"""Fast MMM sky estimator (IDL Astronomy Users Library / DAOPHOT), scalar and batched.

The reference is the D. Jones port in ``vircampype.external.mmm`` (unchanged).
Experimental (off until validated on real data): the pipeline uses this module only when the setup option
``vectorized_mmm`` is on (``destripe_helper`` rows, ``background_image`` tiles).

``mmm`` performs the same operations on the same data in the same dtypes as the
reference, so its results are bit-identical to it. The only change is that the
four pure-Python loops which move the acceptance window one element at a time are
replaced by closed forms evaluated with ``np.searchsorted`` (see ``_lower_bound``
/ ``_upper_bound``).

``mmm_batch`` runs the same algorithm on every 1-D slice of an array along an axis
at once (rows of an image, mesh tiles, ...). All slices are sorted in one call,
every slice keeps its own iteration state and converges independently, and the
work is chunked so the transient memory stays bounded whatever the input size.
The running sums are accumulated in float64 from the start instead of being
updated with float32 partial sums as the reference does for float32 input, so
its results agree with the reference to float32 rounding (and are more accurate).
A slice that ends in a failure code in the batch is recomputed with the scalar
``mmm``, so the batch never fails where the scalar succeeds. Slices whose finite
content is small compared with their length are also handed to the scalar
``mmm`` (it sorts only the finite values). See the module tests for the measured
deviations.

The one intentional deviation from the reference: ``highbad`` works (the
reference indexes a 0-d scalar and raises on numpy >= 2), and ``highbad=0`` is a
valid level; only ``False`` and ``None`` mean "not set". Everything else follows
the reference, including ``readnoise=0`` meaning "not set", ``nan=False`` with
non-finite values (delegated to the reference verbatim because its window
stepping then depends on NaN comparisons; with ``highbad`` set this raises a
ValueError), and ``nan=False`` with nd input (ravelled). ``mmm_batch`` requires
``minsky >= 1`` and a real (float32/64, integer or bool) dtype.
"""

import numpy as np
from numpy.lib.array_utils import normalize_axis_index

from vircampype.external.mmm import mmm as mmm_reference

__all__ = ["batch_is_vectorised", "mmm", "mmm_batch"]

# Above this slice length the per-slice sort dominates and the scalar path is as
# fast as the batched one while using less memory (measured on M4 Pro, numpy 2.3:
# batch/scalar time 0.45 at N=2048, 0.65 at 4096, 0.94 at 16384, 1.04 at 65536).
_BATCH_MAX_SLICE = 1 << 15
# Below this many slices the batched machinery does not pay for itself
# (batch/scalar 2.5 at 4 slices, 0.9 at 16, 0.5 at 256; N=2048).
_BATCH_MIN_SLICES = 16
# Per-slice cost model (seconds; M4 Pro, numpy 2.3) used to hand sparse slices to
# the scalar path: the batch sorts the full slice length N, the scalar only the n
# finite values but pays a fixed overhead and one pass over N.
_SORT_COST = 0.7e-9  # per element x log2(elements)
_BATCH_FIXED = 5e-6
_SCALAR_FIXED = 40e-6
_SCALAR_PASS = 1e-9  # per element (isfinite + compaction)


def batch_is_vectorised(n_slices, slice_length):
    """True if ``mmm_batch`` uses the vectorised core for this input shape.

    Otherwise it loops over the scalar ``mmm`` (very long slices are sort-bound,
    few slices do not amortise the batch set-up); callers can then keep working
    on views instead of materialising a slice matrix.
    """
    return slice_length <= _BATCH_MAX_SLICE and n_slices >= _BATCH_MIN_SLICES


def _scalar_cheaper(n_finite, slice_length):
    """Per slice: is the scalar path (sorting ``n_finite`` values) faster?"""
    n = np.maximum(n_finite, 2).astype(np.float64)
    scalar = _SCALAR_FIXED + _SCALAR_PASS * slice_length + _SORT_COST * n * np.log2(n)
    batch = _BATCH_FIXED + _SORT_COST * slice_length * np.log2(max(slice_length, 2))
    return scalar < batch


def _highbad_level(highbad):
    """``highbad`` as a float level, or None when not set (``False``/``None``)."""
    if highbad is None or (isinstance(highbad, (bool, np.bool_)) and not highbad):
        return None
    if isinstance(highbad, (bool, np.bool_)):
        raise ValueError("highbad must be a pixel level, False or None")
    return highbad


def _chauvenet_r(nsky):
    """Chauvenet rejection factor as a function of the number of sky values."""
    r = np.log10(nsky)
    return np.maximum(2.0, (-0.1042 * r + 1.1695) * r + 0.8895)


# ---------------------------------------------------------------------------
# Closed forms for the reference's window stepping loops.
#
# With the sorted array ``sky`` (length n, nlast = n - 1), the window is
# sky[minimm + 1 : maximm + 1]. For a cut value c let
#   q = #(sky < c)   (np.searchsorted side="left")
#   p = #(sky <= c)  (np.searchsorted side="right").
# Lower end, tst_min = sky[minimm + 1] >= cut1:
#   tst_min and minimm == -1            -> no change
#   tst_min and sky[minimm] < cut1       -> no change
#   tst_min (else)                       -> newmin = min(p1 - 1, minimm - 1)
#   not tst_min                          -> newmin = min(q1 - 1, nlast)
# Upper end, tst_max = sky[maximm] <= cut2:
#   tst_max and maximm == nlast          -> no change
#   tst_max and sky[maximm + 1] > cut2   -> no change
#   tst_max (else)                       -> newmax = min(nlast, max(maximm + 1, q2 - 1))
#   not tst_max                          -> newmax = p2 - 1
# The reference loop scans one element at a time and stops at the first index
# satisfying sky[i] <= cut <= sky[i + 1] (or at the array ends); on a sorted
# array that index is the one given above. Exact ties at the cut are advanced by
# one element per iteration by the reference, which the min/max with
# minimm - 1 / maximm + 1 reproduce. The forms do not hold when ``sky`` contains
# NaN (nan=False with non-finite input), which is delegated to the reference.
# ---------------------------------------------------------------------------


def _lower_bound(sky, minimm, nlast, cut1):
    """New ``minimm`` for the lower window end, or None if it does not move."""
    tst_min = sky[minimm + 1] >= cut1
    if tst_min:
        if minimm == -1 or sky[minimm] < cut1:
            return None
        p1 = int(np.searchsorted(sky, cut1, side="right"))
        return min(p1 - 1, minimm - 1)
    q1 = int(np.searchsorted(sky, cut1, side="left"))
    return min(q1 - 1, nlast)


def _upper_bound(sky, maximm, nlast, cut2):
    """New ``maximm`` for the upper window end, or None if it does not move."""
    tst_max = sky[maximm] <= cut2
    if tst_max:
        if maximm == nlast or sky[maximm + 1] > cut2:
            return None
        q2 = int(np.searchsorted(sky, cut2, side="left"))
        return min(nlast, max(maximm + 1, q2 - 1))
    p2 = int(np.searchsorted(sky, cut2, side="right"))
    return p2 - 1


def mmm(
    sky_vector,
    highbad=False,
    debug=False,
    readnoise=False,
    mxiter=50,
    minsky=20,
    nan=True,
):
    """Estimate the sky background in a stellar contaminated field (DAOPHOT MMM).

    MMM assumes that contaminated sky pixel values overwhelmingly display POSITIVE
    departures from the true value. Bit-identical to the original port
    (``vircampype.external.mmm.mmm``) on all inputs it accepts, except for the
    ``highbad`` handling; see the module docstring.

    Parameters
    ----------
    sky_vector : array_like
        Sky values (any shape; flattened).
    highbad : float, optional
        Lowest "bad" pixel level (cosmic rays, saturation); the first acceptance
        window is limited to ``highbad - median``. ``False`` or ``None`` (the
        default) disables it; ``0`` is a valid level.
    debug : bool, optional
        Print diagnostics.
    readnoise : float, optional
        Read noise; widens the central fraction used for the robust median on
        quantised data. ``False`` disables it.
    mxiter : int, optional
        Maximum number of rejection iterations. Default 50.
    minsky : int, optional
        Minimum number of sky values required. Default 20.
    nan : bool, optional
        Drop non-finite values first. Default True.

    Returns
    -------
    skymod, sigma, skew : float
        Mode, standard deviation and skewness of the sky peak. On failure
        ``sigma`` is NaN (too few values, no convergence) or -1.0 (window below
        ``minsky`` at the start of an iteration) and ``skymod`` is the last
        estimate (NaN if the input was too small).
    """
    hb = _highbad_level(highbad)
    sky_vector = np.asarray(sky_vector)
    if nan:
        sky_vector = sky_vector[np.isfinite(sky_vector)]
    else:
        sky_vector = sky_vector.ravel()
        if sky_vector.dtype.kind == "f" and not np.isfinite(sky_vector).all():
            if hb is not None:
                raise ValueError(
                    "highbad is not supported with nan=False and non-finite values"
                )
            return mmm_reference(
                sky_vector,
                highbad=False,
                debug=debug,
                readnoise=readnoise,
                mxiter=mxiter,
                minsky=minsky,
                nan=False,
            )
    nsky = sky_vector.size

    if nsky < minsky:
        if debug:
            print(
                "ERROR -Input vector must contain at least " + str(minsky) + " elements"
            )
        return np.nan, np.nan, np.nan

    nlast = nsky - 1
    if debug:
        print("Processing " + str(nsky) + " element array")

    sky = np.sort(sky_vector)

    # Median of all sky values (stays in the input dtype, like the reference)
    skymid = 0.5 * sky[(nsky - 1) // 2] + 0.5 * sky[nsky // 2]

    cut1 = np.min([skymid - sky[0], sky[nlast] - skymid])
    if hb is not None:
        cut1 = np.min([cut1, hb - skymid])
    cut2 = skymid + cut1
    cut1 = skymid - cut1

    # Pixels between cut1 and cut2 form a contiguous range of the sorted array
    lo = int(np.searchsorted(sky, cut1, side="left"))
    hi = int(np.searchsorted(sky, cut2, side="right"))
    if hi - lo <= 0:
        if debug:
            print(
                "ERROR - No sky values fall within " + str(cut1) + " and " + str(cut2)
            )
        return np.nan, np.nan, np.nan

    delta = sky[lo:hi] - skymid  # Subtract median to improve arithmetic accuracy
    mysum = np.sum(delta.astype("float64"))
    sumsq = np.sum(delta.astype("float64") ** 2)

    maximm = hi - 1  # Highest value accepted at upper end of vector
    minimm = lo - 1  # Highest value rejected at lower end of vector

    # Compute mean and sigma (from the first pass).
    skymed = (
        0.5 * sky[(minimm + maximm + 1) // 2] + 0.5 * sky[(minimm + maximm) // 2 + 1]
    )
    skymn = mysum / (maximm - minimm)
    sigma = np.sqrt(sumsq / (maximm - minimm) - skymn**2)
    skymn = skymn + skymid

    # If mean is less than the mode, contamination is slight and the mean is what
    # we want.
    if skymed < skymn:
        skymod = 3.0 * skymed - 2.0 * skymn
    else:
        skymod = skymn

    # Rejection and recomputation loop
    niter = 0
    clamp = 1
    old = 0
    redo = True
    while redo:
        niter = niter + 1
        if niter > mxiter:
            if debug:
                print(
                    "ERROR - Too many ("
                    + str(mxiter)
                    + ") iterations,"
                    + " unable to compute sky"
                )
            return skymod, np.nan, np.nan

        if maximm - minimm < minsky:
            if debug:
                print(
                    "ERROR - Too few ("
                    + str(maximm - minimm)
                    + ") valid sky elements, unable to compute sky"
                )
            return skymod, -1.0, 0.0

        # Chauvenet rejection criterion and limits (symmetric about the mode)
        r = np.log10(float(maximm - minimm))
        r = np.max([2.0, (-0.1042 * r + 1.1695) * r + 0.8895])
        cut = r * sigma + 0.5 * np.abs(skymn - skymod)
        cut1 = skymod - cut
        cut2 = skymod + cut
        if np.isnan(cut1) or np.isnan(cut2):
            # A NaN cut (sigma NaN from the first pass, or an overflowed mode)
            # makes the reference scan to the array end at that side, which
            # empties the window and fails the minsky check below.
            if debug:
                print("ERROR - Outlier rejection left too few sky elements")
            return skymod, np.nan, np.nan

        # Recompute mean and sigma by adding and/or subtracting sky values at
        # both ends of the interval of acceptable values.
        redo = False
        newmin = _lower_bound(sky, minimm, nlast, cut1)
        if newmin is not None:
            if newmin < minimm:
                delta = sky[newmin + 1 : minimm + 1] - skymid
                mysum = mysum + np.sum(delta)
                sumsq = sumsq + np.sum(delta**2)
            else:
                delta = sky[minimm + 1 : newmin + 1] - skymid
                mysum = mysum - np.sum(delta)
                sumsq = sumsq - np.sum(delta**2)
            redo = True
            minimm = newmin

        newmax = _upper_bound(sky, maximm, nlast, cut2)
        if newmax is not None:
            if newmax > maximm:
                delta = sky[maximm + 1 : newmax + 1] - skymid
                mysum = mysum + np.sum(delta)
                sumsq = sumsq + np.sum(delta**2)
            else:
                delta = sky[newmax + 1 : maximm + 1] - skymid
                mysum = mysum - np.sum(delta)
                sumsq = sumsq - np.sum(delta**2)
            redo = True
            maximm = newmax

        # Compute mean and sigma (from this pass).
        nsky = maximm - minimm
        if nsky < minsky:
            if debug:
                print("ERROR - Outlier rejection left too few sky elements")
            return skymod, np.nan, np.nan

        skymn = mysum / nsky
        var = sumsq / nsky - skymn**2
        if var < 0:
            var = 0
        sigma = float(np.sqrt(var))
        skymn = skymn + skymid

        # Robust median: mean of the central 20% of sky values, symmetric about
        # the window centre.
        center = (minimm + 1 + maximm) / 2.0
        side = np.round(0.2 * (maximm - minimm)) / 2.0 + 0.25
        j = np.round(center - side)
        k = np.round(center + side)

        # Quantised data: expand the range until both limiting values differ from
        # the central value by at least 0.25 times the read noise.
        if readnoise:
            ll = round(center - 0.25)
            mm = round(center + 0.25)
            rr = 0.25 * readnoise
            while (
                (j > 0)
                and (k < nsky - 1)
                and (
                    ((sky[int(ll)] - sky[int(j)]) < rr)
                    or ((sky[int(k)] - sky[int(mm)]) < rr)
                )
            ):
                j -= 1
                k += 1

        skymed = np.sum(sky[int(j) : int(k + 1)]) / (k - j + 1)

        # If the mean is less than the median, contamination is slight and the
        # mean is what we really want.
        if skymed < skymn:
            dmod = 3.0 * skymed - 2.0 * skymn - skymod
        else:
            dmod = skymn - skymod

        # Prevent oscillations by clamping down if sky adjustments change sign
        if dmod * old < 0:
            clamp = 0.5 * clamp
        skymod = skymod + clamp * dmod
        old = dmod

    skew = float((skymn - skymod) / max([1.0, sigma]))

    if debug:
        print(
            "% MMM: Number of unrejected sky elements: ",
            str(maximm - minimm),
            "    Number of iterations: ",
            str(niter),
        )
        print("% MMM: Mode, Sigma, Skew of sky vector:", skymod, sigma, skew)

    return skymod, sigma, skew


# ---------------------------------------------------------------------------
# Batched implementation
# ---------------------------------------------------------------------------


def _row_searchsorted(s, rows, n, values, strict):
    """Vectorised per-row searchsorted on an array of sorted rows.

    Returns, for each query ``i``, the number of elements of ``s[rows[i], :n[i]]``
    that are ``< values[i]`` (``strict[i]`` True, side="left") or ``<= values[i]``
    (side="right"). Comparisons happen in the promoted dtype, exactly as the
    scalar comparisons of the reference. A bisection with O(log N) gathers per
    query keeps the per-iteration cost independent of the slice length.
    """
    lo = np.zeros(rows.size, dtype=np.int64)
    hi = n.astype(np.int64, copy=True)
    last = s.shape[1] - 1
    while True:
        active = lo < hi
        if not active.any():
            return lo
        mid = (lo + hi) >> 1
        v = s[rows, np.minimum(mid, last)]
        # NaN queries sort after everything, as in np.searchsorted
        go_right = (
            np.where(strict, v < values, v <= values) | np.isnan(values)
        ) & active
        lo = np.where(go_right, mid + 1, lo)
        hi = np.where(active & ~go_right, mid, hi)


def _segment_sums(delta, rows, starts, stops, squares=False):
    """Sums of ``delta[rows[i], starts[i]:stops[i]]`` for all i (empty -> 0).

    Gathers only the addressed elements, so the cost is proportional to the
    number of elements that enter or leave a window, not to the slice length.
    With ``squares`` the sums of the squared elements are returned as well.
    """
    length = np.maximum(stops - starts, 0)
    sums = np.zeros(rows.size)
    sq = np.zeros(rows.size) if squares else None
    nz = np.flatnonzero(length)
    if nz.size:
        reps = length[nz]
        offs = np.cumsum(reps) - reps
        cols = np.arange(int(reps.sum())) - np.repeat(offs - starts[nz], reps)
        vals = delta[np.repeat(rows[nz], reps), cols]
        sums[nz] = np.add.reduceat(vals, offs)
        if squares:
            sq[nz] = np.add.reduceat(vals * vals, offs)
    return (sums, sq) if squares else sums


def _mmm_rows(x, hb, readnoise, mxiter, minsky, nan):
    """MMM on every row of the 2-D array ``x``; returns three float64 vectors.

    ``hb`` is the highbad level or None (see ``_highbad_level``).
    """
    nrows, ncols = x.shape
    out_mod = np.full(nrows, np.nan)
    out_sig = np.full(nrows, np.nan)
    out_skew = np.full(nrows, np.nan)
    if nrows == 0 or ncols == 0:
        return out_mod, out_sig, out_skew

    # Non-finite handling. With nan=True the non-finite values are replaced by +inf
    # so they sort to the end and every row's valid values form the prefix [0, n).
    n = np.full(nrows, ncols, dtype=np.int64)
    s = None
    if x.dtype.kind == "f":
        finite = np.isfinite(x)
        if not finite.all():
            if nan:
                n = np.count_nonzero(finite, axis=1).astype(np.int64)
                s = np.where(finite, x, np.inf)
                s.sort(axis=1)
            else:
                # The reference's stepping depends on NaN comparisons here: run
                # it verbatim on the affected rows and the batch on the others.
                if hb is not None:
                    raise ValueError(
                        "highbad is not supported with nan=False and non-finite values"
                    )
                bad = ~finite.all(axis=1)
                for i in np.flatnonzero(bad):
                    out_mod[i], out_sig[i], out_skew[i] = mmm_reference(
                        x[i],
                        highbad=False,
                        readnoise=readnoise,
                        mxiter=mxiter,
                        minsky=minsky,
                        nan=False,
                    )
                good = np.flatnonzero(~bad)
                if good.size:
                    res = _mmm_rows(x[good], hb, readnoise, mxiter, minsky, nan)
                    out_mod[good], out_sig[good], out_skew[good] = res
                return out_mod, out_sig, out_skew
        del finite

    valid = n >= minsky
    if not valid.all():
        rows_valid = np.flatnonzero(valid)
        if rows_valid.size == 0:
            return out_mod, out_sig, out_skew
        res = _mmm_rows(x[rows_valid], hb, readnoise, mxiter, minsky, nan)
        out_mod[rows_valid], out_sig[rows_valid], out_skew[rows_valid] = res
        return out_mod, out_sig, out_skew

    if s is None:
        s = np.sort(x, axis=1)
    del x
    nlast = n - 1
    ri = np.arange(nrows)

    # Median of all sky values, in the input dtype like the reference
    skymid = 0.5 * s[ri, (n - 1) // 2] + 0.5 * s[ri, n // 2]

    # First acceptance window: symmetric about the median, limited by the nearer
    # extreme (and by highbad).
    cut1 = np.minimum(skymid - s[:, 0], s[ri, nlast] - skymid)
    if hb is not None:
        cut1 = np.minimum(cut1, hb - skymid)
    cut2 = skymid + cut1
    cut1 = skymid - cut1
    lo = _row_searchsorted(s, ri, n, cut1, np.ones(nrows, dtype=bool))
    hi = _row_searchsorted(s, ri, n, cut2, np.zeros(nrows, dtype=bool))
    ngood = hi - lo
    active = ngood > 0  # rows with an empty first window return NaN
    if not active.any():
        return out_mod, out_sig, out_skew
    minimm = lo - 1  # highest index rejected at the lower end
    maximm = hi - 1  # highest index accepted at the upper end

    # Deltas to the median in float64. The subtraction runs in the input dtype
    # (its result is then cast), exactly like the reference's ``sky - skymid``.
    # A zero guard column keeps the reduceat boundaries below in range.
    stride = ncols + 1
    delta = np.zeros((nrows, stride))
    np.subtract(s, skymid[:, None], out=delta[:, :ncols])
    bounds = np.empty(2 * nrows, dtype=np.int64)
    bounds[0::2] = ri * stride + lo
    bounds[1::2] = ri * stride + hi
    mysum = np.add.reduceat(delta.reshape(-1), bounds)[0::2]
    sumsq = np.add.reduceat(np.square(delta).reshape(-1), bounds)[0::2]
    mysum = np.where(active, mysum, 0.0)
    sumsq = np.where(active, sumsq, 0.0)
    del bounds

    # First pass: mean, sigma and mode
    ngood_safe = np.where(active, ngood, 1)
    skymed = (
        0.5 * s[ri, np.maximum((minimm + maximm + 1) // 2, 0)]
        + 0.5 * s[ri, np.maximum((minimm + maximm) // 2 + 1, 0)]
    )
    skymn = mysum / ngood_safe
    with np.errstate(invalid="ignore"):
        # Negative variance (rounding) gives a NaN sigma that fails the row in
        # the loop, as in the reference; inactive rows hold garbage here.
        sigma = np.sqrt(sumsq / ngood_safe - skymn**2)
    skymn = skymn + skymid
    skymod = np.where(skymed < skymn, 3.0 * skymed - 2.0 * skymn, skymn)

    # Per-row iteration state. ``cen`` tracks the sum of the deltas over the
    # central range [prev_j, prev_k] used for the robust median.
    niter = np.zeros(nrows, dtype=np.int64)
    clamp = np.ones(nrows)
    old = np.zeros(nrows)
    prev_j = np.zeros(nrows, dtype=np.int64)
    prev_k = np.full(nrows, -1, dtype=np.int64)
    cen = np.zeros(nrows)
    rr = 0.25 * readnoise if readnoise else None

    while True:
        a = np.flatnonzero(active)
        if a.size == 0:
            break
        niter[a] += 1

        # Entry checks, in the reference's order
        f_iter = niter[a] > mxiter
        if f_iter.any():
            fa = a[f_iter]
            out_mod[fa] = skymod[fa]
            active[fa] = False
            a = a[~f_iter]
        nsky = maximm[a] - minimm[a]
        f_few = nsky < minsky
        if f_few.any():
            fa = a[f_few]
            out_mod[fa] = skymod[fa]
            out_sig[fa] = -1.0
            out_skew[fa] = 0.0
            active[fa] = False
            a = a[~f_few]
            nsky = nsky[~f_few]
        if a.size == 0:
            continue

        r = _chauvenet_r(nsky.astype(np.float64))
        cut = r * sigma[a] + 0.5 * np.abs(skymn[a] - skymod[a])
        c1 = skymod[a] - cut
        c2 = skymod[a] + cut
        nan_cut = np.isnan(c1) | np.isnan(c2)
        if nan_cut.any():
            # A NaN cut makes the reference scan to the array end at that side,
            # which empties the window and fails the minsky check.
            fa = a[nan_cut]
            out_mod[fa] = skymod[fa]
            active[fa] = False
            keep = ~nan_cut
            a, c1, c2 = a[keep], c1[keep], c2[keep]
            if a.size == 0:
                continue
        mn = minimm[a]
        mx = maximm[a]
        nl = nlast[a]
        na = n[a]

        # Which ends move (see the closed forms above)
        tst_min = s[a, mn + 1] >= c1
        move_lo = ~(tst_min & ((mn == -1) | (s[a, np.maximum(mn, 0)] < c1)))
        tst_max = s[a, mx] <= c2
        move_hi = ~(tst_max & ((mx == nl) | (s[a, np.minimum(mx + 1, nl)] > c2)))

        # One bisection per moving end: strict count (q) when rejecting at the
        # lower end / including at the upper end, non-strict (p) otherwise.
        cnt = _row_searchsorted(
            s,
            np.concatenate([a[move_lo], a[move_hi]]),
            np.concatenate([na[move_lo], na[move_hi]]),
            np.concatenate([c1[move_lo], c2[move_hi]]),
            np.concatenate([~tst_min[move_lo], tst_max[move_hi]]),
        )
        n_lo = np.count_nonzero(move_lo)
        cnt_lo, cnt_hi = cnt[:n_lo], cnt[n_lo:]
        new_mn = mn.copy()
        new_mn[move_lo] = np.where(
            tst_min[move_lo],
            np.minimum(cnt_lo - 1, mn[move_lo] - 1),
            np.minimum(cnt_lo - 1, nl[move_lo]),
        )
        new_mx = mx.copy()
        new_mx[move_hi] = np.where(
            tst_max[move_hi],
            np.minimum(nl[move_hi], np.maximum(mx[move_hi] + 1, cnt_hi - 1)),
            cnt_hi - 1,
        )
        redo = move_lo | move_hi
        minimm[a] = new_mn
        maximm[a] = new_mx

        # Update the running sums with the elements that entered (+) or left (-)
        # the window at either end.
        lo_in = new_mn < mn
        hi_in = new_mx > mx
        starts = np.concatenate(
            [np.where(lo_in, new_mn + 1, mn + 1), np.where(hi_in, mx + 1, new_mx + 1)]
        )
        stops = np.concatenate(
            [np.where(lo_in, mn + 1, new_mn + 1), np.where(hi_in, new_mx + 1, mx + 1)]
        )
        signs = np.concatenate([np.where(lo_in, 1.0, -1.0), np.where(hi_in, 1.0, -1.0)])
        ds, dsq = _segment_sums(delta, np.concatenate([a, a]), starts, stops, True)
        m = a.size
        mysum[a] += signs[:m] * ds[:m] + signs[m:] * ds[m:]
        sumsq[a] += signs[:m] * dsq[:m] + signs[m:] * dsq[m:]

        # Mean and sigma from this pass
        nsky = new_mx - new_mn
        f_few = nsky < minsky
        if f_few.any():
            fa = a[f_few]
            out_mod[fa] = skymod[fa]
            active[fa] = False
            keep = ~f_few
            a, new_mn, new_mx, nsky, redo = (
                a[keep],
                new_mn[keep],
                new_mx[keep],
                nsky[keep],
                redo[keep],
            )
            m = a.size
            if m == 0:
                continue
        smn = mysum[a] / nsky
        var = np.maximum(sumsq[a] / nsky - smn**2, 0.0)
        sig = np.sqrt(var)
        smn = smn + skymid[a]

        # Robust median: mean of the central 20% of the window
        center = (new_mn + 1 + new_mx) / 2.0
        side = np.round(0.2 * nsky) / 2.0 + 0.25
        j = np.round(center - side).astype(np.int64)
        k = np.round(center + side).astype(np.int64)
        if rr is not None:
            ll = np.round(center - 0.25).astype(np.int64)
            mm = np.round(center + 0.25).astype(np.int64)
            sll = s[a, ll]
            smm = s[a, mm]
            while True:
                cond = (j > 0) & (k < nsky - 1)
                if cond.any():
                    cond &= ((sll - s[a, j]) < rr) | ((s[a, k] - smm) < rr)
                if not cond.any():
                    break
                j[cond] -= 1
                k[cond] += 1
        # Move the tracked central range [prev_j, prev_k] to [j, k] by adding and
        # removing the elements at its ends (first use starts from an empty range).
        first = niter[a] == 1
        pj = np.where(first, j, prev_j[a])
        pk = np.where(first, j - 1, prev_k[a])
        j_in = j < pj
        k_in = k > pk
        cs = _segment_sums(
            delta,
            np.concatenate([a, a]),
            np.concatenate([np.where(j_in, j, pj), np.where(k_in, pk + 1, k + 1)]),
            np.concatenate([np.where(j_in, pj, j), np.where(k_in, k + 1, pk + 1)]),
        )
        cen[a] += (
            np.where(j_in, 1.0, -1.0) * cs[:m] + np.where(k_in, 1.0, -1.0) * cs[m:]
        )
        prev_j[a] = j
        prev_k[a] = k
        nmed = k - j + 1
        smed = (cen[a] + nmed * skymid[a]) / nmed

        dmod = np.where(smed < smn, 3.0 * smed - 2.0 * smn - skymod[a], smn - skymod[a])
        flip = dmod * old[a] < 0
        clamp[a[flip]] *= 0.5
        smod = skymod[a] + clamp[a] * dmod
        skymod[a] = smod
        old[a] = dmod
        skymn[a] = smn
        sigma[a] = sig

        done = ~redo
        if done.any():
            fa = a[done]
            out_mod[fa] = smod[done]
            out_sig[fa] = sig[done]
            out_skew[fa] = (smn[done] - smod[done]) / np.maximum(1.0, sig[done])
            active[fa] = False

    return out_mod, out_sig, out_skew


def mmm_batch(
    data,
    axis=-1,
    highbad=False,
    readnoise=False,
    mxiter=50,
    minsky=20,
    nan=True,
    max_chunk_elements=1 << 20,
):
    """Run ``mmm`` on every 1-D slice of ``data`` along ``axis``, vectorised.

    Parameters
    ----------
    data : array_like
        Input array; each 1-D slice along ``axis`` is one sky vector.
    axis : int, optional
        Axis of the sky vectors. Default -1.
    highbad, readnoise, mxiter, minsky, nan
        As for ``mmm``.
    max_chunk_elements : int, optional
        Slices are processed in chunks of at most this many elements; the
        transient memory is about 30 bytes per element of a chunk. Default 2**20.

    Returns
    -------
    skymod, sigma, skew : np.ndarray
        float64 arrays of shape ``data.shape`` without ``axis``, holding what
        ``mmm`` returns for the corresponding slice, failure codes included.
        Integer and float64 input agree with ``mmm`` to float64 rounding. For
        float32 input the batch keeps its running sums in float64 where the
        reference sums float32 slices, so results differ by float32 rounding
        (a few 1e-7 of the sky level), and a sample lying within that rounding
        of a cut can be classified differently; such decisions follow the
        float64 result and are rare (none seen on real rows or tiles, ~1e-4
        of adversarial fuzz rows with tiny sigma at high levels). A slice that
        fails in the batch is recomputed with the scalar ``mmm``, so the batch
        never fails where the scalar succeeds. Integer input beyond ~2**53 / n
        differs because the reference's int64 sums overflow there. float16 and
        complex input are rejected.

    Notes
    -----
    Slices whose finite content is small compared with their length (heavily
    masked tiles) are computed with the scalar ``mmm`` when that is faster (it
    sorts only the finite values); the decision is a fixed cost model, so
    results do not depend on timing. Memory: about 20 bytes per element of a
    chunk. For ``axis`` other than the last on a non-contiguous layout the
    input is additionally copied once in full (``moveaxis`` + ``reshape``); 2-D
    input with ``axis=1`` is a view.
    """
    data = np.asarray(data)
    if data.ndim == 0:
        raise ValueError("mmm_batch needs at least a 1-D input")
    if data.dtype.kind not in "fiub" or data.dtype == np.float16:
        # complex is meaningless; float16 cannot be matched to rounding because
        # the reference sums float16 slices (cast to float32 first).
        raise TypeError(f"mmm_batch does not support dtype {data.dtype}")
    if minsky < 1:
        raise ValueError("mmm_batch needs minsky >= 1")
    hb = _highbad_level(highbad)
    axis = normalize_axis_index(axis, data.ndim)
    out_shape = data.shape[:axis] + data.shape[axis + 1 :]
    nrows, ncols = int(np.prod(out_shape, dtype=np.int64)), data.shape[axis]
    rows = np.moveaxis(data, axis, -1).reshape(nrows, ncols)
    out = tuple(np.full(nrows, np.nan) for _ in range(3))

    def scalar(indices):
        for i in indices:
            out[0][i], out[1][i], out[2][i] = mmm(
                rows[i],
                highbad=highbad,
                readnoise=readnoise,
                mxiter=mxiter,
                minsky=minsky,
                nan=nan,
            )

    if not batch_is_vectorised(nrows, ncols):
        scalar(range(nrows))
        return tuple(o.reshape(out_shape) for o in out)

    step = max(1, int(max_chunk_elements) // max(ncols, 1))
    sparse_dispatch = nan and rows.dtype.kind == "f"
    for start in range(0, nrows, step):
        stop = min(start + step, nrows)
        chunk = rows[start:stop]
        dense = None
        if sparse_dispatch:
            n_finite = np.count_nonzero(np.isfinite(chunk), axis=1)
            sparse = _scalar_cheaper(n_finite, ncols)
            if sparse.any():
                scalar(start + np.flatnonzero(sparse))
                dense = np.flatnonzero(~sparse)
                if dense.size == 0:
                    continue
                chunk = chunk[dense]
        res = _mmm_rows(chunk, hb, readnoise, mxiter, minsky, nan)
        if dense is None:
            out[0][start:stop], out[1][start:stop], out[2][start:stop] = res
        else:
            out[0][start + dense], out[1][start + dense], out[2][start + dense] = res

    # Failed slices are recomputed with the scalar mmm: at float32 resolution the
    # float64 sums can (rarely) keep a window moving until mxiter where the
    # reference's float32 sums converge, so the batch must not have the last word
    # on failures. Slices that fail for lack of data are cheap to redo.
    failed = np.flatnonzero(~(np.isfinite(out[1]) & (out[1] >= 0.0)))
    if failed.size:
        scalar(failed)

    return tuple(o.reshape(out_shape) for o in out)
