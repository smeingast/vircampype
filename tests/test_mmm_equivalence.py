"""Equivalence of ``vircampype.tools.mmmtools`` with the original MMM port.

The oracle is the untouched original, ``vircampype.external.mmm.mmm``. Three
layers:

* ``mmmtools.mmm`` (scalar) must be bit-identical to the oracle (same values, same
  NaN/-1 failure codes, same return types) on randomised and adversarial inputs.
* ``mmm_batch`` must agree with the scalar slice by slice to float rounding, with
  component-specific tolerances (mode: relative to the sky level; sigma: relative
  to sigma plus the data resolution at that level; skew: the mode tolerance over
  max(1, sigma)); exact for integer input. Wherever the scalar succeeds the batch
  must succeed (failed batch slices are recomputed with the scalar).
* Regression tests for every edge case and failure path, and for the two pipeline
  callers (``destripe_helper``, ``background_image``) with the ``vectorized_mmm``
  flag off (original code path, external ``mmm``) and on.
"""

import unittest
import warnings
from unittest import mock

import numpy as np

from vircampype.external.mmm import mmm as mmm_reference
from vircampype.tools import imagetools
from vircampype.tools.imagetools import background_image, destripe_helper
from vircampype.tools.mmmtools import (
    _lower_bound,
    _row_searchsorted,
    _scalar_cheaper,
    _segment_sums,
    _upper_bound,
    batch_is_vectorised,
    mmm,
    mmm_batch,
)


def _same_triplet(a, b):
    """True if two (skymod, sigma, skew) triplets are identical (NaN == NaN)."""
    return all(
        (np.isnan(x) and np.isnan(y)) or (not np.isnan(x) and x == y)
        for x, y in zip(a, b)
    )


def _vector(rng, n, kind, dtype):
    """Random sky vectors of the kinds that matter for MMM."""
    if kind == "gauss":
        x = rng.normal(1000.0, 10.0, n)
    elif kind == "stars":
        x = rng.normal(1000.0, 10.0, n)
        k = max(1, n // 10)
        x[:k] += rng.exponential(500.0, k)
    elif kind == "bright":
        x = rng.normal(1000.0, 10.0, n)
        k = max(1, n // 50)
        x[:k] = 60000.0
    elif kind == "lowtail":
        x = rng.normal(1000.0, 10.0, n)
        k = max(1, n // 20)
        x[:k] -= rng.exponential(300.0, k)
    elif kind == "constant":
        x = np.full(n, 42.0)
    elif kind == "fewlevels":
        x = rng.integers(0, 5, n).astype(float)
    elif kind == "quantised":
        x = np.round(rng.normal(100.0, 2.0, n))
    elif kind == "nans":
        x = rng.normal(0.0, 1.0, n)
        x[rng.random(n) < 0.1] = np.nan
        x[rng.random(n) < 0.02] = np.inf
    elif kind == "zeromean":
        x = rng.normal(0.0, 20.0, n)
    else:
        raise ValueError(kind)
    if np.issubdtype(dtype, np.integer):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(dtype)
    return x.astype(dtype)


KINDS = (
    "gauss",
    "stars",
    "bright",
    "lowtail",
    "constant",
    "fewlevels",
    "quantised",
    "nans",
    "zeromean",
)
DTYPES = (np.float64, np.float32, np.int16, np.int32, np.int64)


class TestScalarExactEquivalence(unittest.TestCase):
    """``mmm`` reproduces ``mmm_reference`` bit for bit."""

    def _check(self, x, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = mmm_reference(x, **kwargs)
            new = mmm(x, **kwargs)
        self.assertTrue(
            _same_triplet(ref, new),
            msg=f"{x.dtype} n={x.size} {kwargs}: {ref} != {new}",
        )
        self.assertEqual([type(v) for v in ref], [type(v) for v in new])

    def test_randomised(self):
        rng = np.random.default_rng(20261016)
        sizes = [1, 2, 3, 5, 9, 10, 11, 19, 20, 21, 25, 40, 64, 100, 257, 2048]
        for _ in range(3000):
            n = int(rng.choice(sizes))
            x = _vector(rng, n, str(rng.choice(KINDS)), rng.choice(DTYPES))
            self._check(
                x,
                minsky=int(rng.choice([1, 2, 5, 10, 20])),
                mxiter=int(rng.choice([1, 2, 3, 50, 500])),
                readnoise=float(rng.choice([0.0, 0.0, 1.0, 5.0])),
            )

    def test_dense_ties_small_integers(self):
        """Exact ties at the cuts drive the one-element-per-iteration stepping."""
        rng = np.random.default_rng(1)
        for _ in range(2000):
            n = int(rng.integers(1, 40))
            x = rng.integers(0, int(rng.integers(2, 6)), n)
            dt = rng.choice([np.int64, np.float32, np.float64])
            self._check(x.astype(dt), minsky=int(rng.integers(1, 6)))

    def test_realistic_rows_and_tiles(self):
        rng = np.random.default_rng(2)
        img = rng.normal(4000.0, 18.0, (64, 2048)).astype(np.float32)
        img[rng.random(img.shape) < 0.05] = np.nan
        img[:, 100:140] += 2000.0
        for row in img:
            self._check(row, minsky=10)
        for tile in img[:, :512].reshape(-1, 64 * 64):
            self._check(tile)

    def test_two_dimensional_input_is_flattened(self):
        rng = np.random.default_rng(3)
        x = rng.normal(10.0, 1.0, (30, 40))
        x[0, 0] = np.nan
        self._check(x)

    def test_failure_paths(self):
        rng = np.random.default_rng(4)
        x = rng.normal(100.0, 5.0, 200)
        # too few elements
        self._check(x[:5], minsky=20)
        self.assertTrue(all(np.isnan(v) for v in mmm(x[:5], minsky=20)))
        # too many iterations
        self._check(x, mxiter=1)
        self._check(x, mxiter=0)
        res = mmm(x, mxiter=0)
        self.assertTrue(np.isfinite(res[0]) and np.isnan(res[1]) and np.isnan(res[2]))
        # window below minsky at loop entry (-1.0 / 0.0 code)
        y = np.concatenate([np.full(30, 10.0), np.full(5, 11.0)])
        self._check(y, minsky=40)
        self._check(y, minsky=33)
        # first window may be empty: median falls between two separated groups
        z = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        self._check(z, minsky=1)


class TestScalarBehaviour(unittest.TestCase):
    def test_return_types(self):
        rng = np.random.default_rng(5)
        skymod, sigma, skew = mmm(rng.normal(100.0, 5.0, 1000))
        self.assertIsInstance(skymod, np.float64)
        self.assertIs(type(sigma), float)
        self.assertIs(type(skew), float)

    def test_highbad_limits_first_window(self):
        rng = np.random.default_rng(6)
        x = rng.normal(100.0, 5.0, 2000)
        x[:200] = 150.0  # "bad" high plateau (e.g. saturation)
        skymod, sigma, _ = mmm(x, highbad=140.0)
        self.assertAlmostEqual(skymod, 100.0, delta=1.0)
        self.assertAlmostEqual(sigma, 5.0, delta=1.0)
        # highbad below the median empties the first window
        self.assertTrue(np.isnan(mmm(x, highbad=90.0)[0]))
        # the reference raises on numpy >= 2 (indexes a 0-d scalar)
        with self.assertRaises((ValueError, IndexError)):
            mmm_reference(x, highbad=140.0)

    def test_highbad_zero_is_a_level_and_none_false_are_unset(self):
        """The one intentional deviation: highbad=0 is a valid level (the
        reference tests it for truth, so 0 meant unset)."""
        rng = np.random.default_rng(29)
        x = rng.normal(-1.0, 1.0, 2000)
        x[:50] = 0.5  # plateau above zero
        unset = mmm_reference(x)
        self.assertTrue(_same_triplet(mmm(x), unset))
        self.assertTrue(_same_triplet(mmm(x, highbad=None), unset))
        self.assertTrue(_same_triplet(mmm(x, highbad=False), unset))
        zero = mmm(x, highbad=0)
        self.assertTrue(_same_triplet(zero, mmm(x, highbad=0.0)))
        self.assertTrue(_same_triplet(zero, mmm(x, highbad=1e-12)))
        self.assertFalse(_same_triplet(zero, unset))
        for a, b in zip(mmm_batch(np.tile(x, (20, 1)), highbad=0), zero):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        with self.assertRaises(ValueError):
            mmm(x, highbad=True)

    def test_nan_false_without_nonfinite_matches_reference(self):
        rng = np.random.default_rng(7)
        x = rng.normal(100.0, 5.0, 500)
        self.assertTrue(_same_triplet(mmm(x, nan=False), mmm_reference(x, nan=False)))

    def test_nan_false_with_nonfinite_delegates_to_reference(self):
        rng = np.random.default_rng(8)
        x = rng.normal(100.0, 5.0, 500)
        x[::50] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertTrue(
                _same_triplet(mmm(x, nan=False), mmm_reference(x, nan=False))
            )

    def test_list_input(self):
        self.assertAlmostEqual(mmm(list(range(100)))[0], 49.5, delta=1.0)

    def test_searchsorted_compares_in_promoted_dtype(self):
        """The closed forms rely on float32 arrays being searched with float64 cuts
        without rounding the cut (numpy promotes to the common dtype)."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.assertEqual(np.searchsorted(a, np.float64(2.0) + 1e-9, side="left"), 2)
        self.assertEqual(np.searchsorted(a, np.float64(2.0) - 1e-9, side="left"), 1)
        b = np.array([1, 2, 3], dtype=np.int32)
        self.assertEqual(np.searchsorted(b, 2.5, side="left"), 2)


class TestClosedFormsAgainstStepping(unittest.TestCase):
    """The window-edge closed forms reproduce the reference's stepping loops."""

    @staticmethod
    def _step_lower(sky, minimm, nlast, cut1):
        newmin = minimm
        tst_min = 1 if sky[newmin + 1] >= cut1 else 0
        done = 1 if (newmin == -1) and tst_min else 0
        if not done:
            skyind = newmin if newmin > 0 else 0
            if (sky[skyind] < cut1) and tst_min:
                done = 1
        if done:
            return None
        istep = 1 - 2 * tst_min
        while not done:
            newmin = newmin + istep
            if (newmin == -1) | (newmin == nlast):
                done = 1
            if not done:
                if (sky[newmin] <= cut1) and (sky[newmin + 1] >= cut1):
                    done = 1
        return newmin

    @staticmethod
    def _step_upper(sky, maximm, nlast, cut2):
        newmax = maximm
        tst_max = 1 if sky[maximm] <= cut2 else 0
        done = 1 if (maximm == nlast) and tst_max else 0
        if not done:
            skyind = maximm + 1 if maximm + 1 < nlast else nlast
            if tst_max and (sky[skyind] > cut2):
                done = 1
        if done:
            return None
        istep = -1 + 2 * tst_max
        while not done:
            newmax = newmax + istep
            if (newmax == nlast) or (newmax == -1):
                done = 1
            if not done:
                if (sky[newmax] <= cut2) and (sky[newmax + 1] >= cut2):
                    done = 1
        return newmax

    def test_exhaustive_small_arrays(self):
        rng = np.random.default_rng(10)
        for _ in range(20000):
            n = int(rng.integers(1, 12))
            sky = np.sort(rng.integers(0, 6, n).astype(float))
            nlast = n - 1
            minimm = int(rng.integers(-1, n - 1))  # window non-empty
            maximm = int(rng.integers(minimm + 1, n))
            cut = float(rng.choice([rng.integers(-1, 7), rng.uniform(-1, 7)]))
            self.assertEqual(
                _lower_bound(sky, minimm, nlast, cut),
                self._step_lower(sky, minimm, nlast, cut),
                msg=f"lower {sky} minimm={minimm} cut={cut}",
            )
            self.assertEqual(
                _upper_bound(sky, maximm, nlast, cut),
                self._step_upper(sky, maximm, nlast, cut),
                msg=f"upper {sky} maximm={maximm} cut={cut}",
            )


class TestBatchHelpers(unittest.TestCase):
    def test_row_searchsorted(self):
        rng = np.random.default_rng(11)
        s = np.sort(rng.integers(0, 10, (50, 37)).astype(np.float32), axis=1)
        n = rng.integers(1, 38, 50)
        s[np.arange(37)[None, :] >= n[:, None]] = np.inf
        s.sort(axis=1)
        vals = rng.uniform(-1, 11, 50)
        vals[::7] = np.round(vals[::7])  # exact ties
        rows = np.arange(50)
        left = _row_searchsorted(s, rows, n, vals, np.ones(50, dtype=bool))
        right = _row_searchsorted(s, rows, n, vals, np.zeros(50, dtype=bool))
        for i in range(50):
            self.assertEqual(left[i], np.searchsorted(s[i, : n[i]], vals[i], "left"))
            self.assertEqual(right[i], np.searchsorted(s[i, : n[i]], vals[i], "right"))

    def test_segment_sums(self):
        rng = np.random.default_rng(12)
        d = rng.normal(size=(6, 20))
        rows = np.array([0, 3, 3, 5, 1])
        starts = np.array([2, 0, 19, 5, 7])
        stops = np.array([9, 20, 20, 5, 6])  # last two: empty / inverted
        sums, sq = _segment_sums(d, rows, starts, stops, squares=True)
        for i in range(5):
            seg = d[rows[i], starts[i] : max(stops[i], starts[i])]
            self.assertAlmostEqual(sums[i], seg.sum(), places=12)
            self.assertAlmostEqual(sq[i], (seg**2).sum(), places=12)


class TestBatchEquivalence(unittest.TestCase):
    """``mmm_batch`` agrees with the scalar ``mmm`` slice by slice."""

    @staticmethod
    def _tolerance(ref, dtype):
        """Component-specific tolerances for (mode, sigma, skew).

        The batch and the reference differ by the rounding of the data dtype
        (float32 partial sums in the reference), so the mode tolerance is a few
        tens of data ulps at the sky level; sigma is compared relative to itself
        plus a few data ulps at the level (it cannot be determined better than
        that from data quantised at the ulp); skew inherits the mode tolerance
        over max(1, sigma).
        """
        eps = np.finfo(np.float32 if dtype == np.float32 else np.float64).eps
        sigma = np.where(np.isfinite(ref[:, 1]), np.abs(ref[:, 1]), 0.0)
        level = np.abs(ref[:, 0]) + sigma
        tol_mod = 32 * eps * level + 1e-300
        tol_sig = 1e-4 * sigma + 8 * eps * level + 1e-300
        tol_skew = 2 * tol_mod / np.maximum(1.0, sigma) + 1e-300
        return np.stack([tol_mod, tol_sig, tol_skew], 1)

    @staticmethod
    def _failed(res):
        return ~(np.isfinite(res[:, 1]) & (res[:, 1] >= 0.0))

    def _check_block(self, X, max_flips=0, **kwargs):
        """Batch vs scalar on a block.

        Wherever the scalar succeeds the batch succeeds; a batch failure is
        exactly the scalar's; successful rows agree within the component
        tolerances. ``max_flips`` rows may differ by a window decision (data
        quantised at the dtype resolution); those must still be sane estimates.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = np.stack(mmm_batch(X, axis=1, **kwargs), 1)
            ref = np.array([mmm(row, **kwargs) for row in X], dtype=float)
        got_failed, ref_failed = self._failed(got), self._failed(ref)
        self.assertFalse(
            np.any(got_failed & ~ref_failed),
            msg=f"{X.dtype} {kwargs}: batch-only failure",
        )
        for i in np.flatnonzero(got_failed):
            self.assertTrue(_same_triplet(got[i], ref[i]), msg=f"{got[i]} != {ref[i]}")
        fin = ~ref_failed & ~got_failed
        if np.issubdtype(X.dtype, np.integer):
            np.testing.assert_array_equal(got[fin], ref[fin])
            return
        diff = np.abs(got[fin] - ref[fin])
        ratio = diff / self._tolerance(ref[fin], X.dtype)
        beyond = np.any(ratio > 1.0, axis=1)
        worst = ratio.max(axis=0) if ratio.size else 0.0
        self.assertLessEqual(
            int(beyond.sum()),
            max_flips,
            msg=f"{X.dtype} {kwargs}: max diff/tol {worst}",
        )
        for g, r in zip(got[fin][beyond], ref[fin][beyond]):
            sig = max(r[1], g[1])
            self.assertLessEqual(abs(g[0] - r[0]), 5.0 * sig)
            self.assertLessEqual(max(r[1], g[1]), 2.0 * min(r[1], g[1]))

    def test_randomised_blocks(self):
        rng = np.random.default_rng(13)
        for dtype in DTYPES:
            for n in (21, 64, 300, 2048):
                for kwargs in (
                    {},
                    {"minsky": 10},
                    {"minsky": 20, "mxiter": 2},
                    {"minsky": 5, "mxiter": 500, "readnoise": 2.0},
                    {"readnoise": 0.5},
                ):
                    X = np.stack(
                        [
                            _vector(rng, n, str(rng.choice(KINDS)), dtype)
                            for _ in range(64)
                        ]
                    )
                    self._check_block(X, **kwargs)

    def test_rows_with_too_few_valid_values(self):
        rng = np.random.default_rng(14)
        X = rng.normal(100.0, 5.0, (40, 200))
        X[3] = np.nan
        X[7, 5:] = np.nan  # 5 valid values < minsky
        X[9, ::2] = np.nan
        self._check_block(X, minsky=10)
        res = mmm_batch(X, axis=1, minsky=10)
        self.assertTrue(all(np.isnan(r[3]) for r in res))
        self.assertTrue(all(np.isnan(r[7]) for r in res))
        self.assertTrue(all(np.isfinite(r[9]) for r in res))

    def test_mixed_failure_rows_in_one_block(self):
        rng = np.random.default_rng(15)
        X = rng.normal(100.0, 5.0, (64, 100))
        X[1] = 42.0  # constant
        X[2, :50] = 0.0
        X[2, 50:] = 1.0  # bimodal ties
        X[4] = np.linspace(0, 1e6, 100)  # huge spread
        X[5, :95] = np.nan
        self._check_block(X, minsky=10)
        self._check_block(X, minsky=10, mxiter=1)

    def test_nan_false_delegates_rows_with_nonfinite(self):
        rng = np.random.default_rng(16)
        X = rng.normal(100.0, 5.0, (20, 200))
        X[4, 10] = np.nan
        X[11, 0] = np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = np.stack(mmm_batch(X, axis=1, nan=False), 1)
            ref = np.array([mmm_reference(r, nan=False) for r in X], dtype=float)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
        fin = np.isfinite(ref).all(axis=1)
        np.testing.assert_allclose(got[fin], ref[fin], rtol=1e-9, atol=1e-9)

    def test_highbad(self):
        rng = np.random.default_rng(17)
        X = rng.normal(100.0, 5.0, (30, 500))
        X[:, :50] = 150.0
        self._check_block(X, highbad=140.0)
        self._check_block(X, highbad=90.0)

    def test_extreme_values_float32(self):
        # Level 1e15 keeps the reference's float32 slice sums of delta**2 below
        # the float32 maximum; beyond ~1e19 the reference overflows there while
        # the float64 batch does not (the documented precision difference).
        rng = np.random.default_rng(18)
        X = rng.normal(1e15, 1e13, (32, 300)).astype(np.float32)
        X[:, :3] = 3e38  # never inside the window
        self._check_block(X)
        # sigma 1e-15: delta**2 (1e-30) stays a normal float32 number; below
        # ~1e-19 the reference's float32 slice sums of delta**2 underflow
        Y = rng.normal(0.0, 1e-15, (32, 300)).astype(np.float32)
        self._check_block(Y)

    def test_axis_and_shape(self):
        rng = np.random.default_rng(19)
        data = rng.normal(100.0, 5.0, (3, 40, 50))
        ref0 = mmm_batch(data, axis=0)
        ref2 = mmm_batch(data, axis=2)
        self.assertEqual(ref0[0].shape, (40, 50))
        self.assertEqual(ref2[0].shape, (3, 40))
        np.testing.assert_allclose(ref0[0], mmm_batch(np.moveaxis(data, 0, -1))[0])
        np.testing.assert_allclose(ref2[0], mmm_batch(data, axis=-1)[0])
        one = mmm_batch(data[0, 0])
        self.assertEqual(one[0].shape, ())
        self.assertAlmostEqual(float(one[0]), mmm(data[0, 0])[0], places=9)
        with self.assertRaises(ValueError):
            mmm_batch(np.float64(3.0))

    def test_empty_and_degenerate_shapes(self):
        a, b, c = mmm_batch(np.empty((0, 100)), axis=1)
        self.assertEqual(a.shape, (0,))
        a, b, c = mmm_batch(np.empty((20, 0)), axis=1)
        self.assertTrue(np.all(np.isnan(a)))

    def test_chunking_does_not_change_results(self):
        rng = np.random.default_rng(20)
        X = rng.normal(100.0, 5.0, (64, 2048)).astype(np.float32)
        X[rng.random(X.shape) < 0.03] = np.nan
        full = mmm_batch(X, axis=1, max_chunk_elements=1 << 30)
        small = mmm_batch(X, axis=1, max_chunk_elements=5 * 2048)
        for f, s in zip(full, small):
            np.testing.assert_array_equal(f, s)

    def test_scalar_dispatch_paths_agree(self):
        """Few slices / long slices go through the scalar loop; same answers."""
        rng = np.random.default_rng(21)
        X = rng.normal(100.0, 5.0, (3, 500))
        got = np.stack(mmm_batch(X, axis=1), 1)
        ref = np.array([mmm(r) for r in X], dtype=float)
        np.testing.assert_array_equal(got, ref)
        Y = rng.normal(100.0, 5.0, (20, 40000)).astype(np.float32)
        got = np.stack(mmm_batch(Y, axis=1), 1)
        ref = np.array([mmm(r) for r in Y], dtype=float)
        np.testing.assert_array_equal(got, ref)


class TestBatchProperties(unittest.TestCase):
    def test_permutation_invariance(self):
        rng = np.random.default_rng(22)
        X = rng.normal(100.0, 5.0, (32, 400))
        P = X[:, rng.permutation(400)]
        for a, b in zip(mmm_batch(X), mmm_batch(P)):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-9)

    def test_shift_and_scale(self):
        rng = np.random.default_rng(23)
        X = rng.normal(0.0, 1.0, (32, 1000))
        m0, s0, k0 = mmm_batch(X)
        m1, s1, k1 = mmm_batch(3.0 * X + 50.0)
        np.testing.assert_allclose(m1, 3.0 * m0 + 50.0, rtol=1e-9, atol=1e-7)
        np.testing.assert_allclose(s1, 3.0 * s0, rtol=1e-9, atol=1e-9)

    def test_contamination_resistance(self):
        rng = np.random.default_rng(24)
        X = rng.normal(200.0, 10.0, (64, 4096))
        X[:, :100] += 800.0
        m, s, k = mmm_batch(X)
        self.assertTrue(np.all(np.abs(m - 200.0) < 3.0))
        self.assertTrue(np.all(np.abs(s - 10.0) < 1.5))

    def test_constant_rows(self):
        X = np.full((20, 300), 7.5)
        m, s, k = mmm_batch(X)
        np.testing.assert_array_equal(m, 7.5)
        np.testing.assert_array_equal(s, 0.0)
        np.testing.assert_array_equal(k, 0.0)


class TestCallers(unittest.TestCase):
    """``destripe_helper`` / ``background_image``: with ``vectorized_mmm`` off they
    run the original code (external ``mmm`` per slice, nothing else); with it on
    they agree with that to float32 rounding."""

    @staticmethod
    def _frame(rng, level, shape=(256, 512)):
        frame = rng.normal(level, 18.0, shape).astype(np.float32)
        frame += rng.normal(0.0, 2.0, shape[0])[:, None]  # stripes
        frame[rng.random(frame.shape) < 0.04] = np.nan  # masked sources
        frame[40:43] = np.nan  # fully masked rows
        frame[100:110, 200:230] += 3000.0  # bright source
        return frame

    def test_flag_off_is_the_original_path(self):
        self.assertIs(imagetools.mmm, mmm_reference)
        frame = self._frame(np.random.default_rng(25), 1000.0)
        with (
            mock.patch.object(imagetools, "mmm", wraps=imagetools.mmm) as ext,
            mock.patch.object(imagetools, "mmm_batch") as batch,
            mock.patch.object(imagetools, "mmm_fast") as fast,
        ):
            destripe_helper(frame, smooth=True)
            self.assertEqual(ext.call_count, frame.shape[0])
            background_image(frame, 64, 3)
            self.assertEqual(ext.call_count, frame.shape[0] + frame.size // 64**2)
            batch.assert_not_called()
            fast.assert_not_called()

    def test_flag_on_uses_batch_or_fast_scalar(self):
        frame = self._frame(np.random.default_rng(25), 1000.0, shape=(512, 1024))
        with (
            mock.patch.object(imagetools, "mmm") as ext,
            mock.patch.object(
                imagetools, "mmm_batch", wraps=imagetools.mmm_batch
            ) as batch,
            mock.patch.object(
                imagetools, "mmm_fast", wraps=imagetools.mmm_fast
            ) as fast,
        ):
            destripe_helper(frame, smooth=True, vectorized_mmm=True)
            background_image(frame, 64, 3, vectorized_mmm=True)
            self.assertEqual(batch.call_count, 2)
            fast.assert_not_called()
            # 256-px tiles are sort-bound: views + fast scalar, no tile matrix
            background_image(frame, 256, 3, vectorized_mmm=True)
            self.assertEqual(batch.call_count, 2)
            self.assertEqual(fast.call_count, frame.size // 256**2)
            ext.assert_not_called()

    def test_destripe_helper_flag_on_matches_flag_off(self):
        rng = np.random.default_rng(25)
        for level in (0.0, 4000.0):
            frame = self._frame(rng, level)
            off = destripe_helper(frame, smooth=True)
            on = destripe_helper(frame, smooth=True, vectorized_mmm=True)
            self.assertTrue(np.all(np.isfinite(on)))
            np.testing.assert_allclose(on, off, atol=2e-3 * (1 + level / 4000.0))

    def test_background_image_flag_on_matches_flag_off(self):
        rng = np.random.default_rng(26)
        for level in (0.0, 4000.0):
            for mesh in (64, 256):
                frame = self._frame(rng, level, shape=(512, 1024))
                bg_off, std_off = background_image(frame, mesh, 3)
                bg_on, std_on = background_image(frame, mesh, 3, vectorized_mmm=True)
                self.assertEqual(bg_on.shape, frame.shape)
                np.testing.assert_allclose(
                    bg_on, bg_off, atol=2e-3 * (1 + level / 4000.0)
                )
                np.testing.assert_allclose(std_on, std_off, atol=1e-4)

    def test_destripe_helper_flag_on_never_fails_where_flag_off_succeeds(self):
        # codex reproducer: sigma of a few float32 ulps at 100 ADU; the batch core
        # alone ran to mxiter here, the flag-off path converged
        o = np.array(
            [
                -94,
                52,
                36,
                -46,
                30,
                7,
                -53,
                69,
                45,
                -91,
                -77,
                -81,
                -97,
                -45,
                82,
                -54,
                -4,
                93,
                -99,
                -94,
            ]
        )
        row = (np.float32(100) + o * np.spacing(np.float32(100))).astype(np.float32)
        frame = np.full((16, 2048), np.nan, dtype=np.float32)
        frame[:, : row.size] = row
        off = destripe_helper(frame, smooth=False)
        on = destripe_helper(frame, smooth=False, vectorized_mmm=True)
        self.assertTrue(np.all(np.isfinite(on)))
        np.testing.assert_allclose(on, off, atol=1e-6)


class TestValidation(unittest.TestCase):
    def test_batch_rejects_bad_arguments(self):
        rng = np.random.default_rng(27)
        X = rng.normal(100.0, 5.0, (20, 100))
        with self.assertRaises(ValueError):
            mmm_batch(X, minsky=0)
        with self.assertRaises(TypeError):
            mmm_batch(X.astype(complex))
        with self.assertRaises(TypeError):
            mmm_batch(X.astype(np.float16))
        # the scalar path still follows the reference for float16
        self.assertTrue(
            _same_triplet(
                mmm(X[0].astype(np.float16)), mmm_reference(X[0].astype(np.float16))
            )
        )
        with self.assertRaises(np.exceptions.AxisError):
            mmm_batch(X, axis=2)
        with self.assertRaises(np.exceptions.AxisError):
            mmm_batch(X, axis=-3)

    def test_highbad_with_nan_false_and_nonfinite_raises_clearly(self):
        rng = np.random.default_rng(28)
        x = rng.normal(100.0, 5.0, 60)
        x[3] = np.nan
        with self.assertRaisesRegex(ValueError, "highbad"):
            mmm(x, highbad=200.0, nan=False)
        X = np.tile(x, (20, 1))
        with self.assertRaisesRegex(ValueError, "highbad"):
            mmm_batch(X, highbad=200.0, nan=False)

    def test_row_searchsorted_nan_query_sorts_last(self):
        s = np.sort(np.arange(12.0).reshape(3, 4), axis=1)
        n = np.full(3, 4)
        vals = np.array([np.nan, 5.5, np.nan])  # row 1 holds 4..7
        res = _row_searchsorted(s, np.arange(3), n, vals, np.ones(3, dtype=bool))
        np.testing.assert_array_equal(res, [4, 2, 4])


class TestBatchNeverFailsWhereScalarSucceeds(unittest.TestCase):
    """Failed batch slices are recomputed with the scalar ``mmm`` (codex finding:
    float64 sums can keep a window moving to mxiter at float32 resolution)."""

    def test_codex_reproducer_sigma_of_a_few_ulps(self):
        o = np.array(
            [
                -94,
                52,
                36,
                -46,
                30,
                7,
                -53,
                69,
                45,
                -91,
                -77,
                -81,
                -97,
                -45,
                82,
                -54,
                -4,
                93,
                -99,
                -94,
            ]
        )
        x = (np.float32(100) + o * np.spacing(np.float32(100))).astype(np.float32)
        ref = mmm(x, minsky=10)
        self.assertTrue(np.isfinite(ref[1]))
        got = np.stack(mmm_batch(np.tile(x, (16, 1)), axis=1, minsky=10, mxiter=500), 1)
        tol = TestBatchEquivalence._tolerance(np.tile(ref, (16, 1)), np.float32)
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertTrue(np.all(np.abs(got - ref) <= tol))

    def test_codex_reproducer_high_level_tiny_sigma(self):
        rng = np.random.default_rng(6133)
        x = rng.normal(10000, 0.1, (64, 25)).astype(np.float32)[31]
        ref = mmm(x, minsky=20, mxiter=3)
        self.assertTrue(np.isfinite(ref[1]))
        got = np.stack(mmm_batch(np.tile(x, (16, 1)), axis=1, minsky=20, mxiter=3), 1)
        self.assertTrue(np.all(np.isfinite(got)))
        tol = TestBatchEquivalence._tolerance(np.tile(ref, (16, 1)), np.float32)
        self.assertTrue(np.all(np.abs(got - ref) <= tol))

    def test_near_quantisation_blocks(self):
        """Data quantised at the dtype resolution: no batch-only failures, and
        agreement within the component tolerances wherever both succeed."""
        rng = np.random.default_rng(30)
        check = TestBatchEquivalence()
        cases = [
            (np.float32, 100.0, 1e-5),
            (np.float32, 100.0, 1e-4),
            (np.float32, 10000.0, 0.1),
            (np.float32, 10000.0, 1.0),
            (np.float32, 30000.0, 0.3),
            (np.float64, 1e8, 5e-7),
            (np.float64, 1e8, 1e-6),
            (np.float64, 1e12, 1e-3),
        ]
        for dtype, level, sigma in cases:
            for n in (25, 200, 2048):
                X = rng.normal(level, sigma, (64, n)).astype(dtype)
                for mxiter in (2, 3, 50):
                    # data quantised at ~1 % of sigma: a float32-vs-float64
                    # decision at a cut may differ for a row or two per block
                    check._check_block(X, max_flips=2, minsky=10, mxiter=mxiter)


class TestDispatch(unittest.TestCase):
    def test_batch_is_vectorised(self):
        self.assertTrue(batch_is_vectorised(1024, 64**2))
        self.assertTrue(batch_is_vectorised(256, 128**2))
        self.assertTrue(batch_is_vectorised(2048, 2048))
        self.assertFalse(batch_is_vectorised(64, 256**2))  # sort-bound tiles
        self.assertFalse(batch_is_vectorised(4, 2048))  # too few slices

    def test_sparse_slices_go_to_the_scalar_path(self):
        # cost model: 20 finite of 16384 -> scalar; dense 16384 -> batch;
        # 2048-element rows -> always batch
        np.testing.assert_array_equal(
            _scalar_cheaper(np.array([20, 16000]), 16384), [True, False]
        )
        np.testing.assert_array_equal(
            _scalar_cheaper(np.array([2, 2048]), 2048), [False, False]
        )

    def test_sparse_and_dense_rows_mixed(self):
        rng = np.random.default_rng(31)
        X = rng.normal(4000.0, 18.0, (64, 16384)).astype(np.float32)
        X[::2, 20:] = np.nan  # every other tile has 20 finite values
        X[1, :] = np.nan
        TestBatchEquivalence()._check_block(X, minsky=10)
        # the sparse rows are computed by the scalar path: exact
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = np.stack(mmm_batch(X, axis=1, minsky=10), 1)
            ref = np.array([mmm(r, minsky=10) for r in X[::2]], dtype=float)
        for g, r in zip(got[::2], ref):
            self.assertTrue(_same_triplet(g, r))


if __name__ == "__main__":
    unittest.main()
