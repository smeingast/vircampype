import logging
import warnings
from functools import lru_cache

import numpy as np
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.units import Unit
from scipy.stats import sem

__all__ = [
    "get_zeropoint",
    "vega2ab",
    "get_default_extinction",
    "resampling_noise_correlation",
    "aperture_error_inflation",
    "limiting_magnitude",
    "saturation_magnitude",
]


def get_zeropoint(
    skycoord1,
    mag1,
    skycoord2,
    mag2,
    mag_limits_ref=None,
    method="weighted",
    magerr1=None,
    magerr2=None,
    plot=False,
):
    """
    Calculate zero point

    Parameters
    ----------
    skycoord1 : SkyCoord
        Astropy SkyCoord object with all coordinates of catalog to be calibrated.
    mag1 : iterable, ndarray
        Magnitudes of sources to be calibrated.
    magerr1 : iterable, ndarray
        Magnitude errors of sources to be calibrated.
    skycoord2 : SkyCoord
        Astropy SkyCoord instance for reference catalog sources.
    mag2 : iterable, ndarray
        Magnitudes of reference sources.
    magerr2 : iterable, ndarray
        Magnitude errors of reference sources.
    mag_limits_ref : tuple, optional
        Tuple of magnitude limits to be applied. e.g. (10, 15)
    method: str, optional
        Method used to calcualte ZP. Either 'median', 'weighted', or 'all'. Default is
        'weighted'.
    plot: bool, optional
        Makes a test plot for testing.
    Returns
    -------
    tuple
        Tuple holding zero point and error in zero point.

    """

    # Copy arrays
    skycoord_cal = skycoord1.copy()
    mag_cal = mag1.copy()
    mag_err_cal = magerr1.copy() if magerr1 is not None else None
    skycoord_ref = skycoord2.copy()
    mag_ref = mag2.copy()
    mag_err_ref = magerr2.copy() if magerr2 is not None else None

    # Dummy check for errors
    if (method.lower() == "weighted") & ((mag_err_cal is None) | (mag_err_ref is None)):
        raise ValueError(
            "For weighted ZP determination, please provide magnitude errors!"
        )

    # Make new array for output
    mag_diff_all = np.full_like(mag_cal, fill_value=np.nan)

    # Restrict reference catalog
    if mag_limits_ref is not None:
        keep = (mag_ref >= mag_limits_ref[0]) & (mag_ref <= mag_limits_ref[1])
        mag_ref, skycoord_ref = mag_ref[keep], skycoord_ref[keep]
        if mag_err_ref is not None:
            mag_err_ref = mag_err_ref[keep]

    # Xmatch science with reference
    zp_idx, zp_d2d, _ = skycoord_cal.match_to_catalog_sky(skycoord_ref)

    # Get good indices in reference catalog and in current field
    idx_ref = zp_idx[zp_d2d < 1 * Unit("arcsec")]
    idx_sci = np.arange(len(zp_idx))[zp_d2d < 1 * Unit("arcsec")]

    # Apply indices filter
    mag_ref = mag_ref[idx_ref]
    mag_cal = mag_cal[idx_sci]

    # Raise error if no sources are found
    if len(mag_ref) == 0:
        raise ValueError(
            "No matched sources found in reference catalog for ZP determination!"
        )
    if len(mag_ref) < 5:
        logging.getLogger(__name__).warning(
            f"Only {len(mag_ref)} matched reference star(s) for ZP determination"
        )

    # Compute ZP for each source
    mag_diff = (mag_ref - mag_cal.T).T

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # Return magdiff for all sources if requested
        if method == "all":
            mag_diff_all[idx_sci] = mag_diff
            return mag_diff_all

        # Median ZP
        if method.lower() == "median":
            # Get sigma-clipped stats
            _, zp, zp_err = sigma_clipped_stats(
                data=mag_diff, sigma=3, maxiters=3, axis=0
            )

        # Weighted ZP
        elif method.lower() == "weighted":
            # Apply match to errors
            mag_err_ref = mag_err_ref[idx_ref]
            mag_err_cal = mag_err_cal[idx_sci]

            # Sigma clip mag_diff array
            clipped = sigma_clip(mag_diff, sigma=2.5, maxiters=5, axis=0).mask
            err_tot = np.sqrt(mag_err_ref**2 + mag_err_cal.T**2)
            weights = (1 / err_tot**2).T.copy()

            # Exclude clipped sources and sources with NaN data or unusable
            # errors (NaN or zero in either catalog) from both the ZP and its error
            bad = clipped | np.isnan(mag_diff) | ~np.isfinite(weights)
            mag_diff[bad], weights[bad] = 0.0, 0.0

            # Compute weighted average
            zp = np.average(mag_diff, weights=weights, axis=0)
            temp = mag_diff.copy()
            temp[bad] = np.nan
            zp_err = sem(temp, nan_policy="omit", axis=0)

            """ I experimented with lots of options to compute the ZP error, including 
            the weighted standard deviation and also a weighted standard error of the 
            mean: 
            https://stats.stackexchange.com/questions/25895/computing-standard-error-in-weighted-mean-estimation
            All result in almost the same value, so I decided to just compute the error 
            in the ZP as the standard error of the masked difference array. """

            # Determine variance
            # zp_err = np.sqrt(np.average((mag_diff - zp)**2, weights=weights, axis=0))

        else:
            raise ValueError(
                "ZP method '{0}' not supported. Use 'median' or 'weighted'."
            )

        # For tests
        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(
                nrows=1, ncols=1, gridspec_kw=None, **dict(figsize=(7, 4))
            )
            kwargs = dict(s=30, lw=0, alpha=1, rasterized=True)
            ax.scatter(mag_ref, mag_diff, fc="crimson", **kwargs)
            if method.lower() == "weighted":
                ax.scatter(mag_ref[~bad], mag_diff[~bad], fc="green", **kwargs)
            ax.axhline(zp, c="black")
            plt.show()

        # Return
        return zp, zp_err


def vega2ab(mag, passband):
    """
    Converts Vega to AB magnitudes for the 2MASS system.
    https://iopscience.iop.org/article/10.1086/429803/pdf (Blanton 2005)

    Parameters
    ----------
    mag : array_like
        Array of magnitudes to convert.
    passband : str
        Passband (Either 'J', 'H' or 'Ks').

    Returns
    -------
    array_like
        Converted magnitudes.

    """
    if passband.lower() == "j":
        cor = 0.91
    elif passband.lower() == "h":
        cor = 1.39
    elif passband.lower() == "ks":
        cor = 1.85
    else:
        raise ValueError(f"Filter {passband} not supported")

    return mag + cor


def get_default_extinction(passband: str) -> float:
    """
    return default extinction coefficient for a near-ir passband.

    Parameters
    ----------
    passband
        passband name (e.g. "j", "h", "k", "ks").

    Returns
    -------
    float
        extinction coefficient.

    Raises
    ------
    ValueError
        if passband is not supported.
    """
    pb = passband.strip().lower()
    dextinct: dict[str, float] = {"j": 0.11, "h": 0.06, "ks": 0.07}

    if "j" in pb:
        return dextinct["j"]
    if "h" in pb:
        return dextinct["h"]
    if "k" in pb:
        return dextinct["ks"]

    raise ValueError(f"passband {passband!r} not supported")


# SWarp interpolation kernels: half-width (taps per side) of each kernel.
_SWARP_KERNEL_HALF_WIDTH = {
    "nearest": 1,
    "bilinear": 1,
    "lanczos2": 2,
    "lanczos3": 3,
    "lanczos4": 4,
}


def _swarp_kernel_weights(
    pos: np.ndarray, kernel: str
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolation weights SWarp applies at input positions ``pos`` (pixels).

    Mirrors SWarp ``src/interpolate.c``: separable kernels, Lanczos weights
    normalised to unit sum (``make_kernel``). Returns the first input tap index
    per position and a ``(len(pos), n_taps)`` weight array.
    """
    if kernel == "nearest":
        return np.round(pos).astype(int), np.ones((pos.size, 1))
    a = _SWARP_KERNEL_HALF_WIDTH[kernel]
    j0 = np.floor(pos).astype(int) - a + 1
    x = pos[:, None] - (j0[:, None] + np.arange(2 * a)[None, :])
    if kernel == "bilinear":
        w = np.clip(1.0 - np.abs(x), 0.0, None)
    else:
        w = np.sinc(x) * np.sinc(x / a)
        w[np.abs(x) >= a] = 0.0
    w /= w.sum(axis=1, keepdims=True)
    return j0, w


def _resampling_matrix(n_out: int, x0: float, ratio: float, kernel: str):
    """1-D resampling matrix ``W[k, j]``: output pixel ``k`` sits at input
    coordinate ``x0 + k * ratio`` and receives ``sum_j W[k, j] * input[j]``."""
    a = _SWARP_KERNEL_HALF_WIDTH[kernel]
    pos = x0 + np.arange(n_out) * ratio + a + 1  # keep every tap at j >= 0
    j0, w = _swarp_kernel_weights(pos, kernel)
    n_in = int(np.ceil(pos.max())) + a + 2
    out = np.zeros((n_out, n_in))
    cols = j0[:, None] + np.arange(w.shape[1])[None, :]
    out[np.repeat(np.arange(n_out), w.shape[1]), cols.ravel()] = w.ravel()
    return out


def _circular_aperture(n: int, diameter: float, cx: float, cy: float, subpix: int):
    """Fractional-area weights of a circular aperture on an ``n x n`` pixel grid."""
    s = (np.arange(n * subpix) + 0.5) / subpix
    inside = (s[None, :] - cx) ** 2 + (s[:, None] - cy) ** 2 <= (0.5 * diameter) ** 2
    return inside.reshape(n, subpix, n, subpix).mean(axis=(1, 3))


@lru_cache(maxsize=32)
def _resampling_noise_correlation_cached(
    diameters: tuple[float, ...], kernel: str, scale_ratio: float, n_phase: int
) -> tuple[float, ...]:
    phases = (np.arange(n_phase) + 0.5) / n_phase
    offsets = (np.arange(3) - 1) / 3.0  # aperture centring within an output pixel

    # Phase-averaged per-pixel variance, as measured by SExtractor on the image
    # (background-RMS meshes span many beat periods of the sampling phase).
    _, w = _swarp_kernel_weights(
        np.linspace(0.0, 1.0, 1024, endpoint=False) + 10, kernel
    )
    pix_var_1d = float(np.mean(np.sum(w**2, axis=1)))

    out = []
    for diameter in diameters:
        n = int(np.ceil(diameter)) + 4
        mats = [_resampling_matrix(n, x0, scale_ratio, kernel) for x0 in phases]
        num, den = 0.0, 0.0
        for dx in offsets:
            for dy in offsets:
                ap = _circular_aperture(n, diameter, 0.5 * n + dx, 0.5 * n + dy, 10)
                area = ap.sum()
                for wy in mats:
                    left = wy.T @ ap
                    for wx in mats:
                        num += np.sum((left @ wx) ** 2)
                        den += area * pix_var_1d**2
        out.append(float(np.sqrt(num / den)))
    return tuple(out)


def resampling_noise_correlation(
    diameters: list[float] | np.ndarray,
    kernel: str = "lanczos3",
    scale_ratio: float = 1.0,
    n_phase: int = 6,
) -> np.ndarray:
    """
    Aperture-error inflation caused by the noise correlation of SWarp resampling.

    SWarp interpolates each input frame onto the output grid with a separable
    kernel. The interpolation correlates neighbouring output pixels and lowers
    their individual variance, so SExtractor's white-noise aperture error
    (``FLUXERR_APER``: measured per-pixel variance times aperture area)
    under-estimates the true variance of the aperture sum. For white input
    noise the exact ratio is

        kappa(D)^2 = < || W_y^T A W_x ||_F^2 > / ( sum(A) * <sum_j W_kj^2>^2 )

    with ``W`` the 1-D resampling matrices, ``A`` the fractional-pixel circular
    aperture of diameter ``D`` and ``<.>`` the average over sampling phase and
    aperture centring. The expectation does not depend on the number of
    co-added frames (numerator and denominator average identically).

    Parameters
    ----------
    diameters : list[float] | np.ndarray
        Aperture diameters in output pixels (SExtractor ``PHOT_APERTURES``).
    kernel : str, optional
        SWarp ``RESAMPLING_TYPE`` (``nearest``, ``bilinear``, ``lanczos2``,
        ``lanczos3``, ``lanczos4``). Default ``lanczos3``.
    scale_ratio : float, optional
        Output / input pixel scale (e.g. 1/3" / 0.337" for VIRCAM tiles).
    n_phase : int, optional
        Sampling phases per axis in the deterministic phase average.

    Returns
    -------
    np.ndarray
        Multiplicative inflation factor for each diameter (dimensionless).
    """
    kernel = kernel.lower()
    if kernel not in _SWARP_KERNEL_HALF_WIDTH:
        raise ValueError(f"Unsupported resampling kernel '{kernel}'")
    if scale_ratio <= 0:
        raise ValueError("scale_ratio must be positive")
    diameters = tuple(float(d) for d in np.atleast_1d(diameters))
    return np.array(
        _resampling_noise_correlation_cached(
            diameters, kernel, round(float(scale_ratio), 4), int(n_phase)
        )
    )


def aperture_error_inflation(
    diameters: list[float] | np.ndarray,
    kernel: str = "lanczos3",
    scale_ratio: float = 1.0,
    noise_scale: float = 1.0,
    background_beta: float = 0.0,
    background_dmax: float | None = None,
) -> np.ndarray:
    """
    Per-aperture factor that turns SExtractor's ``MAGERR_APER`` into a realistic error.

        kappa(D)^2 = noise_scale^2 * kappa_resamp(D)^2
                     + background_beta * pi * min(D, background_dmax)^2 / 4

    ``kappa_resamp`` is the resampling noise correlation
    (:func:`resampling_noise_correlation`). ``background_beta`` adds noise that is
    coherent over the aperture (background-estimation error and sky structure on
    aperture scales, absent from ``FLUXERR_APER``); its variance grows with the
    aperture area squared, i.e. ``beta * area`` relative to the white-noise term.
    ``noise_scale`` rescales the per-pixel noise as a whole. Both are calibrated
    empirically from repeat measurements (see ``doc/correlated-noise-magerr.md``).
    The background term is held constant beyond ``background_dmax`` (the largest
    calibrated aperture): large apertures are used for bright, Poisson- and
    floor-dominated sources, where an area-growing term is neither measurable
    nor physical (the factor multiplies the whole ``MAGERR_APER``, including the
    source Poisson term).

    Parameters
    ----------
    diameters : list[float] | np.ndarray
        Aperture diameters in output pixels.
    kernel : str, optional
        SWarp resampling kernel. Default ``lanczos3``.
    scale_ratio : float, optional
        Output / input pixel scale. Default 1.0.
    noise_scale : float, optional
        Multiplicative per-pixel noise calibration ``c``. Default 1.0.
    background_beta : float, optional
        Coherent-background variance per pixel^2, in units of the per-pixel
        variance. Default 0.0.
    background_dmax : float | None, optional
        Aperture diameter (pixels) beyond which the background term stops
        growing. ``None`` (default) = no cap.

    Returns
    -------
    np.ndarray
        Inflation factor for each diameter.
    """
    if noise_scale <= 0 or background_beta < 0:
        raise ValueError("noise_scale must be > 0 and background_beta >= 0")
    diameters = np.atleast_1d(np.asarray(diameters, dtype=float))
    kappa = resampling_noise_correlation(
        diameters, kernel=kernel, scale_ratio=scale_ratio
    )
    d_bkg = (
        diameters if background_dmax is None else np.minimum(diameters, background_dmax)
    )
    area = np.pi * d_bkg**2 / 4.0
    return np.sqrt(noise_scale**2 * kappa**2 + background_beta * area)


def _bright_point_sources(table, snr_min: float = 50.0) -> np.ndarray:
    """Boolean mask of clean, well-measured point sources in a SExtractor table."""
    return (
        (table["CLASS_STAR"] > 0.8)
        & (table["FLAGS"] == 0)
        & (table["SNR_WIN"] > snr_min)
        & np.isfinite(table["FWHM_IMAGE"])
    )


def limiting_magnitude(table, catalog_header, backsig, setup, nsigma: float = 5.0):
    """
    Point-source limiting magnitude from the background noise (ESO definition).

    The limit is the magnitude of a point source whose flux in the fixed aperture
    closest to 1.35 x FWHM (the optimal aperture for a Gaussian PSF) equals
    ``nsigma`` times the aperture noise. The per-pixel background RMS ``backsig``
    is scaled to the aperture area and inflated by the correlated-noise factor of
    the resampling kernel. The per-aperture zero point ``PYPE ZP MAG_APER j``
    converts the aperture flux to a total magnitude because it already contains
    the mean aperture correction. Classification is only used for the bright stars
    that define the FWHM, so the result does not depend on the unreliable
    CLASS_STAR of faint sources.

    Parameters
    ----------
    table : fits.FITS_rec | Table
        Calibrated source catalog (needs CLASS_STAR, FLAGS, SNR_WIN, FWHM_IMAGE).
    catalog_header : fits.Header
        Catalog extension header with the ``PYPE ZP MAG_APER j`` keywords.
    backsig : float
        Background RMS per pixel of the image (ADU).
    setup : Setup
        Pipeline setup (apertures, resampling kernel, error-model constants).
    nsigma : float, optional
        Detection threshold in units of the aperture noise.

    Returns
    -------
    float
        Limiting magnitude (Vega), NaN if it cannot be determined.

    """
    try:
        stars = _bright_point_sources(table)
        if np.sum(stars) < 5:
            return np.nan
        fwhm = float(np.nanmedian(table["FWHM_IMAGE"][stars]))
        apertures = np.asarray(setup.apertures, dtype=float)
        idx = int(np.argmin(np.abs(apertures - 1.35 * fwhm)))
        zp = float(catalog_header[f"HIERARCH PYPE ZP MAG_APER {idx}"])
    except (KeyError, ValueError, TypeError):
        return np.nan
    kappa = aperture_error_inflation(
        diameters=[apertures[idx]],
        kernel=setup.resampling_kernel,
        scale_ratio=setup.coadd_pixel_scale / setup.native_pixel_scale,
        noise_scale=setup.photometric_error_noise_scale,
        background_beta=setup.photometric_error_background_beta,
        background_dmax=setup.photometric_error_background_dmax,
    )[0]
    noise = nsigma * kappa * backsig * np.sqrt(np.pi * (apertures[idx] / 2) ** 2)
    if not (np.isfinite(noise) and noise > 0):
        return np.nan
    return float(zp - 2.5 * np.log10(noise))


def saturation_magnitude(table):
    """
    Saturation limit: the calibrated total magnitude (MAG_AUTO_CAL) of the
    10th brightest unflagged point source (NaN if there are fewer than 10).
    Sources brighter than this are mostly saturated or otherwise flagged.

    Parameters
    ----------
    table : fits.FITS_rec | Table
        Calibrated source catalog (needs CLASS_STAR, FLAGS, MAG_AUTO_CAL).

    Returns
    -------
    float
        Saturation magnitude (Vega), NaN with fewer than 10 clean point sources.

    """
    try:
        stars = (table["CLASS_STAR"] > 0.8) & (table["FLAGS"] == 0)
        mags = np.asarray(table["MAG_AUTO_CAL"][stars], dtype=float)
    except (KeyError, ValueError, TypeError):
        return np.nan
    # The 10th brightest instead of the brightest, so that a few saturated stars
    # that SExtractor did not flag do not set the limit
    mags = np.sort(mags[np.isfinite(mags)])
    return float(mags[9]) if len(mags) >= 10 else np.nan
