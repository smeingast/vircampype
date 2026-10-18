"""Local cap of SExtractor's windowed-position (WIN) window at the stellar size.

SExtractor's windowed centroid uses a Gaussian window with sigma = 2 * FLUX_RADIUS / 2.35
of each source. Extended sources therefore get wide windows that sample the PSF wings;
an epoch-dependent PSF asymmetry then shifts them relative to the point sources that
calibrate the astrometry, which shows up as a spurious proper motion of galaxies. The
patched SExtractor (``WINPOS_SIGMAP``) caps the window at a map of the local stellar
window sigma, so point sources keep their own window and extended sources are held at
the local PSF size.

The map comes from the statistics machinery: ``build_statistics`` writes, per pawprint
and detector, the squared stellar window sigma measured in the pawprint's SCAMP
catalogue (``psfvar``); ``statscoadd.coadd_psf_sigma_map`` combines these with the
statistics weights; ``write_winpos_sigmap`` converts the result into the map read by
SExtractor (coarse grid with a linear pixel relation to the image, sigma in pixels).
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

__all__ = [
    "pixel_scale_arcsec",
    "stellar_window_sigma",
    "write_winpos_sigmap",
]

_FWHM_TO_SIGMA = 2.0 / 2.35  # SExtractor winpos.c: sig = hl_radius * 2.0 / 2.35


def stellar_window_sigma(
    objects,
    min_snr: float = 50.0,
    max_ellipticity: float = 0.2,
    min_stars: int = 20,
) -> float:
    """Stellar window sigma (pixels) from one detector of a SExtractor catalogue.

    Bright, unflagged, round sources; the 3-sigma-clipped median FLUX_RADIUS of them
    (the stellar locus dominates at high S/N), converted as SExtractor does
    (sigma = 2 * FLUX_RADIUS / 2.35). NaN if fewer than ``min_stars`` remain.
    """
    fr = np.asarray(objects["FLUX_RADIUS"], float)
    good = np.isfinite(fr) & (fr > 0)
    good &= np.asarray(objects["FLAGS"]) == 0
    if "FLAGS_WEIGHT" in objects.dtype.names:
        good &= np.asarray(objects["FLAGS_WEIGHT"]) == 0
    good &= np.asarray(objects["SNR_WIN"], float) >= min_snr
    good &= np.asarray(objects["ELLIPTICITY"], float) <= max_ellipticity
    fr = fr[good]
    for _ in range(5):
        if fr.size < min_stars:
            return np.nan
        med = np.median(fr)
        sig = 1.4826 * np.median(np.abs(fr - med))
        keep = np.abs(fr - med) <= 3 * max(sig, 1e-6)
        if keep.all():
            break
        fr = fr[keep]
    if fr.size < min_stars:
        return np.nan
    return float(np.median(fr) * _FWHM_TO_SIGMA)


def pixel_scale_arcsec(
    header: fits.Header, x: float | None = None, y: float | None = None
):
    """Local pixel scale (arcsec, sqrt of the pixel area) at (x, y) (0-based; centre).

    SCAMP writes TAN with PV distortion terms; these are read as TPV so that the
    distortion (which carries a few per cent of the scale) is included.
    """
    h = header.copy()
    if any(k.startswith("PV1_") for k in h) and h.get("CTYPE1", "").endswith("TAN"):
        h["CTYPE1"], h["CTYPE2"] = "RA---TPV", "DEC--TPV"
    w = WCS(h, relax=True)
    if x is None:
        x = (h.get("NAXIS1", h.get("IMAGEW", 2048)) - 1) / 2
    if y is None:
        y = (h.get("NAXIS2", h.get("IMAGEH", 2048)) - 1) / 2
    px = np.array([[x, y], [x + 1, y], [x, y + 1]], float)
    ra, dec = w.all_pix2world(px[:, 0], px[:, 1], 0)
    cosd = np.cos(np.deg2rad(dec[0]))
    dra = (ra - ra[0] + 180.0) % 360.0 - 180.0  # wrap-safe across RA = 0
    vx = np.array([dra[1] * cosd, dec[1] - dec[0]])
    vy = np.array([dra[2] * cosd, dec[2] - dec[0]])
    return float(np.sqrt(abs(vx[0] * vy[1] - vx[1] * vy[0])) * 3600.0)


def write_winpos_sigmap(
    path_sigma_arcsec: str, header_image: fits.Header, path_out: str
) -> None:
    """SExtractor ``WINPOS_SIGMAP`` from a statistics-grid window-sigma map.

    The statistics grid and the image share projection and reference point, so the
    pixel relation is linear per axis; it is derived from both WCS, checked, and
    written as ``map = WPA * image + WPB`` (1-based pixels, ``WPAX/WPBX/WPAY/WPBY``).
    Sigma is converted from arcsec to image pixels; 0 (no cap) stays 0.
    """
    with fits.open(path_sigma_arcsec) as h:
        sig = np.asarray(h[0].data, np.float32)
        hs = h[0].header
    ws, wi = WCS(hs), WCS(header_image)
    ny, nx = sig.shape
    # Image pixel positions of a grid of map pixels (both 1-based)
    gx, gy = np.meshgrid(np.linspace(1, nx, 5), np.linspace(1, ny, 5))
    ra, dec = ws.all_pix2world(gx.ravel(), gy.ravel(), 1)
    ix, iy = wi.all_world2pix(ra, dec, 1)
    ax, bx = np.polyfit(ix, gx.ravel(), 1)
    ay, by = np.polyfit(iy, gy.ravel(), 1)
    res = max(
        np.max(np.abs(ax * ix + bx - gx.ravel())),
        np.max(np.abs(ay * iy + by - gy.ravel())),
    )
    if res > 0.05:
        raise ValueError(
            f"statistics grid not linearly related to the image ({res:.3f} px)"
        )
    scale = pixel_scale_arcsec(header_image)
    out = np.where(sig > 0, sig / scale, 0.0).astype(np.float32)
    hdr = fits.Header()
    hdr["WPAX"] = (float(ax), "map x = WPAX * image x + WPBX (1-based)")
    hdr["WPBX"] = (float(bx), "map x offset")
    hdr["WPAY"] = (float(ay), "map y = WPAY * image y + WPBY (1-based)")
    hdr["WPBY"] = (float(by), "map y offset")
    hdr["WPSCALE"] = (scale, "Image pixel scale (arcsec) used for sigma")
    hdr["BUNIT"] = ("pixel", "Maximum WIN window sigma (image pixels), 0 = no cap")
    fits.writeto(path_out, out, hdr, overwrite=True)
