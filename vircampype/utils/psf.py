import numpy as np

from astropy.io import fits
from scipy.ndimage import map_coordinates
from scipy.interpolate import UnivariateSpline

# __all__ = ["build_psf"]


# def build_psf(data_image, data_weight, table_sources, psf_size=25, oversampling=2, maxiters=20):
#
#     # Copy data to avoid parallelisation read/write issues
#     image, weight = data_image.copy(), data_weight.copy()
#
#     from astropy.utils.exceptions import AstropyUserWarning
#
#     # Mask
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", category=AstropyUserWarning)
#         image[weight <= 0] = np.nan
#         image = interpolate_replace_nans(image, kernel=Gaussian2DKernel(1))
#
#     # Create coordinate table
#     stars_tbl = Table()
#     stars_tbl["x"] = table_sources["XWIN_IMAGE"]
#     stars_tbl["y"] = table_sources["YWIN_IMAGE"]
#
#     # Extract stats
#     stars = extract_stars(NDData(data=image), stars_tbl, size=psf_size)
#
#     # Build PSF
#     epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=maxiters,
#                                progress_bar=False, smoothing_kernel="quadratic")
#     epsf, fitted_stars = epsf_builder(stars)
#
#     # Return EPSF model
#     return epsf.data


def interpolate_fwhm(axis, scale=1):
    axis /= np.max(axis)
    axis -= 0.5
    x = np.linspace(0, len(axis), len(axis))

    # Do the interpolation
    spline = UnivariateSpline(x, axis, s=0)
    r1, r2 = spline.roots()

    return (r2 - r1) * scale


def snapshots2fwhm(path_snapshot, n_snap):

    # Get extensions
    with fits.open(path_snapshot) as snapshot:

        # Figure out which extensions to use
        idx_ext = [i for i, j in enumerate(snapshot) if isinstance(j, fits.ImageHDU)]

        # Read snapshot data
        data_snap = [snapshot[i].data for i in idx_ext]

    fwhm_ext = []
    for data in data_snap:

        # Determine number of tiles
        m, n = data.shape[0] // n_snap, data.shape[1] // n_snap

        # Chop snaphot into individual PSFs
        tiles = [data[x:x+m, y:y+n] for x in range(0, data.shape[0], m) for y in range(0, data.shape[1], n)]

        fwhm_tile = []
        for tile in tiles:

            # Get vertical and horizontal profiles
            horizontal = np.take(tile, int(tile.shape[0] / 2), axis=0)
            vertical = np.take(tile, int(tile.shape[1] / 2), axis=1)

            # Get also arrays across snapshot
            ndiag = 100
            scale_diag = np.sqrt(np.sum(np.power(tile.shape, 2))) / ndiag
            x, y = np.linspace(0, tile.shape[0] - 1, ndiag), np.linspace(0, tile.shape[1] - 1, ndiag)
            diag1 = map_coordinates(tile, np.vstack((x, y)))
            diag2 = map_coordinates(tile, np.vstack((x, y[::-1])))

            # Get the 4 different FWHM measurements
            scales = [1, 1, scale_diag, scale_diag]
            fwhm = [interpolate_fwhm(x, scale=s) for x, s in zip([horizontal, vertical, diag1, diag2], scales)]

            # Average the four measurements for this PSF
            fwhm_tile.append(np.mean(fwhm))

        fwhm_ext.append(fwhm_tile)
    return fwhm_ext
