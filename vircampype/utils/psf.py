import warnings
import numpy as np

from astropy.table import Table
from astropy.nddata import NDData
from photutils.psf import extract_stars, EPSFBuilder
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

__all__ = ["build_psf"]


def build_psf(data_image, data_weight, table_sources, psf_size=25, oversampling=2, maxiters=20):

    # Copy data to avoid parallelisation read/write issues
    image, weight = data_image.copy(), data_weight.copy()

    from astropy.utils.exceptions import AstropyUserWarning

    # Mask
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=AstropyUserWarning)
        image[weight <= 0] = np.nan
        image = interpolate_replace_nans(image, kernel=Gaussian2DKernel(1))

    # Create coordinate table
    stars_tbl = Table()
    stars_tbl["x"] = table_sources["XWIN_IMAGE"]
    stars_tbl["y"] = table_sources["YWIN_IMAGE"]

    # Extract stats
    stars = extract_stars(NDData(data=image), stars_tbl, size=psf_size)

    # Build PSF
    epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=maxiters,
                               progress_bar=False, smoothing_kernel="quadratic")
    epsf, fitted_stars = epsf_builder(stars)

    # Return EPSF model
    return epsf.data
