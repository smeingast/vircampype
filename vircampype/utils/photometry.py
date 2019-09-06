# =========================================================================== #
# Import
import warnings
import numpy as np

from astropy import modeling
from astropy.units import Unit
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord

# Define objects in this module
__all__ = ["get_aperture_correction", "get_zeropoint", "get_zeropoint_radec"]


def get_aperture_correction(diameters, magnitudes, func="Moffat"):

    # Subtract last aperture from mag to obtain aperture correction for each source
    mag_diff = magnitudes[:, -1][:, np.newaxis] - magnitudes

    # The aperture correction is now the median across all remaining sources
    mag_apcor = np.nanmedian(mag_diff, axis=0)
    magerr_apcor = np.nanstd(mag_diff, axis=0)

    # Choose model based on option
    if func.lower() == "moffat":
        model_init = modeling.models.Moffat1D(amplitude=-100, x_0=-1.75, gamma=0.5, alpha=1.0, name="moffat1d",
                                              bounds={"amplitude": (-300, -20), "x_0": (-3., 0),
                                                      "gamma": (0.1, 0.9), "alpha": (0.5, 1.5)})
    # TODO: Write initial guess and bounds that make sense before allowing these options
    # elif func.lower() == "gaussian":
    #     model_init = models.Gaussian1D()
    # elif func.lower() == "lorentz":
    #     model_init = models.Lorentz1D()
    else:
        raise ValueError("Requested fit function '{0}' not available.".format(func))

    # Fit profile
    fit_m = modeling.fitting.SimplexLSQFitter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The fit may be unsuccessful")
        # noinspection PyTypeChecker
        model = fit_m(model_init, diameters, mag_apcor)

    # Dummy check sums of squares
    ss = np.sum((mag_apcor - model(diameters))**2)
    if ss > 0.5:
        raise ValueError("Fit did not converge")

    # Return aperture correction and model
    return mag_apcor, magerr_apcor, model


def get_zeropoint(skycoo_cal, mag_cal, skycoo_ref, mag_ref, mag_ref_limits=None):
    """
    Calculate zero point

    Parameters
    ----------
    skycoo_cal : SkyCoord
        Astropy SkyCoord object with all coordinates of catalog to be calibrated.
    mag_cal : iterable, ndarray
        Magnitudes of sources to be calibrated.
    skycoo_ref : SkyCoord
        Astropy SkyCoord instance for reference catalog sources.
    mag_ref : iterable, ndarray
        Magnitudes of reference sources
    mag_ref_limits : tuple, optional
        Tuple of magnitude limits to be applied. e.g. (10, 15)

    Returns
    -------
    (float, float)
        Tuple holding zero point and error in zero point.

    """

    # Restrict reference catalog
    if mag_ref_limits is not None:
        keep = (mag_ref >= mag_ref_limits[0]) & (mag_ref <= mag_ref_limits[1])
        mag_ref = mag_ref[keep]
        skycoo_ref = skycoo_ref[keep]

    # Xmatch science with reference
    zp_idx, zp_d2d, _ = skycoo_cal.match_to_catalog_sky(skycoo_ref)

    # Get good indices in reference catalog and in current field
    idx_ref = zp_idx[zp_d2d < 1 * Unit("arcsec")]
    idx_sci = np.arange(len(zp_idx))[zp_d2d < 1 * Unit("arcsec")]

    # Apply indices filter
    mag_ref = mag_ref[idx_ref]
    mag_cal = mag_cal[idx_sci]

    # Apply sigma clipping
    zp_cal = mag_ref - mag_cal
    zp_cal = sigma_clip(data=zp_cal, sigma=3, maxiters=3, copy=True).filled(fill_value=np.nan)

    # Return ZP and standard deviation
    return np.nanmedian(zp_cal), np.nanstd(zp_cal)


def get_zeropoint_radec(ra_cal, dec_cal, ra_ref, dec_ref, **kwargs):
    """ Convenience method """
    sc = SkyCoord(ra=ra_cal, dec=dec_cal, frame="icrs", unit="degree")
    sc_ref = SkyCoord(ra=ra_ref, dec=dec_ref, frame="icrs", unit="degree")
    return get_zeropoint(skycoo_cal=sc, skycoo_ref=sc_ref, **kwargs)
