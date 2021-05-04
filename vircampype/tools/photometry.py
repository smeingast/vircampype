import warnings
import numpy as np

from scipy.stats import sem
from astropy.units import Unit
from astropy.stats import sigma_clip, sigma_clipped_stats

__all__ = ["get_zeropoint", "vega2ab"]


def get_zeropoint(skycoord_cal, mag_cal, skycoord_ref, mag_ref, mag_limits_ref=None,
                  method="weighted", mag_err_cal=None, mag_err_ref=None, plot=False):
    """
    Calculate zero point

    Parameters
    ----------
    skycoord_cal : SkyCoord
        Astropy SkyCoord object with all coordinates of catalog to be calibrated.
    mag_cal : iterable, ndarray
        Magnitudes of sources to be calibrated.
    mag_err_cal : iterable, ndarray
        Magnitude errors of sources to be calibrated.
    skycoord_ref : SkyCoord
        Astropy SkyCoord instance for reference catalog sources.
    mag_ref : iterable, ndarray
        Magnitudes of reference sources.
    mag_err_ref : iterable, ndarray
        Magnitude errors of reference sources.
    mag_limits_ref : tuple, optional
        Tuple of magnitude limits to be applied. e.g. (10, 15)
    method: str, optional
        Method used to calcualte ZP. Either 'median', 'weighted', or 'all'. Default is 'weighted'.
    plot: bool, optional
        Makes a test plot for testing.
    Returns
    -------
    tuple
        Tuple holding zero point and error in zero point.

    """

    # Dummy check for errors
    if (method.lower() == "weighted") & ((mag_err_cal is None) | (mag_err_ref is None)):
        raise ValueError("For weighted ZP determination, please provide magnitude errors!")

    # Make new array for output
    mag_diff_all = np.full_like(mag_cal, fill_value=np.nan)

    # Restrict reference catalog
    if mag_limits_ref is not None:
        keep = (mag_ref >= mag_limits_ref[0]) & (mag_ref <= mag_limits_ref[1])
        mag_ref = mag_ref[keep]

        # Also apply to errors if needed
        if method.lower() == "weighted":
            mag_err_ref = mag_err_ref[keep]

        skycoord_ref = skycoord_ref[keep]

    # Xmatch science with reference
    zp_idx, zp_d2d, _ = skycoord_cal.match_to_catalog_sky(skycoord_ref)

    # Get good indices in reference catalog and in current field
    idx_ref = zp_idx[zp_d2d < 1 * Unit("arcsec")]
    idx_sci = np.arange(len(zp_idx))[zp_d2d < 1 * Unit("arcsec")]

    # Apply indices filter
    mag_ref = mag_ref[idx_ref]
    mag_cal = mag_cal[idx_sci]

    # Return bad value if ZP could not be determined
    if len(mag_ref) == 0:
        return 99., 99.

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
            _, zp, zp_err = sigma_clipped_stats(data=mag_diff, sigma=3, maxiters=3, axis=0)

        # Weighted ZP
        elif method.lower() == "weighted":

            # Apply match to errors
            mag_err_ref = mag_err_ref[idx_ref]
            mag_err_cal = mag_err_cal[idx_sci]

            # Sigma clip mag_diff array and set weights of outliers to 0
            mask = sigma_clip(mag_diff, sigma=2.5, maxiters=5, axis=0).mask
            err_tot = np.sqrt(mag_err_ref**2 + mag_err_cal.T**2)
            weights = (1/err_tot**2).T
            weights[mask] = 0.

            # Compute weighted average
            zp = np.average(mag_diff, weights=weights, axis=0)
            temp = mag_diff.copy()
            temp[mask] = np.nan
            zp_err = sem(temp, nan_policy="omit", axis=0)

            """ I experimented with lots of options to compute the ZP error, including the weighted standard deviation
            and also a weighted standard error of the mean: 
            https://stats.stackexchange.com/questions/25895/computing-standard-error-in-weighted-mean-estimation
            All result in almost the same value, so I decided to just compute the error in the ZP as the standard
            error of the masked difference array.
            """

            # Determine variance
            # zp_err = np.sqrt(np.average((mag_diff - zp)**2, weights=weights, axis=0))

        else:
            raise ValueError("ZP method '{0}' not supported. Use 'median' or 'weighted'.")

        # For tests
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(nrows=1, ncols=1, gridspec_kw=None, **dict(figsize=(7, 4)))
            kwargs = dict(s=30, lw=0, alpha=1)
            ax.scatter(mag_ref, mag_diff, fc="crimson", **kwargs)
            if method.lower() == "weighted":
                ax.scatter(mag_ref[~mask], mag_diff[~mask], fc="green", **kwargs)
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
        raise ValueError("Filter {0} not supported".format(passband))

    return mag + cor
