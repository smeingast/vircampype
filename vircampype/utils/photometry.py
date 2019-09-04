# =========================================================================== #
# Import
import warnings
import numpy as np

from astropy.modeling import models, fitting


def get_aperture_correction(diameters, magnitudes, func="Moffat"):

    # Subtract last aperture from mag to obtain aperture correction for each source
    mag_diff = magnitudes[:, -1][:, np.newaxis] - magnitudes

    # The aperture correction is now the median across all remaining sources
    mag_apcor = np.nanmedian(mag_diff, axis=0)
    magerr_apcor = np.nanstd(mag_diff, axis=0)

    # Choose model based on option
    if func.lower() == "moffat":
        model_init = models.Moffat1D(amplitude=-100, x_0=-1.75, gamma=0.5, alpha=1.0,
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
    fit_m = fitting.SimplexLSQFitter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The fit may be unsuccessful")
        # noinspection PyTypeChecker
        model = fit_m(model_init, diameters, mag_apcor)

    # Dummy check sums of squares
    ss = np.sum((mag_apcor - model(diameters))**2)
    if ss > 0.3:
        raise ValueError("Fit did not converge")

    # Return aperture correction and model
    return mag_apcor, magerr_apcor, model
