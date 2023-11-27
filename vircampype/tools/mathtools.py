import warnings
import numpy as np

from astropy.units import Unit
from fractions import Fraction
from typing import Union, Callable
from astropy.coordinates import SkyCoord
from vircampype.tools.miscellaneous import *
from astropy.stats import sigma_clipped_stats


__all__ = [
    "apply_sigma_clip",
    "linearize_data",
    "apply_along_axes",
    "ceil_value",
    "floor_value",
    "meshgrid",
    "estimate_background",
    "centroid_sphere",
    "clipped_median",
    "clipped_stdev",
    "fraction2float",
    "round_decimals_up",
    "round_decimals_down",
    "linearity_fitfunc",
    "clipped_mean",
    "cart2pol",
]


def apply_sigma_clip(
    data: np.ndarray,
    sigma_level: Union[int, float] = 3,
    sigma_iter: int = 1,
    center_metric: Callable = np.nanmedian,
    axis: int = 0
) -> np.ndarray:
    """
    Performs sigma clipping of data.

    Parameters
    ----------
    data : ndarray
        Input data.
    sigma_level : int, float, optional
        Sigma level for clipping (e.g. 3-sigma).
    sigma_iter : int, optional
        Number of iterations.
    center_metric : callable, optional
        Metric which calculates the center around which clipping occurs.
    axis : int, optional
        Axis along which to perform clipping.

    Returns
    -------
    np.ndarray
        Array with clipped values set to NaN.

    """

    # Ignore the numpy warnings upon masking
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for _ in range(sigma_iter):

            # Calculate standard deviation
            std = np.nanstd(data, axis=axis)

            # find values outside limits and set to NaN
            data[
                (data > center_metric(data, axis=axis) + sigma_level * std)
                | (data < center_metric(data, axis=axis) - sigma_level * std)
            ] = np.nan

    # Return the clipped array
    return data


def clipped_mean(data, **kwargs):
    """Helper function to return the clipped mean of an array via astropy."""
    return sigma_clipped_stats(data, **kwargs)[0]  # noqa


def clipped_median(data, **kwargs):
    """Helper function to return the clipped median of an array via astropy."""
    return sigma_clipped_stats(data, **kwargs)[1]  # noqa


def clipped_stdev(data, **kwargs):
    """Helper function to return the clipped median of an array via astropy."""
    return sigma_clipped_stats(data, **kwargs)[2]  # noqa


def cuberoot_idl(
    c0: (int, float), c1: (int, float), c2: (int, float), c3: (int, float)
):
    """
    Copied from the IDL implementation.
    Function to return the roots of a cubic polynomial c0 + c1*x + c2*x^2 + c3*x^3 = 0


    Parameters
    ----------
    c0 : int, float
        0th order polynomial coefficient.
    c1 : int, float
        1st order polynomial coefficient.
    c2 : int, float
        2nd order polynomial coefficient.
    c3 : int, float
        3rd order polynomial coefficient.

    Returns
    -------
    tuple
        Tuple containing the three possible (real) solutions.

    """

    # Get data into shape
    c1 = np.full_like(c0, fill_value=c1)
    c2 = np.full_like(c0, fill_value=c2)
    c3 = np.full_like(c0, fill_value=c3)

    # Make solution arrays
    solution1 = np.full_like(c0, fill_value=np.nan)
    solution2 = np.full_like(c0, fill_value=np.nan)
    solution3 = np.full_like(c0, fill_value=np.nan)

    # Normalize to a + bx + cx^2 + x^3=0
    a, b, c = c0 / c3, c1 / c3, c2 / c3

    q = (c ** 2 - 3 * b) / 9
    r = (2 * c ** 3 - 9 * c * b + 27 * a) / 54

    index1 = r ** 2 < q ** 3
    index2 = ~index1
    count1, count2 = np.sum(index1), np.sum(index2)

    # Filter case r^2 < q^3
    if count1 > 0:
        rf = r[index1]
        qf = q[index1]
        cf = c[index1]

        theta = np.arccos(rf / qf ** 1.5)
        solution1[index1] = -2 * np.sqrt(qf) * np.cos(theta / 3) - cf / 3
        solution2[index1] = -2 * np.sqrt(qf) * np.cos((theta + 2 * np.pi) / 3) - cf / 3
        solution3[index1] = -2 * np.sqrt(qf) * np.cos((theta - 2 * np.pi) / 3) - cf / 3

    # All other cases
    if count2 > 0:

        rf = r[index2]
        qf = q[index2]
        cf = c[index2]

        # Get the one real root
        h = -rf / np.abs(rf) * (np.abs(rf) + np.sqrt(rf ** 2 - qf ** 3)) ** (1 / 3)
        k = h.copy()

        zindex = np.isclose(h, 0, atol=1.0e-5)
        cindex = ~zindex
        zcount, ccount = np.sum(zindex), np.sum(cindex)
        if zcount > 0:
            k[zindex] = 0.0
        if ccount > 0:
            k[cindex] = qf / h

        solution1[index2] = (h + k) - cf / 3
        solution2[index2] = np.nan
        solution3[index2] = np.nan

    # Return solutions
    return solution1, solution2, solution3


def cuberoot(a, b, c, d, return_real=False):
    """
    Function to return the roots of a cubic polynomial a + bx + cx^2 + dx^3 = 0
    Uses the general formula listed in Wikipedia.

    Parameters
    ----------
    a : np.ndarray, float, int
        Intercept.
    b : float, int
        Linear coefficient.
    c : float, int
        Square coefficient.
    d : float, int
        Cubic coefficient.
    return_real : bool, optional
        If set, only return the real part of the solutions.

    Returns
    -------
    np.ndarray
        Roots of polynomial.

    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in true_divide"
        )

        # Transform to complex numbers
        d, c, b = complex(d), complex(c), complex(b)

        # Calculate stuff to get the roots
        delta0, delta1 = (
            c ** 2.0 - 3.0 * d * b,
            2.0 * c ** 3.0 - 9.0 * d * c * b + 27.0 * d ** 2.0 * a,
        )
        z = ((delta1 + np.sqrt(delta1 ** 2.0 - 4.0 * delta0 ** 3.0)) / 2.0) ** (
            1.0 / 3.0
        )

        u1, u2, u3 = 1.0, (-1.0 + 1j * np.sqrt(3)) / 2.0, (-1.0 - 1j * np.sqrt(3)) / 2.0

        # Just return real part
        if return_real:
            return [
                (-(1.0 / (3.0 * d)) * (c + u * z + (delta0 / (u * z)))).real
                for u in [u1, u2, u3]
            ]

        # Return all solutions
        else:
            return [
                -(1.0 / (3.0 * d)) * (c + u * z + (delta0 / (u * z)))
                for u in [u1, u2, u3]
            ]


def squareroot(a, b, c, return_real=False):
    """
    Function to return the roots of a quadratic polynomial a + bx + cx^2 = 0

    Parameters
    ----------
    a : np.ndarray, float, int
        Intercept.
    b : float, int
        Linear coefficient.
    c : float, int
        Square coefficient.
    return_real : bool, optional
        If set, only return the real part of the solutions.

    Returns
    -------
    np.ndarray
        Roots of polynomial

    """

    # Transform to complex numbers
    c, b = complex(c), complex(b)

    # Calculate stuff to get the roots
    delta = np.sqrt(b ** 2.0 - 4 * c * a)
    x1, x2 = (-b + delta) / (2 * c), (-b - delta) / (2 * c)

    # Just return real part
    if return_real:
        return [x1.real, x2.real]

    # Return all solutions
    else:
        return [x1, x2]


def linearity_fitfunc(x, b1, b2, b3):
    """Fitting function used for non-linearity correction."""
    coeff, mindit = [b1, b2, b3], 1.0011
    kk = mindit / x
    return np.sum(
        [
            coeff[j - 1] * x ** j * ((1 + kk) ** j - kk ** j)
            for j in range(1, len(coeff) + 1)
        ],
        axis=0,
    )


def linearize_data(data, coeff, texptime, reset_read_overhead):
    """
    General single-threaded linearization for arbitrary input data.

    Parameters
    ----------
    data : np.ndarray
        Input data to be linearized.
    coeff : list[floats], ndarray
        List of coefficients.
    texptime : int, float
        Total exptime.
    reset_read_overhead : float
        Reset-read-overhead in seconds.

    Returns
    -------
    np.ndarray
        Linearized data

    """

    # Dummy check
    if not isinstance(texptime, (int, float)):
        raise ValueError("texptime must be of type float or int.")

    # Determine order of fit
    order = len(coeff) - 1

    # Coefficient modification factor based on TEXPTIME and reset overhead
    kk = reset_read_overhead / texptime
    f = (1 + kk) ** np.arange(order + 1) - kk ** np.arange(order + 1)

    # Set coeff modifier to 1 for tests
    # f[:] = 1.

    # Copy, apply modification, and set intercept to data for inversion
    coeff_copy = list(coeff.copy() * f)
    coeff_copy[0] -= data.ravel()

    # Get the roots of all data points
    if order == 2:
        roots = squareroot(*coeff_copy, return_real=True)
    elif order == 3:
        roots = cuberoot_idl(*coeff_copy)
    else:
        raise ValueError("Order '{0}' not supported".format(order))

    # Select closest value from the real roots
    data_lin = (
        np.nanmin(np.abs([r - coeff[0] + coeff_copy[0] for r in roots]), axis=0)
        + data.ravel()
    ).reshape(data.shape)

    # Mask too large values
    data_lin[data_lin > 65536] = np.nan

    # Reset negative values just in case
    data_lin[data < 0] = data[data < 0]

    return data_lin


def apply_along_axes(array, method="median", axis=None, norm=True, copy=True):
    """
    Destripes arbitrary input arrays.

    Parameters
    ----------
    array : np.ndarray
        Array to destripe.
    method : callable
        Method to apply along given axes. Default is np.nanmedian.
    axis : int, tuple[int]
        Axes along which to destripe.
    norm : bool, optional
        Whether to normalize the data.
    copy : bool, optional
        Whether the data should be copied. If false, the original array will be
        overwritten.

    Returns
    -------
    ndarray
        Modified array.

    """

    # Create copy if set
    if copy:
        array = array.copy()

    # Fetch function
    func = string2func(method)

    # Calculate median along axis
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        med = func(array, axis=axis)

    # Expand dimensions in case of tuple for array broadcasting
    if axis is not None:
        if isinstance(axis, tuple):
            for ax in axis:
                med = np.expand_dims(med, ax)
        else:
            med = np.expand_dims(med, axis=axis)

    # Normalize if set
    if norm:
        return array - med + func(array)
    else:
        return array - med


def ceil_value(data, value):
    """
    Round data to a given value.

    Parameters
    ----------
    data : int, float, np.ndarray
    value : in, float

    Returns
    -------
    int, float, np.ndarray
        Rounded data.
    """

    return np.ceil(data / value) * value


def floor_value(data, value):
    """
    Round data to a given value.

    Parameters
    ----------
    data : int, float, np.ndarray
    value : in, float

    Returns
    -------
    int, float, np.ndarray
        Rounded data.
    """

    return np.floor(data / value) * value


def meshgrid(array, size=128):
    """
    Generates a pixel coordinate grid from an array with given mesh sizes. The mesh
    size is approximated when the data shape is not a multiple of the mesh size (almost
    always the case). For smaller mesh sizes, the output mesh will match the input more
    closely.

    Parameters
    ----------
    array : np.ndarray
        Input data for which the pixel grid should be generated
    size : int, float, tuple, optional
        Size of the grid. The larger this value rel to the image dimensions, the less
        the requested grid size will match to the output grid size.

    Returns
    -------
    tuple
        Tuple for grid coordinates in each axis. Pixel coordinates start with index 0!

    Raises
    ------
    ValueError
        In case the data has more than 3 dimensions (can be extended easily)

    """

    if isinstance(size, int):
        size = [
            size,
        ] * array.ndim

    # Get number of cells per axis
    n = [np.ceil(sh / si) for sh, si in zip(array.shape, size)]

    """In contrast to meshgrid, this has the 
    advantage that the edges are always included!"""
    # Return
    if array.ndim == 1:
        return np.uint32((np.mgrid[0: array.shape[0] - 1: complex(n[0])]))
    if array.ndim == 2:
        return np.uint32(
            (
                np.mgrid[
                    0: array.shape[0] - 1: complex(n[0]),
                    0: array.shape[1] - 1: complex(n[1]),
                ]
            )
        )
    if array.ndim == 3:
        return np.uint32(
            (
                np.mgrid[
                    0: array.shape[0] - 1: complex(n[0]),
                    0: array.shape[1] - 1: complex(n[1]),
                    0: array.shape[2] - 1: complex(n[2]),
                ]
            )
        )
    else:
        raise ValueError("{0:d}-dimensional data not supported".format(array.ndim))


def estimate_background(array, max_iter=20, force_clipping=True, axis=None):
    """
    Estimates the background sky level based on an iterative 3-sigma clipping
    algorithm. In principle the data are iterativley clipped around the median. At each
    iteration the mean of the clipped histogram is calculated. If the change from one
    iteration to the next is less than 1%, estimates for the background and standard
    deviation in the background are returned. Here, we return the mean if the last
    iteration did not show more than 20% relative change compared to the fist
    iteration. Otherwise, the field is assumed to be crowded and the mode is estimated
    with 2.5 * median - 1.5 * mean (see SExtractor doc). Ultimatley at the 'max_iter'
    iteration, the mode estimate is always returned.

    Parameters
    ----------
    array : np.ndarray
        Input data
    max_iter : int, optional
        Maximum iterations. If convergence is not reached, return an estimate of the
        background
    force_clipping : bool, optional
        If set, then even without convergence, the result will be returned after
        max_iter.
    axis : int, tuple, optional
        The axis along which the sky background should be analaysed

    Returns
    -------
    tuple
        Sky background and sky sigma estimates

    """

    # Check for integer data
    if "int" in str(array.dtype).lower():
        raise TypeError("integer data not supported")

    # Immediately return if mostly bad input data
    if np.sum(~np.isfinite(array)) > 0.9 * array.size:
        return np.nan, np.nan

    masked, idx, sky_save, sky_ini = array.copy(), 0, None, None
    while True:

        # 3-sigma clip each plane around the median
        med = np.nanmedian(masked, axis=axis)
        std = np.nanstd(masked, axis=axis)

        # Expand dimensions if required
        if axis is not None:

            # If its an integer, we just expand once
            if isinstance(axis, int):
                med = np.expand_dims(med, axis=axis)
                std = np.expand_dims(std, axis=axis)

            # If it's a tuple, we need to iteratively add axes
            if isinstance(axis, tuple):
                for i in axis:
                    med = np.expand_dims(med, axis=i)
                    std = np.expand_dims(std, axis=i)

        # Mask everything outside 3-sigma around the median in each plane
        with np.errstate(invalid="ignore"):
            masked[(masked < med - 3 * std) | (masked > med + 3 * std)] = np.nan

        # Estimate the sky and sigma with the mean of the clipped histogram
        sky, skysig = np.nanmean(masked, axis=axis), np.nanstd(masked, axis=axis)

        # Save this value in first iteration
        if idx == 0:
            sky_ini = sky.copy()

        # Only upon the second iteration we evaluate
        elif (idx > 0) & (idx < max_iter):

            # If we have no change within 2% of previous iteration we return
            if np.mean(np.abs(sky_save / sky - 1)) < 0.02:

                # If (compared to the initial value) the mean has changed by less
                # than 20%, the field is not crowded
                if np.mean(np.abs(sky / sky_ini - 1)) < 0.2 or force_clipping is True:
                    return sky, skysig

                # Otherwise the field is crowded and we return an estimate of the mode
                else:
                    return (
                        2.5 * np.nanmedian(masked, axis=axis)
                        - 1.5 * np.nanmean(masked, axis=axis),
                        skysig,
                    )

            # Otherwise we do one more iteration
            else:
                pass

        # If we have no convergence after 10 iterations, we return an estimate
        elif idx >= max_iter:

            if force_clipping is True:
                return sky, skysig
            else:
                return (
                    2.5 * np.nanmedian(masked, axis=axis)
                    - 1.5 * np.nanmean(masked, axis=axis),
                    skysig,
                )

        # For the iterations between 1 and 10 when no convergence is reached
        sky_save = sky.copy()

        # Increase loop index
        idx += 1


def centroid_sphere(skycoord):
    """
    Calculate the centroid on a sphere. Strictly valid only for a unit sphere and for a
    coordinate system with latitudes from -90 to 90 degrees and longitudes from 0 to
    360 degrees.

    Parameters
    ----------
    skycoord : SkyCoord
        SkyCoord instance

    Returns
    -------
    SkyCoord
        Centroid as SkyCoord instance.

    """

    # Filter finite entries in arrays
    good = np.isfinite(skycoord.spherical.lon) & np.isfinite(skycoord.spherical.lat)

    # 3D mean
    mean_x = np.mean(skycoord[good].cartesian.x)
    mean_y = np.mean(skycoord[good].cartesian.y)
    mean_z = np.mean(skycoord[good].cartesian.z)

    # Push mean to triangle surface
    cenlen = np.sqrt(mean_x ** 2 + mean_y ** 2 + mean_z ** 2)
    xsur, ysur, zsur = mean_x / cenlen, mean_y / cenlen, mean_z / cenlen

    # Convert back to spherical coordinates and return
    outlon = np.arctan2(ysur, xsur)

    # Convert back to 0-2pi range if necessary
    if outlon < 0:
        outlon += 2 * np.pi * Unit("rad")
    outlat = np.arcsin(zsur)

    return SkyCoord(outlon, outlat, frame=skycoord.frame)


def fraction2float(fraction):
    """
    Converts a fraction given by a string to a float

    Parameters
    ----------
    fraction : str
        String. e.g. '1/3'.

    Returns
    -------
    float
        Converted fraction
    """
    return float(Fraction(fraction))


def round_decimals_up(number: float, decimals: int = 2):
    """Returns a value rounded up to a specific number of decimal places."""
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return np.ceil(number)

    factor = 10 ** decimals
    return np.ceil(number * factor) / factor


def round_decimals_down(number: float, decimals: int = 2):
    """Returns a value rounded down to a specific number of decimal places."""
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return np.floor(number)

    factor = 10 ** decimals
    return np.floor(number * factor) / factor


def cart2pol(x, y):
    """
    Transforms cartesian to polar coordinates

    Parameters
    ----------
    x: int, float, np.ndarray
    y: int, float, np.ndarray

    Returns
    -------
    float, float
        Theta and Rho polar coordinates

    """
    return np.arctan2(y, x), np.hypot(x, y)
