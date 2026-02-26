import warnings
from fractions import Fraction
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.units import Unit
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

from vircampype.tools.miscellaneous import *

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
    "get_nearest_neighbors",
    "interpolate_value",
    "find_neighbors_within_distance",
    "convert_position_error",
]


def apply_sigma_clip(
    data: np.ndarray,
    sigma_level: Union[int, float] = 3,
    sigma_iter: int = 1,
    center_metric: Callable = np.nanmedian,
    axis: int = 0,
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


def clipped_mean(data: np.ndarray, **kwargs) -> float:
    """
    Return the sigma-clipped mean of an array.

    Thin wrapper around ``astropy.stats.sigma_clipped_stats``.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    **kwargs
        Additional keyword arguments forwarded to ``sigma_clipped_stats``.

    Returns
    -------
    float
        Sigma-clipped mean.
    """
    return sigma_clipped_stats(data, **kwargs)[0]  # noqa


def clipped_median(data: np.ndarray, **kwargs) -> float:
    """
    Return the sigma-clipped median of an array.

    Thin wrapper around ``astropy.stats.sigma_clipped_stats``.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    **kwargs
        Additional keyword arguments forwarded to ``sigma_clipped_stats``.

    Returns
    -------
    float
        Sigma-clipped median.
    """
    return sigma_clipped_stats(data, **kwargs)[1]  # noqa


def clipped_stdev(data: np.ndarray, **kwargs) -> float:
    """
    Return the sigma-clipped standard deviation of an array.

    Thin wrapper around ``astropy.stats.sigma_clipped_stats``.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    **kwargs
        Additional keyword arguments forwarded to ``sigma_clipped_stats``.

    Returns
    -------
    float
        Sigma-clipped standard deviation.
    """
    return sigma_clipped_stats(data, **kwargs)[2]  # noqa


def cuberoot_idl(
    c0: Union[int, float],
    c1: Union[int, float],
    c2: Union[int, float],
    c3: Union[int, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    q = (c**2 - 3 * b) / 9
    r = (2 * c**3 - 9 * c * b + 27 * a) / 54

    index1 = r**2 < q**3
    index2 = ~index1
    count1, count2 = np.sum(index1), np.sum(index2)

    # Filter case r^2 < q^3
    if count1 > 0:
        rf = r[index1]
        qf = q[index1]
        cf = c[index1]

        theta = np.arccos(rf / qf**1.5)
        solution1[index1] = -2 * np.sqrt(qf) * np.cos(theta / 3) - cf / 3
        solution2[index1] = -2 * np.sqrt(qf) * np.cos((theta + 2 * np.pi) / 3) - cf / 3
        solution3[index1] = -2 * np.sqrt(qf) * np.cos((theta - 2 * np.pi) / 3) - cf / 3

    # All other cases
    if count2 > 0:
        rf = r[index2]
        qf = q[index2]
        cf = c[index2]

        # Get the one real root
        h = -rf / np.abs(rf) * (np.abs(rf) + np.sqrt(rf**2 - qf**3)) ** (1 / 3)
        k = h.copy()

        zindex = np.isclose(h, 0, atol=1.0e-5)
        cindex = ~zindex
        zcount, ccount = np.sum(zindex), np.sum(cindex)
        if zcount > 0:
            k[zindex] = 0.0
        if ccount > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                k[cindex] = qf[cindex] / h[cindex]

        solution1[index2] = (h + k) - cf / 3
        solution2[index2] = np.nan
        solution3[index2] = np.nan

    # Return solutions
    return solution1, solution2, solution3


def cuberoot(
    a: Union[np.ndarray, float, int],
    b: Union[float, int],
    c: Union[float, int],
    d: Union[float, int],
    return_real: bool = False,
) -> list:
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
            c**2.0 - 3.0 * d * b,
            2.0 * c**3.0 - 9.0 * d * c * b + 27.0 * d**2.0 * a,
        )
        z = ((delta1 + np.sqrt(delta1**2.0 - 4.0 * delta0**3.0)) / 2.0) ** (1.0 / 3.0)

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


def squareroot(
    a: Union[np.ndarray, float, int],
    b: Union[float, int],
    c: Union[float, int],
    return_real: bool = False,
) -> list:
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
    delta = np.sqrt(b**2.0 - 4 * c * a)
    x1, x2 = (-b + delta) / (2 * c), (-b - delta) / (2 * c)

    # Just return real part
    if return_real:
        return [x1.real, x2.real]

    # Return all solutions
    else:
        return [x1, x2]


def linearity_fitfunc(
    x: np.ndarray,
    b1: float,
    b2: float,
    b3: float,
) -> np.ndarray:
    """
    Fitting function used for non-linearity correction.

    Evaluates a polynomial model that accounts for the reset-read overhead
    in VIRCAM detector integrations.

    Parameters
    ----------
    x : np.ndarray
        Input pixel values (ADU).
    b1 : float
        First-order polynomial coefficient.
    b2 : float
        Second-order polynomial coefficient.
    b3 : float
        Third-order polynomial coefficient.

    Returns
    -------
    np.ndarray
        Model pixel values after applying the linearity correction.
    """
    coeff, mindit = [b1, b2, b3], 1.0011
    kk = mindit / x
    return np.sum(
        [
            coeff[j - 1] * x**j * ((1 + kk) ** j - kk**j)
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
        raise ValueError(f"Order '{order}' not supported")

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
    method : str
        Method to apply along given axes. Default is "median".
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
        return np.uint32((np.mgrid[0 : array.shape[0] - 1 : complex(n[0])]))
    if array.ndim == 2:
        return np.uint32(
            (
                np.mgrid[
                    0 : array.shape[0] - 1 : complex(n[0]),
                    0 : array.shape[1] - 1 : complex(n[1]),
                ]
            )
        )
    if array.ndim == 3:
        return np.uint32(
            (
                np.mgrid[
                    0 : array.shape[0] - 1 : complex(n[0]),
                    0 : array.shape[1] - 1 : complex(n[1]),
                    0 : array.shape[2] - 1 : complex(n[2]),
                ]
            )
        )
    else:
        raise ValueError(f"{array.ndim:d}-dimensional data not supported")


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
            with np.errstate(divide="ignore", invalid="ignore"):
                converged = np.mean(np.abs(sky_save / sky - 1)) < 0.02
            if converged:
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
    cenlen = np.sqrt(mean_x**2 + mean_y**2 + mean_z**2)
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


def round_decimals_up(
    number: Union[float, np.ndarray], decimals: int = 2
) -> Union[float, np.ndarray]:
    """
    Rounds a number or each element of an array up to a specific number of decimal places.

    Parameters:
    - number (float or np.ndarray): The number or array of numbers to round up.
    - decimals (int): The number of decimal places to round up to. Must be a non-negative integer.

    Returns:
    - float or np.ndarray: The rounded up number or array of numbers.

    Raises:
    - TypeError: If 'decimals' is not an integer.
    - ValueError: If 'decimals' is negative.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return np.ceil(number)

    factor = 10**decimals
    return np.ceil(number * factor) / factor


def round_decimals_down(
    number: Union[float, np.ndarray], decimals: int = 2
) -> Union[float, np.ndarray]:
    """
    Rounds a number or each element of an array down to a specific number of decimal places.

    Parameters:
    - number (float or np.ndarray): The number or array of numbers to round down.
    - decimals (int): The number of decimal places to round down to. Must be a non-negative integer.

    Returns:
    - float or np.ndarray: The rounded down number or array of numbers.

    Raises:
    - TypeError: If 'decimals' is not an integer.
    - ValueError: If 'decimals' is negative.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return np.floor(number)

    factor = 10**decimals
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


def get_nearest_neighbors(
    x: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    n_neighbors: int = 100,
    max_dis: float = 540.0,
    n_fixed: int = 20,
    n_jobs: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the nearest neighbors for a set of coordinates.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the target points.
    y : np.ndarray
        The y-coordinates of the target points.
    x0 : np.ndarray
        The x-coordinates of the source points.
    y0 : np.ndarray
        The y-coordinates of the source points.
    n_neighbors : int, optional
        The number of nearest neighbors to find. Default is 100.
    max_dis : float, optional
        The maximum distance to consider when finding nearest neighbors. Distances
        larger than this value will be set to NaN. Default is 540.
    n_fixed : int, optional
        The minimum number of neighbors to include, regardless of distance.
        Default is 20.
    n_jobs : int or None, optional
        The number of parallel jobs to run for neighbors search. None means 1.
        -1 means using all processors. Default is None.
    **kwargs
        Additional keyword arguments to pass to the NearestNeighbors constructor.

    Returns
    -------
    nn_dis : np.ndarray
        Array of distances to the nearest neighbors, shaped (len(x), n_neighbors).
    nn_idx : np.ndarray
        Array of indices of the nearest neighbors, shaped (len(x), n_neighbors).

    Notes
    -----
    This function uses a KDTree to efficiently find the nearest neighbors of each
    target point (x, y) from a set of source points (x0, y0). If the distance to
    a neighbor exceeds `max_dis`, it is replaced with NaN. At least `n_fixed`
    neighbors are always included, regardless of the distance.
    """

    # Set n_neighbors and n_fixed to valid values based on the input size
    n0 = len(x0)
    n_neighbors = min(n0, n_neighbors)
    n_fixed = min(n0, n_fixed)

    # Stack coordinates
    stacked0 = np.column_stack([x0, y0])
    stacked = np.column_stack([x, y])

    # Grab nearest neighbors
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs, **kwargs).fit(
        stacked0
    )
    nn_dis, nn_idx = nn_model.kneighbors(stacked)

    # Ensure at least n_fixed neighbors are considered
    nn_dis_fixed = nn_dis[:, :n_fixed]
    nn_dis = np.where(nn_dis > max_dis, np.nan, nn_dis)
    nn_dis[:, :n_fixed] = nn_dis_fixed

    # Return nearest neighbors
    return nn_dis, nn_idx


def interpolate_value(
    x: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    val0: np.ndarray,
    additional_weights0: np.ndarray,
    n_neighbors: int = 100,
    max_dis: float = 540,
    n_fixed: int = 20,
    nn_dis: Optional[np.ndarray] = None,
    nn_idx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate a value and its standard deviation based on the nearest neighbors' data.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinate of the point to interpolate.
    y : np.ndarray
        The y-coordinate of the point to interpolate.
    x0 : np.ndarray
        The x-coordinates of the input data points.
    y0 : np.ndarray
        The y-coordinates of the input data points.
    val0 : np.ndarray
        The values of the input data points.
    additional_weights0 : np.ndarray
        Additional weights to apply to each data point during interpolation.
    n_neighbors : int, optional
        The number of nearest neighbors to consider (default is 100).
    max_dis : float, optional
        The maximum distance to consider for interpolation (default is 540).
    n_fixed : int, optional
        The minimum number of neighbors to include, regardless of distance
        (default is 20).
    nn_dis : np.ndarray, optional
        The distances to the nearest neighbors. Enables reusing the nearest
        neighbors for multiple interpolations.
    nn_idx : np.ndarray, optional
        The indices of the nearest neighbors. Enables reusing the nearest
        neighbors for multiple interpolations.

    Returns
    -------
    val : np.ndarray
        The interpolated values.
    val_std : np.ndarray
        The standard deviations of the interpolated values.
    n_sources_per_source : np.ndarray
        The number of sources used for each interpolation.
    max_dis_per_source : np.ndarray
        The maximum distance used for each source during interpolation.
    """

    # If nearest neighbors are not provided, compute them
    if nn_dis is None or nn_idx is None:
        nn_dis, nn_idx = get_nearest_neighbors(
            x=x,
            y=y,
            x0=x0,
            y0=y0,
            n_neighbors=n_neighbors,
            max_dis=max_dis,
            n_fixed=n_fixed,
        )

    # Grab nearest neighbor data and
    nn_data = val0[nn_idx]
    nn_additional_weights = additional_weights0[nn_idx]

    # Compute weight based on gaussian distance metric and additional weights
    weights = np.exp(-0.5 * (nn_dis / (max_dis / 2)) ** 2) * nn_additional_weights

    # Replicate weights to third dimension if required
    if nn_data.ndim == 3:
        weights = np.repeat(weights[:, :, np.newaxis], nn_data.shape[2], axis=2)

    # Compute mask based on sigma-clipped data array
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input data contains invalid values")
        wmask = sigma_clip(nn_data, axis=1, sigma=2.5, maxiters=2, masked=True).mask

    # Set weights to zero for masked values
    weights[wmask] = 0.0
    bad_idx = ~np.isfinite(nn_data)
    nn_data[bad_idx], weights[bad_idx] = 0, 0

    # Compute number of sources and max distance per source
    if weights.ndim == 3:
        neighbor_used = np.any(weights > 0, axis=2)
    else:
        neighbor_used = weights > 0
    n_sources_per_source = np.sum(neighbor_used, axis=1)
    nn_dis_masked = nn_dis.copy()
    nn_dis_masked[~neighbor_used] = -np.inf
    max_dis_per_source = np.max(nn_dis_masked, axis=1)
    max_dis_per_source[max_dis_per_source == -np.inf] = np.nan

    # Compute weighted average and std
    vals = np.average(nn_data, axis=1, weights=weights)
    vals_std = np.sqrt(
        np.average((nn_data - vals[:, np.newaxis]) ** 2, axis=1, weights=weights)
    )

    # Return interpolated value, standard deviation, number of sources, and max distance
    return vals, vals_std, n_sources_per_source, max_dis_per_source


def find_neighbors_within_distance(
    coords1: SkyCoord,
    coords2: SkyCoord,
    distance_limit_arcmin: float,
    compute_distances: bool = False,
) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
    """
    Find all neighbors within a given distance limit using KD-Tree.

    Parameters
    ----------
    coords1 : SkyCoord
        The set of SkyCoord instances to find neighbors for.
    coords2 : SkyCoord
        The set of SkyCoord instances to search within.
    distance_limit_arcmin : float
        The distance limit within which to search for neighbors, in arcminutes.
    compute_distances : bool, optional
        Whether to compute and return the distances to the neighbors. Default is False.

    Returns
    -------
    neighbors : List[List[int]]
        A list where each entry contains the indices of neighbors within the distance limit for each entry in coords1.
    distances_arcmin : Optional[List[List[float]]]
        A list where each entry contains the distances to the neighbors within the distance limit for each entry in coords1,
        in arcminutes. Returns None if compute_distances is False.
    """
    # Convert RA/Dec to Cartesian coordinates for KD-Tree
    xyz1 = np.vstack(
        [
            coords1.cartesian.x.value,
            coords1.cartesian.y.value,
            coords1.cartesian.z.value,
        ]
    ).T
    xyz2 = np.vstack(
        [
            coords2.cartesian.x.value,
            coords2.cartesian.y.value,
            coords2.cartesian.z.value,
        ]
    ).T

    # Build KD-Trees
    tree1 = cKDTree(xyz1)
    tree2 = cKDTree(xyz2)

    # Query all points within the distance limit
    # Convert the distance limit to radians (since Cartesian coordinates are unitless)
    distance_limit_rad = np.radians(distance_limit_arcmin / 60.0)
    neighbors = tree1.query_ball_tree(tree2, distance_limit_rad)

    distances_arcmin = None
    if compute_distances:
        # Calculate the actual distances in arcminutes
        distances_arcmin = []
        for i, neighbor_indices in enumerate(neighbors):
            source_coord = coords1[i]
            neighbor_coords = coords2[neighbor_indices]
            separations = source_coord.separation(neighbor_coords)
            distances_arcmin.append(separations.to_value("arcmin").tolist())

    return neighbors, distances_arcmin


# TODO: Make sure this works correctly with PAs East of North
def convert_position_error(
    errmaj: Union[np.ndarray, list, tuple],
    errmin: Union[np.ndarray, list, tuple],
    errpa: Union[np.ndarray, list, tuple],
    degrees: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Right Ascension (RA) and Declination (Dec) errors and
    their correlation coefficient from major and minor errors and position angle.

    Parameters
    ----------
    errmaj : np.ndarray, list, tuple
        Major axis errors.
    errmin : np.ndarray, list, tuple
        Minor axis errors.
    errpa : np.ndarray, list, tuple
        Position angle of error ellipse (East of North);
        in degrees if degrees=True, otherwise in radians.
    degrees : bool, optional
        Indicates whether the position angle is in degrees (default is True).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        RA error, Dec error, and RA-Dec correlation coefficient, each as an np.ndarray.

    Notes
    -----
    The function converts position angle errors from degrees to radians if necessary,
    calculates cosine and sine of these angles, and uses these to compute elements of
    the covariance matrix. Variances along the RA and Dec directions and the covariance
    between RA and Dec are also computed, which are then used to derive the standard
    deviations (errors) and correlation coefficient.

    """
    # Make sure the input is a numpy array
    errmaj, errmin = np.asarray(errmaj), np.asarray(errmin)

    # Convert position angles from degrees to radians
    if degrees:
        theta_rad = np.deg2rad(errpa)
    else:
        theta_rad = np.asarray(errpa)

    # Calculate the components of the rotation matrix for each set
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    # Preallocate the 3D matrix array (N sets of 2x2 matrices)
    cc = np.zeros((len(errmaj), 2, 2))

    # Define each element of the covariance matrices
    cc[:, 0, 0] = cos_theta**2 * errmin**2 + sin_theta**2 * errmaj**2
    cc[:, 0, 1] = (cos_theta * sin_theta) * (errmaj**2 - errmin**2)
    cc[:, 1, 0] = cc[:, 0, 1]
    cc[:, 1, 1] = sin_theta**2 * errmin**2 + cos_theta**2 * errmaj**2

    # Compute the RA and Dec errors and correlation coefficients
    ra_error = np.sqrt(cc[:, 0, 0])
    dec_error = np.sqrt(cc[:, 1, 1])
    ra_dec_corr = cc[:, 0, 1] / np.sqrt(cc[:, 0, 0] * cc[:, 1, 1])

    # Return
    return ra_error, dec_error, ra_dec_corr
