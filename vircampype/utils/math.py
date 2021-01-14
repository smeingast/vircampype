# =========================================================================== #
# Import
import warnings
import itertools
import numpy as np

# noinspection PyUnresolvedReferences
# from astroscrappy.astroscrappy import detect_cosmics

from itertools import repeat
from fractions import Fraction
from astropy.units import Unit
from joblib import Parallel, delayed
from scipy.interpolate import griddata
from scipy.ndimage import median_filter
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic_2d
from vircampype.utils.miscellaneous import str2func
from scipy.interpolate import SmoothBivariateSpline
from astropy.convolution import Gaussian2DKernel, Kernel2D, CustomKernel, convolve, interpolate_replace_nans


# Define objects in this module
__all__ = ["estimate_background", "sigma_clip", "cuberoot", "squareroot", "linearize_data", "ceil_value", "floor_value",
           "interpolate_image", "chop_image", "merge_chopped", "meshgrid", "background_cube", "apply_along_axes",
           "distance_sky", "distance_euclid2d", "connected_components", "centroid_sphere", "centroid_sphere_skycoord",
           "haversine", "fraction2float", "grid_value_2d", "grid_value_2d_griddata", "point_density", "upscale_image"]


def estimate_background(array, max_iter=10, force_clipping=False, axis=None):
    """
    Estimates the background sky level based on an iterative 3-sigma clipping algorithm. In principle the data are
    iterativley clipped around the median. At each iteration the mean of the clipped histogram is calculated. If the
    change from one iteration to the next is less than 1%, estimates for the background and standard deviation in the
    background are returned. Here, we return the mean if the last iteration did not show more than 20% relative change
    compared to the fist iteration. Otherwise, the field is assumed to be crowded and the mode is estimated with
    2.5 * median - 1.5 * mean (see SExtractor doc). Ultimatley at the 'max_iter' iteration, the mode estimate is
    always returned.

    Parameters
    ----------
    array : np.ndarray
        Input data
    max_iter : int, optional
        Maximum iterations. If convergence is not reached, return an estimate of the background
    force_clipping : bool, optional
        If set, then even without convergence, the result will be returned after max_iter.
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
        elif idx > 0:

            # If we have no change within 2% of previous iteration we return
            if np.mean(np.abs(sky_save / sky - 1)) < 0.02:

                # If (compared to the initial value) the mean has changed by less than 20%, the field is not crowded
                if np.mean(np.abs(sky / sky_ini - 1)) < 0.2 or force_clipping is True:
                    return sky, skysig

                # Otherwise the field is crowded and we return an estimate of the mode
                else:
                    return 2.5 * np.nanmedian(masked, axis=axis) - 1.5 * np.nanmean(masked, axis=axis), skysig

            # Otherwise we do one more iteration
            else:
                pass

        # If we have no convergence after 10 iterations, we return an estimate
        elif idx > max_iter:

            if force_clipping is True:
                return sky, skysig
            else:
                return 2.5 * np.nanmedian(masked, axis=axis) - 1.5 * np.nanmean(masked, axis=axis), skysig

        # For the iterations between 1 and 10 when no convergence is reached
        sky_save = sky.copy()

        # Increase loop index
        idx += 1


def sigma_clip(data, sigma_level=3, sigma_iter=1, center_metric=np.nanmedian, axis=0):
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
    center_metric : callable, float, int, optional
        Metric which calculates the center around which clipping occurs.
    axis : int, optional
        Axis along which to perform clipping.

    Returns
    -------
    ndarray
        Array with clipped values set to NaN.

    """

    # Ignore the numpy warnings upon masking
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for _ in range(sigma_iter):

            # Calculate center with given metric
            if callable(center_metric):
                center = center_metric(data, axis=axis)
            else:
                center = center_metric

            # Calculate standard deviation
            std = np.nanstd(data, axis=axis)

            # find values outside limits and set to NaN
            data[(data > center + sigma_level * std) | (data < center - sigma_level * std)] = np.nan

    # Return the clipped array
    return data


def cuberoot(a, b, c, d, return_real=False):
    """
    Function to return the roots of a cubic polynomial ax^3 + bx^2 + cx + d = 0
    Uses the general formula listed in Wikipedia.

    Parameters
    ----------
    a : float, int
        Cubic coefficient.
    b : float, int
        Square coefficient.
    c : float, int
        Linear coefficient.
    d : np.ndarray, float, int
        Intercept.
    return_real : bool, optional
        If set, only return the real part of the solutions.

    Returns
    -------
    np.ndarray
        Roots of polynomial.

    """

    # Transform to complex numbers
    a, b, c = complex(a), complex(b), complex(c)

    # Calculate stuff to get the roots
    delta0, delta1 = b ** 2. - 3. * a * c, 2. * b ** 3. - 9. * a * b * c + 27. * a ** 2. * d
    z = ((delta1 + np.sqrt(delta1 ** 2. - 4. * delta0 ** 3.)) / 2.) ** (1. / 3.)

    u1, u2, u3 = 1., (- 1. + 1J * np.sqrt(3)) / 2., (- 1. - 1J * np.sqrt(3)) / 2.

    # Just return real part
    if return_real:
        return [(-(1. / (3. * a)) * (b + u * z + (delta0 / (u * z)))).real for u in [u1, u2, u3]]

    # Return all solutions
    else:
        return [-(1. / (3. * a)) * (b + u * z + (delta0 / (u * z))) for u in [u1, u2, u3]]


def squareroot(a, b, c, return_real=False):
    """
    Function to return the roots of a quadratic polynomial ax^2 + bx + c = 0

    Parameters
    ----------
    a : float, int
        Square coefficient.
    b : float, int
        Linear coefficient.
    c : np.ndarray, float, int
        Intercept.
    return_real : bool, optional
        If set, only return the real part of the solutions.

    Returns
    -------
    np.ndarray
        Roots of polynomial

    """

    # Transform to complex numbers
    a, b = complex(a), complex(b)

    # Calculate stuff to get the roots
    delta = np.sqrt(b ** 2. - 4 * a * c)
    x1, x2 = (-b + delta) / (2 * a), (-b - delta) / (2 * a)

    # Just return real part
    if return_real:
        return [x1.real, x2.real]

    # Return all solutions
    else:
        return [x1, x2]


def linearize_data(data, coeff):
    """
    General single-threaded linearization for arbitrary input data.

    Parameters
    ----------
    data : np.ndarray
        Input data to be linearized.
    coeff : list[floats]
        List of coefficients.

    Returns
    -------
    np.ndarray
        Linearized data

    """

    # Determine order of fit
    order = len(coeff) - 1

    # Prepare data
    coeff_copy = coeff.copy()
    coeff_copy[-1] -= data.ravel()

    # Get the roots of all data points
    if order == 2:
        roots = squareroot(*coeff_copy, return_real=True)
    elif order == 3:
        roots = cuberoot(*coeff_copy, return_real=True)
    else:
        raise ValueError("Order '{0}' not supported".format(order))

    # Select closest value from the real roots, and return
    return (np.min(np.abs([r - coeff[-1] + coeff_copy[-1] for r in roots]), axis=0) + data.ravel()).reshape(data.shape)


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


def interpolate_image(data, kernel=None, max_bad_neighbors=None):
    """
    Interpolates NaNs in an image. NaNs are replaced by convolving the original image with a kernel from which
    the pixel values are copied. This technique is much faster than other aporaches involving spline fitting
    (e.g. griddata or scipy inteprolation methods.)

    Parameters
    ----------
    data : np.ndarray
        2D numpy array to interpolate.
    kernel : Kernel2D, np.ndarray, optional
        Kernel used for interpolation.
    max_bad_neighbors : int, optional
        Maximum bad neighbors a pixel can have to be interpolated. Default is None.

    Returns
    -------
    np.ndarray
        Interpolated image

    """

    # Copy data to avoid "read_only issue"
    array = data.copy()

    # Determine NaNs
    nans = ~np.isfinite(array)

    # If there are no NaNs, we return
    if np.sum(nans) == 0:
        return array

    # In case we want to exclude pixels surrounded by other bad pixels
    if max_bad_neighbors is not None:

        # Make kernel for neighbor counts
        nan_kernel = np.ones(shape=(3, 3))

        # Convolve NaN data
        nans_conv = convolve(nans, kernel=nan_kernel, boundary="extend", normalize_kernel=False)

        # Get the ones with a maximum of 'max_bad_neighbors' bad neighbors
        # noinspection PyTypeChecker
        nans_fil = (nans_conv <= max_bad_neighbors) & (nans == 1)

        # If there are no NaNs at the stage, we return
        if np.sum(nans_fil) == 0:
            return array

        # Get the NaNs which where skipped
        nans_skipped = (nans_fil == 0) & (nans == 1)

        # Set those to the median
        array[nans_skipped] = np.nanmedian(array)

        # Assign new NaNs
        nans = nans_fil

    # Just for editor warnings
    else:
        nans_skipped = None

    # Set kernel
    if kernel is None:
        kernel = Gaussian2DKernel(1)
    elif isinstance(kernel, np.ndarray):
        # noinspection PyTypeChecker
        kernel = CustomKernel(kernel)
    else:
        if not isinstance(kernel, Kernel2D):
            raise ValueError("Supplied kernel not supported")

    # Convolve
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        conv = convolve(array, kernel=kernel, boundary="extend")

    # Fill interpolated NaNs in
    array[nans] = conv[nans]

    # Fill skipped NaNs back in
    array[nans_skipped] = np.nan

    # Return
    return array


def chop_image(array, npieces, axis=0, overlap=None):
    """
    Chops a numpy 2D (image) array into subarrays.

    Parameters
    ----------
    array : np.array
        The array to chop.
    npieces : int
        Number of pieces in the chopped output.
    axis : int, optional
        The axis along which to chop.
    overlap : int, optional
        The overlap in the output split output arrays. Default is None.

    Returns
    -------
    list
        List of sub-arrays constructed from the input

    """

    # Axis must be 0 or 1
    if axis not in [0, 1]:
        raise ValueError("Axis={0:0d} not supported".format(axis))

    # If there is no overlap, we can just u se the numpy function
    if overlap is None:
        return np.array_split(ary=array, indices_or_sections=npieces, axis=axis)

    # Determine where to chop
    cut = list(np.int32(np.round(np.linspace(0, array.shape[axis], npieces + 1), decimals=0)))

    # Force the first and last cut location just to be safe from any integer conversion issues
    cut[0], cut[-1] = 0, array.shape[axis]

    chopped = []
    for i in range(npieces):

        if axis == 0:

            # First slice
            if i == 0:
                chopped.append(array[cut[i]:cut[i+1] + overlap, :])

            # Last slice
            elif i == npieces - 1:
                chopped.append(array[cut[i] - overlap:cut[i+1], :])

            # Everything else
            else:
                chopped.append(array[cut[i] - overlap:cut[i+1] + overlap, :])

        elif axis == 1:

            # First slice
            if i == 0:
                chopped.append(array[:, cut[i]:cut[i+1] + overlap])

            # Last slice
            elif i == npieces - 1:
                chopped.append(array[:, cut[i] - overlap:cut[i+1]])

            # Everything else
            else:
                chopped.append(array[:, cut[i] - overlap:cut[i+1] + overlap])

    # Return list of chopped arrays
    return chopped, cut


def merge_chopped(arrays, locations, axis=0, overlap=0):
    """
    Complementary to the above function, this one merges the chopped array back into the original.

    Parameters
    ----------
    arrays : iterable
        List of arrays to merge.
    locations : iterable
        List of locations where the cut occured (returned by chop_image)
    axis : int, optional
        Axis along which the cop occured. Default is 0.
    overlap : int, optional
        Overlap used in chopping.

    Returns
    -------
    np.ndarray
        Merged array.

    """

    # Axis must be 0 or 1
    if axis not in [0, 1]:
        raise ValueError("Axis={0:0d} not supported".format(axis))

    # Get other axis
    otheraxis = 1 if axis == 0 else 0

    # Determine size of output
    shape = (locations[-1], arrays[0].shape[otheraxis]) if axis == 0 else (arrays[0].shape[otheraxis], locations[-1])

    merged = np.empty(shape=shape, dtype=arrays[0].dtype)
    for i in range(len(arrays)):

        if axis == 0:

            # First slice
            if i == 0:
                merged[0:locations[i + 1], :] = arrays[i][:arrays[i].shape[0] - overlap, :]

            # Last slice
            elif i == len(arrays) - 1:
                merged[locations[i]:, :] = arrays[i][overlap:, :]

            # In between
            else:
                merged[locations[i]:locations[i+1], :] = arrays[i][overlap:-overlap, :]

        elif axis == 1:

            # First slice
            if i == 0:
                merged[:, 0:locations[i + 1]] = arrays[i][:, :arrays[i].shape[1] - overlap]

            # Last slice
            elif i == len(arrays) - 1:
                merged[:, locations[i]:] = arrays[i][:, overlap:]

            # In between
            else:
                merged[:, locations[i]:locations[i+1]] = arrays[i][:, overlap:-overlap]

    return merged


def meshgrid(array, size=128):
    """
    Generates a pixel coordinate grid from an array with given mesh sizes. The mesh size is approximated when the
    data shape is not a multiple of the mesh size (almost always the case). For smaller mesh sizes, the output mesh
    will match the input more closely.

    Parameters
    ----------
    array : np.ndarray
        Input data for which the pixel grid should be generated
    size : int, float, tuple, optional
        Size of the grid. The larger this value rel to the image dimensions, the less the requested grid size will match
        to the output grid size.

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
        size = [size, ] * array.ndim

    # Get number of cells per axis
    n = [np.ceil(sh / si) for sh, si in zip(array.shape, size)]

    """In contrast to meshgrid, this has the advantage that the edges are always included!"""
    # Return
    if array.ndim == 1:
        return np.uint32((np.mgrid[0:array.shape[0] - 1:complex(n[0])]))
    if array.ndim == 2:
        return np.uint32((np.mgrid[0:array.shape[0] - 1:complex(n[0]),
                                   0:array.shape[1] - 1:complex(n[1])]))
    if array.ndim == 3:
        return np.uint32((np.mgrid[0:array.shape[0] - 1:complex(n[0]),
                                   0:array.shape[1] - 1:complex(n[1]),
                                   0:array.shape[2] - 1:complex(n[2])]))
    else:
        raise ValueError("{0:d}-dimensional data not supported".format(array.ndim))


def background_cube(cube, mesh_size=128, mesh_filtersize=3, max_iter=10, n_threads=None):
    """
    Generates a background mesh for the cube based on a robust background estimation (similar to SExtractor).
    In addition to the background estimate also the 1-sigma standard deviation in the clipped data is computed.

    Parameters
    ----------
    cube : ndarray
        input cube
    mesh_size : int, optional
        Requested mesh size in pixels. Actual mesh size will vary depending on input shape (default = 128 pix).
    mesh_filtersize : int, optional
        2D median filter size for meshes (default = 3).
    max_iter : int, optional
        Maximum iterations. If convergence is not reached, return an estimate of the background.
    n_threads : int, optional
        Number of threads to use.

    Returns
    -------
    tuple[ndarray, ndarray]
        Tuple holding (background, noise).
         If return_grid is set (background, noise, x grid coordinates, y grid coordinates).

    """

    # Generate the pixel coordinate grid
    ygrid, xgrid = meshgrid(array=cube[0], size=mesh_size)
    ygrid_flat, xgrid_flat = ygrid.ravel(), xgrid.ravel()

    # Determine actual grid half-size
    y2size, x2size = ygrid[1][0] // 2, xgrid[0][1] // 2

    # For each of these grid points create the appropriate sub-region
    sub = []
    for y, x in zip(ygrid_flat, xgrid_flat):

        if (x > 0) & (y > 0):
            sub.append(cube[:, y - y2size:y + y2size, x - x2size:x + x2size])
        elif (x > 0) & (y == 0):
            sub.append(cube[:, y:y + y2size, x - x2size:x + x2size])
        elif (x == 0) & (y > 0):
            sub.append(cube[:, y - y2size:y + y2size, x:x + x2size])
        else:
            sub.append(cube[:, y:y + y2size, x:x + x2size])

    # For each sub-region estimate the background and noise
    if n_threads == 1:
        mp = []
        for s in sub:
            mp.append(estimate_background(array=s, max_iter=max_iter, force_clipping=True, axis=(1, 2)))

    elif n_threads > 1:
        with Parallel(n_jobs=n_threads) as parallel:
            mp = parallel(delayed(estimate_background)(s, i, mi, a)
                          for s, i, mi, a in zip(sub, repeat(max_iter), repeat(True), repeat((1, 2))))

    else:
        raise ValueError("'n_threads' not correctly set (n_threads = {0})".format(n_threads))

    # Unpack results
    background, noise = np.array(list(zip(*mp)))

    # Reshape for cube
    background = background.T.reshape((len(cube),) + xgrid.shape)
    noise = noise.T.reshape((len(cube),) + xgrid.shape)

    # Interpolate NaNs if any
    if np.sum(~np.isfinite(background)) > 0:
        background = interpolate_image(data=background, kernel=np.ones((3, 3)), max_bad_neighbors=None)
        noise = interpolate_image(data=noise, kernel=np.ones((3, 3)), max_bad_neighbors=None)

    # Median-filter low-res images if set and reshape into image
    if mesh_filtersize > 1:
        for idx in range(len(cube)):
            background[idx, :, :] = median_filter(input=background[idx, :, :], size=mesh_filtersize)
            noise[idx, :, :] = median_filter(input=noise[idx, :, :], size=mesh_filtersize)

    # Scale back to original size
    with Parallel(n_jobs=n_threads) as parallel:
        cube_background = parallel(delayed(upscale_image)(i, j, k, l) for i, j, k, l
                                   in zip(background, repeat(cube.shape[1:]), repeat("spline"), repeat(2)))
        cube_noise = parallel(delayed(upscale_image)(i, j, k, l) for i, j, k, l
                              in zip(noise, repeat(cube.shape[1:]), repeat("spline"), repeat(2)))

    # Return scaled cubes
    return np.array(cube_background), np.array(cube_noise)


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
        Whether the data should be copied. If false, the original array will be overwritten.

    Returns
    -------
    ndarray
        Modified array.

    """

    # Create copy if set
    if copy:
        array = array.copy()

    # Fetch function
    func = str2func(method)

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


def distance_sky(lon1, lat1, lon2, lat2, unit="radians"):
    """
    Returns the distance between two objects on a sphere. Also works with arrays.

    Parameters
    ----------
    lon1 : int, float, np.ndarray
        Longitude (e.g. Right Ascension) of first object
    lat1 : int, float, np.ndarray
        Latitude (e.g. Declination) of first object
    lon2 : int, float, np.ndarray
        Longitude of object to calculate the distance to.
    lat2 : int, float, np.ndarray
        Longitude of object to calculate the distance to.
    unit : str, optional
        The unit in which the coordinates are given. Either 'radians' or 'degrees'. Default is 'radians'. Output will
        be in the same units.

    Returns
    -------
    float, np.ndarray
        On-sky distances between given objects.

    """

    # Calculate distance on sphere
    if "rad" in unit:

        # Spherical law of cosines (not so suitable for very small numbers)
        # dis = np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2))

        # Haversine distance (better for small numbers)
        dis = 2 * np.arcsin(np.sqrt(np.sin((lat1 - lat2) / 2.) ** 2 +
                                    np.cos(lat1) * np.cos(lat2) * np.sin((lon1 - lon2) / 2.) ** 2))

    elif "deg" in unit:

        # Spherical law of cosines (not so suitable for very small numbers)
        # dis = np.degrees(np.arccos(np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) +
        #                            np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
        #                            np.cos(np.radians(lon1 - lon2))))

        # Haversine distance (better for small numbers)
        dis = 2 * np.degrees(np.arcsin(np.sqrt(np.sin((np.radians(lat1) - np.radians(lat2)) / 2.) ** 2 +
                                               np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
                                               np.sin((np.radians(lon1) - np.radians(lon2)) / 2.) ** 2)))

    # If given unit is not supported.
    else:
        raise ValueError("Unit {0:s} not supported".format(unit))

    # Return distance
    return dis


def distance_euclid2d(x1, y1, x2, y2):
    """
    For the very lazy ones a convenience function to calculate the euclidean distance between points.

    Parameters
    ----------
    x1 : int, float
        X coordinate of first object
    y1 : int, float
        Y coordinate of first object
    x2 : int, float, np.ndarray
        X coordinate of second object
    y2 : int, float, np.ndarray
        Y coordinate of second object

    Returns
    -------
    float, np.ndarray
        Distances between the data points.

    """

    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def connected_components(xarr, yarr, max_distance, metric="euclidean", units="degrees"):
    """
    Calculate connected groups separated by a maximum distance. Available metrics are euclidean and haversine.

    Parameters
    ----------
    xarr : np.array
        X coordinates of data.
    yarr : np.array
        Y coordinates of data.
    max_distance : float
        Maximum allowed distance within a group.
    metric : str, optional
        Distance metric. Either 'euclidean' or 'haversine'. Default is 'metric'.
    units : str, optional
        Units of input in case of haversine metric.

    Returns
    -------
    list
        List with the same length as input. Contains for each object the group it was assigned to.

    """

    # Start with as many groups as object
    groups = np.arange(len(xarr))

    # Loop over all coordinates
    for x, y in zip(xarr, yarr):

        # Calculate distance to all other data points
        if "euclid" in metric:
            dis = distance_euclid2d(x1=x, y1=y, x2=xarr, y2=yarr)
        elif metric == "haversine":
            dis = distance_sky(lon1=x, lat1=y, lon2=xarr, lat2=yarr, unit=units)
        else:
            raise ValueError("Metric {0:s} not suppoerted".format(metric))

        # Get index of those which are within the given limits
        idx = np.where(dis <= max_distance)[0]

        # If there is no other object, we continue
        if len(idx) == 0:
            continue

        # Set common group for all within the limits
        for i in idx:
            groups[groups == groups[i]] = min(groups[idx])

    # Rewrite labels starting with 0
    for old, new in zip(set(groups), range(len(set(groups)))):
        idx = [i for i, j in enumerate(groups) if j == old]
        groups[idx] = new

    # Return groups for each object
    return list(groups)


def centroid_sphere(lon, lat, units="radian"):
    """
    Calcualte the centroid on a sphere. Strictly valid only for a unit sphere and for a coordinate system with latitudes
    from -90 to 90 degrees and longitudes from 0 to 360 degrees.

    Parameters
    ----------
    lon : list, np.array
        Input longitudes
    lat : list, np.array
        Input latitudes
    units : str, optional
        Input units. Either 'radian' or 'degree'. Default is 'radian'.

    Returns
    -------
    tuple
        Tuple with (lon, lat) of centroid

    """

    # Convert to radians if degrees
    if "deg" in units.lower():
        mlon, mlat = np.radians(lon), np.radians(lat)
    else:
        mlon, mlat = lon, lat

    # Convert to cartesian coordinates
    x, y, z = np.cos(mlat) * np.cos(mlon), np.cos(mlat) * np.sin(mlon), np.sin(mlat)

    # 3D centroid
    xcen, ycen, zcen = np.sum(x) / len(x), np.sum(y) / len(y), np.sum(z) / len(z)

    # Push centroid to triangle surface
    cenlen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
    xsur, ysur, zsur = xcen / cenlen, ycen / cenlen, zcen / cenlen

    # Convert back to spherical coordinates and return
    outlon = np.arctan2(ysur, xsur)

    # Convert back to 0-2pi range if necessary
    if outlon < 0:
        outlon += 2 * np.pi
    outlat = np.arcsin(zsur)

    # Return
    if "deg" in units.lower():
        return np.degrees(outlon), np.degrees(outlat)
    else:
        return outlon, outlat


# TODO: Centroiding methods should be unified
def centroid_sphere_skycoord(skycoord):
    """
    Calculate the centroid on a sphere. Strictly valid only for a unit sphere and for a coordinate system with latitudes
    from -90 to 90 degrees and longitudes from 0 to 360 degrees.

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


def haversine(theta, units="radian"):
    """
    Haversine function.

    Parameters
    ----------
    theta : int, float, np.ndarray
        Angle(s)
    units : stro, optional
        Either 'radian' or 'degree'. Default is 'radian'.

    Returns
    -------

    """

    if "rad" in units.lower():
        return np.sin(theta / 2.)**2
    elif "deg" in units.lower():
        return np.degrees(np.sin(np.radians(theta) / 2.)**2)


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


def grid_value_2d_griddata(x, y, value, x_min, y_min, x_max, y_max, nx, ny,
                           conv=True, kernel_scale=0.1, method="cubic"):

    """
    Grids (non-uniformly) data onto a 2D array with size (naxis1, naxis2)

    Parameters
    ----------
    x : iterable, ndarray
        X coordinates
    y : iterable, ndarray
        Y coordinates
    value : iterable, ndarray
        Values for the X/Y coordinates
    x_min : int, float
        Minimum X position for grid.
    x_max : int, float
        Maximum X position for grid.
    y_min : int, float
        Minimum Y position for grid.
    y_max : int, float
        Maximum Y position for grid.
    nx : int
        Number of pixels for grid in X.
    ny : int
        Number of pixels for grid in Y.
    conv : bool, optional
        If set, convolve the grid before resampling to final size.
    kernel_scale : float, optional
        Convolution kernel scale relative to initial grid size.
    method : {"linear", "nearest", "cubic"}, optional
        Method of interpolation.

    Returns
    -------
    ndarray
        2D array with gridded data.

    """

    # Filter infinite values
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(value)

    # Make grid
    xg, yg = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))

    # Map ZPs onto grid
    gridded = griddata(points=np.stack([x[good], y[good]], axis=1), values=value[good],
                       xi=(xg, yg), method=method)

    # Smooth
    if conv:
        gridded = convolve(gridded, kernel=Gaussian2DKernel(x_stddev=np.mean([nx, ny]) * kernel_scale),
                           boundary="extend")

    # Rescale
    return gridded


def grid_value_2d(x, y, value, x_min, y_min, x_max, y_max, nx, ny, conv=True,
                  kernel_size=2, weights=None, upscale=True, interpolate_nan=True):
    """
    Grids (non-uniformly) data onto a 2D array with size (naxis1, naxis2)

    Parameters
    ----------
    x : iterable, ndarray
        X coordinates
    y : iterable, ndarray
        Y coordinates
    value : iterable, ndarray
        Values for the X/Y coordinates.
    x_min : int, float
        Minimum X position for grid.
    x_max : int, float
        Maximum X position for grid.
    y_min : int, float
        Minimum Y position for grid.
    y_max : int, float
        Maximum Y position for grid.
    nx : int
        Number of bins in X.
    ny : int
        Number of bins in Y.
    conv : bool, optional
        If set, convolve the grid before resampling to final size.
    kernel_size : float, optional
        Convolution kernel size in pix. Default is 2.
    weights : ndarray, optional
        Optionally provide weights for weighted average.
    upscale : bool, optional
        If True, rescale outout to (x_max - x_min, y_max  - y_min). Default it True.
    interpolate_nan : bool, optional
        In case there are NaN values in the grid, interpolate them before returning.

    Returns
    -------
    ndarray
        2D array with gridded data.

    """

    # Filter infinite values
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(value)

    # Grid
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # noinspection PyTypeChecker
        stat, xe, ye, (nbx, nby) = binned_statistic_2d(x=x[good], y=y[good], values=value[good], bins=[nx, ny],
                                                       range=[(x_min, x_max), (y_min, y_max)], statistic=np.nanmedian,
                                                       expand_binnumbers=True)

        # Convert bin number to index
        nbx, nby = nbx - 1, nby - 1

        # Compute weighted average instead of median if weights are provided
        if weights is not None:

            # Empty stat matrix
            stat = np.full((nx, ny), fill_value=np.nan)

            # Get all combinations of indices
            idx_combinations = list(itertools.product(np.arange(nx), np.arange(ny)))

            # Evaluate statistic for each bin
            for cidx in idx_combinations:

                # Get filter for current bin
                fil = (nbx == cidx[0]) & (nby == cidx[1])

                # sigma clip each bin separately
                mask = np.isfinite(sigma_clip(value[good][fil], sigma_level=3, sigma_iter=3))

                # Check sum of weights
                if np.sum(weights[good][fil]) < 0.0001:
                    stat[cidx[0], cidx[1]] = np.nan
                else:
                    # Compute weighted average for this bin
                    stat[cidx[0], cidx[1]] = np.average(value[good][fil][mask], weights=weights[good][fil][mask])

        # Transpose
        stat = stat.T

    # Smooth
    if conv:
        stat = convolve(stat, kernel=Gaussian2DKernel(x_stddev=kernel_size), boundary="extend")

    if interpolate_nan:
        stat = interpolate_replace_nans(stat, kernel=Gaussian2DKernel(2))

    # Upscale with spline
    if upscale:
        return upscale_image(image=stat, new_size=(x_max - x_min, y_max - y_min))

    return stat


def upscale_image(image, new_size, method="pil", order=3):
    """
    Resizes a 2D array to tiven new size.

    An example of how to upscale with PIL:
    apc_plot = np.array(Image.fromarray(apc_grid).resize(size=(hdr["NAXIS1"], hdr["NAXIS2"]), resample=Image.LANCZOS))

    Parameters
    ----------
    image : array_like
        numpy 2D array.
    new_size : tuple
        New size (xsize, ysize)
    method : str, optional
        Method to use for scaling. Either 'splines' or 'pil'.
    order : int
        Order for spline fit.

    Returns
    -------
    array_like
        Resized image.

    """

    if "pil" in method.lower():
        from PIL import Image
        return np.array(Image.fromarray(image).resize(size=new_size, resample=Image.LANCZOS))
    elif "spline" in method.lower():

        # Detemrine edge coordinates of input wrt output size
        xedge, yedge = np.linspace(0, new_size[0], image.shape[0]+1), np.linspace(0, new_size[1], image.shape[1]+1)

        # Determine pixel center coordinates
        xcenter, ycenter = (xedge[1:] + xedge[:-1]) / 2, (yedge[1:] + yedge[:-1]) / 2

        # Make coordinate grid
        xcenter, ycenter = np.meshgrid(xcenter, ycenter)

        # Fit spline to grid
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            spline_fit = SmoothBivariateSpline(xcenter.ravel(), ycenter.ravel(), image.ravel(), kx=order, ky=order).ev

            # Return interplated spline
            return spline_fit(*np.meshgrid(np.arange(new_size[0]), np.arange(new_size[1])))
    else:
        raise ValueError("Method '{0}' not supported".format(method))


def _point_density(x, y, xdata, ydata, xsize, ysize):
    """ Parallelisation method for point_average function. """

    d = (xdata > x - xsize / 2.) & \
        (xdata < x + xsize / 2.) & \
        (ydata > y - ysize / 2.) & \
        (ydata < y + ysize / 2.)

    return np.sum(d)


def point_density(xdata, ydata, xsize, ysize, norm=False, njobs=None):
    """ Compute singe point density. """

    # Import
    from joblib import Parallel, delayed, cpu_count

    # Create outarray
    out_dens = np.array([xdata]).reshape(-1)

    # Clean from NaNs
    goodindex = np.isfinite(xdata) & np.isfinite(ydata)

    # Apply to data
    xdata = xdata[goodindex]
    ydata = ydata[goodindex]

    # Set parallel jobs
    njobs = cpu_count() // 2 if njobs is None else njobs

    # Run
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Parallel(n_jobs=njobs) as parallel:
            mp = parallel(delayed(_point_density)(a, b, c, d, e, f) for a, b, c, d, e, f in
                          zip(xdata, ydata, repeat(xdata), repeat(ydata),
                              repeat(xsize), repeat(ysize)))

    # Fill indices
    out_dens[goodindex] = np.array(mp)
    out_dens[~goodindex] = np.nan

    # Return
    if norm:
        return out_dens / np.nanmax(out_dens)
    else:
        return out_dens
