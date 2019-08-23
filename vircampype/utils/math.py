# =========================================================================== #
# Import
import astropy
import warnings
import numpy as np
from PIL import Image
import multiprocessing

from itertools import repeat
from scipy.ndimage import median_filter
from astropy.convolution import Gaussian2DKernel, Kernel2D, CustomKernel


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


def sigma_clip(data, kappa=3, ikappa=1, center_metric=np.nanmedian, axis=0):
    """
    Performs sigma clipping of data.

    Parameters
    ----------
    data : ndarray
        Input data.
    kappa : int, optional
        kappa-factor (e.g. 3-sigma).
    ikappa : int, optional
        Number of iterations.
    center_metric : callable, optional
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

        for _ in range(ikappa):

            # Calculate center with given metric
            center = center_metric(data, axis=axis)

            # Calculate standard deviation
            sigma = np.nanstd(data, axis=axis)

            # find values outside limits and set to NaN
            data[(data > center + kappa * sigma) | (data < center - kappa * sigma)] = np.nan

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


def interpolate_image(array, kernel=None, max_bad_neighbors=None):
    """
    Interpolates NaNs in an image. NaNs are replaced by convolving the original image with a kernel from which
    the pixel values are copied. This technique is much faster than other aporaches involving spline fitting
    (e.g. griddata or scipy inteprolation methods.)

    Parameters
    ----------
    array : np.ndarray
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
        nans_conv = astropy.convolution.convolve(nans, kernel=nan_kernel, boundary="extend")

        # Get the ones with a maximum of 'max_bad_neighbors' bad neighbors
        # noinspection PyTypeChecker
        nans_fil = (nans_conv <= max_bad_neighbors) & (nans > 0)

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
        conv = astropy.convolution.convolve(array=array, kernel=kernel, boundary="extend")

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


# ----------------------------------------------------------------------
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
    with multiprocessing.Pool(processes=n_threads) as pool:
        # TODO: Added a repeat(10) here for max_iter; check if ok!
        mp = pool.starmap(estimate_background, zip(sub, repeat(10), repeat(max_iter), repeat((1, 2))))

    # Unpack results
    background, noise = np.array(list(zip(*mp)))

    # Reshape for cube
    background = background.T.reshape((len(cube),) + xgrid.shape)
    noise = noise.T.reshape((len(cube),) + xgrid.shape)

    # Interpolate NaNs if any
    if np.sum(~np.isfinite(background)) > 0:
        background = interpolate_image(array=background, kernel=np.ones((3, 3)), max_bad_neighbors=None)
        noise = interpolate_image(array=noise, kernel=np.ones((3, 3)), max_bad_neighbors=None)

    # Median-filter low-res images if set and reshape into image
    if mesh_filtersize > 1:
        for idx in range(len(cube)):
            background[idx, :, :] = median_filter(input=background[idx, :, :], size=mesh_filtersize)
            noise[idx, :, :] = median_filter(input=noise[idx, :, :], size=mesh_filtersize)

    # Scale back to original size
    cube_background, cube_noise = [], []
    for bg, nn in zip(background, noise):
        cube_background.append(np.array(Image.fromarray(bg).resize(size=cube.shape[1:], resample=Image.BICUBIC)))
        cube_noise.append(np.array(Image.fromarray(nn).resize(size=cube.shape[1:], resample=Image.BICUBIC)))

        """
        This is how it works with griddata.
        Imresize is MUCH faster than griddata, but the results are different and it only works for 32bit float data!
    
        from scipy.interpolate import griddata
        xgrid, ygrid = grid_like(array=array)
        back_map = griddata(np.stack([ygrid_flat, xgrid_flat], axis=1), back_map.ravel(), (ygrid, xgrid),
                            method=method)
        noise_map = griddata(np.stack([ygrid_flat, xgrid_flat], axis=1), noise_map.ravel(), (ygrid, xgrid),
                             method=method)
    
        """

    # Return scaled cubes
    return np.array(cube_background), np.array(cube_noise)


def apply_along_axes(array, func=np.nanmedian, axis=None, norm=True, copy=True):
    """
    Destripes arbitrary input arrays.

    Parameters
    ----------
    array : np.ndarray
        Array to destripe.
    func : callable
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
