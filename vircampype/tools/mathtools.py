import warnings
import itertools
import numpy as np

from astropy.units import Unit
from fractions import Fraction
from scipy.ndimage import median_filter
from vircampype.external.mmm import mmm
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic_2d
from vircampype.tools.miscellaneous import *
from astropy.stats import sigma_clipped_stats
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import SmoothBivariateSpline
from astropy.convolution import convolve, Gaussian2DKernel, CustomKernel, Kernel2D, interpolate_replace_nans, \
    Box2DKernel

__all__ = ["sigma_clip", "linearize_data", "apply_along_axes", "chop_image", "interpolate_image", "merge_chopped",
           "ceil_value", "floor_value", "meshgrid", "estimate_background", "upscale_image", "centroid_sphere",
           "clipped_median", "clipped_stdev", "grid_value_2d", "fraction2float", "round_decimals_up", "circular_mask",
           "round_decimals_down", "background_image", "grid_value_2d_nn", "linearity_fitfunc", "destripe_helper"]


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


def clipped_median(data, **kwargs):
    """ Hlper function to return the clipped median of an array via astropy. """
    return sigma_clipped_stats(data, **kwargs)[1]  # noqa


def clipped_stdev(data, **kwargs):
    """ Hlper function to return the clipped median of an array via astropy. """
    return sigma_clipped_stats(data, **kwargs)[2]  # noqa


def cuberoot_idl(c0: (int, float), c1: (int, float), c2: (int, float), c3: (int, float)):
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
    a, b, c = c0/c3, c1/c3, c2/c3

    q = (c**2 - 3*b) / 9
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
        # TODO: Perhaps mask these? They are all bad solutions actually as far as I have seen so far
        rf = r[index2]
        qf = q[index2]
        cf = c[index2]

        # Get the one real root
        h = -rf / np.abs(rf) * (np.abs(rf) + np.sqrt(rf**2 - qf**3))**(1 / 3)
        k = h.copy()

        zindex = np.isclose(h, 0, atol=1.E-5)
        cindex = ~zindex
        zcount, ccount = np.sum(zindex), np.sum(cindex)
        if zcount > 0:
            k[zindex] = 0.
        if ccount > 0:
            k[cindex] = qf / h

        # TODO: Double-check this solution with IDL
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
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")

        # Transform to complex numbers
        d, c, b = complex(d), complex(c), complex(b)

        # Calculate stuff to get the roots
        delta0, delta1 = c ** 2. - 3. * d * b, 2. * c ** 3. - 9. * d * c * b + 27. * d ** 2. * a
        z = ((delta1 + np.sqrt(delta1 ** 2. - 4. * delta0 ** 3.)) / 2.) ** (1. / 3.)

        u1, u2, u3 = 1., (- 1. + 1J * np.sqrt(3)) / 2., (- 1. - 1J * np.sqrt(3)) / 2.

        # Just return real part
        if return_real:
            return [(-(1. / (3. * d)) * (c + u * z + (delta0 / (u * z)))).real for u in [u1, u2, u3]]

        # Return all solutions
        else:
            return [-(1. / (3. * d)) * (c + u * z + (delta0 / (u * z))) for u in [u1, u2, u3]]


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
    delta = np.sqrt(b ** 2. - 4 * c * a)
    x1, x2 = (-b + delta) / (2 * c), (-b - delta) / (2 * c)

    # Just return real part
    if return_real:
        return [x1.real, x2.real]

    # Return all solutions
    else:
        return [x1, x2]


def linearity_fitfunc(x, b1, b2, b3):
    """ Fitting function used for non-linearity correction. """
    coeff, resettime = [b1, b2, b3], 1.0011
    return np.sum([coeff[j-1] * x**j * ((1 + resettime/x)**j - (resettime/x)**j)
                   for j in range(1, len(coeff) + 1)], axis=0)


def linearize_data(data, coeff, dit, reset_read_overhead):
    """
    General single-threaded linearization for arbitrary input data.

    Parameters
    ----------
    data : np.ndarray
        Input data to be linearized.
    coeff : list[floats], ndarray
        List of coefficients.
    dit : int, float
        DIT of input data.
    reset_read_overhead : float
        Reset-read-overhead in seconds.

    Returns
    -------
    np.ndarray
        Linearized data

    """

    # Dummy check
    if not isinstance(dit, (int, float)):
        raise ValueError("DIT must be of type float or int.")

    # Determine order of fit
    order = len(coeff) - 1

    # Coefficient modification factor based on DIT and reset overhead
    f = (1 + reset_read_overhead / dit)**np.arange(order + 1) - \
        (reset_read_overhead / dit)**np.arange(order + 1)
    # TODO: Try also to set f = 1 and compare with VISION
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

    # Select closest value from the real roots, and return
    return (np.nanmin(np.abs([r - coeff[0] + coeff_copy[0] for r in roots]), axis=0) + data.ravel()).reshape(data.shape)


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
        nans_fil = (nans_conv <= max_bad_neighbors) & (nans == 1)  # noqa

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
        kernel = CustomKernel(kernel)  # noqa
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


def estimate_background(array, max_iter=20, force_clipping=True, axis=None):
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
        elif idx >= max_iter:

            if force_clipping is True:
                return sky, skysig
            else:
                return 2.5 * np.nanmedian(masked, axis=axis) - 1.5 * np.nanmean(masked, axis=axis), skysig

        # For the iterations between 1 and 10 when no convergence is reached
        sky_save = sky.copy()

        # Increase loop index
        idx += 1


def background_image(image, mesh_size, mesh_filtersize=3):

    # Image must be 2D
    if len(image.shape) != 2:
        raise ValueError("Please supply array with 2 dimensions. "
                         "The given data has {0} dimensions".format(len(image.shape)))

    # Back size and image dimensions must be compatible
    if (image.shape[0] % mesh_size != 0) | (image.shape[1] % mesh_size != 0):
        raise ValueError("Image dimensions {0} must be multiple of backsize mesh size ({1})"
                         "".format(image.shape, mesh_size))

    # Tile image
    tiles = [image[x:x + mesh_size, y:y + mesh_size] for x in
             range(0, image.shape[0], mesh_size) for y in range(0, image.shape[1], mesh_size)]

    # Estimate background for each tile
    bg, bg_std, _ = list(zip(*[mmm(t) for t in tiles]))

    # Scale back to 2D array
    n_tiles_x, n_tiles_y = image.shape[1] // mesh_size, image.shape[0] // mesh_size
    bg, bg_std = np.array(bg).reshape(n_tiles_y, n_tiles_x), np.array(bg_std).reshape(n_tiles_y, n_tiles_x)

    # Interpolate NaN values in grid
    bg = interpolate_replace_nans(bg, kernel=Gaussian2DKernel(3), boundary="extend")
    bg_std = interpolate_replace_nans(bg_std, kernel=Gaussian2DKernel(3), boundary="extend")

    # Apply median filter
    bg, bg_std = median_filter(input=bg, size=mesh_filtersize), median_filter(input=bg_std, size=mesh_filtersize)

    # Convolve
    bg = convolve(bg, kernel=Gaussian2DKernel(1), boundary="extend")
    bg_std = convolve(bg_std, kernel=Gaussian2DKernel(1), boundary="extend")

    # Return upscaled data
    return upscale_image(bg, new_size=image.shape), upscale_image(bg_std, new_size=image.shape)  # noqa


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
        return np.array(Image.fromarray(image).resize(size=new_size, resample=Image.BICUBIC))
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


def centroid_sphere(skycoord):
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

        stat, xe, ye, (nbx, nby) = binned_statistic_2d(x=x[good], y=y[good], values=value[good], bins=[nx, ny],
                                                       statistic=clipped_median, expand_binnumbers=True,
                                                       range=[(x_min, x_max), (y_min, y_max)])  # noqa

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


def grid_value_2d_nn(x, y, values, n_nearest_neighbors, n_bins_x, n_bins_y, x_min, x_max, y_min, y_max):
    """
    Grids values to a 2D array based on nearest neighbor interpolation.

    Parameters
    ----------
    x : np.ndarray
        X coordinates of input data.
    y : np.ndarray
        Y coordinates of input data.
    values : np.ndarray
        Values of datapoints.
    n_nearest_neighbors : int
        Number of nearest neighbors to use in interpolation.
    n_bins_x : int
        Number of gridpoints in x.
    n_bins_y : int
        Number of gridpoints in y.
    x_min : int, float
        Minimum X coordinate of original data.
    x_max : int, float
        Maximum X coordinate of original data.
    y_min : int, float
        Minimum Y coordinate of original data.
    y_max : int, float
        Maximum Y coordinate of original data.


    Returns
    -------
    np.ndarray
        Interpolated 2D array.

    """

    # Determine step size in grid in X and Y
    step_x, step_y = (x_max - x_min) / n_bins_x, (y_max - y_min) / n_bins_y

    # Create grid of pixel centers
    xg, yg = np.meshgrid(np.linspace(x_min + step_x / 2, x_max - step_x / 2, n_bins_x),
                         np.linspace(y_min + step_y / 2, y_max - step_y / 2, n_bins_y))

    # Get nearest neighbors to grid pixel centers
    stacked_grid = np.stack([xg.ravel(), yg.ravel()]).T
    stacked_data = np.stack([x, y]).T
    _, idx = NearestNeighbors(n_neighbors=n_nearest_neighbors).fit(stacked_data).kneighbors(stacked_grid)

    # Obtain median values at each grid pixel
    _, gv, _ = sigma_clipped_stats(values[idx], axis=1)
    gv = gv.reshape(n_bins_y, n_bins_x)

    # Apply some filters and return
    if np.sum(~np.isfinite(gv)) > 0:
        gv = interpolate_replace_nans(gv, kernel=Box2DKernel(3))
    gv = median_filter(gv, size=3)
    return convolve(gv, kernel=Gaussian2DKernel(1), boundary="extend")


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
    """ Returns a value rounded up to a specific number of decimal places. """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return np.ceil(number)

    factor = 10 ** decimals
    return np.ceil(number * factor) / factor


def round_decimals_down(number: float, decimals: int = 2):
    """ Returns a value rounded down to a specific number of decimal places. """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return np.floor(number)

    factor = 10 ** decimals
    return np.floor(number * factor) / factor


def destripe_helper(array, mask=None):
    """
    Destripe helper for parallelisation.

    Parameters
    ----------
    array : ndarray
    mask : ndarray, optional

    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # Do destriping on masked array if set
        if mask is not None:
            array_copy = array.copy()
            array_copy[mask > 0] = np.nan
            med = np.expand_dims(clipped_median(array_copy, axis=1, sigma_lower=3, sigma_upper=2), axis=1)

        # Otherwise on full array
        else:
            med = np.expand_dims(clipped_median(array, axis=1, sigma_lower=3, sigma_upper=2), axis=1)

        # Return destriped array
        return array - med + clipped_median(array, sigma_lower=3, sigma_upper=2)


def circular_mask(array, coordinates, radius):
    """ Construct circular mask """
    (i1, i2), (nx, ny) = coordinates, array.shape
    y, x = np.ogrid[-i1:nx-i1, -i2:ny-i2]
    return x*x + y*y <= radius * radius
