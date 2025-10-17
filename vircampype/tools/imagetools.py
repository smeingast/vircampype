import warnings
import itertools
import numpy as np

from typing import List
from astropy import wcs
from skimage.draw import disk
from skimage.filters import sobel
from scipy.ndimage import median_filter
from vircampype.external.mmm import mmm
from vircampype.tools.mathtools import *
from astropy.coordinates import SkyCoord
from skimage import measure as skmeasure
from scipy.stats import binned_statistic_2d
from astropy.stats import sigma_clipped_stats
from skimage import morphology as skmorphology
from sklearn.neighbors import NearestNeighbors
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.modeling.functional_models import Gaussian1D
from astropy.stats import sigma_clip as astropy_sigma_clip
from scipy.ndimage import binary_closing, generate_binary_structure
from scipy.interpolate import UnivariateSpline, SmoothBivariateSpline
from astropy.convolution import (
    convolve,
    interpolate_replace_nans,
    Gaussian2DKernel,
    Kernel2D,
    Box2DKernel,
    Gaussian1DKernel,
)

__all__ = [
    "interpolate_image",
    "chop_image",
    "merge_chopped",
    "background_image",
    "upscale_image",
    "grid_value_2d",
    "grid_value_2d_nn",
    "destripe_helper",
    "source_mask",
    "get_edge_skycoord_weight",
    "coordinate_array",
    "tile_image",
]


def interpolate_image(data, kernel=None, max_bad_neighbors=None):
    """
    Interpolates NaNs in an image. NaNs are replaced by convolving the original image
    with a kernel from which the pixel values are copied. This technique is much faster
    than other aporaches involving spline fitting (e.g. griddata or scipy inteprolation
     methods.)

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
        nans_conv = convolve(
            nans, kernel=nan_kernel, boundary="extend", normalize_kernel=False
        )

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
    List
        List of sub-arrays constructed from the input

    """

    # Axis must be 0 or 1
    if axis not in [0, 1]:
        raise ValueError("Axis={0:0d} not supported".format(axis))

    # If there is no overlap, we can just u se the numpy function
    if overlap is None:
        return np.array_split(ary=array, indices_or_sections=npieces, axis=axis)

    # Determine where to chop
    cut = list(
        np.int32(np.round(np.linspace(0, array.shape[axis], npieces + 1), decimals=0))
    )

    # Force the first and last cut location just to be safe from any integer conversion
    # issues
    cut[0], cut[-1] = 0, array.shape[axis]

    chopped = []
    for i in range(npieces):
        if axis == 0:
            # First slice
            if i == 0:
                chopped.append(array[cut[i] : cut[i + 1] + overlap, :])

            # Last slice
            elif i == npieces - 1:
                chopped.append(array[cut[i] - overlap : cut[i + 1], :])

            # Everything else
            else:
                chopped.append(array[cut[i] - overlap : cut[i + 1] + overlap, :])

        elif axis == 1:
            # First slice
            if i == 0:
                chopped.append(array[:, cut[i] : cut[i + 1] + overlap])

            # Last slice
            elif i == npieces - 1:
                chopped.append(array[:, cut[i] - overlap : cut[i + 1]])

            # Everything else
            else:
                chopped.append(array[:, cut[i] - overlap : cut[i + 1] + overlap])

    # Return list of chopped arrays
    return chopped, cut


def merge_chopped(arrays, locations, axis=0, overlap=0):
    """
    Complementary to the above function, this one merges the chopped array back into
    the original.

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
    shape = (
        (locations[-1], arrays[0].shape[otheraxis])
        if axis == 0
        else (arrays[0].shape[otheraxis], locations[-1])
    )

    merged = np.empty(shape=shape, dtype=arrays[0].dtype)
    for i in range(len(arrays)):
        if axis == 0:
            # First slice
            if i == 0:
                merged[0 : locations[i + 1], :] = arrays[i][
                    : arrays[i].shape[0] - overlap, :
                ]

            # Last slice
            elif i == len(arrays) - 1:
                merged[locations[i] :, :] = arrays[i][overlap:, :]

            # In between
            else:
                merged[locations[i] : locations[i + 1], :] = arrays[i][
                    overlap:-overlap, :
                ]

        elif axis == 1:
            # First slice
            if i == 0:
                merged[:, 0 : locations[i + 1]] = arrays[i][
                    :, : arrays[i].shape[1] - overlap
                ]

            # Last slice
            elif i == len(arrays) - 1:
                merged[:, locations[i] :] = arrays[i][:, overlap:]

            # In between
            else:
                merged[:, locations[i] : locations[i + 1]] = arrays[i][
                    :, overlap:-overlap
                ]

    return merged


def background_image(image, mesh_size, mesh_filtersize=3):
    # Image must be 2D
    if len(image.shape) != 2:
        raise ValueError(
            f"Please supply array with 2 dimensions. The given data has {len(image.shape)} dimensions"
        )

    # Back size and image dimensions must be compatible
    if (image.shape[0] % mesh_size != 0) | (image.shape[1] % mesh_size != 0):
        raise ValueError(
            f"Image dimensions {image.shape} must be multiple of backsize mesh size ({mesh_size})"
        )

    # Tile image
    tiles = [
        image[x : x + mesh_size, y : y + mesh_size]
        for x in range(0, image.shape[0], mesh_size)
        for y in range(0, image.shape[1], mesh_size)
    ]

    # Estimate background for each tile
    bg, bg_std, _ = list(zip(*[mmm(t) for t in tiles]))

    # Scale back to 2D array
    n_tiles_x, n_tiles_y = image.shape[1] // mesh_size, image.shape[0] // mesh_size
    bg, bg_std = (
        np.array(bg).reshape(n_tiles_y, n_tiles_x),
        np.array(bg_std).reshape(n_tiles_y, n_tiles_x),
    )

    # Interpolate NaN values in grid
    bg = interpolate_replace_nans(bg, kernel=Gaussian2DKernel(3), boundary="extend")
    bg_std = interpolate_replace_nans(
        bg_std, kernel=Gaussian2DKernel(3), boundary="extend"
    )

    # Apply median filter
    bg, bg_std = (
        median_filter(input=bg, size=mesh_filtersize),
        median_filter(input=bg_std, size=mesh_filtersize),
    )

    # Convolve
    bg = convolve(bg, kernel=Gaussian2DKernel(1), boundary="extend")
    bg_std = convolve(bg_std, kernel=Gaussian2DKernel(1), boundary="extend")

    # Return upscaled data
    return upscale_image(bg, new_size=image.shape), upscale_image(  # noqa
        bg_std,
        new_size=image.shape,  # noqa
    )


def upscale_image(image, new_size, method="pil", order=3):
    """
    Resizes a 2D array to tiven new size.

    An example of how to upscale with PIL:
    apc_plot = np.array(Image.fromarray(apc_grid).resize(size=(hdr["NAXIS1"],
    hdr["NAXIS2"]), resample=Image.LANCZOS))

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

        return np.array(
            Image.fromarray(image).resize(size=new_size, resample=Image.BICUBIC)  # noqa
        )
    elif "spline" in method.lower():
        # Detemrine edge coordinates of input wrt output size
        xedge, yedge = (
            np.linspace(0, new_size[0], image.shape[0] + 1),
            np.linspace(0, new_size[1], image.shape[1] + 1),
        )

        # Determine pixel center coordinates
        xcenter, ycenter = (xedge[1:] + xedge[:-1]) / 2, (yedge[1:] + yedge[:-1]) / 2

        # Make coordinate grid
        xcenter, ycenter = np.meshgrid(xcenter, ycenter)

        # Fit spline to grid
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            spline_fit = SmoothBivariateSpline(
                xcenter.ravel(), ycenter.ravel(), image.ravel(), kx=order, ky=order
            ).ev

            # Return interplated spline
            return spline_fit(
                *np.meshgrid(np.arange(new_size[0]), np.arange(new_size[1]))
            )
    else:
        raise ValueError(f"Method '{method}' not supported")


def grid_value_2d(
    x,
    y,
    value,
    x_min,
    y_min,
    x_max,
    y_max,
    nx,
    ny,
    conv=True,
    kernel_size=2,
    weights=None,
    upscale=True,
    interpolate_nan=True,
):
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

        stat, xe, ye, (nbx, nby) = binned_statistic_2d(
            x=x[good],
            y=y[good],
            values=value[good],
            bins=[nx, ny],
            statistic=clipped_median,
            expand_binnumbers=True,
            range=[(x_min, x_max), (y_min, y_max)],  # noqa
        )  # noqa

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
                # TODO: This should use astropy sigma clipping
                value_mask = value[good][fil].copy()
                mask = np.isnan(
                    apply_sigma_clip(value_mask, sigma_level=3, sigma_iter=3)
                )

                # Check sum of weights
                if np.sum(weights[good][fil]) < 0.0001:
                    stat[cidx[0], cidx[1]] = np.nan
                else:
                    # Compute weighted average for this bin
                    stat[cidx[0], cidx[1]] = np.average(
                        value[good][fil][mask], weights=weights[good][fil][mask]
                    )

        # Transpose
        stat = stat.T

    # Smooth
    if conv:
        stat = convolve(
            stat, kernel=Gaussian2DKernel(x_stddev=kernel_size), boundary="extend"
        )

    if interpolate_nan:
        stat = interpolate_replace_nans(stat, kernel=Gaussian2DKernel(2))

    # Upscale with spline
    if upscale:
        return upscale_image(image=stat, new_size=(x_max - x_min, y_max - y_min))

    return stat


def grid_value_2d_nn(
    x,
    y,
    values,
    n_nearest_neighbors,
    n_bins_x,
    n_bins_y,
    x_min,
    x_max,
    y_min,
    y_max,
    metric="median",
    weights=None,
):
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
    metric : str, optional
        Method to use to calculate grid values.
    weights : np.ndarray, optional
        If metric is weighted, supply weights.

    Returns
    -------
    np.ndarray
        Interpolated 2D array.

    """

    # Copy arrays, apply global sigma clip, and throw out bad values
    xc, yc, vc = x.copy(), y.copy(), values.copy()
    keep = np.isfinite(xc) & np.isfinite(yc) & np.isfinite(vc)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        keep &= ~astropy_sigma_clip(vc, sigma=2.5).mask
    xc, yc, vc = xc[keep], yc[keep], vc[keep]
    wc = weights.copy()[keep] if weights is not None else None

    # Determine step size in grid in X and Y
    step_x, step_y = (x_max - x_min) / n_bins_x, (y_max - y_min) / n_bins_y

    # Create grid of pixel centers
    xg, yg = np.meshgrid(
        np.linspace(x_min + step_x / 2, x_max - step_x / 2, n_bins_x),
        np.linspace(y_min + step_y / 2, y_max - step_y / 2, n_bins_y),
    )

    # Get nearest neighbors to grid pixel centers
    stacked_grid = np.stack([xg.ravel(), yg.ravel()]).T
    stacked_data = np.stack([xc, yc]).T
    nn = len(xc) if len(xc) < n_nearest_neighbors else n_nearest_neighbors
    nn_dis, idx = (
        NearestNeighbors(n_neighbors=nn).fit(stacked_data).kneighbors(stacked_grid)
    )

    # Obtain median values at each grid pixel
    if metric == "weighted":
        # Weights must be provided
        if weights is None:
            raise ValueError("Must provide weights")

        # Sigma-clip weights
        # vv, ww = vc[idx].copy(), wc[idx].copy()
        vv, ww = (
            vc[idx].copy(),
            wc[idx] * Gaussian1D(amplitude=1, mean=0, stddev=540)(nn_dis),
        )
        ww[astropy_sigma_clip(vv, sigma=2.5, maxiters=5, axis=1).mask] = 0.0

        # Compute weighted average
        gv = np.average(vv, weights=ww, axis=1)

    # Clipped mean
    elif metric == "mean":
        gv, _, _ = sigma_clipped_stats(vc[idx], axis=1)

    # CLipped median
    elif metric == "median":
        _, gv, _ = sigma_clipped_stats(vc[idx], axis=1)

    else:
        raise ValueError(f"Metric '{metric}' not supported")

    # Reshape to image
    gv = gv.reshape(n_bins_y, n_bins_x)

    # Apply some filters and return
    while True:
        i = 0
        if np.sum(~np.isfinite(gv)) == 0:
            break
        gv = interpolate_replace_nans(gv, kernel=Box2DKernel(3))
        if i > 50:
            raise ValueError("Interpolation did not converge")
        i += 1

    # Median filter
    gv = median_filter(gv, size=3)

    # Return convolved array (kernel size ~ 10% of image)
    return convolve(
        gv, kernel=Gaussian2DKernel((n_bins_x + n_bins_y) / 20), boundary="extend"
    )


def destripe_helper(array, mask=None, smooth=False):
    """
    Destripe helper for parallelisation.

    Parameters
    ----------
    array : np.ndarray
    mask : np.ndarray, optional
    smooth : bool, optional

    """

    # Copy array
    array_copy = array.copy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # Apply mask if set
        if mask is not None:
            array_copy[mask > 0] = np.nan

        # Compute sky values in each row
        med_destripe = np.array([mmm(v, minsky=10)[0] for v in array_copy])

        # Interpolate NaNs until there are none.
        ii = 0
        while True:
            if np.sum(~np.isfinite(med_destripe)) == 0:
                break
            med_destripe = interpolate_replace_nans(
                med_destripe, kernel=Gaussian1DKernel(5), boundary="extend"
            )
            ii += 1
            if ii > 100:
                raise ValueError("Destriping failed")

        # Apply smoothing if set
        if smooth:
            yy = np.arange(len(med_destripe))  # noqa
            med_destripe_interp = UnivariateSpline(
                yy, med_destripe, k=3, s=100 * len(yy)
            )(yy)
            med_destripe -= med_destripe_interp

        # Return stripe pattern
        return med_destripe - np.nanmedian(med_destripe)


def source_mask(
    image: np.ndarray,
    kappa: (int, float),
    min_area: int = 3,
    max_area: int = 100000,
    mask_bright_sources: bool = True,
):
    """
    Create source mask from input images based on thresholding

    Parameters
    ----------
    image : np.ndarray
        Input image.
    kappa : float, int
        Threshold in background sigmas
    min_area : int, optional
        Minimum area of sources in pixels.
    max_area : int, optional
        Maximum area of sources in pixels.
    mask_bright_sources : bool, optional
        Whether additional circualr patches should be drawn on bright sources.

    Returns
    -------
    np.ndarray
        Source mask.

    """

    # Make empty label image
    image_labels = np.full_like(image, dtype=np.uint8, fill_value=0)

    # Create background map
    bg, bgstd = background_image(image=image, mesh_size=128, mesh_filtersize=3)

    # Threshold source map
    sources = image > bg + kappa * bgstd

    # Label regions
    labels, n_labels = skmeasure.label(sources, return_num=True)

    # If no significant region, return
    if n_labels < 1:
        return image_labels

    # Generate regionprops
    regionprops = skmeasure.regionprops(labels)

    # Measure size of regions
    sizes = np.array([r.area for r in regionprops])

    # Get index of large labels
    if mask_bright_sources:
        idx_large_all = [
            i for i, x in enumerate((sizes > 200) & (sizes < max_area)) if x
        ]
    else:
        idx_large_all = []

    # Empty list to store masks
    mask_large = np.full_like(image, dtype=np.uint16, fill_value=0)

    # Create large region circular masks
    if len(idx_large_all) > 0:
        # Loop over large sources
        for idx_large in idx_large_all:
            # Skip if eccentricity is too large (e.g. bad rows or columns)
            if regionprops[idx_large].eccentricity > 0.8:
                continue

            # Grab current size and coordinates
            csize = sizes[idx_large]

            # Approximate radius of circular mask with radius (and add 20%)
            crad = 1.2 * np.sqrt(csize / np.pi)

            # Determine centroid
            centroid = regionprops[idx_large].centroid

            # Construct mask for current source
            aa, bb = disk(
                centroid, np.ceil(2 * crad).astype(int), shape=mask_large.shape
            )
            mask_large[aa, bb] = 1

    # Find those sources outside the given thresholds and set to 0
    # bad_sources = (sizes < minarea) | (sizes > maxarea)
    # if maxarea is not None else sizes < minarea
    bad_sources = (
        (sizes < min_area) | (sizes > max_area)
        if max_area is not None
        else sizes < min_area
    )
    assert isinstance(bad_sources, np.ndarray)

    # Only if there are bad sources
    if np.sum(bad_sources) > 0:
        labels[bad_sources[labels - 1]] = 0  # labels starts with 1

    # Set background to 0, sources to 1 and convert to 8bit unsigned integer
    labels[labels > 0], labels[labels < 0] = 1, 0
    labels = labels.astype(bool)

    # Dilate the mask
    labels = skmorphology.binary_dilation(skmorphology.binary_closing(labels))

    # Apply mask
    image_labels[labels] = 1

    # Apply large region mask
    image_labels[mask_large > 0] = 1

    # Return mask
    return image_labels


def get_edge_skycoord_weight(weight_img: np.ndarray, weight_wcs: wcs.WCS) -> SkyCoord:
    """
    Determine the (sorted) sky coordinates of the edges in a weightmap.

    Parameters
    ----------
    weight_img : np.ndarray
        Weight image array.
    weight_wcs : wcs.WCS
        WCS instance of the weight image for coordinate transformation.

    Returns
    -------
    SkyCoord
        SkyCoord instance with coordinates of the (sorted) edge pixels.

    """

    # Determine frame edge
    weight_flat = np.full_like(weight_img, fill_value=0, dtype=int)
    weight_flat[weight_img > 0] = 1
    weight_flat = binary_closing(
        weight_flat, structure=generate_binary_structure(2, 2), iterations=50
    )
    weight_edge = sobel(weight_flat)  # noqa
    weight_edge[weight_edge > 0] = 1  # noqa
    weight_edge = weight_edge.astype(np.int8)  # noqa

    # Determine edge coordinates
    coo_edge_y, coo_edge_x = np.where(weight_edge > 0)

    # Transform so that origin is approximately in center
    temp_x = coo_edge_x - np.mean(coo_edge_x)
    temp_y = coo_edge_y - np.mean(coo_edge_y)

    # Transform to polar coordinates for sorting
    theta, _ = cart2pol(temp_x, temp_y)
    sort_idx = np.argsort(theta)

    # Compute and sort skycoordinates by polar angle
    skycoo_edge = weight_wcs.wcs_pix2world(
        coo_edge_x[sort_idx], coo_edge_y[sort_idx], 0
    )
    return SkyCoord(
        *skycoo_edge, frame=wcs_to_celestial_frame(weight_wcs), unit="degree"
    )


def coordinate_array(image: np.ndarray) -> (np.ndarray, np.ndarray):
    """Creates arrays of same shape as input, but holds x and y pixel coordinates."""
    return np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))


def tile_image(image, s1: int, s2: int) -> List:
    """Splits an image into tiles of given size s1 and s2."""
    return [
        image[x : x + s1, y : y + s2]
        for x in range(0, image.shape[0], s1)
        for y in range(0, image.shape[1], s2)
    ]
