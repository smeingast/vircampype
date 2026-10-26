import itertools
import warnings
from typing import List, Tuple

import numpy as np
from astropy import wcs
from astropy.convolution import (
    Box2DKernel,
    Gaussian1DKernel,
    Gaussian2DKernel,
    Kernel2D,
    convolve,
    interpolate_replace_nans,
)
from astropy.coordinates import SkyCoord
from astropy.modeling.functional_models import Gaussian1D
from astropy.stats import sigma_clip as astropy_sigma_clip
from astropy.stats import sigma_clipped_stats
from astropy.wcs.utils import wcs_to_celestial_frame
from scipy import ndimage as ndi
from scipy.interpolate import SmoothBivariateSpline, UnivariateSpline
from scipy.ndimage import binary_closing, generate_binary_structure, median_filter
from scipy.stats import binned_statistic_2d
from skimage.filters import sobel
from sklearn.neighbors import NearestNeighbors

from vircampype.external.mmm import mmm
from vircampype.tools.mathtools import *
from vircampype.tools.mmmtools import batch_is_vectorised
from vircampype.tools.mmmtools import mmm as mmm_fast
from vircampype.tools.mmmtools import mmm_batch

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
    "always_masked_fraction",
    "get_edge_skycoord_weight",
    "coordinate_array",
    "tile_image",
]


def interpolate_image(data, kernel=None, max_bad_neighbors=None):
    """
    Interpolates NaNs in an image. NaNs are replaced by convolving the original image
    with a kernel from which the pixel values are copied. This technique is much faster
    than other approaches involving spline fitting (e.g. griddata or scipy interpolation
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

        # Get the NaNs which were skipped
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
    if isinstance(kernel, Kernel2D):
        kernel_array = kernel.array
    elif isinstance(kernel, np.ndarray):
        kernel_array = kernel
    else:
        raise ValueError("Supplied kernel not supported")
    if (kernel_array.shape[0] % 2 == 0) or (kernel_array.shape[1] % 2 == 0):
        # astropy.convolution also rejects even-sized kernels (no center px)
        raise ValueError("Kernel dimensions must be odd")
    # convolve() flips the kernel (convolution, not correlation); flip here
    # too so asymmetric kernels keep the historical orientation.
    kernel_array = kernel_array[::-1, ::-1]

    # Compute interpolation values only at NaN positions, not over the full
    # frame (~1-2% of pixels vs ~20x cost for astropy convolve). Replicates
    # convolve(array, kernel, boundary="extend", nan_treatment="interpolate",
    # normalize_kernel=True): kernel-weighted mean over finite neighbors,
    # renormalized by their weight sum, out-of-frame neighbors edge-clipped.
    n_rows, n_cols = array.shape
    k_rows, k_cols = kernel_array.shape
    all_row, all_col = np.where(nans)
    off_row = np.arange(k_rows) - k_rows // 2
    off_col = np.arange(k_cols) - k_cols // 2
    # Chunked to bound the (N, k, k) gather on heavily masked frames. Assign
    # only after all chunks are gathered, so every pixel sees its original
    # (pre-fill) neighbors.
    filled = []
    for c0 in range(0, all_row.size, 1_000_000):
        idx_row = all_row[c0 : c0 + 1_000_000]
        idx_col = all_col[c0 : c0 + 1_000_000]
        neigh_row = np.clip(idx_row[:, None] + off_row[None, :], 0, n_rows - 1)
        neigh_col = np.clip(idx_col[:, None] + off_col[None, :], 0, n_cols - 1)
        vals = array[neigh_row[:, :, None], neigh_col[:, None, :]].astype(np.float64)
        finite = np.isfinite(vals)
        weights = np.broadcast_to(kernel_array, vals.shape)
        num = np.where(finite, vals * weights, 0.0).sum(axis=(1, 2))
        den = np.where(finite, weights, 0.0).sum(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            filled.append(num / den)
    array[all_row, all_col] = np.concatenate(filled) if filled else []

    # Fill skipped NaNs back in
    if nans_skipped is not None:
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
        raise ValueError(f"Axis={axis:0d} not supported")

    # If there is no overlap, we can just use the numpy function
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
        List of locations where the cut occurred (returned by chop_image)
    axis : int, optional
        Axis along which the chop occurred. Default is 0.
    overlap : int, optional
        Overlap used in chopping.

    Returns
    -------
    np.ndarray
        Merged array.

    """

    # Axis must be 0 or 1
    if axis not in [0, 1]:
        raise ValueError(f"Axis={axis:0d} not supported")

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


def background_image(image, mesh_size, mesh_filtersize=3, vectorized_mmm=False):
    """
    Estimate a 2-D background map and its standard deviation.

    Tiles the image into a grid of *mesh_size* × *mesh_size* pixels, estimates
    the sky background in each tile with the MMM algorithm, applies a median
    filter and Gaussian convolution for smoothing, and finally upscales the
    result back to the original image dimensions.

    Parameters
    ----------
    image : np.ndarray
        2-D input image. Dimensions must be exact multiples of *mesh_size*.
    mesh_size : int
        Side length of each background mesh tile in pixels.
    mesh_filtersize : int, optional
        Size of the median filter applied to the mesh grid. Default is 3.
    vectorized_mmm : bool, optional
        Estimate all tiles at once with the vectorised MMM
        (``vircampype.tools.mmmtools``); results equal to float32 rounding.
        Default False (original per-tile estimator).

    Returns
    -------
    background : np.ndarray
        2-D background image, same shape as *image*.
    background_std : np.ndarray
        2-D background standard-deviation image, same shape as *image*.

    Raises
    ------
    ValueError
        If *image* is not 2-D or if its dimensions are not multiples of
        *mesh_size*.
    """
    # Image must be 2D
    if len(image.shape) != 2:
        raise ValueError(
            f"Please supply array with 2 dimensions. The given data has {len(image.shape)} dimensions"
        )

    # Back size and image dimensions must be compatible
    if (image.shape[0] % mesh_size != 0) or (image.shape[1] % mesh_size != 0):
        raise ValueError(
            f"Image dimensions {image.shape} must be multiple of backsize mesh size ({mesh_size})"
        )

    n_tiles_y, n_tiles_x = image.shape[0] // mesh_size, image.shape[1] // mesh_size
    if vectorized_mmm and batch_is_vectorised(n_tiles_y * n_tiles_x, mesh_size**2):
        # One row per mesh tile (row-major tile order), all tiles in one call
        tiles = (
            image.reshape(n_tiles_y, mesh_size, n_tiles_x, mesh_size)
            .swapaxes(1, 2)
            .reshape(n_tiles_y * n_tiles_x, mesh_size * mesh_size)
        )
        bg, bg_std, _ = mmm_batch(tiles, axis=1)
        del tiles
    else:
        # Tile image (views) and estimate the background tile by tile. Large
        # tiles are sort-bound, so the batch would fall back to this anyway.
        tiles = [
            image[x : x + mesh_size, y : y + mesh_size]
            for x in range(0, image.shape[0], mesh_size)
            for y in range(0, image.shape[1], mesh_size)
        ]
        estimator = mmm_fast if vectorized_mmm else mmm
        bg, bg_std, _ = list(zip(*[estimator(t) for t in tiles]))
        bg, bg_std = np.array(bg), np.array(bg_std)

    # Scale back to 2D array
    bg, bg_std = (
        bg.reshape(n_tiles_y, n_tiles_x),
        bg_std.reshape(n_tiles_y, n_tiles_x),
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
    return upscale_image(bg, new_size=image.shape[::-1]), upscale_image(  # noqa
        bg_std,
        new_size=image.shape[::-1],  # noqa
    )


def upscale_image(image, new_size, method="pil", order=3):
    """
    Resizes a 2D array to given new size.

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
        # Determine edge coordinates of input wrt output size
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

            # Return interpolated spline
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
        If True, rescale output to (x_max - x_min, y_max - y_min). Default is True.
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

    # Clipped median
    elif metric == "median":
        _, gv, _ = sigma_clipped_stats(vc[idx], axis=1)

    else:
        raise ValueError(f"Metric '{metric}' not supported")

    # Reshape to image
    gv = gv.reshape(n_bins_y, n_bins_x)

    # Apply some filters and return
    i = 0
    while True:
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


def destripe_helper(array, mask=None, smooth=False, vectorized_mmm=False):
    """
    Destripe helper for parallelisation.

    Parameters
    ----------
    array : np.ndarray
    mask : np.ndarray, optional
    smooth : bool, optional
    vectorized_mmm : bool, optional
        Estimate all row statistics at once with the vectorised MMM
        (``vircampype.tools.mmmtools``); results equal to float32 rounding.
        Default False (original per-row estimator).

    Returns
    -------
    np.ndarray
        1-D array of estimated row stripe values (zero-meaned).

    Notes
    -----
    The row profile itself is never sigma-clipped: on the profile a strong stripe
    and a bad row are indistinguishable, so any such clip removes exactly the
    strongest stripes from the correction. Rows are instead rejected only when the
    row statistic failed (mmm sigma not finite or <= 0) and are then interpolated
    from their neighbours.
    """

    # Copy array
    array_copy = array.copy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # Apply mask if set
        if mask is not None:
            array_copy[mask > 0] = np.nan

        # Compute sky mode and sigma in each row
        if vectorized_mmm:
            med_destripe, sig_destripe, _ = mmm_batch(array_copy, axis=1, minsky=10)
        else:
            row_stats = [mmm(v, minsky=10) for v in array_copy]
            med_destripe = np.array([s[0] for s in row_stats], dtype=float)
            sig_destripe = np.array([s[1] for s in row_stats], dtype=float)

        # Reject only rows where the row statistic failed
        bad_rows = ~(np.isfinite(sig_destripe) & (sig_destripe > 0))
        med_destripe[bad_rows] = np.nan

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

        # Remove the large-scale variation along y if set. The smoothing budget is
        # tied to the profile variance so that FITPACK always returns the
        # least-squares cubic (an absolute budget makes the spline follow the
        # stripes once their RMS exceeds it and the correction then cancels).
        if smooth:
            yy = np.arange(len(med_destripe))  # noqa
            ss = 2 * np.sum((med_destripe - np.mean(med_destripe)) ** 2)
            med_destripe_interp = UnivariateSpline(yy, med_destripe, k=3, s=ss)(yy)
            med_destripe -= med_destripe_interp

        # Return stripe pattern
        return med_destripe - np.nanmedian(med_destripe)


def _clipped_stats(values: np.ndarray, n_iter: int = 5, kappa: float = 3.0):
    """Iterative kappa-sigma clipped median and MAD-based sigma of finite values."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.nan, np.nan
    med, sig = np.median(v), 1.4826 * np.median(np.abs(v - np.median(v)))
    for _ in range(n_iter):
        vv = v[np.abs(v - med) < kappa * sig]
        if vv.size == 0:  # zero scatter: keep the last estimate
            break
        med, sig = np.median(vv), 1.4826 * np.median(np.abs(vv - np.median(vv)))
    return med, sig


def _drop_small(mask: np.ndarray, n_min: int):
    """Remove connected components smaller than n_min; also return labels and areas."""
    labels, n = ndi.label(mask)
    if n == 0:
        return mask, labels, np.zeros(1, int)
    area = np.bincount(labels.ravel())
    area[0] = 0
    return (area >= n_min)[labels], labels, area


def source_mask(
    image: np.ndarray,
    kappa_compact: float = 3.0,
    kappa_extended: float = 3.0,
    max_fraction: float = 0.75,
) -> np.ndarray:
    """
    Boolean source mask of one detector plane for the sky build.

    1. One global level and noise per plane (clipped median / MAD; no local
       background mesh, so extended emission is not absorbed into the background).
    2. Compact sources: the 1 px Gaussian-smoothed plane above level +
       kappa_compact sigma, components >= 5 px, grown by 2 px. Components >= 50 px
       also get a disk of radius 1.5 r_eq + 3 (saturated stars, halos).
    3. Extended emission: 8x8 block medians (compact sources excluded), 3x3
       smoothed, minus a robustly fitted plane (absorbs sky gradients), above
       kappa_extended sigma_b (bin-to-bin noise from the high-pass residual),
       components >= 6 bins, grown by 1 bin.
    4. Union. If more than max_fraction of the plane is masked, only the compact
       mask without the halo disks is returned (keeps sky in crowded planes).

    Parameters
    ----------
    image : np.ndarray
        2D image; NaN marks bad pixels (never detected).
    kappa_compact : float, optional
        Detection threshold of the compact sources (sigma of the smoothed plane).
    kappa_extended : float, optional
        Detection threshold of the extended emission (sigma of the binned plane).
    max_fraction : float, optional
        Maximum masked fraction before falling back to the compact mask.

    Returns
    -------
    np.ndarray
        Boolean source mask.

    """
    bs = 8  # bin size of the extended component
    x = np.asarray(image, dtype=np.float32)
    ny, nx = x.shape
    bad = ~np.isfinite(x)
    lvl, _ = _clipped_stats(x[::4, ::4])
    if not np.isfinite(lvl):  # no valid pixels: nothing to detect
        return np.zeros(x.shape, dtype=bool)
    xf = np.where(bad, lvl, x)

    # Compact sources on the 1 px smoothed plane
    xs = ndi.gaussian_filter(xf, 1.0)
    _, sig_s = _clipped_stats(xs[::4, ::4])
    sig_s = sig_s if sig_s > 0 else np.inf  # flat plane: no compact detections
    core, labels, area = _drop_small((xs > lvl + kappa_compact * sig_s) & ~bad, 5)
    mask_compact = ndi.binary_dilation(core, iterations=2)
    mask = mask_compact.copy()

    # Halo disks (radius 1.5 r_eq + 3) for components >= 50 px
    big = np.where(area >= 50)[0]
    if len(big):
        cy, cx = np.array(ndi.center_of_mass(core, labels, big)).T
        rr = 1.5 * np.sqrt(area[big] / np.pi) + 3
        yy, xx = np.ogrid[:ny, :nx]
        for y0, x0, r in zip(cy, cx, rr):
            y1, y2 = int(max(y0 - r, 0)), int(min(y0 + r + 1, ny))
            x1, x2 = int(max(x0 - r, 0)), int(min(x0 + r + 1, nx))
            sub = (yy[y1:y2] - y0) ** 2 + (xx[:, x1:x2] - x0) ** 2 <= r * r
            mask[y1:y2, x1:x2] |= sub

    # Extended emission on 8x8 block medians (compact sources and bad pixels excluded)
    nyb, nxb = ny // bs, nx // bs
    xb = np.where(core | bad, np.nan, x)[: nyb * bs, : nxb * bs]
    xb = xb.reshape(nyb, bs, nxb, bs).swapaxes(1, 2).reshape(nyb, nxb, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks
        xb = np.nanmedian(xb, axis=2)
    xb = np.where(np.isfinite(xb), xb, lvl)
    xbs = ndi.uniform_filter(xb, 3)
    _, sig_b = _clipped_stats(xbs - ndi.median_filter(xbs, 7))
    sig_b = sig_b if sig_b > 0 else np.inf  # flat binned plane: no extended part

    # Robust plane fit (absorbs residual sky gradients, not compact nebulae)
    yb, xbi = np.mgrid[:nyb, :nxb] / nyb - 0.5
    amat = np.array([np.ones_like(yb).ravel(), xbi.ravel(), yb.ravel()]).T
    use = np.ones(amat.shape[0], bool)
    for _ in range(6):
        coeff = np.linalg.lstsq(amat[use], xbs.ravel()[use], rcond=None)[0]
        res = xbs.ravel() - amat @ coeff
        use = (res < kappa_extended * sig_b) & (res > -3 * sig_b)
    ext, _, _ = _drop_small((res > kappa_extended * sig_b).reshape(xbs.shape), 6)
    ext = ndi.binary_dilation(ext, iterations=1)
    ext = np.kron(ext, np.ones((bs, bs), bool))
    mask[: ext.shape[0], : ext.shape[1]] |= ext

    # Cap: a confusion-limited plane keeps only its compact mask
    if mask.mean() > max_fraction:
        return mask_compact
    return mask


def always_masked_fraction(
    masks: np.ndarray, bpm: np.ndarray | None = None, frac: float = 0.8
) -> float:
    """
    Fraction of pixels masked in at least frac of the frames (QC of fixed features).

    Parameters
    ----------
    masks : np.ndarray
        Boolean source masks of one detector, shape (n_frames, ny, nx).
    bpm : np.ndarray, optional
        Boolean bad pixel mask (ny, nx); its pixels are not counted.
    frac : float, optional
        Minimum fraction of frames.

    Returns
    -------
    float
        Fraction of all plane pixels that are masked in >= frac of the frames.

    """
    n_masked = masks.sum(axis=0, dtype=np.uint16)
    always = n_masked >= np.ceil(frac * len(masks))
    if bpm is not None:
        always &= ~bpm
    return float(always.mean())


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


def coordinate_array(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Creates arrays of same shape as input, but holds x and y pixel coordinates."""
    return np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))


def tile_image(image, s1: int, s2: int) -> List:
    """Splits an image into tiles of given size s1 and s2."""
    return [
        image[x : x + s1, y : y + s2]
        for x in range(0, image.shape[0], s1)
        for y in range(0, image.shape[1], s2)
    ]
