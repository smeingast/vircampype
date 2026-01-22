import os
import warnings
from itertools import repeat
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from joblib import Parallel, delayed
from numpy.typing import ArrayLike

from vircampype.external.mmm import mmm
from vircampype.pipeline.setup import Setup
from vircampype.tools.imagetools import *
from vircampype.tools.mathtools import *
from vircampype.tools.systemtools import *


class ImageCube(object):
    def __init__(self, setup, cube=None):
        """
        Parameters
        ----------
        setup : str, Setup
            Setup dictionary passed from the pipeline instance.
        cube : np.ndarray, optional

        """

        # Set setup
        self.setup = Setup.load_pipeline_setup(setup)

        # Check supplied data and load into attribute
        if cube is not None:
            if len(cube.shape) == 2:
                self.cube = np.expand_dims(cube, axis=0)
            elif len(cube.shape) != 3:
                raise ValueError("Supplied data not compatible with ImageCube")
            else:
                self.cube = cube
        else:
            self.cube = cube

    def __str__(self):
        """
        Defines behavior for when str() is called on an instance of ImageCube

        Returns
        -------
        str
            Same as for numpy array

        """
        return str(self.cube)

    def __repr__(self):
        """
        Defines behavior for when repr() is called on an instance of ImageCube

        Returns
        -------
        str
            Same as for numpy array

        """
        return repr(self.cube)

    def __len__(self):
        """
        Returns the length of the cube.

        Returns
        -------
        int
            Length of cube

        """

        if self.cube is not None:
            return len(self.cube)
        else:
            return 0

    def __getitem__(self, key):
        """
        Defines behavior for when an item is accessed

        Parameters
        ----------
        key : int, tuple

        """
        return self.cube[key]

    def __le__(self, value):
        """
        Defines behavior for the less-than-or-equal-to operator, <=

        Parameters
        ----------
        value : float, int

        Returns
        -------
        np.ndarray

        """

        with np.errstate(invalid="ignore"):
            return self.cube <= value

    def __lt__(self, value):
        """
        Defines behavior for the less-than operator, <

        Parameters
        ----------
        value : float, int

        Returns
        -------
        np.ndarray

        """

        with np.errstate(invalid="ignore"):
            return self.cube < value

    def __ge__(self, value):
        """
        Defines behavior for the greater-than-or-equal-to operator, >=

        Parameters
        ----------
        value : float, int

        Returns
        -------
        np.ndarray

        """

        with np.errstate(invalid="ignore"):
            return self.cube >= value

    def __gt__(self, value):
        """
        Defines behavior for the greater-than operator, >

        Parameters
        ----------
        value : float, int

        Returns
        -------
        np.ndarray

        """

        with np.errstate(invalid="ignore"):
            return self.cube > value

    def __add__(self, other):
        """
        Implements addition for ImageData. Here the cubes are added together and a new
        ImageCube instance is returned.

        Parameters
        ----------
        other : ImageCube, np.ndarray
            Second ImageData instance or ndarray from which the data is to be added

        Returns
        -------
        ImageCube
            New ImageData instance with added data

        Raises
        ------
        TypeError
            When instances are not compatible

        """

        if isinstance(other, ImageCube):
            return ImageCube(setup=self.setup, cube=self.cube + other.cube)
        elif isinstance(other, (np.ndarray, float, np.float32, int)):
            return ImageCube(setup=self.setup, cube=self.cube + other)
        else:
            raise TypeError(f"Addition for {str(type(other)):s} not implemented")

    def __iadd__(self, other):
        """
        Implements addition with assignment for ImageData. Both ImageCube and ndarrays
        are supported as input.

        Parameters
        ----------
        other : ImageCube, np.ndarray
            Second ImageData instance or ndarray with same shape from which the data is
            to be added.

        Returns
        -------
        ImageCube
            Self with added data

        Raises
        ------
        TypeError
            When instances are not compatible

        """

        # Return based on instance
        if isinstance(other, ImageCube):
            self.cube += other.cube
            return self
        elif isinstance(other, (np.ndarray, float, np.float32, int)):
            self.cube += other
            return self
        else:
            raise TypeError(f"Addition for {str(type(other)):s} not implemented")

    def __sub__(self, other):
        """
        Implements subtraction for ImageData. Here the cubes are subtracted and a new
        ImageCube instance is returned.

        Parameters
        ----------
        other : ImageCube, np.ndarray
            Second ImageData instance or ndarray to be subtracted

        Returns
        -------
        ImageCube
            New ImageData instance with subracted data

        Raises
        ------
        TypeError
            When instances are not compatible

        """

        if isinstance(other, ImageCube):
            return ImageCube(setup=self.setup, cube=self.cube - other.cube)
        elif isinstance(other, (np.ndarray, float, np.float32, int)):
            return ImageCube(setup=self.setup, cube=self.cube - other)
        else:
            raise TypeError(f"Subtraction for {str(type(other)):s} not implemented")

    def __isub__(self, other):
        """
        Implements subtraction with assignment for ImageData. Both ImageCube and
        ndarrays are supported as input.

        Parameters
        ----------
        other : ImageCube, np.ndarray
            Second ImageData instance or ndarray with same shape which is to be
            subtracted.

        Returns
        -------
        ImageCube
            Self with subtracted data

        Raises
        ------
        TypeError
            When instances are not compatible

        """

        # Return based on instance
        if isinstance(other, ImageCube):
            self.cube -= other.cube
            return self
        elif isinstance(other, (np.ndarray, float, np.float32, int)):
            self.cube -= other
            return self
        else:
            raise TypeError(f"Subtraction for {str(type(other)):s} not implemented")

    def __mul__(self, other):
        """
        Implements multiplication for ImageData. Here the cubes are multiplied and a
        new ImageCube instance is returned.

        Parameters
        ----------
        other : ImageCube, np.ndarray
            Second ImageData instance or ndarray

        Returns
        -------
        ImageCube
            New ImageData instance with multiplied data

        Raises
        ------
        TypeError
            When instances are not compatible

        """

        if isinstance(other, ImageCube):
            return ImageCube(setup=self.setup, cube=self.cube * other.cube)
        elif isinstance(other, (np.ndarray, float, np.float32, int)):
            return ImageCube(setup=self.setup, cube=self.cube * other)
        else:
            raise TypeError(f"Multiplication for {str(type(other)):s} not implemented")

    def __imul__(self, other):
        """
        Implements multiplication with assignment for ImageData. Both ImageCube and
        ndarrays are supported as input.

        Parameters
        ----------
        other : ImageCube, np.ndarray
            Second ImageData instance or ndarray with same shape which is to be
            multiplied with self.

        Returns
        -------
        ImageCube
            Self with multiplied data

        Raises
        ------
        TypeError
            When instances are not compatible

        """

        # Return based on instance
        if isinstance(other, ImageCube):
            self.cube *= other.cube
            return self
        elif isinstance(other, (np.ndarray, float, np.float32, int)):
            self.cube *= other
            return self
        else:
            raise TypeError(f"Multiplication for {str(type(other)):s} not implemented")

    def __truediv__(self, other):
        """
        Implements division for ImageData. Here the cubes are divided and a new
        ImageCube instance is returned.

        Parameters
        ----------
        other : ImageCube, np.ndarray
            Second ImageData instance or ndarray

        Returns
        -------
        ImageCube
            New ImageData instance with divided data

        Raises
        ------
        TypeError
            When instances are not compatible

        """

        if isinstance(other, ImageCube):
            return ImageCube(setup=self.setup, cube=self.cube / other.cube)
        elif isinstance(other, (np.ndarray, float, np.float32, int)):
            return ImageCube(setup=self.setup, cube=self.cube / other)
        else:
            raise TypeError(f"Division for {str(type(other)):s} not implemented")

    def __itruediv__(self, other):
        """
        Implements division with assignment for ImageData. Both ImageCube and ndarrays
        are supported as input.

        Parameters
        ----------
        other : ImageCube, np.ndarray
            Second ImageData instance or ndarray

        Returns
        -------
        ImageCube
            Self with divided data

        Raises
        ------
        TypeError
            When instances are not compatible

        """

        # Return based on instance
        if isinstance(other, ImageCube):
            self.cube /= other.cube
            return self
        elif isinstance(other, (np.ndarray, float, np.float32, int)):
            self.cube /= other
            return self
        else:
            raise TypeError(f"Division for {str(type(other)):s} not implemented")

    def __copy__(self):
        """
        Implements copy for ImageData. Here the cube is copied and a new ImageCube
        instance is returned.

        Returns
        -------
        ImageCube
            New ImageData instance with copied data

        """

        return ImageCube(setup=self.setup, cube=self.cube.copy())

    def copy(self):
        """
        Implements copy for ImageData. Here the cube is copied and a new ImageCube
        """

        return self.__copy__()

    # =========================================================================== #
    # I/O
    # =========================================================================== #
    def write_mef(
        self, path, prime_header=None, data_headers=None, overwrite=False, dtype=None
    ):
        """
        Write MEF Fits file to disk

        Parameters
        ----------
        path : str
            Output file path
        prime_header : fits.Header, optional
            Primary header.
        data_headers : sized, optional
            Data headers.
        overwrite : bool, optional
            Whether existing files should be overwritten.
        dtype
            Optionally force certain data type. e.g. 'float32'.

        """

        # Create List of nothing
        if data_headers is None:
            data_headers = [None] * len(self)

        # Dummy check
        if len(self) != len(data_headers):
            raise ValueError("Supplied headers are not compatible with data format")

        if dtype is None:
            dtype = self.cube.dtype

        # Make HDUList from data and headers
        hdulist = fits.HDUList(
            hdus=[
                fits.ImageHDU(data=d.astype(dtype), header=h)
                for d, h in zip(self.cube[:], data_headers)
            ]
        )

        # Prepend PrimaryHDU and write
        hdulist.insert(0, fits.PrimaryHDU(header=prime_header))
        hdulist.writeto(fileobj=path, overwrite=overwrite)

    # =========================================================================== #
    # Masking
    # =========================================================================== #
    def _mask_max(self):
        """
        Masks the maximum pixel value in the stack with NaN

        Returns
        -------

        """

        # Create cube with bad columns and set the columns to a finite value temporarily
        if np.sum(self.bad_columns) > 0:
            bad = np.empty_like(self.cube, dtype=bool)
            bad[:] = self.bad_columns[np.newaxis, :, :]
            self.cube[bad] = 1
        else:
            bad = None

        # Find maximum and replace with NaN
        pos_max = np.expand_dims(np.nanargmax(self.cube, axis=0), axis=0)
        pos_max_idx = (
            np.arange(self.cube.shape[0]).reshape((len(self), 1, 1)) == pos_max
        )
        self.cube[pos_max_idx] = np.nan

        # In case we replaced bad columns, put back the NaNs
        if bad is not None:
            self.cube[bad] = np.nan

    def _mask_min(self):
        """
        Masks the minimum pixel value in the stack with NaN

        Returns
        -------

        """

        # Create cube with bad columns and set the columns to a finite value temporarily
        if np.sum(self.bad_columns) > 0:
            bad = np.empty_like(self.cube, dtype=bool)
            bad[:] = self.bad_columns[np.newaxis, :, :]
            self.cube[bad] = 1
        else:
            bad = None

        # Find minimum and replace with NaN
        pos_min = np.expand_dims(np.nanargmin(self.cube, axis=0), axis=0)
        pos_min_idx = (
            np.arange(self.cube.shape[0]).reshape((len(self), 1, 1)) == pos_min
        )
        self.cube[pos_min_idx] = np.nan

        # In case we replaced bad columns, put back the NaNs
        if bad is not None:
            self.cube[bad] = np.nan

    def _mask_below(self, value):
        """
        Set all entries in the cube below 'value' to NaN

        Parameters
        ----------
        value : float, int
            Value below which everything is to be masked

        Returns
        -------

        """

        # Mask values
        with np.errstate(invalid="ignore"):
            self.cube[:][self.cube < value] = np.nan

    def _mask_above(self, value):
        """
        Set all entries in the cube above 'value' to NaN

        Parameters
        ----------
        value : float, int
            Value above which everything is to be masked

        Returns
        -------

        """

        # Mask values
        with np.errstate(invalid="ignore"):
            self.cube[:][self > value] = np.nan

    def _mask_badpix(self, bpm):
        """
        Applys a bad pixel mask to self

        Parameters
        ----------
        bpm : ImageCube, optional
            Bad pixel mask as ImageCube instance. Must have same shape as self

        Returns
        -------

        """

        # Shape must match
        if self.shape != bpm.shape:
            raise ValueError("Shapes do not match")

        # Mask bad pixels with NaN
        self.cube[bpm.cube > 0] = np.nan

    def _sigma_clip(self, sigma_level=3, sigma_iter=1, center_metric=np.nanmedian):
        """
        Performs sigma clipping on cube. Replaces all rejected values with NaN.

        Parameters
        ----------
        sigma_level : float, int, optional
            sigma level; default is 3.
        sigma_iter : int, optional
            Number of iterations; default is 1.
        center_metric : callable, optional
            Metric to calculate the center of the data; default is np.nanmedian.

        """

        # Perform sigma clipping along first axis
        self.cube = apply_sigma_clip(
            data=self.cube,
            sigma_level=sigma_level,
            sigma_iter=sigma_iter,
            center_metric=center_metric,
            axis=0,
        )

    def apply_masks(
        self,
        bpm=None,
        sources=None,
        mask_min=False,
        mask_max=False,
        mask_below=None,
        mask_above=None,
        sigma_level=None,
        sigma_iter=1,
        sigma_center_metric=np.nanmedian,
    ):
        """
        Applies the above given masking methods to instance cube.

        Parameters
        ----------
        bpm : ImageCube, optional
            Bad pixel mask as ImageCube instance. Must have same shape as self
        sources: ImageCube, optional
            Similar to the bad pixel mask, a cube that holds a mask where sources are
            located.
        mask_min : bool, optional
            Whether the minimum in the stack should be masked
        mask_max : bool, optional
            Whether the maximum in the stack should be masked
        mask_below : float, int, optional
            Values below are masked in the entire cube
        mask_above : float, int, optional
            Values above are masked in the entire cube
        sigma_level : float, int, optional
            sigma-level in clipping.
        sigma_iter : int, optional
            Iterations of sigma clipping.
        sigma_center_metric : callable, optional
            Metric to calculate the center of the data; default is np.nanmedian.

        """

        # Mask bad pixels
        if bpm is not None:
            self._mask_badpix(bpm=bpm)

        if sources is not None:
            self._mask_badpix(bpm=sources)

        # Mask minimum in cube
        if mask_min:
            self._mask_min()

        # Mask maximum in cube
        if mask_max:
            self._mask_max()

        # Mask below
        if mask_below is not None:
            self._mask_below(value=mask_below)

        # Mask above
        if mask_above is not None:
            self._mask_above(value=mask_above)

        # Sigma clipping
        if (sigma_level is not None) & (sigma_iter > 0):
            self._sigma_clip(
                sigma_level=sigma_level,
                sigma_iter=sigma_iter,
                center_metric=sigma_center_metric,
            )

    def apply_masks_plane(self, sigma_level, sigma_iter):
        """
        Applys masks on per-plane basis.

        Parameters
        ----------
        sigma_level : int, float
            Sigma level.
        sigma_iter : int
            Number of iterations on sigma clipping
        """

        for plane in self:
            plane[:] = apply_sigma_clip(
                data=plane,
                sigma_level=sigma_level,
                sigma_iter=sigma_iter,
                center_metric=np.nanmedian,
            )

    def discard_nan_planes(self, threshold: float = 0.9) -> ArrayLike:
        """
        Removes planes (first axis) where the fraction of NaN pixels is >= threshold.
        Modifies self.cube in place.

        Parameters
        ----------
        threshold : float
            Fraction of NaN pixels above which a plane is discarded (between 0 and 1).

        Returns
        -------
        np.ndarray
            Boolean array indicating which planes were kept (True) and which were
            discarded (False).

        """
        # Calculate the fraction of NaNs in each plane
        nan_fraction = np.isnan(self.cube).sum(axis=(1, 2)) / (
            self.cube.shape[1] * self.cube.shape[2]
        )
        # Keep only planes with NaN fraction less than threshold
        keep = nan_fraction < threshold
        self.cube = self.cube[keep]

        # Raise error if all planes are discarded
        if self.cube.size == 0:
            raise ValueError(
                f"All planes were discarded (NaN fraction >= {threshold})."
            )

        return keep

    # =========================================================================== #
    # Data manipulation
    # =========================================================================== #
    def scale_planes(self, scales: Union[ArrayLike, List]):
        """
        Scales each plane by the given value.

        Parameters
        ----------
        scales : np.ndarray, List
            The scales for each plane. Must match length of cube

        """
        if len(scales) != len(self):
            raise ValueError("Provide scales for each plane!")

        if isinstance(scales, list):
            scales = np.array(scales)

        # Apply scales along first axis
        self.cube *= scales[:, np.newaxis, np.newaxis]

    def extend(self, data):
        """
        Extends the ImageCube instance with a new plane.

        Parameters
        ----------
        data : np.ndarray

        """

        # Get supplied data into correct shape
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        elif len(data.shape) != 3:
            raise ValueError("Supplied data not compatible with ImageCube")

        # If we have an empty instance we assign the data
        if self.cube is None:
            self.cube = data

        # Otherwise we stack on top of the old
        else:
            self.cube = np.vstack((self.cube, data))

    def flatten(self, metric=np.nanmedian, weights=None, axis=0, dtype=None):
        """
        Flattens the ImageCube data to a 2D numpy array based on various options.

        Parameters
        ----------
        metric : callable, optional
            Metric to be used to collapse cube
        axis : int, optional
            axis along which to flatten (usually 0 if the shape of the data is not
            tampered with).
        weights : ImageCube, optional
            Optionally an ImageCube instance containing the weights for a weighted
            average flattening.
        dtype : callable, optional
            Output data type

        Returns
        -------
        np.ndarray
            2D numpy array of flattened cube

        """

        # # In case a weighted average should be calculated (only possible with a
        # masked array)
        # if weights is not None:
        #
        #     # Weights must match input data
        #     if self.shape != weights.shape:
        #         raise ValueError("Weights don't match input")
        #
        #     # Calculate weighted average
        #     flat = np.ma.average(np.ma.masked_invalid(self.cube), axis=axis,
        #     weights=weights)
        #     """:type : np.ma.MaskedArray"""
        #
        #     # Fill NaNs back in and return
        #     with warnings.catch_warnings():
        #         warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        #         return flat.filled(fill_value=np.nan).astype(dtype=dtype, copy=False)

        # In case a weighted average should be calculated (only possible with a masked
        # array)
        if (weights is not None) and (metric == "weighted"):
            # Weights must match input data
            if self.shape != weights.shape:
                raise ValueError("Weights don't match input")

            # Calculate weighted average
            flat = np.ma.average(  # noqa
                np.ma.masked_invalid(self.cube), axis=axis, weights=weights
            )
            """:type : np.ma.MaskedArray"""

            # Fill NaNs back in and return
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
                return flat.filled(fill_value=np.nan).astype(dtype=dtype, copy=False)

        # Just flatten
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            warnings.filterwarnings("ignore", r"Mean of empty slice")
            return metric(self.cube, axis=axis).astype(dtype=dtype, copy=False)

    def normalize(self, norm):
        """
        Normalizes the ImageCube.

        Parameters
        ----------
        norm : int, float, np.ndarray
            Data by which to normalize the cube. In case of an integer, or float, we
            divide everything. In case of a 1D array, we normalize each plane.

        Raises
        -------
        ValueError
            If normalization shape not supported.

        """

        # If we have a float or integer
        if (isinstance(norm, (int, np.integer))) | (
            isinstance(norm, (int, np.floating))
        ):
            self.cube /= norm

        # If we have an array...
        elif isinstance(norm, (np.ndarray, list)):
            if isinstance(norm, list):
                norm = np.asarray(norm)

            # ...with one dimension
            if norm.ndim == 1:
                # Dimensions must match!
                if len(self) != len(norm):
                    raise ValueError("Normalization shape incorrect")

                # Norm
                self.cube /= norm[:, np.newaxis, np.newaxis]

            # ...with more dimensions...
            else:
                # ...the norm-shape must match the cube shape
                if self.shape != norm.shape:
                    raise ValueError("Normalization cube shape incorrect")

                # Norm
                self.cube /= norm

        # If we have something else, raise Error
        else:
            raise ValueError("Normalization not supported")

    def linearize(self, coeff, texptime):
        """
        Linearizes the data cube based on non-linearity coefficients. Will created
        multiple parallel processes (up to 4) for better performance.

        Parameters
        ----------
        coeff : iterable
            If a list of coefficients (floats), the same non-linear inversion will be
            applied to all planes of the cube If a list of lists with coefficients, it
            must have the length of the cube and each plane will be linearized with the
            corresponding coefficients.
        texptime : iterable
            TEXPTIME for each plane in cube.

        Returns
        -------

        """

        # Check if it's list of lists
        if isinstance(coeff[0], list):
            cff = coeff

            # Provided number of coefficient lists must match the cube
            if len(coeff) != len(self):
                raise ValueError("Coefficient list does not match data")

        # If there is just one list of coefficients, we apply it to the entire cube
        else:
            cff = repeat(coeff)

        # If just a single value for the dit is provided
        if isinstance(texptime, (int, float)):
            texptime = repeat(texptime)

        # Only launch Pool if more than one thread is requested
        if self.setup.n_jobs == 1:
            mp = []
            for a, b, c in zip(self.cube, cff, texptime):
                mp.append(
                    linearize_data(
                        data=a, coeff=b, texptime=c, reset_read_overhead=1.0011
                    )
                )

        elif self.setup.n_jobs > 1:
            # Start multithreaded processing of linearization
            with Parallel(
                n_jobs=self.setup.n_jobs, prefer=self.setup.joblib_backend
            ) as parallel:
                mp = parallel(
                    delayed(linearize_data)(a, b, c, d)
                    for a, b, c, d in zip(self.cube, cff, texptime, repeat(1.0011))
                )

        else:
            raise ValueError(
                f"'n_threads' not correctly set (n_threads = {self.setup.n_jobs})"
            )

        # Concatenate results and overwrite cube
        self.cube = np.stack(mp, axis=0)

    def destripe(
        self,
        masks=None,
        smooth=False,
        path_plot=None,
        combine_bad_planes=False,
        force_combine_planes=False,
    ):
        """
        Destripes VIRCAM images

        Parameters
        ----------
        masks : ImageCube
            Cube instance containing masks.
        smooth : bool, optional
            Whether the destriping array should be smoothed with a spline before being
            applied.
        path_plot : str, optional
            If set, path of QC plot.
        combine_bad_planes : bool, optional
            Whether bad planes should be combined.
        force_combine_planes : bool, optional
            Whether bad planes should be combined even if they are not flagged as bad.

        """

        # Set masks to iterable if not set
        if masks is None:
            masks = np.full_like(self.cube, False, dtype=bool)

        # Destripe in parallel
        with Parallel(
            n_jobs=self.setup.n_jobs, prefer=self.setup.joblib_backend
        ) as parallel:
            mp = parallel(
                delayed(destripe_helper)(a, b, c)
                for a, b, c in zip(self.cube, masks, repeat(smooth))
            )

        # Count bad pixels in each plane
        nbad = np.sum(masks, axis=2)

        # Identify rows with more than 50% masked pixels
        bad_rows = nbad > (nbad.shape[1] * 0.5)

        # Count number of bad rows in each plane
        nbad_rows = np.sum(bad_rows, axis=1)

        # Identify bad planes as those that have more than 1/3 bad rows
        bad_planes = nbad_rows > (self.shape[2] * 1 / 3)

        cube_destripe_avg = self.copy()
        if combine_bad_planes:
            # Combine stripes for each detector row
            destripe_01_04 = np.nanmedian(np.stack(mp[0:4]), axis=0)
            destripe_05_08 = np.nanmedian(np.stack(mp[4:8]), axis=0)
            destripe_09_12 = np.nanmedian(np.stack(mp[8:12]), axis=0)
            destripe_13_16 = np.nanmedian(np.stack(mp[12:16]), axis=0)

            # Apply average destriping to separate cube
            cube_destripe_avg.cube[0:4] -= np.expand_dims(destripe_01_04, axis=1)
            cube_destripe_avg.cube[4:8] -= np.expand_dims(destripe_05_08, axis=1)
            cube_destripe_avg.cube[8:12] -= np.expand_dims(destripe_09_12, axis=1)
            cube_destripe_avg.cube[12:16] -= np.expand_dims(destripe_13_16, axis=1)

        # Loop over planes
        # TODO: Replace all bad rows with average destriping instead of full planes?
        for idx, bad in enumerate(bad_planes):
            if force_combine_planes:
                self.cube[idx] = cube_destripe_avg[idx]
            # If bad plane, take average destriping
            elif bad & combine_bad_planes:
                self.cube[idx] = cube_destripe_avg[idx]
            # If good plane, take single destriping
            else:
                self.cube[idx] -= np.expand_dims(mp[idx], axis=1)

        # Plot destripe pattern if requested
        if path_plot is not None:
            fig, ax = plt.subplots(1, 1, figsize=(25, 5))
            kw_plot = dict(lw=0.4, alpha=0.8)
            pcolors = ["crimson", "green", "blue", "black"]
            for pidx, pc in zip(range(0, 16, 4), pcolors):
                for tidx in range(4):
                    ax.plot(mp[pidx + tidx] + 3 * pidx, c=pc, **kw_plot)
                ax.plot(
                    np.nanmedian(np.stack(mp[pidx : pidx + 4]), axis=0) + 3 * pidx,
                    c=pc,
                    lw=1.0,
                )
            ax.set_ylim(-30, 60)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                fig.savefig(path_plot, bbox_inches="tight", dpi=300)
            plt.close("all")

    def interpolate_nan(self):
        """
        Interpolates NaNs for each plane in the cube. Interpolation is (for performance
        reasons) kept very simple, where the original image is convolved with a given
        kernel and the NaNs are then replace with the convolved pixel values.

        """

        # Hardcoded Kernel
        kernel = Gaussian2DKernel(1)

        # Overlap is half the kernel size
        overlap = int(np.ceil(np.max(kernel.shape) / 2))

        # Also the overlap must be even
        # if overlap % 2 != 0:
        #     overlap = np.int(np.ceil(overlap / 2.) * 2)

        # Always chop along the longer axis
        chop_ax = 0 if self.shape[1] > self.shape[2] else 1

        # Loop through planes and interpolate
        for plane in self:
            # Chop in smaller sub-regions for better performance
            chopped, loc = chop_image(
                array=plane,
                npieces=self.setup.n_jobs * 2,
                axis=chop_ax,
                overlap=overlap,
            )

            # Do interpolation
            if self.setup.n_jobs == 1:
                mp = []
                for ch in chopped:
                    mp.append(
                        interpolate_image(
                            data=ch,
                            kernel=kernel,
                            max_bad_neighbors=self.setup.interpolate_max_bad_neighbors,
                        )
                    )

            elif self.setup.n_jobs > 1:
                with Parallel(
                    n_jobs=self.setup.n_jobs, prefer=self.setup.joblib_backend
                ) as parallel:
                    mp = parallel(
                        delayed(interpolate_image)(d, k, m)
                        for d, k, m in zip(
                            chopped,
                            repeat(kernel),
                            repeat(self.setup.interpolate_max_bad_neighbors),
                        )
                    )

            else:
                raise ValueError(
                    f"'n_threads' not correctly set (n_threads = {self.setup.n_jobs})"
                )

            # Merge back into plane and put into cube
            plane[:] = merge_chopped(
                arrays=mp, locations=loc, axis=chop_ax, overlap=overlap
            )

    def replace_nan(self, value):
        """
        Replaces NaNs in the cube with a given finite value.

        Parameters
        ----------
        value : int, float
            Value which will replace NaNs

        """

        # Replace values
        self.cube[np.isnan(self.cube)] = value

    def build_source_masks(self):
        """
        Create a source mask for current cube

        Returns
        -------
        ImageCube
            Source mask.

        """

        # Build mask for each plane in parallel
        with Parallel(
            n_jobs=self.setup.n_jobs, prefer=self.setup.joblib_backend
        ) as parallel:
            mp = parallel(
                delayed(source_mask)(a, b, c, d, e)
                for a, b, c, d, e in zip(
                    self.cube,
                    repeat(self.setup.mask_sources_thresh),
                    repeat(self.setup.mask_sources_min_area),
                    repeat(self.setup.mask_sources_max_area),
                    repeat(self.setup.mask_bright_sources),
                    repeat(self.setup.mask_sources_dilate),
                )
            )

        return ImageCube(cube=np.array(mp).astype(bool), setup=self.setup)

    def build_source_masks_noisechisel(self):
        """
        Runs noisechisel on the cube and returns the masks.

        Returns
        -------
        ImageCube
            Noisechisel masks.

        """

        # Write cube to temp file on disk
        path_temp_data = make_path_system_tempfile(suffix=".fits")

        self.write_mef(
            path=path_temp_data,
            overwrite=True,
            dtype=np.float32,
        )

        # Create temp paths for noisechisel
        paths_temp_noisechisel = [
            path_temp_data.replace(".fits", f"_{i:02d}.fits")
            for i in range(1, len(self) + 1)
        ]

        # Build NoiseChisel commands
        cmds = [
            (
                f"{which(self.setup.bin_noisechisel)} {path_temp_data} "
                f"--hdu {i} "
                f"-o {pto} "
                f"-N 1 "
                f"-q --rawoutput --cleangrowndet "
                f"--qthresh {self.setup.noisechisel_qthresh} "
                f"--erode {self.setup.noisechisel_erode} "
                f"--detgrowquant {self.setup.noisechisel_detgrowquant} "
                f"--tilesize={self.setup.noisechisel_tilesize} "
                f"--meanmedqdiff {self.setup.noisechisel_meanmedqdiff}"
            )
            for i, pto in zip(range(1, len(self) + 1), paths_temp_noisechisel)
        ]

        # Run noisechisel in parallel
        run_commands_shell_parallel(cmds=cmds, silent=False, n_jobs=self.setup.n_jobs)

        # Read results
        masks_noisechisel = np.array(
            [
                fits.getdata(filename=path, ext=1).astype(np.float32)
                for path in paths_temp_noisechisel
            ]
        )

        # Replace NaNs with 1
        masks_noisechisel[np.isnan(masks_noisechisel)] = 1

        # Delete temp files
        os.remove(path_temp_data)
        [os.remove(path) for path in paths_temp_noisechisel]

        # Return
        return ImageCube(cube=masks_noisechisel.astype(bool), setup=self.setup)

    # =========================================================================== #
    # Properties
    # =========================================================================== #
    @property
    def shape(self):
        """
        Returns shape of cube

        Returns
        -------
        iterable
            Shape of numpy array

        """

        return self.cube.shape

    @property
    def bad_columns(self):
        """
        Identifes bad columns in the cube

        Returns
        -------
        np.ndarray

        """

        # Ignore NaN warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            # Identify bad columns
            return ~np.isfinite(np.nanmax(self.cube, axis=0))

    def var(self, axis=None):
        """
        Variance of the cube.

        Parameters
        ----------
        axis : int, tuple, optional
            Axis along which to calculate the standard deviation. Default is None.

        Returns
        -------
        float, np.ndarray
            Variance.

        """

        return np.nanvar(self.cube, axis=axis)

    def mean(self, axis=None):
        """
        Returns the mean of the cube

        Parameters
        ----------
        axis : int, tuple, optional
            Axis along which to calculate the mean. Default is None.

        Returns
        -------
        float, np.ndarray

        """

        return np.nanmean(self.cube, axis=axis)

    def median(self, axis=None):
        """
        Returns the median of the cube

        Parameters
        ----------
        axis : int, tuple, optional
            Axis along which to calculate the median. Default is None.

        Returns
        -------
        float, np.ndarray

        """

        return np.nanmedian(self.cube, axis=axis)

    def mad(self, axis=None):
        """
        Median absolute deviation

        Parameters
        ----------
        axis : int, tuple, optional
            Axis along which to calculate the standard deviation. Default is None.

        Returns
        -------
        float, np.ndarray
            Median absolute deviation along specified axes.

        """

        # If no axes are specified, just return with the standard formula.
        if axis is None:
            return np.nanmedian(np.abs(self.cube - np.nanmedian(self.cube)))

        # Otherwise we need to expand the dimensions
        else:
            med = np.nanmedian(self.cube, axis)

            if isinstance(axis, tuple):
                for a in axis:
                    med = np.expand_dims(med, axis=a)

            elif isinstance(axis, int):
                med = np.expand_dims(med, axis=axis)

            else:
                raise ValueError("Supplied axis format incorrect")

            # Return
            return np.nanmedian(np.abs(self.cube - med), axis)

    def background_planes(self):
        """
        Calculates sky level and noise estimates for each plane in the cube.

        Returns
        -------
        ndarray, ndarray

        """

        # Calculate the sky values for each plane in the cube
        if self.setup.n_jobs == 1:
            back, back_sig = list(zip(*[mmm(sky_vector=c) for c in self.cube]))[:2]
        else:
            with Parallel(
                n_jobs=self.setup.n_jobs, prefer=self.setup.joblib_backend
            ) as parallel:
                mp = parallel(
                    delayed(mmm)(a, b, c, d, e, f, g)
                    for a, b, c, d, e, f, g in zip(
                        self.cube,
                        repeat(False),
                        repeat(False),
                        repeat(False),
                        repeat(500),
                        repeat(20),
                        repeat(True),
                    )
                )
            # Unpack
            back, back_sig = list(zip(*mp))[:2]

        # Return
        return np.asarray(back), np.asarray(back_sig)

    def background(self, mesh_size, mesh_filtersize):
        """
        Creates background and noise cubes.

        Parameters
        ----------
        mesh_size : int
            Requested mesh size in pixels.
        mesh_filtersize : int
            2D median filter size for meshes.

        Returns
        -------
        ImageCube, ImageCube
            Tuple of ImageCubes (background, noise)

        """
        # Submit parallel jobs for background estimation in each cube plane
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with Parallel(
                n_jobs=self.setup.n_jobs, prefer=self.setup.joblib_backend
            ) as parallel:
                mp = parallel(
                    delayed(background_image)(a, b, c)
                    for a, b, c in zip(
                        self.cube, repeat(mesh_size), repeat(mesh_filtersize)
                    )
                )

        # Unpack result, put into ImageCubes, and return
        bg, bg_std = list(zip(*mp))
        return ImageCube(cube=np.array(bg), setup=self.setup), ImageCube(
            cube=np.array(bg_std), setup=self.setup
        )
