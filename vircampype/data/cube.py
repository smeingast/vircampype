# =========================================================================== #
# Import
import warnings
import numpy as np

from scipy import ndimage
from astropy.io import fits
from itertools import repeat
from joblib import Parallel, delayed
from vircampype.tools.mathtools import *
from vircampype.pipeline.setup import Setup
from astropy.convolution import Gaussian2DKernel


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

    # =========================================================================== #
    # Magic methods
    # =========================================================================== #
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
        Implements addition for ImageData. Here the cubes are added together and a new ImageCube instance is returned

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
            raise TypeError("Addition for {0:s} not implemented".format(str(type(other))))

    def __iadd__(self, other):
        """
        Implements addition with assignment for ImageData. Both ImageCube and ndarrays are supported as input.

        Parameters
        ----------
        other : ImageCube, np.ndarray
            Second ImageData instance or ndarray with same shape from which the data is to be added

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
            raise TypeError("Addition for {0:s} not implemented".format(str(type(other))))

    def __sub__(self, other):
        """
        Implements subtraction for ImageData. Here the cubes are subtracted and a new ImageCube instance is returned.

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
            raise TypeError("Subtraction for {0:s} not implemented".format(str(type(other))))

    def __isub__(self, other):
        """
        Implements subtraction with assignment for ImageData. Both ImageCube and ndarrays are supported as input.

        Parameters
        ----------
        other : ImageCube, np.ndarray
            Second ImageData instance or ndarray with same shape which is to be subtracted

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
            raise TypeError("Subtraction for {0:s} not implemented".format(str(type(other))))

    def __mul__(self, other):
        """
        Implements multiplication for ImageData. Here the cubes are multiplied and a new ImageCube instance is returned.

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
            raise TypeError("Multiplication for {0:s} not implemented".format(str(type(other))))

    def __imul__(self, other):
        """
        Implements multiplication with assignment for ImageData. Both ImageCube and ndarrays are supported as input.

        Parameters
        ----------
        other : ImageCube, np.ndarray
            Second ImageData instance or ndarray with same shape which is to be multiplied with self

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
            raise TypeError("Multiplication for {0:s} not implemented".format(str(type(other))))

    def __truediv__(self, other):
        """
        Implements division for ImageData. Here the cubes are divided and a new ImageCube instance is returned.

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
            raise TypeError("Division for {0:s} not implemented".format(str(type(other))))

    def __itruediv__(self, other):
        """
        Implements division with assignment for ImageData. Both ImageCube and ndarrays are supported as input.

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
            raise TypeError("Division for {0:s} not implemented".format(str(type(other))))

    # =========================================================================== #
    # I/O
    # =========================================================================== #
    def write_mef(self, path, prime_header=None, data_headers=None, overwrite=False, dtype=None):
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
        hdulist = fits.HDUList(hdus=[fits.ImageHDU(data=d.astype(dtype), header=h)
                                     for d, h in zip(self.cube[:], data_headers)])

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
        pos_max_idx = np.arange(self.cube.shape[0]).reshape((len(self), 1, 1)) == pos_max
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
        pos_min_idx = np.arange(self.cube.shape[0]).reshape((len(self), 1, 1)) == pos_min
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
        self.cube = sigma_clip(data=self.cube, sigma_level=sigma_level, sigma_iter=sigma_iter,
                               center_metric=center_metric, axis=0)

    def apply_masks(self, bpm=None, sources=None, mask_min=False, mask_max=False, mask_below=None,
                    mask_above=None, sigma_level=None, sigma_iter=1):
        """
        Applies the above given masking methods to instance cube.

        Parameters
        ----------
        bpm : ImageCube, optional
            Bad pixel mask as ImageCube instance. Must have same shape as self
        sources: ImageCube, optional
            Similar to the bad pixel mask, a cube that holds a mask where sources are located.
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
            self._sigma_clip(sigma_level=sigma_level, sigma_iter=sigma_iter, center_metric=np.nanmedian)

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
            plane[:] = sigma_clip(data=plane, sigma_level=sigma_level,
                                  sigma_iter=sigma_iter, center_metric=np.nanmedian)

    def mask_cosmics(self, gain, rdnoise, return_mask=False):
        """
        Mask cosmics (and hot pixels) based on the L.A.Cosmic algorithm using astroscrappy.

        Parameters
        ----------
        gain : list, iterable
            List of gains
        rdnoise : list, iterable
            List of read noise values.
        return_mask : bool, optional
            Whether the mask should be returned. Default is False.

        """

        # Import
        from astroscrappy.astroscrappy import detect_cosmics

        # Run in parallel
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered")
            with Parallel(n_jobs=self.setup.n_jobs) as parallel:
                mp = parallel(delayed(detect_cosmics)(a, b, c, d, e, f, g)
                              for a, b, c, d, e, f, g in
                              zip(self.cube, repeat(None), repeat(4.5), repeat(0.3), repeat(5.0), gain, rdnoise))

            # Unpack result
            mask, clean = list(zip(*mp))

        # If mask should be returned
        if return_mask:
            return ImageCube(setup=self.setup, cube=np.array(mask, dtype=bool))

        # Otherwise overwrite self
        self.cube = np.array(clean)

    # =========================================================================== #
    # Data manipulation
    # =========================================================================== #
    def scale_planes(self, scales):
        """
        Scales each plane by the given value.

        Parameters
        ----------
        scales : np.ndarray
            The scales for each plane. Must match length of cube

        """
        if len(scales) != len(self):
            raise ValueError("Provide scales for each plane!")

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
            axis along which to flatten (usually 0 if the shape of the data is not tampered with)
        weights : ImageCube, optional
            Optionally an ImageCube instance containing the weights for a weighted average flattening
        dtype : callable, optional
            Output data type

        Returns
        -------
        np.ndarray
            2D numpy array of flattened cube

        """

        # # In case a weighted average should be calculated (only possible with a masked array)
        # if weights is not None:
        #
        #     # Weights must match input data
        #     if self.shape != weights.shape:
        #         raise ValueError("Weights don't match input")
        #
        #     # Calculate weighted average
        #     flat = np.ma.average(np.ma.masked_invalid(self.cube), axis=axis, weights=weights)
        #     """:type : np.ma.MaskedArray"""
        #
        #     # Fill NaNs back in and return
        #     with warnings.catch_warnings():
        #         warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        #         return flat.filled(fill_value=np.nan).astype(dtype=dtype, copy=False)

        # In case a weighted average should be calculated (only possible with a masked array)
        if (weights is not None) and (metric == "weighted"):

            # Weights must match input data
            if self.shape != weights.shape:
                raise ValueError("Weights don't match input")

            # Calculate weighted average
            flat = np.ma.average(np.ma.masked_invalid(self.cube), axis=axis, weights=weights)
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
            Data by which to normalize the cube. In case of an integer, or float, we divide everything. In case of a
            1D array, we normalize each plane.

        Raises
        -------
        ValueError
            If normalization shape not supported.

        """

        # If we have a float or integer
        if (isinstance(norm, (int, np.integer))) | (isinstance(norm, (int, np.floating))):
            self.cube /= norm

        # If we have an array...
        elif isinstance(norm, np.ndarray):

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

    def linearize(self, coeff):
        """
        Linearizes the data cube based on non-linearity coefficients. Will created multiple parallel processes (up to 4)
        for better performance.

        Parameters
        ----------
        coeff : iterable
            If a list of coefficients (floats), the same non-linear inversion will be applied to all planes of the cube
            If a list of lists with coefficients, it must have the length of the cube and each plane will be linearized
            with the corresponding coefficients.

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

        # Only launch Pool if more than one thread is requested
        if self.setup.n_jobs == 1:
            mp = []
            for p, c in zip(self.cube, cff):
                mp.append(linearize_data(data=p, coeff=c))

        elif self.setup.n_jobs > 1:
            # Start multithreaded processing of linearization
            with Parallel(n_jobs=self.setup.n_jobs) as parallel:
                mp = parallel(delayed(linearize_data)(d, c) for d, c in zip(self.cube, cff))

        else:
            raise ValueError("'n_threads' not correctly set (n_threads = {0})"
                             .format(self.setup.n_jobs))

        # Concatenate results and overwrite cube
        self.cube = np.stack(mp, axis=0)

    def destripe(self):
        """ Destripes the cube along a given axis (e.g. for VIRCAM axis=2) """

        # Apply metric along given axis
        self.cube = apply_along_axes(self.cube, method=self.setup.destripe_metric, axis=self.setup.destripe_axis,
                                     norm=True, copy=False)

    def process_raw(self, dark=None, flat=None, linearize=None, sky=None, norm_before=None, norm_after=None):
        """
        Applies calibration steps to the ImageCube.
        (0) normalization
        (1) Dark
        (2) Linearity
        (3) flat-field
        (4) normalization

        Parameters
        ----------
        dark : ImageCube, optional
            The dark cube that should be subtracted.
        flat : ImageCube, optional
            The flat cube by which the data should be divided.
        linearize : iterable, optional
            The linearity coefficients when the cube should be linearized.
        sky : ImageCube, optional
            Sky data when a background correction should be applied.
        norm_before : int, float, np.ndarray, optional
            The normalization data applied before processing.
        norm_after : int, float, np.ndarray, optional
            The normalization data applied after processing.

        """

        # Normalize
        if norm_before is not None:
            self.normalize(norm=norm_before)

        # Subtract dark
        if dark is not None:

            # Shape must match
            if self.shape != dark.shape:
                raise ValueError("Shapes do not match")

            # Normalize dark also to NDIT=1
            if norm_before is not None:
                dark.normalize(norm=norm_before)
                """ This needs to be recoded if at some point DIT scaling is implemented for darks. """

            # Subtract dark
            self.cube -= dark.cube

        # Linearize cube
        if linearize:
            self.linearize(coeff=linearize)

        # Apply flat-field
        if flat is not None:

            # Shape must match
            if self.shape != flat.shape:
                raise ValueError("Shapes do not match")

            # Apply flat
            self.cube /= flat.cube

        # Apply background correction
        if sky is not None:

            # Shape must match
            if self.shape != flat.shape:
                raise ValueError("Shapes do not match")

            # Subtract background
            self.cube -= sky.cube

        # Normalize
        if norm_after is not None:
            self.normalize(norm=norm_after)

    def interpolate_nan(self):
        """
        Interpolates NaNs for each plane in the cube. Interpolation is (for performance reasons) kept very simple,
        where the original image is convolved with a given kernel and the NaNs are then replace with the convolved
        pixel values.

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
            chopped, loc = chop_image(array=plane, npieces=self.setup.n_jobs * 2,
                                      axis=chop_ax, overlap=overlap)

            # Do interpolation
            if self.setup.n_jobs == 1:
                mp = []
                for ch in chopped:
                    mp.append(interpolate_image(data=ch, kernel=kernel,
                                                max_bad_neighbors=self.setup.interpolate_max_bad_neighbors))

            elif self.setup.n_jobs > 1:
                with Parallel(n_jobs=self.setup.n_jobs) as parallel:
                    mp = parallel(delayed(interpolate_image)(d, k, m) for d, k, m in
                                  zip(chopped, repeat(kernel), repeat(self.setup.interpolate_max_bad_neighbors)))

            else:
                raise ValueError(
                    "'n_threads' not correctly set (n_threads = {0})".format(self.setup.n_jobs))

            # Merge back into plane and put into cube
            plane[:] = merge_chopped(arrays=mp, locations=loc, axis=chop_ax, overlap=overlap)

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

        # Get background and noise Cubes with a relatively large mesh size
        cube_bg, cube_bg_std = self.background(mesh_size=128, mesh_filtersize=3)

        # Make the threshold cube (to avoid an editor warning I use np.add here)
        thresh_cube = np.add(cube_bg.cube, self.setup.mask_sources_thresh * cube_bg_std.cube)

        # Make empty new cube if only labels should be returned
        cube_labels = np.full_like(self.cube, dtype=np.uint8, fill_value=0)

        # Loop over cube planes and
        for idx in range(len(self)):

            # Resize threshold map to image size and get pixels above threshold
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in greater")
                sources = self.cube[idx] > thresh_cube[idx]

            # Label regions
            labels, n_labels = ndimage.measurements.label(input=sources, structure=np.ones(shape=(3, 3)))
            assert isinstance(labels, np.ndarray)

            # If there are no sources, continue
            if n_labels < 1:
                continue

            # Measure region sizes
            sizes = ndimage.measurements.sum(input=sources, labels=labels, index=range(1, n_labels + 1))

            # Find those sources outside the given thresholds and set to 0
            # bad_sources = (sizes < minarea) | (sizes > maxarea) if maxarea is not None else sizes < minarea
            bad_sources = (sizes < self.setup.mask_sources_min_area) | (sizes > self.setup.mask_sources_max_area) \
                if self.setup.mask_sources_max_area is not None else sizes < self.setup.mask_sources_min_area
            assert isinstance(bad_sources, np.ndarray)

            # Only if there are bad sources
            if np.sum(bad_sources) > 0:
                labels[bad_sources[labels-1]] = 0  # labels starts with 1

            # Set background to 0, sources to 1 and convert to 8bit unsigned integer
            labels[labels > 0], labels[labels < 0] = 1, 0
            labels = labels.astype(bool)

            # Dilate the mask
            labels = ndimage.binary_closing(labels, iterations=3)

            # Apply mask
            cube_labels[:][idx][labels] = 1

            # Alternatively apply mask
            # self.cube[:][idx][labels] = np.nan

        # Return labels
        return ImageCube(cube=cube_labels, setup=self.setup)

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
        return estimate_background(array=self.cube[:], axis=(1, 2))

    def background(self, mesh_size=None, mesh_filtersize=None):
        """
        Creates background and noise cubes.

        Parameters
        ----------
        mesh_size : int, optional
            Requested mesh size in pixels. Defaults to value in setup.
        mesh_filtersize : int, optional
            2D median filter size for meshes. Defaults to value in setup.

        Returns
        -------
        ImageCube, ImageCube
            Tuple of ImageCubes (background, noise)

        """
        # Set defaults if not specified otherwise
        if mesh_size is None:
            mesh_size = self.setup.background_mesh_size
        if mesh_filtersize is None:
            mesh_filtersize = self.setup.background_mesh_filtersize

        # Submit parallel jobs for background estimation in each cube plane
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with Parallel(n_jobs=self.setup.n_jobs, prefer="threads") as parallel:
                mp = parallel(delayed(background_image)(a, b, c) for a, b, c
                              in zip(self.cube, repeat(mesh_size), repeat(mesh_filtersize)))

        # Unpack result, put into ImageCubes, and return
        bg, bg_std = list(zip(*mp))
        return ImageCube(cube=np.array(bg), setup=self.setup), ImageCube(cube=np.array(bg_std), setup=self.setup)
