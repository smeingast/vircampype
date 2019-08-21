# =========================================================================== #
# Import
import warnings
import numpy as np

from astropy.io import fits
from vircampype.utils.math import sigma_clip


class ImageCube(object):

    def __init__(self, cube=None):
        """
        Parameters
        ----------
        cube : np.ndarray, optional

        """

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
            return ImageCube(self.cube + other.cube)
        elif isinstance(other, np.ndarray):
            return ImageCube(self.cube + other)
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
        elif isinstance(other, np.ndarray):
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
            return ImageCube(self.cube - other.cube)
        elif isinstance(other, np.ndarray):
            return ImageCube(self.cube - other)
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
        elif isinstance(other, np.ndarray):
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
            return ImageCube(self.cube * other.cube)
        elif isinstance(other, np.ndarray):
            return ImageCube(self.cube * other)
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
        elif isinstance(other, np.ndarray):
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
            return ImageCube(self.cube / other.cube)
        elif isinstance(other, np.ndarray):
            return ImageCube(self.cube / other)
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
        elif isinstance(other, np.ndarray):
            self.cube /= other
            return self
        else:
            raise TypeError("Division for {0:s} not implemented".format(str(type(other))))

    # =========================================================================== #
    # I/O
    # =========================================================================== #
    def write_mef(self, path, prime_header, data_headers, overwrite=True):
        """
        Write MEF Fits file to disk

        Parameters
        ----------
        path : str
            Output file path
        overwrite : bool
            Whether to overwrite already existing files
        prime_header : fits.Header
            Primary header.
        data_headers : sized
            Data headers.

        """

        # Dummy check
        if len(self) != len(data_headers):
            raise ValueError("Supplied headers are not compatible with data format")

        # Make HDUList from data and headers
        hdulist = fits.HDUList(hdus=[fits.ImageHDU(data=d, header=h) for d, h in zip(self.cube[:], data_headers)])

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
            bad = np.empty_like(self.cube, dtype=np.bool)
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
            bad = np.empty_like(self.cube, dtype=np.bool)
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

    def _kappa_sigma(self, kappa=3, ikappa=1, center_metric=np.nanmedian):
        """
        Performs Kappa-sigma clipping on cube. Replaces all rejected values with NaN.

        Parameters
        ----------
        kappa : float, int, optional
            kappa-factor (kappa * sigma clipping); default is 3.
        ikappa : int, optional
            Number of iterations; default is 1.
        center_metric : callable, optional
            Metric to calculate the center of the data; default is np.nanmedian.

        """

        # Perform sigma clipping along first axis
        self.cube = sigma_clip(data=self.cube, kappa=kappa, ikappa=ikappa, center_metric=center_metric, axis=0)

    def apply_masks(self, bpm=None, mask_min=False, mask_max=False, mask_below=None, mask_above=None,
                    kappa=None, ikappa=1):
        """
        Applies the above given masking methods to instance cube.

        Parameters
        ----------
        bpm : ImageCube, optional
            Bad pixel mask as ImageCube instance. Must have same shape as self
        mask_min : bool, optional
            Whether the minimum in the stack should be masked
        mask_max : bool, optional
            Whether the maximum in the stack should be masked
        mask_below : float, int, optional
            Values below are masked in the entire cube
        mask_above : float, int, optional
            Values above are masked in the entire cube
        kappa : float, int, optional
            kappa-factor in kappa-sigma clipping.
        ikappa : int, optional
            Iterations of kappa-sigma clipping.

        """

        # Mask bad pixels
        if bpm is not None:
            self._mask_badpix(bpm=bpm)

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
        if kappa:
            self._kappa_sigma(kappa=kappa, ikappa=ikappa, center_metric=np.nanmedian)

    # =========================================================================== #
    # Data manipulation
    # =========================================================================== #
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

    def flatten(self, metric=np.nanmedian, axis=0, dtype=None):
        """
        Flattens the ImageCube data to a 2D numpy array based on various options.

        Parameters
        ----------
        metric : callable, optional
            Metric to be used to collapse cube
        axis : int, optional
            axis along which to flatten (usually 0 if the shape of the data is not tampered with)
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
        # noinspection PyUnresolvedReferences
        if (isinstance(norm, (int, np.int))) | (isinstance(norm, (int, np.float))):
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

    def calibrate(self, dark=None, norm_before=None, norm_after=None, mask=None):
        """
        Applies calibration steps to the ImageCube.
        (0) normalization
        (1) Dark
        # (2) Linearity
        (3) flat-field
        (4) normalization

        Parameters
        ----------
        dark : ImageCube, optional
            The dark cube that should be subtracted.
        norm_before : int, float, np.ndarray, optional
            The normalization data applied before processing.
        norm_after : int, float, np.ndarray, optional
            The normalization data applied after processing.
        mask : ImageCube, optional
            Cube containing the mask.

        """

        # Normalize
        if norm_before is not None:
            self.normalize(norm=norm_before)

        # Subtract dark
        if dark is not None:

            # Shape must match
            if self.shape != dark.shape:
                raise ValueError("Shapes do not match")

            # Subtract dark
            self.cube -= dark.cube

        # Normalize
        if norm_after is not None:
            self.normalize(norm=norm_after)

        if mask:
            self.apply_masks(bpm=mask)

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
