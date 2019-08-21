# =========================================================================== #
# Import
import glob
import warnings
import numpy as np

from astropy.io import fits
from vircampype.data.cube import ImageCube
from vircampype.fits.common import FitsFiles


class FitsImages(FitsFiles):

    def __init__(self, file_paths=None):
        """
        Class for Fits images based on FitsFiles. Contains specific methods and functions applicable only to images

        Parameters
        ----------
        file_paths : iterable
            List of input file paths pointing to the Fits images

        Returns
        -------

        """

        super(FitsImages, self).__init__(file_paths=file_paths)

    _dit = None

    @property
    def dit(self):
        """
        Property to get all DITs of the files.

        Returns
        -------
        iterable
            List with DITs for all files

        """

        # Check if already determined
        if self._dit is not None:
            return self._dit

        self._dit = self.primeheaders_get_keys(keywords=[self.setup["keywords"]["dit"]])[0]
        return self._dit

    # =========================================================================== #
    # Data properties
    # =========================================================================== #
    _ndit = None

    @property
    def ndit(self):
        """
        Property to get all NDITs of the files. If not found, automatically set to 1 for each file

        Returns
        -------
        iterable
            List with NDITs for all files. If not found, automatically set to 1.

        """

        # Check if already determined
        if self._ndit is not None:
            return self._ndit

        # If available, read it, else set 1 for all files
        try:
            self._ndit = self.primeheaders_get_keys(keywords=[self.setup["keywords"]["dit"]])[0]
        except KeyError:
            self._ndit = [1] * self.n_files

        return self._ndit

    @property
    def ndit_norm(self):
        """
        Convenience method for retrieving the NDITs of the current instance as ndarray.

        Returns
        -------

        """

        return np.array(self.ndit)

    _filter = None

    @property
    def filter(self):
        """
        Property to return the filter entries from the main header

        Returns
        -------
        iterable
            List with filters for all files.

        """

        # Check if already determined
        if self._filter is not None:
            return self._filter

        self._filter = self.primeheaders_get_keys(keywords=[self.setup["keywords"]["filter"]])[0]
        return self._filter

    _dtypes = None

    @property
    def dtypes(self):
        """
        Gets the data type info from the fits headers. For each file and each extension the data type is extracted.

        Returns
        -------
        iterable
            List of lists of data types for each FitsImage entry and each extension

        """

        # Get data types from fits info
        # dtypes = [[info[idx][6] for idx in range(it)] for info, it in zip(self.fitsinfo, self.n_hdu)]
        # Get bitpix keyword
        bitpix = self.primeheaders_get_keys(keywords=["BITPIX"])[0]

        # Loop through everything and set types
        dtypes = []
        for bp in bitpix:
            if bp == 8:
                app = np.uint8
            elif bp == 16:
                app = np.uint16
            # elif "float32" in dtype:
            #     dtypes[tidx][bidx] = np.float32
            else:
                raise NotImplementedError("Data type '{0}' not implemented!".format(bp))

            dtypes.append(app)

        self._dtypes = dtypes
        return self._dtypes

    # =========================================================================== #
    # I/O
    # =========================================================================== #
    def hdu2cube(self, hdu_index=0, dtype=None):
        """
        Reads a given extension from all files in the given instance into a numpy array

        Parameters
        ----------
        hdu_index : int, optional
            Iterable of hdu indices to load, default is to load all HDUs with data
        dtype : dtype, optional
            Data type (e.g. np.float32)

        Returns
        -------
        ImageCube
            ImageCube instance containing data for a given extension for all files

        """

        # The specified extension must be in the range of existing ones
        if hdu_index > min(self.n_hdu) - 1:
            raise ValueError("The specified extension is out of range!")

        # Data type
        if dtype is None:
            dtype = np.float32

        # Create empty numpy cube
        cube = np.empty((self.n_files, self.setup["data"]["dim_y"], self.setup["data"]["dim_x"]), dtype=dtype)

        # Fill cube with data
        for path, plane in zip(self.full_paths, cube):
            with fits.open(path) as f:
                plane[:] = f[hdu_index].data

        # Return
        return ImageCube(cube=cube)

    def file2cube(self, file_index=0, hdu_index=None, dtype=None):
        """
        Reads all extensions of a given file into a numpy array.

        Parameters
        ----------
        file_index : int
            Integer index of the file in the current FitsFiles instance.
        hdu_index : iterable
            Iterable of hdu indices to load, default is to load all HDUs with data.
        dtype : optional, dtype
            Data type (e.g. np.float32).

        Returns
        -------
        ImageCube
            ImageCube instance containing data of the requested file.

        """

        if hdu_index is None:
            hdu_index = self.data_hdu[file_index]

        # Data type
        if dtype is None:
            dtype = set([self.dtypes[file_index][idx] for idx in hdu_index])
            if len(dtype) > 1:
                warnings.warn("Mixed data types in file", Warning)
            dtype = list(dtype)[0]

        # Create empty cube
        cube = np.empty((len(hdu_index), self.setup["data"]["dim_y"], self.setup["data"]["dim_x"]), dtype=dtype)

        # Fill cube with data
        with fits.open(name=self[file_index]) as f:
            for plane, idx in zip(cube, hdu_index):
                plane[:] = f[idx].data

        # Return
        return ImageCube(cube=cube)

    # =========================================================================== #
    # Splitter
    # =========================================================================== #
    def _split_filter(self):
        """
        Splits self files based on unique filter entries in the FITS headers.

        Returns
        -------
        iterable
            List if FitsImages.

        """

        # Filter keyword must be present!
        return self.split_keywords(keywords=[self.setup["keywords"]["filter"]])

    def _split_expsequence(self):
        """
        Splits input files based on unique exposure sequences (DIT and NDIT) entries in the FITS headers.

        Returns
        -------
        ImageList
            ImageList instance with split FitsImages entries based on unique exposure sequences (DIT and NDIT).

        """

        # When the keyword is present, we can just use the standard method
        """ Removing this break compatibility with non-ESO data. """
        # try:
        return self.split_keywords(keywords=[self.setup["keywords"]["dit"], self.setup["keywords"]["ndit"]])

        # Otherwise, we set NDIT to 1
        # except KeyError:
        #
        #     # Construct list of tuples for DIT and NDIT
        #     tup = [(i, k) for i, k in zip(self.dit, self.ndit)]
        #
        #     # Find unique entries
        #     utup = set(tup)
        #
        #     # Get the split indices
        #     split_indices = [[i for i, j in enumerate(tup) if j == k] for k in utup]
        #
        #     split_list = []
        #     for s_idx in split_indices:
        #         split_list.append(self.__class__([self.file_paths[idx] for idx in s_idx]))
        #
        #     return split_list

    # =========================================================================== #
    # Master images
    # =========================================================================== #
    def _get_masterimages(self):
        """
        Gets all MasterImages for current instance

        Returns
        -------
        MasterImages
            MasterImages instance with all master fits files in the mastercalibration directory

        Raises
        ------
        ValueError
            If no files are found.

        """

        # Get paths in the master calibration directory
        paths = glob.glob(self.path_master + "*.fits")

        # If there is nothing, issue error
        if len(paths) < 1:
            raise ValueError("No master calibration images found!")

        return MasterImages(file_paths=paths)

    # TODO: Rename this to get_*
    def match_masterbpm(self):
        """
        Get for each file in self the corresponding MasterBadPixelMask.

        Returns
        -------
        MasterBadPixelMask
            MasterBadPixelMask instance holding for each file in self the corresponding MasterBadPixelMask file.

        """

        # Match and return
        return self.match_mjd(match_to=self._get_masterimages().bpm, max_lag=self.setup["master"]["max_lag"])

    def match_masterdark(self):
        """
        Get for each file in self the corresponding MasterDark.

        Returns
        -------
        MasterDark
            MasterDark instance holding for each file in self the corresponding MasterDark file.

        """

        # Fetch all master darks
        mdarks = self._get_masterimages().dark

        # Match DIT and NDIT and MJD
        return self._match_exposure(match_to=mdarks, max_lag=self.setup["master"]["max_lag"])

    # =========================================================================== #
    # Matcher
    # =========================================================================== #
    # noinspection PyTypeChecker
    def _match_exposure(self, match_to, max_lag=None, ignore_dit=False, ignore_ndit=False):
        """
        Matches all entries in the current instance with the match_to instance so that DIT and NDIT fit. In case there
        are multiple matches, will return the closest in time!

        Parameters
        ----------
        match_to : FitsImages
            FitsImages (or any child) instance out of which the matches should be drawn.
        max_lag : int, float, optional
            Maximum allowed time difference for matching in days. Default is None.
        ignore_dit : bool, optional
            Whether to ignore DIT values in matching. Default is False.
        ignore_ndit: bool, optional
            Whether to ignore NDIT values in matching. Default is False.

        """

        # Check if input is indeed of FitsImages class
        if not isinstance(match_to, FitsImages):
            raise ValueError("Input objects are not FitsImages class")

        # Fetch DIT and NDIT information for filtering options
        dit_a = [1 for _ in self.dit] if ignore_dit else self.dit
        ndit_a = [1 for _ in self.ndit] if ignore_ndit else self.ndit
        dit_b = [1 for _ in match_to.dit] if ignore_dit else match_to.dit
        ndit_b = [1 for _ in match_to.ndit] if ignore_ndit else match_to.ndit

        # Construct list of tuples for easier matching
        pair_a = [(i, k) for i, k in zip(dit_a, ndit_a)]
        pair_b = [(i, k) for i, k in zip(dit_b, ndit_b)]

        # Get matching indices (for each entry in pair_a get the indices in pair_b)
        indices = [[i for i, j in enumerate(pair_b) if j == k] for k in pair_a]

        # Create list for output
        matched = []

        # Now get the closest in time for each entry
        for f, idx in zip(self, indices):

            # Construct FitsFiles class
            a = self.__class__([f])
            b = match_to.__class__([match_to.file_paths[i] for i in idx])

            # Raise error if nothing is found
            if len(a) < 1 or len(b) < 1:
                raise ValueError("No matching exposure found.")

            # Get the closest in time
            matched.extend(a.match_mjd(match_to=b, max_lag=max_lag).full_paths)

        # Return
        return match_to.__class__(matched)

    # =========================================================================== #
    # Other methods
    # =========================================================================== #
    def check_compatibility(self, n_files_min=None, n_files_max=None, n_hdu_min=None, n_hdu_max=None, n_dit_min=None,
                            n_dit_max=None, n_ndit_min=None, n_ndit_max=None, n_filter_min=None, n_filter_max=None):
        """
        Applies various checks for sequence compatibility. For example one can set the minimum number of available
        files or the maximum number of different filters in a sequence.

        Parameters
        ----------
        n_files_min : int, optional
            Minimum number of files which must be available.
        n_files_max : int, optional
            Maximum number of files which must be available.
        n_hdu_min : int, optional
            Minimum different HDU counts in sequence.
        n_hdu_max : int, optional
            Maximum different HDU counts in sequence.
        n_dit_min : int, optional
            Minimum number of DITs in sequence.
        n_dit_max : int, optional
            Maximum number of DITs in sequence.
        n_ndit_min : int, optional
            Minimum number of NDITs in sequence.
        n_ndit_max : int, optional
            Maximum number of NDITs in sequence.
        n_filter_min : int, optional
            Minimum number of filters in sequence.
        n_filter_max : int, optional
            Minimum number of filters in sequence.

        Raises
        ------
        ValueError
            If any of the conditions is not met.

        """

        # Check number of available files
        if n_files_min is not None:
            if len(self) < n_files_min:
                raise ValueError("Less than {0:0g} files available".format(n_files_min))

        if n_files_max is not None:
            if len(self) > n_files_max:
                raise ValueError("More than {0:0g} files available".format(n_files_max))

        # Check number of HDUs
        if n_hdu_min is not None:
            if len(set(self.n_hdu)) < n_hdu_min:
                raise ValueError("File extensions not matching!")

        if n_hdu_max is not None:
            if len(set(self.n_hdu)) > n_hdu_max:
                raise ValueError("File extensions not matching!")

        # Check DITs
        if n_dit_min is not None:
            if len(set(self.dit)) < n_dit_min:
                raise ValueError("Found {0:0g} different DITs; min = {1:0g}".format(len(set(self.dit)), n_dit_min))

        if n_dit_max is not None:
            if len(set(self.dit)) > n_dit_max:
                raise ValueError("Found {0:0g} different DITs; max = {1:0g}".format(len(set(self.dit)), n_dit_max))

        # Check NDITs
        if n_ndit_min is not None:
            if len(set(self.ndit)) < n_ndit_min:
                raise ValueError("Found {0:0g} different NDITs; min = {1:0g}".format(len(set(self.dit)), n_ndit_min))

        if n_ndit_max is not None:
            if len(set(self.ndit)) > n_ndit_max:
                raise ValueError("Found {0:0g} different NDITs; max = {1:0g}".format(len(set(self.dit)), n_ndit_max))

        # Check filters
        if n_filter_min is not None:
            if len(set(self.filter)) < n_filter_min:
                raise ValueError("Found {0:0g} different filters; "
                                 "min = {1:0g}".format(len(set(self.filter)), n_filter_min))
        if n_filter_max is not None:
            if len(set(self.filter)) > n_filter_max:
                raise ValueError("Found {0:0g} different filters; "
                                 "max = {1:0g}".format(len(set(self.filter)), n_filter_min))

    def create_masterpath(self, basename, idx=0, dit=False, ndit=False, mjd=False, filt=False, table=False):
        """
        Build the path for master calibration files based on information in the FITS header

        Parameters
        ----------
        basename : str
            The generated filename will start with this string
        idx : int
            Index of entry in fits headers of self (default = 0).
        dit : bool, optional
            Whether the DIT should be mentioned in the filename.
        ndit : bool, optional
            Whether the NDIT should be mentioned in the filename.
        mjd : bool, optional
            Whether the MJD should be mentioned in the filename.
        filt : bool, optional
            Whether the Filter should be mentioned in the filename.
        table : bool, optional
            If set, append '.tab' to the end of the filename.

        Returns
        -------
        str
            Master calibration file path.

        """

        # Common name
        path = self.path_master + basename

        # Append options
        if dit:
            path += ".DIT_" + str(self.dit[idx])
        if ndit:
            path += ".NDIT_" + str(self.ndit[idx])
        if mjd:
            path += ".MJD_" + str(np.round(self.mjd_mean, decimals=3))
        if filt:
            path += ".FIL_" + self.filter[idx]

        # File extensions
        path += ".fits"

        if table:
            path += ".tab"

        # Return
        return path


class MasterImages(FitsImages):

    def __init__(self, file_paths=None):
        super(MasterImages, self).__init__(file_paths=file_paths)

    @property
    def types(self):
        """
        Property which holds the calibration types

        Returns
        -------
        iterable
            Ordered list of calibration types
        """

        return self.primeheaders_get_keys([self.setup["keywords"]["object"]])[0]

    @property
    def bpm(self):
        """
        Holds all MasterBadPixelMask images.

        Returns
        -------
        MasterBadPixelMask
            All MasterBadPixelMask images as a MasterBadPixelMask instance.

        """

        # Import
        from vircampype.fits.images.bpm import MasterBadPixelMask

        # Get the masterbpm files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-BPM"]

        return MasterBadPixelMask(file_paths=[self.file_paths[idx] for idx in index])

    @property
    def dark(self):
        """
        Holds all MasterDark images.

        Returns
        -------
        MasterDark
            All MasterDark images as a MasterDark instance.

        """

        # Import
        from vircampype.fits.images.dark import MasterDark

        # Get the masterbpm files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-DARK"]

        return MasterDark(file_paths=[self.file_paths[idx] for idx in index])
