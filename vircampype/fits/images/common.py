import os
import numpy as np

from astropy.io import fits
from vircampype.data.cube import ImageCube
from vircampype.fits.common import FitsFiles


class FitsImages(FitsFiles):

    def __init__(self, setup, file_paths=None):
        """ Class for Fits images that includees specific methods and functions applicable only to images. """
        super(FitsImages, self).__init__(setup=setup, file_paths=file_paths)

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

        self._dit = self.read_from_prime_headers(keywords=[self.setup.keywords.dit])[0]
        return self._dit

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
            self._ndit = self.read_from_prime_headers(keywords=[self.setup.keywords.ndit])[0]
        except KeyError:
            self._ndit = [1] * self.n_files

        return self._ndit

    _texptime = None

    @property
    def texptime(self):

        # Check if already determined
        if self._texptime is not None:
            return self._texptime

        self._texptime = [dit * ndit for dit, ndit in zip(self.dit, self.ndit)]
        return self._texptime

    _passband = None

    @property
    def passband(self):
        """
        Property to return the filter entries from the main header

        Returns
        -------
        iterable
            List with filters for all files.

        """

        # Check if already determined
        if self._passband is not None:
            return self._passband

        self._passband = self.read_from_prime_headers(keywords=[self.setup.keywords.filter_name])[0]
        return self._passband

    @property
    def dit_norm(self):
        """ Convenience method for retrieving the DITs of the current instance as ndarray. """
        return np.array(self.dit)

    @property
    def ndit_norm(self):
        """ Convenience method for retrieving the NDITs of the current instance as ndarray. """
        return np.array(self.ndit)

    _filter = None

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
            if len(set(self.passband)) < n_filter_min:
                raise ValueError("Found {0:0g} different filters; "
                                 "min = {1:0g}".format(len(set(self.passband)), n_filter_min))
        if n_filter_max is not None:
            if len(set(self.passband)) > n_filter_max:
                raise ValueError("Found {0:0g} different filters; "
                                 "max = {1:0g}".format(len(set(self.passband)), n_filter_min))

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

        # Get bitpix keyword
        bitpix = [x[0] for x in self.read_from_data_headers(keywords=["BITPIX"])[0]]

        # Loop through everything and set types
        dtypes = []
        for bp in bitpix:
            if bp == 8:
                app = np.uint8
            elif bp == 16:
                app = np.uint16
            elif bp in (-32, 32):
                app = np.float32
            else:
                raise NotImplementedError("Data type '{0}' not implemented!".format(bp))

            dtypes.append(app)

        self._dtypes = dtypes
        return self._dtypes

    # =========================================================================== #
    # Data splitting
    # =========================================================================== #
    def split_types(self):
        """
        Basic file splitting routine for raw VIRCAM data.

        Returns
        -------
        dict
            Dictionary with subtypes.

        """

        # Import
        from vircampype.fits.images.dark import DarkImages
        from vircampype.fits.images.sky import SkyImagesRawScience, SkyImagesRawOffset, SkyImagesRawStd
        from vircampype.fits.images.flat import FlatTwilight, FlatLampLin, FlatLampCheck, FlatLampGain

        # Get the type, and category from the primary header
        types, category = self.read_from_prime_headers([self.setup.keywords.type, self.setup.keywords.category])

        # Extract the various data types for VIRCAM
        science_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                         c == "SCIENCE" and "OBJECT" in t]
        science = None if len(science_index) < 1 else \
            SkyImagesRawScience(setup=self.setup, file_paths=[self.paths_full[i] for i in science_index])

        offset_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                        c == "SCIENCE" and "SKY" in t]
        offset = None if len(offset_index) < 1 else \
            SkyImagesRawOffset(setup=self.setup, file_paths=[self.paths_full[i] for i in offset_index])

        dark_science_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                              c == "CALIB" and t == "DARK"]
        dark_science = None if len(dark_science_index) < 1 else \
            DarkImages(setup=self.setup, file_paths=[self.paths_full[i] for i in dark_science_index])

        flat_twilight_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                               c == "CALIB" and t == "FLAT,TWILIGHT"]
        flat_twilight = None if len(flat_twilight_index) < 1 else \
            FlatTwilight(setup=self.setup, file_paths=[self.paths_full[i] for i in flat_twilight_index])

        dark_lin_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                          c == "CALIB" and t == "DARK,LINEARITY"]
        dark_lin = None if len(dark_lin_index) < 1 else \
            DarkImages(setup=self.setup, file_paths=[self.paths_full[i] for i in dark_lin_index])

        flat_lamp_lin_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                               c == "CALIB" and t == "FLAT,LAMP,LINEARITY"]
        flat_lamp_lin = None if len(flat_lamp_lin_index) < 1 else \
            FlatLampLin(setup=self.setup, file_paths=[self.paths_full[i] for i in flat_lamp_lin_index])

        flat_lamp_check_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                                 c == "CALIB" and t == "FLAT,LAMP,CHECK"]
        flat_lamp_check = None if len(flat_lamp_check_index) < 1 else \
            FlatLampCheck(setup=self.setup, file_paths=[self.paths_full[i] for i in flat_lamp_check_index])

        dark_check_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                            c == "CALIB" and t == "DARK,CHECK"]
        dark_check = None if len(dark_check_index) < 1 else \
            DarkImages(setup=self.setup, file_paths=[self.paths_full[i] for i in dark_check_index])

        dark_gain_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                           c == "CALIB" and t == "DARK,GAIN"]
        dark_gain = None if len(dark_gain_index) < 1 else \
            DarkImages(setup=self.setup, file_paths=[self.paths_full[i] for i in dark_gain_index])

        flat_lamp_gain_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                                c == "CALIB" and t == "FLAT,LAMP,GAIN"]
        flat_lamp_gain = None if len(flat_lamp_gain_index) < 1 else \
            FlatLampGain(setup=self.setup, file_paths=[self.paths_full[i] for i in flat_lamp_gain_index])

        std_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                     c == "CALIB" and t == "STD,FLUX"]
        std = None if len(std_index) < 1 else \
            SkyImagesRawStd(setup=self.setup, file_paths=[self.paths_full[i] for i in std_index])

        return dict(science=science, offset=offset, std=std, dark_science=dark_science, dark_lin=dark_lin,
                    dark_gain=dark_gain, dark_check=dark_check, flat_twilight=flat_twilight,
                    flat_lamp_lin=flat_lamp_lin, flat_lamp_check=flat_lamp_check, flat_lamp_gain=flat_lamp_gain)

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

        # Read header of first file and given extension
        header = self.headers[0][hdu_index]

        # Create empty numpy cube
        cube = np.empty((self.n_files, header["NAXIS2"], header["NAXIS1"]), dtype=dtype)

        # Fill cube with data
        for path, plane in zip(self.paths_full, cube):
            with fits.open(path) as f:
                plane[:] = f[hdu_index].data

        # Return
        return ImageCube(setup=self.setup, cube=cube)

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
            hdu_index = self.iter_data_hdu[file_index]

        # Data type
        if dtype is None:
            dtype = self.dtypes[file_index]

        # Read header of requested file and first extension
        header = self.headers_data[file_index][0]

        # Create empty cube
        cube = np.empty((len(hdu_index), header["NAXIS2"], header["NAXIS1"]), dtype=dtype)

        # Fill cube with data
        with fits.open(name=self.paths_full[file_index]) as f:
            for plane, idx in zip(cube, hdu_index):
                plane[:] = f[idx].data

        # Return
        return ImageCube(setup=self.setup, cube=cube)

    # =========================================================================== #
    # Matching
    # =========================================================================== #
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
            a = self.__class__(setup=self.setup, file_paths=[f])
            b = match_to.__class__(setup=self.setup, file_paths=[match_to.paths_full[i] for i in idx])

            # Raise error if nothing is found
            if len(a) < 1 or len(b) < 1:
                raise ValueError("No matching exposure found.")

            # Get the closest in time
            matched.extend(a.match_mjd(match_to=b, max_lag=max_lag).paths_full)

        # Return
        return match_to.__class__(setup=self.setup, file_paths=matched)

    def match_passband(self, match_to, max_lag=None):
        """
        Matches all entries in the current instance with the match_to instance so that the filters match. In case there
        are multiple matches, will return the closest in time!

        Parameters
        ----------
        match_to : FitsImages
            FitsImages (or any child) instance out of which the matches should be drawn.
        max_lag : int, float, optional
            Maximum allowed time difference for matching in days. Default is None.

        Returns
        -------

        """

        # Check if input is indeed of FitsImages class
        if not isinstance(match_to, FitsImages):
            raise ValueError("Input objects are not FitsImages class")

        # Get matching indices (for each entry in pair_a get the indices in pair_b)
        indices = [[i for i, j in enumerate(match_to.passband) if j == k] for k in self.passband]

        # Create list for output
        matched = []

        # Now get the closest in time for each entry
        for f, idx in zip(self, indices):

            # Issue error if no files can be found
            if len(idx) < 1:
                raise ValueError("Could not find matching filter")

            # Otherwise append closest in time
            else:

                # Construct FitsFiles class
                a = self.__class__(setup=self.setup, file_paths=[f])
                b = match_to.__class__(setup=match_to.setup, file_paths=[match_to.paths_full[i] for i in idx])

                # Get the closest in time
                matched.extend(a.match_mjd(match_to=b, max_lag=max_lag).paths_full)

        # Return
        return match_to.__class__(setup=match_to.setup, file_paths=matched)

    # =========================================================================== #
    # Master images
    # =========================================================================== #
    def get_master_bpm(self):
        """
        Get for each file in self the corresponding MasterBadPixelMask.

        Returns
        -------
        MasterBadPixelMask
            MasterBadPixelMask instance holding for each file in self the corresponding MasterBadPixelMask file.

        """

        # Match and return
        return self.match_mjd(match_to=self.get_master_images().bpm, max_lag=self.setup.master_max_lag_bpm)

    def get_master_dark(self, ignore_dit=False):
        """
        Get for each file in self the corresponding MasterDark.

        Parameters
        ----------
        ignore_dit : bool, optional
            Whether to ignore DIT values. Default is False.

        Returns
        -------
        MasterDark
            MasterDark instance holding for each file in self the corresponding MasterDark file.

        """

        # Match DIT and NDIT and MJD
        return self._match_exposure(match_to=self.get_master_images().dark, max_lag=self.setup.master_max_lag_dark,
                                    ignore_dit=ignore_dit)

    def get_master_flat(self):
        """
        Get for each file in self the corresponding MasterFlat.

        Returns
        -------
        MasterFlat
            MasterFlat instance holding for each file in self the corresponding Masterflat file.

        """

        # Match and return
        return self.match_passband(match_to=self.get_master_images().flat, max_lag=self.setup.master_max_lag_flat)

    def get_unique_master_flats(self):
        """ Returns unique Master Flats as MasterFlat instance. """
        from vircampype.fits.images.flat import MasterFlat
        return MasterFlat(setup=self.setup, file_paths=list(set(self.get_master_flat().paths_full)))

    def get_master_source_mask(self):
        """
        Fetches the corresponding master source mask files, based on a MJD match.

        Returns
        -------
        MasterSourceMask
            MasterSourceMask instance holding all matches for the current instance.

        """
        return self.match_mjd(match_to=self.get_master_images().source_mask, max_lag=1 / 86400)

    def get_master_sky(self, mode: str):
        """
        Get for each file in self the corresponding Mastersky.

        Parameters
        ----------
        mode : str
            Either 'static' or 'dynamic'

        Returns
        -------
        MasterSky
            MasterSky instance holding for each file in self the corresponding MasterSky file.

        """

        # Match and return
        if "static" in mode.lower():
            return self.match_passband(match_to=self.get_master_images().sky_static,
                                       max_lag=self.setup.master_max_lag_sky / 1440.)
        elif "dynamic" in mode.lower():
            return self.match_passband(match_to=self.get_master_images().sky_dynamic,
                                       max_lag=self.setup.master_max_lag_sky / 1440.)
        else:
            raise ValueError("Mode '{0}' not supported for fetching master sky".format(mode))

    def get_master_weight_global(self):
        """
        Searches for MasterWeights in the following order:
        1. Local files with extention *.weight.fits
        2. Weight maps in the master folder

        Returns
        -------
        MasterWeight
            MasterWeight instance.

        Raises
        ------
        ValueError
            When not all images have an associated weight.

        """

        # Import
        from vircampype.fits.images.flat import MasterWeight

        # Look for weights in same folder with fitting name
        master_weight_paths = [x.replace(".fits", ".weight.fits") for x in self.paths_full]
        if sum([os.path.isfile(x) for x in master_weight_paths]) == len(self):
            return MasterWeight(file_paths=master_weight_paths, setup=self.setup)

        # If no local paths are found, fetch image weights
        master_weight_paths = self.match_passband(match_to=self.get_master_images().weight_global,
                                                  max_lag=self.setup.master_max_lag_weight).paths_full
        if sum([os.path.isfile(x) for x in master_weight_paths]) == len(self):
            return MasterWeight(file_paths=master_weight_paths, setup=self.setup)

        # If not all images have associated weights now, then something went wrong
        raise ValueError("Not all images have weights")

    def get_master_weight_image(self):

        # Import
        from vircampype.fits.images.flat import MasterWeight

        # If no local paths are founds, try to get a weight for each image
        master_weight_paths = self.match_mjd(match_to=self.get_master_images().weight_image,
                                             max_lag=self.setup.master_max_lag_weight).paths_full
        if sum([os.path.isfile(x) for x in master_weight_paths]) == len(self):
            return MasterWeight(file_paths=master_weight_paths, setup=self.setup)

    def get_master_illumination_correction(self):
        """
        Get for all files in self the corresponding MasterIlluminationCorrection (split by minutes from setup).

        Returns
        -------
        MasterIlluminationCorrection
            MasterIlluminationCorrection instance with matching files

        """
        return self.match_passband(match_to=self.get_master_images().illumination_correction, max_lag=1 / 1440.)

    # =========================================================================== #
    # Master tables
    # =========================================================================== #
    def get_master_linearity(self):
        """
        Get for each file in self the corresponding Masterlinearity table.

        Returns
        -------
        MasterLinearity
            MasterLinearity instance holding for each file in self the corresponding Masterlinearity table.

        """

        # Match and return
        return self.match_mjd(match_to=self.get_master_tables().linearity, max_lag=self.setup.master_max_lag_linearity)

    def get_master_gain(self):
        """
        Get for each file in self the corresponding MasterGain table.

        Returns
        -------
        MasterGain
            MasterGain instance holding for each file in self the corresponding Masterlinearity table.

        """

        # Match and return
        return self.match_mjd(match_to=self.get_master_tables().gain, max_lag=self.setup.master_max_lag_gain)


class MasterImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(MasterImages, self).__init__(setup=setup, file_paths=file_paths)

    @property
    def types(self):
        """
        Property which holds the calibration types

        Returns
        -------
        iterable
            Ordered list of calibration types
        """

        return self.read_from_prime_headers([self.setup.keywords.object])[0]

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

        return MasterBadPixelMask(setup=self.setup, file_paths=[self.paths_full[idx] for idx in index])

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

        return MasterDark(setup=self.setup, file_paths=[self.paths_full[idx] for idx in index])

    @property
    def flat(self):
        """
        Holds all MasterFlat images.

        Returns
        -------
        MasterFlat
            All MasterFlat images as a MasterFlat instance.

        """

        # Import
        from vircampype.fits.images.flat import MasterFlat

        # Get the masterbpm files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-FLAT"]

        return MasterFlat(setup=self.setup, file_paths=[self.paths_full[idx] for idx in index])

    @property
    def source_mask(self):

        # Import
        from vircampype.fits.images.sky import MasterSourceMask

        # Get the masterbpm files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-SOURCE-MASK"]

        return MasterSourceMask(setup=self.setup, file_paths=[self.paths_full[idx] for idx in index])

    @property
    def sky_static(self):
        """
        Retrieves all static MasterSky images.

        Returns
        -------
        MasterSky
            All MasterSky images as a MasterSky instance.

        """

        # Import
        from vircampype.fits.images.sky import MasterSky

        # Get the masterbpm files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-SKY-STATIC"]

        return MasterSky(setup=self.setup, file_paths=[self.paths_full[idx] for idx in index])

    @property
    def sky_dynamic(self):
        """
        Retrieves all static MasterSky images.

        Returns
        -------
        MasterSky
            All MasterSky images as a MasterSky instance.

        """

        # Import
        from vircampype.fits.images.sky import MasterSky

        # Get the masterbpm files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-SKY-DYNAMIC"]

        return MasterSky(setup=self.setup, file_paths=[self.paths_full[idx] for idx in index])

    @property
    def weight_global(self):
        """
        Retrieves all global MasterWeight images.

        Returns
        -------
        MasterWeight
            All MasterWeight images as a MasterWeight instance.

        """

        # Import
        from vircampype.fits.images.flat import MasterWeight

        # Get the masterbpm files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-WEIGHT-GLOBAL"]

        # Return MasterWeight instance
        return MasterWeight(setup=self.setup, file_paths=[self.paths_full[idx] for idx in index])

    @property
    def weight_image(self):
        """
        Retrieves all global MasterWeight images.

        Returns
        -------
        MasterWeight
            All MasterWeight images as a MasterWeight instance.

        """

        # Import
        from vircampype.fits.images.flat import MasterWeight

        # Get the masterbpm files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-WEIGHT-IMAGE"]

        # Return MasterWeight instance
        return MasterWeight(setup=self.setup, file_paths=[self.paths_full[idx] for idx in index])

    @property
    def illumination_correction(self):
        """
        Holds all MasterIlluminationCorrection images.

        Returns
        -------
        MasterIlluminationCorrection
            All MasterIlluminationCorrection images.

        """

        # Import
        from vircampype.fits.images.flat import MasterIlluminationCorrection

        # Get the masterbpm files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-ILLUMINATION-CORRECTION"]

        return MasterIlluminationCorrection(setup=self.setup, file_paths=[self.paths_full[idx] for idx in index])
