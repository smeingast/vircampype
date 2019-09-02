# =========================================================================== #
# Import
from itertools import repeat
from vircampype.data.cube import ImageCube
from vircampype.utils.system import run_cmds
from vircampype.fits.common import FitsFiles
from vircampype.utils.miscellaneous import *
from vircampype.utils.astromatic import yml2config
from vircampype.fits.tables.sextractor import SextractorTable


class FitsImages(FitsFiles):

    def __init__(self, setup, file_paths=None):
        """
        Class for Fits images based on FitsFiles. Contains specific methods and functions applicable only to images

        Parameters
        ----------
        setup : str, dict
            YML setup. Can be either path to setup, or a dictionary.

        Returns
        -------

        """

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
            self._ndit = self.primeheaders_get_keys(keywords=[self.setup["keywords"]["ndit"]])[0]
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

        # Get bitpix keyword
        bitpix = [x[0] for x in self.dataheaders_get_keys(keywords=["BITPIX"])[0]]

        # Loop through everything and set types
        dtypes = []
        for bp in bitpix:
            if bp == 8:
                app = np.uint8
            elif bp == 16:
                app = np.uint16
            elif bp in (-32, 32):
                app = np.float32
            # elif "float32" in dtype:
            #     dtypes[tidx][bidx] = np.float32
            else:
                raise NotImplementedError("Data type '{0}' not implemented!".format(bp))

            dtypes.append(app)

        self._dtypes = dtypes
        return self._dtypes

    @property
    def paths_calibrated(self):
        """
        Generates paths for calibrated images.

        Returns
        -------
        List
            List with paths for each file.

        """
        return ["{0}{1}.cal{2}".format(d, n, e) for d, n, e in
                zip(repeat(self.setup["paths"]["calibrated"]), self.file_names, self.file_extensions)]

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
            hdu_index = self.data_hdu[file_index]

        # Data type
        if dtype is None:
            dtype = self.dtypes[file_index]

        # Create empty cube
        cube = np.empty((len(hdu_index), self.setup["data"]["dim_y"], self.setup["data"]["dim_x"]), dtype=dtype)

        # Fill cube with data
        with fits.open(name=self.full_paths[file_index]) as f:
            for plane, idx in zip(cube, hdu_index):
                plane[:] = f[idx].data

        # Return
        return ImageCube(setup=self.setup, cube=cube)

    # =========================================================================== #
    # Splitter
    # =========================================================================== #
    def split_filter(self):
        """
        Splits self files based on unique filter entries in the FITS headers.

        Returns
        -------
        iterable
            List if FitsImages.

        """

        # Filter keyword must be present!
        return self.split_keywords(keywords=[self.setup["keywords"]["filter"]])

    def split_exposure(self):
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
    def get_master_bpm(self):
        """
        Get for each file in self the corresponding MasterBadPixelMask.

        Returns
        -------
        MasterBadPixelMask
            MasterBadPixelMask instance holding for each file in self the corresponding MasterBadPixelMask file.

        """

        # Match and return
        return self.match_mjd(match_to=self.get_master_images().bpm,
                              max_lag=self.setup["master"]["max_lag_bpm"])

    def get_master_dark(self):
        """
        Get for each file in self the corresponding MasterDark.

        Returns
        -------
        MasterDark
            MasterDark instance holding for each file in self the corresponding MasterDark file.

        """

        # Match DIT and NDIT and MJD
        return self._match_exposure(match_to=self.get_master_images().dark,
                                    max_lag=self.setup["master"]["max_lag_dark"])

    def get_master_flat(self):
        """
        Get for each file in self the corresponding MasterFlat.

        Returns
        -------
        MasterFlat
            MasterFlat instance holding for each file in self the corresponding Masterflat file.

        """

        # Match and return
        return self.match_filter(match_to=self.get_master_images().flat,
                                 max_lag=self.setup["master"]["max_lag_flat"])

    def get_master_weight(self):
        """
        Get for each file in self the corresponding MasterWeight.

        Returns
        -------
        MasterWeight
            MasterWeight instance holding for each file in self the corresponding MasterWeight file.

        """

        # Match and return
        return self.match_filter(match_to=self.get_master_images().weight,
                                 max_lag=self.setup["master"]["max_lag_flat"])

    def get_unique_master_flats(self):
        """ Returns unique Master Flats as MasterFlat instance. """
        from vircampype.fits.images.flat import MasterFlat
        return MasterFlat(setup=self.setup, file_paths=list(set(self.get_master_flat().full_paths)))

    def get_master_sky(self):
        """
        Get for each file in self the corresponding Mastersky.

        Returns
        -------
        MasterSky
            MasterSky instance holding for each file in self the corresponding MasterSky file.

        """

        # Match and return
        return self.match_filter(match_to=self.get_master_images().sky,
                                 max_lag=self.setup["master"]["max_lag_sky"] / 1440.)

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
        return self.match_mjd(match_to=self.get_master_tables().linearity,
                              max_lag=self.setup["master"]["max_lag_linearity"])

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
            a = self.__class__(setup=self.setup, file_paths=[f])
            b = match_to.__class__(setup=self.setup, file_paths=[match_to.file_paths[i] for i in idx])

            # Raise error if nothing is found
            if len(a) < 1 or len(b) < 1:
                raise ValueError("No matching exposure found.")

            # Get the closest in time
            matched.extend(a.match_mjd(match_to=b, max_lag=max_lag).full_paths)

        # Return
        return match_to.__class__(setup=self.setup, file_paths=matched)

    # noinspection PyTypeChecker
    def match_filter(self, match_to, max_lag=None):
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
        indices = [[i for i, j in enumerate(match_to.filter) if j == k] for k in self.filter]

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
                b = match_to.__class__(setup=match_to.setup, file_paths=[match_to.file_paths[i] for i in idx])

                # Get the closest in time
                matched.extend(a.match_mjd(match_to=b, max_lag=max_lag).full_paths)

        # Return
        return match_to.__class__(setup=match_to.setup, file_paths=matched)

    # =========================================================================== #
    # Main data calibration
    # =========================================================================== #
    def calibrate(self):
        """ Main science calibration method. All options are set in the setup. """
        # TODO: Write better docstring

        # import
        from vircampype.fits.images.sky import MasterSky
        from vircampype.fits.images.dark import MasterDark
        from vircampype.fits.images.flat import MasterFlat
        from vircampype.fits.tables.linearity import MasterLinearity

        # Processing info
        tstart = message_mastercalibration(master_type="CALIBRATION", silent=self.setup["misc"]["silent"], right="")

        # Fetch the Masterfiles
        master_dark = self.get_master_dark()  # type: MasterDark
        master_flat = self.get_master_flat()  # type: MasterFlat
        master_sky = self.get_master_sky()  # type: MasterSky
        master_linearity = self.get_master_linearity()  # type: MasterLinearity

        # Loop over files and apply calibration
        for idx in range(self.n_files):

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=self.paths_calibrated[idx], silent=self.setup["misc"]["silent"]):
                continue

            # Print processing info
            message_calibration(n_current=idx + 1, n_total=self.n_files, name=self.paths_calibrated[idx],
                                d_current=None, d_total=None, silent=self.setup["misc"]["silent"])

            # Read file into cube
            calib_cube = self.file2cube(file_index=idx, hdu_index=None, dtype=np.float32)

            # Get master calibration
            dark = master_dark.file2cube(file_index=idx, hdu_index=None, dtype=np.float32)
            flat = master_flat.file2cube(file_index=idx, hdu_index=None, dtype=np.float32)
            sky = master_sky.file2cube(file_index=idx, hdu_index=None, dtype=np.float32)
            lin = master_linearity.file2coeff(file_index=idx, hdu_index=None)
            norm_before = self.ndit_norm[idx]

            # Do calibration
            calib_cube.calibrate(dark=dark, flat=flat, linearize=lin, sky=sky, norm_before=norm_before)

            # Apply cosmetics
            if self.setup["cosmetics"]["interpolate_bad"]:
                calib_cube.interpolate_nan()
            if self.setup["cosmetics"]["destripe"]:
                calib_cube.destripe()

            # Write to disk
            calib_cube.write_mef(path=self.paths_calibrated[idx], prime_header=self.headers_primary[idx],
                                 data_headers=self.headers_data[idx])
        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

        # Return new instance of calibrated images
        return self.__class__(setup=self.setup, file_paths=self.paths_calibrated)

    # =========================================================================== #
    # Astromatic
    # =========================================================================== #
    def sextractor(self, preset="scamp"):
        """
        Runs sextractor based on given presets.

        Parameters
        ----------
        preset : str
            Preset name.

        Returns
        -------
        SextractorTable
            SextractorTable instance with the generated catalogs.

        """

        # Shortcut for resources
        package = "vircampype.resources.astromatic.sextractor"

        # Find executable
        path_exe = which(self.setup["astromatic"]["bin_sex"])

        # Find setup file
        path_filter = get_resource_path(package=package, resource="gauss_2.5_5x5.conv")
        # path_default_config = get_resource_path(package=package, resource="default.config")

        # Construct output catalog paths
        path_tables = [x.replace(end, "{0}.{1}tab".format(end, preset)) for x, end
                       in zip(self.full_paths, self.file_extensions)]

        # Check for existing files
        path_tables_clean = []
        if not self.setup["misc"]["overwrite"]:
            for pt in path_tables:
                if not os.path.isfile(pt):
                    path_tables_clean.append(pt)

        # Fetch masterweights
        master_weight = self.get_master_weight()

        # Fetch param file
        if preset == "scamp":
            path_param = get_resource_path(package=package, resource="presets/sextractor_scamp.param")
            ss = yml2config(path=get_resource_path(package=package, resource="presets/sextractor_scamp.yml"),
                            filter_name=path_filter, parameters_name=path_param, skip=["catalog_name", "weight_image"])

        else:
            raise ValueError("Preset '{0}' not supported".format(preset))

        # Construct commands for source extraction
        cmds = ["{0} {1} -CATALOG_NAME {2} -WEIGHT_IMAGE {3} {4}".format(path_exe, image, catalog, weight, ss)
                for image, catalog, weight in zip(self.full_paths, path_tables_clean, master_weight.full_paths)]

        # Run Sextractor
        run_cmds(cmds=cmds, silent=False, n_processes=self.setup["misc"]["n_threads"])

        # Return Table instance
        return SextractorTable(setup=self.setup, file_paths=path_tables)

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

    def build_master_path(self, basename, idx=0, dit=False, ndit=False, mjd=False, filt=False, table=False):
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
            path += ".MJD_" + str(np.round(self.mjd_mean, decimals=4))
        if filt:
            path += ".FIL_" + self.filter[idx]

        # File extensions
        path += ".fits"

        if table:
            path += ".tab"

        # Return
        return path

    def get_saturation_hdu(self, hdu_index):
        """

        Parameters
        ----------
        hdu_index : int
            Integer index of HDU for which the saturation should be returned

        Returns
        -------
        float
            Saturation of requested hdu from self.setup

        """

        # Need -1 here since the coefficients do not take an empty primary header into account
        if hdu_index-1 < 0:
            raise ValueError("HDU with index {0} does not exits".format(hdu_index-1))

        # Return HDU saturation limit
        return self.setup["data"]["saturate"][hdu_index-1]


class MasterImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(MasterImages, self).__init__(setup=setup, file_paths=file_paths)

    @property
    def paths_qc_plots(self):
        return [x.replace(".fits", ".pdf") for x in self.full_paths]

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

        return MasterBadPixelMask(setup=self.setup, file_paths=[self.file_paths[idx] for idx in index])

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

        return MasterDark(setup=self.setup, file_paths=[self.file_paths[idx] for idx in index])

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

        return MasterFlat(setup=self.setup, file_paths=[self.file_paths[idx] for idx in index])

    @property
    def sky(self):
        """
        Retrieves all MasterSky images.

        Returns
        -------
        MasterSky
            All MasterSky images as a MasterSky instance.

        """

        # Import
        from vircampype.fits.images.sky import MasterSky

        # Get the masterbpm files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-SKY"]

        return MasterSky(setup=self.setup, file_paths=[self.file_paths[idx] for idx in index])

    @property
    def weight(self):
        """
        Retrieves all MasterWeight images.

        Returns
        -------
        MasterWeight
            All MasterWeight images as a MasterSky instance.

        """

        # Import
        from vircampype.fits.images.flat import MasterWeight

        # Get the masterbpm files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-WEIGHT"]

        return MasterWeight(setup=self.setup, file_paths=[self.file_paths[idx] for idx in index])
