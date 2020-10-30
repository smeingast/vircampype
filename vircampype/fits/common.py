# =========================================================================== #
# Import
import os
import glob
import pickle
import warnings
import numpy as np

from typing import Dict
from astropy.io import fits
from joblib import cpu_count
from astropy.time import Time
from vircampype.utils import *
from astropy.io.fits.hdu.image import ImageHDU, PrimaryHDU


class FitsFiles:

    def __init__(self, setup, file_paths=None):
        """
        Top-level class of FITS files. Contains basic information on file structure and the headers.

        Parameters
        ----------
        setup : Dict[str, Dict[str, Any]]
            YML setup. Can be either path to setup, or a dictionary.
        file_paths : iterable, optional.
            Paths to FITS files.
        """

        # Setup
        if isinstance(setup, str):
            self.setup = read_setup(path_yaml=setup)
        elif isinstance(setup, dict):
            self.setup = setup
        else:
            raise ValueError("Please provide a pipeline setup")

        if file_paths is None:
            self.file_paths = []
        elif isinstance(file_paths, str):
            self.file_paths = [file_paths]
        else:
            self.file_paths = sorted(file_paths)

        # Some checks
        if not isinstance(self.file_paths, list):
            raise ValueError("file_paths must be non-empty list")

        # Check setup
        self.__check_setup()

        # Paths and properties
        self.base_names = [os.path.basename(x) for x in self.file_paths]
        self.file_names = [os.path.splitext(x)[0] for x in self.base_names]
        self.file_extensions = [os.path.splitext(x)[1] for x in self.base_names]
        self.full_paths = [os.path.abspath(x) for x in self.file_paths]
        self.file_directories = [os.path.dirname(x) + "/" for x in self.full_paths]
        self.n_files = len(self.file_paths)

        # Set name for current batch
        self.name = self.setup["paths"]["name"]

        # Initialize folders and set attributes manually
        self.path_pype = self.setup["paths"]["pype"]
        self.path_object = "{0}{1}/".format(self.path_pype, self.name)
        self.path_temp = "{0}{1}/".format(self.path_object, "temp")
        self.path_headers = "{0}{1}/".format(self.path_object, "headers")
        self.path_processed = "{0}{1}/".format(self.path_object, "processed")
        self.path_superflatted = "{0}{1}/".format(self.path_object, "superflatted")
        self.path_resampled = "{0}{1}/".format(self.path_object, "resampled")
        self.path_coadd = "{0}{1}/".format(self.path_object, "coadd")

        # Obs parameters
        self.path_obspar = "{0}{1}/".format(self.path_object, "obspar")
        self.path_apcor = "{0}{1}/".format(self.path_obspar, "apcor")

        # Master paths
        self.path_master_common = "{0}{1}/".format(self.path_pype, "master")
        self.path_master_object = "{0}{1}/".format(self.path_object, "master")

        # QC
        self.path_qc_object = "{0}{1}/".format(self.path_object, "qc")

        # Common QC
        self.path_qc_bpm = "{0}{1}/".format(self.path_qc_object, "bpm")
        self.path_qc_dark = "{0}{1}/".format(self.path_qc_object, "dark")
        self.path_qc_gain = "{0}{1}/".format(self.path_qc_object, "gain")
        self.path_qc_linearity = "{0}{1}/".format(self.path_qc_object, "linearity")
        self.path_qc_flat = "{0}{1}/".format(self.path_qc_object, "flat")

        # Sequence specific QC
        self.path_qc_sky = "{0}{1}/".format(self.path_qc_object, "sky")
        self.path_qc_photometry = "{0}{1}/".format(self.path_qc_object, "photometry")
        self.path_qc_astrometry = "{0}{1}/".format(self.path_qc_object, "astrometry")
        self.path_qc_apcor = "{0}{1}/".format(self.path_qc_object, "aperture_correction")
        self.path_qc_superflat = "{0}{1}/".format(self.path_qc_object, "superflat")

        # ESO
        self.path_phase3 = "{0}{1}/{2}/".format(self.path_pype, "phase3", self.name)

        # Common paths
        paths_common = [self.path_pype, self.path_headers, self.path_master_common, self.path_object]

        # calibration-specific paths
        paths_cal = [self.path_qc_bpm, self.path_qc_dark, self.path_qc_gain, self.path_qc_linearity, self.path_qc_flat]

        # Object-specific paths
        paths_obj = [self.path_temp, self.path_processed, self.path_resampled, self.path_coadd, self.path_obspar,
                     self.path_apcor, self.path_master_object, self.path_qc_object, self.path_qc_sky,
                     self.path_qc_photometry, self.path_qc_astrometry, self.path_qc_apcor, self.path_phase3,
                     self.path_qc_superflat, self.path_superflatted]

        # Generate common paths
        for path in paths_common:
            make_folder(path)

        # Create common calibration path only if we run a calibration unit
        if "calibration" in self.name.lower():
            for path in paths_cal:
                make_folder(path=path)

        # Other wise make object paths
        else:
            for path in paths_obj:
                make_folder(path=path)

        # Generate paths
        self._header_paths = ["{0}{1}.header".format(self.path_headers, x) for x in self.base_names]

    # =========================================================================== #
    # Setup check
    # =========================================================================== #
    def __check_setup(self):
        """ Makes some consitency checks in setup. """

        # Only single threads for python are allowed at the moment
        # if self.setup["misc"]["n_jobs"] != 1:
        #     raise SetupError(BColors.FAIL +
        #                      "Multiple threads in Python are not supported at the moment, "
        #                      "'n_jobs' = " "{0}".format(self.setup["misc"]["n_jobs"])
        #                      + BColors.ENDC)

        # Raise error when more threads than available are requested
        # if self.setup["misc"]["n_jobs"] > cpu_count():
        #     raise SetupError(BColors.FAIL +
        #                      "More threads reuqested than available. {0} > {1}"
        #                      "".format(self.setup["misc"]["n_jobs"], cpu_count())
        #                      + BColors.ENDC)

        # Raise error when more threads than available are requested
        if self.setup["misc"]["n_jobs"] > cpu_count():
            raise SetupError(BColors.FAIL +
                             "More threads reuqested than available. {0} > {1}"
                             "".format(self.setup["misc"]["n_jobs"], cpu_count())
                             + BColors.ENDC)

        # Check if astroscrappy is installed when found in setup
        # if self.setup["cosmetics"]["mask_cosmics"]:
        #     if not module_exists("astroscrappy"):
        #         raise SystemError("For cosmic ray detection, you need 'astroscrappy' installed.")

    # =========================================================================== #
    #   Magic methods
    # =========================================================================== #
    def __str__(self):
        return str(self.full_paths)

    def __repr__(self):
        return str(self.full_paths)

    def __iter__(self):
        return iter(self.full_paths)

    def __setitem__(self, key, item):
        # Only if file exists
        if os.path.isfile(item):
            self.full_paths[key] = item
        else:
            raise FileNotFoundError("The specified file '{0}' does not exist.".format(item))

    def __getitem__(self, key):
        return self.__class__(setup=self.setup, file_paths=self.full_paths[key])

    def __len__(self):
        return self.n_files

    def __add__(self, other):
        return self.__class__(setup=self.setup, file_paths=self.full_paths + other.full_paths)

    def __iadd__(self, other):
        return self.__class__(setup=self.setup, file_paths=self.full_paths + other.full_paths)

    # =========================================================================== #
    #   I/O
    # =========================================================================== #
    @classmethod
    def from_folder(cls, path, setup, pattern=None, exclude=None):
        """
        Loads all files from the given folder into a FitsFiles (or child) instance.

        Parameters
        ----------
        path : str
            Path to folder.
        setup : str, dict
            YML setup. Can be either path to setup, or a dictionary.
        pattern : str, optional
            Substring to identify FITS files. Default is None, which loads all files in the folder.
        exclude : str, optional
            Substring contained in filenames for files that should be excluded.

        Returns
        -------
            Instance with the found files built from the requested class.

        """

        # Append / if not set
        if not path.endswith("/"):
            path += "/"

        if pattern is not None:
            file_paths = glob.glob(path + pattern)
        else:
            file_paths = glob.glob(path + "*")

        if exclude is not None:
            file_paths = [x for x in file_paths if exclude not in x]

        # Return new instance
        return cls(setup=setup, file_paths=file_paths)

    @classmethod
    def from_setup(cls, setup):
        """
        Creates instance by reading paths from setup

        Parameters
        ----------
        setup : str, dict
            YML setup. Can be either path to setup, or a dictionary.

        Returns
        -------
            Instance with the found files built from the requested class.

        """

        # Read setup
        setup = read_setup(path_yaml=setup)

        # Fix path if necessary
        path = setup["paths"]["data"]
        if not path.endswith("/"):
            path += "/"

        return cls(setup=setup, file_paths=glob.glob(path + setup["paths"]["pattern"]))

    # =========================================================================== #
    # Headers
    # =========================================================================== #
    _headers = None

    # noinspection DuplicatedCode
    @property
    def headers(self):

        # Check if already determined
        if self._headers is not None:
            return self._headers

        headers = []
        for idx in range(self.n_files):

            # Try to read the database
            try:
                with open(self._header_paths[idx], "rb") as f:

                    # If the file is there, load the headers...
                    headers.append(pickle.load(f))

                    # And continue with next file
                    continue

            # If not found we move on to read the headers from the fits file
            except FileNotFoundError:

                with fits.open(self.full_paths[idx]) as hdulist:

                    fileheaders = []
                    for hdu in hdulist:

                        # Load header
                        hdr = hdu.header

                        if self.setup["data"]["fix_vircam_header"]:
                            try:
                                hdr.remove("HIERARCH ESO DET CHIP PXSPACE")
                            except KeyError:
                                pass

                        # Check if header has been fixed already
                        try:
                            fixed = hdr["HIERARCH PYPE WCS RESET"]
                        except KeyError:
                            fixed = False

                        # Reset WCS if set
                        if self.setup["data"]["reset_wcs"] and not fixed:

                            # Save Target coordinate
                            if isinstance(hdu, PrimaryHDU):

                                try:
                                    tra = str(hdr["HIERARCH ESO TEL TARG ALPHA"])
                                    tde = str(hdr["HIERARCH ESO TEL TARG DELTA"])

                                    # Silly fix for short ALPHA/DELTA strings
                                    if len(tra.split(".")[0]) == 5:
                                        tra = "0" + tra
                                    elif len(tra.split(".")[0]) == 4:
                                        tra = "00" + tra
                                    elif len(tra.split(".")[0]) == 3:
                                        tra = "000" + tra
                                    # TODO: This does not work when the string starts with a '-'
                                    #  (but the astropy should throw and error as it can't read the string)
                                    if len(tde.split(".")[0]) == 5:
                                        tde = "0" + tde
                                    elif len(tde.split(".")[0]) == 4:
                                        tde = "00" + tde
                                    elif len(tde.split(".")[0]) == 3:
                                        tde = "000" + tde

                                    field_ra = 15 * (float(tra[:2]) + float(tra[2:4]) / 60 + float(tra[4:]) / 3600)

                                    if tde[0].startswith("-"):
                                        field_de = float(tde[:3]) - float(tde[3:5]) / 60 - float(tde[5:]) / 3600
                                    else:
                                        field_de = float(tde[:2]) + float(tde[2:4]) / 60 + float(tde[4:]) / 3600

                                except KeyError:
                                    field_ra, field_de = None, None

                            if isinstance(hdu, ImageHDU):

                                # Overwrite with consistently working keyword
                                try:
                                    hdr["CRVAL1"] = field_ra if field_ra is not None else hdr["CRVAL1"]
                                    hdr["CRVAL2"] = field_de if field_ra is not None else hdr["CRVAL2"]
                                except KeyError:
                                    pass

                                with warnings.catch_warnings():
                                    warnings.filterwarnings("ignore")
                                    hdr = header_reset_wcs(hdr)
                                    hdr["HIERARCH PYPE WCS RESET"] = True

                        # Save cleaned header
                        fileheaders.append(hdr)

                # When done for all headers dump them into the designated database
                with open(self._header_paths[idx], "wb") as d:
                    pickle.dump(fileheaders, d)

                headers.append(fileheaders)

        # Return all headers
        self._headers = headers
        return self._headers

    @property
    def headers_primary(self):
        """
        Returns a list of primary headers for all files.

        Returns
        -------
        iterable
            List of primary headers.

        """

        return [hdrs[0] for hdrs in self.headers]

    def primeheaders_get_keys(self, keywords):
        """
        Simple method to return a list with lists for the individual values of the supplied keys from the primary
        headers

        Parameters
        ----------
        keywords : list[str]
            List of FITS header keys in the primary header

        Returns
        -------
        iterable
            List of lists for all input keywords (unpackable) containing the stored value in the headers

        Raises
        ------
        TypeError
            When the supplied keywords are not in a list.

        """

        if not isinstance(keywords, list):
            raise TypeError("Keywords must be in a list!")

        # Return values
        return [[h[k] for h in self.headers_primary] if k is not None else [None for _ in range(len(self))]
                for k in keywords]

    @property
    def headers_data(self):
        """
        Returns a list of primary headers for all files.

        Returns
        -------
        iterable
            List of primary headers.

        """

        return [[hdrs[i] for i in idx] for hdrs, idx in zip(self.headers, self.data_hdu)]

    def dataheaders_get_keys(self, keywords, file_index=None):
        """
        Method to return a list with lists for the individual values of the supplied keys from the data headers

        Parameters
        ----------
        keywords : list[str]
            List of FITS header keys in the primary header
        file_index : int, optional
            If set, only retrieve values from given file.

        Returns
        -------
        iterable
            Triple stacked list: List which contains a list of all keywords which in turn contain the values from all
            data headers

        Raises
        ------
        TypeError
            When the supplied keywords are not in a list.

        """

        if not isinstance(keywords, list):
            raise TypeError("Keywords must be in a list!")

        if file_index is None:
            headers_data = self.headers_data[:]
        else:
            headers_data = [self.headers_data[file_index]]

        # Return values
        return [[[e[k] for e in h] for h in headers_data] for k in keywords]

    def _get_dataheaders_sequence(self, keyword):
        """
        Retrieves values from dataheaders that are atored in a sequence like 'keyword 0' - 'keyword 1' - ...

        Parameters
        ----------
        keyword : str
            Keyword in header

        Returns
        -------
        List
            List of values

        """

        idx, temp = 0, []
        while True:
            try:
                temp.append(self.dataheaders_get_keys(keywords=["{0} {1}".format(keyword, idx)])[0])
                idx += 1
            except KeyError:
                break

        temp = np.rollaxis(np.array(temp), axis=1)
        return [t.T.tolist() for t in temp]

    def delete_headers_temp(self, file_index=None):
        """ Removes temporary header files. """

        if file_index is not None:
            remove_file(path=self._header_paths[file_index])
        else:
            for p in self._header_paths:
                remove_file(path=p)

    def add_dataheader_key(self, key, values, comments=None):
        """
        Add key/values to dataheaders. Values lists must match the data format exactly. i.e. top list must match length
        of self. Nested lists must correspond to data hdus.

        Parameters
        ----------
        key : str
            Which keyword to add.
        values : iterable
            Values to add to each data HDU.
        comments : iterable, optional
            If set, comments to add

        """

        # Dummy check
        if len(self) != len(values):
            raise ValueError("Values must be provided for each image")

        # Values must also match for all extensions
        for d, v in zip(self.data_hdu, values):
            if len(d) != len(v):
                raise ValueError("Values must be provided for each extension")

        # Loop over files and add values
        for idx in range(len(self)):
            add_key_file(path=self.full_paths[idx], key=key, values=values[idx], comments=comments,
                         hdu_data=self.data_hdu[idx])

            # Force removing temp header
            self.delete_headers_temp(file_index=idx)

    # =========================================================================== #
    # Data properties
    # =========================================================================== #
    _n_hdu = None

    @property
    def n_hdu(self):
        """
        Property which holds the total number of HDUs for all input Fits files

        Returns
        -------
        iterable
            List of the number of HDUs for each input file
        """

        # Check if already determined
        if self._n_hdu is not None:
            return self._n_hdu

        self._n_hdu = [len(h) for h in self.headers]

        # Return
        return self._n_hdu

    _time_obs = None

    @property
    def data_hdu(self):
        """
        Property which holds an iterator for each file containing the indices for data access. In general it is assumed
        here that the primary HDU always has index 0. If there is only one HDU, then this is assumed to be the primary
        HDU which also contains data. If there are extensions, then the primary HDU is not assumed to contain data!

        Returns
        -------
        iterable
            List of iterators for header indices of HDUs which hold data.
        """
        return [range(0, 1) if len(hdrs) == 1 else range(1, len(hdrs)) for hdrs in self.headers]

    @property
    def time_obs(self):

        # Check if already determined
        if self._time_obs is not None:
            return self._time_obs
        else:
            pass
        self._time_obs = Time(self.primeheaders_get_keys([self.setup["keywords"]["date_mjd"]])[0],
                              scale="utc", format="mjd")

        return self._time_obs

    @property
    def time_obs_mean(self):
        return Time(self.mjd_mean, format="mjd").fits

    @property
    def mjd(self):
        """
        Property to hold all MJDs for the observations in a list.

        Returns
        -------
        iterable
            List of MJDs for all files

        """

        return self.time_obs.mjd

    @property
    def mjd_mean(self):
        """
        Mean MJD for all files of the current instance.

        Returns
        -------
        float
            Mean MJD of instance files.
        """

        return np.mean(self.mjd)

    @property
    def mjd_range(self):
        """
        MJD range of instance (max - min)

        Returns
        -------
        float
            MJD difference.
        """

        return self.mjd.max() - self.mjd.min()

    def sort_by_mjd(self):
        """
        Sorts input by MJD and returns new instance.

        Returns
        -------
            Same class as input, but sorted by MJD.

        """
        sorted_paths = [x for _, x in sorted(zip(self.mjd, self.full_paths))]
        return self.__class__(setup=self.setup, file_paths=sorted_paths)

    # =========================================================================== #
    # Data splitter
    # =========================================================================== #
    def split_keywords(self, keywords):
        """
        General file-splitting method for any keywords in the primary FITS header.

        Parameters
        ----------
        keywords : list[str]
            List of header keywords

        Returns
        -------
        iterable
            List with split FitsFiles entries based on unique keyword-value combinations.

        """

        # Get the entries for the keywords
        entries = self.primeheaders_get_keys(keywords=keywords)

        # Construct list of tuples with the keywords
        tup = [tuple(i) for i in zip(*entries)]

        # Find unique entries
        utup = set(tup)

        # Get the split indices
        split_indices = [[i for i, j in enumerate(tup) if j == k] for k in utup]

        split_list = []
        for s_idx in split_indices:
            split_list.append(self.__class__(setup=self.setup, file_paths=[self.file_paths[idx] for idx in s_idx]))

        return split_list

    def split_lag(self, max_lag, sort_mjd=False):
        """
        Splitting function which splits the input files based on a given maximum time difference.

        Parameters
        ----------
        max_lag : float, integer
            Maximum allowed time difference between split sets in hours
        sort_mjd : bool, optional
            If set, sort the output list by increasing mjd.

        Returns
        -------
        List
            List holding individual FitsFiles instance based on determined splits.

        """

        # Sort input by MJD
        mjd_sorted = sorted(self.mjd)
        sort_index = sorted(range(self.n_files), key=lambda k: self.mjd[k])
        sorted_paths = [self.file_paths[i] for i in sort_index]

        # Get for all files the integer hour relative to the start and the time between
        hour = [(m - min(mjd_sorted)) * 24 for m in mjd_sorted]

        # Get lag
        lag = [a - b for a, b in zip(hour[1:], hour[:-1])]

        # Get the indices where the data is spread out over more than max_lag
        split_indices = [i + 1 for k, i in zip(lag, range(len(lag))) if k > max_lag]

        # Add first and last index
        split_indices.insert(0, 0)
        split_indices.append(len(mjd_sorted))

        # Now just split at the indices
        split_list = []
        for idx in range(len(split_indices)):

            try:
                # Get current upper and lower
                lower = split_indices[idx]
                upper = split_indices[idx + 1]

                # Append files
                split_list.append(self.__class__(setup=self.setup, file_paths=sorted_paths[lower:upper]))

            # On the last iteration we get an Index error since there is no idx + 1
            except IndexError:
                pass

        # Sort by MJD
        if sort_mjd:
            sidx = np.argsort([s.mjd_mean for s in split_list])
            split_list = [split_list[i] for i in sidx]

        # Return the list which contains the separated file paths
        return split_list

    def split_window(self, window, remove_duplicates=True):
        """
        Splits input files based on time intervals

        Parameters
        ----------
        window : float, int
            Time window in minutes for which to create FitsList entries (+/- around self.mjd).
        remove_duplicates : bool, optional
            When set, list duplicates will be removed (default = True)

        Returns
        -------
        List
            List holding individual FitsFiles instance based on determined splits.

        """

        # Keep everything within interval
        split_indices = [[i for i, t in enumerate([abs(1440 * (mjd - m)) for m in self.mjd]) if t < window / 2] for
                         mjd in self.mjd]

        # Remove duplicates
        if remove_duplicates:
            split_indices = [list(x) for x in set(tuple(x) for x in split_indices)]

        # Create FitsFiles entries for list
        split_list = []
        for s_idx in split_indices:
            split_list.append(self.__class__(setup=self.setup, file_paths=[self.file_paths[idx] for idx in s_idx]))

        # Return List
        return split_list

    def split_values(self, values):
        """
        Split self based on some given value. The length of the value iterable must match the length of self.

        Parameters
        ----------
        values : iterable
            Some value list or array. Unique entries will be split into different instances.

        Returns
        -------
        List
            List holding individual FitsFiles instance based on determined splits.

        """

        if len(self) != len(values):
            raise ValueError("Split value must be provided for every file in self.")

        # Get split indices for unique parameters
        split_indices = [[i for i, j in enumerate(values) if j == u] for u in list(set(values))]

        split_list = []
        for s_idx in split_indices:
            split_list.append(self.__class__(setup=self.setup, file_paths=[self.file_paths[idx] for idx in s_idx]))

        # Return List
        return split_list

    # =========================================================================== #
    # Matcher
    # =========================================================================== #
    def match_mjd(self, match_to, max_lag=None):
        """
        Matches all entries in the current instance with the match_to instance according to the closest match in time.

        Parameters
        ----------
        match_to : FitsFiles
            FitsFiles (or any child) instance out of which the matches should be drawn.
        max_lag : int, float, optional
            Maximum allowed time difference for matching in days. Default is None.

        Returns
        -------
            Matched files

        Raises
        ------
        ValueError
            When the maximum time difference is exceeded.

        """

        # Check if input is indeed a FitsFiles class
        # if self.__class__.__mro__[-2] != match_to.__class__.__mro__[-2]:
        #     raise ValueError("Input objects are not FitsFiles class")

        # Loop through each file
        matched = []
        for mjd in self.mjd:

            # Calculate differences (the float here suppresses a weird warning)
            mjd_diff = [abs(float(mjd) - x) for x in match_to.mjd]

            # Raise Error when there is no file available within the maximum lag
            if max_lag is not None:
                if np.min(mjd_diff) > max_lag:
                    raise ValueError("No match within allowed time frame ({0} < {1})"
                                     .format(max_lag, np.round(np.min(mjd_diff), decimals=4)))

            # Get minimum index and append files
            matched.append(match_to.full_paths[mjd_diff.index(min(mjd_diff))])

        # Return
        return match_to.__class__(setup=self.setup, file_paths=matched)

    # =========================================================================== #
    # Master finder
    # =========================================================================== #
    def get_master_images(self):
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
        from vircampype.fits.images.common import MasterImages

        # Get paths in the master calibration directory
        paths = glob.glob(self.path_master_common + "*.fits") + glob.glob(self.path_master_object + "*.fits")

        # If there is nothing, issue error
        if len(paths) < 1:
            raise ValueError("No master calibration images found!")

        return MasterImages(setup=self.setup, file_paths=paths)

    def get_master_tables(self):
        """
        Gets all MasterTables for current instance

        Returns
        -------
        MasterTables
            MasterTables instance with all master fits tables in the mastercalibration directory

        Raises
        ------
        ValueError
            If no files are found.

        """
        from vircampype.fits.tables.common import MasterTables

        # Get paths in the master calibration directory
        paths = glob.glob(self.path_master_common + "*.fits.tab") + glob.glob(self.path_master_object + "*.fits.tab")

        # If there is nothing, issue error
        if len(paths) < 1:
            raise ValueError("No master calibration tables found!")

        return MasterTables(setup=self.setup, file_paths=paths)

    def get_aperture_correction(self, diameter=None):
        """
        Fetches all aperture correction images in the obspar directory.

        Parameters
        ----------
        diameter : int, float, optional
            If given, return only aperture correction files for given aperture.

        Returns
        -------
        ApcorImages
            ApcorImages instance with all aperture correction files.

        """

        # Import
        from vircampype.fits.images.obspar import ApcorImages

        # Get aperture correction files matching the file name exactly
        if diameter is None:
            paths = flat_list([glob.glob("{0}{1}*apcor*.fits".format(self.path_apcor, fn))
                               for fn in self.file_names])
        else:
            paths = flat_list([glob.glob("{0}{1}*apcor{2}.fits".format(self.path_apcor, fn, diameter))
                               for fn in self.file_names])

        # Return matched aperture correction files
        return ApcorImages(setup=self.setup, file_paths=paths)

    # =========================================================================== #
    # Master tables
    # =========================================================================== #
    def get_master_photometry(self):
        """
        Get for all files in self the corresponding MasterPhotometry table.

        Returns
        -------
        MasterPhotometry
            MasterPhotometry instance holding for all files in self the corresponding MasterPhotometry table.

        """
        return self.get_master_tables().photometry

    # =========================================================================== #
    # Setup stuff
    # =========================================================================== #
    @property
    def _aperture_save_idx(self):
        return [[i for i, x in enumerate(apertures_all) if x == b][0] for b in apertures_out]
