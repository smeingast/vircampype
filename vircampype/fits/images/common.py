# =========================================================================== #
# Import
import os
import numpy as np

from astropy.io import fits
from vircampype.utils import *
from vircampype.setup import *
from vircampype.data.cube import ImageCube
from vircampype.fits.common import FitsFiles
from vircampype.fits.tables.sextractor import SextractorCatalogs


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
    def paths_processed(self):
        """
        Generates paths for processed images.

        Returns
        -------
        iterable
            List with paths for each file.
        """
        return ["{0}{1}.proc{2}".format(self.path_processed, n, e) for n, e
                in zip(self.file_names, self.file_extensions)]

    @property
    def paths_superflatted(self):
        """
        Generates paths for superflatted images.

        Returns
        -------
        iterable
            List with paths for each file.
        """
        return ["{0}{1}.sf{2}".format(self.path_superflatted, n, e) for n, e
                in zip(self.file_names, self.file_extensions)]

    @property
    def _paths_aheaders(self):
        """
        Generates path for aheads (if any).

        Returns
        -------
        iterable
            List with aheader paths.

        """

        return [x.replace(".fits", ".ahead") for x in self.full_paths]

    @property
    def pixel_scale(self):
        """
        Reads the pixel scale from the setup and converts it to a float in degrees.

        Returns
        -------
        float

        """
        return fraction2float(self.setup["astromatic"]["pixel_scale"]) / 3600.

    # =========================================================================== #
    # I/O
    # =========================================================================== #
    def get_pixel_value(self, skycoo, file_index, hdu_index):
        """
        Fetches the pixel value directly from image based on coordinates.

        Parameters
        ----------
        skycoo : SkyCoord
            Input astropy SkyCoord object for which the aperture correction should be obtained.
        file_index : int
            Index of file in self.
        hdu_index : int
            Index of HDU

        Returns
        -------
        ndarray
            Array with pixel values

        """

        # Get data and header for given file and HDU
        data, header = fits.getdata(filename=self.full_paths[file_index], header=True, ext=hdu_index)

        # Read pixel coordinate
        return get_value_image(ra=skycoo.icrs.ra.deg, dec=skycoo.icrs.dec.deg, data=data, header=header)

    def hdu2arrays(self, hdu_index=0):
        """
        Reads data from a given HDU from all files in self into a list of arrays.

        Parameters
        ----------
        hdu_index : int, optional
            HDU index to read

        Returns
        -------
        iterable
            List of arrays

        """
        arraylist = []
        for path in self:
            with fits.open(path) as hdulist:
                arraylist.append(hdulist[hdu_index].data)
        return arraylist

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
        cube = np.empty((self.n_files,
                         self.headers_data[0][hdu_index]["NAXIS2"],
                         self.headers_data[0][hdu_index]["NAXIS1"]), dtype=dtype)

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

    def get_unique_master_flats(self):
        """ Returns unique Master Flats as MasterFlat instance. """
        from vircampype.fits.images.flat import MasterFlat
        return MasterFlat(setup=self.setup, file_paths=list(set(self.get_master_flat().full_paths)))

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

    def get_unique_master_weights(self):
        """ Returns unique MasterWeights as MasterWeights instance. """
        from vircampype.fits.images.flat import MasterWeight
        return MasterWeight(setup=self.setup, file_paths=list(set(self.get_master_weight().full_paths)))

    def get_master_weight_coadd(self):
        """
        Get for each file in self the corresponding MasterWeightCoadd.

        Returns
        -------
        MasterWeightCoadd
            MasterWeightCoadd instance holding for each file in self the corresponding MasterWeightCoadd file.

        """

        # Match and return
        return self.match_filter(match_to=self.get_master_images().weight_coadd,
                                 max_lag=self.setup["master"]["max_lag_flat"])

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

    def get_master_superflat(self):
        """
        Get for all files in self the corresponding MasterSuperflat (split by minutes from setup).

        Returns
        -------
        MasterSuperflat
            MasterSuperflat instance holding for all files in self the corresponding MasterSuperflat images.

        """
        return self.match_filter(match_to=self.get_master_images().superflat,
                                 max_lag=self.setup["master"]["max_lag_superflat"] / 1440.)

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

    def get_master_gain(self):
        """
        Get for each file in self the corresponding MasterGain table.

        Returns
        -------
        MasterGain
            MasterGain instance holding for each file in self the corresponding Masterlinearity table.

        """

        # Match and return
        return self.match_mjd(match_to=self.get_master_tables().gain,
                              max_lag=self.setup["master"]["max_lag_gain"])

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
                raise ValueError(BColors.FAIL + "Could not find matching filter" + BColors.ENDC)

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
    def process_raw(self):
        """ Main science calibration method. All options are set in the setup. """
        # TODO: Write better docstring

        # import
        from vircampype.fits.images.sky import MasterSky
        from vircampype.fits.images.dark import MasterDark
        from vircampype.fits.images.flat import MasterFlat
        from vircampype.fits.tables.gain import MasterGain
        from vircampype.fits.images.bpm import MasterBadPixelMask
        from vircampype.fits.tables.linearity import MasterLinearity

        # Processing info
        tstart = message_mastercalibration(master_type="PROCESSING RAW", silent=self.setup["misc"]["silent"], right="")

        # Fetch the Masterfiles
        master_bpm = self.get_master_bpm()  # type: MasterBadPixelMask
        master_dark = self.get_master_dark()  # type: MasterDark
        master_flat = self.get_master_flat()  # type: MasterFlat
        master_gain = self.get_master_gain()  # type: MasterGain
        master_sky = self.get_master_sky()  # type: MasterSky
        master_linearity = self.get_master_linearity()  # type: MasterLinearity

        # Loop over files and apply calibration
        for idx in range(self.n_files):

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=self.paths_processed[idx], silent=self.setup["misc"]["silent"]):
                continue

            # Print processing info
            message_calibration(n_current=idx + 1, n_total=self.n_files, name=self.paths_processed[idx],
                                d_current=None, d_total=None, silent=self.setup["misc"]["silent"])

            # Read file into cube
            calib_cube = self.file2cube(file_index=idx, hdu_index=None, dtype=np.float32)

            # Get master calibration
            bpm = master_bpm.file2cube(file_index=idx, hdu_index=None, dtype=np.uint8)
            dark = master_dark.file2cube(file_index=idx, hdu_index=None, dtype=np.float32)
            flat = master_flat.file2cube(file_index=idx, hdu_index=None, dtype=np.float32)
            sky = master_sky.file2cube(file_index=idx, hdu_index=None, dtype=np.float32)
            lin = master_linearity.file2coeff(file_index=idx, hdu_index=None)

            # Add Gain, read noise, and saturation limit to headers
            for h, g, r, s in zip(self.headers_data[idx], master_gain.gain[idx],
                                  master_gain.rdnoise[idx], self.setup["data"]["saturate"]):
                h[self.setup["keywords"]["gain"]] = (np.round(g, decimals=3), "Gain (e-/ADU)")
                h[self.setup["keywords"]["rdnoise"]] = (np.round(r, decimals=3), "Read noise (e-)")
                h[self.setup["keywords"]["saturate"]] = (s, "Saturation limit (ADU)")

            # Do calibration
            calib_cube.process_raw(dark=dark, flat=flat, linearize=lin, sky=sky, norm_before=self.ndit_norm[idx])

            # Apply cosmetics
            if self.setup["cosmetics"]["mask_cosmics"]:
                calib_cube.mask_cosmics(bpm=bpm, gain=master_gain.gain[idx], readnoise=master_gain.rdnoise[idx],
                                        satlevel=self.setup["data"]["saturate"], sepmed=False, cleantype="medmask")
            if self.setup["cosmetics"]["interpolate_nan"]:
                calib_cube.interpolate_nan()
            if self.setup["cosmetics"]["destripe"]:
                calib_cube.destripe()

            # Add file info to main header
            phdr = self.headers_primary[idx].copy()
            phdr["BPMFILE"] = master_bpm.file_names[idx]
            phdr["DARKFILE"] = master_dark.file_names[idx]
            phdr["FLATFILE"] = master_flat.file_names[idx]
            phdr["SKYFILE"] = master_sky.file_names[idx]
            phdr["LINFILE"] = master_linearity.file_names[idx]

            # Write to disk
            calib_cube.write_mef(path=self.paths_processed[idx], prime_header=phdr,
                                 data_headers=self.headers_data[idx], dtype="float32")

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

        # Return new instance of calibrated images
        return self.__class__(setup=self.setup, file_paths=self.paths_processed)

    def apply_superflat(self):
        """ Applies superflat to (processed) images. """

        # Processing info
        tstart = message_mastercalibration(master_type="APPLYING SUPERFLAT",
                                           silent=self.setup["misc"]["silent"], right="")

        # Build paths for aheaders (need to be copied too for resampling)
        path_aheaders = [x.replace(".fits", ".ahead") for x in self.paths_superflatted]

        # Fetch superflat for each image in self
        superflats = self.get_master_superflat()

        # Loop over self and superflats
        for idx_file in range(len(self)):

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=self.paths_superflatted[idx_file], silent=self.setup["misc"]["silent"]):
                continue

            # Print processing info
            message_calibration(n_current=idx_file + 1, n_total=self.n_files, name=self.paths_superflatted[idx_file],
                                d_current=None, d_total=None, silent=self.setup["misc"]["silent"])

            # Read data
            cube_self = self.file2cube(file_index=idx_file)
            cube_flat = superflats.file2cube(file_index=idx_file)

            # Determine cube background
            background = cube_self.background(mesh_size=256)[0]

            # Apply background
            cube_self -= background

            # Normalize
            cube_self /= cube_flat

            # Add background back in
            cube_self += background

            # Write back to disk
            cube_self.write_mef(self.paths_superflatted[idx_file], prime_header=self.headers_primary[idx_file],
                                data_headers=self.headers_data[idx_file])

            # Copy aheader for swarping
            copy_file(self._paths_aheaders[idx_file], path_aheaders[idx_file])

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

        # Return new instance of calibrated images
        return self.__class__(setup=self.setup, file_paths=self.paths_superflatted)

    # =========================================================================== #
    # Sextractor
    # =========================================================================== #
    @property
    def _bin_sex(self):
        return which(self.setup["astromatic"]["bin_sex"])

    @property
    def _sex_default_config(self):
        """
        Searches for default config file in resources.

        Returns
        -------
        str
            Path to default config

        """
        return get_resource_path(package="vircampype.resources.astromatic.sextractor", resource="default.config")

    @property
    def _sex_preset_package(self):
        """
        Internal package preset path for sextractor.

        Returns
        -------
        str
            Package path.
        """

        return "vircampype.resources.astromatic.presets"

    @property
    def _sex_default_filter(self):
        """
        Path for default convolution filter.

        Returns
        -------
        str
            Path to file.
        """

        return get_resource_path(package="vircampype.resources.astromatic.sextractor", resource="gauss_2.5_5x5.conv")

    @property
    def _sex_default_nnw(self):
        """
        Path for default nnw file.

        Returns
        -------
        str
            Path to file.
        """

        return get_resource_path(package="vircampype.resources.astromatic.sextractor", resource="default.nnw")

    def _sex_paths_tables(self, prefix=""):
        """
        Path to sextractor tables for files in instance.

        Returns
        -------
        iterable
            List with table names.
        """

        if prefix is None:
            prefix = ""

        return [x.replace(".fits", ".{0}.sources.fits".format(prefix)).replace("..", ".") for x in self.full_paths]

    def _sex_path_param(self, preset):
        """
        Returns path to sextractor param file, given preset.

        Parameters
        ----------
        preset : str
            Which preset to use

        Returns
        -------
        str
            Path to preset param.
        """

        if preset == "scamp":
            return get_resource_path(package=self._sex_preset_package, resource="sextractor_scamp.param")
        elif preset == "full":
            return get_resource_path(package=self._sex_preset_package, resource="sextractor_full.param")
        elif preset == "superflat":
            return get_resource_path(package=self._sex_preset_package, resource="sextractor_superflat.param")
        else:
            raise ValueError("Sextractor parameter name '{0}' not supported".format(preset))

    def sextractor(self, preset="scamp", prefix=None, **kwargs):
        """
        Runs sextractor based on given presets.

        Parameters
        ----------
        preset : str
            Preset name.
        prefix : str, optional
            Prefix to be used for catalogs.

        Returns
        -------
        SextractorCatalogs
            SextractorCatalog instance with the generated catalogs.

        """

        # Processing info
        tstart = message_mastercalibration(master_type="SOURCE DETECTION", silent=self.setup["misc"]["silent"],
                                           left="Running Sextractor with preset '{0}' on {1} files"
                                                "".format(preset, len(self)), right=None)

        # Check for existing files
        path_tables_clean = []
        if not self.setup["misc"]["overwrite"]:
            for pt in self._sex_paths_tables(prefix=prefix):
                check_file_exists(file_path=pt, silent=self.setup["misc"]["silent"])
                if not os.path.isfile(pt):
                    path_tables_clean.append(pt)

        # Look for local weights
        master_weight_paths = [x.replace(".fits", ".weight.fits") for x in self.full_paths]
        if sum([os.path.isfile(x) for x in master_weight_paths]) != len(self):
            master_weight_paths = self.get_master_weight().full_paths

        # Fetch param file
        if preset == "scamp":
            ss = yml2config(path=get_resource_path(package=self._sex_preset_package, resource="sextractor_scamp.yml"),
                            filter_name=self._sex_default_filter, parameters_name=self._sex_path_param(preset=preset),
                            satur_key=self.setup["keywords"]["saturate"], gain_key=self.setup["keywords"]["gain"],
                            skip=["catalog_name", "weight_image"])
        elif preset == "full":

            ss = yml2config(path=get_resource_path(package=self._sex_preset_package, resource="sextractor_full.yml"),
                            filter_name=self._sex_default_filter, parameters_name=self._sex_path_param(preset=preset),
                            phot_apertures=list2str(apertures_all, sep=","),
                            satur_key=self.setup["keywords"]["saturate"], gain_key=self.setup["keywords"]["gain"],
                            skip=["catalog_name", "weight_image", "starnnw_name"] + list(kwargs.keys()))
        elif preset == "superflat":
            ss = yml2config(path=get_resource_path(package=self._sex_preset_package,
                                                   resource="sextractor_superflat.yml"),
                            filter_name=self._sex_default_filter, parameters_name=self._sex_path_param(preset=preset),
                            satur_key=self.setup["keywords"]["saturate"], gain_key=self.setup["keywords"]["gain"],
                            skip=["catalog_name", "weight_image", "starnnw_name"] + list(kwargs.keys()))
        else:
            raise ValueError("Preset '{0}' not supported".format(preset))

        # Construct commands for source extraction
        cmds = ["{0} -c {1} {2} -STARNNW_NAME {3} -CATALOG_NAME {4} -WEIGHT_IMAGE {5} {6}"
                "".format(self._bin_sex, self._sex_default_config, image, self._sex_default_nnw, catalog, weight, ss)
                for image, catalog, weight in zip(self.full_paths, path_tables_clean, master_weight_paths)]

        # Add kwargs
        for key, val in kwargs.items():
            for cmd_idx in range(len(cmds)):
                cmds[cmd_idx] += "-{0} {1}".format(key.upper(), val[cmd_idx])

        # Run Sextractor
        run_cmds(cmds=cmds, silent=True, n_processes=self.setup["misc"]["n_threads_shell"])

        # Add some keywords to primary header
        for cat, img in zip(path_tables_clean, self.full_paths):
            copy_keywords(path_1=cat, path_2=img, hdu_1=0, hdu_2=0,
                          keywords=[self.setup["keywords"]["object"], self.setup["keywords"]["filter"]])

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

        # Return Table instance
        return SextractorCatalogs(setup=self.setup, file_paths=self._sex_paths_tables(prefix=prefix))

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
        if ("sky" in basename.lower()) or ("photometry" in basename.lower()):
            path = self.path_master_object + basename
        else:
            path = self.path_master_common + basename

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

    @property
    def weight_coadd(self):
        """
        Retrieves all MasterWeightCoadd images.

        Returns
        -------
        MasterWeightCoadd
            All MasterWeightCoadd images as a MasterSky instance.

        """

        # Import
        from vircampype.fits.images.flat import MasterWeightCoadd

        # Get the masterbpm files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-WEIGHT-COADD"]

        return MasterWeightCoadd(setup=self.setup, file_paths=[self.file_paths[idx] for idx in index])

    @property
    def superflat(self):
        """
        Holds all MasterSuperflat images.

        Returns
        -------
        MasterSuperflat
            All MasterSuperflat images.

        """

        # Import
        from vircampype.fits.images.flat import MasterSuperflat

        # Get the masterbpm files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-SUPERFLAT"]

        return MasterSuperflat(setup=self.setup, file_paths=[self.file_paths[idx] for idx in index])
