import numpy as np
from vircampype.utils.system import *


class Setup(dict):
    """ Pipeline setup. """

    def __init__(self, *arg, **kw):

        # Initialize dict
        super(Setup, self).__init__(*arg, **kw)

        # Add folder structure to self
        self.__add_folder_tree()

        # Add folder attribute
        self.folders = self["folders"]

        # Make folder structure
        self.__create_folder_tree()

        # =========================================================================== #
        # Set default attribute values

        # Data setup
        self.__purge_headers = True
        self.__reset_wcs = True

        # Superflat
        self.__superflat_window = 60
        self.__superflat_n_min = 5

        # Astromatic
        self.__bin_sex = "sex"
        self.__bin_scamp = "scamp"
        self.__bin_swarp = "swarp"
        self.__swarp_back_size = 128
        self.__sex_back_size = 64
        self.__sex_back_filtersize = 3

        # Try to override property from setup
        for key, val in self.items():
            setattr(self, key, val)

    @property
    def path_tile(self):
        return "{0}{1}.fits".format(self.folders["tile"], self["name"])

    @property
    def path_tile_weight(self):
        return self.path_tile.replace(".fits", ".weight.fits")

    @property
    def path_tile_header(self):
        return self.path_tile.replace(".fits", ".ahead")

    def __add_folder_tree(self):
        """ Adds pipeline folder tree to setup. """

        # Start adding folder structure to setup
        self["folders"] = dict()
        self["folders"]["pype"] = self["path_pype"]
        self["folders"]["raw"] = self["path_data"]
        self["folders"]["object"] = "{0}{1}/".format(self["path_pype"], self["name"])
        self["folders"]["headers"] = "{0}{1}/".format(self["folders"]["object"], "headers")

        # Master paths
        self["folders"]["master_common"] = "{0}{1}/".format(self["path_pype"], "master")
        self["folders"]["master_object"] = "{0}{1}/".format(self["folders"]["object"], "master")

        # Processing folders
        self["folders"]["temp"] = "{0}{1}/".format(self["folders"]["object"], "temp")
        self["folders"]["processed"] = "{0}{1}/".format(self["folders"]["object"], "processed")
        self["folders"]["superflat"] = "{0}{1}/".format(self["folders"]["object"], "superflatted")
        self["folders"]["resampled"] = "{0}{1}/".format(self["folders"]["object"], "resampled")

        # QC
        self["folders"]["qc"] = "{0}{1}/".format(self["folders"]["object"], "qc")

        # Common QC
        self["folders"]["qc_bpm"] = "{0}{1}/".format(self["folders"]["qc"], "bpm")
        self["folders"]["qc_dark"] = "{0}{1}/".format(self["folders"]["qc"], "dark")
        self["folders"]["qc_gain"] = "{0}{1}/".format(self["folders"]["qc"], "gain")
        self["folders"]["qc_linearity"] = "{0}{1}/".format(self["folders"]["qc"], "linearity")
        self["folders"]["qc_flat"] = "{0}{1}/".format(self["folders"]["qc"], "flat")

        # Sequence specific QC
        self["folders"]["qc_sky"] = "{0}{1}/".format(self["folders"]["qc"], "sky")
        self["folders"]["qc_astrometry"] = "{0}{1}/".format(self["folders"]["qc"], "astrometry")
        self["folders"]["qc_superflat"] = "{0}{1}/".format(self["folders"]["qc"], "superflat")
        self["folders"]["qc_photometry"] = "{0}{1}/".format(self["folders"]["qc"], "photometry")

        # Tile paths
        self["folders"]["tile"] = "{0}{1}/".format(self["folders"]["object"], "tile")

        # Phase 3
        self["folders"]["phase3"] = "{0}{1}{2}/".format(self["path_pype"], "phase3/", self["name"])

    def __create_folder_tree(self):
        """ Creates the folder tree for the pipeline"""

        # Common paths
        folders_common = [self["folders"]["pype"], self["folders"]["headers"], self["folders"]["phase3"],
                          self["folders"]["master_common"], self["folders"]["temp"]]

        # calibration-specific paths
        folders_cal = [self["folders"]["qc_bpm"], self["folders"]["qc_dark"], self["folders"]["qc_gain"],
                       self["folders"]["qc_linearity"], self["folders"]["qc_flat"]]

        # Object-specific paths
        folders_object = [self["folders"]["master_object"], self["folders"]["qc"], self["folders"]["qc_sky"],
                          self["folders"]["processed"], self["folders"]["qc_astrometry"], self["folders"]["superflat"],
                          self["folders"]["qc_superflat"], self["folders"]["resampled"], self["folders"]["tile"],
                          self["folders"]["qc_photometry"]]

        # Generate common paths
        for path in folders_common:
            make_folder(path)

        # Create common calibration path only if we run a calibration unit
        if "calibration" in self["name"].lower():
            for path in folders_cal:
                make_folder(path=path)

        # Other wise make object paths
        else:
            for path in folders_object:
                make_folder(path=path)

    @classmethod
    def load_pipeline_setup(cls, setup):
        """
        Load a setup instance.

        Parameters
        ----------
        setup : str, Setup
            Either a string pointing to the location of a pipeline YML, or a Setup instance

        Returns
        -------
        Setup
            Setup instance.

        """

        # If given as string, load YML
        if isinstance(setup, str):
            return cls(read_yml(path_yml=setup))

        # If given as Setup instance, just return it again
        elif isinstance(setup, cls):
            return setup

        # If something else was provided
        else:
            raise ValueError("Please provide a pipeline setup")

    # =========================================================================== #
    # Hard-coded pipeline setup
    # =========================================================================== #
    # Data setup
    @property
    def saturation_levels(self):
        """ Saturation levels for each detector. """
        return [33000, 32000, 33000, 32000,
                24000, 24000, 35000, 33000,
                35000, 35000, 37000, 34000,
                33000, 35000, 34000, 34000]

    @property
    def fpa_layout(self):
        """ Focal plane array layout. """
        return [4, 4]

    @property
    def fix_vircam_header(self):
        """ Whether the (sometimes silly) VIRCAM  headers should be fixed. """
        return True

    @property
    def reset_wcs(self):
        """
        Whether the WCS info in the original headers should be reset.
        This is necessary because sometimes the data flow does not propagate WCS keywords to all images.
        """
        return self.__reset_wcs

    @reset_wcs.setter
    def reset_wcs(self, reset_wcs):
        self.__reset_wcs = reset_wcs

    @property
    def purge_headers(self):
        return self.__purge_headers

    @purge_headers.setter
    def purge_headers(self, purge_headers):
        self.__purge_headers = purge_headers

    # =========================================================================== #
    # Keywords
    @property
    def keywords(self):
        return HeaderKeywords()

    # =========================================================================== #
    # Master
    @property
    def max_lag_bpm(self):
        """ Maximum time difference to MasterBPM in days. """
        return 10

    @property
    def max_lag_dark(self):
        """ Maximum time difference to MasterDark in days. """
        return 10

    @property
    def max_lag_flat(self):
        """ Maximum time difference to MasterFlat in days. """
        return 10

    @property
    def max_lag_superflat(self):
        """ Maximum time difference to MasterSuperflat in minutes. """
        return 60

    @property
    def max_lag_sky(self):
        """ Maximum time difference to MasterSky in minutes. """
        return 60

    @property
    def max_lag_gain(self):
        """ Maximum time difference to MasterGain in days. """
        return 10

    @property
    def max_lag_weight(self):
        """ Maximum time difference to MasterWeight in days. """
        return 10

    @property
    def max_lag_linearity(self):
        """ Maximum time difference to MasterLinearity in days. """
        return 10

    # =========================================================================== #
    # Photometry
    @property
    def apertures(self):
        """
        Aperture diameters used in photometric calibration.
        All apertures will be matched to the last in the list.
        """
        return [3.0, 4.0, 5.0, 6.0, 9.0,
                12.0, 15.0, 18.0, 21.0, 24.0]

    @property
    def target_zp(self):
        """ All data will be scaled to approximately match this zero-point """
        return 25

    @property
    def reference_catalog(self):
        """ Reference catalog to be used. """
        return "2mass"

    # =========================================================================== #
    # Cosmetics
    @property
    def destripe(self):
        """ Whether destriping should be done. """
        return True

    @property
    def destripe_axis(self):
        """ Axis along which to destripe. """
        return 2

    @property
    def destripe_metric(self):
        """ Metric used in destriping. """
        return "median"

    @property
    def interpolate_nan_bool(self):
        """ Whether to interpolate NaN values. """
        return True

    @property
    def interpolate_max_bad_neighbors(self):
        """ How many bad neighbors a NaN can have so that it still gets interpolated. """
        return 3

    # =========================================================================== #
    # Source masks
    @property
    def mask_sources_thresh(self):
        """ Significance threshold above background for pixels to be masked. """
        return 3

    @property
    def mask_sources_min_area(self):
        """ Minimum area of sources that are masked. """
        return 3

    @property
    def mask_sources_max_area(self):
        """ Maximum area of sources that are masked (500x500 pix). """
        return 250000

    # =========================================================================== #
    # Sky
    @property
    def sky_mix_science(self):
        return True

    @property
    def sky_window(self):
        """ Window in minutes around which sky images are created. """
        return 180

    @property
    def sky_n_min(self):
        """ Minimum number of images to merge to an offset image. """
        return 5

    @property
    def sky_mask_min(self):
        return False

    @property
    def sky_mask_max(self):
        return True

    @property
    def sky_sigma_level(self):
        return 3

    @property
    def sky_sigma_iter(self):
        return 1

    @property
    def sky_background_mesh_size(self):
        return 256

    @property
    def sky_background_mesh_filter_size(self):
        return 3

    @property
    def sky_metric(self):
        return "median"

    # =========================================================================== #
    # Superflat
    @property
    def superflat_window(self):
        """ Superflat window in minutes centered on each image. """
        return self.__superflat_window

    @superflat_window.setter
    def superflat_window(self, superflat_window):
        self.__superflat_window = superflat_window

    @property
    def superflat_n_min(self):
        """ Minimum number of files that need to go into a superflat. """
        return self.__superflat_n_min

    @superflat_n_min.setter
    def superflat_n_min(self, superflat_n_min):
        self.__superflat_n_min = superflat_n_min

    # =========================================================================== #
    # Astromatic
    @property
    def bin_sex(self):
        return self.__bin_sex

    @bin_sex.setter
    def bin_sex(self, bin_sex):
        self.__bin_sex = bin_sex

    @property
    def bin_scamp(self):
        return self.__bin_scamp

    @bin_scamp.setter
    def bin_scamp(self, bin_scamp):
        self.__bin_scamp = bin_scamp

    @property
    def bin_swarp(self):
        return self.__bin_swarp

    @bin_swarp.setter
    def bin_swarp(self, bin_swarp):
        self.__bin_swarp = bin_swarp

    @property
    def swarp_back_size(self):
        return self.__swarp_back_size

    @swarp_back_size.setter
    def swarp_back_size(self, swarp_back_size):
        self.__swarp_back_size = swarp_back_size

    @property
    def sex_back_size(self):
        return self.__sex_back_size

    @sex_back_size.setter
    def sex_back_size(self, sex_back_size):
        self.__sex_back_size = sex_back_size

    @property
    def sex_back_filtersize(self):
        return self.__sex_back_filtersize

    @sex_back_filtersize.setter
    def sex_back_filtersize(self, sex_back_filtersize):
        self.__sex_back_filtersize = sex_back_filtersize

    # =========================================================================== #
    # Other
    @property
    def seeing_test_range(self):
        return np.around(np.arange(0.5, 1.36, 0.05), decimals=2)

    @property
    def pixel_scale_arcsec(self):
        return 1/3

    @property
    def pixel_scale_degrees(self):
        return self.pixel_scale_arcsec / 3600.

    @property
    def silent(self):
        """ Whether the pipeline should operatre in silent mode. """
        return False

    @property
    def overwrite(self):
        """ Whether the pipeline overwrites existing files. """
        return False

    @property
    def qc_plots(self):
        """ Whether QC plots should be generated. """
        return True


class HeaderKeywords:

    def __init__(self, obj="OBJECT", filter_name="HIERARCH ESO INS FILT1 NAME", dpr_type="HIERARCH ESO DPR TYPE",
                 dpr_category="HIERARCH ESO DPR CATG", dit="HIERARCH ESO DET DIT", ndit="HIERARCH ESO DET NDIT",
                 date_mjd="MJD-OBS", date_ut="DATE-OBS", gain="GAIN", rdnoise="RDNOISE", saturate="SATURATE"):

        self.object = obj
        self.filter_name = filter_name
        self.type = dpr_type
        self.category = dpr_category
        self.dit = dit
        self.ndit = ndit
        self.date_mjd = date_mjd
        self.date_ut = date_ut
        self.gain = gain
        self.rdnoise = rdnoise
        self.saturate = saturate
