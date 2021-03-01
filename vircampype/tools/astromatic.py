from astropy.io import fits
from vircampype.tools.systemtools import *
from vircampype.pipeline.main import Setup

__all__ = ["sextractor2imagehdr", "SextractorSetup", "SwarpSetup", "ScampSetup"]


def sextractor2imagehdr(path):
    """
    Obtains image headers from sextractor catalogs.

    Parameters
    ----------
    path : str
        Path to Sextractor FITS table.

    Returns
    -------
    iterable
        List of image headers found in file.

    """

    # Read image headers into tables
    with fits.open(path) as hdulist:
        headers = [fits.Header.fromstring("\n".join(hdulist[i].data["Field Header Card"][0]), sep="\n")
                   for i in range(1, len(hdulist), 2)]

    # Convert to headers and return
    return headers


class AstromaticSetup:
    def __init__(self, setup):
        self.setup = Setup.load_pipeline_setup(setup)
        if self.bin is None:
            raise ValueError("Cannot find executable '{0}'".format(self.bin_name))

    @property
    def bin(self):
        return which(self.bin_name)

    @property
    def bin_name(self):
        return ""


class SextractorSetup(AstromaticSetup):

    def __init__(self, setup):
        super(SextractorSetup, self).__init__(setup=setup)

    @property
    def bin_name(self):
        return self.setup.bin_sex

    @property
    def package(self):
        """
        Internal package preset path for sextractor.

        Returns
        -------
        str
            Package path.
        """

        return "vircampype.resources.astromatic.sextractor"

    @property
    def package_presets(self):
        """
        Internal package preset path for sextractor.

        Returns
        -------
        str
            Package path.
        """

        return "{0}.presets".format(self.package)

    @property
    def default_config(self):
        """
        Searches for default config file in resources.

        Returns
        -------
        str
            Path to default config

        """
        return get_resource_path(package=self.package, resource="default.config")

    @property
    def default_filter(self):
        """
        Path for default convolution filter.

        Returns
        -------
        str
            Path to file.
        """

        return get_resource_path(package=self.package, resource="gauss_2.5_5x5.conv")

    @property
    def default_nnw(self):
        """
        Path for default nnw file.

        Returns
        -------
        str
            Path to file.
        """

        return get_resource_path(package=self.package, resource="default.nnw")

    def path_yml(self, preset):
        """
        Returns path to sextractor yml file, given preset.

        Parameters
        ----------
        preset : str
            Which preset to use.

        Returns
        -------
        str
            Path to preset yml.
        """

        return get_resource_path(package=self.package_presets, resource="{0}.yml".format(preset))

    def path_param(self, preset):
        """
        Returns path to sextractor param file, given preset.

        Parameters
        ----------
        preset : str
            Which preset to use.

        Returns
        -------
        str
            Path to preset param.
        """

        return get_resource_path(package=self.package_presets, resource="{0}.param".format(preset))


class SwarpSetup(AstromaticSetup):

    def __init__(self, setup):
        super(SwarpSetup, self).__init__(setup=setup)

    @property
    def bin_name(self):
        return self.setup.bin_swarp

    @property
    def package(self):
        """
        Internal package path for swarp.

        Returns
        -------
        str
            Package path.
        """

        return "vircampype.resources.astromatic.swarp"

    @property
    def package_presets(self):
        """
        Internal package preset path for swarp.

        Returns
        -------
        str
            Package path.
        """

        return "{0}.presets".format(self.package)

    @property
    def default_config(self):
        """
        Searches for default config file in resources.

        Returns
        -------
        str
            Path to default config.

        """
        return get_resource_path(package=self.package, resource="default.config")

    @property
    def resample_suffix(self):
        """
        Returns resample suffix.Y

        Returns
        -------
        str
            Resample suffix.
        """
        return ".resamp.fits"

    @property
    def preset_resampling(self):
        """
        Obtains path to resampling preset for swarp.

        Returns
        -------
        str
            Path to preset.
        """
        return get_resource_path(package=self.package_presets, resource="resampling.yml")

    @property
    def preset_coadd(self):
        """
        Obtains path to coadd preset for swarp.

        Returns
        -------
        str
            Path to preset.
        """
        return get_resource_path(package=self.package_presets, resource="coadd.yml")


class ScampSetup(AstromaticSetup):

    def __init__(self, setup):
        super(ScampSetup, self).__init__(setup=setup)

    @property
    def bin_name(self):
        return self.setup.bin_scamp

    @property
    def package(self):
        """
        Internal package preset path for scamp.

        Returns
        -------
        str
            Package path.
        """

        return "vircampype.resources.astromatic.scamp"

    @property
    def package_presets(self):
        """
        Internal package preset path for scamp.

        Returns
        -------
        str
            Package path.
        """

        return "{0}.presets".format(self.package)

    @property
    def default_config(self):
        """
        Searches for default config file in resources.

        Returns
        -------
        str
            Path to default config

        """
        return get_resource_path(package=self.package, resource="default.config")

    @staticmethod
    def qc_types(joined=False):
        """
        QC check plot types for scamp.

        Parameters
        ----------
        joined : bool, optional
            If set, list will be joined by ',' to make it readable for scamp.

        Returns
        -------
        iterable, str
            List or str with QC checkplot types.

        """
        types = ["SKY_ALL", "FGROUPS", "DISTORTION", "ASTR_INTERROR1D", "ASTR_INTERROR2D", "ASTR_REFERROR1D",
                 "ASTR_REFERROR2D", "ASTR_PIXERROR1D", "ASTR_SUBPIXERROR1D", "ASTR_CHI2", "ASTR_REFSYSMAP",
                 "ASTR_XPIXERROR2D", "ASTR_YPIXERROR2D", "SHEAR_VS_AIRMASS"]
        if joined:
            return ",".join(types)
        else:
            return types

    def qc_names(self, joined=False):
        """
        List or str containing scamp QC plot names.

        Parameters
        ----------
        joined : bool, optional
            If set, list will be joined by ',' to make it readable for scamp.

        Returns
        -------
        iterable, str
            List or str with QC checkplot types.

        """
        names = ["{0}scamp_{1}".format(self.setup.folders["qc_astrometry"], qt.lower()) for qt in
                 self.qc_types(joined=False)]
        if joined:
            return ",".join(names)
        else:
            return names
