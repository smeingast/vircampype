import numpy as np

from astropy.io import fits
from collections.abc import Iterable
from vircampype.tools.systemtools import *

__all__ = [
    "sextractor2imagehdr",
    "read_aheaders",
    "write_aheaders",
    "SextractorSetup",
    "SwarpSetup",
    "ScampSetup",
    "PSFExSetup",
]


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
        headers = []
        for i in range(1, len(hdulist), 2):
            fhc = hdulist[i].data["Field Header Card"][0]
            fhc = np.array(
                [
                    item.decode("UTF-8") if isinstance(item, bytes) else item
                    for item in fhc
                ]
            )
            header_string = "\n".join(fhc)
            header = fits.Header.fromstring(header_string, sep="\n")
            headers.append(header)

    # Convert to headers and return
    return headers


def read_aheaders(path: str):
    """
    Reads aheader files created by scamp into FITS header instances.

    Parameters
    ----------
    path : str
        Path to aheader file

    Returns
    -------
    list
        List containing each extension as a fits Header.

    """

    # Create empty list of output headers
    headers = []

    # Open file
    with open(path, "r") as file:
        # Read textfile
        data = file.read()

    # Split into HDUs
    hdus = data.split("END")

    # Loop over HDUs
    for hdu in hdus:
        # Skip if only whitespace
        if hdu.isspace():
            continue

        # Create cards
        cards = [fits.Card.fromstring(x) for x in hdu.split("\n")]

        # Verify cards
        [c.verify(option="silentfix+ignore") for c in cards]

        # Clean cards
        cards = [c for c in cards if not c.is_blank]

        # Create header from cards
        headers.append(fits.Header(cards))

    # Return list of headers
    return headers


def write_aheaders(headers: Iterable, path: str):
    """
    Write a list of FITS headers into the astromatic aheader format.

    Parameters
    ----------
    headers : Iterable
        List of headers.
    path : str
        File path to write to.

    """

    # Convert to strings
    headers_str = [h.tostring(endcard=True, padding=False, sep="\n") for h in headers]

    # Join all extensions together
    ss = "\n".join(headers_str)

    # Write to disk
    with open(path, "w") as file:
        file.write(ss)


class AstromaticSetup:
    def __init__(self, setup):
        from vircampype.pipeline.main import Setup

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

        return get_resource_path(package=self.package, resource="gauss_2.0_5x5.conv")

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

        return get_resource_path(
            package=self.package_presets, resource="{0}.yml".format(preset)
        )

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

        return get_resource_path(
            package=self.package_presets, resource="{0}.param".format(preset)
        )


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
        Returns resample suffix.

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
        return get_resource_path(
            package=self.package_presets, resource="resampling.yml"
        )

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

        return f"{self.package}.presets"

    @property
    def path_config_default(self) -> str:
        """
        Searches for default config file in resources.

        Returns
        -------
        str
            Path to default config

        """
        return get_resource_path(package=self.package, resource="default.config")

    @property
    def path_config_preset(self) -> str:
        """
        Searches for default config file in resources.

        Returns
        -------
        str
            Path to default config

        """
        if self.setup.scamp_mode == "loose":
            return self.__path_config_loose
        elif self.setup.scamp_mode == "fix_focalplane":
            return self.__path_config_fix_focalplane
        else:
            raise ValueError(f"Scamp mode '{self.setup.scamp_mode}' not supported")

    @property
    def __path_config_loose(self) -> str:
        """
        Searches for loose config file in resources.

        Returns
        -------
        str
            Path to default config

        """
        return get_resource_path(package=self.package_presets,
                                 resource="scamp_loose.yml")

    @property
    def __path_config_fix_focalplane(self) -> str:
        """
        Searches for fix focalplane config file in resources.

        Returns
        -------
        str
            Path to default config

        """
        return get_resource_path(package=self.package_presets,
                                 resource="scamp_ffp.yml")

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
        types = [
            "SKY_ALL",
            "FGROUPS",
            "DISTORTION",
            "ASTR_INTERROR1D",
            "ASTR_INTERROR2D",
            "ASTR_REFERROR1D",
            "ASTR_REFERROR2D",
            "ASTR_PIXERROR1D",
            "ASTR_SUBPIXERROR1D",
            "ASTR_CHI2",
            "ASTR_REFSYSMAP",
            "ASTR_XPIXERROR2D",
            "ASTR_YPIXERROR2D",
            "SHEAR_VS_AIRMASS",
        ]
        # "PHOT_ERROR", "PHOT_ERRORVSMAG", "PHOT_ZPCORR", "PHOT_ZPCORR3D"]
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
        names = [
            "{0}scamp_{1}".format(self.setup.folders["qc_astrometry"], qt.lower())
            for qt in self.qc_types(joined=False)
        ]
        if joined:
            return ",".join(names)
        else:
            return names


class PSFExSetup(AstromaticSetup):
    def __init__(self, setup):
        super(PSFExSetup, self).__init__(setup=setup)

    @property
    def bin_name(self):
        return self.setup.bin_psfex

    @property
    def package(self):
        """
        Internal package preset path for psfex.

        Returns
        -------
        str
            Package path.
        """

        return "vircampype.resources.astromatic.psfex"

    @property
    def package_presets(self):
        """
        Internal package preset path for psfex.

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

    def path_yml(self, preset):
        """
        Returns path to PSFEx yml file, given preset.

        Parameters
        ----------
        preset : str
            Which preset to use.

        Returns
        -------
        str
            Path to preset yml.
        """

        return get_resource_path(
            package=self.package_presets, resource="{0}.yml".format(preset)
        )

    @staticmethod
    def checkplot_types(joined=False):
        """
        QC check plot types for psfex.

        Parameters
        ----------
        joined : bool, optional
            If set, list will be joined by ',' to make it readable for psfex.

        Returns
        -------
        iterable, str
            List or str with QC checkplot types.

        """
        types = [
            "SELECTION_FWHM",
            "FWHM",
            "ELLIPTICITY",
            "COUNTS",
            "COUNT_FRACTION",
            "CHI2",
            "RESIDUALS",
        ]
        if joined:
            return ",".join(types)
        else:
            return types

    def checkplot_names(self, joined=False):
        """
        List or str containing psfex QC plot names.

        Parameters
        ----------
        joined : bool, optional
            If set, list will be joined by ',' to make it readable for psfex.

        Returns
        -------
        iterable, str
            List or str with QC checkplot names.

        """
        names = [
            "{0}{1}".format(self.setup.folders["qc_psf"], ct.lower())
            for ct in self.checkplot_types(joined=False)
        ]
        if joined:
            return ",".join(names)
        else:
            return names

    @staticmethod
    def checkimage_types(joined=False):
        """
        QC check image types for psfex.

        Parameters
        ----------
        joined : bool, optional
            If set, list will be joined by ',' to make it readable for psfex.

        Returns
        -------
        iterable, str
            List or str with QC checkimage types.

        """
        types = ["SNAPSHOTS_IMRES", "SAMPLES", "RESIDUALS"]
        if joined:
            return ",".join(types)
        else:
            return types

    def checkimage_names(self, joined=False):
        """
        List or str containing psfex check image names.

        Parameters
        ----------
        joined : bool, optional
            If set, list will be joined by ',' to make it readable for psfex.

        Returns
        -------
        iterable, str
            List or str with QC check image names.

        """
        names = [
            "{0}{1}".format(self.setup.folders["qc_psf"], qt.lower())
            for qt in self.checkimage_types(joined=False)
        ]
        if joined:
            return ",".join(names)
        else:
            return names
