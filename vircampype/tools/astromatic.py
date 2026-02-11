import os

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
            fhc = [
                item.decode("UTF-8") if isinstance(item, bytes) else item
                for item in fhc
            ]
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
        # Skip if empty or only whitespace
        if not hdu.strip():
            continue

        # Create cards
        cards = [fits.Card.fromstring(x) for x in hdu.split("\n")]

        # Verify cards
        for c in cards:
            c.verify(option="silentfix+ignore")

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
    _package_name = ""

    def __init__(self, setup):
        from vircampype.pipeline.main import Setup

        self.setup = Setup.load_pipeline_setup(setup)
        if not os.path.isfile(self.bin):
            raise ValueError(f"Cannot find executable '{self.bin_name}'")

    @property
    def bin(self):
        return which(self.bin_name)

    @property
    def bin_name(self):
        return ""

    @property
    def package(self):
        return f"vircampype.resources.astromatic.{self._package_name}"

    @property
    def package_presets(self):
        return f"{self.package}.presets"

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


class SextractorSetup(AstromaticSetup):
    _package_name = "sextractor"

    def __init__(self, setup):
        super().__init__(setup=setup)

    @property
    def bin_name(self):
        return self.setup.bin_sex

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
            package=self.package_presets, resource=f"{preset}.yml"
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
            package=self.package_presets, resource=f"{preset}.param"
        )


class SwarpSetup(AstromaticSetup):
    _package_name = "swarp"

    def __init__(self, setup):
        super().__init__(setup=setup)

    @property
    def bin_name(self):
        return self.setup.bin_swarp

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
    _package_name = "scamp"

    def __init__(self, setup):
        super().__init__(setup=setup)

    @property
    def bin_name(self):
        return self.setup.bin_scamp

    @property
    def preset_config(self) -> str:
        """
        Returns path to the scamp preset config based on scamp_mode.

        Returns
        -------
        str
            Path to preset config.

        """
        if self.setup.scamp_mode == "loose":
            return self._path_config_loose
        elif self.setup.scamp_mode == "fix_focalplane":
            return self._path_config_fix_focalplane
        else:
            raise ValueError(f"Scamp mode '{self.setup.scamp_mode}' not supported")

    @property
    def _path_config_loose(self) -> str:
        """
        Searches for loose config file in resources.

        Returns
        -------
        str
            Path to loose config.

        """
        return get_resource_path(package=self.package_presets,
                                 resource="scamp_loose.yml")

    @property
    def _path_config_fix_focalplane(self) -> str:
        """
        Searches for fix focalplane config file in resources.

        Returns
        -------
        str
            Path to fix focalplane config.

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
            f"{self.setup.folders['qc_astrometry']}scamp_{qt.lower()}"
            for qt in self.qc_types(joined=False)
        ]
        if joined:
            return ",".join(names)
        else:
            return names


class PSFExSetup(AstromaticSetup):
    _package_name = "psfex"

    def __init__(self, setup):
        super().__init__(setup=setup)

    @property
    def bin_name(self):
        return self.setup.bin_psfex

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
            package=self.package_presets, resource=f"{preset}.yml"
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
            f"{self.setup.folders['qc_psf']}{ct.lower()}"
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
            f"{self.setup.folders['qc_psf']}{qt.lower()}"
            for qt in self.checkimage_types(joined=False)
        ]
        if joined:
            return ",".join(names)
        else:
            return names