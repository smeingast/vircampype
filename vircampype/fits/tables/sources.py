import numpy as np

from astropy.time import Time
from typing import Tuple, List
from astropy.coordinates import SkyCoord
from vircampype.fits.tables.common import FitsTables


class SourceCatalogs(FitsTables):
    def __init__(
        self,
        setup,
        file_paths=None,
        key_ra="RA",
        key_dec="DEC",
    ):
        super(SourceCatalogs, self).__init__(
            file_paths=file_paths,
            setup=setup,
        )
        self.key_ra = key_ra
        self.key_dec = key_dec

    @property
    def passband(self):
        """
        Grabs filter keyword from header and puts in into list.

        Returns
        -------
        iterable
            List of filters for all tables in instance.

        """
        return self.read_from_prime_headers(keywords=[self.setup.keywords.filter_name])[
            0
        ]

    _ra = None

    @property
    def ra(self) -> List[List[float]]:
        """
        Extracts all RA entries in tables.

        Returns
        -------
        List
            List of lists containing RAs

        """

        # Check if already determined
        if self._ra is not None:
            return self._ra

        # Retun columns
        self._ra = [[y for y in x] for x in self.get_columns(column_name=self.key_ra)]
        return self._ra

    _dec = None

    @property
    def dec(self) -> List[List[float]]:
        """
        Extracts all RA entries in tables.

        Returns
        -------
        List
            List of lists containing RAs

        """

        # Check if already determined
        if self._dec is not None:
            return self._dec

        # Get data from columns
        self._dec = [[y for y in x] for x in self.get_columns(column_name=self.key_dec)]
        return self._dec

    @property
    def skycoord(self):
        """
        Constructs SkyCoord object from ra/dec

        Returns
        -------
        iterable
            List of lists holding SkyCoord objects

        """

        skycoord_files = []
        for fra, fdec in zip(self.ra, self.dec):
            skycoord_ext = []
            for ra, dec in zip(fra, fdec):
                skycoord_ext.append(SkyCoord(ra=ra, dec=dec, frame="icrs", unit="deg"))
            skycoord_files.append(skycoord_ext)

        return skycoord_files


class MasterPhotometry(SourceCatalogs):
    def __init__(self, setup, file_paths=None, **kwargs):
        super(MasterPhotometry, self).__init__(
            file_paths=file_paths, setup=setup, **kwargs
        )


class MasterPhotometry2Mass(MasterPhotometry):
    def __init__(self, setup, file_paths=None):
        super(MasterPhotometry2Mass, self).__init__(
            file_paths=file_paths,
            setup=setup,
            key_ra="RAJ2000",
            key_dec="DEJ2000",
        )

    @staticmethod
    def _passband2idx(passband):
        """Helper to return filter index for flags."""
        if "j" in passband.lower():
            return 0
        elif "h" in passband.lower():
            return 1
        elif "k" in passband.lower():
            return 2
        else:
            raise ValueError("Filter '{0}' not defined".format(passband))

    __qflag = None

    @property
    def _qflags(self):
        """Reads quality flag column from catalog."""

        if self.__qflag is not None:
            return self.__qflag

        self.__qflag = self.get_columns(column_name="Qflg")
        return self.__qflag

    def qflags(self, passband):
        """Return Quality flag for given filter."""
        return [
            [[x[self._passband2idx(passband=passband)] for x in y] for y in z]
            for z in self._qflags
        ]

    __cflag = None

    @property
    def _cflags(self):
        """Reads contamintation and confusion flag column from catalog."""

        if self.__cflag is not None:
            return self.__cflag

        self.__cflag = self.get_columns(column_name="Cflg")
        return self.__cflag

    def cflags(self, passband):
        """Return contamination and confusion flag for given filter."""
        return [
            [[x[self._passband2idx(passband=passband)] for x in y] for y in z]
            for z in self._cflags
        ]

    @staticmethod
    def translate_passband(passband):
        """
        Translates input filter from e.g. VISTA to 2MASS names.

        Parameters
        ----------
        passband : str
            The input magnitude name. e.g. 'J'.

        Returns
        -------
        str
            Translated magnitude key. e.g. 'Ks' will be translated to 'Kmag'.
        """

        if "j" in passband.lower():
            return "Jmag"
        elif "h" in passband.lower():
            return "Hmag"
        elif "k" in passband.lower():
            return "Kmag"
        else:
            raise ValueError("Passband '{0}' not defined".format(passband))

    def mag_lim(self, passband: str) -> Tuple[float, float]:
        """
        Fetches magnitude limits in setup.

        Parameters
        ----------
        passband : str
            The passband for which to fetch the magnitude limits.

        Returns
        -------
        tuple
            Tuple with lower and upper magnitude limits.

        Raises
        ------
        ValueError
            If the passband is not available.

        """
        if None not in (self.setup.reference_mag_lo, self.setup.reference_mag_hi):
            return self.setup.reference_mag_lo, self.setup.reference_mag_hi

        # In case this is not given in the setup, default to standard values for bands
        if "j" in passband.lower():
            return 12.0, 15.5
        elif "h" in passband.lower():
            return 11.5, 15.0
        elif "k" in passband.lower():
            return 11.0, 14.5
        else:
            raise ValueError("Passband '{0}' not available".format(passband))

    def mag(self, passband):
        """
        Returns magnitude in catalog based on keyword.

        Parameters
        ----------
        passband : str
            Key in catalog for magnitude.

        Returns
        -------
        iterable
            List of lists for each catalog and extension in self.

        """
        return [
            [y for y in x]
            for x in self.get_columns(
                column_name=self.translate_passband(passband=passband)
            )
        ]

    def mag_err(self, passband):
        """
        Returns magnitude in catalog based on keyword.

        Parameters
        ----------
        passband : str
            Key in catalog for magnitude.

        Returns
        -------
        iterable
            List of lists for each catalog and extension in self.

        """
        return [
            [y for y in x]
            for x in self.get_columns(
                column_name="e_" + self.translate_passband(passband=passband)
            )
        ]

    def get_purge_index(
        self, passband, allowed_qflags: str = "A", allowed_cflags: str = "0c"
    ):
        """
        Cleans catalog from bad measurements in a given passband.

        Parameters
        ----------
        passband : str
            Passband.
        allowed_qflags : str
            Specifies which Qflags are not purged. e.g. 'A', or 'ABC'. Default is 'A'.
        allowed_cflags : str
            Specifies which Cflags are not purged. e.g. '0c', or '0cd'. Default is '0c'.

        Returns
        -------
        array
            Index array for cleaned sources.

        """
        return np.array(
            [
                True if (q[0] in allowed_qflags) & (c[0] in allowed_cflags) else False
                for q, c in zip(
                    self.qflags(passband=passband)[0][0],
                    self.cflags(passband=passband)[0][0],
                )
            ]
        )


class MasterAstrometry(SourceCatalogs):
    def __init__(self, setup, file_paths=None, **kwargs):
        super(MasterAstrometry, self).__init__(
            file_paths=file_paths, setup=setup, **kwargs
        )


class MasterAstrometryGaia(MasterAstrometry):
    def __init__(
        self,
        setup,
        file_paths=None,
        key_pmra="pmra",
        key_pmdec="pmdec",
    ):
        super(MasterAstrometryGaia, self).__init__(
            file_paths=file_paths,
            setup=setup,
            key_ra="ra",
            key_dec="dec",
        )
        self.key_pmra = key_pmra
        self.key_pmdec = key_pmdec

    @property
    def epoch(self):
        """Returns epoch of catalog."""
        return self.read_from_prime_headers(keywords=["EPOCH"])[0][0]

    @property
    def iter_data_hdu(self):
        """Override iter_data_hdu"""
        return [range(2, len(hdrs), 2) for hdrs in self.headers]

    _pmra = None

    @property
    def pmra(self):
        # Check if already determined
        if self._pmra is not None:
            return self._pmra

        # Retun columns
        self._pmra = [
            [y for y in x] for x in self.get_columns(column_name=self.key_pmra)
        ]
        return self._pmra

    _pmdec = None

    @property
    def pmdec(self):
        # Check if already determined
        if self._pmdec is not None:
            return self._pmdec

        # Retun columns
        self._pmdec = [
            [y for y in x] for x in self.get_columns(column_name=self.key_pmdec)
        ]
        return self._pmdec

    @property
    def skycoord(self):
        """
        Constructs SkyCoord object from ra/dec (overrides SourceCatalog property).

        Returns
        -------
        iterable
            List of lists holding SkyCoord objects

        """

        from astropy.units import Unit

        udeg = Unit("deg")
        umasyr = Unit("mas/yr")

        skycoord_files = []
        for fra, fdec, fpmra, fpmdec in zip(
            self.ra,
            self.dec,
            self.pmra,
            self.pmdec,
        ):
            skycoord_ext = []
            for ra, dec, pmra, pmdec in zip(fra, fdec, fpmra, fpmdec):
                skycoord_ext.append(
                    SkyCoord(
                        ra=ra * udeg,
                        dec=dec * udeg,
                        pm_ra_cosdec=pmra * umasyr,
                        pm_dec=pmdec * umasyr,
                        frame="icrs",
                        obstime=Time(self.epoch, format="decimalyear"),
                    )
                )
            skycoord_files.append(skycoord_ext)

        return skycoord_files
