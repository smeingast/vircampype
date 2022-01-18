import numpy as np

from astropy.time import Time
from astropy.coordinates import SkyCoord
from vircampype.fits.tables.common import FitsTables


class SourceCatalogs(FitsTables):
    def __init__(self, setup, file_paths=None):
        super(SourceCatalogs, self).__init__(file_paths=file_paths, setup=setup)

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

    @property
    def _key_ra(self):
        return "RA"

    @property
    def _key_dec(self):
        return "DEC"

    _ra = None

    def ra(self, key=None):
        """
        Extracts all RA entries in tables.

        Parameters
        ----------
        key : str, optional
            Used to override default key for RA

        Returns
        -------
        iterable
            List of lists containing RAs

        """

        # Check if already determined
        if self._ra is not None:
            return self._ra

        # Override RA key
        kra = key if key is not None else self._key_ra

        # Retun columns
        self._ra = [[y for y in x] for x in self.get_columns(column_name=kra)]
        return self._ra

    _dec = None

    def dec(self, key=None):
        """
        Extracts all RA entries in tables.

        Parameters
        ----------
        key : str, optional
            Used to override default key for RA

        Returns
        -------
        iterable
            List of lists containing RAs

        """

        # Check if already determined
        if self._dec is not None:
            return self._dec

        # Override DEC key
        kdec = key if key is not None else self._key_dec

        # Get data from columns
        self._dec = [[y for y in x] for x in self.get_columns(column_name=kdec)]
        return self._dec

    def skycoord(self, key_ra=None, key_dec=None):
        """
        Constructs SkyCoord object from ra/dec
        Parameters
        ----------
        key_ra : str, optional
            Key for RA in table.
        key_dec : str, optional
            Key for DEC in table.

        Returns
        -------
        iterable
            List of lists holding SkyCoord objects

        """

        skycoord_files = []
        for fra, fdec in zip(self.ra(key=key_ra), self.dec(key=key_dec)):
            skycoord_ext = []
            for ra, dec in zip(fra, fdec):
                skycoord_ext.append(SkyCoord(ra=ra, dec=dec, frame="icrs", unit="deg"))
            skycoord_files.append(skycoord_ext)

        return skycoord_files


class MasterPhotometry(SourceCatalogs):
    def __init__(self, setup, file_paths=None):
        super(MasterPhotometry, self).__init__(file_paths=file_paths, setup=setup)


class MasterPhotometry2Mass(MasterPhotometry):
    def __init__(self, setup, file_paths=None):
        super(MasterPhotometry2Mass, self).__init__(file_paths=file_paths, setup=setup)

    @property
    def _key_ra(self):
        return "RAJ2000"

    @property
    def _key_dec(self):
        return "DEJ2000"

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

    def mag_lim(self, passband):
        """
        Fetches magnitude limits in setup.

        Returns
        -------
        tuple
            Tuple with upper and lower magnitude limits.
        """
        if self.setup.reference_mag_lim is not None:
            return self.setup.reference_mag_lim

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

    def get_purge_index(self, passband):
        """
        Cleans catalog from bad measurements in a given passband

        Parameters
        ----------
        passband : str
            Passband.

        Returns
        -------
        array
            Index array for cleaned sources.

        """
        return np.array(
            [
                True if (q[0] == "A") & (c[0] in "0c") else False
                for q, c in zip(
                    self.qflags(passband=passband)[0][0],
                    self.cflags(passband=passband)[0][0],
                )
            ]
        )


class MasterAstrometry(SourceCatalogs):
    def __init__(self, setup, file_paths=None):
        super(MasterAstrometry, self).__init__(file_paths=file_paths, setup=setup)

    @property
    def _key_pmra(self):
        return "pmra"

    @property
    def _key_pmdec(self):
        return "pmdec"

    @property
    def epoch(self):
        return 2000.0

    _pmra = None

    def pmra(self, key=None):

        # Check if already determined
        if self._pmra is not None:
            return self._pmra

        # Override RA key
        kpmra = key if key is not None else self._key_pmra

        # Retun columns
        self._pmra = [[y for y in x] for x in self.get_columns(column_name=kpmra)]
        return self._pmra

    _pmdec = None

    def pmdec(self, key=None):

        # Check if already determined
        if self._pmdec is not None:
            return self._pmdec

        # Override RA key
        kpmdec = key if key is not None else self._key_pmdec

        # Retun columns
        self._pmdec = [[y for y in x] for x in self.get_columns(column_name=kpmdec)]
        return self._pmdec

    def skycoord(self, key_ra=None, key_dec=None, key_pmra=None, key_pmdec=None):
        """
        Constructs SkyCoord object from ra/dec
        Parameters
        ----------
        key_ra : str, optional
            Key for RA in table.
        key_dec : str, optional
            Key for DEC in table.
        key_pmra : str, optional
            Key for PM along RA in table.
        key_pmdec : str, optional
            Key for PM along Dec in table.

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
            self.ra(key=key_ra),
            self.dec(key=key_dec),
            self.pmra(key=key_pmra),
            self.pmdec(key=key_pmdec),
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


class MasterAstrometryGaia(MasterAstrometry):
    def __init__(self, setup, file_paths=None):
        super(MasterAstrometryGaia, self).__init__(file_paths=file_paths, setup=setup)

    @property
    def _key_ra(self):
        return "RA_ICRS"

    @property
    def _key_dec(self):
        return "DE_ICRS"

    @property
    def _key_pmra(self):
        return "pmRA"

    @property
    def _key_pmdec(self):
        return "pmDE"

    @property
    def epoch(self):
        return 2016.0
