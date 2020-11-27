# =========================================================================== #
# Import
from vircampype.utils.miscellaneous import str2list
from vircampype.fits.tables.common import FitsTables


class SourceCatalogs(FitsTables):

    def __init__(self, setup, file_paths=None):
        super(SourceCatalogs, self).__init__(file_paths=file_paths, setup=setup)

    # =========================================================================== #
    # Properties
    # =========================================================================== #
    @property
    def _key_ra(self):
        """
        Keyword for right ascension in tables.

        Returns
        -------
        str

        """
        return "RA"

    @property
    def _key_dec(self):
        """
        Keyword for Declination in tables.

        Returns
        -------
        str

        """
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

    def ra_file(self, idx_file, key=None):
        """
        Extract RA from a given file.

        Parameters
        ----------
        idx_file : int
            Index of file
        key: str, optional
            Column name of RA.

        Returns
        -------
        iterable
            List of RAs for each extension

        """
        # Override RA key
        kra = key if key is not None else self._key_ra

        # Return for given file
        return self.get_column_file(idx_file=idx_file, column_name=kra)

    def dec_file(self, idx_file, key=None):
        """
        Extract DEC from a given file.

        Parameters
        ----------
        idx_file : int
            Index of file
        key: str, optional
            Column name of RA.

        Returns
        -------
        iterable
            List of DECs for each extension

        """
        # Override RA key
        kdec = key if key is not None else self._key_dec

        # Return for given file
        return self.get_column_file(idx_file=idx_file, column_name=kdec)

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

        from astropy.coordinates import SkyCoord

        skycoord_files = []
        for fra, fdec in zip(self.ra(key=key_ra), self.dec(key=key_dec)):
            skycoord_ext = []
            for ra, dec in zip(fra, fdec):
                skycoord_ext.append(SkyCoord(ra=ra, dec=dec, frame="icrs", unit="deg"))
            skycoord_files.append(skycoord_ext)

        return skycoord_files

    def skycoord_file(self, idx_file, key_ra=None, key_dec=None):
        """
        Constructs SkyCoord object from ra/dec for a given file.
        Parameters
        ----------
        idx_file : int
            Index of file
        key_ra : str, optional
            Key for RA in table.
        key_dec : str, optional
            Key for DEC in table.

        Returns
        -------
        iterable
            List of Skycoords for each extension of the given catalog.

        """

        # Import
        from astropy.coordinates import SkyCoord

        if key_ra is None:
            key_ra = self._key_ra
        if key_dec is None:
            key_dec = self._key_dec

        skycoord = []
        for ra, dec in zip(*self.get_columns_file(idx_file=idx_file, column_names=[key_ra, key_dec])):
            skycoord.append(SkyCoord(ra=ra, dec=dec, frame="icrs", unit="deg"))

        return skycoord

    @property
    def filter(self):
        """
        Grabs filter keyword from header and puts in into list.

        Returns
        -------
        iterable
            List of filters for all tables in instance.

        """
        return self.primeheaders_get_keys(keywords=[self.setup["keywords"]["filter"]])[0]


class ESOSourceCatalogs(SourceCatalogs):

    def __init__(self, setup, file_paths=None):
        super(ESOSourceCatalogs, self).__init__(file_paths=file_paths, setup=setup)


class MasterPhotometry(SourceCatalogs):

    def __init__(self, setup, file_paths=None):
        super(MasterPhotometry, self).__init__(file_paths=file_paths, setup=setup)

    @property
    def mag_lim(self):
        """
        Fetches magnitude limits in setup.

        Returns
        -------
        iterable
            List with magnitude limits.
        """
        return str2list(s=self.setup["photometry"]["mag_limits_ref"], sep=",", dtype=float)


class MasterPhotometry2Mass(MasterPhotometry):

    def __init__(self, setup, file_paths=None):
        super(MasterPhotometry2Mass, self).__init__(file_paths=file_paths, setup=setup)

    # =========================================================================== #
    # Coordinates
    # =========================================================================== #
    @property
    def _key_ra(self):
        return "RAJ2000"

    @property
    def _key_dec(self):
        return "DEJ2000"

    @staticmethod
    def translate_filter(key):
        """
        Translates input filter from e.g. VISTA to 2MASS names.

        Parameters
        ----------
        key : str
            The input magnitude name. e.g. 'J'.

        Returns
        -------
        str
            Translated magnitude key. e.g. 'Ks' will be translated to 'Kmag'.
        """

        if "j" in key.lower():
            return "Jmag"
        elif "h" in key.lower():
            return "Hmag"
        elif "k" in key.lower():
            return "Kmag"
        else:
            raise ValueError("Filter '{0}' not defined".format(key))

    @staticmethod
    def _key2idx(key):
        """ Helper to return filter index for flags. """
        if "j" in key.lower():
            return 0
        elif "h" in key.lower():
            return 1
        elif "k" in key.lower():
            return 2
        else:
            raise ValueError("Filter '{0}' not defined".format(key))

    __qflag = None

    @property
    def _qflags(self):
        """ Reads quality flag column from catalog. """

        if self.__qflag is not None:
            return self.__qflag

        self.__qflag = self.get_columns(column_name="Qflg")
        return self.__qflag

    def qflags(self, key):
        """ Return Quality flag for given filter. """
        return [[[x[self._key2idx(key=key)] for x in y] for y in z] for z in self._qflags]

    __cflag = None

    @property
    def _cflags(self):
        """ Reads contamintation and confusion flag column from catalog. """

        if self.__cflag is not None:
            return self.__cflag

        self.__cflag = self.get_columns(column_name="Cflg")
        return self.__cflag

    def cflags(self, key):
        """ Return contamination and confusion flag for given filter. """
        return [[[x[self._key2idx(key=key)] for x in y] for y in z] for z in self._cflags]

    def mag(self, band):
        """
        Returns magnitude in catalog based on keyword.

        Parameters
        ----------
        band : str
            Key in catalog for magnitude.

        Returns
        -------
        iterable
            List of lists for each catalog and extension in self.

        """
        return [[y for y in x] for x in self.get_columns(column_name=band)]

    def mag_err(self, band):
        """
        Returns magnitude in catalog based on keyword.

        Parameters
        ----------
        band : str
            Key in catalog for magnitude.

        Returns
        -------
        iterable
            List of lists for each catalog and extension in self.

        """
        return [[y for y in x] for x in self.get_columns(column_name="e_" + band)]
