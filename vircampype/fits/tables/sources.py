# =========================================================================== #
# Import
from vircampype.fits.tables.common import FitsTables


class SourceCatalogs(FitsTables):

    def __init__(self, setup, file_paths=None):
        super(SourceCatalogs, self).__init__(file_paths=file_paths, setup=setup)

    # =========================================================================== #
    # Coordinates
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
        self._ra = [[y.data for y in x] for x in self.get_columns(column_name=kra)]
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
        self._dec = [[y.data for y in x] for x in self.get_columns(column_name=kdec)]
        return self._dec

    # =========================================================================== #
    # Other
    # =========================================================================== #
    @property
    def filters(self):
        """
        Grabs filter keyword from header and puts in into list

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
