# =========================================================================== #
# Import
from vircampype.fits.tables.common import FitsTables


class SourceCatalog(FitsTables):

    def __init__(self, setup, file_paths=None):
        super(SourceCatalog, self).__init__(file_paths=file_paths, setup=setup)

    # =========================================================================== #
    # Other methods
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


class ESOSourceCatalog(SourceCatalog):

    def __init__(self, setup, file_paths=None):
        super(ESOSourceCatalog, self).__init__(file_paths=file_paths, setup=setup)


class MasterPhotometry(SourceCatalog):

    def __init__(self, setup, file_paths=None):
        super(MasterPhotometry, self).__init__(file_paths=file_paths, setup=setup)
