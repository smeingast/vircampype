# =========================================================================== #
# Import
from vircampype.fits.common import FitsFiles


class FitsTables(FitsFiles):

    def __init__(self, file_paths):
        """
        Class for Fits tables based on FitsFiles. Contains specific methods and functions applicable only to tables

        Parameters
        ----------
        file_paths : iterable
            List of input file paths pointing to the Fits tables.

        Returns
        -------

        """

        super(FitsTables, self).__init__(file_paths=file_paths)


class MasterTables(FitsTables):

    def __init__(self, file_paths):
        super(MasterTables, self).__init__(file_paths=file_paths)
