# =========================================================================== #
# Import
from astropy.table import Table
from vircampype.fits.common import FitsFiles


class FitsTables(FitsFiles):

    def __init__(self, setup, file_paths=None):
        """
        Class for Fits tables based on FitsFiles. Contains specific methods and functions applicable only to tables

        Parameters
        ----------
        file_paths : iterable
            List of input file paths pointing to the Fits tables.

        Returns
        -------

        """

        super(FitsTables, self).__init__(setup=setup, file_paths=file_paths)

    _types = None

    @property
    def types(self):
        """
        Property which holds the table types.

        Returns
        -------
        iterable
            Ordered list of table types.
        """

        # Check if already determined
        if self._types is not None:
            return self._types

        self._types = self.primeheaders_get_keys(["OBJECT"])[0]
        return self._types

    # =========================================================================== #
    # I/O
    # =========================================================================== #
    def file2table(self, file_index):
        """
        Extracts columns from a FITS table in and FitsTables instance.

        Parameters
        ----------
        file_index : int
            The index of the table in the FitsTables instance.

        Returns
        -------
        Table
            Astropy Table instance.

        """

        return Table.read(self.full_paths[file_index])

    def get_column(self, column_name):
        """
        Extracts a single column (indentified by name) across all given tables in the current instance.

        Parameters
        ----------
        column_name : str
            Name of column.

        Returns
        -------
        iterable
            List of Columns for all files in instance.

        """
        return [Table.read(f)[column_name] for f in self.full_paths]


class MasterTables(FitsTables):

    def __init__(self, setup, file_paths=None):
        super(MasterTables, self).__init__(setup=setup, file_paths=file_paths)
