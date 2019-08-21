# =========================================================================== #
# Import
import numpy as np
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
        list[Table]
            List of astropy Table instances.

        """

        return [Table.read(self.full_paths[file_index], hdu=h) for h in self.data_hdu[file_index]]

    def hdu2table(self, hdu_index):
        """
        Reads all tables in current instance in a given HDU.

        Parameters
        ----------
        hdu_index : int
            Index of HDU.

        Returns
        -------
        list[Table]
            List of astropy Table instances.

        """
        return [Table.read(f, hdu=hdu_index) for f in self.full_paths]

    def get_column(self, hdu_index, column_name):
        """
        Extracts a single column for a given HDU across all given tables in the current instance.

        Parameters
        ----------
        hdu_index : int
            Index of HDU from where to extract column.
        column_name : str
            Name of column.

        Returns
        -------
        iterable
            List of Columns for all files in instance.

        """
        return [Table.read(f, hdu=hdu_index)[column_name] for f in self.full_paths]

    def get_columns(self, column_name):
        """
        Column reader cross all files and HDUs.

        Parameters
        ----------
        column_name : str
            name of the column to extract

        Returns
        -------
        iterable

        """

        temp = np.array([self.get_column(column_name=column_name, hdu_index=h) for h in self.data_hdu[0]])

        # Rearrange
        temp = np.rollaxis(np.array(temp), axis=1)
        return [t.tolist() for t in temp]


class MasterTables(FitsTables):

    def __init__(self, setup, file_paths=None):
        super(MasterTables, self).__init__(setup=setup, file_paths=file_paths)
