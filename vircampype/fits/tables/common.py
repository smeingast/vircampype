# =========================================================================== #
# Import
from astropy.io import fits
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
    # Master
    # =========================================================================== #
    def get_master_zeropoint(self):
        """
        Get for each file in self the corresponding MasterGain table.

        Returns
        -------
        MasterZeroPoint
            MasterZeroPoint instance holding for each file in self the corresponding MasterZeroPoint table.

        """

        # Get all MASTER-ZP files
        master_zp_all = self.get_master_tables().zeropoint

        # Extract PROV info from all ZP files
        master_zp_all_prov = master_zp_all.primeheaders_get_keys(keywords=["PROV1"])[0]

        # Find match for each file in self
        match_idx = []
        for bn in self.base_names:
            match_idx.append(master_zp_all_prov.index(bn))

        # Return exact match
        return master_zp_all.__class__(setup=self.setup, file_paths=[master_zp_all.full_paths[idx]
                                                                     for idx in match_idx])

    # =========================================================================== #
    # I/O
    # =========================================================================== #
    def filehdu2table(self, file_index, hdu_index):
        """
        Extracts columns from a FITS table in and FitsTables instance.

        Parameters
        ----------
        file_index : int
            The index of the table in the FitsTables instance.
        hdu_index : int
            Index of HDU.

        Returns
        -------
        Table
            Astropy table

        """

        return Table.read(self.full_paths[file_index], hdu=hdu_index)

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

    def get_column_hdu(self, idx_hdu, column_name):
        """
        Extracts a single column for a given HDU across all given tables in the current instance.

        Parameters
        ----------
        idx_hdu : int
            Index of HDU from where to extract column.
        column_name : str
            Name of column.

        Returns
        -------
        iterable
            List of Columns for all files in instance.

        """
        return [Table.read(f, hdu=idx_hdu)[column_name] for f in self.full_paths]

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

        data_files = []
        for file, dhus in zip(self.full_paths, self.data_hdu):
            data_hdus = []
            with fits.open(file) as f:
                for hdu in dhus:
                    data_hdus.append(f[hdu].data[column_name])
            data_files.append(data_hdus)

        return data_files

    def get_column_file(self, idx_file, column_name):
        """
        Fetches a given column for a given file across all extensions of this this file.

        Parameters
        ----------
        idx_file : int
            File index.
        column_name : str
            Name of column to read.

        Returns
        -------
        iterable
            List of arrays for given column.

        """
        with fits.open(self.full_paths[idx_file]) as f:
            columns = []
            for hdu in self.data_hdu[idx_file]:
                columns.append(f[hdu].data[column_name])
        return columns

    def get_columns_file(self, idx_file, column_names):
        """
        Extracts data for given column names across all data HDUs.

        Parameters
        ----------
        idx_file : int
            Index of file in current instance.
        column_names : iterable, list
            List of file names

        Returns
        -------
            List of extracted data for each column name [column_name1[hdu1,...], column_name2[hdu1, ...], ...]

        """
        with fits.open(self.full_paths[idx_file]) as f:

            # Read file
            data_hdus = [f[hdu].data for hdu in self.data_hdu[idx_file]]

            # Store data and return
            columns = []
            for cn in column_names:
                columns.append([dh[cn] for dh in data_hdus])

        return columns


class MasterTables(FitsTables):

    def __init__(self, setup, file_paths=None):
        super(MasterTables, self).__init__(setup=setup, file_paths=file_paths)

    @property
    def linearity(self):
        """
        Holds all MasterLinearity tables.

        Returns
        -------
        MasterLinearity
            All MasterLinearity tables as a MasterLinearity instance.

        """

        # Import
        from vircampype.fits.tables.linearity import MasterLinearity

        # Get the masterlinearity files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-LINEARITY"]

        return MasterLinearity(setup=self.setup, file_paths=[self.file_paths[idx] for idx in index])

    @property
    def gain(self):
        """
        Holds all MasterGain tables.

        Returns
        -------
        MasterGain
            All MasterGain tables as a MasterLinearity instance.

        """
        # Import
        from vircampype.fits.tables.gain import MasterGain

        # Get the mastergain files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-GAIN"]

        return MasterGain(setup=self.setup, file_paths=[self.file_paths[idx] for idx in index])

    @property
    def zeropoint(self):
        """
        Holds all MasterZeroPoint tables.

        Returns
        -------
        MasterZeroPoint
            All MasterZeroPoint tables as a MasterZeroPoint instance.

        """
        # Import
        from vircampype.fits.tables.zeropoint import MasterZeroPoint

        # Get the mastergain files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-ZEROPOINT"]

        return MasterZeroPoint(setup=self.setup, file_paths=[self.file_paths[idx] for idx in index])

    @property
    def photometry(self):
        """
        Holds all MasterPhotometry tables.

        Returns
        -------
        MasterPhotometry
            All MasterPhotometry tables as a MasterLinearity instance.

        """

        # Import
        from vircampype.fits.tables.sources import MasterPhotometry, MasterPhotometry2Mass

        # Get the masterphotometry files
        index = [idx for idx, key in enumerate(self.types) if key == "MASTER-PHOTOMETRY"]

        # Return photometry catalog
        if self.setup["photometry"]["reference"] == "2mass":
            return MasterPhotometry2Mass(setup=self.setup, file_paths=[self.file_paths[idx] for idx in index])
        else:
            return MasterPhotometry(setup=self.setup, file_paths=[self.file_paths[idx] for idx in index])
