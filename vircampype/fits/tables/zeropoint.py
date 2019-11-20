# =========================================================================== #
# Import
import numpy as np
from vircampype.fits.tables.common import MasterTables
from vircampype.utils.plots import plot_value_detector


class MasterZeroPoint(MasterTables):

    def __init__(self, setup, file_paths=None):
        super(MasterZeroPoint, self).__init__(setup=setup, file_paths=file_paths)

    @property
    def _zp_colnames(self):
        return [k for k in self.zp_table[0].columns.keys() if "ZP_" in k]

    @property
    def _zperr_colnames(self):
        return [k for k in self.zp_table[0].columns.keys() if "ZPERR_" in k]

    _zp_table = None

    @property
    def zp_table(self):
        """
        Constructs a list of tables containing all saved zeropoints

        Returns
        -------
        Iterable
            List of astropy tables.

        """

        if self._zp_table is not None:
            return self._zp_table

        self._zp_table = [self.file2table(file_index=idx_file)[0] for idx_file in range(len(self))]
        return self._zp_table

    def zp_diameter(self, diameter):
        """
        Extracts the ZP for a given aperture diameter.

        Parameters
        ----------
        diameter : str
            Aperture diameter given as sting.

        Returns
        -------
        iterable
            List of ZPs for each file for specific aperture.

        """
        return [t["ZP_{0}".format(diameter)] for t in self.zp_table]

    def zperr_diameter(self, diameter):
        """
        Extracts the ZP errors for a given aperture diameter.

        Parameters
        ----------
        diameter : str
            Aperture diameter given as sting.

        Returns
        -------
        iterable
            List of ZP errors for each file for specific aperture.

        """
        return [t["ZPERR_{0}".format(diameter)] for t in self.zp_table]

    @property
    def zp_mean(self):
        """
        Computes the mean ZP across all apertures for each file and each detector in self

        Returns
        -------
        iterable
            List of averages for each file containing the mean ZP for each detector.

        """
        return [np.mean(np.stack([tab[name] for name in self._zp_colnames]), axis=0) for tab in self.zp_table]

    @property
    def zperr_mean(self):
        """
        Computes the mean ZP across all apertures for each file and each detector in self

        Returns
        -------
        iterable
            List of averages for each file containing the mean ZP for each detector.

        """
        return [np.mean(np.stack([tab[name] for name in self._zperr_colnames]), axis=0) for tab in self.zp_table]

    @property
    def zp_mean_std(self):
        return [np.std(np.stack([tab[name] for name in self._zp_colnames]), axis=0) for tab in self.zp_table]

    def paths_qc_plots(self, paths=None, prefix="zp"):
        """ Constructs paths for ZP QC plots. """
        if paths is None:
            return ["{0}{1}.{2}.pdf".format(self.path_qc_photometry, fp, prefix) for fp in self.file_names]
        else:
            return paths

    def qc_plot_zp(self, paths=None, axis_size=5, overwrite=False):
        """ Generates ZP QC plot. """

        # Generate path for plots
        paths = self.paths_qc_plots(paths=paths, prefix="zp")

        # Loop over files and create plots
        for zp, zpstd, path in zip(self.zp_mean, self.zp_mean_std, paths):
            plot_value_detector(values=zp, errors=zpstd, path=path, ylabel="ZP (mag)", axis_size=axis_size,
                                overwrite=overwrite, yrange=(np.median(zp) - 0.05, np.median(zp) + 0.05))
