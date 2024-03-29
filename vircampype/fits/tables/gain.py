from vircampype.tools.plottools import *
from vircampype.fits.tables.common import MasterTables


class MasterGain(MasterTables):
    def __init__(self, setup, file_paths=None):
        super(MasterGain, self).__init__(setup=setup, file_paths=file_paths)

    _gain = None

    @property
    def gain(self):

        # Check if already determined
        if self._gain is not None:
            return self._gain

        self._gain = [x[0] for x in self.get_columns(column_name="gain")]
        return self._gain

    _rdnoise = None

    @property
    def rdnoise(self):

        # Check if already determined
        if self._rdnoise is not None:
            return self._rdnoise

        self._rdnoise = [x[0] for x in self.get_columns(column_name="rdnoise")]
        return self._rdnoise

    def qc_plot_gain(self, paths=None, axis_size=5):

        # Generate path for plots
        paths = self.paths_qc_plots(paths=paths, prefix="gain")

        # Loop over files and create plots
        for gain, path in zip(self.gain, paths):
            plot_value_detector(
                values=gain, path=path, ylabel="Gain (e-/ADU)", axis_size=axis_size
            )

    def qc_plot_rdnoise(self, paths=None, axis_size=5):

        # Generate path for plots
        paths = self.paths_qc_plots(paths=paths, prefix="rdnoise")

        # Loop over files and create plots
        for rdn, path in zip(self.rdnoise, paths):
            plot_value_detector(
                values=rdn, path=path, ylabel="Read Noise (e-)", axis_size=axis_size
            )

    def paths_qc_plots(self, paths, prefix=""):

        if paths is None:
            return [
                "{0}{1}.{2}.pdf".format(self.setup.folders["qc_gain"], fp, prefix)
                for fp in self.basenames
            ]
        else:
            return paths
