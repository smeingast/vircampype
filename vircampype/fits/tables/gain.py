# ----------------------------------------------------------------------
# Import stuff
from vircampype.fits.tables.common import MasterTables
from vircampype.utils.plots import plot_value_detector


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class MasterGain(MasterTables):

    def __init__(self, file_paths):
        super(MasterGain, self).__init__(file_paths=file_paths)

    _gain = None

    @property
    def gain(self):

        # Check if already determined
        if self._gain is not None:
            return self._gain

        self._gain = self.get_column(column_name="gain")
        return self._gain

    _rdnoise = None

    @property
    def rdnoise(self):

        # Check if already determined
        if self._rdnoise is not None:
            return self._rdnoise

        self._rdnoise = self.get_column(column_name="rdnoise")
        return self._rdnoise

    def qc_plot_gain(self, paths=None, axis_size=5, overwrite=False):

        # Generate path for plots
        paths = self._make_plot_paths(paths=paths, prefix="gain")

        # Loop over files and create plots
        for gain, path in zip(self.gain, paths):
            plot_value_detector(values=gain, path=path, ylabel="Gain (e-/ADU)",
                                axis_size=axis_size, overwrite=overwrite)

    def qc_plot_rdnoise(self, paths=None, axis_size=5, overwrite=False):

        # Generate path for plots
        paths = self._make_plot_paths(paths=paths, prefix="rdnoise")

        # Loop over files and create plots
        for rdn, path in zip(self.rdnoise, paths):
            plot_value_detector(values=rdn, path=path, ylabel="Read Noise (e-)",
                                axis_size=axis_size, overwrite=overwrite)

    def _make_plot_paths(self, paths, prefix=""):

        if paths is None:
            return [x.replace(".tab", ".{0}.pdf".format(prefix)) for x in self.full_paths]
        else:
            return paths
