# ----------------------------------------------------------------------
# Import stuff
from vircampype.fits.tables.common import MasterTables
from vircampype.utils.plots import plot_value_detector


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class MasterGain(MasterTables):

    def __init__(self, file_paths):
        super(MasterGain, self).__init__(file_paths=file_paths)

    _gains = None

    @property
    def gains(self):

        # Check if already determined
        if self._gains is not None:
            return self._gains

        self._gains = self.get_column(column_name="gain")
        return self._gains

    def qc_plot_gain(self, paths=None, axis_size=5):

        # Generate path for plots
        if paths is None:
            paths = [x.replace(".tab", ".pdf") for x in self.full_paths]

        # Loop over files and create plots
        for gain, path in zip(self.gains, paths):
            plot_value_detector(values=gain, path=path, ylabel="Gain (e-/ADU)",
                                axis_size=axis_size, overwrite=self.setup["misc"]["overwrite"])
