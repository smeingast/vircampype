from vircampype.tools.plottools import *
from vircampype.fits.images.common import MasterImages


class MasterBadPixelMask(MasterImages):
    def __init__(self, setup, file_paths=None):
        super(MasterImages, self).__init__(setup=setup, file_paths=file_paths)

    @property
    def bpmfracs(self):
        """Bad pixel fraction for each image and extension."""
        return self.read_from_data_headers(keywords=["PYPE BADFRAC"])[0]

    @property
    def nbadpix(self):
        """Number of bad pixels for each image and extension."""
        return self.read_from_data_headers(keywords=["PYPE NBADPIX"])[0]

    def qc_plot_bpm(self, paths=None, axis_size=5):
        """
        Generates a simple QC plot for BPMs.

        Parameters
        ----------
        paths : list, optional
            Path of the QC plot file. If None (default), use relative path
        axis_size : int, float, optional
            Axis size. Default is 5.

        """

        # Generate path for plots
        if paths is None:
            paths = [
                "{0}{1}.pdf".format(self.setup.folders["qc_bpm"], fp)
                for fp in self.basenames
            ]

        # Loop over files and create plots
        for bpm, path in zip(self.bpmfracs, paths):
            plot_value_detector(
                values=[x * 100 for x in bpm],
                path=path,
                ylabel="Bad pixel fraction (%)",
                axis_size=axis_size,
            )
