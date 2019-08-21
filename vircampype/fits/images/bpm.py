# =========================================================================== #
# Import
from vircampype.fits.images.common import FitsImages
from vircampype.utils.plots import plot_value_detector


class MasterBadPixelMask(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(FitsImages, self).__init__(setup=setup, file_paths=file_paths)

    @property
    def bpmfracs(self):
        """Bad pixel fraction for each image and extension."""
        return self.dataheaders_get_keys(keywords=["PYPE BADFRAC"])[0]

    @property
    def nbadpix(self):
        """Number of bad pixels for each image and extension."""
        return self.dataheaders_get_keys(keywords=["PYPE NBADPIX"])[0]

    def qc_plot_bpm(self, paths=None, axis_size=5, overwrite=False):
        """
        Generates a simple QC plot for BPMs.

        Parameters
        ----------
        paths : list, optional
            Path of the QC plot file. If None (default), use relative path
        axis_size : int, float, optional
            Axis size. Default is 5.
        overwrite : optional, bool
            Whether an exisiting plot should be overwritten. Default is False.

        """

        # Generate path for plots
        if paths is None:
            paths = [x.replace(".fits", ".pdf") for x in self.full_paths]

        # Loop over files and create plots
        for bpm, path in zip(self.bpmfracs, paths):
            plot_value_detector(values=[x * 100 for x in bpm], path=path, ylabel="Bad pixel fraction (%)",
                                axis_size=axis_size, overwrite=overwrite)
