# =========================================================================== #
# Import
from vircampype.utils import *
from vircampype.fits.images.sky import SkyImages


class ApcorImages(SkyImages):

    def __init__(self, setup, file_paths=None):
        super(ApcorImages, self).__init__(setup=setup, file_paths=file_paths)

    @property
    def _swarp_preset_apcor_path(self):
        """
        Obtains path to coadd preset for swarp.

        Returns
        -------
        str
            Path to preset.
        """
        return get_resource_path(package=self._swarp_preset_package, resource="swarp_apcor.yml")

    def get_apcor(self, skycoo, file_index, hdu_index):
        """
        Fetches aperture correction directly from image

        Parameters
        ----------
        skycoo : SkyCoord
            Input astropy SkyCoord object for which the aperture correction should be obtained.
        file_index : int
            Index of file in self.
        hdu_index : int
            Index of HDU

        Returns
        -------
        ndarray
            Array with aperture corrections.

        """
        return self.get_pixel_value(skycoo=skycoo, file_index=file_index, hdu_index=hdu_index)

    def coadd_apcor(self):

        # Processing info
        tstart = message_mastercalibration(master_type="COADDING", silent=self.setup["misc"]["silent"], right=None)

        # Create output header
        header = self.header_coadd(scale=self.cdelt1_mean)

        # Write header to disk
        outpath = "/Users/stefan/Desktop/test.fits"
        header.totextfile(outpath.replace(".fits", ".ahead"), overwrite=True, endcard=True)

        ss = yml2config(path=self._swarp_preset_apcor_path, imageout_name=outpath, weight_type="None",
                        weightout_name=outpath.replace(".fits", ".weight.fits"), resample_dir=self.path_temp,
                        nthreads=self.setup["misc"]["n_threads"], skip=["weight_thresh", "weight_image"])

        # Construct commands for source extraction
        cmd = "{0} {1} -c {2} {3}".format(self.bin_swarp, " ".join(self.full_paths), self._swarp_default_config, ss)

        # Run Swarp
        if not check_file_exists(file_path=self._swarp_path_coadd, silent=self.setup["misc"]["silent"]) \
                and not self.setup["misc"]["overwrite"]:
            run_command_bash(cmd=cmd, silent=False)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])
