# =========================================================================== #
# Import
from vircampype.utils import *
from vircampype.fits.images.sky import SkyImages


class ApcorImages(SkyImages):

    def __init__(self, setup, file_paths=None):
        super(ApcorImages, self).__init__(setup=setup, file_paths=file_paths)

    @property
    def diameters(self):
        """
        Fetches aperture diameters from headers.

        Returns
        -------
        iterable
            List of lists for each file and each detector.

        """
        return self.dataheaders_get_keys(keywords=["APCDIAM"])[0]

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

        # Split by aperture diameter
        split_apcor = self.split_keywords(keywords=["APCDIAM"])

        for split in split_apcor:  # type: ApcorImages

            # Get current diameter
            diameter = split.diameters[0][0]

            # Create output header
            header = split.header_coadd(scale=self.cdelt1_mean)

            # Write header to disk
            outpath = "/Users/stefan/Desktop/test{0}.fits".format(diameter)
            header.totextfile(outpath.replace(".fits", ".ahead"), overwrite=True, endcard=True)

            ss = yml2config(path=split._swarp_preset_apcor_path, imageout_name=outpath, weight_type="None",
                            weightout_name=outpath.replace(".fits", ".weight.fits"), resample_dir=self.path_temp,
                            nthreads=split.setup["misc"]["n_threads"], skip=["weight_thresh", "weight_image"])

            # Construct commands for source extraction
            cmd = "{0} {1} -c {2} {3}".format(split.bin_swarp, " ".join(split.full_paths),
                                              split._swarp_default_config, ss)

            # Run Swarp
            if not check_file_exists(file_path=split._swarp_path_coadd, silent=split.setup["misc"]["silent"]) \
                    and not split.setup["misc"]["overwrite"]:
                run_command_bash(cmd=cmd, silent=False)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])
