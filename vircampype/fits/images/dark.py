# =========================================================================== #
# Import
import numpy as np

from astropy.io import fits
from vircampype.data.cube import ImageCube
from vircampype.utils.miscellaneous import *
from vircampype.utils.plots import plot_value_detector
from vircampype.fits.images.common import FitsImages, MasterImages


class DarkImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(DarkImages, self).__init__(setup=setup, file_paths=file_paths)

    def build_master_dark(self):
        """ Create master darks. """

        # Processing info
        tstart = message_mastercalibration(master_type="MASTER-DARK", silent=self.setup["misc"]["silent"])

        """ This does not work with Dark normalisation. There has to be a set for each DIT/NDIT combination! """
        # Split files first on DIT and NDIT, then on lag
        split = self.split_exposure()
        split = flat_list([s.split_lag(max_lag=self.setup["dark"]["max_lag"]) for s in split])

        # Now loop through separated files and build the Masterdarks
        for files, fidx in zip(split, range(1, len(split) + 1)):  # type: DarkImages, int

            # Check sequence suitability for Dark (same number of HDUs and NDIT)
            files.check_compatibility(n_hdu_max=1, n_ndit_max=1)

            # Create master dark name
            outpath = files.build_master_path(basename="MASTER-DARK", idx=0, dit=True, ndit=True, mjd=True)

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]) \
                    and not self.setup["misc"]["overwrite"]:
                continue

            # Instantiate output
            master_cube = ImageCube(setup=self.setup)

            # Get Masterbpm if set
            master_bpm = files.get_master_bpm()

            # Start looping over detectors
            data_headers = []
            for d in files.data_hdu[0]:

                # Print processing info
                if not self.setup["misc"]["silent"]:
                    message_calibration(n_current=fidx, n_total=len(split), name=outpath,
                                        d_current=d, d_total=max(files.data_hdu[0]))

                # Get data
                cube = files.hdu2cube(hdu_index=d, dtype=np.float32)

                # Get master calibration
                bpm = master_bpm.hdu2cube(hdu_index=d, dtype=np.uint8)

                # Masking methods
                cube.apply_masks(mask_min=self.setup["dark"]["mask_min"], mask_max=self.setup["dark"]["mask_max"],
                                 sigma_level=self.setup["dark"]["sigma_level"],
                                 sigma_iter=self.setup["dark"]["sigma_iter"], bpm=bpm)

                # Collapse extensions
                collapsed = cube.flatten(metric=str2func(self.setup["dark"]["metric"]))

                # Determine dark current as median
                dc = np.nanmedian(collapsed) / (files.dit[0] * files.ndit[0])

                # Write DC into data header
                cards = make_cards(keywords=["HIERARCH PYPE DC"], values=[np.round(dc, decimals=3)],
                                   comments=["Dark current in ADU/s"])
                data_headers.append(fits.Header(cards=cards))

                # Append to output
                master_cube.extend(data=collapsed.astype(np.float32))

            # Make cards for primary headers
            prime_cards = make_cards(keywords=[self.setup["keywords"]["dit"], self.setup["keywords"]["ndit"],
                                               self.setup["keywords"]["date_mjd"], self.setup["keywords"]["date_ut"],
                                               self.setup["keywords"]["object"], "HIERARCH PYPE N_FILES"],
                                     values=[files.dit[0], files.ndit[0],
                                             files.mjd_mean, files.time_obs_mean,
                                             "MASTER-DARK", len(files)])
            prime_header = fits.Header(cards=prime_cards)

            # Write to disk
            master_cube.write_mef(path=outpath, prime_header=prime_header, data_headers=data_headers)

            # QC plot
            if self.setup["misc"]["qc_plots"]:
                mdark = MasterDark(setup=self.setup, file_paths=outpath)
                mdark.qc_plot_dark(paths=None, axis_size=5, overwrite=self.setup["misc"]["overwrite"])

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])


class MasterDark(MasterImages):

    def __init__(self, setup, file_paths=None):
        super(MasterDark, self).__init__(setup=setup, file_paths=file_paths)

    @property
    def darkcurrent(self):
        return self.dataheaders_get_keys(keywords=["HIERARCH PYPE DC"])[0]

    def qc_plot_dark(self, paths=None, axis_size=5, overwrite=False):
        """
        Generates a simple QC plot for BPMs.

        Parameters
        ----------
        paths : list, optional
            Paths of the QC plot files. If None (default), use relative paths.
        axis_size : int, float, optional
            Axis size. Default is 5.
        overwrite : optional, bool
            Whether an exisiting plot should be overwritten. Default is False.

        """

        # Generate path for plots
        if paths is None:
            paths = ["{0}{1}.pdf".format(self.path_qc_dark, fp) for fp in self.file_names]

        # Loop over files and create plots
        for dc, path in zip(self.darkcurrent, paths):
            plot_value_detector(values=dc, path=path, ylabel="Dark Current (e-/s)",
                                axis_size=axis_size, overwrite=overwrite)
