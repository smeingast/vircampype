# =========================================================================== #
# Import
from vircampype.data.cube import ImageCube
from vircampype.utils.miscellaneous import *
from vircampype.fits.images.common import FitsImages
from vircampype.fits.images.bpm import MasterBadPixelMask


class FlatImages(FitsImages):

    def __init__(self, file_paths=None):
        super(FlatImages, self).__init__(file_paths=file_paths)

    def build_master_bpm(self):
        """ Builds a Bad pixel mask from image data. """

        # Processing info
        tstart = mastercalibration_message(master_type="MASTER-BPM", silent=self.setup["misc"]["silent"])

        # Split files based on maximum time lag is set
        split = self.split_lag(max_lag=self.setup["bpm"]["max_lag"])

        # Now loop through separated files and build Masterbpm
        outpaths = []
        for files, fidx in zip(split, range(1, len(split) + 1)):  # type: FlatImages, int

            # Check sequence compatibility
            files.check_compatibility(n_hdu_max=1, n_dit_max=1, n_ndit_max=1, n_files_min=3)

            # Create Masterbpm name
            outpaths.append(files.create_masterpath(basename="MASTER-BPM", ndit=True, mjd=True))

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpaths[-1], silent=self.setup["misc"]["silent"]) \
                    and not self.setup["misc"]["overwrite"]:
                continue

            # Instantiate output
            master_bpm = ImageCube()

            # Start looping over detectors
            data_headers = []
            for d in files.data_hdu[0]:

                # Print processing info
                if not self.setup["misc"]["silent"]:
                    calibration_message(n_current=fidx, n_total=len(split), name=outpaths[-1],
                                        d_current=d, d_total=len(files.data_hdu[0]))

                # Get data
                cube = files.hdu2cube(hdu_index=d, dtype=np.float32)

                # Mask low and high absolute values
                cube.apply_masks(mask_below=self.setup["bpm"]["abs_lo"], mask_above=self.setup["bpm"]["abs_hi"])

                # Collapse cube with median
                flat = cube.flatten(metric=str2func(self.setup["bpm"]["collapse_metric"]), axis=0)

                # Normalize cube with flattened data
                cube.cube = cube.cube / flat

                # Mask low and high relative values
                cube.apply_masks(mask_below=self.setup["bpm"]["rel_lo"], mask_above=self.setup["bpm"]["rel_hi"])

                # Count how many bad pixels there are in the stack and normalize to the number of input images
                nbad_pix = np.sum(~np.isfinite(cube.cube), axis=0) / files.n_files

                # Get those pixels where the number of bad pixels is greater than the given input threshold
                bpm = np.array(nbad_pix > self.setup["bpm"]["frac"], dtype=np.uint8)

                # Make header cards
                cards = make_cards(keywords=["HIERARCH PYPE NBADPIX", "HIERARCH PYPE BADFRAC"],
                                   values=[np.int(np.sum(bpm)), np.round(np.sum(bpm) / bpm.size, decimals=5)],
                                   comments=["Number of bad pixels", "Fraction of bad pixels"])
                data_headers.append(fits.Header(cards=cards))

                # Append HDU
                master_bpm.extend(data=bpm)

            # Make cards for primary headers
            prime_cards = make_cards(keywords=[self.setup["keywords"]["dit"], self.setup["keywords"]["ndit"],
                                               self.setup["keywords"]["date_mjd"], self.setup["keywords"]["date_ut"],
                                               self.setup["keywords"]["object"], "HIERARCH PYPE N_FILES"],
                                     values=[files.dit[0], files.ndit[0],
                                             files.mjd_mean, files.time_obs_mean,
                                             "MASTER-BPM", len(files)])
            prime_header = fits.Header(cards=prime_cards)

            # Write to disk
            master_bpm.write_mef(path=outpaths[-1], prime_header=prime_header, data_headers=data_headers)

        # QC plot
        if self.setup["misc"]["qc_plots"]:
            mbpm = MasterBadPixelMask(file_paths=outpaths)
            mbpm.qc_plot_bpm(paths=None, axis_size=5, overwrite=self.setup["misc"]["overwrite"])

        # Print time
        if not self.setup["misc"]["silent"]:
            print("-> Elapsed time: {0:.2f}s".format(time.time() - tstart))
