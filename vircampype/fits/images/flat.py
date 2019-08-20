# =========================================================================== #
# Import
import numpy as np

from vircampype.setup import *
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
        tstart = mastercalibration_message(master_type="MASTER-BPM", silent=setup_silent)

        # Split files based on maximum time lag is set
        split = self._split_lag(max_lag=setup_bpm_maxlag)

        # Now loop through separated files and build Masterbpm
        outpaths = []
        for files, fidx in zip(split, range(1, len(split) + 1)):  # type: FlatImages, int

            # Check sequence compatibility
            files.check_compatibility(n_hdu_max=1, n_dit_max=1, n_ndit_max=1, n_files_min=3)

            # Create Masterbpm name
            outpaths.append(files.create_masterpath(basename="MASTER-BPM", ndit=True, mjd=True))

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpaths[-1], silent=setup_silent) and not setup_overwrite:
                continue

            # Instantiate output
            master_bpm = ImageCube(cube=None)

            # Start looping over detectors
            data_headers = []
            for d in files.data_hdu[0]:

                # Print processing info
                if not setup_silent:
                    calibration_message(n_current=fidx, n_total=len(split), name=outpaths[-1],
                                        d_current=d, d_total=len(files.data_hdu[0]))

                # Get data
                cube = files.hdu2cube(hdu_index=d, dtype=np.float32)

                # Mask low and high absolute values
                cube.apply_masks(mask_below=setup_bpm_abslo, mask_above=setup_bpm_abshi)

                # Collapse cube with median
                flat = cube.flatten(metric=np.nanmedian, axis=0)

                # Normalize cube with flattened data
                cube.cube = cube.cube / flat

                # Mask low and high relative values
                cube.apply_masks(mask_below=setup_bpm_rello, mask_above=setup_bpm_relhi)

                # Count how many bad pixels there are in the stack and normalize to the number of input images
                nbad_pix = np.sum(~np.isfinite(cube.cube), axis=0) / files.n_files

                # Get those pixels where the number of bad pixels is greater than the given input threshold
                bpm = np.array(nbad_pix > setup_bpm_frac, dtype=np.uint8)

                # Make header cards
                cards = make_cards(keywords=["NBADPIX", "BADFRAC"],
                                   values=[np.int(np.sum(bpm)), np.round(np.sum(bpm) / bpm.size, decimals=5)],
                                   comments=["Number of bad pixels", "Fraction of bad pixels"], hierarch=True)
                data_headers.append(fits.Header(cards=cards))

                # Append HDU
                master_bpm.extend(data=bpm)

            # Make cards for primary headers
            prime_cards = make_cards(keywords=[setup_kw_dit, setup_kw_ndit, setup_kw_mjd, setup_kw_object, "N_FILES"],
                                     values=[files.dit[0], files.ndit[0], files.mjd_mean, "MASTER-BPM", len(files)],
                                     hierarch=True)
            prime_header = fits.Header(cards=prime_cards)

            # Write to disk
            master_bpm.write_mef(path=outpaths[-1], prime_header=prime_header, data_headers=data_headers)

        # Create output
        mbpm = MasterBadPixelMask(file_paths=outpaths)

        # QC plots
        if setup_qc_plots:

            # Generate plot names
            plot_paths = [x.replace(".fits", ".pdf") for x in mbpm.full_paths]

            # Plot
            mbpm.qc_plot_bpm(paths=plot_paths, axis_size=5, overwrite=setup_overwrite)

        # Print time
        if not setup_silent:
            print("-> Elapsed time: {0:.2f}s".format(time.time() - tstart))
