import time
import warnings
import numpy as np

from astropy.io import fits
from vircampype.tools.plottools import *
from vircampype.tools.messaging import *
from vircampype.tools.fitstools import *
from vircampype.data.cube import ImageCube
from vircampype.tools.miscellaneous import *
from astropy.stats import sigma_clipped_stats
from vircampype.fits.images.common import FitsImages
from vircampype.fits.images.common import MasterImages


class DarkImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(DarkImages, self).__init__(setup=setup, file_paths=file_paths)

    def build_master_dark(self):
        """ Create master darks. """

        # Processing info
        print_header(header="MASTER-DARK", silent=self.setup.silent)
        tstart = time.time()

        # Split files first on NDIT, then on lag
        split = self.split_keywords(keywords=[self.setup.keywords.ndit])
        split = flat_list([s.split_lag(max_lag=self.setup.dark_max_lag) for s in split])

        # Now loop through separated files and build the Masterdarks
        for files, fidx in zip(split, range(1, len(split) + 1)):  # type: DarkImages, int

            # Check sequence suitability for Dark (same number of HDUs and NDIT)
            files.check_compatibility(n_hdu_max=1, n_ndit_max=1)

            # Create Mastedark name
            outpath = "{0}MASTER-DARK.NDIT_{1}.MJD_{2:0.4f}.fits" \
                      "".format(files.setup.folders["master_common"], files.ndit[0], files.mjd_mean)

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent) and not self.setup.overwrite:
                continue

            # Instantiate output
            master_cube = ImageCube(setup=self.setup)

            # Get Masterbpm if set
            master_bpm = files.get_master_bpm()

            # Start looping over detectors
            data_headers = []
            for d in files.iter_data_hdu[0]:

                # Print processing info
                if not self.setup.silent:
                    message_calibration(n_current=fidx, n_total=len(split), name=outpath,
                                        d_current=d, d_total=max(files.iter_data_hdu[0]))

                # Get data
                cube = files.hdu2cube(hdu_index=d, dtype=np.float32)

                # Scale cube with DIT
                cube /= files.dit_norm[:, np.newaxis, np.newaxis]

                # Get master calibration
                bpm = master_bpm.hdu2cube(hdu_index=d, dtype=np.uint8)

                # Masking methods
                cube.apply_masks(mask_min=self.setup.dark_mask_min, mask_max=self.setup.dark_mask_max, bpm=bpm,
                                 sigma_level=self.setup.dark_sigma_level, sigma_iter=self.setup.dark_sigma_iter)

                # Extract mask
                mask = ~np.isfinite(cube.cube)

                # Fill masked cube with temporary values
                cube.cube[mask] = 0.

                # Make weight cube
                weights = np.full_like(cube.cube, fill_value=1, dtype=np.float32)
                weights[:] = files.dit_norm[:, np.newaxis, np.newaxis]
                weights[mask] = 0.

                # Compute weighted average
                collapsed = np.ma.average(cube.cube, weights=weights, axis=0).filled(fill_value=np.nan)

                # Determine dark current and std
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    _, dc, dc_std = sigma_clipped_stats(collapsed)

                # Write DC into data header
                header = fits.Header()
                add_int_to_header(header=header, key="HIERARCH PYPE DC NIMAGES", value=len(files),
                                  comment="Number of darkframes used")
                add_float_to_header(header=header, key="HIERARCH PYPE DC", value=dc,
                                    decimals=3, comment="Dark current in ADU/s")
                add_float_to_header(header=header, key="HIERARCH PYPE DC STD", value=dc_std,
                                    decimals=3, comment="Dark current standard deviation in ADU")
                data_headers.append(header)

                # Append to output
                master_cube.extend(data=collapsed.astype(np.float32))

            # Make cards for primary headers
            prime_cards = make_cards(keywords=[self.setup.keywords.dit, self.setup.keywords.ndit,
                                               self.setup.keywords.date_mjd, self.setup.keywords.date_ut,
                                               self.setup.keywords.object, "HIERARCH PYPE N_FILES"],
                                     values=[1, files.ndit[0],
                                             files.mjd_mean, files.time_obs_mean,
                                             "MASTER-DARK", len(files)])
            prime_header = fits.Header(cards=prime_cards)

            # Write to disk
            master_cube.write_mef(path=outpath, prime_header=prime_header, data_headers=data_headers)

            # QC plot
            if self.setup.qc_plots:
                mdark = MasterDark(setup=self.setup, file_paths=outpath)
                mdark.qc_plot_dark(paths=None, axis_size=5)

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")


class MasterDark(MasterImages):

    def __init__(self, setup, file_paths=None):
        super(MasterDark, self).__init__(setup=setup, file_paths=file_paths)

    @property
    def darkcurrent(self):
        return self.read_from_data_headers(keywords=["HIERARCH PYPE DC"])[0]

    @property
    def darkcurrent_std(self):
        return self.read_from_data_headers(keywords=["HIERARCH PYPE DC STD"])[0]

    @property
    def n_images(self):
        return self.read_from_data_headers(keywords=["HIERARCH PYPE DC NIMAGES"])[0]

    def qc_plot_dark(self, paths=None, axis_size=5):
        """
        Generates a simple QC plot for BPMs.

        Parameters
        ----------
        paths : list, optional
            Paths of the QC plot files. If None (default), use relative paths.
        axis_size : int, float, optional
            Axis size. Default is 5.

        """

        # Generate path for plots
        if paths is None:
            paths = ["{0}{1}.pdf".format(self.setup.folders["qc_dark"], fp) for fp in self.basenames]

        # Loop over files and create plots
        for path, dc, dcstd in zip(paths, self.darkcurrent, self.darkcurrent_std):
            plot_value_detector(values=dc, errors=dcstd, path=path, ylabel="Dark Current (ADU/s)",
                                axis_size=axis_size, hlines=[0])
