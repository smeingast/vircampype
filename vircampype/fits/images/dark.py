import time
import warnings

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from vircampype.data.cube import ImageCube
from vircampype.fits.images.common import FitsImages, MasterImages
from vircampype.pipeline.log import PipelineLog
from vircampype.tools.fitstools import *
from vircampype.tools.messaging import *
from vircampype.tools.miscellaneous import *
from vircampype.tools.plottools import *


class DarkImages(FitsImages):
    def __init__(self, setup, file_paths=None):
        super(DarkImages, self).__init__(setup=setup, file_paths=file_paths)

    def build_master_dark(self):
        """Create master darks."""

        # Processing info
        print_header(header="MASTER-DARK", silent=self.setup.silent)

        # Fetch log
        log = PipelineLog()
        log.info(
            f"Building master darks from {self.n_files} files:\n{self.basenames2log}"
        )
        tstart = time.time()

        # Split files first on NDIT, then on lag
        split = self.split_keywords(
            keywords=[self.setup.keywords.ndit, self.setup.keywords.dit]
        )
        split = flat_list([s.split_lag(max_lag=self.setup.dark_max_lag) for s in split])

        # Remove sequences with too few images
        split = prune_list(split, n_min=3)
        log.info(f"Number of dark groups: {len(split)}")

        # Now loop through separated files and build the Masterdarks
        for files, fidx in zip(split, range(1, len(split) + 1)):  # type: DarkImages, int
            # Check sequence suitability for Dark (same number of HDUs and NDIT)
            files.check_compatibility(n_hdu_max=1, n_ndit_max=1, n_dit_max=1)

            # Create Mastedark name
            outpath = (
                f"{files.setup.folders['master_common']}"
                f"MASTER-DARK.DIT_{files.dit[0]}"
                f".NDIT_{files.ndit[0]}"
                f".MJD_{files.mjd_mean:0.4f}"
                f".fits"
            )
            log.info(f"Processing dark group {fidx}/{len(split)}: {outpath}")

            # Check if the file is already there and skip if it is
            if (
                check_file_exists(file_path=outpath, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                log.info("File already exists, skipping")
                continue

            # Instantiate output
            master_cube = ImageCube(setup=self.setup)

            # Get master linearity
            master_linearity = files.get_master_linearity()
            log.info(f"Master linearity:\n{master_linearity.basenames2log}")
            master_bpm = files.get_master_bpm()
            log.info(f"Master BPM:\n{master_bpm.basenames2log}")

            # Start looping over detectors
            data_headers = []
            for d in files.iter_data_hdu[0]:
                # Print processing info
                if not self.setup.silent:
                    message_calibration(
                        n_current=fidx,
                        n_total=len(split),
                        name=outpath,
                        d_current=d,
                        d_total=max(files.iter_data_hdu[0]),
                    )

                # Get data
                cube = files.hdu2cube(hdu_index=d, dtype=np.float32)

                # Read linearity coefficients
                lcff = master_linearity.hdu2coeff(hdu_index=d)

                # Scale cube with NDIT (each DIT starts from a fresh reset in DCR mode,
                # so non-linearity is at the per-DIT level; normalise first so the
                # signal level matches the NDIT=1 linearity calibration frames)
                cube.scale_planes(scales=1 / files.ndit_norm)

                # Linearize data (use DIT, not DIT×NDIT: after NDIT normalisation the
                # signal represents one DIT integration, matching the NDIT=1
                # calibration frames; the reset-overhead factor kk = 1.0011/DIT
                # is only correct at the per-DIT level)
                cube.linearize(coeff=lcff, texptime=files.dit_norm)

                # Masking methods
                bpm = master_bpm.hdu2cube(hdu_index=d, dtype=bool)
                cube.apply_masks(
                    bpm=bpm,
                    mask_min=self.setup.dark_mask_min,
                    mask_max=self.setup.dark_mask_max,
                )

                # Discard a plane if only NaNs
                good_planes = cube.discard_nan_planes(threshold=0.9)
                bpm = bpm[good_planes]

                # Destripe
                if self.setup.destripe:
                    cube.destripe(masks=bpm, combine_bad_planes=False, smooth=True)

                # Flatten data
                collapsed = cube.flatten(
                    metric=string2func(self.setup.dark_metric), axis=0, dtype=None
                )

                # Determine dark current and std
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    dc, _, dc_std = sigma_clipped_stats(collapsed)

                # Norm to 1s also via DIT
                dc, dc_std = dc / files.dit[0], dc_std / files.dit[0]
                log.info(f"Detector {d}: dark current = {dc:.4f} ± {dc_std:.4f} ADU/s")

                # Write DC into data header
                header = fits.Header()
                add_int_to_header(
                    header=header,
                    key="HIERARCH PYPE DC NIMAGES",
                    value=len(files),
                    comment="Number of darkframes used",
                )
                add_float_to_header(
                    header=header,
                    key="HIERARCH PYPE DC",
                    value=dc,
                    decimals=3,
                    comment="Dark current in ADU/s",
                )
                add_float_to_header(
                    header=header,
                    key="HIERARCH PYPE DC STD",
                    value=dc_std,
                    decimals=3,
                    comment="Dark current standard deviation in ADU/s",
                )
                data_headers.append(header)

                # Append to output
                master_cube.extend(data=collapsed.astype(np.float32))

            # Make cards for primary headers
            prime_cards = make_cards(
                keywords=[
                    self.setup.keywords.dit,
                    self.setup.keywords.ndit,
                    self.setup.keywords.date_mjd,
                    self.setup.keywords.date_ut,
                    self.setup.keywords.object,
                    "HIERARCH PYPE N_FILES",
                ],
                values=[
                    files.dit[0],
                    files.ndit[0],
                    files.mjd_mean,
                    files.time_obs_mean,
                    "MASTER-DARK",
                    len(files),
                ],
            )
            prime_header = fits.Header(cards=prime_cards)

            # Write to disk
            master_cube.write_mef(
                path=outpath, prime_header=prime_header, data_headers=data_headers
            )
            log.info(f"Written: {outpath}")

            # QC plot
            if self.setup.qc_plots:
                mdark = MasterDark(setup=self.setup, file_paths=outpath)
                mdark.qc_plot_dark(paths=None, axis_size=5)

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )


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
            paths = [
                f"{self.setup.folders['qc_dark']}{fp}.pdf" for fp in self.basenames
            ]

        # Loop over files and create plots
        for path, dc, dcstd in zip(paths, self.darkcurrent, self.darkcurrent_std):
            plot_value_detector(
                values=dc,
                errors=dcstd,
                path=path,
                ylabel="Dark Current (ADU/s)",
                axis_size=axis_size,
                hlines=[0],
            )
