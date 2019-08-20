# =========================================================================== #
# Import
from vircampype.setup import *
from vircampype.data.cube import ImageCube
from vircampype.utils.miscellaneous import *
from vircampype.fits.images.common import FitsImages
from vircampype.utils.math import estimate_background
from vircampype.utils.plots import plot_value_detector


class DarkImages(FitsImages):

    def __init__(self, file_paths=None):
        super(DarkImages, self).__init__(file_paths=file_paths)

    def build_master_dark(self):
        """ Create master darks. """

        # Processing info
        tstart = mastercalibration_message(master_type="MASTER-DARK", silent=setup_silent)

        """ This does not work with Dark normalisation. There has to be a set for each DIT/NDIT combination! """
        # Split files first on DIT and NDIT, then on lag
        split = self._split_expsequence()
        split = flat_list([s.split_lag(max_lag=setup_dark_maxlag) for s in split])

        # Now loop through separated files and build the Masterdarks
        outpaths = []
        for files, fidx in zip(split, range(1, len(split) + 1)):  # type: DarkImages, int

            # Check sequence suitability for Dark (same number of HDUs and NDIT)
            files.check_compatibility(n_hdu_max=1, n_ndit_max=1)

            # Create master dark name
            outpaths.append(files.create_masterpath(basename="MASTER-DARK", idx=0, dit=True, ndit=True, mjd=True))

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpaths[-1], silent=setup_silent) and not setup_overwrite:
                continue

            # Instantiate output
            master_dark = ImageCube()

            # Get Masterbpm if set
            master_bpm = files.match_masterbpm()

            # Start looping over detectors
            data_headers = []
            for d in files.data_hdu[0]:

                # Print processing info
                if not setup_silent:
                    calibration_message(n_current=fidx, n_total=len(split), name=outpaths[-1],
                                        d_current=d, d_total=max(files.data_hdu[0]))

                # Get data
                cube = files.hdu2cube(hdu_index=d, dtype=np.float32)

                # Get master calibration
                bpm = master_bpm.hdu2cube(hdu_index=d, dtype=np.uint8)

                # Masking methods
                cube.apply_masks(mask_min=setup_dark_maskmin, mask_max=setup_dark_maskmax, bpm=bpm)

                # Collapse extensions
                collapsed = cube.flatten(metric=str2func(setup_dark_metric))

                # Determine dark current
                dc, _ = estimate_background(collapsed, max_iter=10, force_clipping=True)
                dc /= (files.dit[0] * files.ndit[0])  # This only works because we do not normalize

                # Write DC into data header
                cards = make_cards(keywords=["HIERARCH PYPE DC"], values=[np.round(dc, decimals=3)],
                                   comments=["Dark current in ADU/s"])
                data_headers.append(fits.Header(cards=cards))

                # Append to output
                master_dark.extend(data=collapsed.astype(np.float32))

            # Make cards for primary headers
            prime_cards = make_cards(keywords=[setup_kw_dit, setup_kw_ndit,  setup_kw_mjd,
                                               setup_kw_dateut, setup_kw_object, "HIERARCH PYPE N_FILES"],
                                     values=[files.dit[0], files.ndit[0],  files.mjd_mean,
                                             files.time_obs_mean, "MASTER-DARK", len(files)])
            prime_header = fits.Header(cards=prime_cards)

            # Write to disk
            master_dark.write_mef(path=outpaths[-1], prime_header=prime_header, data_headers=data_headers)

        # QC plot
        if setup_qc_plots:
            mdark = MasterDark(file_paths=outpaths)
            mdark.qc_plot_dark(paths=None, axis_size=5, overwrite=setup_overwrite)

        # Print time
        if not setup_silent:
            print("-> Elapsed time: {0:.2f}s".format(time.time() - tstart))


class MasterDark(FitsImages):

    def __init__(self, file_paths=None):
        super(MasterDark, self).__init__(file_paths=file_paths)

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
            paths = [x.replace(".fits", ".pdf") for x in self.full_paths]

        # Loop over files and create plots
        for dc, path in zip(self.darkcurrent, paths):
            plot_value_detector(values=dc, path=path, ylabel="Dark Current (ADU/s)",
                                axis_size=axis_size, overwrite=overwrite)
