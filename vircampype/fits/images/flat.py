# =========================================================================== #
# Import
from vircampype.data.cube import ImageCube
from vircampype.utils.miscellaneous import *
from vircampype.utils.math import linearize_data
from vircampype.fits.images.dark import MasterDark
from vircampype.fits.images.common import FitsImages
from vircampype.fits.images.bpm import MasterBadPixelMask
from vircampype.fits.tables.linearity import MasterLinearity


class FlatImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(FlatImages, self).__init__(setup=setup, file_paths=file_paths)

    def build_master_bpm(self):
        """ Builds a Bad pixel mask from image data. """

        # Processing info
        tstart = mastercalibration_message(master_type="MASTER-BPM", silent=self.setup["misc"]["silent"])

        # Split files based on maximum time lag is set
        split = self.split_lag(max_lag=self.setup["bpm"]["max_lag"])

        # Now loop through separated files and build Masterbpm
        for files, fidx in zip(split, range(1, len(split) + 1)):  # type: FlatImages, int

            # Check sequence compatibility
            files.check_compatibility(n_hdu_max=1, n_dit_max=1, n_ndit_max=1, n_files_min=3)

            # Create Masterbpm name
            outpath = files.create_masterpath(basename="MASTER-BPM", ndit=True, mjd=True)

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]) \
                    and not self.setup["misc"]["overwrite"]:
                continue

            # Instantiate output
            master_cube = ImageCube()

            # Start looping over detectors
            data_headers = []
            for d in files.data_hdu[0]:

                # Print processing info
                if not self.setup["misc"]["silent"]:
                    calibration_message(n_current=fidx, n_total=len(split), name=outpath,
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
                master_cube.extend(data=bpm)

            # Make cards for primary headers
            prime_cards = make_cards(keywords=[self.setup["keywords"]["dit"], self.setup["keywords"]["ndit"],
                                               self.setup["keywords"]["date_mjd"], self.setup["keywords"]["date_ut"],
                                               self.setup["keywords"]["object"], "HIERARCH PYPE N_FILES"],
                                     values=[files.dit[0], files.ndit[0],
                                             files.mjd_mean, files.time_obs_mean,
                                             "MASTER-BPM", len(files)])
            prime_header = fits.Header(cards=prime_cards)

            # Write to disk
            master_cube.write_mef(path=outpath, prime_header=prime_header, data_headers=data_headers)

            # QC plot
            if self.setup["misc"]["qc_plots"]:
                mbpm = MasterBadPixelMask(setup=self.setup, file_paths=outpath)
                mbpm.qc_plot_bpm(paths=None, axis_size=5, overwrite=self.setup["misc"]["overwrite"])

        # Print time
        if not self.setup["misc"]["silent"]:
            print("-> Elapsed time: {0:.2f}s".format(time.time() - tstart))

    def build_master_linearity(self):
        """ Calculates the non-linearity coefficients based on a series of dome flats. """

        # Order can't be greater than 3 at the moment
        if self.setup["linearity"]["order"] not in [2, 3]:
            raise NotImplementedError("Order not supported")

        # Processing info
        tstart = mastercalibration_message(master_type="MASTER-LINEARITY", silent=self.setup["misc"]["silent"])

        # Split based on lag and filter
        split = self.split_filter()
        split = flat_list([s.split_lag(max_lag=self.setup["gain"]["max_lag"]) for s in split])

        # Now loop through separated files and build the Masterdarks
        for files, fidx in zip(split, range(1, len(split) + 1)):  # type: FlatImages, int

            # Check sequence suitability for linearity (same nHDU, at least five different exposure times and same NDIT)
            files.check_compatibility(n_hdu_max=1, n_dit_min=5, n_ndit_max=1)

            # Create master name
            outpath = files.create_masterpath(basename="MASTER-LINEARITY", idx=0, mjd=True, table=True)

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]) \
                    and not self.setup["misc"]["overwrite"]:
                continue

            # Fetch the Masterfiles
            master_bpms = files.match_masterbpm()  # type: MasterBadPixelMask
            master_darks = files.match_masterdark()  # type: MasterDark

            # initialize empty lists
            table_hdus = []

            # Start looping over detectors
            for d in files.data_hdu[0]:

                # Print processing info
                calibration_message(n_current=fidx, n_total=len(split), name=outpath, d_current=d,
                                    d_total=max(files.data_hdu[0]), silent=self.setup["misc"]["silent"])

                # Get data
                cube = files.hdu2cube(hdu_index=d, dtype=np.float32)

                # Get master calibration
                bpm = master_bpms.hdu2cube(hdu_index=d, dtype=np.uint8)
                dark = master_darks.hdu2cube(hdu_index=d, dtype=np.float32)
                sat = self.get_saturation_hdu(hdu_index=d-1)
                norm_before = files.ndit_norm

                # Do calibration
                cube.calibrate(dark=dark, norm_before=norm_before, mask=bpm)

                # Estimate flux for each plane as the median
                flux = cube.median(axis=(1, 2))

                # Get flux filter
                badflux = flux > sat

                # Get dit
                dit = np.array(files.dit)

                # Fit a polynomial through good fluxes
                coeff = np.polyfit(dit[~badflux], flux[~badflux], deg=self.setup["linearity"]["order"])

                # The coefficients now must be scaled (check VISTA DR library design document for an explanation)
                coeff_norm = np.divide(coeff, coeff[-2] ** np.arange(self.setup["linearity"]["order"] + 1)[::-1])

                # Round coefficients
                coeff, coeff_norm = list(np.around(coeff, decimals=12)), list(np.around(coeff_norm, decimals=12))

                # Determine non-linearity@10000ADU
                nl10000 = (linearize_data(data=np.array([10000]), coeff=coeff_norm) / 10000 - 1)[0] * 100
                nl10000 = np.round(nl10000, decimals=5)

                # Make fits cards for coefficients
                cards_poly_coeff, cards_norm_coeff = [], []
                for cidx in range(len(coeff)):
                    cards_poly_coeff.append(fits.Card(keyword="HIERARCH PYPE COEFF POLY {0}".format(cidx),
                                                      value=coeff[cidx],
                                                      comment="Polynomial coefficient {0}".format(cidx)))
                    cards_norm_coeff.append(fits.Card(keyword="HIERARCH PYPE COEFF LINEAR {0}".format(cidx),
                                                      value=coeff_norm[cidx],
                                                      comment="Linearity coefficient {0}".format(cidx)))

                # Make header cards
                cards = make_cards(keywords=["HIERARCH PYPE QC NL10000", "HIERARCH PYPE QC SATURATION",
                                             "HIERARCH PYPE LIN ORDER"],
                                   values=[nl10000, sat, self.setup["linearity"]["order"]],
                                   comments=["Non-linearity at 10000 ADU in %", "Saturation limit (ADU)",
                                             "Order of linearity fit"])

                # Merge cards
                cards = cards + cards_poly_coeff + cards_norm_coeff

                # Add dit and flux to another HDU
                table_hdus.append(fits.TableHDU.from_columns(columns=[fits.Column(name="dit", format="D", array=dit),
                                                                      fits.Column(name="flux", format="D", array=flux)],
                                                             header=fits.Header(cards=cards)))

            # Make cards for primary headers
            prime_cards = make_cards(keywords=[self.setup["keywords"]["date_mjd"], self.setup["keywords"]["date_ut"],
                                               self.setup["keywords"]["object"], "HIERARCH PYPE N_FILES"],
                                     values=[files.mjd_mean, files.time_obs_mean,
                                             "MASTER-LINEARITY", len(files)])
            prime_header = fits.Header(cards=prime_cards)

            # Save table
            thdulist = fits.HDUList([fits.PrimaryHDU(header=prime_header)] + table_hdus)
            thdulist.writeto(fileobj=outpath, overwrite=self.setup["misc"]["overwrite"])

            # Initialize plot if set
            if self.setup["misc"]["qc_plots"]:
                mlinearity = MasterLinearity(setup=self.setup, file_paths=outpath)
                mlinearity.qc_plot_linearity(paths=None, axis_size=5, overwrite=self.setup["misc"]["overwrite"])

        # Print time
        if not self.setup["misc"]["silent"]:
            print("-> Elapsed time: {0:.2f}s".format(time.time() - tstart))
