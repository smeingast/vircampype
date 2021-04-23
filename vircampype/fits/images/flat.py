import time
import warnings
import numpy as np

from astropy.io import fits
from scipy.optimize import curve_fit
from vircampype.tools.plottools import *
from vircampype.tools.messaging import *
from vircampype.tools.mathtools import *
from vircampype.tools.fitstools import *
from vircampype.data.cube import ImageCube
from vircampype.tools.miscellaneous import *
from vircampype.fits.tables.gain import MasterGain
from vircampype.fits.images.bpm import MasterBadPixelMask
from vircampype.fits.tables.linearity import MasterLinearity
from vircampype.fits.images.common import FitsImages, MasterImages


class FlatImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(FlatImages, self).__init__(setup=setup, file_paths=file_paths)


class FlatTwilight(FlatImages):

    def __init__(self, setup, file_paths=None):
        super(FlatTwilight, self).__init__(setup=setup, file_paths=file_paths)

    def build_master_flat(self):
        """
        Builds a masterflat from the given flat fields. Also calculates a global gain harmonization which is applied
        to all detectors. Different filters (as reported in the fits header) will be split into different masterflats.
        In addition, if masterweights should be created, this method will also create them after building all
        masterflats.

        """

        # Processing info
        print_header(header="MASTER-FLAT", right=None, silent=self.setup.silent)
        tstart = time.time()

        # Split based on lag and filter
        split = self.split_keywords(keywords=[self.setup.keywords.filter_name])
        split = flat_list([s.split_lag(max_lag=self.setup.flat_max_lag) for s in split])

        # Now loop through separated files and build the Masterdarks
        for files, fidx in zip(split, range(1, len(split) + 1)):

            # Check flat sequence (at least three files, same nHDU, same NDIT, and same filter)
            files.check_compatibility(n_files_min=3, n_hdu_max=1, n_ndit_max=1, n_filter_max=1)

            # Create Master name
            outpath = "{0}MASTER-FLAT.MJD_{1:0.4f}.FIL_{2}.fits" \
                      "".format(files.setup.folders["master_common"], files.mjd_mean, files.passband[0])

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent):
                continue

            # Fetch the Masterfiles
            master_bpms = files.get_master_bpm()
            master_darks = files.get_master_dark(ignore_dit=True)
            master_linearity = files.get_master_linearity()

            for master in [master_bpms, master_darks, master_linearity]:
                if len(master) != len(files):
                    raise ValueError("Fetched Master sequences do not match")

            # initialize empty lists and data structures
            master_flat, flux = ImageCube(setup=self.setup, cube=None), []

            # Start looping over detectors
            data_headers = []
            for d in files.iter_data_hdu[0]:

                # Print processing info
                message_calibration(n_current=fidx, n_total=len(split), name=outpath, d_current=d,
                                    d_total=max(files.iter_data_hdu[0]), silent=self.setup.silent)

                # Get data
                cube = files.hdu2cube(hdu_index=d, dtype=np.float32)

                # Get master calibration
                bpm = master_bpms.hdu2cube(hdu_index=d, dtype=np.uint8)
                dark = master_darks.hdu2cube(hdu_index=d, dtype=np.float32)
                dark.scale_planes(scales=files.dit_norm)
                lin = master_linearity.hdu2coeff(hdu_index=d)
                sat = self.setup.saturation_levels[d-1]
                norm_before = files.ndit_norm

                # Do calibration
                cube.process_raw(dark=dark, linearize=(lin, files.dit), norm_before=norm_before)

                # Apply masks (only BPM and saturation before scaling)
                cube.apply_masks(bpm=bpm, mask_above=sat)

                # Determine flux for each plane of the cube
                flux.append(cube.median(axis=(1, 2)))

                # Scale the cube with the fluxes
                cube.scale_planes(scales=1/np.array(flux[-1]))

                # After flux scaling we can also safely apply the remaining masks
                cube.apply_masks(mask_min=self.setup.flat_mask_min, mask_max=self.setup.flat_mask_max,
                                 mask_below=self.setup.flat_rel_lo, mask_above=self.setup.flat_rel_hi,
                                 sigma_level=self.setup.flat_sigma_level, sigma_iter=self.setup.flat_sigma_iter)

                # Create weights if needed
                if self.setup.flat_metric == "weighted":
                    weights = np.empty_like(cube.cube)
                    weights[:] = flux[-1][:, np.newaxis, np.newaxis]
                    weights[~np.isfinite(cube.cube)] = 0.
                else:
                    weights = None

                # Flatten data
                flat = cube.flatten(metric=self.setup.flat_metric, axis=0, weights=weights, dtype=None)

                # Create header with flux measurements
                cards_flux = []
                for cidx in range(len(flux[-1])):
                    cards_flux.append(make_cards(keywords=["HIERARCH PYPE FLAT FLUX {0}".format(cidx),
                                                           "HIERARCH PYPE FLAT MJD {0}".format(cidx)],
                                                 values=[np.round(flux[-1][cidx], 3), np.round(files.mjd[cidx], 5)],
                                                 comments=["Measured flux (ADU)", "MJD of measured flux"]))
                data_headers.append(fits.Header(cards=flat_list(cards_flux)))

                # Flatten cube
                master_flat.extend(data=flat.astype(np.float32))

            # Convert to array
            flux = np.array(flux)

            # Determine exposure scale
            scale_exposure = np.mean(flux, axis=0)

            # Norm fluxes
            flux_normed = flux / scale_exposure

            # Now we can determine the flat field error
            flat_error = np.std(flux_normed, axis=1)

            # And the global flat field coefficients
            coeff_global = np.mean(flux_normed, axis=1)

            # Finally, I apply the global coefficients to the normed master flat
            master_flat.scale_planes(scales=coeff_global)

            # Make primary header
            prime_header = fits.Header()
            add_float_to_header(header=prime_header, key=self.setup.keywords.dit,
                                value=files.dit[0], decimals=2)
            add_int_to_header(header=prime_header, key=self.setup.keywords.ndit,
                              value=files.ndit[0])
            prime_header[self.setup.keywords.filter_name] = files.passband[0]
            add_float_to_header(header=prime_header, key=self.setup.keywords.date_mjd,
                                value=files.mjd_mean, decimals=6)
            prime_header[self.setup.keywords.date_ut] = files.time_obs_mean.fits
            prime_header[self.setup.keywords.object] = "MASTER-FLAT"
            add_int_to_header(header=prime_header, key="HIERARCH PYPE N_FILES", value=len(files))

            # Add global scale factors and flat field error to data headers
            for dh, gs, fe in zip(data_headers, coeff_global, flat_error):
                add_float_to_header(header=dh, key="HIERARCH PYPE FLAT SCALE",
                                    value=gs, decimals=4)
                add_float_to_header(header=dh, key="HIERARCH PYPE FLAT ERROR",
                                    value=fe, decimals=4)

            # Write to disk
            master_flat.write_mef(path=outpath, prime_header=prime_header, data_headers=data_headers)

            # QC plot
            if self.setup.qc_plots:
                mflat = MasterFlat(setup=self.setup, file_paths=outpath)
                mflat.qc_plot_flat(paths=None, axis_size=5, overwrite=self.setup.overwrite)

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

    # =========================================================================== #
    # Master Weight
    # =========================================================================== #
    def build_master_weight_global(self):
        """
        Creates master weights from master flats. The difference between them is that NaNs are replaced with 0s and
        there is an additional option to mask relative and absolute values.

        """

        # Processing info
        print_header(header="MASTER-WEIGHT-GLOBAL", right=None, silent=self.setup.silent)
        tstart = time.time()

        # Get unique Master flats
        master_flats = self.get_unique_master_flats()

        # Generate outpaths
        outpaths = [x.replace("MASTER-FLAT", "MASTER-WEIGHT-GLOBAL") for x in master_flats.paths_full]

        # Loop over files and apply calibration
        for idx in range(len(master_flats)):

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpaths[idx], silent=self.setup.silent) \
                    and not self.setup.overwrite:
                continue

            # Print processing info
            if not self.setup.silent:
                message_calibration(n_current=idx + 1, n_total=len(master_flats),
                                    name=outpaths[idx], d_current=None, d_total=None)

            # Read file into cube
            cube = master_flats.file2cube(file_index=idx, hdu_index=None, dtype=None)

            # Apply absolute masks
            cube.apply_masks(mask_below=self.setup.weight_mask_abs_min, mask_above=self.setup.weight_mask_abs_max)

            # Get median for each plane
            median = cube.median(axis=(1, 2))[:, np.newaxis, np.newaxis]

            # Norm each plane by its median
            cube /= median

            # Mask relative values
            cube.apply_masks(mask_below=self.setup.weight_mask_rel_min, mask_above=self.setup.weight_mask_rel_max)

            # Scale back to original
            cube *= median

            # Interpolate NaNs if set (cosmetic function can't be used since this would also apply e.g. de-striping)
            if self.setup.interpolate_nan_bool:
                cube.interpolate_nan()

            # Replace remaining NaNs with 0 weight
            cube.replace_nan(value=0)

            # Modify type in primary header
            prime_header = master_flats.headers_primary[idx].copy()
            prime_header[self.setup.keywords.object] = "MASTER-WEIGHT-GLOBAL"

            # Write to file
            cube.write_mef(path=outpaths[idx], prime_header=prime_header, data_headers=master_flats.headers_data[idx])

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")


class FlatLampLin(FlatImages):

    def __init__(self, setup, file_paths=None):
        super(FlatLampLin, self).__init__(setup=setup, file_paths=file_paths)

    # =========================================================================== #
    # Master Linearity
    # =========================================================================== #
    def build_master_linearity(self, darks):

        # Processing info
        print_header(header="MASTER-LINEARITY", silent=self.setup.silent)
        tstart = time.time()

        # Split based on lag and filter
        split_flat = self.split_lag(max_lag=self.setup.linearity_max_lag)
        split_dark = darks.split_lag(max_lag=self.setup.linearity_max_lag)

        for sflats, sdarks, fidx in zip(split_flat, split_dark, range(len(split_flat))):

            # Check exposure sequence
            if (sflats.dit != sdarks.dit) | (sflats.ndit != sdarks.ndit):
                raise ValueError("Linearity flat/dark sequence is broken")

            # Check sequence suitability for linearity (same nHDU, at least five different exposure times and same NDIT)
            sflats.check_compatibility(n_hdu_max=1, n_dit_min=5, n_ndit_max=1)

            # Create Master name
            outpath = "{0}MASTER-LINEARITY.MJD_{1:0.4f}.fits.tab" \
                      "".format(sflats.setup.folders["master_common"], sflats.mjd_mean)

            # Find BPMs
            master_bpm = sflats.get_master_bpm()

            # Initialize empty list for table HDUs
            table_hdus = []

            # Start detector loop
            for idx_hdu in sflats.iter_data_hdu[0]:

                # Print processing info
                message_calibration(n_current=fidx+1, n_total=len(split_flat), name=outpath, d_current=idx_hdu,
                                    d_total=max(sflats.iter_data_hdu[0]), silent=self.setup.silent)

                # Read data
                cube_flat = sflats.hdu2cube(hdu_index=idx_hdu)
                cube_dark = sdarks.hdu2cube(hdu_index=idx_hdu)
                cube_bpm = master_bpm.hdu2cube(hdu_index=idx_hdu)

                # Subtract dark from flat
                cube_flat -= cube_dark

                # Mask bad pixels
                cube_flat.apply_masks(bpm=cube_bpm)

                # Determine flux and flux sigma
                flux, flux_err = cube_flat.background_planes()

                # Grab saturation level
                satlevel = self.setup.saturation_levels[idx_hdu-1]

                # Find values above saturation limit
                saturated = flux > satlevel

                # Grab clean DIT and clean flux
                flux_clean, flux_err_clean = flux[~saturated], flux_err[~saturated]
                dit = np.array(sflats.dit)
                dit_clean = dit[~saturated]

                # Do curve fit
                coeff, _ = curve_fit(linearity_fitfunc, dit_clean, flux_clean,
                                     sigma=flux_err_clean, absolute_sigma=True)

                # Compute normalized final coefficients
                coeff_norm = [coeff[i] / coeff[0]**(i+1) for i in range(0, len(coeff))]

                # Add 0 order term
                coeff_norm = np.insert(coeff_norm, 0, 0)

                # Compute non-linearity at 10000 ADU for DIT=2
                nl10000 = (linearize_data(data=np.array([10000]), coeff=coeff_norm, dit=2,
                                          reset_read_overhead=1.0011) / 10000 - 1)[0] * 100

                # Linearize data
                flux_lin = []
                for f, t in zip(flux, dit):
                    flux_lin.append(linearize_data(data=f, coeff=coeff_norm, dit=t, reset_read_overhead=1.0011))
                flux_lin = np.asarray(flux_lin)

                # Make fits cards for coefficients
                cards_coeff = []
                for cidx in range(len(coeff_norm)):
                    cards_coeff.append(fits.Card(keyword="HIERARCH PYPE COEFF LINEAR {0}".format(cidx),
                                                 value=coeff_norm[cidx],
                                                 comment="Linearity coefficient {0}".format(cidx)))

                # Create header
                hdr = fits.Header(cards=cards_coeff)

                # Add some more values
                add_float_to_header(header=hdr, key="HIERARCH PYPE QC SATURATION", value=satlevel,
                                    decimals=1, comment="Saturation limit (ADU)")
                add_float_to_header(header=hdr, key="HIERARCH PYPE QC NL10000", value=nl10000,
                                    decimals=3, comment="Non-linearity at 10000 ADU for DIT=2 in %")

                # Linearize data
                flux_clean_lin = []
                for f, t in zip(flux_clean, dit_clean):
                    flux_clean_lin.append(linearize_data(data=f, coeff=coeff_norm, dit=t,
                                                         reset_read_overhead=1.0011))

                # Add data to HDU
                cdit = fits.Column(name="dit", format="D", array=dit)
                cflux = fits.Column(name="flux", format="D", array=flux)
                cflux_lin = fits.Column(name="flux_lin", format="D", array=flux_lin)
                cmask = fits.Column(name="saturated", format="L", array=saturated)
                table_hdus.append(fits.TableHDU.from_columns(columns=[cdit, cflux, cflux_lin, cmask], header=hdr))

            # Make cards for primary headers
            prime_cards = make_cards(keywords=[self.setup.keywords.date_mjd, self.setup.keywords.date_ut,
                                               self.setup.keywords.object, "HIERARCH PYPE N_FILES"],
                                     values=[sflats.mjd_mean, sflats.time_obs_mean,
                                             "MASTER-LINEARITY", len(sflats)])
            prime_header = fits.Header(cards=prime_cards)

            # Save table
            thdulist = fits.HDUList([fits.PrimaryHDU(header=prime_header)] + table_hdus)
            thdulist.writeto(fileobj=outpath, overwrite=self.setup.overwrite)

            # Initialize plot if set
            # if self.setup.qc_plots:
            #     mlinearity = MasterLinearity(setup=self.setup, file_paths=outpath)
            #     mlinearity.qc_plot_linearity_detector(paths=None, axis_size=5, overwrite=self.setup.overwrite)
            #     mlinearity.qc_plot_linearity(paths=None, axis_size=5, overwrite=self.setup.overwrite)
            #     mlinearity.qc_plot_linearity_2(paths=None, axis_size=5, overwrite=self.setup.overwrite)

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")


class FlatLampCheck(FlatImages):

    def __init__(self, setup, file_paths=None):
        super(FlatLampCheck, self).__init__(setup=setup, file_paths=file_paths)

    # =========================================================================== #
    # Master Bad Pixel Mask
    # =========================================================================== #
    def build_master_bpm(self):
        """ Builds a Bad pixel mask from image data. """

        # Processing info
        print_header(header="MASTER-BPM", silent=self.setup.silent)
        tstart = time.time()

        # Split files based on maximum time lag is set
        split = self.split_lag(max_lag=self.setup.bpm_max_lag)

        # Now loop through separated files and build Masterbpm
        for files, idx_print in zip(split, range(1, len(split) + 1)):

            # Check sequence compatibility
            files.check_compatibility(n_hdu_max=1, n_dit_max=1, n_ndit_max=1, n_files_min=3)

            # Create Master name
            outpath = "{0}MASTER-BPM.NDIT_{1}.MJD_{2:0.4f}.fits" \
                      "".format(files.setup.folders["master_common"], files.ndit[0], files.mjd_mean)

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent) \
                    and not self.setup.overwrite:
                continue

            # Instantiate output
            master_cube = ImageCube(setup=self.setup)

            # Start looping over detectors
            data_headers = []
            for d in files.iter_data_hdu[0]:

                # Print processing info
                if not self.setup.silent:
                    message_calibration(n_current=idx_print, n_total=len(split), name=outpath,
                                        d_current=d, d_total=len(files.iter_data_hdu[0]))

                # Get data
                cube = files.hdu2cube(hdu_index=d, dtype=np.float32)

                # Mask low and high absolute values
                cube.apply_masks(mask_below=self.setup.bpm_abs_lo, mask_above=self.setup.bpm_abs_hi)

                # Sigma clipping per plane
                cube.apply_masks_plane(sigma_level=self.setup.bpm_sigma_level, sigma_iter=self.setup.bpm_sigma_iter)

                # Collapse cube with median
                flat = cube.flatten(metric=string2func(self.setup.bpm_metric), axis=0)

                # Normalize cube with flattened data
                cube.cube = cube.cube / flat

                # Mask low and high relative values
                cube.apply_masks(mask_below=self.setup.bpm_rel_lo, mask_above=self.setup.bpm_rel_hi)

                # Count how many bad pixels there are in the stack and normalize to the number of input images
                nbad_pix = np.sum(~np.isfinite(cube.cube), axis=0) / files.n_files

                # Get those pixels where the number of bad pixels is greater than the given input threshold
                bpm = np.array(nbad_pix > self.setup.bpm_frac, dtype=np.uint8)

                # Make header cards
                cards = make_cards(keywords=["HIERARCH PYPE NBADPIX", "HIERARCH PYPE BADFRAC"],
                                   values=[int(np.sum(bpm)), np.round(np.sum(bpm) / bpm.size, decimals=5)],
                                   comments=["Number of bad pixels", "Fraction of bad pixels"])
                data_headers.append(fits.Header(cards=cards))

                # Append HDU
                master_cube.extend(data=bpm)

            # Make cards for primary headers
            prime_cards = make_cards(keywords=[self.setup.keywords.dit, self.setup.keywords.ndit,
                                               self.setup.keywords.date_mjd, self.setup.keywords.date_ut,
                                               self.setup.keywords.object, "HIERARCH PYPE N_FILES"],
                                     values=[files.dit[0], files.ndit[0],
                                             files.mjd_mean, files.time_obs_mean.fits,
                                             "MASTER-BPM", len(files)])
            prime_header = fits.Header(cards=prime_cards)

            # Write to disk
            master_cube.write_mef(path=outpath, prime_header=prime_header, data_headers=data_headers)

            # QC plot
            if self.setup.qc_plots:
                mbpm = MasterBadPixelMask(setup=self.setup, file_paths=outpath)
                mbpm.qc_plot_bpm(paths=None, axis_size=5)

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")


class FlatLampGain(FlatImages):

    def __init__(self, setup, file_paths=None):
        super(FlatLampGain, self).__init__(setup=setup, file_paths=file_paths)

    def build_master_gain(self, darks):
        """
        Preliminary (not universal) routine to calculate gain and Flat tables. For the moment only works with VIRCAM and
        maybe not even under all circumstance. The gain and read noise are calculated using Janesick's method.

        See e.g. Hand book of CCD astronomy.

        Parameters
        ----------
        darks : DarkImages
            Corresponding dark images for the method.


        """

        # Processing info
        print_header(header="MASTER-GAIN", right=None, silent=self.setup.silent)
        tstart = time.time()

        # Split based on lag
        split_flats = self.split_lag(max_lag=self.setup.gain_max_lag, sort_mjd=True)
        split_darks = darks.split_lag(max_lag=self.setup.gain_max_lag, sort_mjd=True)

        if len(split_flats) != len(split_darks):
            raise ValueError("Provided darks do not match to input flats!")

        # Now loop through separated files and build the Gain Table
        for idx in range(len(split_flats)):

            # Grab files
            flats, darks = split_flats[idx], split_darks[idx]

            # Check sequence suitability for Dark (same number of HDUs and NDIT)
            flats.check_compatibility(n_hdu_max=1, n_ndit_max=1, n_filter_max=1)
            if len(flats) != len(flats):
                raise ValueError("Gain sequence not compatible!")

            # Also DITs must match
            if (np.sum(np.abs(np.array(flats.dit) - np.array(darks.dit)) < 0.001)) != len(flats):
                raise ValueError("Gain sequence not compatible!")

            # Create master  name
            outpath = "{0}MASTER-GAIN.NDIT_{1}.MJD_{2:0.4f}.fits.tab" \
                      "".format(flats.setup.folders["master_common"], flats.ndit[0], flats.mjd_mean)

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent) \
                    and not self.setup.overwrite:
                continue

            # Print processing info
            if not self.setup.silent:
                message_calibration(n_current=idx+1, n_total=len(split_flats),
                                    name=outpath, d_current=None, d_total=None)

            # Get BPM
            mbpms = flats.get_master_bpm()

            # Read data
            f0 = flats.file2cube(file_index=0, dtype=np.float32)
            f1 = flats.file2cube(file_index=1, dtype=np.float32)
            d0 = darks.file2cube(file_index=0, dtype=np.float32)
            d1 = darks.file2cube(file_index=1, dtype=np.float32)
            m0 = mbpms.file2cube(file_index=0, dtype=np.uint8)
            m1 = mbpms.file2cube(file_index=1, dtype=np.uint8)

            # Mask bad pixels
            f0.apply_masks(bpm=m0), f1.apply_masks(bpm=m1)
            d0.apply_masks(bpm=m0), d1.apply_masks(bpm=m1)

            # Get variance in difference images
            fvar, dvar = (f0 - f1).var(axis=(1, 2)), (d0 - d1).var(axis=(1, 2))

            # Calculate gain
            gain = ((f0.mean(axis=(1, 2)) + f1.mean(axis=(1, 2))) -
                    (d0.mean(axis=(1, 2)) + d1.mean(axis=(1, 2)))) / (fvar - dvar)

            # Calculate readout noise
            rdnoise = gain * np.sqrt(dvar) / np.sqrt(2)

            # Make header cards
            prime_cards = make_cards(keywords=[self.setup.keywords.dit, self.setup.keywords.ndit,
                                               self.setup.keywords.date_mjd, self.setup.keywords.date_ut,
                                               self.setup.keywords.object, "HIERARCH PYPE N_FILES"],
                                     values=[flats.dit[0], flats.ndit[0],
                                             flats.mjd_mean, flats.time_obs_mean,
                                             "MASTER-GAIN", len(flats)])
            prhdu = fits.PrimaryHDU(header=fits.Header(cards=prime_cards))

            # Create table HDU for output
            tbhdu = fits.TableHDU.from_columns([fits.Column(name="gain", format="D", array=gain),
                                                fits.Column(name="rdnoise", format="D", array=rdnoise)])
            thdulist = fits.HDUList([prhdu, tbhdu])

            # Write
            thdulist.writeto(fileobj=outpath, overwrite=self.setup.overwrite)

            # QC plot
            if self.setup.qc_plots:
                mgain = MasterGain(setup=self.setup, file_paths=outpath)
                mgain.qc_plot_gain(paths=None, axis_size=5)
                mgain.qc_plot_rdnoise(paths=None, axis_size=5)

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")


class MasterFlat(MasterImages):

    def __init__(self, setup, file_paths=None):
        super(MasterFlat, self).__init__(setup=setup, file_paths=file_paths)

    # =========================================================================== #
    # Properties
    # =========================================================================== #
    _flux = None

    @property
    def flux(self):
        """
        Reads flux measurements of used flats into list.

        Returns
        -------
        List
            List of fluxes.

        """

        # Check if already determined
        if self._flux is not None:
            return self._flux

        self._flux = self._get_dataheaders_sequence(keyword="HIERARCH PYPE FLAT FLUX")
        return self._flux

    _flux_mjd = None

    @property
    def flux_mjd(self):
        """
        Reads MJDs of flux measurements of used flats into list.

        Returns
        -------
        List
            List of MJDs.

        """

        # Check if already determined
        if self._flux_mjd is not None:
            return self._flux_mjd

        self._flux_mjd = self._get_dataheaders_sequence(keyword="HIERARCH PYPE FLAT MJD")
        return self._flux_mjd

    @property
    def gainscale(self):
        """
        Read used gainscale from data headers.

        Returns
        -------
        List
            List of gainscales.

        """
        return self.read_from_data_headers(keywords=["HIERARCH PYPE FLAT SCALE"])[0]

    def paths_qc_plots(self, paths):
        """
        Generates paths for QC plots

        Parameters
        ----------
        paths : iterable
            Input paths to override internal paths

        Returns
        -------
        iterable
            List of paths.
        """

        if paths is None:
            return ["{0}{1}.pdf".format(self.setup.folders["qc_flat"], fp) for fp in self.basenames]
        else:
            return paths

    def qc_plot_flat(self, paths=None, axis_size=4, overwrite=False):
        """
        Creates the QC plot for the flat fields. Should only be used together with the above method.

        Parameters
        ----------
        paths : List, optional
            Paths of the QC plot files. If None (default), use relative path
        axis_size : int, float, optional
            Axis size. Default is 4.
        overwrite : optional, bool
            Whether an exisiting plot should be overwritten. Default is False.

        """

        # Import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Generate path for plots
        paths = self.paths_qc_plots(paths=paths)

        for flux, mjd, gs, path in zip(self.flux, self.flux_mjd, self.gainscale, paths):

            # Check if plot already exits
            if check_file_exists(file_path=path, silent=True) and not overwrite:
                continue

            # Get plot grid
            fig, axes = get_plotgrid(layout=self.setup.fpa_layout, xsize=axis_size, ysize=axis_size)
            axes = axes.ravel()

            # Helpers
            mjd_floor = np.floor(np.min(mjd))
            xmin, xmax = 0.9999 * np.min(24 * (mjd - mjd_floor)), 1.0001 * np.max(24 * (mjd - mjd_floor))
            allflux = flat_list(flux)
            ymin, ymax = 0.98 * np.min(allflux), 1.02 * np.max(allflux)

            # Loop and plot
            for idx in range(len(flux)):

                # Grab axes
                ax = axes[idx]

                # Plot flux
                ax.scatter(24 * (mjd[idx] - mjd_floor), flux[idx], c="#DC143C", lw=0, s=40, alpha=0.7, zorder=0)

                # Annotate detector ID and gain scale
                ax.annotate("Scale={0:.3f}".format(gs[idx]),
                            xy=(0.96, 0.96), xycoords="axes fraction", ha="right", va="top")
                ax.annotate("Det.ID: {0:0d}".format(idx + 1),
                            xy=(0.04, 0.04), xycoords="axes fraction", ha="left", va="bottom")

                # Modify axes
                if idx < self.setup.fpa_layout[1]:
                    ax.set_xlabel("MJD (h) + {0:0n}d".format(mjd_floor))
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if idx % self.setup.fpa_layout[0] == self.setup.fpa_layout[0] - 1:
                    ax.set_ylabel("ADU")
                else:
                    ax.axes.yaxis.set_ticklabels([])

                # Set ranges
                ax.set_xlim(xmin=floor_value(data=xmin, value=0.02), xmax=ceil_value(data=xmax, value=0.02))
                ax.set_ylim(ymin=floor_value(data=ymin, value=1000), ymax=ceil_value(data=ymax, value=1000))

                # Set ticks
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator())

                # Hide first tick label
                xticks, yticks = ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()
                xticks[0].set_visible(False)
                yticks[0].set_visible(False)

            # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
                fig.savefig(path, bbox_inches="tight")
            plt.close("all")


class MasterWeight(MasterImages):

    def __init__(self, setup, file_paths=None):
        super(MasterWeight, self).__init__(setup=setup, file_paths=file_paths)


class MasterSuperflat(MasterImages):

    def __init__(self, setup, file_paths=None):
        super(MasterSuperflat, self).__init__(setup=setup, file_paths=file_paths)

    @property
    def nsources(self):
        return self.read_from_data_headers(keywords=["HIERARCH PYPE SFLAT NSOURCES"])[0]

    @property
    def flx_std(self):
        return self.read_from_data_headers(keywords=["HIERARCH PYPE SFLAT STD"])[0]

    def qc_plot_superflat(self, paths=None, axis_size=4):

        # Import
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Generate path for plots
        if paths is None:
            paths = ["{0}{1}.pdf".format(self.setup.folders["qc_superflat"], fp) for fp in self.basenames]

        for idx_file in range(self.n_files):

            # Create figure
            fig, ax_file = get_plotgrid(layout=self.setup.fpa_layout, xsize=axis_size, ysize=axis_size)
            ax_file = ax_file.ravel()
            cax = fig.add_axes([0.3, 0.92, 0.4, 0.02])

            # Read data
            cube = self.file2cube(file_index=idx_file)

            # Determine vmin/vmax
            vmin, vmax = np.percentile(cube, 0.1), np.percentile(cube, 99.9)

            for idx_hdu in range(len(self.iter_data_hdu[idx_file])):

                # Fetch current axes
                ax = ax_file[idx_hdu]

                # Draw image
                im = ax.imshow(cube[idx_hdu], vmin=vmin, vmax=vmax, cmap=get_cmap("RdYlBu_r", 30), origin="lower")

                # Add colorbar
                cbar = plt.colorbar(mappable=im, cax=cax, orientation="horizontal", label="Relative Flux")
                cbar.ax.xaxis.set_ticks_position("top")
                cbar.ax.xaxis.set_label_position("top")

                # Limits
                ax.set_xlim(0, self.headers_data[idx_file][idx_hdu]["NAXIS1"] - 1)
                ax.set_ylim(0, self.headers_data[idx_file][idx_hdu]["NAXIS2"] - 1)

                # Annotate detector ID
                ax.annotate("Det.ID: {0:0d}".format(idx_hdu + 1), xy=(0.02, 1.005),
                            xycoords="axes fraction", ha="left", va="bottom")

                # Annotate number of sources used
                ax.annotate("N = {0:0d}".format(self.nsources[idx_file][idx_hdu]), xy=(0.98, 1.005),
                            xycoords="axes fraction", ha="right", va="bottom")

                # Modify axes
                if idx_hdu < self.setup.fpa_layout[1]:
                    ax.set_xlabel("X (pix)")
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if idx_hdu % self.setup.fpa_layout[0] == self.setup.fpa_layout[0] - 1:
                    ax.set_ylabel("Y (pix)")
                else:
                    ax.axes.yaxis.set_ticklabels([])

                # Set equal aspect ratio
                ax.set_aspect("equal")

                # Set ticks
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
                fig.savefig(paths[idx_file], bbox_inches="tight")
            plt.close("all")
