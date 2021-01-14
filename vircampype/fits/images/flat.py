# =========================================================================== #
# Import
import warnings
import numpy as np

from astropy.io import fits
from vircampype.utils.math import *
from vircampype.utils.plots import *
from vircampype.data.cube import ImageCube
from vircampype.utils.miscellaneous import *
from sklearn.neighbors import NearestNeighbors
from vircampype.fits.images.dark import MasterDark
from vircampype.fits.images.bpm import MasterBadPixelMask
from vircampype.fits.tables.linearity import MasterLinearity
from vircampype.fits.images.common import FitsImages, MasterImages


class FlatImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(FlatImages, self).__init__(setup=setup, file_paths=file_paths)

    # =========================================================================== #
    # Master Bad Pixel Mask
    # =========================================================================== #
    def build_master_bpm(self):
        """ Builds a Bad pixel mask from image data. """

        # Processing info
        tstart = message_mastercalibration(master_type="MASTER-BPM", silent=self.setup["misc"]["silent"])

        # Split files based on maximum time lag is set
        split = self.split_lag(max_lag=self.setup["bpm"]["max_lag"])

        # Now loop through separated files and build Masterbpm
        for files, idx_print in zip(split, range(1, len(split) + 1)):

            # Check sequence compatibility
            files.check_compatibility(n_hdu_max=1, n_dit_max=1, n_ndit_max=1, n_files_min=3)

            # Create Masterbpm name
            outpath = files.build_master_path(basename="MASTER-BPM", ndit=True, mjd=True)

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]) \
                    and not self.setup["misc"]["overwrite"]:
                continue

            # Instantiate output
            master_cube = ImageCube(setup=self.setup)

            # Start looping over detectors
            data_headers = []
            for d in files.data_hdu[0]:

                # Print processing info
                if not self.setup["misc"]["silent"]:
                    message_calibration(n_current=idx_print, n_total=len(split), name=outpath,
                                        d_current=d, d_total=len(files.data_hdu[0]))

                # Get data
                cube = files.hdu2cube(hdu_index=d, dtype=np.float32)

                # Mask low and high absolute values
                cube.apply_masks(mask_below=self.setup["bpm"]["abs_lo"], mask_above=self.setup["bpm"]["abs_hi"])

                # Sigma clipping per plane
                cube.apply_masks_plane(sigma_level=self.setup["bpm"]["sigma_level"],
                                       sigma_iter=self.setup["bpm"]["sigma_iter"])

                # Collapse cube with median
                flat = cube.flatten(metric=str2func(self.setup["bpm"]["metric"]), axis=0)

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
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    # =========================================================================== #
    # Master Linearity
    # =========================================================================== #
    # noinspection DuplicatedCode
    def build_master_linearity(self):
        """ Calculates the non-linearity coefficients based on a series of dome flats. """

        # Order can't be greater than 3 at the moment
        if self.setup["linearity"]["order"] not in [2, 3]:
            raise NotImplementedError("Order not supported")

        # Processing info
        tstart = message_mastercalibration(master_type="MASTER-LINEARITY", silent=self.setup["misc"]["silent"])

        # Split based on lag and filter
        split = self.split_filter()
        split = flat_list([s.split_lag(max_lag=self.setup["gain"]["max_lag"]) for s in split])

        # Now loop through separated files and build the Masterdarks
        for files, fidx in zip(split, range(1, len(split) + 1)):  # type: FlatImages, int

            # Check sequence suitability for linearity (same nHDU, at least five different exposure times and same NDIT)
            files.check_compatibility(n_hdu_max=1, n_dit_min=5, n_ndit_max=1)

            # Create master name
            outpath = files.build_master_path(basename="MASTER-LINEARITY", idx=0, mjd=True, table=True)

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]):
                continue

            # Fetch the Masterfiles
            master_bpms = files.get_master_bpm()  # type: MasterBadPixelMask
            master_darks = files.get_master_dark()  # type: MasterDark

            # initialize empty lists
            table_hdus = []

            # Start looping over detectors
            for d in files.data_hdu[0]:

                # Print processing info
                message_calibration(n_current=fidx, n_total=len(split), name=outpath, d_current=d,
                                    d_total=max(files.data_hdu[0]), silent=self.setup["misc"]["silent"])

                # Get data
                cube = files.hdu2cube(hdu_index=d, dtype=np.float32)

                # Get master calibration
                bpm = master_bpms.hdu2cube(hdu_index=d, dtype=np.uint8)
                dark = master_darks.hdu2cube(hdu_index=d, dtype=np.float32)
                sat = self.get_saturation_hdu(hdu_index=d)
                norm_before = files.ndit_norm

                # Do calibration
                cube.process_raw(dark=dark, norm_before=norm_before)

                # Apply BPM
                cube.apply_masks(bpm=bpm)

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
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    # =========================================================================== #
    # Master Flat
    # =========================================================================== #
    # noinspection DuplicatedCode
    def build_master_flat(self):
        """
        Builds a masterflat from the given flat fields. Also calculates a global gain harmonization which is applied
        to all detectors. Different filters (as reported in the fits header) will be split into different masterflats.
        In addition, if masterweights should be created, this method will also create them after building all
        masterflats.

        """

        # Processing info
        tstart = message_mastercalibration(master_type="MASTER-FLAT", silent=self.setup["misc"]["silent"])

        # Split based on lag and filter
        split = self.split_filter()
        split = flat_list([s.split_lag(max_lag=self.setup["flat"]["max_lag"]) for s in split])

        # Now loop through separated files and build the Masterdarks
        for files, fidx in zip(split, range(1, len(split) + 1)):  # type: FlatImages, int

            # Check flat sequence (at least three files, same nHDU, same NDIT, and same filter)
            files.check_compatibility(n_files_min=3, n_hdu_max=1, n_ndit_max=1, n_filter_max=1)

            # Create master name
            outpath = files.build_master_path(basename="MASTER-FLAT", idx=0, mjd=True, filt=True, table=False)

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]):
                continue

            # Fetch the Masterfiles
            master_bpms = files.get_master_bpm()  # type: MasterBadPixelMask
            master_darks = files.get_master_dark()  # type: MasterDark
            master_linearity = files.get_master_linearity()  # type: MasterLinearity

            # initialize empty lists and data structures
            master_flat, flux = ImageCube(setup=self.setup, cube=None), []

            # Start looping over detectors
            data_headers = []
            for d in files.data_hdu[0]:

                # Print processing info
                message_calibration(n_current=fidx, n_total=len(split), name=outpath, d_current=d,
                                    d_total=max(files.data_hdu[0]), silent=self.setup["misc"]["silent"])

                # Get data
                cube = files.hdu2cube(hdu_index=d, dtype=np.float32)

                # Get master calibration
                bpm = master_bpms.hdu2cube(hdu_index=d, dtype=np.uint8)
                dark = master_darks.hdu2cube(hdu_index=d, dtype=np.float32)
                lin = master_linearity.hdu2coeff(hdu_index=d)
                sat = files.get_saturation_hdu(hdu_index=d)
                norm_before = files.ndit_norm

                # Do calibration
                cube.process_raw(dark=dark, linearize=lin, norm_before=norm_before)

                # Apply masks (only BPM and saturation before scaling)
                cube.apply_masks(bpm=bpm, mask_above=sat)

                # Determine flux for each plane of the cube
                flux.append(cube.median(axis=(1, 2)))

                # Scale the cube with the fluxes
                cube.cube /= flux[-1][:, np.newaxis, np.newaxis]

                # After flux scaling we can also safely apply the remaining masks
                cube.apply_masks(mask_min=self.setup["flat"]["mask_min"], mask_max=self.setup["flat"]["mask_min"],
                                 mask_below=self.setup["flat"]["rel_lo"], mask_above=self.setup["flat"]["rel_hi"],
                                 sigma_level=self.setup["flat"]["sigma_level"],
                                 sigma_iter=self.setup["flat"]["sigma_iter"])

                # Create weights if needed
                if self.setup["flat"]["metric"] == "weighted":
                    weights = np.empty_like(cube.cube)
                    weights[:] = flux[-1][:, np.newaxis, np.newaxis]
                else:
                    weights = None

                # Flatten data
                flat = cube.flatten(metric=self.setup["flat"]["metric"], axis=0, weights=weights, dtype=None)

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

            """The current masterflat contains for each detector the flattened data after scaling each plane with the
            flux. Now we calculate a first order gain harmonization across the focal plane by calculating the relative
            fluxes between the detectors"""

            # Calculate an average exposure scale (scale for each image in the sequence preserving the relative flux)
            exposure_scale = [sum(f) / len(f) for f in zip(*flux)]

            # Now scale all measurements with the exposure scale
            flux_scaled = [[x / s for x, s in zip(f, exposure_scale)] for f in flux]

            # The gain harmonization factor is then the mean of all scaled exposures
            gainscale = [sum(f) / len(f) for f in flux_scaled]

            # Apply the gain harmonization
            master_flat.cube *= np.array(gainscale)[:, np.newaxis, np.newaxis]

            # Make cards for primary headers
            prime_cards = make_cards(keywords=[self.setup["keywords"]["dit"], self.setup["keywords"]["ndit"],
                                               self.setup["keywords"]["filter"],
                                               self.setup["keywords"]["date_mjd"], self.setup["keywords"]["date_ut"],
                                               self.setup["keywords"]["object"], "HIERARCH PYPE N_FILES"],
                                     values=[files.dit[0], files.ndit[0],
                                             files.filter[0],
                                             files.mjd_mean, files.time_obs_mean,
                                             "MASTER-FLAT", len(files)])
            prime_header = fits.Header(cards=prime_cards)

            # Add gainscale to data headers
            for dh, gs in zip(data_headers, gainscale):
                dh["HIERARCH PYPE FLAT SCALE"] = np.round(gs, decimals=4)

            # Write to disk
            master_flat.write_mef(path=outpath, prime_header=prime_header, data_headers=data_headers)

            # QC plot
            if self.setup["misc"]["qc_plots"]:
                mflat = MasterFlat(setup=self.setup, file_paths=outpath)
                mflat.qc_plot_flat(paths=None, axis_size=5, overwrite=self.setup["misc"]["overwrite"])

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    # =========================================================================== #
    # Master Weight
    # =========================================================================== #
    def build_master_weight_global(self):
        """
        Creates master weights from master flats. The difference between them is that NaNs are replaced with 0s and
        there is an additional option to mask relative and absolute values.

        """

        # Processing info
        tstart = message_mastercalibration(master_type="MASTER-WEIGHT-GLOBAL",
                                           silent=self.setup["misc"]["silent"], right=None)

        # Get unique Master flats
        master_flats = self.get_unique_master_flats()

        # Generate outpaths
        outpaths = [x.replace("MASTER-FLAT", "MASTER-WEIGHT-GLOBAL") for x in master_flats.full_paths]

        # Loop over files and apply calibration
        for idx in range(len(master_flats)):

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpaths[idx], silent=self.setup["misc"]["silent"]) \
                    and not self.setup["misc"]["overwrite"]:
                continue

            # Print processing info
            if not self.setup["misc"]["silent"]:
                message_calibration(n_current=idx + 1, n_total=len(master_flats),
                                    name=outpaths[idx], d_current=None, d_total=None)

            # Read file into cube
            cube = master_flats.file2cube(file_index=idx, hdu_index=None, dtype=None)

            # Apply absolute masks
            cube.apply_masks(mask_below=self.setup["weight"]["mask_abs_min"],
                             mask_above=self.setup["weight"]["mask_abs_max"])

            # Get median for each plane
            median = cube.median(axis=(1, 2))[:, np.newaxis, np.newaxis]

            # Norm each plane by its median
            cube /= median

            # Mask relative values
            cube.apply_masks(mask_below=self.setup["weight"]["mask_rel_min"],
                             mask_above=self.setup["weight"]["mask_rel_max"])

            # Scale back to original
            cube *= median

            # Interpolate NaNs if set (cosmetic function can't be used since this would also apply e.g. de-striping)
            if self.setup["interpolation"]["interpolate_nan"]:
                cube.interpolate_nan()

            # Replace remaining NaNs with 0 weight
            cube.replace_nan(value=0)

            # Modify type in primary header
            prime_header = master_flats.headers_primary[idx].copy()
            prime_header[self.setup["keywords"]["object"]] = "MASTER-WEIGHT-GLOBAL"

            # Write to file
            cube.write_mef(path=outpaths[idx], prime_header=prime_header, data_headers=master_flats.headers_data[idx])

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

        # Return master weights
        return MasterWeight(setup=self.setup, file_paths=outpaths)

    def build_master_weight_coadd(self):

        # Processing info
        tstart = message_mastercalibration(master_type="MASTER-WEIGHT-COADD", right=None,
                                           silent=self.setup["misc"]["silent"])

        # Get unique Master flats
        master_weights = self.get_unique_master_weights()

        # Generate outpaths
        outpaths = [x.replace("MASTER-WEIGHT", "MASTER-WEIGHT-COADD") for x in master_weights.full_paths]

        # Loop over files and apply calibration
        for idx in range(len(master_weights)):

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpaths[idx], silent=self.setup["misc"]["silent"]) \
                    and not self.setup["misc"]["overwrite"]:
                continue

            # Print processing info
            if not self.setup["misc"]["silent"]:
                message_calibration(n_current=idx + 1, n_total=len(master_weights),
                                    name=outpaths[idx], d_current=None, d_total=None)

            # Read file into cube
            cube = master_weights.file2cube(file_index=idx, hdu_index=None, dtype=None)

            # Edge arrays
            e1 = np.array([np.full(cube.shape[1], fill_value=0, dtype=int),
                           np.arange(0, cube.shape[2], 1).astype(int)])
            e2 = np.array([np.arange(0, cube.shape[1], 1).astype(int),
                           np.full(cube.shape[2], fill_value=0, dtype=int)])
            e3 = np.array([np.full(cube.shape[1], fill_value=cube.shape[1], dtype=int),
                           np.arange(0, cube.shape[2], 1).astype(int)])
            e4 = np.array([np.arange(0, cube.shape[1], 1).astype(int),
                           np.full(cube.shape[2], fill_value=cube.shape[2], dtype=int)])
            edges_stacked = np.hstack([e1, e2, e3, e4])

            # Create coorainate array for entire image
            coo_image_x, coo_image_y = np.meshgrid(np.arange(cube.shape[1]), np.arange(cube.shape[2]))
            coo_stacked = np.stack([coo_image_x.ravel(), coo_image_y.ravel()])

            dis, _ = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(edges_stacked.T).kneighbors(coo_stacked.T)
            dis = dis[:, -1].reshape(cube.shape[1], cube.shape[2])

            # Create weight
            weight = dis / self.setup["weight"]["coadd_edge_gradient"]

            # Mask too large weights in center of image
            weight[weight > 1] = 1.

            # Modify ipout weight
            cube.cube *= weight

            # Modify type in primary header
            prime_header = master_weights.headers_primary[idx].copy()
            prime_header[self.setup["keywords"]["object"]] = "MASTER-WEIGHT-COADD"

            # Write to file
            cube.write_mef(path=outpaths[idx], prime_header=prime_header, data_headers=master_weights.headers_data[idx])

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

        # Return master weights
        return MasterWeightCoadd(setup=self.setup, file_paths=outpaths)


class WeightImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(WeightImages, self).__init__(setup=setup, file_paths=file_paths)


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
        return self.dataheaders_get_keys(keywords=["HIERARCH PYPE FLAT SCALE"])[0]

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
            return ["{0}{1}.pdf".format(self.path_qc_flat, fp) for fp in self.file_names]
        else:
            return paths

    # noinspection DuplicatedCode
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

            # Read focal play array layout
            fpa_layout = str2list(self.setup["data"]["fpa_layout"], dtype=int)

            # Get plot grid
            fig, axes = get_plotgrid(layout=fpa_layout, xsize=axis_size, ysize=axis_size)
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
                if idx < fpa_layout[1]:
                    ax.set_xlabel("MJD (h) + {0:0n}d".format(mjd_floor))
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if idx % fpa_layout[0] == fpa_layout[0] - 1:
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


class MasterSuperflat(MasterImages):

    def __init__(self, setup, file_paths=None):
        super(MasterSuperflat, self).__init__(setup=setup, file_paths=file_paths)

    @property
    def nsources(self):
        return self.dataheaders_get_keys(keywords=["HIERARCH PYPE SFLAT NSOURCES"])[0]

    @property
    def flx_std(self):
        return self.dataheaders_get_keys(keywords=["HIERARCH PYPE SFLAT STD"])[0]

    def qc_plot_superflat(self, paths=None, axis_size=4):

        # Import
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Generate path for plots
        if paths is None:
            paths = ["{0}{1}.pdf".format(self.path_qc_superflat, fp) for fp in self.file_names]

        for idx_file in range(len(self)):

            # Read focal play array layout
            fpa_layout = str2list(self.setup["data"]["fpa_layout"], dtype=int)

            # Create figure
            fig, ax_file = get_plotgrid(layout=fpa_layout, xsize=axis_size, ysize=axis_size)
            ax_file = ax_file.ravel()
            cax = fig.add_axes([0.3, 0.92, 0.4, 0.02])

            # Read data
            cube = self.file2cube(file_index=idx_file)

            for idx_hdu in range(len(self.data_hdu[idx_file])):

                # Fetch current axes
                ax = ax_file[idx_hdu]

                # Draw image
                im = ax.imshow(cube[idx_hdu], vmin=0.85, vmax=1.15, cmap=get_cmap("RdYlBu_r", 30), origin="lower")

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
                if idx_hdu < fpa_layout[1]:
                    ax.set_xlabel("X (pix)")
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if idx_hdu % fpa_layout[0] == fpa_layout[0] - 1:
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


class MasterWeight(MasterImages):

    def __init__(self, setup, file_paths=None):
        super(MasterWeight, self).__init__(setup=setup, file_paths=file_paths)


class MasterWeightCoadd(MasterImages):

    def __init__(self, setup, file_paths=None):
        super(MasterWeightCoadd, self).__init__(setup=setup, file_paths=file_paths)
