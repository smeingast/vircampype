import time
import pickle
import warnings
import numpy as np

from astropy import wcs
from astropy.io import fits
from itertools import repeat
from astropy.time import Time
from astropy.table import vstack
from joblib import Parallel, delayed
from vircampype.tools.plottools import *
from vircampype.tools.systemtools import *
from vircampype.tools.tabletools import *
from astropy.coordinates import SkyCoord
from vircampype.tools.messaging import *
from vircampype.tools.fitstools import *
from vircampype.tools.mathtools import *
from vircampype.tools.astromatic import *
from vircampype.tools.photometry import *
from vircampype.data.cube import ImageCube
from sklearn.neighbors import KernelDensity
from vircampype.tools.miscellaneous import *
from vircampype.tools.tabletools import add_zp_2mass
from vircampype.fits.tables.sources import SourceCatalogs


class SextractorCatalogs(SourceCatalogs):

    def __init__(self, setup, file_paths=None):
        super(SextractorCatalogs, self).__init__(file_paths=file_paths, setup=setup)

    @property
    def _key_ra(self):
        return "ALPHA_J2000"

    @property
    def _key_dec(self):
        return "DELTA_J2000"

    @property
    def iter_data_hdu(self):
        """
        Overrides the normal table data_hdu property.

        Returns
        -------
        iterable
            List of iterators for header indices of HDUs which hold data.
        """
        return [range(2, len(hdrs), 2) for hdrs in self.headers]

    def read_from_image_headers(self, keywords, file_index=None):

        if not isinstance(keywords, list):
            raise TypeError("Keywords must be in a list!")

        if file_index is None:
            headers_image = self.image_headers[:]
        else:
            headers_image = [self.image_headers[file_index]]

        # Return values
        return [[[e[k] for e in h] for h in headers_image] for k in keywords]

    # =========================================================================== #
    # Scamp
    # =========================================================================== #
    def _scamp_header_paths(self, joined=False):
        """
        List or str containing scamp header names.

        Parameters
        ----------
        joined : bool, optional
            If set, list will be joined by ',' to make it readable for scamp.

        Returns
        -------
        iterable, str
            List or str with header names.

        """

        names = [x.replace(".scamp.fits.tab", ".ahead") for x in self.paths_full]
        if joined:
            return ",".join(names)
        else:
            return names

    @property
    def _scamp_catalog_paths(self):
        """
        Concatenates full paths to make them readable for scamp.

        Returns
        -------
        str
            Joined string with full paths of self.
        """
        return " ".join(self.paths_full)

    def scamp(self):

        # Load Scamp setup
        scs = ScampSetup(setup=self.setup)

        # Get passband
        bands = list(set(self.passband))
        print(bands)
        if len(bands) != 1:
            raise ValueError("Sequence contains multiple filters")
        else:
            band = bands[0][0]  # This should only keep J,H, and K for 2MASS (First band and first letter)
            band = "Ks" if "k" in band.lower() else band

        # Load preset
        options = yml2config(path_yml=get_resource_path(package=scs.package_presets, resource="scamp.yml"),
                             nthreads=self.setup.n_jobs,
                             checkplot_type=scs.qc_types(joined=True),
                             checkplot_name=scs.qc_names(joined=True),
                             skip=["HEADER_NAME", "AHEADER_NAME", "ASTREF_BAND"])

        # Construct commands for source extraction
        cmd = "{0} {1} -c {2} -HEADER_NAME {3} -ASTREF_BAND {4} {5}" \
              "".format(scs.bin, self._scamp_catalog_paths, scs.default_config,
                        self._scamp_header_paths(joined=True), band, options)

        # Run Scamp
        run_command_bash(cmd, silent=False)

    # =========================================================================== #
    # Other properties
    # =========================================================================== #
    @property
    def _paths_image_headers(self):
        # return ["{0}{1}.image.header".format(self.setup.folders["headers"], x) for x in self.basenames]
        return [x.replace(".header", ".imageheader") for x in self.paths_headers]

    _image_headers = None

    @property
    def image_headers(self):
        """
        Obtains image headers from sextractor catalogs

        Returns
        -------
        iterable
            List of lists containing the image headers for each table and each extension.

        """

        # Check if already determined
        if self._image_headers is not None:
            return self._image_headers

        self._image_headers = []
        for idx in range(self.n_files):

            # Try to read the database
            try:
                with open(self._paths_image_headers[idx], "rb") as f:

                    # If the file is there, load the headers...
                    self._image_headers.append(pickle.load(f))

                    # And continue with next file
                    continue

            # If not found we move on to read the headers from the fits file
            except FileNotFoundError:

                headers = sextractor2imagehdr(path=self.paths_full[idx])

                # When done for all headers dump them into the designated database
                with open(self._paths_image_headers[idx], "wb") as d:
                    pickle.dump(headers, d)

                # Append converted headers
                self._image_headers.append(headers)

        # Return
        return self._image_headers

    # =========================================================================== #
    # Time
    # =========================================================================== #
    _time_obs = None

    @property
    def time_obs(self):

        # Check if already determined
        if self._time_obs is not None:
            return self._time_obs
        else:
            pass

        self._time_obs = Time([hdr[0][self.setup.keywords.date_mjd]
                               for hdr in self.image_headers], scale="utc", format="mjd")
        return self._time_obs


class AstrometricCalibratedSextractorCatalogs(SextractorCatalogs):

    def __init__(self, setup, file_paths=None):
        super(AstrometricCalibratedSextractorCatalogs, self).__init__(file_paths=file_paths, setup=setup)

    def build_master_superflat(self):
        """ Superflat construction method. """

        # Import
        from vircampype.fits.images.flat import MasterSuperflat

        # Processing info
        print_header(header="MASTER-SUPERFLAT", right=None, silent=self.setup.silent)
        tstart = time.time()

        # Split based on passband and interval
        split = self.split_keywords(keywords=[self.setup.keywords.filter_name])
        split = flat_list([s.split_window(window=self.setup.superflat_window, remove_duplicates=True) for s in split])

        # Remove too short entries
        split = prune_list(split, n_min=self.setup.superflat_n_min)

        # Get master photometry catalog
        master_phot = self.get_master_photometry()

        # Now loop through separated files
        for files, idx_print in zip(split, range(1, len(split) + 1)):

            # Create master dark name
            outpath = self.setup.folders["master_object"] + "MASTER-SUPERFLAT_{0:11.5f}.fits".format(files.mjd_mean)

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent):
                continue

            # Grab current passband
            passband = files.passband[0]

            # Fetch clean index for current filter
            mkeep = master_phot.get_purge_index(passband=passband)

            # Fetch magnitude and coordinates for master catalog
            mag_master = master_phot.mag(passband=passband)[0][0][mkeep]
            skycoord_master = master_phot.skycoord()[0][0][mkeep]

            data_headers, flx_scale, n_sources = [], [], []
            for idx_hdu, idx_hdr in zip(files.iter_data_hdu[0], range(len(files.iter_data_hdu[0]))):

                # Print processing info
                message_calibration(n_current=idx_print, n_total=len(split), name=outpath, d_current=idx_hdr + 1,
                                    d_total=len(files.iter_data_hdu[0]), silent=self.setup.silent)

                # Read current HDU for all files into a single table
                tab = vstack(files.hdu2table(hdu_index=idx_hdu))

                # Read header of current extension in first file
                header = files.image_headers[0][idx_hdr]

                # Clean table
                tab = clean_source_table(table=tab, image_header=header, flux_max=header["SEXSATLV"] / 2)

                # Get difference to reference catalog magnitudes
                zp_all = get_zeropoint(skycoord_cal=SkyCoord(tab[self._key_ra], tab[self._key_dec], unit="deg"),
                                       mag_cal=tab["MAG_AUTO"], skycoord_ref=skycoord_master, mag_ref=mag_master,
                                       mag_limits_ref=master_phot.mag_lim(passband=passband), method="all")

                # Remove all table entries without ZP entry
                tab, zp_all = tab[np.isfinite(zp_all)], zp_all[np.isfinite(zp_all)]

                binsize = get_binsize(table=tab, key_x="XWIN_IMAGE", key_y="YWIN_IMAGE", n_neighbors=50)
                n_bins_x, n_bins_y = int(header["NAXIS1"] / binsize), int(header["NAXIS1"] / binsize)

                # Set minimum and maximum number of bins
                n_bins_x = 3 if n_bins_x < 3 else n_bins_x
                n_bins_y = 3 if n_bins_y < 3 else n_bins_y
                n_bins_x = 15 if n_bins_x > 15 else n_bins_x
                n_bins_y = 15 if n_bins_y > 15 else n_bins_y

                # Grid values to detector size array
                grid_zp = grid_value_2d(x=tab["XWIN_IMAGE"], y=tab["YWIN_IMAGE"], value=zp_all, x_min=0, y_min=0,
                                        weights=None, x_max=header["NAXIS1"], y_max=header["NAXIS2"], nx=n_bins_x,
                                        ny=n_bins_y, conv=True, kernel_size=1.0, upscale=True)

                # Convert to flux scale
                flx_scale.append(10**((grid_zp - self.setup.target_zp) / 2.5))

                # # Plot sources on top of superflat
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(nrows=1, ncols=1, gridspec_kw=dict(top=0.98, right=0.99),
                #                        **dict(figsize=(6, 5)))
                #
                # im = ax.imshow(flx_scale[-1] / np.nanmedian(flx_scale[-1]), cmap="RdYlBu_r", vmin=0.95, vmax=1.04,
                #                origin="lower", extent=[0, header["NAXIS1"], 0, header["NAXIS2"]])
                # flux = 10**(zp_all / 2.5)
                # ax.scatter(tab["XWIN_IMAGE"], tab["YWIN_IMAGE"], c=flux / np.nanmedian(flux), s=5, lw=0.5,
                #            cmap="RdYlBu_r", ec="black", vmin=0.95, vmax=1.04)
                # plt.colorbar(im)
                # plt.show()
                # exit()

                # Save number of sources
                n_sources.append(np.sum(np.isfinite(zp_all)))

            # Instantiate output
            superflat = ImageCube(setup=self.setup)

            # Loop over extensions and construct final superflat
            for idx_hdu, fscl, nn in zip(files.iter_data_hdu[0], flx_scale, n_sources):

                # Append to output
                superflat.extend(data=fscl.astype(np.float32))

                # Create extension header cards
                # TODO: Change this to add_float_to_header
                data_cards = make_cards(keywords=["HIERARCH PYPE SFLAT NSOURCES", "HIERARCH PYPE SFLAT STD"],
                                        values=[nn, float(str(np.round(np.nanstd(fscl), decimals=2)))],
                                        comments=["Number of sources used", "Standard deviation in relative flux"])

                # Append header
                data_headers.append(fits.Header(cards=data_cards))

            # Make primary header
            prime_cards = make_cards(keywords=[self.setup.keywords.object, self.setup.keywords.date_mjd,
                                               self.setup.keywords.filter_name, self.setup.keywords.date_ut,
                                               "HIERARCH PYPE N_FILES"],
                                     values=["MASTER-SUPERFLAT", files.mjd_mean,
                                             files.passband[0], files.time_obs_mean,
                                             len(files)])
            prime_header = fits.Header(cards=prime_cards)

            # Write to disk
            superflat.write_mef(path=outpath, prime_header=prime_header, data_headers=data_headers)

            # QC plot
            if self.setup.qc_plots:
                msf = MasterSuperflat(setup=self.setup, file_paths=outpath)
                msf.qc_plot_superflat(paths=None, axis_size=5)

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

    def crunch_source_catalogs(self):

        # Processing info
        print_header(header="PHOTOMETRY", silent=self.setup.silent, right=None)
        tstart = time.time()

        # Get master photometry catalog
        master_phot = self.get_master_photometry()[0]
        mkeep = master_phot.get_purge_index(passband=self.passband[0][0])
        table_master = master_phot.file2table(file_index=0)[0][mkeep]

        # Instantiate classification tables
        tables_class = SourceCatalogs(setup=self.setup,
                                      file_paths=[x.replace(".full.", ".cs.") for x in self.paths_full])

        # Start loop over files
        for idx_file in range(self.n_files):

            # Create output path
            outpath = self.paths_full[idx_file].replace(".fits.tab", ".fits.ctab")

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent) \
                    and not self.setup.overwrite:
                continue

            # Print processing info
            message_calibration(n_current=idx_file + 1, n_total=self.n_files, name=outpath,
                                d_current=None, d_total=None, silent=self.setup.silent)

            # Load and clean table
            tables_file = self.file2table(file_index=idx_file)
            image_headers_file = self.image_headers[idx_file]

            # Add aperture correction to tables
            [t.add_column((t["MAG_APER"].data.T - t["MAG_APER"].data[:, -1]).T, name="MAG_APER_COR")
             for t in tables_file]

            # Add smoothed stats to tables
            parameters = ["FWHM_WORLD", "ELLIPTICITY", "MAG_APER_COR"]
            with Parallel(n_jobs=self.setup.n_jobs) as parallel:
                tables_file = parallel(delayed(add_smoothed_value)(i, j, k) for i, j, k
                                       in zip(tables_file, image_headers_file, repeat(parameters)))

            # Load classification table for current file
            tables_class_file = tables_class.file2table(file_index=idx_file)

            # Interpolate classification in parallel for each extension
            with Parallel(n_jobs=self.setup.n_jobs) as parallel:
                tables_file = parallel(delayed(interpolate_classification)(i, j, k) for i, j, k
                                       in zip(tables_file, tables_class_file, repeat(self.setup.seeing_test_range)))

            # Match apertures and add to table
            [t.add_column((t["MAG_APER"] - t["MAG_APER_COR_INTERP"]), name="MAG_APER_MATCHED")
             for t in tables_file]

            # Compute ZP and add calibrated photometry to catalog
            columns_mag = ["MAG_APER_MATCHED", "MAG_AUTO", "MAG_ISO", "MAG_ISOCOR", "MAG_PETRO"]
            columns_magerr = ["MAGERR_APER", "MAGERR_AUTO", "MAGERR_ISO", "MAGERR_ISOCOR", "MAGERR_PETRO"]

            # Open input catalog
            table_hdulist = fits.open(self.paths_full[idx_file], mode="readonly")

            for table_hdu, idx_table_hdu in zip(tables_file, self.iter_data_hdu[idx_file]):
                add_zp_2mass(table_hdu, table_2mass=table_master, key_ra=self._key_ra, key_dec=self._key_dec,
                             mag_lim_ref=master_phot.mag_lim(passband=self.passband[idx_file]), method="weighted",
                             passband_2mass=master_phot.translate_passband(self.passband[idx_file][0]),
                             columns_mag=columns_mag, columns_magerr=columns_magerr)

                # Compute total errors for aperture photometry as combination of errors from
                # photometry, ZP error, and aperture matching
                magerr_zp = np.array([table_hdu.zperr["HIERARCH PYPE ZP ERR MAG_APER_MATCHED {0}".format(i+1)]
                                      for i in range(len(self.setup.apertures))])
                table_hdu["MAGERR_APER_MATCHED_TOT"] = np.sqrt(table_hdu["MAGERR_APER"]**2 +
                                                               table_hdu["MAG_APER_COR_STDEV"]**2 +
                                                               magerr_zp**2).astype(np.float32)

                # Replace original HDU
                table_hdulist[idx_table_hdu] = table2bintablehdu(table=table_hdu)

                # Add ZPs to header
                for attr in ["zp", "zperr"]:
                    for key, val in getattr(table_hdu, attr).items():
                        if attr == "zp":
                            comment = "Zero point (mag)"
                        else:
                            comment = "Standard error of ZP (mag)"
                        add_float_to_header(header=table_hdulist[idx_table_hdu].header,
                                            key=key, value=val, comment=comment, decimals=4)

            # Write to new output file
            table_hdulist.writeto(outpath, overwrite=self.setup.overwrite)

            # Close original file
            table_hdulist.close()

            # QC plot
            if self.setup.qc_plots:
                csc = PhotometricCalibratedSextractorCatalogs(setup=self.setup, file_paths=outpath)
                csc.plot_qc_zp(axis_size=5)
                csc.plot_qc_ref(axis_size=5)

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

    def add_statistics(self):

        # Import
        from vircampype.fits.images.common import FitsImages

        # Processing info
        print_header(header="ADDING STATISTICS", silent=self.setup.silent, right=None)
        tstart = time.time()

        for idx_file in range(self.n_files):

            # Find statistics images
            mjdeff_image = FitsImages.from_folder(path=self.directories[idx_file],
                                                  pattern="*.mjdeff.fits", setup=self.setup)
            exptime_image = FitsImages.from_folder(path=self.directories[idx_file],
                                                   pattern="*.exptime.fits", setup=self.setup)
            ndet_image = FitsImages.from_folder(path=self.directories[idx_file],
                                                pattern="*.ndet.fits", setup=self.setup)

            # There can only be one match
            if mjdeff_image.n_files * exptime_image.n_files * ndet_image.n_files != 1:
                raise ValueError("Matches for image statistics are not unique")

            # Open current table file
            hdul = fits.open(self.paths_full[idx_file], mode="update")

            for idx_hdu_self, idx_hdu_stats in zip(self.iter_data_hdu[idx_file], mjdeff_image.iter_data_hdu[0]):

                # Read table
                table_hdu = self.filehdu2table(file_index=idx_file, hdu_index=idx_hdu_self)

                # Read stats
                mjdeff = fits.getdata(mjdeff_image.paths_full[0], idx_hdu_stats)
                exptime = fits.getdata(exptime_image.paths_full[0], idx_hdu_stats)
                ndet = fits.getdata(ndet_image.paths_full[0], idx_hdu_stats)
                weight = fits.getdata(mjdeff_image.paths_full[0].replace(".fits", ".weight.fits"), idx_hdu_stats)
                weight /= np.median(weight)

                # Obtain wcs for statistics images (they all have the same projection)
                wcs_stats = wcs.WCS(header=mjdeff_image.headers_data[0][idx_hdu_stats])

                # Convert to X/Y
                xx, yy = wcs_stats.wcs_world2pix(table_hdu[self._key_ra], table_hdu[self._key_dec], 0)

                # Get values for each source from data arrays
                mjdeff_sources = mjdeff[yy.astype(int), xx.astype(int)]
                exptime_sources = exptime[yy.astype(int), xx.astype(int)]
                ndet_sources = ndet[yy.astype(int), xx.astype(int)]
                weight_sources = weight[yy.astype(int), xx.astype(int)]

                # Mask bad sources
                bad = weight_sources < 0.0001
                mjdeff_sources[bad] = np.nan
                exptime_sources[bad] = 0
                ndet_sources[bad] = 0

                # Append new columns
                orig_cols = hdul[idx_hdu_self].data.columns
                new_cols = fits.ColDefs([fits.Column(name="MJDEFF", format="D", array=mjdeff_sources),
                                         fits.Column(name='EXPTIME', format="J", array=exptime_sources, unit="seconds"),
                                         fits.Column(name='NOBS', format="J", array=ndet_sources)])

                # Replace HDU
                try:
                    hdul[idx_hdu_self] = fits.BinTableHDU.from_columns(orig_cols + new_cols,
                                                                       header=hdul[idx_hdu_self].header)
                except ValueError:
                    pass

            hdul.flush()

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")


class PhotometricCalibratedSextractorCatalogs(AstrometricCalibratedSextractorCatalogs):

    def __init__(self, setup, file_paths=None):
        super(PhotometricCalibratedSextractorCatalogs, self).__init__(file_paths=file_paths, setup=setup)

    def write_coadd_flux_scale(self):
        """ Constructs flux scale for coaddition from different zero points across all images and detectors. """

        # Convert ZPs to flux scaling factor
        flx_scale = 10**(np.array(self.read_from_data_headers(keywords=["HIERARCH PYPE ZP MAG_AUTO"])[0]) / -2.5)

        # Normalize the scaling across all input catalogs
        flx_scale = (flx_scale / np.mean(flx_scale)).tolist()

        # Loop over files and write to disk
        for idx_file in range(self.n_files):

            # Create output path
            outpath = self.paths_full[idx_file].replace(".full.fits.ctab", ".coadd_ahead")

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent) and not self.setup.overwrite:
                continue

            # Create string to write from all scaling factor for current file
            s = "\n".join(["FLXSCALE={0:0.5f}\nEND".format(x) for x in flx_scale[idx_file]])

            # Write to disk
            with open(outpath, "w") as file:
                file.write(s)

    def paths_qc_plots(self, paths, prefix=""):

        if paths is None:
            return ["{0}{1}.{2}.pdf".format(self.setup.folders["qc_photometry"], fp, prefix) for fp in self.basenames]
        else:
            return paths

    def plot_qc_zp(self, paths=None, axis_size=5):
        """ Generates ZP QC plot. """

        for idx_file in range(self.n_files):

            # Generate path for plot
            path = self.paths_qc_plots(paths=paths, prefix="zp")[idx_file]

            zp_auto = self.read_from_data_headers(keywords=["HIERARCH PYPE ZP MAG_AUTO"])[0][idx_file]
            zperr_auto = self.read_from_data_headers(keywords=["HIERARCH PYPE ZP ERR MAG_AUTO"])[0][idx_file]

            # Create plot
            plot_value_detector(values=zp_auto, errors=zperr_auto, path=path, ylabel="ZP AUTO (mag)",
                                axis_size=axis_size, yrange=(np.median(zp_auto) - 0.1, np.median(zp_auto) + 0.1))

    def plot_qc_ref(self, axis_size=5):

        # Import
        from astropy.units import Unit
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Generate output paths
        outpaths_1d = self.paths_qc_plots(paths=None, prefix="phot.1D")

        for idx_file in range(len(self)):

            # Get passband
            passband = self.passband[idx_file][0]

            # Get master photometry catalog
            master_phot = self.get_master_photometry()[0]
            mkeep = master_phot.get_purge_index(passband=passband)

            # Fetch magnitude and coordinates for master catalog
            mag_master = master_phot.mag(passband=passband)[idx_file][0][mkeep]
            master_skycoord = master_phot.skycoord()[idx_file][0][mkeep]

            # Read tables
            tab_file = self.file2table(file_index=idx_file)

            # Clean tables
            tab_file = [clean_source_table(table=t) for t in tab_file]

            # Make plot grid
            if len(self.iter_data_hdu[idx_file]) == 1:
                fig, ax_file = get_plotgrid(layout=(1, 1), xsize=2*axis_size, ysize=2*axis_size)
                ax_file = [ax_file]
            else:
                fig, ax_file = get_plotgrid(layout=self.setup.fpa_layout, xsize=axis_size, ysize=axis_size / 2)
                ax_file = ax_file.ravel()

            # Loop over extensions
            for idx_hdu in range(len(self.iter_data_hdu[idx_file])):

                # Grab axes
                ax = ax_file[idx_hdu]

                # Grab current catalog
                tab_hdu = tab_file[idx_hdu]

                # Clean table
                tab_hdu = clean_source_table(table=tab_hdu)

                # Construct skycoordinates
                sc_hdu = SkyCoord(ra=tab_hdu[self._key_ra], dec=tab_hdu[self._key_dec], unit="deg")

                # Xmatch science with reference
                zp_idx, zp_d2d, _ = sc_hdu.match_to_catalog_sky(master_skycoord)

                # Get good indices in reference catalog and in current field
                idx_master = zp_idx[zp_d2d < 1 * Unit("arcsec")]
                idx_final = np.arange(len(zp_idx))[zp_d2d < 1 * Unit("arcsec")]

                # Apply indices filter
                mag_hdu_match = tab_hdu["MAG_AUTO_CAL"][idx_final]
                mag_master_match = mag_master[idx_master]

                # Compute difference between reference and self
                mag_delta = mag_master_match - mag_hdu_match

                # Draw ZP
                ax.axhline(0, zorder=1, c="black", alpha=0.5)

                # KDE for ZP mag interval
                keep = (mag_master_match >= master_phot.mag_lim(passband=passband)[0]) & \
                       (mag_master_match <= master_phot.mag_lim(passband=passband)[1])

                # Draw photometry for all matched sources
                ax.scatter(mag_master_match, mag_delta, c="crimson",
                           vmin=0, vmax=1.0, s=6, lw=0, alpha=0.6, zorder=0)

                # Draw for sources within mag limits
                ax.scatter(mag_master_match[keep], mag_delta[keep], c="crimson",
                           vmin=0, vmax=1.0, s=7, lw=0, alpha=1.0, zorder=0)

                # Evaluate KDE
                kde = KernelDensity(kernel="gaussian", bandwidth=0.1, metric="euclidean")
                kde_grid = np.arange(np.floor(-1), np.ceil(1), 0.01)
                # noinspection PyUnresolvedReferences
                dens_zp = np.exp(kde.fit((mag_delta[keep]).reshape(-1, 1)).score_samples(kde_grid.reshape(-1, 1)))

                # Draw KDE
                ax_kde = ax.twiny()
                ax_kde.plot(dens_zp, kde_grid, lw=1, c="black", alpha=0.8)
                ax_kde.axis("off")

                # Annotate detector ID
                ax.annotate("Det.ID: {0:0d}".format(idx_hdu + 1), xy=(0.98, 0.04),
                            xycoords="axes fraction", ha="right", va="bottom")

                # Set limits
                ax.set_xlim(10, 18)
                ylim = (-1, 1)
                ax.set_ylim(ylim)
                ax_kde.set_ylim(ylim)

                # Modify axes
                if idx_hdu < self.setup.fpa_layout[1]:
                    ax.set_xlabel("{0} {1} (mag)".format(self.setup.reference_catalog.upper(), self.passband[idx_file]))
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if idx_hdu % self.setup.fpa_layout[0] == self.setup.fpa_layout[0] - 1:
                    ax.set_ylabel(r"$\Delta${0} (mag)".format(self.passband[idx_file]))
                else:
                    ax.axes.yaxis.set_ticklabels([])

                # Set ticks
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
                fig.savefig(outpaths_1d[-1], bbox_inches="tight")
            plt.close("all")
