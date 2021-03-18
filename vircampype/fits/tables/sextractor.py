import os
import time
import pickle
import warnings
import numpy as np

from astropy.io import fits
from itertools import repeat
from astropy.time import Time
from joblib import Parallel, delayed
from astropy.table import vstack, Table
from vircampype.tools.plottools import *
from astropy.coordinates import SkyCoord
from vircampype.tools.messaging import *
from vircampype.tools.fitstools import *
from vircampype.tools.mathtools import *
from vircampype.tools.tabletools import *
from vircampype.tools.astromatic import *
from vircampype.tools.photometry import *
from vircampype.tools.systemtools import *
from vircampype.data.cube import ImageCube
from sklearn.neighbors import KernelDensity
from vircampype.tools.miscellaneous import *
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS, FITSFixedWarning
from sklearn.neighbors import NearestNeighbors
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

        # Construct XML path
        path_xml = "{0}scamp.xml".format(self.setup.folders["qc_astrometry"])

        # Check if XML is already there and just return if it is
        if check_file_exists(file_path=path_xml, silent=self.setup.silent):
            return

        # Load Scamp setup
        scs = ScampSetup(setup=self.setup)

        # Load preset
        options = yml2config(path_yml=get_resource_path(package=scs.package_presets, resource="scamp.yml"),
                             nthreads=self.setup.n_jobs,
                             checkplot_type=scs.qc_types(joined=True),
                             checkplot_name=scs.qc_names(joined=True),
                             skip=["HEADER_NAME", "AHEADER_NAME", "ASTREF_CATALOG", "ASTREF_BAND",
                                   "FLAGS_MASK", "WEIGHTFLAGS_MASK", "ASTR_FLAGSMASK", "XML_NAME"])

        # Get passband
        if "gaia" in self.setup.astr_reference_catalog.lower():
            catalog_name = "GAIA-EDR3"
            band_name = "G"
        elif "2mass" in self.setup.astr_reference_catalog.lower():
            catalog_name = "2MASS"
            bands = list(set(self.passband))
            if len(bands) != 1:
                raise ValueError("Sequence contains multiple filters")
            else:
                band_name = bands[0][0]  # This should only keep J,H, and K for 2MASS (First band and first letter)
                band_name = "Ks" if "k" in band_name.lower() else band_name
        else:
            raise ValueError("Astrometric reference catalog '{0}' not supported"
                             "".format(self.setup.astr_reference_catalog))

        # Construct command for scamp
        cmd = "{0} {1} -c {2} -HEADER_NAME {3} -ASTREF_CATALOG {4} -ASTREF_BAND {5} -XML_NAME {6} {7}" \
              "".format(scs.bin, self._scamp_catalog_paths, scs.default_config,
                        self._scamp_header_paths(joined=True), catalog_name, band_name, path_xml, options)

        # Run Scamp
        run_command_shell(cmd, silent=False)

        # Some basic QC on XML
        xml = Table.read(path_xml, format="votable", table_id=1)
        if np.max(xml["AstromSigma_Internal"].data.ravel() * 1000) > 100:
            raise ValueError("Astrometric solution may be crap, please check")

    # =========================================================================== #
    # PSFEx
    # =========================================================================== #
    def psfex(self, preset):

        # Processing info
        print_header(header="PSFEx", left="Running PSFEx on {0} files"
                                          "".format(len(self)), right=None, silent=self.setup.silent)
        tstart = time.time()

        # Load Sextractor setup
        psfs = PSFExSetup(setup=self.setup)

        # Read setup based on preset
        if preset.lower() in ["pawprints", "tile"]:
            ss = read_yml(path_yml=psfs.path_yml(preset=preset))
        else:
            raise ValueError("Preset '{0}' not supported".format(preset))

        # Construct output psf paths
        psf_paths = ["{0}{1}".format(self.setup.folders["master_object"], fn).replace(".tab", ".psf")
                     for fn in self.basenames]

        # Check for existing PSF models
        done = [os.path.exists(p) for p in psf_paths]

        # Print statement on already existing files
        for p in psf_paths:
            check_file_exists(file_path=p, silent=self.setup.silent)

        # Return if nothing to be done
        if sum(done) == len(self):
            print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")
            return

        # Read source tables and stack all HDUs in a file
        options = []
        for i in range(self.n_files):

            # Read tables for current file
            tables = self.file2table(file_index=i)

            # Clean tables
            fwhm_range = [float(x) for x in ss["SAMPLE_FWHMRANGE"].split(",")]
            tables = [clean_source_table(table=t, min_snr=ss["SAMPLE_MINSN"], max_ellipticity=ss["SAMPLE_MAXELLIP"],
                                         nndis_limit=5, min_fwhm=fwhm_range[0], max_fwhm=fwhm_range[1]) for t in tables]

            # Loop over each HDU
            fwhm_hdu = [clipped_median(t["FWHM_IMAGE"]) for t in tables]

            # Determine sample FWHM range
            sample_fwhmrange = [0.8 * np.percentile(fwhm_hdu, 5),
                                1.5 * np.percentile(fwhm_hdu, 95)]

            # Construct XML path
            xml_name = "{0}XML_{1}.xml".format(self.setup.folders["qc_psf"], self.basenames[i])

            # Construct PSFEx options for current file
            options.append(yml2config(nthreads=1,
                                      checkplot_type=psfs.checkplot_types(joined=True),
                                      checkplot_name=psfs.checkplot_names(joined=True),
                                      checkimage_type=psfs.checkimage_types(joined=True),
                                      checkimage_name=psfs.checkimage_names(joined=True),
                                      sample_fwhmrange=",".join(["{0:0.2f}".format(x) for x in sample_fwhmrange]),
                                      sample_variability=0.5, xml_name=xml_name,
                                      psf_dir=self.setup.folders["master_object"], skip=["homokernel_dir"],
                                      path_yml=psfs.path_yml(preset=preset)))

        # Construct commands
        cmds = ["{0} {1} -c {2} {3}".format(psfs.bin, tab, psfs.default_config, o)
                for tab, o in zip(self.paths_full, options)]

        # Clean commands
        cmds = [c for c, d in zip(cmds, done) if not d]

        # Set number of parallel jobs
        n_jobs_psfex = 1
        n_jobs_shell = self.setup.n_jobs

        # If there are less files than parallel jobs, optimize psfex jobs
        if (len(cmds) > 0) & (len(cmds) < self.setup.n_jobs):
            n_jobs_shell = len(cmds)
            while n_jobs_psfex * n_jobs_shell < self.setup.n_jobs:
                n_jobs_psfex += 1

            # Adapt commands
            cmds = [c.replace("NTHREADS 1", "NTHREADS {0}".format(n_jobs_psfex)) for c in cmds]

        # Run PSFEX
        run_commands_shell_parallel(cmds=cmds, silent=True, n_jobs=n_jobs_shell)

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

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
                tab = clean_source_table(table=tab, image_header=header, flux_max=header["SEXSATLV"]/2, border_pix=20,
                                         nndis_limit=None, min_fwhm=1.0, max_fwhm=5.0, max_ellipticity=0.2)

                # Get difference to reference catalog magnitudes
                zp_all = get_zeropoint(skycoord_cal=SkyCoord(tab[self._key_ra], tab[self._key_dec], unit="deg"),
                                       mag_cal=tab["MAG_AUTO"], skycoord_ref=skycoord_master, mag_ref=mag_master,
                                       mag_limits_ref=master_phot.mag_lim(passband=passband), method="all")

                # Remove all table entries without ZP entry
                tab, zp_all = tab[np.isfinite(zp_all)], zp_all[np.isfinite(zp_all)]

                # Grid with NN interpolation
                grid_zp = grid_value_2d_nn(x=tab["XWIN_IMAGE"], y=tab["YWIN_IMAGE"], values=zp_all,
                                           n_bins_x=header["NAXIS1"] // 100, n_bins_y=header["NAXIS2"] // 100,
                                           x_min=1, y_min=1, x_max=header["NAXIS1"], y_max=header["NAXIS2"],
                                           n_nearest_neighbors=100 if len(tab) > 100 else len(tab))

                # Resize to original image size
                grid_zp = upscale_image(grid_zp, new_size=(header["NAXIS1"], header["NAXIS2"]), method="PIL")

                # Convert to flux scale
                flx_scale.append(10**((grid_zp - self.setup.target_zp) / 2.5))

                # Save number of sources
                n_sources.append(np.sum(np.isfinite(zp_all)))

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

            # Instantiate output
            superflat = ImageCube(setup=self.setup)

            # Loop over extensions and construct final superflat
            for idx_hdu, fscl, nn in zip(files.iter_data_hdu[0], flx_scale, n_sources):

                # Append to output
                superflat.extend(data=fscl.astype(np.float32))

                # Create empty header
                data_header = fits.Header()

                # Add data to header
                add_int_to_header(header=data_header, key="HIERARCH PYPE SFLAT NSOURCES", value=nn,
                                  comment="Number of sources used")
                add_float_to_header(header=data_header, key="HIERARCH PYPE SFLAT STD", value=np.nanstd(fscl),
                                    decimals=4, comment="Standard deviation in relative flux")

                # Append header
                data_headers.append(data_header)

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
                tables_file = parallel(delayed(interpolate_classification)(i, j) for i, j
                                       in zip(tables_file, tables_class_file))

            # Match apertures and add to table
            [t.add_column((t["MAG_APER"] - t["MAG_APER_COR_INTERP"]), name="MAG_APER_MATCHED")
             for t in tables_file]

            # Compute ZP and add calibrated photometry to catalog
            columns_mag = ["MAG_APER", "MAG_APER_MATCHED", "MAG_AUTO",
                           "MAG_ISO", "MAG_ISOCOR", "MAG_PETRO"]
            columns_magerr = ["MAGERR_APER", "MAGERR_APER", "MAGERR_AUTO",
                              "MAGERR_ISO", "MAGERR_ISOCOR", "MAGERR_PETRO"]

            # Open input catalog
            table_hdulist = fits.open(self.paths_full[idx_file], mode="readonly")

            for table_hdu, idx_table_hdu in zip(tables_file, self.iter_data_hdu[idx_file]):
                add_zp_2mass(table_hdu, table_2mass=table_master, key_ra=self._key_ra, key_dec=self._key_dec,
                             mag_lim_ref=master_phot.mag_lim(passband=self.passband[idx_file]), method="weighted",
                             passband_2mass=master_phot.translate_passband(self.passband[idx_file][0]),
                             columns_mag=columns_mag, columns_magerr=columns_magerr)

                # TODO: Fix this (perhaps by using standard error of mean?)
                # Compute total errors for aperture photometry as combination of errors from
                # photometry, ZP error, and aperture matching
                # magerr_zp = np.array([table_hdu.zperr["HIERARCH PYPE ZP ERR MAG_APER_MATCHED {0}".format(i+1)]
                #                       for i in range(len(self.setup.apertures))])
                # table_hdu["MAGERR_APER_MATCHED_TOT"] = np.sqrt(table_hdu["MAGERR_APER"]**2 +
                #                                                table_hdu["MAG_APER_COR_STD"]**2 +
                #                                                magerr_zp**2).astype(np.float32)

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
                pcsc = PhotometricCalibratedSextractorCatalogs(setup=self.setup, file_paths=outpath)
                pcsc.plot_qc_phot_zp(axis_size=5)
                pcsc.plot_qc_phot_ref1d(axis_size=5)
                pcsc.plot_qc_phot_ref2d(axis_size=5)
                if len(pcsc) >= 2:
                    pcsc.plot_qc_phot_interror()

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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    wcs_stats = WCS(header=mjdeff_image.headers_data[0][idx_hdu_stats])

                # Convert to X/Y
                xx, yy = wcs_stats.wcs_world2pix(table_hdu[self._key_ra], table_hdu[self._key_dec], 0)
                xx_image, yy_image = xx.astype(int), yy.astype(int)

                # Mark bad data
                bad = (xx_image >= mjdeff.shape[1]) | (xx_image < 0) | \
                      (yy_image >= mjdeff.shape[0]) | (yy_image < 0)

                # Just to be sort of safe, let's say we can't have more than 0.05% of sources at the edges
                if sum(bad) > 0.0005 * len(bad):
                    raise ValueError("Too many sources are close to the image edge ({0}/{1}). "
                                     "Please check for issues.".format(sum(bad), len(bad)))

                # Reset bad coordinates to 0/0
                xx_image[bad], yy_image[bad] = 0, 0

                # Get values for each source from data arrays
                mjdeff_sources = mjdeff[yy_image, xx_image]
                exptime_sources = exptime[yy_image, xx_image]
                ndet_sources = ndet[yy_image, xx_image]
                weight_sources = weight[yy_image, xx_image]

                # Mask bad sources
                bad &= weight_sources < 0.0001
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
        zps = np.array(self.read_from_data_headers(keywords=["HIERARCH PYPE ZP MAG_AUTO"])[0])
        flx_scale = (10**((zps - self.setup.target_zp) / 2.5)).tolist()

        # Loop over files and write to disk
        for idx_file in range(self.n_files):

            # Create output path
            outpath = self.paths_full[idx_file].replace(".full.fits.ctab", ".ahead")

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent) and not self.setup.overwrite:
                continue

            # Create string to write from all scaling factor for current file
            s = "\n".join(["FLXSCALE= {0:0.5f}\nEND".format(x) for x in flx_scale[idx_file]]) + "\n"

            # Write to disk
            with open(outpath, "w") as file:
                file.write(s)

    def paths_qc_plots(self, paths, prefix=""):

        if paths is None:
            return ["{0}{1}.{2}.pdf".format(self.setup.folders["qc_photometry"], fp, prefix) for fp in self.basenames]
        else:
            return paths

    def plot_qc_phot_interror(self):

        # Only works if there are multiple catalogs available
        if len(self) <= 1:
            raise ValueError("QC plot requires multiple catalogs as input")

        # Import
        from astropy import table
        import matplotlib.pyplot as plt
        from astropy.utils.metadata import MergeConflictWarning

        # Construct output path name
        outpath = "{0}{1}.phot.interror.pdf".format(self.setup.folders["qc_photometry"], self.setup.name)

        # Read and clean all tables
        tables_all = flat_list([self.file2table(file_index=i) for i in range(self.n_files)])
        tables_all = [clean_source_table(t) for t in tables_all]

        # Stack all tables into a single master table
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MergeConflictWarning)
            table_master = table.vstack(tables_all)

        # Remove all sources without a match from the master table
        stacked = np.stack([np.deg2rad(table_master[self._key_ra]), np.deg2rad(table_master[self._key_dec])]).T
        dis, idx = NearestNeighbors(n_neighbors=2, metric="haversine").fit(stacked).kneighbors(stacked)
        dis_arsec = np.rad2deg(dis) * 3600
        dis_arcsec_nn = dis_arsec[:, 1]
        good = dis_arcsec_nn < 0.2
        table_master = table_master[good]

        # Remove duplicates
        table_master = remove_duplicates_wcs(table=table_master, sep=1, key_lon=self._key_ra, key_lat=self._key_dec,
                                             temp_dir=self.setup.folders["temp"], bin_name=self.setup.bin_stilts)

        # Create empty array to store all matched magnitudes
        matched_phot = np.full((len(table_master), len(tables_all)), fill_value=np.nan, dtype=np.float32)
        matched_photerr = np.full((len(table_master), len(tables_all)), fill_value=np.nan, dtype=np.float32)

        # Do NN search in parallel (this takes the most time in a loop)
        def __match_catalogs(t, m):
            return NearestNeighbors(n_neighbors=1, metric="haversine").fit(t).kneighbors(m)
        stacked_master = np.stack([np.deg2rad(table_master[self._key_ra]),
                                   np.deg2rad(table_master[self._key_dec])]).T
        stacked_table = [np.stack([np.deg2rad(tt[self._key_ra]),
                                   np.deg2rad(tt[self._key_dec])]).T for tt in tables_all]
        with Parallel(n_jobs=self.setup.n_jobs) as parallel:
            mp = parallel(delayed(__match_catalogs)(i, j) for i, j in zip(stacked_table, repeat(stacked_master)))
        dis_all, idx_all = list(zip(*mp))

        # Now loop over all individual tables and find matches
        for tidx in range(len(tables_all)):

            # Grad current match
            dis, idx = dis_all[tidx], idx_all[tidx]

            # Determine bad matches
            bad_dis = np.rad2deg(dis[:, 0]) * 3600 > 0.2

            # Write nearest neighbor photometry into matched photometry array
            matched_phot[:, tidx] = tables_all[tidx][idx[:, 0]]["MAG_AUTO_CAL"]
            matched_photerr[:, tidx] = tables_all[tidx][idx[:, 0]]["MAGERR_AUTO"]

            # Mask bad matches
            matched_phot[bad_dis, tidx] = np.nan
            matched_photerr[bad_dis, tidx] = np.nan

        # Compute internal phot error
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _, phot_median, phot_err = sigma_clipped_stats(matched_phot, axis=1)
            _, photerr_median, _ = sigma_clipped_stats(matched_photerr, axis=1)

        # Make 1D disperion histograms
        mag_ranges = [0, 14, 15, 16, 17, 18, 25]
        fig, ax_all = plt.subplots(nrows=2, ncols=3,
                                   gridspec_kw=dict(hspace=0.4, wspace=0.3, left=0.06,
                                                    right=0.97, bottom=0.1, top=0.97),
                                   **dict(figsize=(14, 8)))
        ax_all = ax_all.ravel()
        for idx in range(len(mag_ranges) - 1):

            # Grab current axes and sources
            ax = ax_all[idx]
            mag_lo, mag_hi = mag_ranges[idx], mag_ranges[idx + 1]
            idx_phot = (phot_median >= mag_lo) & (phot_median < mag_hi)

            # Get median photometric error for current bin
            median_photerr_median = np.nanmedian(photerr_median[idx_phot])

            # Remove axis is no sources are present
            if np.sum(idx_phot) == 0:
                ax.remove()

            # Draw histogram
            ax.hist(phot_err[idx_phot], bins=np.logspace(np.log10(0.0001), np.log10(2), 50),
                    ec="black", histtype="step", lw=2)
            ax.set_xscale("log")

            # Draw median
            ax.axvline(np.nanmedian(phot_err[idx_phot]), c="#1f77b4", lw=1.5)
            ax.axvline(median_photerr_median, c="crimson", lw=1.5)

            # Labels and annotations
            ax.set_xlabel("Internal photometric dispersion (mag)")
            ax.set_ylabel("Number of sources")
            ax.annotate("[{0:0.1f},{1:0.1f}) mag".format(mag_lo, mag_hi), xy=(0.02, 0.99),
                        xycoords="axes fraction", va="top", ha="left")
            ax.annotate("N = {0}/{1}".format(np.sum(idx_phot), len(idx_phot)), xy=(0.98, 0.98),
                        xycoords="axes fraction", ha="right", va="top")
            ax.annotate("Internal photometric dispersion {0:0.4f} mag".format(np.nanmedian(phot_err[idx_phot])),
                        xy=(0.01, 1.01), xycoords="axes fraction", ha="left", va="bottom", c="#1f77b4")
            ax.annotate("Median photometric error {0:0.4f} mag".format(median_photerr_median),
                        xy=(0.01, 1.07), xycoords="axes fraction", ha="left", va="bottom", c="crimson")

        # Save plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
            fig.savefig(outpath, bbox_inches="tight")
        plt.close("all")

    def plot_qc_phot_zp(self, paths=None, axis_size=5):
        """ Generates ZP QC plot. """

        for idx_file in range(self.n_files):

            # Generate path for plot
            path = self.paths_qc_plots(paths=paths, prefix="zp")[idx_file]

            zp_auto = self.read_from_data_headers(keywords=["HIERARCH PYPE ZP MAG_AUTO"])[0][idx_file]
            zperr_auto = self.read_from_data_headers(keywords=["HIERARCH PYPE ZP ERR MAG_AUTO"])[0][idx_file]

            # Create plot
            plot_value_detector(values=zp_auto, errors=zperr_auto, path=path, ylabel="ZP AUTO (mag)",
                                axis_size=axis_size, yrange=(np.median(zp_auto) - 0.1, np.median(zp_auto) + 0.1))

    def plot_qc_phot_ref1d(self, axis_size=5):

        # Import
        from astropy.units import Unit
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Generate output paths
        outpaths = self.paths_qc_plots(paths=None, prefix="phot.1D")

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

            # Make plot grid
            if len(self.iter_data_hdu[idx_file]) == 1:
                fig, ax_file = plt.subplots(nrows=1, ncols=1, gridspec_kw=None, **dict(figsize=(8, 5)))
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
                dens_zp = \
                    np.exp(kde.fit((mag_delta[keep]).reshape(-1, 1)).score_samples(kde_grid.reshape(-1, 1)))  # noqa

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
                if (idx_hdu < self.setup.fpa_layout[1]) | (len(ax_file) == 1):
                    ax.set_xlabel("{0} {1} (mag)".format(self.setup.phot_reference_catalog.upper(),
                                                         self.passband[idx_file]))
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if (idx_hdu % self.setup.fpa_layout[0] == self.setup.fpa_layout[0] - 1) | (len(ax_file) == 1):
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
                fig.savefig(outpaths[-1], bbox_inches="tight")
            plt.close("all")

    def plot_qc_phot_ref2d(self, axis_size=5):

        # Import
        from astropy.units import Unit
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        for idx_file in range(self.n_files):

            # Generate output path
            outpath = self.paths_qc_plots(paths=None, prefix="phot.2D")[idx_file]

            # Create figure for current file
            if len(self.iter_data_hdu[idx_file]) == 1:
                fig, ax = plt.subplots(nrows=1, ncols=1, gridspec_kw=None, **dict(figsize=(9, 9)))
                ax_file = [ax]
            else:
                fig, ax_file = get_plotgrid(layout=self.setup.fpa_layout, xsize=axis_size, ysize=axis_size)
                ax_file = ax_file.ravel()
            cax = fig.add_axes([0.25, 0.92, 0.5, 0.02])

            # Get passband
            passband = self.passband[idx_file][0]

            # Get master photometry catalog
            master_phot = self.get_master_photometry()[0]
            mkeep = master_phot.get_purge_index(passband=passband)

            # Fetch magnitude and coordinates for master catalog
            mag_master = master_phot.mag(passband=passband)[0][0][mkeep]
            skycoord_master = master_phot.skycoord()[0][0][mkeep]

            # Keep only soruces within mag limit
            mag_lim = master_phot.mag_lim(self.passband[idx_file])
            keep = (mag_master >= mag_lim[0]) & (mag_master <= mag_lim[1])
            mag_master, skycoord_master = mag_master[keep], skycoord_master[keep]

            # Read sources table for current files
            table_file = self.file2table(file_index=idx_file)

            im = None
            for idx_hdu in range(len(self.iter_data_hdu[idx_file])):

                # Grab axes
                ax = ax_file[idx_hdu]

                # Get current image header
                header = self.image_headers[idx_file][idx_hdu]

                # Grab current catalog
                tab_hdu = table_file[idx_hdu]

                # Clean table
                tab_hdu = clean_source_table(table=tab_hdu)

                # Construct skycoordinates
                sc_hdu = SkyCoord(ra=tab_hdu[self._key_ra], dec=tab_hdu[self._key_dec], unit="deg")

                # Xmatch science with reference
                zp_idx, zp_d2d, _ = sc_hdu.match_to_catalog_sky(skycoord_master)

                # Get good indices in reference catalog and in current field
                idx_master = zp_idx[zp_d2d < 1 * Unit("arcsec")]
                idx_final = np.arange(len(zp_idx))[zp_d2d < 1 * Unit("arcsec")]

                # Apply indices filter
                mag_hdu_match = tab_hdu["MAG_AUTO_CAL"][idx_final]
                mag_master_match = mag_master[idx_master]

                # Compute difference between reference and self
                mag_delta = mag_master_match - mag_hdu_match

                # Grab X/Y coordinates
                x_hdu, y_hdu = tab_hdu["X_IMAGE"][idx_final], tab_hdu["Y_IMAGE"][idx_final]

                # Grid data with nearest neighbors
                grid = grid_value_2d_nn(x=x_hdu, y=y_hdu, values=mag_delta,
                                        n_nearest_neighbors=len(x_hdu) if len(x_hdu) < 50 else 50,
                                        n_bins_x=header["NAXIS1"] // 100, n_bins_y=header["NAXIS2"] // 100,
                                        x_min=1, y_min=1, x_max=header["NAXIS1"], y_max=header["NAXIS2"])

                # Draw
                kwargs = {"vmin": -0.1, "vmax": +0.1, "cmap": get_cmap("RdBu", 20)}
                extent = [1, header["NAXIS1"], 1, header["NAXIS2"]]
                im = ax.imshow(grid, extent=extent, origin="lower", **kwargs)
                ax.scatter(x_hdu, y_hdu, c=mag_delta, s=7, lw=0.5, ec="black", **kwargs)

                # Draw contour
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="No contour levels were found within the data range")
                    cs = ax.contour(grid, np.linspace(-0.1, 0.1, 21), colors="k",
                                    origin="lower", extent=extent, vmin=-0.1, vmax=0.1)
                    ax.clabel(cs, inline=True, fontsize=10, fmt="%0.2f")

                # Annotate detector ID
                ax.annotate("Det.ID: {0:0d}".format(idx_hdu + 1), xy=(0.02, 1.01),
                            xycoords="axes fraction", ha="left", va="bottom")

                # Modify axes
                if (idx_hdu < self.setup.fpa_layout[1]) | (len(ax_file) == 1):
                    ax_file[idx_hdu].set_xlabel("X (pix)")
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if (idx_hdu % self.setup.fpa_layout[0] == self.setup.fpa_layout[0] - 1) | (len(ax_file) == 1):
                    ax_file[idx_hdu].set_ylabel("Y (pix)")
                else:
                    ax.axes.yaxis.set_ticklabels([])

                # Equal aspect ratio
                ax.set_aspect("equal")

                # Set ticks
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator())

                # Set limits
                ax.set_xlim(1, header["NAXIS1"])
                ax.set_ylim(1, header["NAXIS2"])

            # Add colorbar
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal", label="Zero Point (mag)",
                                ticks=np.arange(-0.1, 0.11, 0.05))
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")

            # # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
                fig.savefig(outpath, bbox_inches="tight")
            plt.close("all")
