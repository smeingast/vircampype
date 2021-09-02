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
from vircampype.tools.imagetools import *
from vircampype.tools.systemtools import *
from vircampype.data.cube import ImageCube
from sklearn.neighbors import KernelDensity
from vircampype.tools.miscellaneous import *
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS, FITSFixedWarning
from sklearn.neighbors import NearestNeighbors
from vircampype.tools.tabletools import add_zp_2mass
from vircampype.fits.tables.sources import SourceCatalogs
from vircampype.tools.messaging import message_qc_astrometry


class SextractorCatalogs(SourceCatalogs):

    def __init__(self, setup, file_paths=None):
        super(SextractorCatalogs, self).__init__(file_paths=file_paths, setup=setup)

    @property
    def _key_ra(self):
        return "ALPHA_SKY"

    @property
    def _key_dec(self):
        return "DELTA_SKY"

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

        # # Normalize FLXSCALE across all headers
        # flxscale = []
        # for path_ahead in self._scamp_header_paths():
        #     hdrs = read_aheaders(path=path_ahead)
        #     flxscale.extend([h["FLXSCALE"] for h in hdrs])
        #
        #     # Make a backup
        #     path_backup = path_ahead + ".backup"
        #     if not os.path.isfile(path_backup):
        #         copyfile(path_ahead, path_backup)
        #
        # # Compute norm
        # flxscale_norm = np.mean(flxscale)
        #
        # # Loop again and rewrite this time
        # for path_ahead in self._scamp_header_paths():
        #     hdrs = read_aheaders(path=path_ahead)
        #     for h in hdrs:
        #         h["FSCLORIG"] = (h["FLXSCALE"], "Original scamp flux scale")
        #         h["FSCLSTCK"] = (h["FLXSCALE"] / flxscale_norm, "SCAMP relative flux scale for stacks")
        #         del h["FLXSCALE"]
        #
        #     # Rewrite file with new scale
        #     write_aheaders(headers=hdrs, path=path_ahead)

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
        if preset.lower() in ["pawprint", "tile"]:
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

    def build_master_illumination_correction(self):
        """ Illumination correction construction method. """

        # Import
        from vircampype.fits.images.flat import MasterIlluminationCorrection

        # Processing info
        print_header(header="MASTER-ILLUMINATION-CORRECTION", right=None, silent=self.setup.silent)
        tstart = time.time()

        # At the moment, this only works when there is only one passband
        if len(list(set(self.passband))) != 1:
            raise ValueError("Only one passband allowed")
        passband = self.passband[0]

        # Split based on passband and interval
        split = self.split_keywords(keywords=[self.setup.keywords.filter_name])
        split = flat_list([s.split_window(window=0.5, remove_duplicates=True) for s in split])

        # Get master photometry catalog
        master_phot = self.get_master_photometry()
        mkeep = master_phot.get_purge_index(passband=passband)
        mag_master = master_phot.mag(passband=passband)[0][0][mkeep]
        skycoord_master = master_phot.skycoord()[0][0][mkeep]

        # Read, stack, clean tables; determine mean ZP
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tab_all = [self.file2table(file_index=i) for i in range(len(self))]
            tab_all = clean_source_table(vstack(flat_list(tab_all)))
            zp_all = get_zeropoint(skycoord1=SkyCoord(tab_all[self._key_ra], tab_all[self._key_dec], unit="deg"),
                                   mag1=tab_all["MAG_AUTO"], skycoord2=skycoord_master, mag2=mag_master,
                                   mag_limits_ref=master_phot.mag_lim(passband=passband), method="all")
            _, zp_all_median, _ = sigma_clipped_stats(zp_all)

        # Now loop through separated files
        for files, idx_print in zip(split, range(1, len(split) + 1)):

            # Create master dark name
            outpath = self.setup.folders["master_object"] + "MASTER-ILLUMINATION-CORRECTION_" \
                                                            "{0:11.5f}.fits".format(files.mjd_mean)

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent):
                continue

            # Grab current passband
            passband = files.passband[0]

            # Fetch clean index for current filter
            mkeep = master_phot.get_purge_index(passband=passband)

            # Fetch magnitude and coordinates for master catalog
            mag_master = master_phot.mag(passband=passband)[0][0][mkeep]
            # magerr_master = master_phot.mag_err(passband=passband)[0][0][mkeep]
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
                tab = clean_source_table(table=tab, image_header=header, flux_max=header["SEXSATLV"] / 2, border_pix=10,
                                         nndis_limit=5, min_fwhm=0.8, max_fwhm=6.0, max_ellipticity=0.25)

                # Compute illumination correction
                zp_all = get_zeropoint(skycoord1=SkyCoord(tab[self._key_ra], tab[self._key_dec], unit="deg"),
                                       mag1=tab["MAG_AUTO"], skycoord2=skycoord_master, mag2=mag_master,
                                       mag_limits_ref=master_phot.mag_lim(passband=passband), method="all")

                # Remove all table entries without ZP entry
                tab, zp_all = tab[np.isfinite(zp_all)], zp_all[np.isfinite(zp_all)]

                # Grid with NN interpolation
                grid_zp = grid_value_2d_nn(x=tab["XWIN_IMAGE"], y=tab["YWIN_IMAGE"], values=zp_all,
                                           n_bins_x=header["NAXIS1"] // 50, n_bins_y=header["NAXIS2"] // 50,
                                           x_min=1, y_min=1, x_max=header["NAXIS1"], y_max=header["NAXIS2"],
                                           n_nearest_neighbors=50, metric="weighted", weights=1 / tab["MAGERR_AUTO"]**2)

                # Resize to original image size
                grid_zp = upscale_image(grid_zp, new_size=(header["NAXIS1"], header["NAXIS2"]), method="PIL")

                # Constant value
                # zp = get_zeropoint(skycoord_cal=SkyCoord(tab[self._key_ra], tab[self._key_dec], unit="deg"),
                #                    mag_cal=tab["MAG_AUTO"], skycoord_ref=skycoord_master, mag_ref=mag_master,
                #                    mag_limits_ref=master_phot.mag_lim(passband=passband), method="weighted",
                #                    mag_err_cal=tab["MAGERR_AUTO"], mag_err_ref=magerr_master)[0]
                # grid_zp = np.full((header["NAXIS1"], header["NAXIS2"]), fill_value=zp, dtype=np.float32)

                # Convert to flux scale
                flx_scale.append(10**((grid_zp - zp_all_median) / 2.5))

                # Save number of sources
                n_sources.append(len(tab))

                # Plot sources on top of illumination correction
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
            illumcorr = ImageCube(setup=self.setup)

            # Loop over extensions and construct final illumination correction
            for idx_hdu, fscl, nn in zip(files.iter_data_hdu[0], flx_scale, n_sources):

                # Append to output
                illumcorr.extend(data=fscl.astype(np.float32))

                # Create empty header
                data_header = fits.Header()

                # Add data to header
                add_int_to_header(header=data_header, key="HIERARCH PYPE IC NSOURCES", value=nn,
                                  comment="Number of sources used")
                add_float_to_header(header=data_header, key="HIERARCH PYPE IC STD", value=np.nanstd(fscl),  # noqa
                                    decimals=4, comment="Standard deviation in relative flux")

                # Append header
                data_headers.append(data_header)

            # Make primary header
            prime_cards = make_cards(keywords=[self.setup.keywords.object, self.setup.keywords.date_mjd,
                                               self.setup.keywords.filter_name, self.setup.keywords.date_ut,
                                               "HIERARCH PYPE N_FILES"],
                                     values=["MASTER-ILLUMINATION-CORRECTION", files.mjd_mean,
                                             files.passband[0], files.time_obs_mean,
                                             len(files)])
            prime_header = fits.Header(cards=prime_cards)

            # Write to disk
            illumcorr.write_mef(path=outpath, prime_header=prime_header, data_headers=data_headers)

            # QC plot
            if self.setup.qc_plots:
                msf = MasterIlluminationCorrection(setup=self.setup, file_paths=outpath)
                msf.qc_plot2d(paths=None, axis_size=5)

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
        outpaths = []
        plot_photerr_internal = False
        for idx_file in range(self.n_files):

            # Create output path
            outpaths.append(self.paths_full[idx_file].replace(".fits.tab", ".fits.ctab"))

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpaths[-1], silent=self.setup.silent) \
                    and not self.setup.overwrite:
                continue

            # Print processing info
            message_calibration(n_current=idx_file + 1, n_total=self.n_files, name=outpaths[-1],
                                d_current=None, d_total=None, silent=self.setup.silent)

            # Load table
            tables_file = self.file2table(file_index=idx_file)

            # Add aperture correction to tables
            [t.add_column((t["MAG_APER"].data[:, -1] - t["MAG_APER"].data.T).T, name="MAG_APER_COR")
             for t in tables_file]

            # Add smoothed stats to tables
            parameters = ["FWHM_WORLD", "ELLIPTICITY", "MAG_APER_COR"]
            for par in parameters:
                with Parallel(n_jobs=self.setup.n_jobs, prefer="threads") as parallel:
                    tables_file = parallel(delayed(add_smoothed_value)(i, j) for i, j in zip(tables_file, repeat(par)))

            # Load classification table for current file
            try:
                tables_class_file = tables_class.file2table(file_index=idx_file)

                # Interpolate classification in parallel for each extension
                with Parallel(n_jobs=self.setup.n_jobs, prefer="threads") as parallel:
                    tables_file = parallel(delayed(interpolate_classification)(i, j) for i, j
                                           in zip(tables_file, tables_class_file))
            except FileNotFoundError:
                pass

            # Match apertures and add to table
            [t.add_column((t["MAG_APER"] + t["MAG_APER_COR_INTERP"]), name="MAG_APER_MATCHED")
             for t in tables_file]

            # Compute ZP and add calibrated photometry to catalog
            columns_mag = ["MAG_APER", "MAG_APER_MATCHED", "MAG_AUTO",
                           "MAG_ISO", "MAG_ISOCOR", "MAG_PETRO"]
            columns_magerr = ["MAGERR_APER", "MAGERR_APER", "MAGERR_AUTO",
                              "MAGERR_ISO", "MAGERR_ISOCOR", "MAGERR_PETRO"]

            # Open input catalog
            table_hdulist = fits.open(self.paths_full[idx_file], mode="readonly")

            # Compute zero points
            for table_hdu, idx_table_hdu in zip(tables_file, self.iter_data_hdu[idx_file]):
                add_zp_2mass(table_hdu, table_2mass=table_master, key_ra=self._key_ra, key_dec=self._key_dec,
                             mag_lim_ref=master_phot.mag_lim(passband=self.passband[idx_file]), method="weighted",
                             passband_2mass=master_phot.translate_passband(self.passband[idx_file][0]),
                             columns_mag=columns_mag, columns_magerr=columns_magerr)

                # Add correction factor for the main calibrated mag measurement
                sc1 = SkyCoord(table_hdu[self._key_ra], table_hdu[self._key_dec], unit="degree")
                sc2 = SkyCoord(table_master["RAJ2000"], table_master["DEJ2000"], unit="degree")
                pb = master_phot.translate_passband(self.passband[idx_file][0])
                zp_auto = get_zeropoint(skycoord1=sc1, mag1=table_hdu["MAG_AUTO_CAL"], skycoord2=sc2,
                                        method="all", mag2=table_master[pb],
                                        mag_limits_ref=master_phot.mag_lim(passband=self.passband[idx_file]))
                zp_aper = get_zeropoint(skycoord1=sc1, mag1=table_hdu["MAG_APER_MATCHED_CAL"], skycoord2=sc2,
                                        method="all", mag2=table_master[pb],
                                        mag_limits_ref=master_phot.mag_lim(passband=self.passband[idx_file]))
                table_hdu.add_column(zp_auto, name="MAG_AUTO_CAL_ZPC")
                table_hdu.add_column(zp_aper, name="MAG_APER_MATCHED_CAL_ZPC")
                parameters = ["MAG_AUTO_CAL_ZPC", "MAG_APER_MATCHED_CAL_ZPC"]
                for par in parameters:
                    add_smoothed_value(table=table_hdu, parameter=par, n_neighbors=150, max_dis=1800)

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
            table_hdulist.writeto(outpaths[-1], overwrite=self.setup.overwrite)

            # Close original file
            table_hdulist.close()

            # QC plot
            if self.setup.qc_plots:
                pcsc = PhotometricCalibratedSextractorCatalogs(setup=self.setup, file_paths=outpaths[-1])
                pcsc.plot_qc_phot_zp(axis_size=5)
                pcsc.plot_qc_phot_ref1d(axis_size=5)
                pcsc.plot_qc_phot_ref2d(axis_size=5)
                plot_photerr_internal = True

        # Plot internal dispersion if set
        if plot_photerr_internal & (len(outpaths) > 1):
            all_catalogs = PhotometricCalibratedSextractorCatalogs(setup=self.setup, file_paths=outpaths)
            all_catalogs.plot_qc_photerr_internal()

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

    def add_statistics(self):

        # Import
        from vircampype.fits.images.common import FitsImages

        # Processing info
        print_header(header="ADDING STATISTICS", silent=self.setup.silent, right=None)
        tstart = time.time()

        for idx_file in range(self.n_files):

            # Find files
            path_mjd = self.paths_full[idx_file].replace(".full.fits.tab", ".mjdeff.fits")
            path_exptime = self.paths_full[idx_file].replace(".full.fits.tab", ".exptime.fits")
            path_nimg = self.paths_full[idx_file].replace(".full.fits.tab", ".nimg.fits")
            path_weight = self.paths_full[idx_file].replace(".full.fits.tab", ".weight.fits")

            # Check if files are available
            if not os.path.isfile(path_mjd) & os.path.isfile(path_exptime) & os.path.isfile(path_nimg):
                raise ValueError("Matches for image statistics not found")

            # Instantiate
            image_mjdeff = FitsImages(file_paths=path_mjd, setup=self.setup)

            # Open current table file
            hdul = fits.open(self.paths_full[idx_file], mode="update")

            # Check if the last HDU was already modified
            if "MJDEFF" in hdul[self.iter_data_hdu[idx_file][-1]].columns.names:
                print_message(message="{0} already modified.".format(os.path.basename(self.paths_full[idx_file])),
                              kind="warning", end=None)
                continue

            # Loop over extensions
            for idx_hdu_self, idx_hdu_stats in zip(self.iter_data_hdu[idx_file], range(image_mjdeff.n_data_hdu[0])):

                # Read table
                table_hdu = self.filehdu2table(file_index=idx_file, hdu_index=idx_hdu_self)

                # Read stats
                try:
                    mjdeff = fits.getdata(path_mjd, idx_hdu_stats)
                    exptime = fits.getdata(path_exptime, idx_hdu_stats)
                    nimg = fits.getdata(path_nimg, idx_hdu_stats)
                    weight = fits.getdata(path_weight, idx_hdu_stats)
                except IndexError:
                    mjdeff = fits.getdata(path_mjd, idx_hdu_stats+1)
                    exptime = fits.getdata(path_exptime, idx_hdu_stats+1)
                    nimg = fits.getdata(path_nimg, idx_hdu_stats+1)
                    weight = fits.getdata(path_weight, idx_hdu_stats+1)

                # Renormalize weight
                weight /= np.median(weight)

                # Obtain wcs for statistics images (they all have the same projection)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    wcs_stats = WCS(header=image_mjdeff.headers_data[0][idx_hdu_stats])

                # Convert to X/Y
                xx, yy = wcs_stats.wcs_world2pix(table_hdu[self._key_ra], table_hdu[self._key_dec], 0)
                xx_image, yy_image = xx.astype(int), yy.astype(int)

                # Mark bad data
                bad = (xx_image >= mjdeff.shape[1]) | (xx_image < 0) | \
                      (yy_image >= mjdeff.shape[0]) | (yy_image < 0)

                # Just to be sort of safe, let's say we can't have more than 5% of sources outside the edges
                if sum(bad) / len(bad) > 0.05:
                    raise ValueError("Too many sources are close to the image edge ({0}/{1}). "
                                     "Please check for issues. (file: {2}, TableHDU: {3})"
                                     "".format(sum(bad), len(bad), self.paths_full[idx_file], idx_hdu_self))

                # Reset bad coordinates to 0/0
                xx_image[bad], yy_image[bad] = 0, 0

                # Get values for each source from data arrays
                mjdeff_sources, exptime_sources = mjdeff[yy_image, xx_image], exptime[yy_image, xx_image]
                nimg_sources, weight_sources = nimg[yy_image, xx_image], weight[yy_image, xx_image]

                # Mask bad sources
                bad &= weight_sources < 0.0001
                mjdeff_sources[bad], exptime_sources[bad], nimg_sources[bad] = np.nan, 0, 0

                # Make new columns
                new_cols = fits.ColDefs([fits.Column(name="MJDEFF", format="D", array=mjdeff_sources),
                                         fits.Column(name='EXPTIME', format="J", array=exptime_sources, unit="seconds"),
                                         fits.Column(name='NIMG', format="J", array=nimg_sources)])

                # Append new columns and replace HDU
                hdul[idx_hdu_self] = fits.BinTableHDU.from_columns(hdul[idx_hdu_self].data.columns + new_cols,
                                                                   header=hdul[idx_hdu_self].header)

            hdul.flush()

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

    def plot_qc_astrometry_1d(self, axis_size=5):

        # Import
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Processing info
        print_header(header="QC ASTROMETRY 1D", silent=self.setup.silent)
        tstart = time.time()

        # Get FPA layout
        fpa_layout = self.setup.fpa_layout

        # Obtain master coordinates
        sc_master_raw = self.get_master_astrometry().skycoord()[0][0]

        # Apply space motion to match data obstime
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sc_master_equal = sc_master_raw.apply_space_motion(new_obstime=self.time_obs_mean)

        # Loop over files
        for idx_file in range(len(self)):

            # Generate outpath
            outpath_sep = "{0}{1}_astr_referr_sep.pdf".format(self.setup.folders["qc_astrometry"], self.names[idx_file])
            outpath_ang = "{0}{1}_astr_referr_ang.pdf".format(self.setup.folders["qc_astrometry"], self.names[idx_file])

            # Check if file already exists
            if check_file_exists(file_path=outpath_ang, silent=self.setup.silent):
                continue

            # Grab coordinates
            sc_file = self.skycoord()[idx_file]

            # Coadd mode
            if len(self) == 1:
                fig1, ax_all1 = get_plotgrid(layout=(1, 1), xsize=4*axis_size, ysize=4*axis_size)
                ax_all1 = [ax_all1]
                fig2, ax_all2 = get_plotgrid(layout=(1, 1), xsize=4*axis_size, ysize=4*axis_size)
                ax_all2 = [ax_all2]
            else:
                fig1, ax_all1 = get_plotgrid(layout=fpa_layout, xsize=axis_size, ysize=axis_size)
                ax_all1 = ax_all1.ravel()
                fig2, ax_all2 = get_plotgrid(layout=fpa_layout, xsize=axis_size, ysize=axis_size)
                ax_all2 = ax_all2.ravel()

            # Loop over extensions
            for idx_hdu in range(len(sc_file)):

                # Print processing info
                message_calibration(n_current=idx_file+1, n_total=len(self), name=outpath_sep, d_current=idx_hdu+1,
                                    d_total=len(sc_file), silent=self.setup.silent)

                # Grab data for current HDU
                sc_hdu = sc_file[idx_hdu]

                # Get separations and position angles between matched master and current table
                idx, sep2d_equal = sc_hdu.match_to_catalog_sky(sc_master_equal, nthneighbor=1)[:2]
                i1 = sep2d_equal.arcsec <= 0.5
                i2, sep2d_equal = idx[i1], sep2d_equal[i1]
                ang_equal = sc_hdu[i1].position_angle(sc_master_equal[i2])

                # Get separations and position angles between matched master and current table
                idx, sep2d_raw = sc_hdu.match_to_catalog_sky(sc_master_raw, nthneighbor=1)[:2]
                i1 = sep2d_raw.arcsec <= 0.5
                i2, sep2d_raw = idx[i1], sep2d_raw[i1]
                ang_raw = sc_hdu[i1].position_angle(sc_master_raw[i2])

                # Draw separation histograms
                kwargs_hist = dict(range=(0, 100), bins=20, histtype="step", lw=2.0, ls="solid", alpha=0.7)
                ax_all1[idx_hdu].hist(sep2d_equal.mas, ec="crimson", label="Equalized epoch", **kwargs_hist)
                ax_all1[idx_hdu].hist(sep2d_raw.mas, ec="dodgerblue", label="Raw epoch", **kwargs_hist)
                ax_all1[idx_hdu].axvline(0, c="black", ls="dashed", lw=1)

                # Draw position angle histgrams
                kwargs_hist = dict(range=(0, 360), bins=20, histtype="step", lw=2.0, ls="solid", alpha=0.7)
                ax_all2[idx_hdu].hist(ang_equal.degree, ec="crimson", label="Equalized epoch", **kwargs_hist)
                ax_all2[idx_hdu].hist(ang_raw.degree, ec="dodgerblue", label="Raw epoch", **kwargs_hist)

                # Modify axes
                for ax, ll in zip([ax_all1[idx_hdu], ax_all2[idx_hdu]], ["Separation (mas)", "Position angle (deg)"]):

                    # Annotate detector ID
                    ax.annotate("Det.ID: {0:0d}".format(idx_hdu + 1), xy=(0.02, 1.01),
                                xycoords="axes fraction", ha="left", va="bottom")

                    # Modify axes
                    if idx_hdu < fpa_layout[1]:
                        ax.set_xlabel(ll)
                    else:
                        ax.axes.xaxis.set_ticklabels([])
                    if (idx_hdu + 1) % fpa_layout[0] == 0:
                        ax.set_ylabel("N")
                    else:
                        ax.axes.yaxis.set_ticklabels([])

                    # Set ticks
                    ax.xaxis.set_major_locator(MaxNLocator(5))
                    ax.xaxis.set_minor_locator(AutoMinorLocator())
                    ax.yaxis.set_major_locator(MaxNLocator(5))
                    ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Set label on last iteration
            for ax in [ax_all1[-1], ax_all2[-1]]:
                ax.legend(loc="lower left", bbox_to_anchor=(0.01, 1.02), ncol=2,
                          fancybox=False, shadow=False, frameon=False)

            # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
                fig1.savefig(outpath_sep, bbox_inches="tight")
                fig2.savefig(outpath_ang, bbox_inches="tight")
            plt.close("all")

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

    def plot_qc_astrometry_2d(self, axis_size=5, key_x="XWIN_IMAGE", key_y="YWIN_IMAGE"):

        # Import
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Processing info
        print_header(header="QC ASTROMETRY 2D", silent=self.setup.silent)
        tstart = time.time()

        # Get FPA layout
        fpa_layout = self.setup.fpa_layout

        # Obtain master coordinates
        sc_master = self.get_master_astrometry().skycoord()[0][0]

        # Loop over files
        for idx_file in range(len(self)):

            # Generate outpath
            outpath = "{0}{1}_astr_referror2d.pdf".format(self.setup.folders["qc_astrometry"], self.names[idx_file])

            # Check if file exists
            if check_file_exists(file_path=outpath, silent=self.setup.silent):
                continue

            # Grab coordinates
            xx_file = self.get_column_file(idx_file=idx_file, column_name=key_x)
            yy_file = self.get_column_file(idx_file=idx_file, column_name=key_y)
            snr_file = self.get_column_file(idx_file=idx_file, column_name="SNR_WIN")
            sc_file = self.skycoord()[idx_file]

            # Apply space motion to match data obstime
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                sc_master_matched = sc_master.apply_space_motion(new_obstime=self.time_obs[idx_file])

            # Coadd mode
            if len(self) == 1:
                fig, ax_all = get_plotgrid(layout=(1, 1), xsize=4*axis_size, ysize=4*axis_size)
                ax_all = [ax_all]
            else:
                fig, ax_all = get_plotgrid(layout=fpa_layout, xsize=axis_size, ysize=axis_size)
                ax_all = ax_all.ravel()
            cax = fig.add_axes([0.3, 0.92, 0.4, 0.02])

            # Loop over extensions
            im, sep_all = None, []
            for idx_hdu in range(len(sc_file)):

                # Print processing info
                message_calibration(n_current=idx_file+1, n_total=len(self), name=outpath, d_current=idx_hdu+1,
                                    d_total=len(sc_file), silent=self.setup.silent)

                # Read header
                header = self.image_headers[idx_file][idx_hdu]

                # Get separations between master and current table
                i1, sep, _ = sc_file[idx_hdu].match_to_catalog_sky(sc_master_matched)

                # Extract position angles between master catalog and input
                # sc1 = sc_master_astrometry[i1]
                # ang = sc1.position_angle(sc_hdu)

                # Keep only those with a maximum of 0.5 arcsec
                keep = sep.arcsec < 0.5
                sep, x_hdu, y_hdu = sep[keep], xx_file[idx_hdu][keep], yy_file[idx_hdu][keep]
                snr_hdu = snr_file[idx_hdu][keep]

                # Determine number of bins (with given radius at least 10 sources)
                stacked = np.stack([x_hdu, y_hdu]).T
                dis, _ = NearestNeighbors(n_neighbors=51, algorithm="auto").fit(stacked).kneighbors(stacked)
                maxdis = np.percentile(dis[:, -1], 95)
                n_bins_x, n_bins_y = int(header["NAXIS1"] / maxdis), int(header["NAXIS2"] / maxdis)

                # Minimum number of 3 bins
                n_bins_x = 3 if n_bins_x <= 3 else n_bins_x
                n_bins_y = 3 if n_bins_y <= 3 else n_bins_y

                # Grid value into image
                grid = grid_value_2d_nn(x=x_hdu, y=y_hdu, values=sep.mas, n_nearest_neighbors=20, n_bins_x=n_bins_x,
                                        n_bins_y=n_bins_y, x_min=1, y_min=1, x_max=header["NAXIS1"],
                                        y_max=header["NAXIS2"], metric="weighted", weights=snr_hdu)

                # Save high SN separations
                sep_all.append(sep.mas[snr_hdu > np.nanpercentile(snr_hdu, 90)])

                # Draw
                kwargs = dict(vmin=0, vmax=100, cmap="Spectral_r")
                extent = [0, header["NAXIS1"], 0, header["NAXIS2"]]
                im = ax_all[idx_hdu].imshow(grid, extent=extent, origin="lower", **kwargs)
                ax_all[idx_hdu].scatter(x_hdu, y_hdu, c=sep.mas, s=5, lw=0.5, ec="black", alpha=0.5, **kwargs)

                # Annotate detector ID
                ax_all[idx_hdu].annotate("Det.ID: {0:0d}".format(idx_hdu + 1), xy=(0.02, 1.01),
                                         xycoords="axes fraction", ha="left", va="bottom")

                # Modify axes
                if idx_hdu < fpa_layout[1]:
                    ax_all[idx_hdu].set_xlabel("X (pix)")
                else:
                    ax_all[idx_hdu].axes.xaxis.set_ticklabels([])
                if (idx_hdu + 1) % fpa_layout[0] == 0:
                    ax_all[idx_hdu].set_ylabel("Y (pix)")
                else:
                    ax_all[idx_hdu].axes.yaxis.set_ticklabels([])

                ax_all[idx_hdu].set_aspect("equal")

                # Set ticks
                ax_all[idx_hdu].xaxis.set_major_locator(MaxNLocator(5))
                ax_all[idx_hdu].xaxis.set_minor_locator(AutoMinorLocator())
                ax_all[idx_hdu].yaxis.set_major_locator(MaxNLocator(5))
                ax_all[idx_hdu].yaxis.set_minor_locator(AutoMinorLocator())

                # Left limit
                ax_all[idx_hdu].set_xlim(extent[0], extent[1])
                ax_all[idx_hdu].set_ylim(extent[2], extent[3])

            # Add colorbar
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal", label="Average separation (mas)")
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")

            # Print external error stats
            message_qc_astrometry(separation=flat_list(sep_all))

            # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
                fig.savefig(outpath, bbox_inches="tight")
            plt.close("all")

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")


class PhotometricCalibratedSextractorCatalogs(AstrometricCalibratedSextractorCatalogs):

    def __init__(self, setup, file_paths=None):
        super(PhotometricCalibratedSextractorCatalogs, self).__init__(file_paths=file_paths, setup=setup)

    def _merged_table(self, clean=True):

        # Import
        from astropy import table
        from astropy.utils.metadata import MergeConflictWarning

        # Read all tables
        tables_all = flat_list([self.file2table(file_index=i) for i in range(self.n_files)])

        # Clean if set
        if clean:
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

        # Return
        return table_master

    def _photerr_internal(self):

        # Only works if there are multiple catalogs available
        if len(self) <= 1:
            raise ValueError("Internal photometric error requires multiple catalogs.")

        # Read all tables
        tables_all = flat_list([self.file2table(file_index=i) for i in range(self.n_files)])

        # Get merged master table
        table_master = self._merged_table(clean=True)

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
        with Parallel(n_jobs=self.setup.n_jobs, prefer="threads") as parallel:
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

        # Return
        return phot_median, phot_err, photerr_median

    def photerr_internal(self):

        # Get internal photometric error stats
        phot_median, phot_err, photerr_median = self._photerr_internal()

        # Get the 5% brightest sources
        good = phot_median >= self.setup.reference_mag_lim[0]
        idx_bright = phot_median[good] < np.percentile(phot_median[good], 5)

        # Get median error of those
        return clipped_median(phot_err[good][idx_bright], sigma=2)

    def paths_qc_plots(self, paths, prefix=""):

        if paths is None:
            return ["{0}{1}.{2}.pdf".format(self.setup.folders["qc_photometry"], fp, prefix) for fp in self.basenames]
        else:
            return paths

    def plot_qc_photerr_internal(self):

        # Import
        import matplotlib.pyplot as plt

        # Create output path
        outpath = "{0}{1}.phot.interror.pdf".format(self.setup.folders["qc_photometry"], self.setup.name)

        # Get internal photometric error stats
        phot_median, phot_err, photerr_median = self._photerr_internal()

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
                ax.scatter(mag_master_match, mag_delta, c="black",
                           vmin=0, vmax=1.0, s=2, lw=0, alpha=0.4, zorder=0)

                # Draw for sources within mag limits
                ax.scatter(mag_master_match[keep], mag_delta[keep], c="crimson",
                           vmin=0, vmax=1.0, s=4, lw=0, alpha=1.0, zorder=0)

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
                # ax.scatter(x_hdu, y_hdu, c=mag_delta, s=7, lw=0.5, ec="black", **kwargs)

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
