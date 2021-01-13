# =========================================================================== #
# Import
import os
import pickle
import warnings
import numpy as np

from astropy.io import fits
from astropy.time import Time
from vircampype.setup import *
from astropy.table import vstack
from vircampype.utils.wcs import *
from vircampype.utils.math import *
from astropy.wcs import wcs as awcs
from vircampype.utils.plots import *
from vircampype.utils.tables import *
from vircampype.utils.vizier import *
from vircampype.utils.system import *
from vircampype.utils.fitstools import *
from astropy.coordinates import SkyCoord
from vircampype.utils.photometry import *
from vircampype.utils.astromatic import *
from vircampype.data.cube import ImageCube
from sklearn.neighbors import KernelDensity
from vircampype.utils.miscellaneous import *
from sklearn.neighbors import NearestNeighbors
from vircampype.fits.tables.sources import SourceCatalogs
from astropy.stats import sigma_clip as astropy_sigma_clip
from vircampype.utils.system import run_cmds, run_command_bash, which


class SextractorCatalogs(SourceCatalogs):

    def __init__(self, setup, file_paths=None):
        super(SextractorCatalogs, self).__init__(file_paths=file_paths, setup=setup)

        # Generate image header paths
        self._image_header_paths = ["{0}{1}.image.header".format(self.path_headers, x) for x in self.base_names]

    # =========================================================================== #
    # Some defintions
    # =========================================================================== #
    @property
    def _key_ra(self):
        return "ALPHA_J2000"

    @property
    def _key_dec(self):
        return "DELTA_J2000"

    @property
    def apertures(self):
        return str2list(self.setup["photometry"]["apertures"], sep=",", dtype=float)

    # =========================================================================== #
    # Scamp
    # =========================================================================== #
    @property
    def _bin_scamp(self):
        """
        Searches for scamp executable and returns path.

        Returns
        -------
        str
            Path to scamp executable.

        """
        return which(self.setup["astromatic"]["bin_scamp"])

    @property
    def _scamp_default_config(self):
        """
        Searches for default config file in resources.

        Returns
        -------
        str
            Path to default config

        """
        return get_resource_path(package="vircampype.resources.astromatic.scamp", resource="default.config")

    @property
    def _scamp_preset_package(self):
        """
        Internal package preset path for scamp.

        Returns
        -------
        str
            Package path.
        """

        return "vircampype.resources.astromatic.scamp.presets"

    @staticmethod
    def _scamp_qc_types(joined=False):
        """
        QC check plot types for scamp.

        Parameters
        ----------
        joined : bool, optional
            If set, list will be joined by ',' to make it readable for scamp.

        Returns
        -------
        iterable, str
            List or str with QC checkplot types.

        """
        types = ["FGROUPS", "DISTORTION", "ASTR_INTERROR2D", "ASTR_INTERROR1D", "ASTR_REFERROR2D", "ASTR_REFERROR1D"]
        if joined:
            return ",".join(types)
        else:
            return types

    def _scamp_qc_names(self, joined=False):
        """
        List or str containing scamp QC plot names.

        Parameters
        ----------
        joined : bool, optional
            If set, list will be joined by ',' to make it readable for scamp.

        Returns
        -------
        iterable, str
            List or str with QC checkplot types.

        """
        names = ["{0}scamp_{1}".format(self.path_qc_astrometry, qt.lower()) for qt in
                 self._scamp_qc_types(joined=False)]
        if joined:
            return ",".join(names)
        else:
            return names

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

        names = [x.replace(".scamp.sources.fits", ".ahead") for x in self.full_paths]
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
        return " ".join(self.full_paths)

    def scamp(self):

        # Get passband
        bands = list(set(self.filter))
        if len(bands) != 1:
            raise ValueError("Sequence contains multiple filter")
        else:
            band = bands[0][0]  # THIS should only keep J,H, and K for 2MASS (First band and first letter)
            band = "Ks" if "k" in band.lower() else band

        # Load preset
        options = yml2config(nthreads=self.setup["misc"]["n_jobs"],
                             checkplot_type=self._scamp_qc_types(joined=True),
                             checkplot_name=self._scamp_qc_names(joined=True),
                             skip=["HEADER_NAME", "AHEADER_NAME", "ASTREF_BAND"],
                             path=get_resource_path(package=self._scamp_preset_package, resource="scamp.yml"))

        # Construct commands for source extraction
        cmd = "{0} {1} -c {2} -HEADER_NAME {3} -ASTREF_BAND {4} {5}" \
              "".format(self._bin_scamp, self._scamp_catalog_paths, self._scamp_default_config,
                        self._scamp_header_paths(joined=True), band, options)

        # Run Scamp
        run_command_bash(cmd, silent=False)

    def fwhm_from_columns(self):
        """ Estimates average PSF_FWHMs based on measured values by sextractor. """

        psf_fwhm_files = []
        for idx_file in range(len(self)):

            # Read entire table
            tab_file = self.file2table(file_index=idx_file)

            psf_fwhm_hdus = []
            for hdu_idx in range(len(tab_file)):

                # Table for current extension
                tab_hdu = tab_file[hdu_idx]

                # Apply source filter
                tab_hdu_clean = clean_source_table(table=tab_hdu)

                # Sigma clipping
                good = ~astropy_sigma_clip(tab_hdu_clean["FWHM_IMAGE"]).mask

                # Apply filter
                data, weights = tab_hdu_clean["FWHM_IMAGE"][good], tab_hdu_clean["FWHM_IMAGE"][good]

                # Get weighted average and append
                avg = np.average(data, weights=weights)
                err = np.sqrt(np.average((data - avg)**2, weights=weights))

                # Append
                psf_fwhm_hdus.append((avg, err))

            # Append to file lists
            psf_fwhm_files.append(psf_fwhm_hdus)

        # Return
        return psf_fwhm_files

    # =========================================================================== #
    # PSFEX
    # =========================================================================== #
    @property
    def _bin_psfex(self):
        """
        Searches for psfex executable and returns path.

        Returns
        -------
        str
            Path to psfex executable.

        """
        return which(self.setup["astromatic"]["bin_psfex"])

    @property
    def _psfex_default_config(self):
        """
        Searches for default config file in resources.

        Returns
        -------
        str
            Path to default config

        """
        return get_resource_path(package="vircampype.resources.astromatic.psfex", resource="default.config")

    @property
    def _psfex_preset_package(self):
        """
        Internal package preset path for psfex.

        Returns
        -------
        str
            Package path.
        """

        return "vircampype.resources.astromatic.psfex.presets"

    @staticmethod
    def _psfex_checkplot_types(joined=False):
        """
        QC check plot types for psfex.

        Parameters
        ----------
        joined : bool, optional
            If set, list will be joined by ',' to make it readable for psfex.

        Returns
        -------
        iterable, str
            List or str with QC checkplot types.

        """
        types = ["SELECTION_FWHM", "FWHM", "ELLIPTICITY", "COUNTS", "COUNT_FRACTION", "CHI2", "RESIDUALS"]
        if joined:
            return ",".join(types)
        else:
            return types

    def _psfex_checkplot_names(self, joined=False):
        """
        List or str containing psfex QC plot names.

        Parameters
        ----------
        joined : bool, optional
            If set, list will be joined by ',' to make it readable for psfex.

        Returns
        -------
        iterable, str
            List or str with QC checkplot names.

        """
        names = ["{0}psfex_checkplot_{1}".format(self.path_qc_psf, qt.lower()) for qt in
                 self._psfex_checkplot_types(joined=False)]
        if joined:
            return ",".join(names)
        else:
            return names

    @staticmethod
    def _psfex_checkimage_types(joined=False):
        """
        QC check image types for psfex.

        Parameters
        ----------
        joined : bool, optional
            If set, list will be joined by ',' to make it readable for psfex.

        Returns
        -------
        iterable, str
            List or str with QC checkimage types.

        """
        types = ["SNAPSHOTS_IMRES", "SAMPLES", "RESIDUALS"]
        if joined:
            return ",".join(types)
        else:
            return types

    def _psfex_checkimage_names(self, joined=False):
        """
        List or str containing psfex check image names.

        Parameters
        ----------
        joined : bool, optional
            If set, list will be joined by ',' to make it readable for psfex.

        Returns
        -------
        iterable, str
            List or str with QC check image names.

        """
        names = ["{0}psfex_checkimage_{1}".format(self.path_qc_psf, qt.lower()) for qt in
                 self._psfex_checkimage_types(joined=False)]
        if joined:
            return ",".join(names)
        else:
            return names

    def psfex(self):

        # Processing info
        tstart = message_mastercalibration(master_type="PSFEX", silent=self.setup["misc"]["silent"],
                                           left="Running PSFEx on {0} files with {1} threads"
                                                "".format(len(self), self.setup["misc"]["n_jobs"]), right=None)

        # Construct output psf paths
        psf_paths = ["{0}{1}.psfex".format(self.path_master_object, fn) for fn in self.file_names]

        # Check for existing PSF models
        done = [os.path.exists(p) for p in psf_paths]

        # Print statement on already existing files
        for p in psf_paths:
            check_file_exists(file_path=p, silent=self.setup["misc"]["silent"])

        # Return if nothing to be done
        if sum(done) == len(self):
            message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])
            return

        # Read PSFEX setup
        o = read_setup(path_yaml=get_resource_path(package=self._psfex_preset_package, resource="psfex.yml"))

        # Read source tables and stack all HDUs in a file
        options = []
        for i in range(self.n_files):

            # Read tables for current file
            tables = self.file2table(file_index=i)

            # Clean tables
            fwhm_range = [float(x) for x in o["SAMPLE_FWHMRANGE"].split(",")]
            kwargs_clean = dict(snr_limit=o["SAMPLE_MINSN"], nndis_limit=5, max_ellipticity=o["SAMPLE_MAXELLIP"],
                                min_fwhm=fwhm_range[0], max_fwhm=fwhm_range[1])
            tables = [clean_source_table(table=t, **kwargs_clean) for t in tables]

            # Loop over each HDU
            psf_var_hdu = []
            for t in tables:

                # Determine PSF_VAR based on number of sources
                if len(t) < 500:
                    psf_var = 2
                elif len(t) < 5000:
                    psf_var = 3
                elif len(t) < 20000:
                    psf_var = 4
                elif len(t) < 50000:
                    psf_var = 5
                elif len(t) < 100000:
                    psf_var = 6
                else:
                    psf_var = 7

                psf_var_hdu.append(psf_var)

            # Take minimum across all HDUs but at least 2
            psfvar_degrees = min(psf_var_hdu) if min(psf_var_hdu) > 2 else 2

            # Set degree of PSF variability (it actually would make more sense to select PSFVAR_DEGREES and not N_SNAP
            n_snap = psfvar_degrees * 10

            # Construct PSFEx options for current file
            options.append(yml2config(nthreads=1, psfvar_degrees=psfvar_degrees, psfvar_nsnap=n_snap,
                                      checkplot_type=self._psfex_checkplot_types(joined=True),
                                      checkplot_name=self._psfex_checkplot_names(joined=True),
                                      checkimage_type=self._psfex_checkimage_types(joined=True),
                                      checkimage_name=self._psfex_checkimage_names(joined=True),
                                      psf_dir=self.path_master_object, skip=["homokernel_dir"],
                                      path=get_resource_path(package=self._psfex_preset_package, resource="psfex.yml")))

        # Construct commands
        cmds = ["{0} {1} -c {2} {3}".format(self._bin_psfex, tab, self._psfex_default_config, o)
                for tab, o in zip(self.full_paths, options)]

        # Clean commands
        cmds = [c for c, d in zip(cmds, done) if not d]

        # Set number of parallel jobs
        n_jobs_psfex = 1
        n_jobs_shell = self.setup["misc"]["n_jobs"]

        # If there are less files than parallel jobs, optimize psfex jobs
        if (len(cmds) > 0) & (len(cmds) < self.setup["misc"]["n_jobs"]):
            n_jobs_shell = len(cmds)
            while n_jobs_psfex * n_jobs_shell < self.setup["misc"]["n_jobs"]:
                n_jobs_psfex += 1

            # Adapt commands
            cmds = [c.replace("NTHREADS 1", "NTHREADS {0}".format(n_jobs_psfex)) for c in cmds]

        # Run PSFEX
        run_cmds(cmds=cmds, silent=True, n_processes=n_jobs_shell)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    # =========================================================================== #
    # Aperture matching
    # =========================================================================== #
    def aperture_matching(self):
        """
        Computes aperture corrections for each detector.

        Returns
        -------
        ApcorImages

        """

        # Processing info
        tstart = message_mastercalibration(master_type="APERTURE MATCHING", silent=self.setup["misc"]["silent"],
                                           right=None)

        # Find weights
        from vircampype.fits.images.flat import WeightImages
        paths_weights = [x.replace(".full.sources.", ".weight.") for x in self.full_paths]

        # Dummy check weights
        fe = [os.path.exists(p) for p in paths_weights]
        if np.sum(fe) != len(self):
            raise ValueError("Not all files have associated weights! n_images = {0}; n_weight = {1}"
                             "".format(len(self), np.sum(fe)))

        # Read weights into new instance
        weight_images = WeightImages(setup=self.setup, file_paths=paths_weights)

        # Loop over files
        for idx in range(len(self)):

            # Generate output names
            path_file = "{0}{1}.apcor.fits".format(self.path_apcor, self.file_names[idx])
            path_weight = "{0}{1}.apcor.weight.fits".format(self.path_apcor, self.file_names[idx])

            if check_file_exists(file_path=path_file.replace(".apcor.", ".apcor{0}.".format(self.apertures[0])),
                                 silent=self.setup["misc"]["silent"]):
                continue

            # Read currect catalog
            tables = self.file2table(file_index=idx)

            # Get current image header
            headers = self.image_headers[idx]

            # Make output Apcor Image HDUlist
            hdulist_base = fits.HDUList(hdus=[fits.PrimaryHDU(header=self.headers_primary[idx].copy())])
            hdulist_apcor = [hdulist_base.copy() for _ in range(len(self.apertures))]
            hdulist_weight = hdulist_base.copy()

            # Dummy check
            if len(tables) != len(headers):
                raise ValueError("Number of tables and headers no matching")

            # Loop over extensions and get aperture correction after filtering
            for tab, hdr, whdu_idx in zip(tables, headers, weight_images.data_hdu[idx]):

                # Print processing info
                message_calibration(n_current=idx+1, n_total=len(self), name=path_file, d_current=whdu_idx,
                                    d_total=len(headers), silent=self.setup["misc"]["silent"])

                # Retain only clean sources
                tab = clean_source_table(table=tab, image_header=hdr, return_filter=False, snr_limit=50,
                                         flux_max=hdr["SEXSATLV"] / 2)

                # Compute aperture correction for each source
                mag_apcor = tab["MAG_APER"][:, -1][:, np.newaxis] - tab["MAG_APER"]

                # Shrink original image header and keep only WCS info
                ohdr = resize_header(header=hdr, factor=self.setup["photometry"]["apcor_image_scale"])
                naxis1, naxis2 = ohdr["NAXIS1"], ohdr["NAXIS2"]
                ohdr = awcs.WCS(ohdr).to_header()
                ohdr["NAXIS1"], ohdr["NAXIS2"], ohdr["NAXIS"] = naxis1, naxis2, 2

                # Remove DATE-OBS and MJD-OBS (would lead to annoying warnings when reading files)
                for kw in ["DATE-OBS", "MJD-OBS", "MJDREF"]:
                    ohdr = delete_keyword(header=ohdr, keyword=kw)

                # Determine output size
                output_size = (ohdr["NAXIS1"], ohdr["NAXIS2"])

                # Loop over apertures
                for mag, aidx in zip(mag_apcor.T, range(len(self.apertures))):

                    # Determine number of bins (with given radius at least 10 sources)
                    stacked = np.stack([tab["XWIN_IMAGE"], tab["YWIN_IMAGE"]]).T
                    dis, _ = NearestNeighbors(n_neighbors=11, algorithm="auto").fit(stacked).kneighbors(stacked)
                    maxdis = np.percentile(dis[:, -1], 95)
                    n_bins_x, n_bins_y = int(hdr["NAXIS1"] / maxdis), int(hdr["NAXIS2"] / maxdis)

                    # Minimum size of 3
                    n_bins_x = 3 if n_bins_x <= 3 else n_bins_x
                    n_bins_y = 3 if n_bins_y <= 3 else n_bins_y

                    # Grid
                    apc_grid = grid_value_2d(x=tab["XWIN_IMAGE"], y=tab["YWIN_IMAGE"], value=mag, x_min=0, y_min=0,
                                             x_max=hdr["NAXIS1"], y_max=hdr["NAXIS2"], nx=n_bins_x, ny=n_bins_y,
                                             conv=True, weights=tab["SNR_WIN"], upscale=False, kernel_size=2)

                    # Rescale to given size
                    apc_grid = upscale_image(apc_grid, new_size=output_size, method="spline", order=2)

                    # Get weighted mean aperture correction
                    apc_average = np.average(mag, weights=tab["SNR_WIN"])
                    apc_err = np.sqrt(np.average((mag - apc_average)**2, weights=tab["SNR_WIN"]))

                    # # This plots all sources on top of the current aperture correction image
                    # from astropy.visualization import simple_norm
                    # import matplotlib.pyplot as plt
                    # fig, ax = plt.subplots(nrows=1, ncols=1, gridspec_kw=None, **dict(figsize=(7, 4)))
                    # kwargs = dict(cmap="RdYlBu", norm=simple_norm(apc_grid, min_percent=1, percent=98))
                    # im = ax.imshow(apc_grid, origin="lower", extent=[0, hdr["NAXIS1"], 0, hdr["NAXIS2"]], **kwargs)
                    # ax.scatter(tab["XWIN_IMAGE"], tab["YWIN_IMAGE"], c=mag, lw=0.5, ec="black", s=30, **kwargs)
                    # plt.colorbar(im)
                    # plt.show()
                    # exit()

                    # Construct header to append
                    hdr_temp = ohdr.copy()
                    hdr_temp["NSRCAPC"] = (len(mag), "Number of sources used")
                    hdr_temp["MAGAPC"] = (apc_average, "Average aperture correction (mag)")
                    hdr_temp["STDAPC"] = (apc_err, "Aperture correction std across detector (mag)")
                    hdr_temp["DIAMAPC"] = (self.apertures[aidx], "Aperture diameter (pix)")

                    # Construct and add image HDU
                    # noinspection PyTypeChecker
                    hdulist_apcor[aidx].append(fits.ImageHDU(data=apc_grid.astype(np.float32), header=hdr_temp))

                # Read weight
                weight = fits.getdata(weight_images.full_paths[idx], whdu_idx, header=False)

                # Resize weight with PIL
                weight = upscale_image(weight, new_size=output_size, method="pil")

                # Construct and add image weight HDU
                # noinspection PyTypeChecker
                hdulist_weight.append(fits.ImageHDU(data=weight, header=ohdr))

            # Save aperture correction as MEF
            paths = []
            for hdul, diams in zip(hdulist_apcor, self.apertures):

                # Write aperture correction diameter into primary header too
                hdul[0].header["APCDIAM"] = (diams, "Aperture diameter (pix)")
                hdul[0].header[self.setup["keywords"]["object"]] = "APERTURE-CORRECTION"

                # Save to disk
                paths.append(path_file.replace(".apcor.", ".apcor{0}.".format(diams)))
                hdul.writeto(paths[-1], overwrite=self.setup["misc"]["overwrite"])

            # Save weight to disk
            hdulist_weight.writeto(path_weight, overwrite=self.setup["misc"]["overwrite"])

            # QC plot
            # if self.setup["misc"]["qc_plots"]:
            #     from vircampype.fits.images.obspar import ApcorImages
            #     ApcorImages(file_paths=paths, setup=self.setup).qc_plot_apc()

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

        # return all aperture correction images
        return self.get_aperture_correction(diameter=None)

    # =========================================================================== #
    # Coordinates
    # =========================================================================== #
    _ra_all = None

    def ra_all(self, key_ra=None):
        """ Read all RAs from files. """

        # Return if already read
        if self._ra_all is not None:
            return self._ra_all

        # Set key
        if key_ra is None:
            key_ra = self._key_ra

        self._ra_all = np.hstack(flat_list(self.get_columns(column_name=key_ra)))
        return self._ra_all

    _dec_all = None

    def dec_all(self, key_dec=None):
        """ Read all DECs from files. """

        # Return if already read
        if self._dec_all is not None:
            return self._dec_all

        # Set key
        if key_dec is None:
            key_dec = self._key_dec

        self._dec_all = np.hstack(flat_list(self.get_columns(column_name=key_dec)))
        return self._dec_all

    def centroid_total(self, key_ra=None, key_dec=None):
        """ Return centroid positions for all files in instance together. """

        # Set keys
        if key_ra is None:
            key_ra = self._key_ra
        if key_dec is None:
            key_dec = self._key_dec

        # Return centroid
        return centroid_sphere(lon=self.ra_all(key_ra=key_ra), lat=self.dec_all(key_dec=key_dec), units="degree")

    # =========================================================================== #
    # Names
    # =========================================================================== #
    @property
    def _colname_mag_apc(self):
        """ Constructor for column names. """
        return "MAG_APC"

    @property
    def _colname_mag_cal(self):
        """ Constructor for column names. """
        return "MAG_CAL"

    @property
    def _colname_mag_err(self):
        """ Constructor for column names. """
        return "MAGERR_CAL"

    # =========================================================================== #
    # Astrometry
    # =========================================================================== #
    def calibrate_astrometry(self):

        # Processing info
        tstart = message_mastercalibration(master_type="ASTROMETRY", right=None, left=None,
                                           silent=self.setup["misc"]["silent"])

        # Check how many ahead files are there
        n_ahead = np.sum([os.path.isfile(p) for p in self._scamp_header_paths()])

        # If there are already all files, do not do anything
        if n_ahead == len(self):
            print(BColors.WARNING + "Astrometry already done." + BColors.ENDC)

        # Otherwise Run scamp
        else:

            # Run
            self.scamp()

            # Dummy check
            if len(self._scamp_header_paths()) != len(self):
                raise ValueError("Something wrong with Scamp")

        # Write coadd header if not there yet
        if not os.path.exists(self.path_coadd_header):
            self.coadd_header()

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    def coadd_header(self):

        # Try to read coadd header from disk
        try:
            header_coadd = fits.Header.fromtextfile(self.path_coadd_header)

        except FileNotFoundError:

            # Determine pixel scale
            pixel_scale = fraction2float(self.setup["astromatic"]["pixel_scale"]) / 3600.

            # Now loop over each output scamp header
            ra, dec = [], []
            for fidx in range(len(self)):

                # Convert to astropy Headers
                scamp_headers = read_scamp_header(path=self._scamp_header_paths()[fidx], remove_pv=True)

                naxis1, naxis2 = self.imageheaders_get_keys(keywords=["NAXIS1", "NAXIS2"], file_index=fidx)
                naxis1, naxis2 = naxis1[0], naxis2[0]

                # Mody each header with NAXIS and extract ra,dec of footprint
                for shdr, n1, n2 in zip(scamp_headers, naxis1, naxis2):

                    # Update with NAXIS
                    shdr.update(NAXIS=2, NAXIS1=n1, NAXIS2=n2)

                    # Convert to WCS
                    r, d = awcs.WCS(shdr).calc_footprint().T

                    ra += r.tolist()
                    dec += d.tolist()

            # Construct skycoord
            sc = SkyCoord(ra=ra, dec=dec, frame="icrs", unit="deg")

            # Get optimal rotation of frame
            rotation_test = np.linspace(0, 2 * np.pi, 360)
            area = []
            for rot in rotation_test:
                hdr = skycoord2header(skycoord=sc, proj_code="ZEA", rotation=rot, enlarge=1.02, cdelt=pixel_scale)
                area.append(hdr["NAXIS1"] * hdr["NAXIS2"])

            # Return final header with optimized rotation
            rotation = rotation_test[np.argmin(area)]
            header_coadd = skycoord2header(skycoord=sc, proj_code="ZEA", rotation=rotation,
                                           enlarge=1.02, cdelt=pixel_scale)

            # Dummy check
            if (header_coadd["NAXIS1"] > 100000.) or (header_coadd["NAXIS2"] > 100000.):
                raise ValueError("Double check if the image size is correcti")

            # Write coadd header to disk
            write_header(header=header_coadd, path=self.path_coadd_header)

        return header_coadd

    # =========================================================================== #
    # Superflat
    # =========================================================================== #
    def build_master_superflat(self):
        """ Superflat construction method. """

        # Import
        from vircampype.fits.images.flat import MasterSuperflat

        # Processing info
        tstart = message_mastercalibration(master_type="MASTER-SUPERFLAT", silent=self.setup["misc"]["silent"])

        # Split based on filter and interval
        split = self.split_values(values=self.filter)
        split = flat_list([s.split_window(window=self.setup["superflat"]["window"], remove_duplicates=True)
                           for s in split])

        # Remove too short entries
        split = prune_list(split, n_min=self.setup["superflat"]["n_min"])

        # Get master photometry catalog
        master_phot = self.get_master_photometry()

        # Now loop through separated files
        for files, idx_print in zip(split, range(1, len(split) + 1)):

            # Create master dark name
            outpath = self.path_master_object + "MASTER-SUPERFLAT_{0:11.5f}.fits".format(files.mjd_mean)

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]):
                continue

            # Fetch filter of current catalog
            filter_catalog = files.filter[0]

            # Filter master catalog for good data
            mkeep_qfl = [True if x in "AB" else False for x in master_phot.qflags(key=filter_catalog)[0][0]]
            mkeep_cfl = [True if x == "0" else False for x in master_phot.cflags(key=filter_catalog)[0][0]]

            # Combine quality and contamination flag
            mkeep = mkeep_qfl and mkeep_cfl

            # Fetch magnitude and coordinates for master catalog
            master_mag = master_phot.mag(band=master_phot.translate_filter(key=filter_catalog))[0][0][mkeep]
            master_skycoord = master_phot.skycoord()[0][0][mkeep]

            data_headers, flx_scale, flx_scale_global, n_sources = [], [], [], []
            for idx_hdu, idx_hdr in zip(files.data_hdu[0], range(len(files.data_hdu[0]))):

                # Print processing info
                message_calibration(n_current=idx_print, n_total=len(split), name=outpath, d_current=idx_hdr + 1,
                                    d_total=len(files.data_hdu[0]), silent=self.setup["misc"]["silent"])

                # Read current HDU for all files into a single table
                tab = vstack(files.hdu2table(hdu_index=idx_hdu))

                # Read header of current extension in first file
                header = files.image_headers[0][idx_hdr]

                # Clean table
                tab = clean_source_table(table=tab, image_header=header, flux_max=header["SEXSATLV"] / 2)

                # Get ZP for each single star
                zp = get_zeropoint_radec(ra_cal=tab[self._key_ra], dec_cal=tab[self._key_dec],
                                         mag_cal=tab["MAG_AUTO"].data, ra_ref=master_skycoord.icrs.ra.deg,
                                         dec_ref=master_skycoord.icrs.dec.deg, mag_ref=master_mag,
                                         mag_limits_ref=master_phot.mag_lim, method="all")

                # Remove all table entries without ZP entry
                tab, zp = tab[np.isfinite(zp)], zp[np.isfinite(zp)]

                # Grid values to detector size array
                grid_zp = grid_value_2d(x=tab["XWIN_IMAGE"], y=tab["YWIN_IMAGE"], value=zp, x_min=0, y_min=0,
                                        weights=1/tab["MAGERR_AUTO"]**2, x_max=header["NAXIS1"], y_max=header["NAXIS2"],
                                        nx=self.setup["superflat"]["nbins_x"], ny=self.setup["superflat"]["nbins_y"],
                                        conv=True, kernel_size=2, upscale=True)

                # Convert to flux scale
                flx_scale.append(10**(grid_zp / 2.5))

                # # Plot sources on top of superflat
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(nrows=1, ncols=1, gridspec_kw=None, **dict(figsize=(7, 4)))
                #
                # im = ax.imshow(flx_scale[-1] / np.nanmedian(flx_scale[-1]), cmap="RdYlBu_r", vmin=0.95, vmax=1.04,
                #                origin="lower", extent=[0, header["NAXIS1"], 0, header["NAXIS2"]])
                # flux = 10**(zp / 2.5)
                # ax.scatter(tab["XWIN_IMAGE"], tab["YWIN_IMAGE"], c=flux / np.nanmedian(flux), s=5, lw=0.5,
                #            cmap="RdYlBu_r", ec="black", vmin=0.95, vmax=1.04)
                # plt.colorbar(im)
                # plt.show()
                # exit()

                # Save global scale
                flx_scale_global.append(np.nanmedian(flx_scale[-1]))

                # Save number of sources
                n_sources.append(np.sum(np.isfinite(zp)))

            # Get global flux scale across detectors
            flx_scale_global /= np.median(flx_scale_global)

            # Compute flux scale for each detector
            flx_scale = [f / np.median(f) * g for f, g in zip(flx_scale, flx_scale_global)]

            # Instantiate output
            superflat = ImageCube(setup=self.setup)

            # Loop over extensions and construct final superflat
            for idx_hdu, fscl, nn in zip(files.data_hdu[0], flx_scale, n_sources):

                # Append to output
                superflat.extend(data=fscl.astype(np.float32))

                # Create extension header cards
                data_cards = make_cards(keywords=["HIERARCH PYPE SFLAT NSOURCES", "HIERARCH PYPE SFLAT STD"],
                                        values=[nn, float(str(np.round(np.nanstd(fscl), decimals=2)))],
                                        comments=["Number of sources used", "Standard deviation in relative flux"])

                # Append header
                data_headers.append(fits.Header(cards=data_cards))

            # Make primary header
            prime_cards = make_cards(keywords=[self.setup["keywords"]["object"], self.setup["keywords"]["date_mjd"],
                                               self.setup["keywords"]["filter"], self.setup["keywords"]["date_ut"],
                                               "HIERARCH PYPE N_FILES"],
                                     values=["MASTER-SUPERFLAT", files.mjd_mean,
                                             files.filter[0], files.time_obs_mean,
                                             len(files)])
            prime_header = fits.Header(cards=prime_cards)

            # Write to disk
            superflat.write_mef(path=outpath, prime_header=prime_header, data_headers=data_headers)

            # QC plot
            if self.setup["misc"]["qc_plots"]:
                msf = MasterSuperflat(setup=self.setup, file_paths=outpath)
                msf.qc_plot_superflat(paths=None, axis_size=5)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    # =========================================================================== #
    # Zero points
    # =========================================================================== #
    @property
    def _zp_keys(self):
        return ["HIERARCH PYPE MAGZP {0}".format(i + 1) for i in range(len(self.apertures))]

    @property
    def _zp_comments(self):
        return ["ZP for {0} pix aperture".format(d) for d in self.apertures]

    @property
    def _zperr_keys(self):
        return ["HIERARCH PYPE MAGZPERR {0}".format(i + 1) for i in range(len(self.apertures))]

    @property
    def _zperr_comments(self):
        return ["ZP error for {0} pix aperture".format(d) for d in self.apertures]

    @property
    def _zp_auto_key(self):
        return "HIERARCH PYPE MAGZP AUTO"

    @property
    def _zp_auto_comments(self):
        return "ZP for AUTO aperture"

    @property
    def _zperr_auto_key(self):
        return "HIERARCH PYPE MAGZPERR AUTO"

    @property
    def _zperr_auto_comment(self):
        return "ZP error for AUTO aperture"

    @property
    def _zp_psf_key(self):
        return "HIERARCH PYPE MAGZP PSF"

    @property
    def _zp_psf_comments(self):
        return "ZP for PSF mag"

    @property
    def _zperr_psf_key(self):
        return "HIERARCH PYPE MAGZPERR PSF"

    @property
    def _zperr_psf_comment(self):
        return "ZP error for PSF mag"

    @property
    def _zp_avg_key(self):
        return "HIERARCH PYPE MAGZP AVG"

    @property
    def _zp_avg_comment(self):
        return "Average ZP across apertures"

    @property
    def _zpstd_avg_key(self):
        return "HIERARCH PYPE MAGZPSTD AVG"

    @property
    def _zpstd_avg_comment(self):
        return "Sigma ZP across apertures"

    def plot_qc_astrometry(self, axis_size=5, key_x="XWIN_IMAGE", key_y="YWIN_IMAGE", key_ra=None, key_dec=None):

        # Import
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Processing info
        tstart = message_mastercalibration(master_type="QC ASTROMETRY", right=None, silent=self.setup["misc"]["silent"])

        # Get FPA layout
        fpa_layout = str2list(self.setup["data"]["fpa_layout"], dtype=int)

        # Obtain master coordinates
        sc_master_astrometry = self.get_master_photometry().skycoord()[0][0]

        # Loop over files
        for idx_file in range(len(self)):

            # Generate outpath
            outpath = "{0}{1}_astrometry.pdf".format(self.path_qc_astrometry, self.file_names[idx_file])

            # Check if file exists
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]):
                continue

            # Grab coordinates
            xx_file = self.get_column_file(idx_file=idx_file, column_name=key_x)
            yy_file = self.get_column_file(idx_file=idx_file, column_name=key_y)
            sc_file = self.skycoord_file(idx_file=idx_file, key_ra=key_ra, key_dec=key_dec)

            # Coadd mode
            if len(self) == 1:
                fig, ax_all = get_plotgrid(layout=(1, 1), xsize=2*axis_size, ysize=2*axis_size)
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
                                    d_total=len(sc_file), silent=self.setup["misc"]["silent"])

                # Read header
                header = self.image_headers[idx_file][idx_hdu]

                # Get separations between master and current table
                i1, sep, _ = sc_file[idx_hdu].match_to_catalog_sky(sc_master_astrometry)

                # Extract position angles between master catalog and input
                # sc1 = sc_master_astrometry[i1]
                # ang = sc1.position_angle(sc_hdu)

                # Keep only those with a maximum of 0.5 arcsec
                keep = sep.arcsec < 0.5
                sep, x_hdu, y_hdu = sep[keep], xx_file[idx_hdu][keep], yy_file[idx_hdu][keep]

                # Determine number of bins (with given radius at least 10 sources)
                stacked = np.stack([x_hdu, y_hdu]).T
                dis, _ = NearestNeighbors(n_neighbors=51, algorithm="auto").fit(stacked).kneighbors(stacked)
                maxdis = np.percentile(dis[:, -1], 95)
                n_bins_x, n_bins_y = int(header["NAXIS1"] / maxdis), int(header["NAXIS2"] / maxdis)

                # Minimum number of 3 bins
                n_bins_x = 3 if n_bins_x <= 3 else n_bins_x
                n_bins_y = 3 if n_bins_y <= 3 else n_bins_y

                # Grid value into image
                grid = grid_value_2d(x=x_hdu, y=y_hdu, value=sep.arcsec, x_min=0, x_max=header["NAXIS1"], y_min=0,
                                     y_max=header["NAXIS2"], nx=n_bins_x, ny=n_bins_y, conv=False, upscale=False)

                # Append separations in arcsec
                sep_all.append(sep.arcsec)

                # Draw
                kwargs = {"vmin": 0, "vmax": 0.5, "cmap": "Spectral_r"}
                extent = [0, header["NAXIS1"], 0, header["NAXIS2"]]
                im = ax_all[idx_hdu].imshow(grid, extent=extent, origin="lower", **kwargs)
                ax_all[idx_hdu].scatter(x_hdu, y_hdu, c=sep.arcsec, s=7, lw=0.5, ec="black", **kwargs)

                # Annotate detector ID
                ax_all[idx_hdu].annotate("Det.ID: {0:0d}".format(idx_hdu + 1), xy=(0.02, 1.01),
                                         xycoords="axes fraction", ha="left", va="bottom")

                # Modify axes
                if idx_hdu < fpa_layout[1]:
                    ax_all[idx_hdu].set_xlabel("X (pix)")
                else:
                    ax_all[idx_hdu].axes.xaxis.set_ticklabels([])
                if idx_hdu % fpa_layout[0] == 0:
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
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal", label="Average separation (arcsec)")
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
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    # =========================================================================== #
    # Properties
    # =========================================================================== #
    @property
    def data_hdu(self):
        """
        Overrides the normal table data_hdu property because Sextractor is special...

        Returns
        -------
        iterable
            List of iterators for header indices of HDUs which hold data.
        """
        return [range(2, len(hdrs), 2) for hdrs in self.headers]

    @property
    def image_hdu(self):
        """
        Contains iterators to obtain Image headers saved in the sextractor tables.

        Returns
        -------
        iterable
            List of iterators for each file

        """
        return [range(1, len(hdrs), 2) for hdrs in self.headers]

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
                with open(self._image_header_paths[idx], "rb") as f:

                    # If the file is there, load the headers...
                    self._image_headers.append(pickle.load(f))

                    # And continue with next file
                    continue

            # If not found we move on to read the headers from the fits file
            except FileNotFoundError:

                headers = sextractor2imagehdr(path=self.full_paths[idx])

                # When done for all headers dump them into the designated database
                with open(self._image_header_paths[idx], "wb") as d:
                    pickle.dump(headers, d)

                # Append converted headers
                self._image_headers.append(headers)

        # Return
        return self._image_headers

    def imageheaders_get_keys(self, keywords, file_index=None):
        """
        Method to return a list with lists for the individual values of the supplied keys from the data headers

        Parameters
        ----------
        keywords : list[str]
            List of FITS header keys in the primary header
        file_index : int, optional
            If set, only retrieve values from given file.

        Returns
        -------
        iterable
            Triple stacked list: List which contains a list of all keywords which in turn contain the values from all
            data headers

        Raises
        ------
        TypeError
            When the supplied keywords are not in a list.

        """

        if not isinstance(keywords, list):
            raise TypeError("Keywords must be in a list!")

        if file_index is None:
            headers_images = self.image_headers[:]
        else:
            headers_images = [self.image_headers[file_index]]

        # Return values
        return [[[e[k] for e in h] for h in headers_images] for k in keywords]

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

        self._time_obs = Time([hdr[0][self.setup["keywords"]["date_mjd"]]
                               for hdr in self.image_headers], scale="utc", format="mjd")
        return self._time_obs


class AstrometricCalibratedSextractorCatalogs(SextractorCatalogs):

    def __init__(self, setup, file_paths=None):
        super(AstrometricCalibratedSextractorCatalogs, self).__init__(file_paths=file_paths, setup=setup)

    # =========================================================================== #
    # Photometry
    # =========================================================================== #
    def build_master_photometry(self):

        # Import
        from vircampype.fits.tables.sources import MasterPhotometry2Mass, MasterPhotometry

        # Processing info
        tstart = message_mastercalibration(master_type="MASTER-PHOTOMETRY", right=None,
                                           silent=self.setup["misc"]["silent"])

        # Construct outpath
        outpath = self.path_master_object + "MASTER-PHOTOMETRY.fits.tab"

        # Check if the file is already there and skip if it is
        if not check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]):

            # Print processing info
            message_calibration(n_current=1, n_total=1, name=outpath, d_current=None,
                                d_total=None, silent=self.setup["misc"]["silent"])

            # Obtain field size
            size = np.max(distance_sky(lon1=self.centroid_total()[0], lat1=self.centroid_total()[1],
                                       lon2=self.ra_all(), lat2=self.dec_all(), unit="deg")) * 1.01

            # Download catalog
            if self.setup["photometry"]["reference"] == "2mass":
                table = download_2mass(lon=self.centroid_total()[0], lat=self.centroid_total()[1], radius=2 * size)

            else:
                raise ValueError("Catalog '{0}' not supported".format(self.setup["photometry"]["reference"]))

            # Save catalog
            table.write(outpath, format="fits", overwrite=True)

            # Add object info to primary header
            add_key_primaryhdu(path=outpath, key=self.setup["keywords"]["object"], value="MASTER-PHOTOMETRY")

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

        # Return photometry catalog
        if self.setup["photometry"]["reference"] == "2mass":
            return MasterPhotometry2Mass(setup=self.setup, file_paths=[outpath])
        else:
            return MasterPhotometry(setup=self.setup, file_paths=[outpath])

    def calibrate_photometry(self):

        # Processing info
        tstart = message_mastercalibration(master_type="PHOTOMETRY", silent=self.setup["misc"]["silent"])

        # Extract apertures
        apertures = str2list(self.setup["photometry"]["apertures"], sep=",", dtype=float)

        # Get aperture correction imaages for all aperture diameters
        apc_images = [self.get_aperture_correction(diameter=a) for a in apertures]

        # Get master photometry catalog
        master_phot = self.get_master_photometry()

        # Loop over each file
        outpaths = []
        for idx_file in range(len(self)):

            # Make outpath
            outpaths.append(self.full_paths[idx_file].replace(".sources.fits", ".sources.cal.fits"))

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpaths[-1], silent=self.setup["misc"]["silent"]):
                continue

            # Current passband
            passband = self.filter[idx_file][0]

            # Construct cleaned master photometry
            mkeep_qfl = [True if x in "AB" else False for x in master_phot.qflags(key=passband)[0][0]]
            mkeep_cfl = [True if x == "0" else False for x in master_phot.cflags(key=passband)[0][0]]

            # Combine quality and contamination flag
            mkeep = mkeep_qfl and mkeep_cfl

            # Fetch magnitude and coordinates for master catalog
            master_mag = master_phot.mag(band=master_phot.translate_filter(key=passband))[0][0][mkeep]
            master_magerr = master_phot.mag_err(band=master_phot.translate_filter(key=passband))[0][0][mkeep]
            master_skycoord = master_phot.skycoord()[0][0][mkeep]

            # Get aperture correction subset for current files
            apcs_file = [apc[idx_file] for apc in apc_images]

            # Read source catalog into
            tab_file = self.file2table(file_index=idx_file)

            # Make list to store output tables and ZPs
            tab_out, zp_out, zperr_out, zp_auto, zperr_auto = [], [], [], None, None

            for idx, idx_apc_hdu, hdr in \
                    zip(range(len(self.data_hdu[idx_file])), apcs_file[0].data_hdu[0], self.image_headers[idx_file]):

                # Print info
                message_calibration(n_current=idx_file + 1, n_total=len(self), name=self.file_names[idx_file],
                                    d_current=idx + 1, d_total=len(self.data_hdu[idx_file]))

                # Get table for current HDU
                tab_hdu = tab_file[idx]

                # Get source coordinates
                key_ra, key_dec = "ALPHAWIN_J2000", "DELTAWIN_J2000"
                skycoord_hdu = skycoord_from_tab(tab=tab_hdu, key_ra=key_ra, key_dec=key_dec)

                # Extract aperture corrections
                mag_aper_cor = np.array([apc.get_apcor(skycoo=skycoord_hdu, file_index=0, hdu_index=idx_apc_hdu)
                                         for apc in apcs_file]).T

                # Extract magnitudes and errors
                mag_aper, magerr_aper = tab_hdu["MAG_APER"], tab_hdu["MAGERR_APER"]

                # Get subset of good sources for ZP
                good = clean_source_table(table=tab_hdu, image_header=hdr, return_filter=True,
                                          flux_max=hdr["SEXSATLV"] / 2)

                # Get ZP for each aperture
                zp_aper = [get_zeropoint(skycoo_cal=skycoord_hdu[good], mag_cal=m + apc, mag_err_cal=e,
                                         mag_limits_ref=master_phot.mag_lim, skycoo_ref=master_skycoord,
                                         mag_ref=master_mag, mag_err_ref=master_magerr, method="weighted")
                           for m, apc, e in zip(mag_aper[good, :].T, mag_aper_cor[good, :].T, magerr_aper[good, :].T)]

                # Unpack results and convert to arrays
                zp_aper, zperr_aper = list(zip(*zp_aper))
                zp_aper, zperr_aper = np.array(zp_aper), np.array(zperr_aper)

                # Get ZP for MAG_AUTO
                zp_auto, zperr_auto = get_zeropoint(skycoo_cal=skycoord_hdu[good], mag_cal=tab_hdu["MAG_AUTO"][good],
                                                    mag_err_cal=tab_hdu["MAGERR_AUTO"][good],
                                                    mag_limits_ref=master_phot.mag_lim, skycoo_ref=master_skycoord,
                                                    mag_ref=master_mag, mag_err_ref=master_magerr, method="weighted")

                # ZP for MAG_PSF
                zp_psf, zperr_psf = get_zeropoint(skycoo_cal=skycoord_hdu[good], mag_cal=tab_hdu["MAG_PSF"][good],
                                                  mag_err_cal=tab_hdu["MAGERR_PSF"][good],
                                                  mag_limits_ref=master_phot.mag_lim, skycoo_ref=master_skycoord,
                                                  mag_ref=master_mag, mag_err_ref=master_magerr, method="weighted")

                # Add ZPs and aperture corrections
                mag_aper_cal = mag_aper + mag_aper_cor + zp_aper
                mag_auto_cal = tab_hdu["MAG_AUTO"] + zp_auto
                mag_psf_cal = tab_hdu["MAG_PSF"] + zp_psf

                # Mask bad photometry
                bad_aper = (mag_aper_cal > 50) | (magerr_aper > 50)
                mag_aper_cal[bad_aper], magerr_aper[bad_aper], mag_aper_cor[bad_aper] = np.nan, np.nan, np.nan
                mag_auto_cal[mag_auto_cal > 50], mag_psf_cal[mag_psf_cal > 50] = np.nan, np.nan

                # Make new columns
                col_mag = Column(name=self._colname_mag_cal, data=mag_aper_cal, **kwargs_column_mag)
                col_err = Column(name=self._colname_mag_err, data=magerr_aper, **kwargs_column_mag)
                col_apc = Column(name=self._colname_mag_apc, data=mag_aper_cor, **kwargs_column_mag)
                col_mag_auto = Column(name="MAG_AUTO_CAL", data=mag_auto_cal, **kwargs_column_mag)
                col_mag_psf = Column(name="MAG_PSF_CAL", data=mag_psf_cal, **kwargs_column_mag)

                # Append to table
                tab_hdu.add_columns(cols=[col_mag, col_err, col_apc, col_mag_auto, col_mag_psf])

                # Save data
                tab_out.append(tab_hdu)
                zp_out.append(zp_aper)
                zperr_out.append(zperr_aper)

            # Construct new output file
            with fits.open(self.full_paths[idx_file]) as cat:

                # Loop over table extensions
                for tidx, tab, zp, zperr in zip(self.data_hdu[idx_file], tab_out, zp_out, zperr_out):

                    # Read header
                    hdr = cat[tidx].header

                    # Add keywords to header
                    for aidx in range(len(zp)):
                        add_float_to_header(hdr, self._zp_keys[aidx], zp[aidx], self._zp_comments[aidx])
                        add_float_to_header(hdr, self._zperr_keys[aidx], zperr[aidx], self._zperr_comments[aidx])
                    add_float_to_header(hdr, self._zp_avg_key, float(np.mean(zp)), self._zp_avg_comment)
                    add_float_to_header(hdr, self._zpstd_avg_key, float(np.std(zp)), self._zpstd_avg_comment)
                    add_float_to_header(hdr, self._zp_auto_key, zp_auto, self._zp_auto_comments)
                    add_float_to_header(hdr, self._zperr_auto_key, zperr_auto, self._zperr_auto_comment)

                    # Create new output table
                    cat[tidx] = fits.BinTableHDU(data=tab, header=hdr)

                # Write to disk
                cat.writeto(outpaths[-1], overwrite=self.setup["misc"]["overwrite"])

            # QC plot
            if self.setup["misc"]["qc_plots"]:
                csc = PhotometricCalibratedSextractorCatalogs(setup=self.setup, file_paths=outpaths[-1])
                csc.plot_qc_zp(axis_size=5)
                csc.plot_qc_ref(axis_size=5)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

        # Return new catalog instance
        return PhotometricCalibratedSextractorCatalogs(setup=self.setup, file_paths=outpaths)


class PhotometricCalibratedSextractorCatalogs(AstrometricCalibratedSextractorCatalogs):

    def __init__(self, setup, file_paths=None):
        super(PhotometricCalibratedSextractorCatalogs, self).__init__(file_paths=file_paths, setup=setup)

    @property
    def flux_scale(self):
        """
        Constructs flux scale from different zero points across all images and detectors

        Returns
        -------
        iterable
            List of lists for flux scaling
        """

        # Convert ZPs to dummy flux
        df = 10**(np.array(self.dataheaders_get_keys(keywords=[self._zp_avg_key])[0]) / -2.5)

        # Scale to mean flux (not to mean magnitude)
        return (df / np.mean(df)).tolist()

    @property
    def flux_scale_default(self):
        """
        Returns flux scaling of 1.0 for each image and each extension.

        Returns
        -------
        iterable
            List of lists for default flux scaling (1.0)

        """
        return (np.array(self.flux_scale) / np.array(self.flux_scale)).tolist()

    # =========================================================================== #
    # External headers
    # =========================================================================== #
    def write_coadd_headers(self):

        # Processing info
        tstart = message_mastercalibration(master_type="EXTERNAL COADD HEADERS", silent=self.setup["misc"]["silent"],
                                           right=None)

        # Loop over files
        for idx_file in range(len(self)):

            # Get current ahead path
            path_ahead = self.full_paths[idx_file].replace(".sources.cal.fits", ".ahead")

            if check_file_exists(file_path=path_ahead, silent=self.setup["misc"]["silent"]):
                continue

            # Print info
            message_calibration(n_current=idx_file + 1, n_total=len(self),
                                name=path_ahead, d_current=None, d_total=None)

            # Create empty string
            text = f""

            # Loop over HDUs
            for idx_hdu in range(len(self.data_hdu[idx_file])):

                # Create empty Header
                header = fits.Header()

                # Put flux scale
                header["FLXSCALE"] = self.flux_scale[idx_file][idx_hdu]

                # Get string and append
                text += header.tostring(padding=False, endcard=False) + "\nEND     \n"

            # Dump into file
            with open(path_ahead, "w") as file:
                print(text, file=file)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    # =========================================================================== #
    # QC
    # =========================================================================== #
    def paths_qc_plots(self, prefix, paths=None):
        """ Constructs paths for ZP QC plots. """
        if paths is None:
            return ["{0}{1}.{2}.pdf".format(self.path_qc_photometry, fp, prefix) for fp in self.file_names]
        else:
            return paths

    def plot_qc_zp(self, paths=None, axis_size=5, overwrite=False):
        """ Generates ZP QC plot. """

        # Generate path for plots
        paths = self.paths_qc_plots(paths=paths, prefix="zp")

        zp_mean = self.dataheaders_get_keys(keywords=[self._zp_avg_key])[0]
        zperr_mean = self.dataheaders_get_keys(keywords=[self._zpstd_avg_key])[0]

        # Loop over files and create plots
        for zp, zpstd, path in zip(zp_mean, zperr_mean, paths):
            plot_value_detector(values=zp, errors=zpstd, path=path, ylabel="ZP (mag)", axis_size=axis_size,
                                overwrite=overwrite, yrange=(np.median(zp) - 0.05, np.median(zp) + 0.05))

    def plot_qc_ref(self, axis_size=5):

        # Import
        from astropy.units import Unit
        import matplotlib.pyplot as plt
        # from matplotlib.cm import get_cmap
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Aperture index to use for plotting
        aper_idx = 1

        # Generate output paths
        outpaths_1d = self.paths_qc_plots(paths=None, prefix="phot.1D")
        # outpaths_2d = self.paths_qc_plots(paths=None, prefix="phot.2D")

        for idx_file in range(len(self)):

            # if check_file_exists(file_path=outpath_2d, silent=self.setup["misc"]["silent"]):
            #     continue

            # Read master photometry table
            master_phot = self.get_master_photometry()[idx_file]

            # Get passband
            passband = self.filter[idx_file][0]

            # Construct cleaned master photometry
            mkeep_qfl = [True if x in "AB" else False for x in master_phot.qflags(key=passband)[idx_file][0]]
            mkeep_cfl = [True if x == "0" else False for x in master_phot.cflags(key=passband)[idx_file][0]]

            # Combine quality and contamination flag
            mkeep = mkeep_qfl and mkeep_cfl

            # Fetch magnitude and coordinates for master catalog
            mag_master = master_phot.mag(band=master_phot.translate_filter(key=passband))[idx_file][0][mkeep]
            master_skycoord = master_phot.skycoord()[idx_file][0][mkeep]

            # Plot layout
            fpa_layout = str2list(self.setup["data"]["fpa_layout"], dtype=int)

            # Read tables
            tab_file = self.file2table(file_index=idx_file)

            # Clean tables
            tab_file = [clean_source_table(table=t) for t in tab_file]

            # =========================================================================== #
            # 1D
            # =========================================================================== #
            # Make plot grid
            if len(self.data_hdu[idx_file]) == 1:
                fig, ax_file = get_plotgrid(layout=(1, 1), xsize=2*axis_size, ysize=2*axis_size)
                ax_file = [ax_file]
            else:
                fig, ax_file = get_plotgrid(layout=fpa_layout, xsize=axis_size, ysize=axis_size / 2)
                ax_file = ax_file.ravel()

            # Loop over extensions
            for idx_hdu in range(len(self.data_hdu[idx_file])):

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
                mag_hdu_match = tab_hdu[self._colname_mag_cal][:, aper_idx][idx_final]
                mag_master_match = mag_master[idx_master]

                # Compute difference between reference and self
                mag_delta = mag_master_match - mag_hdu_match

                # Draw photometry
                dens = point_density(xdata=mag_master_match, ydata=mag_delta, xsize=0.25, ysize=0.05, norm=True,
                                     njobs=self.setup["misc"]["n_jobs"])
                sidx = np.argsort(dens)
                ax.scatter(mag_master_match[sidx], mag_delta[sidx], c=np.sqrt(dens[sidx]),
                           vmin=0, vmax=1.0, s=5, lw=0, alpha=1.0, zorder=0, cmap="magma")

                # Draw ZP
                ax.axhline(0, zorder=1, c="black", alpha=0.5)

                # Evalulate KDE
                kde = KernelDensity(kernel="gaussian", bandwidth=0.1, metric="euclidean")
                kde_grid = np.arange(np.floor(-1), np.ceil(1), 0.01)
                # dens = np.exp(kde.fit(mag_delta.reshape(-1, 1)).score_samples(kde_grid.reshape(-1, 1)))

                # KDE for ZP mag interval
                keep = (mag_master_match >= master_phot.mag_lim[0]) & \
                       (mag_master_match <= master_phot.mag_lim[1])

                # Skip if nothing remains
                if np.sum(keep) == 0:
                    continue

                # noinspection PyUnresolvedReferences
                dens_zp = np.exp(kde.fit((mag_delta[keep]).reshape(-1, 1)).score_samples(kde_grid.reshape(-1, 1)))

                # Draw KDE
                ax_kde = ax.twiny()
                ax_kde.plot(dens_zp, kde_grid, lw=1, c="black", alpha=0.8)
                ax_kde.axis("off")

                # Draw normal histogram
                # ax_file[idx_hdu].hist(mag_match - mag_final, orientation="horizontal",
                #                       histtype="step", lw=1, ec="black", bins="scott")

                # Annotate detector ID
                ax.annotate("Det.ID: {0:0d}".format(idx_hdu + 1), xy=(0.98, 0.04),
                            xycoords="axes fraction", ha="right", va="bottom")

                # Set limits
                ax.set_xlim(10, 18)
                ylim = (-1, 1)
                ax.set_ylim(ylim)
                ax_kde.set_ylim(ylim)

                # Modify axes
                if idx_hdu < fpa_layout[1]:
                    ax.set_xlabel("{0} {1} (mag)"
                                  "".format(self.setup["photometry"]["reference"].upper(), self.filter[idx_file]))
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if idx_hdu % fpa_layout[0] == fpa_layout[0] - 1:
                    ax.set_ylabel(r"$\Delta${0} (mag)".format(self.filter[idx_file]))
                else:
                    ax.axes.yaxis.set_ticklabels([])

                # Set ticks
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_locator(MaxNLocator(3))
                ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
                fig.savefig(outpaths_1d[-1], bbox_inches="tight")
            plt.close("all")

            # # =========================================================================== #
            # # 2D
            # # =========================================================================== #
            # # Coadd mode
            # if len(self.data_hdu[idx_file]) == 1:
            #     fig, ax_file = get_plotgrid(layout=(1, 1), xsize=2*axis_size, ysize=2*axis_size)
            #     ax_file = [ax_file]
            # else:
            #     fig, ax_file = get_plotgrid(layout=fpa_layout, xsize=axis_size, ysize=axis_size)
            #     ax_file = ax_file.ravel()
            # cax = fig.add_axes([0.3, 0.92, 0.4, 0.02])
            #
            # im = None
            # for idx_hdu in range(len(self.data_hdu[idx_file])):
            #
            #     # Grab axes
            #     ax = ax_file[idx_hdu]
            #
            #     # Read header
            #     header = self.image_headers[idx_file][idx_hdu]
            #
            #     # Grab current catalog
            #     tab_hdu = tab_file[idx_hdu]
            #
            #     # Construct skycoordinates
            #     sc_hdu = SkyCoord(ra=tab_hdu[self._key_ra], dec=tab_hdu[self._key_dec], unit="deg")
            #
            #     # Xmatch science with reference
            #     zp_idx, zp_d2d, _ = sc_hdu.match_to_catalog_sky(master_skycoord)
            #
            #     # Get good indices in reference catalog and in current field
            #     idx_master = zp_idx[zp_d2d < 1 * Unit("arcsec")]
            #     idx_final = np.arange(len(zp_idx))[zp_d2d < 1 * Unit("arcsec")]
            #
            #     # Apply indices filter
            #     mag_hdu_match = tab_hdu[self._colname_mag_cal][:, aper_idx][idx_final]
            #     magerr_hdu_match = tab_hdu[self._colname_mag_err][:, aper_idx][idx_final]
            #     mag_master_match = mag_master[idx_master]
            #     magerr_master_match = magerr_master[idx_master]
            #     mag_delta = mag_master_match - mag_hdu_match
            #
            #     # Also get X/Y coordinates
            #     x_hdu, y_hdu = tab_hdu["X_IMAGE"][idx_final], tab_hdu["Y_IMAGE"][idx_final]
            #
            #     # Grid value into image
            #     mode = "pawprint"
            #     if mode == "pawprint":
            #         nx, ny, kernel_scale = 3, 3, 0.2
            #     elif mode == "tile":
            #         nx, ny, kernel_scale = 10, 10, 0.05
            #     else:
            #         raise ValueError("Mode '{0}' not supported".format(mode))
            #
            #     grid = grid_value_2d(x=x_hdu, y=y_hdu, value=mag_delta, x_min=0, x_max=header["NAXIS1"],
            #                          y_min=0, y_max=header["NAXIS2"], nx=nx, ny=ny, conv=False, upscale=False,
            #                          weights=1 / (magerr_hdu_match + magerr_master_match))
            #
            #     # Draw
            #     kwargs = {"vmin": -0.2, "vmax": +0.2, "cmap": get_cmap("RdBu", 26)}
            #     extent = [1, header["NAXIS1"], 1, header["NAXIS2"]]
            #     im = ax.imshow(grid, extent=extent, origin="lower", **kwargs)
            #     ax.scatter(x_hdu, y_hdu, c=mag_delta, s=7, lw=0.5, ec="black", **kwargs)
            #
            #     # Annotate detector ID
            #     ax.annotate("Det.ID: {0:0d}".format(idx_hdu + 1), xy=(0.02, 1.01),
            #                 xycoords="axes fraction", ha="left", va="bottom")
            #
            #     # Modify axes
            #     if idx_hdu < fpa_layout[1]:
            #         ax_file[idx_hdu].set_xlabel("X (pix)")
            #     else:
            #         ax.axes.xaxis.set_ticklabels([])
            #     if idx_hdu % fpa_layout[0] == fpa_layout[0] - 1:
            #         ax_file[idx_hdu].set_ylabel("Y (pix)")
            #     else:
            #         ax.axes.yaxis.set_ticklabels([])
            #
            #     # Equal aspect ratio
            #     ax.set_aspect("equal")
            #
            #     # Set ticks
            #     ax.xaxis.set_major_locator(MaxNLocator(5))
            #     ax.xaxis.set_minor_locator(AutoMinorLocator())
            #     ax.yaxis.set_major_locator(MaxNLocator(5))
            #     ax.yaxis.set_minor_locator(AutoMinorLocator())
            #
            #     # Set limits
            #     ax.set_xlim(1, header["NAXIS1"])
            #     ax.set_ylim(1, header["NAXIS2"])
            #
            # # Add colorbar
            # cbar = plt.colorbar(im, cax=cax, orientation="horizontal", label="Zero Point (mag)")
            # cbar.ax.xaxis.set_ticks_position("top")
            # cbar.ax.xaxis.set_label_position("top")
            #
            # # # Save plot
            # with warnings.catch_warnings():
            #     warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
            #     fig.savefig(outpaths_2d[-1], bbox_inches="tight")
            # plt.close("all")
