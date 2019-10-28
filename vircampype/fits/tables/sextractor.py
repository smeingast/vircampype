# =========================================================================== #
# Import
import warnings
import numpy as np
import multiprocessing

from astropy.io import fits
from vircampype.utils import *
from vircampype.fits.tables.sources import SourceCatalogs


class SextractorCatalogs(SourceCatalogs):

    def __init__(self, setup, file_paths=None):
        super(SextractorCatalogs, self).__init__(file_paths=file_paths, setup=setup)

    # =========================================================================== #
    # Coordinates
    # =========================================================================== #
    @property
    def _key_ra(self):
        return "ALPHA_J2000"

    @property
    def _key_dec(self):
        return "DELTA_J2000"

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

        return "vircampype.resources.astromatic.presets"

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

    def _scamp_header_names(self, joined=False):
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
        names = [x.replace(".sources.fits", ".ahead") for x in self.full_paths]
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
        bands = list(set(self.filters))
        if len(bands) != 1:
            raise ValueError("Sequence contains multiple filter")
        else:
            band = bands[0][0]  # THIS should only keep J,H, and K for 2MASS (First band and first letter)
            band = "Ks" if "k" in band.lower() else band

        # Load preset
        options = yml2config(nthreads=self.setup["misc"]["n_threads_shell"],
                             checkplot_type=self._scamp_qc_types(joined=True),
                             checkplot_name=self._scamp_qc_names(joined=True),
                             skip=["HEADER_NAME", "AHEADER_NAME", "ASTREF_BAND"],
                             path=get_resource_path(package=self._scamp_preset_package, resource="scamp.yml"))

        # Construct commands for source extraction
        cmd = "{0} {1} -c {2} -HEADER_NAME {3} -ASTREF_BAND {4} {5}" \
              "".format(self._bin_scamp, self._scamp_catalog_paths, self._scamp_default_config,
                        self._scamp_header_names(joined=True), band, options)

        # Run Scamp
        run_command_bash(cmd, silent=False)

        # Return header paths
        return self._scamp_header_names(joined=False)

    # =========================================================================== #
    # Aperture correction
    # =========================================================================== #
    def build_aperture_correction(self):

        # Processing info
        tstart = message_mastercalibration(master_type="APERTURE CORRECTION", silent=self.setup["misc"]["silent"],
                                           right=None)

        # Obtain diameters from setup and put the in list
        diameters_eval = [float(x) for x in self.setup["photometry"]["apcor_diam_eval"].split(",")]
        diameters_save = [float(x) for x in self.setup["photometry"]["apcor_diam_save"].split(",")]

        # Loop over catalogs and build aperture correction
        for idx in range(len(self)):

            # Generate output names
            path_file = "{0}{1}.apcor.fits".format(self.path_apcor, self.file_names[idx])
            path_plot = "{0}{1}.apcor.pdf".format(self.path_qc_apcor, self.file_names[idx])

            if check_file_exists(file_path=path_file.replace(".apcor.", ".apcor{0}.".format(diameters_save[0])),
                                 silent=self.setup["misc"]["silent"]):
                continue

            # Print processing info
            message_calibration(n_current=idx+1, n_total=len(self), name=path_file, d_current=None,
                                d_total=None, silent=self.setup["misc"]["silent"])

            # Read currect catalog
            tables = self.file2table(file_index=idx)

            # Get current image header
            headers = self.image_headers[idx]

            # Make output Apcor Image HDUlist
            hdulist_base = fits.HDUList(hdus=[fits.PrimaryHDU(header=self.headers_primary[idx].copy())])
            hdulist_save = [hdulist_base.copy() for _ in range(len(diameters_save))]

            # Dummy check
            if len(tables) != len(headers):
                raise ValueError("Number of tables and headers no matching")

            # Lists to save results for this image
            mag_apcor, magerr_apcor, models_apcor, nsources = [], [], [], []

            # Loop over extensions and get aperture correction after filtering
            for tab, hdr in zip(tables, headers):

                # Read magnitudes
                mag = tab["MAG_APER"]

                # Remove bad sources (class, flags, bad mags, bad mag diffs, bad fwhm, bad mag errs)
                good = (tab["CLASS_STAR"] > 0.7) & (tab["FLAGS"] == 0) & \
                       (np.sum(mag > 0, axis=1) == 0) & (np.sum(np.diff(mag, axis=1) > 0, axis=1) == 0) & \
                       (tab["FWHM_IMAGE"] > 0.8) & (tab["FWHM_IMAGE"] < 8.0) & \
                       (np.nanmean(tab["MAGERR_APER"], axis=1) < 0.1)

                # Only keep good sources
                mag = mag[good, :]

                # Obtain aperture correction from cleaned sample
                ma_apcor, me_apcor, mo_apcor = get_aperture_correction(diameters=diameters_eval, magnitudes=mag,
                                                                       func=self.setup["photometry"]["apcor_func"])

                # Obtain aperture correction values for output
                mag_apcor_save = mo_apcor(diameters_save)

                # Shrink image header
                ohdr = resize_header(header=hdr, factor=self.setup["photometry"]["apcor_image_scale"])

                # Loop over apertures and make HDUs
                for aidx in range(len(diameters_save)):

                    # Construct header to append
                    hdr_temp = ohdr.copy()
                    hdr_temp["NSRCAPC"] = (len(mag), "Number of sources used")
                    hdr_temp["APCMAG"] = (mag_apcor_save[aidx], "Aperture correction (mag)")
                    hdr_temp["APCDIAM"] = (diameters_save[aidx], "Aperture diameter (pix)")
                    hdr_temp["APCMODEL"] = (mo_apcor.name, "Aperture correction model name")
                    for i in range(len(mo_apcor.parameters)):
                        hdr_temp["APCMPAR{0}".format(i+1)] = (mo_apcor.parameters[i], "Model parameter {0}".format(i+1))

                    hdulist_save[aidx].append(hdr2imagehdu(header=hdr_temp, fill_value=mag_apcor_save[aidx],
                                                           dtype=np.float32))

                # Append to lists for QC plot
                nsources.append(len(mag))
                mag_apcor.append(ma_apcor)
                magerr_apcor.append(me_apcor)
                models_apcor.append(mo_apcor)

            # Save aperture correction as MEF
            for hdul, diams in zip(hdulist_save, diameters_save):

                # Write aperture correction diameter into primary header too
                hdul[0].header["APCDIAM"] = (diams, "Aperture diameter (pix)")
                hdul[0].header[self.setup["keywords"]["object"]] = "APERTURE-CORRECTION"

                # Save to disk
                hdul.writeto(path_file.replace(".apcor.", ".apcor{0}.".format(diams)),
                             overwrite=self.setup["misc"]["overwrite"])

            # QC plot
            if self.setup["misc"]["qc_plots"]:
                self.qc_plot_apcor(path=path_plot, diameters=diameters_eval, mag_apcor=mag_apcor,
                                   magerr_apcor=magerr_apcor, models=models_apcor, axis_size=4, nsources=nsources,
                                   overwrite=self.setup["misc"]["overwrite"])

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

        # return all aperture correction images
        return self.get_aperture_correction(diameter=None)

    def qc_plot_apcor(self, path, diameters, mag_apcor, magerr_apcor, models, axis_size=4,
                      nsources=None, overwrite=False):

        # Check if plot already exits
        if check_file_exists(file_path=path, silent=True) and not overwrite:
            return

        # Import locally
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Get plot grid
        fig, axes = get_plotgrid(layout=self.setup["instrument"]["layout"], xsize=axis_size, ysize=axis_size)
        axes = axes.ravel()

        # Helper
        ymin = np.floor(np.min(np.array(mag_apcor) - np.array(magerr_apcor))) - 1

        # Loop over detectors
        for idx in range(len(mag_apcor)):

            # Grab axex
            ax = axes[idx]

            # Plot model
            rad_model = np.linspace(0.1, 30, 2000)
            kwargs = {"lw": 0.8, "ls": "solid", "zorder": 10, "c": "black"}
            ax.plot(rad_model, models[idx](rad_model), label=models[idx].name, **kwargs)

            # Scatter plot of measurements
            ax.errorbar(diameters, mag_apcor[idx], yerr=magerr_apcor[idx],
                        fmt="none", ecolor="#08519c", capsize=3, zorder=1, lw=1.0)
            ax.scatter(diameters, mag_apcor[idx],
                       facecolor="white", edgecolor="#08519c", lw=1.0, s=25, marker="o", zorder=2)

            # Limits
            ax.set_ylim(ymin, 0.5)
            ax.set_xlim(min(diameters) - 0.1, max(diameters) + 5.0)
            # ax.legend()

            # Logscale
            ax.set_xscale("log")

            # Labels
            if idx >= len(mag_apcor) - self.setup["instrument"]["layout"][0]:
                ax.set_xlabel("Aperture diameter (pix)")
            else:
                ax.axes.xaxis.set_ticklabels([])
            if idx % self.setup["instrument"]["layout"][0] == 0:
                ax.set_ylabel("Aperture correction (mag)")
            else:
                ax.axes.yaxis.set_ticklabels([])

            # Mark 0,0
            kwargs = {"ls": "dashed", "lw": 1.0, "c": "gray", "alpha": 0.5, "zorder": 0}
            ax.axvline(x=0, **kwargs)
            ax.axhline(y=0, **kwargs)

            # Set ticks
            # ax.xaxis.set_major_locator(MaxNLocator(5))
            # ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Annotate detector ID
            ax.annotate("Det.ID: {0:0d}".format(idx + 1), xy=(0.98, 0.02), xycoords="axes fraction",
                        ha="right", va="bottom")

            # Annotate number of sources used
            if nsources is not None:
                ax.annotate("N: {0:0d}".format(nsources[idx]), xy=(0.02, 0.02), xycoords="axes fraction",
                            ha="left", va="bottom")

        # Save plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
            fig.savefig(path, bbox_inches="tight")
        plt.close("all")

    def add_aperture_correction(self):

        # Processing info
        tstart = message_mastercalibration(master_type="ADDING APETURE CORRECTION", silent=self.setup["misc"]["silent"])

        # Construct aperture corrections dict
        apc_self = [self.get_aperture_correction(diameter=diam) for diam in self._apertures_save]

        # Loop over each file
        for idx_cat_file in range(len(self)):

            # Load current hdulist
            with fits.open(self.full_paths[idx_cat_file], mode="update") as chdulist:

                # Check if aperture correction has already been added
                done = True
                for i, d in zip(self.data_hdu[idx_cat_file], self._apertures_save):
                    if not "MAG_APC_{0}".format(d) in chdulist[i].data.names:
                        done = False
                if done:
                    print(BColors.WARNING + "{0} already modified".format(self.file_names[idx_cat_file]) + BColors.ENDC)
                    continue

                # Get aperture corrections for current file
                apc_file = [apc[idx_cat_file] for apc in apc_self]

                # Get SkyCoord of current file
                skycoord_file = self.skycoord()[idx_cat_file]

                # Loop over detectors
                for idx_cat_hdu, idx_apc_hdu, skycoord_hdu in \
                        zip(self.data_hdu[idx_cat_file], apc_file[0].data_hdu[0], skycoord_file):

                    # Print info
                    message_calibration(n_current=idx_cat_file+1, n_total=len(self), name=self.file_names[idx_cat_file],
                                        d_current=idx_apc_hdu, d_total=len(self.data_hdu[idx_cat_file]))

                    # Get columns for current HDU
                    ccolumns = chdulist[idx_cat_hdu].data.columns

                    # Loop over different apertures
                    new_cols = fits.ColDefs([])
                    for apc, d in zip(apc_file, self._apertures_save):

                        # Extract aperture correction from image
                        a = apc.get_apcor(skycoo=skycoord_hdu, file_index=0, hdu_index=idx_apc_hdu)

                        # Add as new column for each source
                        new_cols.add_col(fits.Column(name="MAG_APC_{0}".format(d), format="E", array=a))

                    # Replace HDU from input catalog
                    chdulist[idx_cat_hdu] = fits.BinTableHDU.from_columns(ccolumns + new_cols)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    _mag_aper = None

    @property
    def mag_aper(self):
        """
        Reads fixed aperture magnitudes from all tables.

        Returns
        -------
        iterable
            List of lists containing MAG_APER from the Sextractor catalogs.

        """
        if self._mag_aper is not None:
            return self._mag_aper

        self._mag_aper = self.get_columns(column_name="MAG_APER")
        return self._mag_aper

    _magerr_aper = None

    @property
    def magerr_aper(self):
        """
        Reads errors for fixed aperture magnitudes from all tables.

        Returns
        -------
        iterable
            List of lists containing MAGERR_APER from the Sextractor catalogs.

        """
        if self._magerr_aper is not None:
            return self._magerr_aper

        self._magerr_aper = self.get_columns(column_name="MAGERR_APER")
        return self._magerr_aper

    _mag_apc_dict = None

    @property
    def mag_apc_dict(self):
        """
        Reads all aperture corrections from files for each extension and each source.

        Returns
        -------
        dict
            Dictionary with aperture corrections.
        """

        if self._mag_apc_dict is not None:
            return self._mag_apc_dict

        self._mag_apc_dict = {}
        for d in self._apertures_save:
            self._mag_apc_dict[d] = self.get_columns(column_name="MAG_APC_{0}".format(d))
        return self._mag_apc_dict

    # =========================================================================== #
    # Zero points
    # =========================================================================== #
    @property
    def _zp_keys(self):
        return ["HIERARCH PYPE MAGZP {0}".format(i + 1) for i in range(len(self._apertures_save))]

    @property
    def _zp_comments(self):
        return ["ZP for {0} pix aperture".format(d) for d in self._apertures_save]

    @property
    def _zperr_keys(self):
        return ["HIERARCH PYPE MAGZPERR {0}".format(i + 1) for i in range(len(self._apertures_save))]

    @property
    def _zperr_comments(self):
        return ["ZP error for {0} pix aperture".format(d) for d in self._apertures_save]

    @property
    def _zp_avg_key(self):
        return "HIERARCH PYPE MAGZP AVG"

    @property
    def _zp_avg_comment(self):
        return "Average ZP across apertures"

    @property
    def _zperr_avg_key(self):
        return "HIERARCH PYPE MAGZPERR AVG"

    @property
    def _zperr_avg_comment(self):
        return "Average ZP error across apertures"

    def delete_zeropoints(self):

        keys = self._zp_keys + self._zperr_keys + [self._zp_avg_key] + [self._zperr_avg_key]

        for idx in range(len(self)):
            for hdu in self.data_hdu[idx]:
                delete_keys_hdu(path=self.full_paths[idx], hdu=hdu, keys=keys)

            # Force reloading header
            self.delete_headers_temp(file_index=idx)

    def get_zeropoints(self):

        # Check if zeropoints have been determined already
        try:
            return self.dataheaders_get_keys(keywords=[self._zp_avg_key, self._zperr_avg_key])
        except KeyError:
            pass

        # Processing info
        tstart = message_mastercalibration(master_type="ZERO POINTS", silent=self.setup["misc"]["silent"])

        # Get master photometry catalog
        master_photometry = self.get_master_photometry()

        # Loop over catalogs
        zp_avg_catalogs, zperr_avg_catalogs = [], []
        for idx_file in range(len(self)):

            # Fetch filter of current catalog
            filter_catalog = self.filters[idx_file]

            # Filter master catalog for good data
            mkeep = [True if x in "AB" else False for x in master_photometry.qflags(key=filter_catalog)[0][0]]

            # Fetch magnitude and coordinates for master catalog
            master_mag = master_photometry.mag(key=master_photometry.translate_filter(key=filter_catalog))[0][0][mkeep]
            master_skycoord = master_photometry.skycoord()[0][0][mkeep]

            # Loop over extensions
            zp_avg, zperr_avg = [], []
            for idx_hdu, idx_file_hdu in zip(range(len(self.data_hdu[idx_file])), self.data_hdu[idx_file]):

                # Message
                message_calibration(n_current=idx_file+1, n_total=len(self), name=self.full_paths[idx_file],
                                    d_current=idx_hdu+1, d_total=len(self.data_hdu[idx_file]),
                                    silent=self.setup["misc"]["silent"])

                # Fetch magnitudes and aperture corrections
                mags = [self.mag_aper[idx_file][idx_hdu][:, idx_apc] for idx_apc in self._aperture_save_idx]
                apcs = [self.mag_apc_dict[d][idx_file][idx_hdu] for d in self._apertures_save]

                # Apply aperture correction to magnitudes
                mags = [m + a for m, a in zip(mags, apcs)]

                # Get zeropoints for each aperture
                zp_values, zperr_values = [], []
                for m in mags:
                    zp, zperr = get_zeropoint(skycoo_cal=self.skycoord()[idx_file][idx_hdu], mag_cal=m,
                                              mag_limits_ref=master_photometry.mag_lim,
                                              skycoo_ref=master_skycoord, mag_ref=master_mag)
                    zp_values.append(float(str(np.round(zp, decimals=4))))      # This forces only 4 decimals to appear
                    zperr_values.append(float(str(np.round(zperr, decimals=4))))  # in headers

                # Get averages across apertures
                zp_avg.append(float(str(np.round(np.mean(zp_values), decimals=4))))
                zperr_avg.append(float(str(np.round(np.std(zp_values), decimals=4))))

                # Add ZP data to header
                k = self._zp_keys + self._zperr_keys + [self._zp_avg_key, self._zperr_avg_key]
                v = zp_values + zperr_values + [zp_avg[-1], zperr_avg[-1]]
                c = self._zp_comments + self._zperr_comments + [self._zp_avg_comment, self._zperr_avg_comment]
                add_keys_hdu(path=self.full_paths[idx_file], hdu=idx_file_hdu, keys=k, values=v, comments=c)

                # Force reloading header after this temporary header
                self.delete_headers_temp(file_index=idx_file)

            # Append all average ZPs for current file
            zp_avg_catalogs.append(zp_avg)
            zperr_avg_catalogs.append(zperr_avg)

            # QC plot
            if self.setup["misc"]["qc_plots"]:
                path_plot = "{0}{1}.zp.pdf".format(self.path_qc_zp, self.file_names[idx_file])
                plot_value_detector(values=zp_avg, errors=zperr_avg, path=path_plot)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

        # Return ZPs
        return zp_avg_catalogs, zperr_avg_catalogs

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
        df = 10**(np.array(self.get_zeropoints()[0]) / -2.5)

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
        return (np.array(self.flux_scale()) / np.array(self.flux_scale())).tolist()

    # =========================================================================== #
    # Properties
    # =========================================================================== #
    @property
    def _apertrure_eval(self):
        """
        Constructs list of apertures from setup.

        Returns
        -------
        iterable
            List of apertures.
        """
        return str2list(s=self.setup["photometry"]["apcor_diam_eval"], sep=",", dtype=float)

    @property
    def _apertures_save(self):
        """
        Constructs list of apertures from setup.

        Returns
        -------
        iterable
            List of apertures.
        """
        return str2list(s=self.setup["photometry"]["apcor_diam_save"], sep=",", dtype=float)

    @property
    def _aperture_save_idx(self):
        return [[i for i, x in enumerate(self._apertrure_eval) if x == b][0] for b in self._apertures_save]

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

        if self._image_headers is not None:
            return self._image_headers

        # Extract Image headers
        if self.setup["misc"]["n_threads_python"] == 1:
            self._image_headers = []
            for p in self.full_paths:
                self._image_headers.append(sextractor2imagehdr(path=p))

        # Only launch a pool when requested
        elif self.setup["misc"]["n_threads_python"] > 1:
            with multiprocessing.Pool(processes=self.setup["misc"]["n_threads_python"]) as pool:
                self._image_headers = pool.starmap(sextractor2imagehdr, zip(self.full_paths))

        else:
            raise ValueError("'n_threads' not correctly set (n_threads = {0})"
                             .format(self.setup["misc"]["n_threads_python"]))

        return self._image_headers

    # =========================================================================== #
    # ESO
    # =========================================================================== #
    def make_phase3_pawprints(self, swarped):

        # Import util
        from vircampype.utils.eso import make_phase3_pawprints

        # Processing info
        tstart = message_mastercalibration(master_type="PHASE 3 PAWPRINTS", right=None,
                                           silent=self.setup["misc"]["silent"])

        # Find keywords that are only in some headers
        tl_ra, tl_dec, tl_ofa = None, None, None
        for idx_file in range(len(self)):

            # Get header
            hdr = fits.getheader(filename=swarped.full_paths[idx_file], ext=0)

            # Try to read the keywords
            try:
                tl_ra = hdr["ESO OCS SADT TILE RA"]
                tl_dec = hdr["ESO OCS SADT TILE DEC"]
                tl_ofa = hdr["ESO OCS SADT TILE OFFANGLE"]
                break
            except KeyError:
                continue

        if (tl_ra is None) | (tl_dec is None) | (tl_ofa is None):
            raise ValueError("Could not determine all silly ESO keywords...")

        # Put in dict
        shitty_kw = {"tl_ra": tl_ra, "tl_dec": tl_dec, "tl_ofa": tl_ofa}

        # Loop over files
        outpaths = []
        for idx_file in range(len(self)):

            outpaths.append("{0}{1}_{2:>02d}.fits".format(self.path_phase3, self.name, idx_file + 1))

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpaths[-1], silent=self.setup["misc"]["silent"]) or \
                    check_file_exists(file_path=outpaths[-1].replace(".fits", ".fits.fz"),
                                      silent=self.setup["misc"]["silent"]):
                continue

            # Status message
            message_calibration(n_current=idx_file + 1, n_total=len(self), name=outpaths[-1])

            # Get paths
            path_pawprint_img = outpaths[-1]
            path_pawprint_cat = outpaths[-1].replace(".fits", ".cat.fits")
            path_pawprint_wei = outpaths[-1].replace(".fits", ".weight.fits")

            # Convert pawprint catalog and image
            make_phase3_pawprints(path_swarped=swarped.full_paths[idx_file], path_sextractor=self.full_paths[idx_file],
                                  setup=swarped.setup, additional=shitty_kw,
                                  outpaths=(path_pawprint_img, path_pawprint_cat),
                                  compressed=self.setup["compression"]["compress_phase3"])

            # There also has to be a weight map
            with fits.open(swarped.full_paths[idx_file].replace(".fits", ".weight.fits")) as weight:

                # Make empty primary header
                prhdr = fits.Header()

                # Fill primary header only with some keywords
                for key, value in weight[0].header.items():
                    if not key.startswith("ESO "):
                        prhdr[key] = value

                # Add PRODCATG before RA key
                prhdr.insert(key="RA", card=("PRODCATG", "ANCILLARY.WEIGHTMAP"))

                # Overwrite primary header
                weight[0].header = prhdr

                # Add EXTNAME
                for eidx in range(1, len(weight)):
                    weight[eidx].header.insert(key="EQUINOX", card=("EXTNAME", "DET1.CHIP{0}".format(eidx)))

                # Save
                weight.writeto(path_pawprint_wei, overwrite=True, checksum=True)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

        # Return images
        from vircampype.fits.images.common import FitsImages
        return FitsImages(setup=self.setup, file_paths=outpaths)

    def make_phase3_tile(self, swarped, prov_images):

        # Import util
        from vircampype.utils.eso import make_phase3_tile

        # There can be only one file in the current instance
        if len(self) != len(swarped) != 1:
            raise ValueError("Only one tile allowed")

        # Processing info
        tstart = message_mastercalibration(master_type="PHASE 3 TILE", right=None,
                                           silent=self.setup["misc"]["silent"])

        # Generate outpath
        path_tile = "{0}{1}_tl.fits".format(self.path_phase3, self.name)
        path_weig = path_tile.replace(".fits", ".weight.fits")

        # Check if the file is already there and skip if it is
        if check_file_exists(file_path=path_tile, silent=self.setup["misc"]["silent"]) or \
                check_file_exists(file_path=path_tile.replace(".fits", ".fits.fz"),
                                  silent=self.setup["misc"]["silent"]):
            return

        # Convert to phase 3 compliant format
        make_phase3_tile(path_swarped=swarped.full_paths[0], path_sextractor=self.full_paths[0],
                         paths_prov=prov_images.full_paths, setup=self.setup, outpath=path_tile,
                         compressed=self.setup["compression"]["compress_phase3"])

        # There also has to be a weight map
        with fits.open(swarped.full_paths[0].replace(".fits", ".weight.fits")) as weight:

            # Add PRODCATG
            weight[0].header["PRODCATG"] = "ANCILLARY.WEIGHTMAP"

            # Save
            weight.writeto(path_weig, overwrite=False, checksum=True)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])
