# =========================================================================== #
# Import
import warnings
import subprocess
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
        options = yml2config(nthreads=self.setup["misc"]["n_threads"], checkplot_type=self._scamp_qc_types(joined=True),
                             checkplot_name=self._scamp_qc_names(joined=True),
                             skip=["HEADER_NAME", "AHEADER_NAME", "ASTREF_BAND"],
                             path=get_resource_path(package=self._scamp_preset_package, resource="scamp.yml"))

        # Construct commands for source extraction
        cmd = "{0} {1} -c {2} -HEADER_NAME {3} -ASTREF_BAND {4} {5}" \
              "".format(self._bin_scamp, self._scamp_catalog_paths, self._scamp_default_config,
                        self._scamp_header_names(joined=True), band, options)

        # Run Scamp
        # cp = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        subprocess.run(cmd, shell=True, executable="/bin/bash")

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
                       (tab["FWHM_IMAGE"] > 1.0) & (tab["FWHM_IMAGE"] < 6.0) & \
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
        # TODO: This should be moved so some sort of MASTER-APCOR class perhaps

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
            chdulist = fits.open(self.full_paths[idx_cat_file], mode="update")

            # Check if aperture correction has alrady been added
            done = True
            for i, d in zip(self.data_hdu[idx_cat_file], self._apertures_save):
                if not "MAG_APC_{0}".format(d) in chdulist[i].data.names:
                    done = False
            if done:
                print("{0} already modified".format(self.file_names[idx_cat_file]))
                continue

            # Get aperture corrections for current file
            apc_file = [apc[idx_cat_file] for apc in apc_self]

            # Get aperture magnitudes for current file
            mag_aper_file = self.mag_aper[idx_cat_file]

            # Get SkyCoord of current file
            skycoord_file = self.skycoord()[idx_cat_file]

            # Loop over detectors
            for idx_cat_hdu, idx_apc_hdu, mag_aper_hdu, skycoord_hdu in \
                    zip(self.data_hdu[idx_cat_file], apc_file[0].data_hdu[0], mag_aper_file, skycoord_file):

                # Print info
                message_calibration(n_current=idx_cat_file+1, n_total=len(self), name=self.file_names[idx_cat_file],
                                    d_current=idx_apc_hdu, d_total=len(self.data_hdu[idx_cat_file]))

                # Get columns for current HDU
                ccolumns = chdulist[idx_cat_hdu].data.columns

                # Extract given apertures
                mag_aper_hdu_save = mag_aper_hdu[:, self._aperture_save_idx]

                # Loop over different apertures
                new_cols = fits.ColDefs([])
                for apc, mag, d in zip(apc_file, mag_aper_hdu_save.T, self._apertures_save):

                    # Extract aperture correction from image
                    a = apc.get_apcor(skycoo=skycoord_hdu, file_index=0, hdu_index=idx_apc_hdu)
                    new_cols.add_col(fits.Column(name="MAG_APC_{0}".format(d), format="E", array=a))

                # Replace HDU from input catalog
                chdulist[idx_cat_hdu] = fits.BinTableHDU.from_columns(ccolumns + new_cols)

            # Overwrite catalog
            chdulist.close()

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
            List of lists containing MAG_APER from the Sextractor catalog.

        """
        if self._mag_aper is not None:
            return self._mag_aper

        self._mag_aper = self.get_columns(column_name="MAG_APER")
        return self._mag_aper

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

        with multiprocessing.Pool(processes=self.setup["misc"]["n_threads"]) as pool:
            self._image_headers = pool.starmap(sextractor2imagehdr, zip(self.full_paths))

        return self._image_headers

    # =========================================================================== #
    # ESO
    # =========================================================================== #
    def make_phase3_catalog(self, mode):

        # Processing info
        tstart = message_mastercalibration(master_type="PHASE 3 CATALOG", silent=self.setup["misc"]["silent"])

        # Loop over files
        for idx_file in range(len(self)):

            # Make outpath
            if mode == "individual":
                outpath = "{0}{1}_{2:>02d}.cat.fits".format(self.path_eso, self.name, idx_file+1)
            elif mode == "coadd":
                outpath = "{0}{1}_{2:>02d}_tl.cat.fits".format(self.path_eso, self.name, idx_file+1)
            else:
                raise ValueError("Mode '{0}' not supported.".format(mode))

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]):
                continue

            # Create empty Table HDUList
            hdulist = fits.HDUList([fits.PrimaryHDU(header=self.headers_primary[idx_file])])

            # Get skycoord for current file
            skycoord_file = self.skycoord_file(idx_file=idx_file)

            # Read data for this file
            fwhm_file = self.get_column_file(idx_file=idx_file, column_name="FWHM_WORLD")
            flags_file = self.get_column_file(idx_file=idx_file, column_name="FLAGS")
            ell_file = self.get_column_file(idx_file=idx_file, column_name="ELLIPTICITY")
            elo_file = self.get_column_file(idx_file=idx_file, column_name="ELONGATION")
            class_file = self.get_column_file(idx_file=idx_file, column_name="CLASS_STAR")

            # Loop over data HDUs
            for idx_catalog_hdu, idx_arrays in zip(self.data_hdu[idx_file], range(len(self.data_hdu[idx_file]))):

                # Print processing info
                message_calibration(n_current=idx_file+1, n_total=len(self), name=outpath, d_current=idx_arrays+1,
                                    d_total=len(self.data_hdu[idx_file]), silent=self.setup["misc"]["silent"])

                # Fetch coordinates, magnitudes, aperture corrections, and zero points
                skycoord_hdu = skycoord_file[idx_arrays]
                mag_aper_hdu = [self.mag_aper[idx_file][idx_arrays][:, idx_apc] for idx_apc in self._aperture_save_idx]
                mag_apc_hdu = [self.mag_apc_dict[d][idx_file][idx_arrays] for d in self._apertures_save]
                mag_zp = self.dataheaders_get_keys(keywords=self._zp_keys, file_index=idx_file)
                mag_zp = [m[0][0] for m in mag_zp]

                # Apply aperture correction to magnitudes
                mags_final = [mag + apc + zp for mag, apc, zp in zip(mag_aper_hdu, mag_apc_hdu, mag_zp)]

                # Mask bad photometry
                amag_final = np.array(mags_final)
                mag_bad = (amag_final > 50.) | (amag_final < 0.)
                amag_final[mag_bad] = np.nan
                mags_final = amag_final.tolist()

                # Throw out bad sources
                keep = fwhm_file[idx_arrays] * 3600 > 0.1

                # Create fits columns
                col_id = fits.Column(name="ID", array=skycoo2visionsid(skycoord=skycoord_hdu[keep]), format="21A")
                col_ra = fits.Column(name="RA", array=skycoord_hdu.icrs.ra.deg[keep], format="D")
                col_dec = fits.Column(name="DEC", array=skycoord_hdu.icrs.dec.deg[keep], format="D")
                col_fwhm = fits.Column(name="FWHM", array=fwhm_file[idx_arrays][keep] * 3600, format="E")
                col_flags = fits.Column(name="FLAGS", array=flags_file[idx_arrays][keep], format="I")
                col_ell = fits.Column(name="ELLIPTICITY", array=ell_file[idx_arrays][keep], format="E")
                col_elo = fits.Column(name="ELONGATION", array=elo_file[idx_arrays][keep], format="E")
                col_class = fits.Column(name="CLASS", array=class_file[idx_arrays][keep], format="E")

                cols_mag = []
                # noinspection PyTypeChecker
                for mag, diam in zip(mags_final, self._apertures_save):
                    cols_mag.append(fits.Column(name="MAG_APER_{0}".format(diam), array=np.array(mag)[keep],
                                                format="E"))

                # Append columns to HDUList
                hdulist.append(fits.BinTableHDU.from_columns([col_id, col_ra, col_dec] + cols_mag +
                                                             [col_fwhm, col_flags, col_ell, col_elo, col_class]))

            hdulist.writeto(outpath, overwrite=True)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])
