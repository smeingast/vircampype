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
        names = ["{0}scamp_{1}".format(self.file_directories[0], qt.lower()) for qt in
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
        names = [x.replace(".sources", ".ahead") for x in self.full_paths]
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

        # Load preset
        options = yml2config(nthreads=self.setup["misc"]["n_threads"], checkplot_type=self._scamp_qc_types(joined=True),
                             checkplot_name=self._scamp_qc_names(), skip=["HEADER_NAME"],
                             path=get_resource_path(package=self._scamp_preset_package, resource="scamp.yml"))

        # Construct commands for source extraction
        cmd = "{0} {1} -c {2} -HEADER_NAME {3} {4}" \
              "".format(self._bin_scamp, self._scamp_catalog_paths, self._scamp_default_config,
                        self._scamp_header_names(joined=True), options)

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
            path_file = "{0}{1}.apcor.fits".format(self.path_obspar, self.file_names[idx])
            path_plot = "{0}{1}.apcor.pdf".format(self.path_qc, self.file_names[idx])

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
            mag_apcor, magerr_apcor, models_apcor = [], [], []

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
                    hdr_temp["APCMAG"] = (mag_apcor_save[aidx], "Aperture correction (mag)")
                    hdr_temp["APCDIAM"] = (diameters_save[aidx], "Aperture diameter (pix)")
                    hdr_temp["APCMODEL"] = (mo_apcor.name, "Aperture correction model name")
                    for i in range(len(mo_apcor.parameters)):
                        hdr_temp["APCMPAR{0}".format(i+1)] = (mo_apcor.parameters[i], "Model parameter {0}".format(i+1))

                    hdulist_save[aidx].append(hdr2imagehdu(header=hdr_temp, fill_value=mag_apcor_save[aidx],
                                                           dtype=np.float32))

                # Append to lists for QC plot
                mag_apcor.append(ma_apcor)
                magerr_apcor.append(me_apcor)
                models_apcor.append(mo_apcor)

            # Save aperture correction as MEF
            for hdul, diams in zip(hdulist_save, diameters_save):

                # Write aperture correction diameter into primary header too
                hdul[0].header["APCDIAM"] = (diams, "Aperture diameter (pix)")

                # Save to disk
                hdul.writeto(path_file.replace(".apcor.", ".apcor{0}.".format(diams)),
                             overwrite=self.setup["misc"]["overwrite"])

            # QC plot
            if self.setup["misc"]["qc_plots"]:
                self.qc_plot_apcor(path=path_plot, diameters=diameters_eval, mag_apcor=mag_apcor,
                                   magerr_apcor=magerr_apcor, models=models_apcor, axis_size=4,
                                   overwrite=self.setup["misc"]["overwrite"])

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    def qc_plot_apcor(self, path, diameters, mag_apcor, magerr_apcor, models, axis_size=4, overwrite=False):
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

        # Save plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
            fig.savefig(path, bbox_inches="tight")
        plt.close("all")

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

        # Processing info
        tstart = message_mastercalibration(master_type="ZERO POINTS", silent=self.setup["misc"]["silent"])

        # Get master photometry catalog
        master_photometry = self.get_master_photometry()

        # Get indices of apertures to save
        apertures_idx = [[i for i, x in enumerate(self._apertrure_eval) if x == b][0] for b in self._apertures_save]

        # Construct aperture corrections dict
        apcors = [self.get_aperture_correction(diameter=diam) for diam in self._apertures_save]

        # Loop over catalogs
        zp_avg_catalogs, zperr_avg_catalogs = [], []
        for idx_catalog in range(len(self)):

            # Check if zeropoints have been determined already
            try:

                # Try reading ZPs from headers
                a = self.dataheaders_get_keys(keywords=[self._zp_avg_key, self._zperr_avg_key], file_index=idx_catalog)
                zp_avg, zperr_avg = a[0][0], a[1][0]

                # If already there, append, and continue with next file
                zp_avg_catalogs.append(zp_avg)
                zperr_avg_catalogs.append(zperr_avg)
                continue

            except KeyError:
                # Make empty list to fill up later
                zp_avg, zperr_avg = [], []

            # Construct outpath of qc plot
            path_plot = "{0}{1}.zp.pdf".format(self.path_qc, self.file_names[idx_catalog])

            # Fetch filter of current catalog
            filter_catalog = self.filters[idx_catalog]

            # Filter master catalog for good data
            mkeep = [True if x in "AB" else False for x in master_photometry.qflags(key=filter_catalog)[0][0]]

            # Fetch magnitude and coordinates for master catalog
            mmag = master_photometry.mag(key=master_photometry.translate_filter(key=filter_catalog))[0][0][mkeep]
            msc = master_photometry.skycoord()[0][0][mkeep]

            # Current aperture corrections
            apcors_catalog = [x[idx_catalog] for x in apcors]

            # Loop over extensions
            for idx_hdu, fidx_hdu in zip(range(len(self.data_hdu[idx_catalog])), self.data_hdu[idx_catalog]):

                # Message
                message_calibration(n_current=idx_catalog+1, n_total=len(self), name=self.full_paths[idx_catalog],
                                    d_current=idx_hdu+1, d_total=len(self.data_hdu[idx_catalog]),
                                    silent=self.setup["misc"]["silent"])

                # Fetch aperture corrections for all sources
                c = [s.get_apcor(skycoo=self.skycoord()[idx_catalog][idx_hdu], file_index=0,
                                 hdu_index=idx_hdu+1) for s in apcors_catalog]

                # Fetch magnitudes
                mags = [self.mag_aper[idx_catalog][idx_hdu][:, idx_apc] for idx_apc in apertures_idx]

                # Apply aperture correction to magnitudes
                mags = [m + a for m, a in zip(mags, c)]

                # Get zeropoints for each aperture
                zp_values, zperr_values = [], []
                for m in mags:
                    zp, zperr = get_zeropoint(skycoo_cal=self.skycoord()[idx_catalog][idx_hdu], mag_cal=m,
                                              mag_limits_ref=master_photometry.mag_lim,
                                              skycoo_ref=msc, mag_ref=mmag)
                    zp_values.append(float(str(np.round(zp, decimals=4))))      # This forces only 4 decimals to appear
                    zperr_values.append(float(str(np.round(zperr, decimals=4))))  # in headers

                # Add zero points to header
                add_keys_hdu(path=self.full_paths[idx_catalog], hdu=fidx_hdu, keys=self._zp_keys,
                             values=zp_values, comments=self._zp_comments)

                # Add zero point errors to header
                add_keys_hdu(path=self.full_paths[idx_catalog], hdu=fidx_hdu, keys=self._zperr_keys,
                             values=zperr_values, comments=self._zperr_comments)

                # Add average aperture-corrected ZP
                zp_avg_vals = [float(str(np.round(np.mean(zp_values), decimals=4))),
                               float(str(np.round(np.std(zp_values), decimals=4)))]
                add_keys_hdu(path=self.full_paths[idx_catalog], hdu=fidx_hdu, values=zp_avg_vals,
                             keys=[self._zp_avg_key, self._zperr_avg_key],
                             comments=[self._zp_avg_comment, self._zperr_avg_comment])

                # Append to lists for plotting
                zp_avg.append(zp_avg_vals[0])
                zperr_avg.append(zp_avg_vals[1])

                # Force reloading header after this temporary header
                self.delete_headers_temp(file_index=idx_catalog)

            # QC plot
            if self.setup["misc"]["qc_plots"]:
                plot_value_detector(values=zp_avg, errors=zperr_avg, path=path_plot)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

        # Return ZPs
        return zp_avg_catalogs, zperr_avg_catalogs

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
