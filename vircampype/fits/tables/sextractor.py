# =========================================================================== #
# Import
import warnings
import numpy as np
import multiprocessing

from astropy.io import fits
from astropy.time import Time
from vircampype.utils import *
from vircampype.setup import *
from vircampype.data.cube import ImageCube
from vircampype.fits.tables.sources import SourceCatalogs
from vircampype.fits.tables.zeropoint import MasterZeroPoint


class SextractorCatalogs(SourceCatalogs):

    def __init__(self, setup, file_paths=None):
        super(SextractorCatalogs, self).__init__(file_paths=file_paths, setup=setup)

    # =========================================================================== #
    # Key defintions
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
        bands = list(set(self.filter))
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

        # Loop over catalogs and build aperture correction
        for idx in range(len(self)):

            # Generate output names
            path_file = "{0}{1}.apcor.fits".format(self.path_apcor, self.file_names[idx])
            path_plot = "{0}{1}.apcor.pdf".format(self.path_qc_apcor, self.file_names[idx])

            if check_file_exists(file_path=path_file.replace(".apcor.", ".apcor{0}.".format(apertures_out[0])),
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
            hdulist_save = [hdulist_base.copy() for _ in range(len(apertures_out))]

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
                ma_apcor, me_apcor, mo_apcor = get_aperture_correction(diameters=apertures_all, magnitudes=mag,
                                                                       func=self.setup["photometry"]["apcor_func"])

                # Obtain aperture correction values for output
                mag_apcor_save = mo_apcor(apertures_out)

                # Shrink image header
                ohdr = resize_header(header=hdr, factor=self.setup["photometry"]["apcor_image_scale"])

                # Loop over apertures and make HDUs
                for aidx in range(len(apertures_out)):

                    # Construct header to append
                    hdr_temp = ohdr.copy()
                    hdr_temp["NSRCAPC"] = (len(mag), "Number of sources used")
                    hdr_temp["APCMAG"] = (mag_apcor_save[aidx], "Aperture correction (mag)")
                    hdr_temp["APCDIAM"] = (apertures_out[aidx], "Aperture diameter (pix)")
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
            for hdul, diams in zip(hdulist_save, apertures_out):

                # Write aperture correction diameter into primary header too
                hdul[0].header["APCDIAM"] = (diams, "Aperture diameter (pix)")
                hdul[0].header[self.setup["keywords"]["object"]] = "APERTURE-CORRECTION"

                # Save to disk
                hdul.writeto(path_file.replace(".apcor.", ".apcor{0}.".format(diams)),
                             overwrite=self.setup["misc"]["overwrite"])

            # QC plot
            if self.setup["misc"]["qc_plots"]:
                self.qc_plot_apcor(path=path_plot, diameters=apertures_all, mag_apcor=mag_apcor,
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

    def build_master_photometry(self):

        # Import
        from vircampype.fits.tables.sources import MasterPhotometry2Mass, MasterPhotometry

        # Processing info
        tstart = message_mastercalibration(master_type="MASTER-PHOTOMETRY", right=None,
                                           silent=self.setup["misc"]["silent"])

        # Construct outpath
        outpath = self.path_master_object + "MASTER-PHOTOMETRY.fits.tab"

        # Print processing info
        message_calibration(n_current=1, n_total=1, name=outpath, d_current=None,
                            d_total=None, silent=self.setup["misc"]["silent"])

        # Check if the file is already there and skip if it is
        if not check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]):

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

    # =========================================================================== #
    # Magnitudes
    # =========================================================================== #
    @property
    def _colnames_apc(self):
        """ Constructor for column names for aperture corrections. """
        return ["MAG_APC_{0}".format(idx+1) for idx in range(len(apertures_out))]

    @property
    def _colnames_aper(self):
        """ Constructor for column names for aperture corrections. """
        return ["MAG_APER_{0}".format(idx+1) for idx in range(len(apertures_out))]

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

    def mag_aper_file(self, idx_file):
        """
        Returns fixed aperture photometry for a given file.

        Parameters
        ----------
        idx_file : int
            Index of file in current instance.

        Returns
        -------
            List of aperture magnitudes for each data extension in given file.

        """
        return self.get_column_file(idx_file=idx_file, column_name="MAG_APER")

    def magerr_aper_file(self, idx_file):
        """
        Returns errors for fixed aperture photometry for a given file.

        Parameters
        ----------
        idx_file : int
            Index of file in current instance.

        Returns
        -------
            List of aperture magnitude errors for each data extension in given file.

        """
        return self.get_column_file(idx_file=idx_file, column_name="MAGERR_APER")

    def mag_aper_dict_file(self, idx_file):
        """
        Constructs a dictionary for all aperture magnitudes (those that will be saved) for a given file.

        Parameters
        ----------
        idx_file : int
            File index in self.

        Returns
        -------
        dict
            Dictionary of aperture magnitudes.

        """

        # Get aperture magnitudes for given file
        mag_aper_file = self.mag_aper_file(idx_file=idx_file)

        # Loop over aperture diameters
        mag_aper_dict_file = {}
        for idx_apc, cname in zip(self._aperture_save_idx, self._colnames_aper):

            # Make empty list for current aperture
            mag_aper_dict_file[cname] = []

            # Loop over HDUs and append aperture magnitudes
            for idx_hdu in range(len(self.data_hdu[idx_file])):

                mag_aper_dict_file[cname].append(mag_aper_file[idx_hdu][:, idx_apc])

        # return filled dictionary
        return mag_aper_dict_file

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
        for cname in self._colnames_apc:
            self._mag_apc_dict[cname] = self.get_columns(column_name=cname)
        return self._mag_apc_dict

    def mag_apc_dict_file(self, idx_file):
        """
        Reads all aperture corrections from a given file for each extension and each source.

        Parameters
        ----------
        idx_file : int
            File index in self.

        Returns
        -------
        dict
            Dictionary with aperture corrections.
        """

        # Get APC columns
        temp = self.get_columns_file(idx_file=idx_file, column_names=self._colnames_apc)

        # Return dictionary with extracted valus
        return {d: mag for d, mag in zip(self._colnames_apc, temp)}

    def _get_columns_zp_method_file(self, idx_file, key_ra=None, key_dec=None):
        """ Convenience method for ZP calculation. """

        # Import
        from astropy.coordinates import SkyCoord

        # Select default RA/DEC keys
        if key_ra is None:
            key_ra = self._key_ra
        if key_dec is None:
            key_dec = self._key_dec

        # Construct column names to extract from file
        cn = ["MAG_APER", key_ra, key_dec] + self._colnames_apc

        # Extract all data at once from file
        mag_aper, ra, dec, *mag_apc = self.get_columns_file(idx_file=idx_file, column_names=cn)

        # Build skycoord
        skycoord = [SkyCoord(ra=r, dec=d, frame="icrs", unit="deg") for r, d in zip(ra, dec)]

        # Return
        return mag_aper, mag_apc, skycoord

    _master_zeropoint = None

    @property
    def master_zeropoint(self):
        """
        Fetches the master zero point file for each file in self.

        Returns
        -------
        MasterZeroPoint
            MasterZeroPoint instance.

        """

        if self._master_zeropoint is not None:
            return self._master_zeropoint

        self._master_zeropoint = self.get_master_zeropoint()
        return self._master_zeropoint

    # =========================================================================== #
    # Catalog modifications
    # =========================================================================== #
    def add_aperture_correction(self):

        # Processing info
        tstart = message_mastercalibration(master_type="ADDING APETURE CORRECTION", silent=self.setup["misc"]["silent"])

        # Construct aperture corrections dict
        apc_self = [self.get_aperture_correction(diameter=diam) for diam in apertures_out]

        # Loop over each file
        for idx_cat_file in range(len(self)):

            # Load current hdulist
            with fits.open(self.full_paths[idx_cat_file], mode="update") as chdulist:

                # Check if aperture correction has already been added
                done = True
                for i, cname in zip(self.data_hdu[idx_cat_file], self._colnames_apc):
                    if cname not in chdulist[i].data.names:
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
                    for apc, cname in zip(apc_file, self._colnames_apc):

                        # Extract aperture correction from image
                        a = apc.get_apcor(skycoo=skycoord_hdu, file_index=0, hdu_index=idx_apc_hdu)

                        # Add as new column for each source
                        new_cols.add_col(fits.Column(name=cname, array=a, **kwargs_column_mag))

                    # Replace HDU from input catalog
                    chdulist[idx_cat_hdu] = fits.BinTableHDU.from_columns(ccolumns + new_cols)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    def add_calibrated_photometry(self):

        # Processing info
        tstart = message_mastercalibration(master_type="ADDING CALIBRATED PHOTOMETRY",
                                           silent=self.setup["misc"]["silent"])

        # Fetch zero points
        master_zp = self.master_zeropoint

        for idx_file in range(len(self)):

            # Read aperture magnitudes
            mag_aper_file = self.mag_aper_dict_file(idx_file=idx_file)
            mag_apc_file = self.mag_apc_dict_file(idx_file=idx_file)

            # Check if already modified
            try:
                self.dataheaders_get_keys(keywords=self._zp_keys, file_index=idx_file)
                print(BColors.WARNING + "{0} already modified".format(self.file_names[idx_file]) + BColors.ENDC)
                continue

            # If error is raised, pass
            except KeyError:
                pass

            # Load current hdulist
            with fits.open(self.full_paths[idx_file], mode="update") as chdulist:

                # Loop over HDUs
                for idx_hdu, idx_hdu_file in zip(range(len(self.data_hdu[idx_file])), self.data_hdu[idx_file]):

                    # Print info
                    message_calibration(n_current=idx_file + 1, n_total=len(self), name=self.file_names[idx_file],
                                        d_current=idx_hdu + 1, d_total=len(self.data_hdu[idx_file]))

                    # Get columns and header for current HDU
                    ccolumns = chdulist[idx_hdu_file].data.columns
                    cheader = chdulist[idx_hdu_file].header

                    # Empy ColDefs for columns to be added
                    new_cols = fits.ColDefs([])

                    # Loop over aperture diameters
                    for apc_name, aper_name, diam, kw, ckw in \
                            zip(self._colnames_apc, self._colnames_aper, apertures_out,
                                self._zp_keys, self._zp_comments):

                        # Get ZP
                        zp = master_zp.zp_diameter(diameter=diam)[idx_file][idx_hdu]

                        # Compute final magnitude
                        mag_final = mag_aper_file[aper_name][idx_hdu] + mag_apc_file[apc_name][idx_hdu] + zp

                        # Add as new column
                        new_cols.add_col(fits.Column(name=aper_name, array=mag_final, **kwargs_column_mag))

                        # Add ZP to header
                        cheader[kw] = (np.round(zp, decimals=4), ckw)

                    # Add new columns and header
                    chdulist[idx_hdu_file] = fits.BinTableHDU.from_columns(ccolumns + new_cols, header=cheader)

            # Remove saved headers
            self.delete_headers_temp(file_index=idx_file)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

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
        master_photometry = self.get_master_photometry()

        # Now loop through separated files
        for files, fidx in zip(split, range(1, len(split) + 1)):

            # Create master dark name
            outpath = self.path_master_object + "MASTER-SUPERFLAT_{0:11.5f}.fits".format(files.mjd_mean)

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]):
                continue

            # Fetch filter of current catalog
            filter_catalog = files.filter[0]

            # Filter master catalog for good data
            mkeep_qfl = [True if x in "AB" else False for x in master_photometry.qflags(key=filter_catalog)[0][0]]
            mkeep_cfl = [True if x == "0" else False for x in master_photometry.cflags(key=filter_catalog)[0][0]]

            # Combine quality and contamination flag
            mkeep = mkeep_qfl and mkeep_cfl

            # Fetch magnitude and coordinates for master catalog
            master_mag = master_photometry.mag(key=master_photometry.translate_filter(key=filter_catalog))[0][0][mkeep]
            master_skycoord = master_photometry.skycoord()[0][0][mkeep]

            data_headers, flx_scale, flx_scale_global, n_sources = [], [], [], []
            for idx_hdu, idx_print in zip(files.data_hdu[0], range(len(files.data_hdu[0]))):

                # Print processing info
                message_calibration(n_current=fidx, n_total=len(split), name=outpath, d_current=idx_print + 1,
                                    d_total=len(files.data_hdu[0]), silent=self.setup["misc"]["silent"])

                # Read current HDU for all files
                data = files.hdu2table(hdu_index=idx_hdu)

                # Extract data for all files for this extension
                aa = np.array(flat_list([d["ALPHA_J2000"] for d in data]))
                dd = np.array(flat_list([d["DELTA_J2000"] for d in data]))
                xx = np.array(flat_list([d["XWIN_IMAGE"] for d in data]))
                yy = np.array(flat_list([d["YWIN_IMAGE"] for d in data]))
                ff = np.array(flat_list([d["FLAGS"] for d in data]))
                mm = np.array(flat_list([d["MAG_AUTO"] for d in data]))
                fwhm = np.array(flat_list([d["FWHM_WORLD"] for d in data])) * 3600
                ee = np.array(flat_list([d["ELLIPTICITY"] for d in data]))

                # Filter for good sources
                good = (np.isfinite(aa) & np.isfinite(dd) & np.isfinite(xx) &
                        np.isfinite(yy) & np.isfinite(ff) & np.isfinite(mm) &
                        (ff == 0) & (ee < 0.2) & (fwhm > 0.3) & (fwhm < 1.5))

                # Apply filter
                aa, dd, xx, yy, mm = aa[good], dd[good], xx[good], yy[good], mm[good]

                # Get ZP for each single star
                zp = get_zeropoint_radec(ra_cal=aa, dec_cal=dd, mag_cal=mm, mag_ref=master_mag,
                                         ra_ref=master_skycoord.icrs.ra.deg, dec_ref=master_skycoord.icrs.dec.deg,
                                         mag_limits_ref=master_photometry.mag_lim, return_all=True)

                # Sigma clip ZP array just to be sure
                zp = sigma_clip(zp, sigma_level=2.5, sigma_iter=5)

                # Grid values to detector size array
                grid_zp = grid_value_2d(x=xx, y=yy, value=zp, ngx=50, ngy=50, kernel_scale=0.1,
                                        naxis1=self.setup["data"]["dim_x"], naxis2=self.setup["data"]["dim_y"])

                # Convert to flux scale
                flx_scale.append(10**(grid_zp / 2.5))

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
        return ["HIERARCH PYPE MAGZP {0}".format(i + 1) for i in range(len(apertures_out))]

    @property
    def _zp_comments(self):
        return ["ZP for {0} pix aperture".format(d) for d in apertures_out]

    @property
    def _zperr_keys(self):
        return ["HIERARCH PYPE MAGZPERR {0}".format(i + 1) for i in range(len(apertures_out))]

    @property
    def _zperr_comments(self):
        return ["ZP error for {0} pix aperture".format(d) for d in apertures_out]

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

    def build_master_zeropoint(self):
        """ Build ZPs. """

        # Processing info
        tstart = message_mastercalibration(master_type="MASTER ZERO POINT", silent=self.setup["misc"]["silent"])

        # Get master photometry catalog
        master_photometry = self.get_master_photometry()

        # Loop over catalogs
        for idx_file in range(len(self)):

            # Create master dark name
            outpath = self.path_master_object + "MASTER-ZEROPOINT.MJD_{0:11.5f}.fits.tab".format(self.mjd[idx_file])

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]):
                continue

            # Fetch filter of current catalog
            filter_catalog = self.filter[idx_file]

            # Filter master catalog for good data
            mkeep_qfl = [True if x in "AB" else False for x in master_photometry.qflags(key=filter_catalog)[0][0]]
            mkeep_cfl = [True if x == "0" else False for x in master_photometry.cflags(key=filter_catalog)[0][0]]

            # Combine quality and contamination flag
            mkeep = mkeep_qfl and mkeep_cfl

            # Fetch magnitude and coordinates for master catalog
            master_mag = master_photometry.mag(key=master_photometry.translate_filter(key=filter_catalog))[0][0][mkeep]
            master_skycoord = master_photometry.skycoord()[0][0][mkeep]

            # Load aperture magnitudes, aperture corrections, and skycoords together from file
            mag_aper_file, mag_apc_file, skycoord_file = self._get_columns_zp_method_file(idx_file=idx_file)

            # Loop over extensions
            zp_hdu = {diam_apc: [] for diam_apc in apertures_out}
            zperr_hdu = {diam_apc: [] for diam_apc in apertures_out}
            for idx_hdu, idx_file_hdu in zip(range(len(self.data_hdu[idx_file])), self.data_hdu[idx_file]):

                # Message
                message_calibration(n_current=idx_file+1, n_total=len(self), name=outpath,
                                    d_current=idx_hdu+1, d_total=len(self.data_hdu[idx_file]),
                                    silent=self.setup["misc"]["silent"])

                # Compute final magnitudes
                mags = {diam_apc: mag_aper_file[idx_hdu][:, idx_apc] + apc[idx_hdu]
                        for idx_apc, diam_apc, apc in zip(self._aperture_save_idx, apertures_out, mag_apc_file)}

                # TODO: This here does the same sky match for all apertures...
                for diam_apc in apertures_out:
                    zp, zperr = get_zeropoint(skycoo_cal=skycoord_file[idx_hdu], mag_cal=mags[diam_apc],
                                              mag_limits_ref=master_photometry.mag_lim,
                                              skycoo_ref=master_skycoord, mag_ref=master_mag)
                    zp_hdu[diam_apc].append(zp)
                    zperr_hdu[diam_apc].append(zperr)

            # Make header cards
            prime_cards = make_cards(keywords=[self.setup["keywords"]["date_mjd"], self.setup["keywords"]["date_ut"],
                                               self.setup["keywords"]["object"], self.setup["keywords"]["filter"]],
                                     values=[self.mjd[idx_file], self.time_obs[idx_file],
                                             "MASTER-ZEROPOINT", self.filter[idx_file]])
            prhdu = fits.PrimaryHDU(header=fits.Header(cards=prime_cards))

            # Create table HDU for output
            cols = [fits.Column(name="ZP_{0}".format(apc_diam), array=zp_hdu[apc_diam], **kwargs_column_mag)
                    for apc_diam in apertures_out] + \
                   [fits.Column(name="ZPERR_{0}".format(apc_diam), array=zperr_hdu[apc_diam], **kwargs_column_mag)
                    for apc_diam in apertures_out]
            tbhdu = fits.TableHDU.from_columns(cols)
            thdulist = fits.HDUList([prhdu, tbhdu])

            # Write
            # thdulist.writeto(fileobj=outpath, overwrite=self.setup["misc"]["overwrite"])
            thdulist.writeto(fileobj=outpath, overwrite=True)

            # QC plot
            if self.setup["misc"]["qc_plots"]:
                MasterZeroPoint(setup=self.setup, file_paths=outpath).qc_plot_zp(overwrite=True)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    @property
    def mean_zeropoints(self):
        """
        Reads ZPs from file headers.

        Returns
        -------
        iterable
            List of lists containing ZPs

        Raises
        ------
        KeyError
            When ZPs not found

        """

        # Return Mean ZP and ZPerr
        return self.master_zeropoint.zp_mean, self.master_zeropoint.zperr_mean

    def zeropoint_diam(self, diameter):
        """
        Fetches the zero points for a given aperture diameter.

        Parameters
        ----------
        diameter : str
            Aperture diameter.

        Returns
        -------
        iterable
            List of lists.

        """

        return (self.master_zeropoint.zp_diameter(diameter=diameter),
                self.master_zeropoint.zperr_diameter(diameter=diameter))

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
        df = 10**(np.array(self.mean_zeropoints[0]) / -2.5)

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
    # QC
    # =========================================================================== #
    def plot_qc_photometry(self, axis_size=4, mode="pawprint"):

        # Import
        from astropy.units import Unit
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
        from astropy.stats import sigma_clip
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Processing info
        tstart = message_mastercalibration(master_type="QC PHOTOMETRY", silent=self.setup["misc"]["silent"])

        # Get ZPs for all data
        zps_all = self.mean_zeropoints[0]

        # Read master photometry table
        master_photometry = self.get_master_photometry()[0]
        sc_master = master_photometry.skycoord()[0][0]

        for idx_file in range(len(self)):

            # Generate outpath
            outpath_1d = "{0}{1}.phot.1D.pdf".format(self.path_qc_photometry, self.file_names[idx_file])
            outpath_2d = "{0}{1}.phot.2D.pdf".format(self.path_qc_photometry, self.file_names[idx_file])

            # Check if file exists
            if check_file_exists(file_path=outpath_2d, silent=self.setup["misc"]["silent"]):
                continue

            # Get magnitudes in master catalog
            mag_master = master_photometry.mag(master_photometry.translate_filter(key=self.filter[idx_file]))[0][0]

            # Get ZPs
            zps_file = zps_all[idx_file]

            # Fetch magnitudes and aperture corrections
            apc_idx = self._aperture_save_idx[3]
            mag_file = self.get_column_file(idx_file=idx_file, column_name="MAG_APER")
            apc_file = self.get_column_file(idx_file=idx_file, column_name=self._colnames_apc[3])

            # Get coordinates
            skycoord_file = self.skycoord_file(idx_file=idx_file)
            x_file = self.get_column_file(idx_file=idx_file, column_name="X_IMAGE")
            y_file = self.get_column_file(idx_file=idx_file, column_name="Y_IMAGE")

            # =========================================================================== #
            # 1D
            # =========================================================================== #
            # Make plot grid
            if len(self) == 1:
                fig, ax_file = get_plotgrid(layout=(1, 1), xsize=2*axis_size, ysize=2*axis_size)
                ax_file = [ax_file]
            else:
                fig, ax_file = get_plotgrid(layout=self.setup["instrument"]["layout"],
                                            xsize=axis_size, ysize=axis_size / 2)
                ax_file = ax_file.ravel()

            for idx_hdu in range(len(self.data_hdu[idx_file])):

                # Get magnitudes into shape
                mag_final = mag_file[idx_hdu][:, apc_idx] + apc_file[idx_hdu]

                # Print processing info
                message_calibration(n_current=idx_file+1, n_total=len(self), name=outpath_1d, d_current=idx_hdu+1,
                                    d_total=len(skycoord_file), silent=self.setup["misc"]["silent"])

                # Xmatch science with reference
                zp_idx, zp_d2d, _ = skycoord_file[idx_hdu].match_to_catalog_sky(sc_master)

                # Get good indices in reference catalog and in current field
                idx_master = zp_idx[zp_d2d < 1 * Unit("arcsec")]
                idx_final = np.arange(len(zp_idx))[zp_d2d < 1 * Unit("arcsec")]

                # Apply indices filter
                mag_final = mag_final[idx_final]
                mag_match = mag_master[idx_master]

                # Draw photometry
                ax_file[idx_hdu].scatter(mag_match, mag_match - mag_final,
                                         s=15, lw=0, alpha=0.4, zorder=0, c="crimson")

                # Draw ZP
                ax_file[idx_hdu].axhline(zps_file[idx_hdu], zorder=1, c="black", alpha=0.5)

                # Annotate detector ID
                ax_file[idx_hdu].annotate("Det.ID: {0:0d}".format(idx_hdu + 1), xy=(0.02, 0.96),
                                          xycoords="axes fraction", ha="left", va="top")

                # Set limits
                ax_file[idx_hdu].set_xlim(10, 18)
                ax_file[idx_hdu].set_ylim(np.floor(np.median(zps_file) - 0.5),
                                          np.ceil(np.median(zps_file) + 0.5))

                # Modify axes
                if idx_hdu >= len(skycoord_file) - self.setup["instrument"]["layout"][0]:
                    ax_file[idx_hdu].set_xlabel("{0} {1} (mag)".format(self.setup["photometry"]["reference"].upper(),
                                                                       self.filter[idx_file]))
                else:
                    ax_file[idx_hdu].axes.xaxis.set_ticklabels([])
                if idx_hdu % self.setup["instrument"]["layout"][0] == 0:
                    ax_file[idx_hdu].set_ylabel(r"$\Delta${0} (mag)".format(self.filter[idx_file]))
                else:
                    ax_file[idx_hdu].axes.yaxis.set_ticklabels([])

                # Set ticks
                ax_file[idx_hdu].xaxis.set_major_locator(MaxNLocator(5))
                ax_file[idx_hdu].xaxis.set_minor_locator(AutoMinorLocator())
                ax_file[idx_hdu].yaxis.set_major_locator(MaxNLocator(3))
                ax_file[idx_hdu].yaxis.set_minor_locator(AutoMinorLocator())

            # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
                fig.savefig(outpath_1d, bbox_inches="tight")
            plt.close("all")

            # =========================================================================== #
            # 2D
            # =========================================================================== #
            # Coadd mode
            if len(self) == 1:
                fig, ax_file = get_plotgrid(layout=(1, 1), xsize=2*axis_size, ysize=2*axis_size)
                ax_file = [ax_file]
            else:
                fig, ax_file = get_plotgrid(layout=self.setup["instrument"]["layout"], xsize=axis_size, ysize=axis_size)
                ax_file = ax_file.ravel()
            cax = fig.add_axes([0.3, 0.92, 0.4, 0.02])

            im = None
            for idx_hdu in range(len(self.data_hdu[idx_file])):

                # Print processing info
                message_calibration(n_current=idx_file+1, n_total=len(self), name=outpath_2d, d_current=idx_hdu+1,
                                    d_total=len(skycoord_file), silent=self.setup["misc"]["silent"])

                # Get magnitudes into shape
                mag_final = mag_file[idx_hdu][:, apc_idx] + apc_file[idx_hdu]

                # Get ZP for each source
                zp_hdu = get_zeropoint(skycoo_cal=skycoord_file[idx_hdu], skycoo_ref=sc_master, mag_cal=mag_final,
                                       mag_ref=mag_master, mag_limits_ref=(12, 15), return_all=True)

                # sigma-clip array
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    zp_hdu = sigma_clip(zp_hdu, masked=True, sigma=3, maxiters=2).filled(np.nan)

                # Apply filter
                fil = np.isfinite(zp_hdu)
                x_hdu, y_hdu, zp_hdu = x_file[idx_hdu][fil], y_file[idx_hdu][fil], zp_hdu[fil]

                # Grid value into image
                if mode == "pawprint":
                    ngx, ngy, kernel_scale = 50, 50, 0.2
                elif mode == "tile":
                    ngx, ngy, kernel_scale = 200, 200, 0.1
                else:
                    raise ValueError("Mode '{0}' not supported".format(mode))
                grid = grid_value_2d(x=x_hdu, y=y_hdu, value=zp_hdu, naxis1=500, naxis2=500,
                                     ngx=ngx, ngy=ngy, kernel_scale=kernel_scale)

                # Draw
                kwargs = {"vmin": np.median(zp_hdu)-0.2, "vmax": np.median(zp_hdu)+0.2, "cmap": get_cmap("RdYlBu", 20)}
                extent = [np.nanmin(x_hdu), np.nanmax(x_hdu), np.nanmin(y_hdu), np.nanmax(y_hdu)]
                im = ax_file[idx_hdu].imshow(grid, extent=extent, origin="lower", **kwargs)
                ax_file[idx_hdu].scatter(x_hdu, y_hdu, c=zp_hdu, s=7, lw=0.5, ec="black", **kwargs)

                # Annotate detector ID
                ax_file[idx_hdu].annotate("Det.ID: {0:0d}".format(idx_hdu + 1), xy=(0.02, 1.01),
                                          xycoords="axes fraction", ha="left", va="bottom")

                # Modify axes
                if idx_hdu >= len(skycoord_file) - self.setup["instrument"]["layout"][0]:
                    ax_file[idx_hdu].set_xlabel("X (pix)")
                else:
                    ax_file[idx_hdu].axes.xaxis.set_ticklabels([])
                if idx_hdu % self.setup["instrument"]["layout"][0] == 0:
                    ax_file[idx_hdu].set_ylabel("Y (pix)")
                else:
                    ax_file[idx_hdu].axes.yaxis.set_ticklabels([])

                ax_file[idx_hdu].set_aspect("equal")

                # Set ticks
                ax_file[idx_hdu].xaxis.set_major_locator(MaxNLocator(5))
                ax_file[idx_hdu].xaxis.set_minor_locator(AutoMinorLocator())
                ax_file[idx_hdu].yaxis.set_major_locator(MaxNLocator(5))
                ax_file[idx_hdu].yaxis.set_minor_locator(AutoMinorLocator())

                # Set limits
                ax_file[idx_hdu].set_xlim(np.min(x_hdu), np.max(x_hdu))
                ax_file[idx_hdu].set_ylim(np.min(y_hdu), np.max(y_hdu))

            # Add colorbar
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal", label="Zero Point (mag)")
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")

            # # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
                fig.savefig(outpath_2d, bbox_inches="tight")
            plt.close("all")

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

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
            path_ahead = self.full_paths[idx_file].replace(".sources.fits", ".ahead")

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
                                  additional=shitty_kw, outpaths=(path_pawprint_img, path_pawprint_cat),
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
                         paths_prov=prov_images.full_paths, outpath=path_tile,
                         compressed=self.setup["compression"]["compress_phase3"])

        # There also has to be a weight map
        with fits.open(swarped.full_paths[0].replace(".fits", ".weight.fits")) as weight:

            # Add PRODCATG
            weight[0].header["PRODCATG"] = "ANCILLARY.WEIGHTMAP"

            # Save
            weight.writeto(path_weig, overwrite=False, checksum=True)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])
