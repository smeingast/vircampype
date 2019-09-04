# =========================================================================== #
# Import
import warnings
import subprocess
import multiprocessing

from vircampype.utils.miscellaneous import *
from vircampype.utils.plots import get_plotgrid
from vircampype.utils.astromatic import yml2config
from vircampype.fits.tables.common import FitsTables
from vircampype.utils.astromatic import sextractor2imagehdr
from vircampype.utils.photometry import get_aperture_correction


class SextractorTable(FitsTables):

    def __init__(self, setup, file_paths=None):
        super(SextractorTable, self).__init__(file_paths=file_paths, setup=setup)

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

    def scamp(self):

        # Find executable
        path_exe = which(self.setup["astromatic"]["bin_scamp"])

        # Shortcut for preset package
        package_presets = "vircampype.resources.astromatic.presets"

        # Find default config
        path_default_config = get_resource_path(package="vircampype.resources.astromatic.scamp",
                                                resource="default.config")

        # QC plots
        qc_types = ["FGROUPS", "DISTORTION", "ASTR_INTERROR2D", "ASTR_INTERROR1D",
                    "ASTR_REFERROR2D", "ASTR_REFERROR1D"]
        qc_names = ",".join(["{0}scamp_{1}".format(self.file_directories[0], qt.lower()) for qt in qc_types])
        qc_types = ",".join(qc_types)

        # Header names
        hdr_names = ",".join([x.replace(".sources", ".ahead") for x in self.full_paths])

        # Load preset
        options = yml2config(nthreads=self.setup["misc"]["n_threads"], checkplot_type=qc_types,
                             checkplot_name=qc_names, skip=["HEADER_NAME"],
                             path=get_resource_path(package=package_presets, resource="scamp.yml"))

        # Get string for catalog paths
        paths_catalogs = " ".join(self.full_paths)

        # Construct commands for source extraction
        cmd = "{0} {1} -c {2} -HEADER_NAME {3} {4}".format(path_exe, paths_catalogs, path_default_config,
                                                           hdr_names, options)

        # Run Scamp
        # cp = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        subprocess.run(cmd, shell=True, executable="/bin/bash")

        # Return header paths
        return hdr_names.split(",")

    def build_aperture_correction(self):

        # Processing info
        tstart = message_mastercalibration(master_type="APERTURE CORRECTION", silent=self.setup["misc"]["silent"],
                                           right=None)

        # Obtain diameters from setup and put the in list
        diameters_eval = [float(x) for x in self.setup["photometry"]["apcor_diam_eval"].split(",")]
        # diameters_save = [float(x) for x in self.setup["photometry"]["apcor_diam_save"].split(",")]

        # Loop over catalogs and build aperture correction
        for idx in range(len(self)):

            # Generate output names
            path_qc_plot = "{0}{1}.pdf".format(self.file_directories[idx], self.file_names[idx])

            # Print processing info
            # TODO: Replace path
            message_calibration(n_current=idx+1, n_total=len(self), name=path_qc_plot, d_current=None,
                                d_total=None, silent=self.setup["misc"]["silent"])

            # Read currect catalog
            tab = self.file2table(file_index=idx)

            # Lists to save results for this image
            mag_apcor, magerr_apcor, models_apcor = [], [], []

            # Loop over extensions and get aperture correction after filtering
            for t in tab:

                # Read magnitudes
                mag = t["MAG_APER"]

                # Remove bad sources
                # bad = (class_star < 0.7) | (flags > 0) | (np.sum(mag > 0, axis=1) > 0) | \
                #       (np.sum(magdiff > 0, axis=1) > 0) | (fwhm < 1.0) | (fwhm > 6.0)

                # Remove bad sources (class, flags, bad mags, bad mag diffs, bad fwhm, bad mag errs)
                good = (t["CLASS_STAR"] > 0.7) & (t["FLAGS"] == 0) & \
                       (np.sum(mag > 0, axis=1) == 0) & (np.sum(np.diff(mag, axis=1) > 0, axis=1) == 0) & \
                       (t["FWHM_IMAGE"] > 1.0) & (t["FWHM_IMAGE"] < 6.0) & (np.nanmean(t["MAGERR_APER"], axis=1) < 0.1)

                # Only keep good sources
                mag = mag[good, :]

                # Obtain aperture correction from cleaned sample
                ma_apcor, me_apcor, mo_apcor = get_aperture_correction(diameters=diameters_eval, magnitudes=mag,
                                                                       func=self.setup["photometry"]["apcor_func"])

                # Append to lists for QC plot
                mag_apcor.append(ma_apcor)
                magerr_apcor.append(me_apcor)
                models_apcor.append(mo_apcor)

                # Obtain aperture correction values for output
                # mag_apcor_save = mo_apcor(diameters_save)
                # print(mag_apcor_save)
                # exit()

            # QC plot
            if self.setup["misc"]["qc_plots"]:
                self.qc_plot_apcor(path=path_qc_plot, diameters=diameters_eval, mag_apcor=mag_apcor,
                                   magerr_apcor=magerr_apcor, models=models_apcor, axis_size=4,
                                   overwrite=self.setup["misc"]["overwrite"])

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    def qc_plot_apcor(self, path, diameters, mag_apcor, magerr_apcor, models, axis_size=4, overwrite=False):

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
