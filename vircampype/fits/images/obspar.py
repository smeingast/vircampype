# =========================================================================== #
# Import
import os
import warnings
import numpy as np

from astropy.io import fits
from vircampype.utils import *
from vircampype.fits.images.sky import SkyImages
from vircampype.fits.images.common import FitsImages


class ApcorImages(SkyImages):

    def __init__(self, setup, file_paths=None):
        super(ApcorImages, self).__init__(setup=setup, file_paths=file_paths)

    @property
    def diameters(self):
        """
        Fetches aperture diameters from headers.

        Returns
        -------
        iterable
            List of lists for each file and each detector.

        """
        return self.dataheaders_get_keys(keywords=["DIAMAPC"])[0]

    @property
    def n_sources(self):
        """
        Fetches number of sources used to build aperture correction.

        Returns
        -------
        iterable
            List of lists for each file and each detector.

        """
        return self.dataheaders_get_keys(keywords=["NSRCAPC"])[0]

    @property
    def mag_apc(self):
        """
        Fetches average aperture correction.

        Returns
        -------
        iterable
            List of lists for each file and each detector.

        """
        return self.dataheaders_get_keys(keywords=["MAGAPC"])[0]

    @property
    def _swarp_preset_apcor_path(self):
        """
        Obtains path to coadd preset for swarp.

        Returns
        -------
        str
            Path to preset.
        """
        return get_resource_path(package=self._swarp_preset_package, resource="apcor.yml")

    @property
    def weight_images(self):

        # Import
        from vircampype.fits.images.flat import WeightImages

        # Search for weight for each image
        weight_paths = [x.replace(".apcor{0}.".format(d[0]), ".apcor.weight.")
                        for x, d in zip(self.full_paths, self.diameters)]

        # For each file there must be a weight
        if len(self) != np.sum([os.path.exists(x) for x in weight_paths]):
            raise ValueError("Not all images have an associated weight.")

        # Return WeightImages instance
        return WeightImages(file_paths=weight_paths, setup=self.setup)

    def get_apcor(self, skycoo, file_index, hdu_index):
        """
        Fetches aperture correction directly from image

        Parameters
        ----------
        skycoo : SkyCoord
            Input astropy SkyCoord object for which the aperture correction should be obtained.
        file_index : int
            Index of file in self.
        hdu_index : int
            Index of HDU

        Returns
        -------
        ndarray
            Array with aperture corrections.

        """
        return self.get_pixel_value(skycoo=skycoo, file_index=file_index, hdu_index=hdu_index)

    def coadd(self):

        # Processing info
        tstart = message_mastercalibration(master_type="COADDING APERTURE CORRECTION",
                                           silent=self.setup["misc"]["silent"], right=None)

        # Split by aperture diameter
        split_apcor = self.split_keywords(keywords=["APCDIAM"])

        for sidx in range(len(split_apcor)):

            # Get current files
            split = split_apcor[sidx]  # type: ApcorImages

            # Get current diameter
            diameter = split.diameters[0][0]

            # Create output path
            outpath = "{0}{1}{2}".format(split.path_apcor, split.coadd_name, ".sources.apcor{0}.fits".format(diameter))

            # Check if file exists and skip if it does
            if check_file_exists(file_path=outpath, silent=split.setup["misc"]["silent"]) \
                    and not split.setup["misc"]["overwrite"]:
                continue

            # Rename weights for Swarp
            orig_paths_weights = split.weight_images.full_paths.copy()
            new_paths_weights = [p.replace(".apcor.", ".apcor{0}.".format(diameter)) for p in orig_paths_weights]
            for orig, new in zip(orig_paths_weights, new_paths_weights):
                os.rename(orig, new)

            # Print processing info
            message_calibration(n_current=sidx + 1, n_total=len(split_apcor), name=outpath,
                                d_current=None, d_total=None, silent=self.setup["misc"]["silent"])

            # Create output header
            header = resize_header(header=split.header_coadd, factor=self.setup["photometry"]["apcor_image_scale"])

            # Write header to disk
            header.totextfile(outpath.replace(".fits", ".ahead"), overwrite=True, endcard=True)

            # Construct swarp options
            ss = yml2config(path=split._swarp_preset_apcor_path, imageout_name=outpath, weight_type="MAP_WEIGHT",
                            weightout_name=outpath.replace(".fits", ".weight.fits"), resample_dir=self.path_temp,
                            nthreads=split.setup["misc"]["n_jobs"], skip=["weight_thresh", "weight_image"])

            # Construct commands for source extraction
            cmd = "{0} {1} -c {2} {3}".format(split.bin_swarp, " ".join(split.full_paths),
                                              split._swarp_default_config, ss)

            # Run Swarp
            run_command_bash(cmd=cmd, silent=True)

            # Rename weights back to original
            for orig, new in zip(orig_paths_weights, new_paths_weights):
                os.rename(new, orig)

            # Remove header and weight
            remove_file(path=outpath.replace(".fits", ".ahead"))
            # remove_file(path=outpath.replace(".fits", ".weight.fits"))

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    def qc_plot_apc(self, paths=None, axis_size=5):

        # Import
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Generate path for plots
        if paths is None:
            paths = ["{0}{1}.pdf".format(self.path_qc_apcor, fp) for fp in self.file_names]

        for idx_file in range(len(self)):

            # Read focal play array layout
            fpa_layout = str2list(self.setup["data"]["fpa_layout"], dtype=int)

            # Create figure
            fig, ax_file = get_plotgrid(layout=fpa_layout, xsize=axis_size, ysize=axis_size)
            ax_file, cax = ax_file.ravel(), fig.add_axes([0.3, 0.92, 0.4, 0.02])

            # Determine plot color range
            vmin, vmax = np.percentile(self.mag_apc[idx_file], 5), np.percentile(self.mag_apc[idx_file], 95)

            # Minimum range of 1% if computed range is too small
            if vmax - vmin < 0.01:
                diff = 0.01 - (vmax - vmin)
                vmin -= diff / 2
                vmax += diff / 2

            for idx_plot, idx_data, dhdr in \
                    zip(range(len(self.data_hdu[idx_file])), self.data_hdu[idx_file], self.headers_data[idx_file]):

                # Grab axes
                ax = ax_file[idx_plot]

                # Read data
                data = fits.getdata(self.full_paths[idx_file], idx_data, header=False)

                # Read weight
                weight = fits.getdata(self.weight_images.full_paths[idx_file], idx_data, header=False)

                # Mask
                data[weight <= 0.00001] = np.nan

                # Dimensions must match
                if data.shape != weight.shape:
                    raise ValueError("Data and weight shapes do not match")

                # Draw image
                im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap="RdYlBu_r", origin="lower")

                # Add colorbar
                cbar = plt.colorbar(mappable=im, cax=cax, orientation="horizontal", label="Relative Flux")
                cbar.ax.xaxis.set_ticks_position("top")
                cbar.ax.xaxis.set_label_position("top")

                # Limits
                ax.set_xlim(0, dhdr["NAXIS1"] - 1)
                ax.set_ylim(0, dhdr["NAXIS2"] - 1)

                # Annotate detector ID
                ax.annotate("Det.ID: {0:0d}".format(idx_data), xy=(0.02, 1.005),
                            xycoords="axes fraction", ha="left", va="bottom")

                # Annotate number of sources used
                ax.annotate("N = {0:0d}".format(self.n_sources[idx_file][idx_data-1]), xy=(0.98, 1.005),
                            xycoords="axes fraction", ha="right", va="bottom")

                # Modify axes
                if idx_plot < fpa_layout[1]:
                    ax.set_xlabel("X (pix)")
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if idx_plot % fpa_layout[0] == fpa_layout[0] - 1:
                    ax.set_ylabel("Y (pix)")
                else:
                    ax.axes.yaxis.set_ticklabels([])

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


class MasterPSF(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(FitsImages, self).__init__(setup=setup, file_paths=file_paths)

    @property
    def nsources(self):
        return self.dataheaders_get_keys(keywords=["NSOURCES"])[0]

    # =========================================================================== #
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
            return ["{0}{1}.pdf".format(self.path_qc_psf, fp) for fp in self.file_names]
        else:
            return paths

    def qc_plot_psf(self, paths=None, axis_size=4):

        # Import
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
        from matplotlib.colors import PowerNorm
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Generate path for plots
        paths = self.paths_qc_plots(paths=paths)

        for idx_file in range(len(self)):

            # Read focal play array layout
            fpa_layout = str2list(self.setup["data"]["fpa_layout"], dtype=int)

            # Create figure
            fig, ax_file = get_plotgrid(layout=fpa_layout, xsize=axis_size, ysize=axis_size)
            ax_file = ax_file.ravel()
            cax = fig.add_axes([0.3, 0.92, 0.4, 0.02])

            # Read data
            cube = self.file2cube(file_index=idx_file)

            # Normalize cube
            cube.normalize(norm=np.nanmax(cube))

            # Loop over extensions
            for idx_hdu in range(len(self.data_hdu[idx_file])):

                # Fetch current axes
                ax = ax_file[idx_hdu]

                # Draw image
                im = ax.imshow(cube[idx_hdu], cmap=get_cmap("viridis"), origin="lower",
                               norm=PowerNorm(gamma=0.4, vmin=0, vmax=1))

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
