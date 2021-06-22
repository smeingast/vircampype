# =========================================================================== #
# Import
import warnings
import numpy as np

from typing import List
from vircampype.tools.plottools import *
from vircampype.tools.mathtools import *
from vircampype.fits.tables.common import MasterTables


class MasterLinearity(MasterTables):

    def __init__(self, file_paths, setup=None):
        super(MasterLinearity, self).__init__(file_paths=file_paths, setup=setup)

    # =========================================================================== #
    # Properties
    # =========================================================================== #
    _nl10000 = None

    @property
    def nl10000(self):
        """
        Fetches all NL10000 values from the headers of the Masterlinearity files

        Returns
        -------
        iterable
            List of lists contaiing the NL10000 values for each file and extension

        """

        # Check if already determined
        if self._nl10000 is not None:
            return self._nl10000

        # Retrieve values and return
        self._nl10000 = self.read_from_data_headers(keywords=["HIERARCH PYPE QC NL10000"])[0]
        return self._nl10000

    _coeff = None

    @property
    def coeff(self):
        """
        Extracts the linearity coefficients stored in the header for linearizing.

        Returns
        -------
        List
            List of coefficients

        """

        # Check if already determined
        if self._coeff is not None:
            return self._coeff

        self._coeff = self._read_sequence_from_data_headers(keyword="HIERARCH PYPE COEFF LINEAR")
        return self._coeff

    _coeff_poly = None

    @property
    def coeff_poly(self):
        """
        Fitting coefficients

        Returns
        -------
        List
            List of coefficients

        """

        # Check if already determined
        if self._coeff_poly is not None:
            return self._coeff_poly

        self._coeff_poly = self._read_sequence_from_data_headers(keyword="HIERARCH PYPE COEFF POLY")
        return self._coeff_poly

    _linearity_dit = None

    @property
    def linearity_dit(self):
        """
        Extracts all DIT values measured in the linearity sequence.

        Returns
        -------
        List
            List of DITs.

        """

        # Check if already determined
        if self._linearity_dit is not None:
            return self._linearity_dit

        self._linearity_dit = self.get_columns(column_name="dit")
        return self._linearity_dit

    _flux = None

    @property
    def flux(self):
        """
        Extracts all Flux values measured in the linearity sequence.

        Returns
        -------
        List
            List of Flux.

        """

        # Check if already determined
        if self._flux is not None:
            return self._flux

        self._flux = self.get_columns(column_name="flux")
        return self._flux

    _flux_linearized = None

    @property
    def flux_linearized(self):

        # Check if already determined
        if self._flux_linearized is not None:
            return self._flux_linearized

        self._flux_linearized = self.get_columns(column_name="flux_lin")
        return self._flux_linearized

    # =========================================================================== #
    # I/O
    # =========================================================================== #
    def hdu2coeff(self, hdu_index):
        """

        Parameters
        ----------
        hdu_index : int, optional
            Which HDU to load.

        Returns
        -------
        List
            List of coefficients.

        """

        # Need -1 here since the coefficients do not take an empty primary header into account
        if hdu_index-1 < 0:
            raise ValueError("HDU with index {0} does not exits".format(hdu_index-1))

        return [f[hdu_index-1] for f in self.coeff]

    def file2coeff(self, file_index, hdu_index=None):
        """

        Parameters
        ----------
        file_index : int
            Integer index of the file in the current FitsFiles instance.
        hdu_index : iterable, optional
            Iterable of hdu indices to load, default is to load all HDUs with data.

        Returns
        -------
        List
            List of coefficients.

        """

        if hdu_index is None:
            hdu_index = self.iter_data_hdu[file_index]

        # Need -1 here since the coefficients do not take an empty primary header into account
        return [self.coeff[file_index][idx - 1] for idx in hdu_index]

    # =========================================================================== #
    # Plots
    # =========================================================================== #
    def qc_plot_linearity_detector(self, paths=None, axis_size=5):

        # Generate path for plots
        if paths is None:
            paths = ["{0}{1}_detector.pdf".format(self.setup.folders["qc_linearity"], fp) for fp in self.basenames]

        # Loop over files and create plots
        for path, nl in zip(paths, self.nl10000):
            plot_value_detector(values=nl, path=path, ylabel="Non-linearity @10000ADU/DIT=2 (%)",
                                axis_size=axis_size, hlines=[0])

    def qc_plot_linearity_fit(self, paths=None, axis_size=4):

        """
        Create the QC plot for the linearity measurements. Should only be used together with the above method.

        Parameters
        ----------
        paths : List, optional
            Paths of the QC plot files. If None (default), use relative path
        axis_size : int, float, optional
            Axis size. Default is 5.

        """

        # Import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Generate path for plots
        if paths is None:
            paths = ["{0}{1}_fit.pdf".format(self.setup.folders["qc_linearity"], fp) for fp in self.basenames]

        # Loop over files and create plots
        for path, dit, flux, flux_lin, nl10000, coeff, coeff_poly in \
                zip(paths, self.linearity_dit, self.flux, self.flux_linearized,
                    self.nl10000, self.coeff, self.coeff_poly):

            # Get plot grid
            fig, axes = get_plotgrid(layout=self.setup.fpa_layout, xsize=axis_size, ysize=axis_size)
            axes = axes.ravel()

            # Helpers
            alldit, allflux = [i for s in dit for i in s], [i for s in flux for i in s]
            xmax = 1.05 * np.max(alldit)
            ymax = 1.10 * np.max(self.setup.saturation_levels)

            # Plot
            for idx in range(len(dit)):

                # Get those above the saturation
                bad = np.array(flux[idx]) > self.setup.saturation_levels[idx]

                # Add axis
                ax = axes[idx]

                # Good raw flux
                ax.scatter(np.array(dit[idx])[~bad], np.array(flux[idx])[~bad],
                           c="#1f77b4", lw=0, s=40, alpha=0.7, zorder=1)

                # Bad raw flux
                ax.scatter(np.array(dit[idx])[bad], np.array(flux[idx])[bad],
                           lw=1, s=40, facecolors="none", edgecolors="#1f77b4")

                # Linearized good flux
                ax.scatter(np.array(dit[idx])[~bad], np.array(flux_lin[idx])[~bad],
                           c="#ff7f0e", lw=0, s=40, alpha=0.7, zorder=2)

                # Fit
                ax.plot(dit[idx], linearity_fitfunc(dit[idx], *coeff_poly[idx][1:]), c="black", lw=0.8, zorder=0)

                # Saturation
                ax.hlines(self.setup.saturation_levels[idx], 0, ceil_value(xmax, value=5),
                          linestyles="dashed", colors="#7F7F7F", lw=1)

                # Annotate non-linearity and detector ID
                ax.annotate("NL$_{{10000}}$ (DIT=2s)$=${}%".format(np.round(nl10000[idx], decimals=2)),
                            xy=(0.03, 0.96), xycoords="axes fraction", ha="left", va="top")
                ax.annotate("Det.ID: {0:0d}".format(idx + 1),
                            xy=(0.96, 0.03), xycoords="axes fraction", ha="right", va="bottom")

                # Modify axes
                if idx < self.setup.fpa_layout[1]:
                    ax.set_xlabel("DIT (s)")
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if idx % self.setup.fpa_layout[0] == self.setup.fpa_layout[0] - 1:
                    ax.set_ylabel("ADU")
                else:
                    ax.axes.yaxis.set_ticklabels([])

                # Set ranges
                ax.set_xlim(0, ceil_value(xmax, value=5))
                ax.set_ylim(0, ceil_value(ymax, value=5000))

                # Set ticks
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator())

                # Hide first tick label
                xticks, yticks = ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()
                xticks[0].set_visible(False)
                yticks[0].set_visible(False)

            # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
                fig.savefig(path, bbox_inches="tight")
            plt.close("all")

    def qc_plot_linearity_delta(self, paths=None, axis_size=4):

        # Import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Generate path for plots
        if paths is None:
            paths = ["{0}{1}_delta.pdf".format(self.setup.folders["qc_linearity"], fp) for fp in self.basenames]

        # Loop over files and create plots
        for idx_file in range(self.n_files):

            # Get plot grid
            fig, axes = get_plotgrid(layout=self.setup.fpa_layout, xsize=axis_size, ysize=axis_size)
            axes = axes.ravel()

            # Grab variable for current file
            coeff_file = self.coeff[idx_file]
            path = paths[idx_file]

            for idx_hdu in range(len(coeff_file)):

                # Grab stuff for current HDU
                ax = axes[idx_hdu]
                coeff_hdu = coeff_file[idx_hdu]

                # Generate test data
                # data = np.linspace(1, self.setup.saturation_levels[idx_hdu] + 5000, 150)
                data = np.linspace(1, 50000, 5000)

                # Sample a few DITs
                dits = [2, 5, 10]
                data_lin = [linearize_data(data=data, coeff=coeff_hdu, dit=d, reset_read_overhead=1.0011) for d in dits]

                # Draw
                for idx_dit in range(len(dits)):
                    ax.plot(data, data_lin[idx_dit] - data, lw=2, alpha=0.8,
                            label="DIT={0}".format(dits[idx_dit]))

                # Draw saturation level
                ax.axvline(self.setup.saturation_levels[idx_hdu], ls="dashed", c="#7F7F7F",
                           lw=1, zorder=0, label="Saturation")

                # 1:1 line
                ax.axhline(0, c="black", lw=1, zorder=0)

                # Annotate Detector ID
                ax.annotate("Det.ID: {0:0d}".format(idx_hdu + 1),
                            xy=(0.96, 0.03), xycoords="axes fraction", ha="right", va="bottom")

                # Modify axes
                if idx_hdu < self.setup.fpa_layout[1]:
                    ax.set_xlabel("Data input (ADU)")
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if idx_hdu % self.setup.fpa_layout[0] == self.setup.fpa_layout[0] - 1:
                    ax.set_ylabel(r"Linearized - Input (ADU)")
                else:
                    ax.axes.yaxis.set_ticklabels([])

                # Set ranges
                ax.set_xlim(1, 50000)
                ax.set_ylim(1, 50000)

                # Logscale
                ax.set_yscale("log")

                # Set ticks
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                # ax.yaxis.set_major_locator(MaxNLocator(5))
                # ax.yaxis.set_minor_locator(AutoMinorLocator())

                # Hide first tick label
                xticks, yticks = ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()
                xticks[0].set_visible(False)
                yticks[0].set_visible(False)

            # Set label on last iteration
            ax.legend(loc="lower left", bbox_to_anchor=(0.01, 1.02), ncol=5,  # noqa
                      fancybox=False, shadow=False, frameon=False)

            # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
                fig.savefig(path, bbox_inches="tight")
            plt.close("all")
