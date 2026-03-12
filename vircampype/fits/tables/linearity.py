# =========================================================================== #
# Import
import warnings

import numpy as np

from vircampype.fits.tables.common import MasterTables
from vircampype.tools.mathtools import *
from vircampype.tools.plottools import *


class MasterLinearity(MasterTables):
    def __init__(self, file_paths, setup=None):
        super(MasterLinearity, self).__init__(file_paths=file_paths, setup=setup)

    # =========================================================================== #
    # Properties
    # =========================================================================== #
    _is_per_channel = None

    @property
    def is_per_channel(self):
        """Whether this linearity file uses per-channel coefficients."""
        if self._is_per_channel is None:
            try:
                mode = self.headers_data[0][0]["HIERARCH PYPE LINEARITY MODE"]
                self._is_per_channel = mode == "channel"
            except KeyError:
                self._is_per_channel = False
        return self._is_per_channel

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
        self._nl10000 = self.read_from_data_headers(
            keywords=["HIERARCH PYPE QC NL10000"]
        )[0]
        return self._nl10000

    _coeff = None

    @property
    def coeff(self):
        """
        Extracts the linearity coefficients stored in the header for linearizing.

        For per-detector files: coeff[file][detector] = [c0, c1, c2, c3]
        For per-channel files: coeff[file][detector][channel] = [c0, c1, c2, c3]

        Returns
        -------
        list
            Nested list of coefficients.

        """

        # Check if already determined
        if self._coeff is not None:
            return self._coeff

        if self.is_per_channel:
            self._coeff = self._read_channel_coefficients(
                keyword_base="HIERARCH PYPE COEFF LINEAR"
            )
        else:
            self._coeff = self._read_sequence_from_data_headers(
                keyword="HIERARCH PYPE COEFF LINEAR"
            )
        return self._coeff

    _coeff_poly = None

    @property
    def coeff_poly(self):
        """
        Fitting coefficients

        For per-detector files: coeff_poly[file][detector] = [c0, c1, c2, c3]
        For per-channel files: coeff_poly[file][detector][channel] = [c0, c1, c2, c3]

        Returns
        -------
        list
            Nested list of coefficients.

        """

        # Check if already determined
        if self._coeff_poly is not None:
            return self._coeff_poly

        if self.is_per_channel:
            self._coeff_poly = self._read_channel_coefficients(
                keyword_base="HIERARCH PYPE COEFF POLY"
            )
        else:
            self._coeff_poly = self._read_sequence_from_data_headers(
                keyword="HIERARCH PYPE COEFF POLY"
            )
        return self._coeff_poly

    def _read_channel_coefficients(self, keyword_base, n_channels=16):
        """
        Read per-channel coefficient keywords from data headers.

        Keywords are expected in the form '{keyword_base} {channel} {order}'.

        Parameters
        ----------
        keyword_base : str
            Base keyword prefix.
        n_channels : int, optional
            Number of readout channels per detector. Default is 16.

        Returns
        -------
        list
            result[file][detector][channel] = [c0, c1, ...]

        """
        result = []
        for headers_file in self.headers_data:
            file_coeffs = []
            for hdr in headers_file:
                det_coeffs = []
                for ch in range(n_channels):
                    ch_coeffs = []
                    order = 0
                    while True:
                        try:
                            ch_coeffs.append(hdr[f"{keyword_base} {ch} {order}"])
                            order += 1
                        except KeyError:
                            break
                    det_coeffs.append(ch_coeffs)
                file_coeffs.append(det_coeffs)
            result.append(file_coeffs)
        return result

    _linearity_texp = None

    @property
    def linearity_texp(self):
        """
        Extracts all TEXPTIME values measured in the linearity sequence.

        Returns
        -------
        List
            List of TEXPTIMEs.

        """

        # Check if already determined
        if self._linearity_texp is not None:
            return self._linearity_texp

        self._linearity_texp = self.get_columns(column_name="texp")
        return self._linearity_texp

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

        # Need -1 here since the coefficients do not take an empty primary header
        # into account
        if hdu_index - 1 < 0:
            raise ValueError(f"HDU with index {hdu_index - 1} does not exits")

        return [f[hdu_index - 1] for f in self.coeff]

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

        # Need -1 here since the coefficients do not take an empty primary header
        # into account
        return [self.coeff[file_index][idx - 1] for idx in hdu_index]

    # =========================================================================== #
    # Plots
    # =========================================================================== #
    def qc_plot_linearity_detector(self, paths=None, axis_size=5):

        # Generate path for plots
        if paths is None:
            paths = [
                f"{self.setup.folders['qc_linearity']}{fp}_detector.pdf"
                for fp in self.basenames
            ]

        # Loop over files and create plots
        for path, nl in zip(paths, self.nl10000):
            plot_value_detector(
                values=nl,
                path=path,
                ylabel="Non-linearity @10000ADU/TEXP=2 (%)",
                axis_size=axis_size,
                hlines=[0],
                dpi=self.setup.qc_plot_dpi,
            )

    def qc_plot_linearity_fit(self, paths=None, axis_size=4):
        """
        Create the QC plot for the linearity measurements. Should only be used together
        with the above method.

        Parameters
        ----------
        paths : List, optional
            Paths of the QC plot files. If None (default), use relative path
        axis_size : int, float, optional
            Axis size. Default is 5.

        """

        # Import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Generate path for plots
        if paths is None:
            paths = [
                f"{self.setup.folders['qc_linearity']}{fp}_fit.pdf"
                for fp in self.basenames
            ]

        # Loop over files and create plots
        for path, texp, flux, flux_lin, nl10000, coeff, coeff_poly in zip(
            paths,
            self.linearity_texp,
            self.flux,
            self.flux_linearized,
            self.nl10000,
            self.coeff,
            self.coeff_poly,
        ):
            # Get plot grid
            fig, axes = get_plotgrid(
                layout=self.setup.fpa_layout, xsize=axis_size, ysize=axis_size
            )
            axes = axes.ravel()

            # Helpers
            alltexp, allflux = (
                [i for s in texp for i in s],
                [i for s in flux for i in s],
            )
            xmax = 1.05 * np.max(alltexp)
            ymax = 1.10 * np.max(self.setup.saturation_levels)

            # Plot
            for idx in range(len(texp)):
                # Get those above the saturation
                bad = np.array(flux[idx]) > self.setup.saturation_levels[idx]

                # Add axis
                ax = axes[idx]

                # Good raw flux
                ax.scatter(
                    np.array(texp[idx])[~bad],
                    np.array(flux[idx])[~bad],
                    c="#1f77b4",
                    lw=0,
                    s=40,
                    alpha=0.7,
                    zorder=1,
                    rasterized=True,
                )

                # Bad raw flux
                ax.scatter(
                    np.array(texp[idx])[bad],
                    np.array(flux[idx])[bad],
                    lw=1,
                    s=40,
                    facecolors="none",
                    edgecolors="#1f77b4",
                    rasterized=True,
                )

                # Linearized good flux
                ax.scatter(
                    np.array(texp[idx])[~bad],
                    np.array(flux_lin[idx])[~bad],
                    c="#ff7f0e",
                    lw=0,
                    s=40,
                    alpha=0.7,
                    zorder=2,
                    rasterized=True,
                )

                # Fit line (use mean across channels for per-channel mode)
                if self.is_per_channel:
                    mean_poly = np.mean(coeff_poly[idx], axis=0)
                    ax.plot(
                        texp[idx],
                        linearity_fitfunc(texp[idx], *mean_poly[1:]),
                        c="black",
                        lw=0.8,
                        zorder=0,
                    )
                else:
                    ax.plot(
                        texp[idx],
                        linearity_fitfunc(texp[idx], *coeff_poly[idx][1:]),
                        c="black",
                        lw=0.8,
                        zorder=0,
                    )

                # Saturation
                ax.hlines(
                    self.setup.saturation_levels[idx],
                    0,
                    ceil_value(xmax, value=5),
                    linestyles="dashed",
                    colors="#7F7F7F",
                    lw=1,
                )

                # Annotate non-linearity and detector ID
                ax.annotate(
                    f"NL$_{{10000}}$ (TEXP=2s)$={np.round(nl10000[idx], decimals=2)}%",
                    xy=(0.03, 0.96),
                    xycoords="axes fraction",
                    ha="left",
                    va="top",
                )
                ax.annotate(
                    f"Det.ID: {idx + 1:0d}",
                    xy=(0.96, 0.03),
                    xycoords="axes fraction",
                    ha="right",
                    va="bottom",
                )

                # Modify axes
                if idx >= self.setup.fpa_layout[0] * (self.setup.fpa_layout[1] - 1):
                    ax.set_xlabel("TEXP (s)")
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if idx % self.setup.fpa_layout[0] == 0:
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
                warnings.filterwarnings(
                    "ignore", message="tight_layout : falling back to Agg renderer"
                )
                fig.savefig(path, bbox_inches="tight", dpi=self.setup.qc_plot_dpi)
            plt.close("all")

    def qc_plot_linearity_delta(self, paths=None, axis_size=4):

        # Import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Generate path for plots
        if paths is None:
            paths = [
                f"{self.setup.folders['qc_linearity']}{fp}_delta.pdf"
                for fp in self.basenames
            ]

        # Loop over files and create plots
        for idx_file in range(self.n_files):
            # Get plot grid
            fig, axes = get_plotgrid(
                layout=self.setup.fpa_layout, xsize=axis_size, ysize=axis_size
            )
            axes = axes.ravel()

            # Grab variable for current file
            coeff_file = self.coeff[idx_file]
            path = paths[idx_file]

            for idx_hdu in range(len(coeff_file)):
                # Grab stuff for current HDU
                ax = axes[idx_hdu]
                coeff_hdu = coeff_file[idx_hdu]

                # Generate test data
                data = np.linspace(1, 50000, 5000)

                # Sample a few TEXPs
                texps = [2, 5, 10]

                if self.is_per_channel:
                    # Per-channel: draw all channels as thin lines, use mean
                    # coefficients for the labeled thick line
                    mean_coeff = np.mean(coeff_hdu, axis=0).tolist()
                    for ch_coeff in coeff_hdu:
                        for t in texps:
                            dl = linearize_data(
                                data=data,
                                coeff=ch_coeff,
                                texptime=t,
                                reset_read_overhead=1.0011,
                            )
                            ax.plot(
                                data,
                                dl - data,
                                lw=0.3,
                                alpha=0.3,
                                c="gray",
                            )
                    data_lin = [
                        linearize_data(
                            data=data,
                            coeff=mean_coeff,
                            texptime=t,
                            reset_read_overhead=1.0011,
                        )
                        for t in texps
                    ]
                else:
                    data_lin = [
                        linearize_data(
                            data=data,
                            coeff=coeff_hdu,
                            texptime=t,
                            reset_read_overhead=1.0011,
                        )
                        for t in texps
                    ]

                # Draw mean/detector-level curves with labels
                for idx_texp in range(len(texps)):
                    ax.plot(
                        data,
                        data_lin[idx_texp] - data,
                        lw=2,
                        alpha=0.8,
                        label=f"TEXP={texps[idx_texp]}",
                    )

                # Draw saturation level
                ax.axvline(
                    self.setup.saturation_levels[idx_hdu],
                    ls="dashed",
                    c="#7F7F7F",
                    lw=1,
                    zorder=0,
                    label="Saturation",
                )

                # 1:1 line
                ax.axhline(0, c="black", lw=1, zorder=0)

                # Annotate Detector ID
                ax.annotate(
                    f"Det.ID: {idx_hdu + 1:0d}",
                    xy=(0.96, 0.03),
                    xycoords="axes fraction",
                    ha="right",
                    va="bottom",
                )

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
            ax.legend(  # noqa
                loc="lower left",
                bbox_to_anchor=(0.01, 1.02),
                ncol=5,  # noqa
                fancybox=False,
                shadow=False,
                frameon=False,
            )

            # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="tight_layout : falling back to Agg renderer"
                )
                fig.savefig(path, bbox_inches="tight", dpi=self.setup.qc_plot_dpi)
            plt.close("all")
