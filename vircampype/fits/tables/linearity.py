# =========================================================================== #
# Import
import warnings
import numpy as np

from typing import List
from vircampype.utils.plots import get_plotgrid
from vircampype.fits.tables.common import MasterTables
from vircampype.utils.math import linearize_data, ceil_value
from vircampype.utils.miscellaneous import check_file_exists


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
        self._nl10000 = self.dataheaders_get_keys(keywords=["HIERARCH PYPE QC NL10000"])[0]
        return self._nl10000

    _coeff_poly = None

    @property
    def coeff_poly(self):
        """
        Extracts the linearity coefficients stored in the header for plotting.

        Returns
        -------
        List
            List of coefficients

        """

        # Check if already determined
        if self._coeff_poly is not None:
            return self._coeff_poly

        temp = []
        for idx in range(self.setup["linearity"]["order"] + 1):
            temp.append(self.dataheaders_get_keys(keywords=["HIERARCH PYPE COEFF POLY {0}".format(idx)])[0])

        temp = np.rollaxis(np.array(temp), axis=1)
        self._coeff_poly = [t.T.tolist() for t in temp]
        return self._coeff_poly

    _coeff_linear = None

    @property
    def coeff_linear(self):
        """
        Extracts the linearity coefficients stored in the header for linearizing.

        Returns
        -------
        List
            List of coefficients

        """

        # Check if already determined
        if self._coeff_linear is not None:
            return self._coeff_linear

        temp = []
        for idx in range(self.setup["linearity"]["order"] + 1):
            temp.append(self.dataheaders_get_keys(keywords=["HIERARCH PYPE COEFF LINEAR {0}".format(idx)])[0])

        temp = np.rollaxis(np.array(temp), axis=1)
        self._coeff_linear = [t.T.tolist() for t in temp]
        return self._coeff_linear

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

    _linearity_flux = None

    @property
    def linearity_flux(self):
        """
        Extracts all Flux values measured in the linearity sequence.

        Returns
        -------
        List
            List of Flux.

        """

        # Check if already determined
        if self._linearity_flux is not None:
            return self._linearity_flux

        self._linearity_flux = self.get_columns(column_name="flux")
        return self._linearity_flux

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

        return [f[hdu_index-1] for f in self.coeff_linear]

    # =========================================================================== #
    # Plots
    # =========================================================================== #
    def qc_plot_linearity(self, paths=None, axis_size=4, overwrite=False):

        """
        Create the QC plot for the linearity measurements. Should only be used together with the above method.

        Parameters
        ----------
        paths : List, optional
            Paths of the QC plot files. If None (default), use relative path
        axis_size : int, float, optional
            Axis size. Default is 5.
        overwrite : optional, bool
            Whether an exisiting plot should be overwritten. Default is False.

        """

        # Import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Generate path for plots
        if paths is None:
            paths = [x.replace(".tab", ".pdf") for x in self.full_paths]

        # Loop over files and create plots
        for dit, flux, path, pcff, lcff, nl10k in zip(self.linearity_dit, self.linearity_flux, paths,
                                                      self.coeff_poly, self.coeff_linear, self.nl10000):

            # Check if plot already exits
            if check_file_exists(file_path=path, silent=True) and not overwrite:
                continue

            # Get plot grid
            fig, axes = get_plotgrid(layout=self.setup["instrument"]["layout"], xsize=axis_size, ysize=axis_size)
            axes = axes.ravel()

            # Helpers
            alldit, allflux = [i for s in dit for i in s], [i for s in flux for i in s]
            xmax = 1.05 * np.max(alldit)
            ymax = 1.10 * np.max(self.setup["data"]["saturate"])

            # Plot
            for idx in range(len(dit)):

                # Get those above the saturation
                bad = np.array(flux[idx]) > self.setup["data"]["saturate"][idx]

                # Add axis
                ax = axes[idx]

                # Polynomial fit
                xdummy = np.arange(start=0, step=0.5, stop=xmax)
                ax.plot(xdummy, np.polyval(pcff[idx], xdummy), color="#7F7F7F", lw=2, zorder=0)

                # Good raw flux
                ax.scatter(np.array(dit[idx])[~bad], np.array(flux[idx])[~bad],
                           c="#1f77b4", lw=0, s=40, alpha=0.7, zorder=1)

                # Bad raw flux
                ax.scatter(np.array(dit[idx])[bad], np.array(flux[idx])[bad],
                           lw=1, s=40, facecolors="none", edgecolors="#1f77b4")

                # Linearized good flux
                lin = linearize_data(data=np.array(flux[idx])[~bad], coeff=lcff[idx])
                ax.scatter(np.array(dit[idx])[~bad], lin, c="#ff7f0e", lw=0, s=40, alpha=0.7, zorder=2)

                # Saturation
                ax.hlines(self.setup["data"]["saturate"][idx], 0,
                          ceil_value(xmax, value=5), linestyles="dashed", colors="#7F7F7F", lw=1)

                # Annotate non-linearity and detector ID
                ax.annotate("NL$_{{10000}}=${}%".format(np.round(nl10k[idx], decimals=2)),
                            xy=(0.03, 0.96), xycoords="axes fraction", ha="left", va="top")
                ax.annotate("Det.ID: {0:0d}".format(idx + 1),
                            xy=(0.96, 0.03), xycoords="axes fraction", ha="right", va="bottom")

                # Modify axes
                if idx >= len(dit) - self.setup["instrument"]["layout"][0]:
                    ax.set_xlabel("DIT (s)")
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if idx % self.setup["instrument"]["layout"][0] == 0:
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
