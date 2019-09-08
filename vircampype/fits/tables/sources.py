# =========================================================================== #
# Import
import warnings
import numpy as np

from vircampype.utils import *
from vircampype.fits.tables.common import FitsTables


class SourceCatalogs(FitsTables):

    def __init__(self, setup, file_paths=None):
        super(SourceCatalogs, self).__init__(file_paths=file_paths, setup=setup)

    # =========================================================================== #
    # Properties
    # =========================================================================== #
    @property
    def _key_ra(self):
        """
        Keyword for right ascension in tables.

        Returns
        -------
        str

        """
        return "RA"

    @property
    def _key_dec(self):
        """
        Keyword for Declination in tables.

        Returns
        -------
        str

        """
        return "DEC"

    _ra = None

    def ra(self, key=None):
        """
        Extracts all RA entries in tables.

        Parameters
        ----------
        key : str, optional
            Used to override default key for RA

        Returns
        -------
        iterable
            List of lists containing RAs

        """

        # Check if already determined
        if self._ra is not None:
            return self._ra

        # Override RA key
        kra = key if key is not None else self._key_ra

        # Retun columns
        self._ra = [[y for y in x] for x in self.get_columns(column_name=kra)]
        return self._ra

    _dec = None

    def dec(self, key=None):
        """
        Extracts all RA entries in tables.

        Parameters
        ----------
        key : str, optional
            Used to override default key for RA

        Returns
        -------
        iterable
            List of lists containing RAs

        """

        # Check if already determined
        if self._dec is not None:
            return self._dec

        # Override DEC key
        kdec = key if key is not None else self._key_dec

        # Get data from columns
        self._dec = [[y for y in x] for x in self.get_columns(column_name=kdec)]
        return self._dec

    def skycoord(self, key_ra=None, key_dec=None):
        """
        Constructs SkyCoord object from ra/dec
        Parameters
        ----------
        key_ra : str, optional
            Key for RA in table.
        key_dec : str, optional
            Key for DEC in table.

        Returns
        -------
        iterable
            List of lists holding SkyCoord objects

        """

        from astropy.coordinates import SkyCoord

        skycoord_files = []
        for fra, fdec in zip(self.ra(key=key_ra), self.dec(key=key_dec)):
            skycoord_ext = []
            for ra, dec in zip(fra, fdec):
                skycoord_ext.append(SkyCoord(ra=ra, dec=dec, frame="icrs", unit="deg"))
            skycoord_files.append(skycoord_ext)

        return skycoord_files

    def mag(self, key):
        """
        Returns magnitude in catalog based on keyword.

        Parameters
        ----------
        key : str
            Key in catalog for magnitude.

        Returns
        -------
        iterable
            List of lists for each catalog and extension in self.

        """
        return [[y for y in x] for x in self.get_columns(column_name=key)]

    @property
    def filters(self):
        """
        Grabs filter keyword from header and puts in into list

        Returns
        -------
        iterable
            List of filters for all tables in instance.

        """
        return self.primeheaders_get_keys(keywords=[self.setup["keywords"]["filter"]])[0]

    # =========================================================================== #
    # Plots
    # =========================================================================== #
    def plot_qc_astrometry(self, axis_size=5, key_x="XWIN_IMAGE", key_y="YWIN_IMAGE", key_ra=None, key_dec=None):

        # Import
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Obtain master coordinates
        sc_master_astrometry = self.get_master_photometry().skycoord()[0][0]

        # Loop over files
        for sc_files, x_files, y_files, name in zip(self.skycoord(key_ra=key_ra, key_dec=key_dec),
                                                    self.get_columns(column_name=key_x),
                                                    self.get_columns(column_name=key_y), self.file_names):

            fig, ax_all = get_plotgrid(layout=self.setup["instrument"]["layout"], xsize=axis_size, ysize=axis_size)
            ax_all = ax_all.ravel()
            cax = fig.add_axes([0.3, 0.92, 0.4, 0.02])

            # Loop over extensions
            im = None
            for idx in range(len(sc_files)):

                # Get separations between master and current table
                i1, sep, _ = sc_files[idx].match_to_catalog_sky(sc_master_astrometry)

                # Extract position angles between master catalog and input
                # sc1 = sc_master_astrometry[i1]
                # ang = sc1.position_angle(sc_hdu)

                # Keep only those with a maximum of 0.5 arcsec
                keep = sep.arcsec < 0.5
                sep, x_hdu, y_hdu = sep[keep], x_files[idx][keep], y_files[idx][keep]
                # ang = ang[keep]

                hist_num, xedges, yedges = np.histogram2d(x_hdu, y_hdu, bins=(5, 5), weights=None, normed=False)
                hist_sep, *_ = np.histogram2d(x_hdu, y_hdu, bins=(5, 5), weights=sep.arcsec, normed=False)
                extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

                im = ax_all[idx].imshow(hist_sep / hist_num, vmin=0, vmax=0.5, extent=extent, cmap="Spectral_r")

                # Show average offset angle
                # hist_sep, *_ = np.histogram2d(x_hdu, y_hdu, bins=(5, 5), weights=ang.degree, normed=False)
                # im = ax.imshow(hist_sep / hist_num, vmin=0, vmax=360, cmap="jet")

                # Annotate detector ID
                ax_all[idx].annotate("Det.ID: {0:0d}".format(idx + 1), xy=(0.02, 1.01),
                                     xycoords="axes fraction", ha="left", va="bottom")

                # Modify axes
                if idx >= len(sc_files) - self.setup["instrument"]["layout"][0]:
                    ax_all[idx].set_xlabel("X (pix)")
                else:
                    ax_all[idx].axes.xaxis.set_ticklabels([])
                if idx % self.setup["instrument"]["layout"][0] == 0:
                    ax_all[idx].set_ylabel("Y(pix)")
                else:
                    ax_all[idx].axes.yaxis.set_ticklabels([])

                ax_all[idx].set_aspect("equal")

                # Set ticks
                ax_all[idx].xaxis.set_major_locator(MaxNLocator(5))
                ax_all[idx].xaxis.set_minor_locator(AutoMinorLocator())
                ax_all[idx].yaxis.set_major_locator(MaxNLocator(5))
                ax_all[idx].yaxis.set_minor_locator(AutoMinorLocator())

            # Add colorbar
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal", label="Average separation (arcsec)")
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")

            # Save plot
            with warnings.catch_warnings():
                path = "{0}{1}_astrometry.pdf".format(self.path_qc, name)
                warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
                fig.savefig(path, bbox_inches="tight")
            plt.close("all")


class ESOSourceCatalogs(SourceCatalogs):

    def __init__(self, setup, file_paths=None):
        super(ESOSourceCatalogs, self).__init__(file_paths=file_paths, setup=setup)


class MasterPhotometry(SourceCatalogs):

    def __init__(self, setup, file_paths=None):
        super(MasterPhotometry, self).__init__(file_paths=file_paths, setup=setup)

    @property
    def mag_lim(self):
        """
        Fetches magnitude limits in setup.

        Returns
        -------
        iterable
            List with magnitude limits.
        """
        return str2list(s=self.setup["photometry"]["mag_limits_ref"], sep=",", dtype=float)


class MasterPhotometry2Mass(MasterPhotometry):

    def __init__(self, setup, file_paths=None):
        super(MasterPhotometry2Mass, self).__init__(file_paths=file_paths, setup=setup)

    # =========================================================================== #
    # Coordinates
    # =========================================================================== #
    @property
    def _key_ra(self):
        return "RAJ2000"

    @property
    def _key_dec(self):
        return "DEJ2000"

    @staticmethod
    def translate_filter(key):
        """
        Translates input filter from e.g. VISTA to 2MASS names.

        Parameters
        ----------
        key : str
            The input magnitude name. e.g. 'J'.

        Returns
        -------
        str
            Translated magnitude key. e.g. 'Ks' will be translated to 'Kmag'.
        """

        if "j" in key.lower():
            return "Jmag"
        elif "h" in key.lower():
            return "Hmag"
        elif "k" in key.lower():
            return "Kmag"
        else:
            raise ValueError("Filter '{0}' not defined".format(key))
