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

    def ra_file(self, idx_file, key=None):
        """
        Extract RA from a given file.

        Parameters
        ----------
        idx_file : int
            Index of file
        key: str, optional
            Column name of RA.

        Returns
        -------
        iterable
            List of RAs for each extension

        """
        # Override RA key
        kra = key if key is not None else self._key_ra

        # Return for given file
        return self.get_column_file(idx_file=idx_file, column_name=kra)

    def dec_file(self, idx_file, key=None):
        """
        Extract DEC from a given file.

        Parameters
        ----------
        idx_file : int
            Index of file
        key: str, optional
            Column name of RA.

        Returns
        -------
        iterable
            List of DECs for each extension

        """
        # Override RA key
        kdec = key if key is not None else self._key_dec

        # Return for given file
        return self.get_column_file(idx_file=idx_file, column_name=kdec)

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

    def skycoord_file(self, idx_file, key_ra=None, key_dec=None):
        """
        Constructs SkyCoord object from ra/dec for a given file.
        Parameters
        ----------
        idx_file : int
            Index of file
        key_ra : str, optional
            Key for RA in table.
        key_dec : str, optional
            Key for DEC in table.

        Returns
        -------
        iterable
            List of Skycoords for each extension of the given catalog.

        """

        # Import
        from astropy.coordinates import SkyCoord

        if key_ra is None:
            key_ra = self._key_ra
        if key_dec is None:
            key_dec = self._key_dec

        skycoord = []
        for ra, dec in zip(*self.get_columns_file(idx_file=idx_file, column_names=[key_ra, key_dec])):
            skycoord.append(SkyCoord(ra=ra, dec=dec, frame="icrs", unit="deg"))

        return skycoord

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
    def filter(self):
        """
        Grabs filter keyword from header and puts in into list.

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

        # Processing info
        tstart = message_mastercalibration(master_type="QC ASTROMETRY", right=None, silent=self.setup["misc"]["silent"])

        # Obtain master coordinates
        sc_master_astrometry = self.get_master_photometry().skycoord()[0][0]

        # Loop over files
        for idx_file in range(len(self)):

            # Generate outpath
            outpath = "{0}{1}_astrometry.pdf".format(self.path_qc_astrometry, self.file_names[idx_file])

            # Check if file exists
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]):
                continue

            # Grab coordinates
            xx_file = self.get_column_file(idx_file=idx_file, column_name=key_x)
            yy_file = self.get_column_file(idx_file=idx_file, column_name=key_y)
            sc_file = self.skycoord_file(idx_file=idx_file, key_ra=key_ra, key_dec=key_dec)

            # Coadd mode
            if len(self) == 1:
                fig, ax_all = get_plotgrid(layout=(1, 1), xsize=2*axis_size, ysize=2*axis_size)
                ax_all = [ax_all]
            else:
                fig, ax_all = get_plotgrid(layout=self.setup["instrument"]["layout"], xsize=axis_size, ysize=axis_size)
                ax_all = ax_all.ravel()
            cax = fig.add_axes([0.3, 0.92, 0.4, 0.02])

            # Loop over extensions
            im, sep_all = None, []
            for idx_hdu in range(len(sc_file)):

                # Print processing info
                message_calibration(n_current=idx_file+1, n_total=len(self), name=outpath, d_current=idx_hdu+1,
                                    d_total=len(sc_file), silent=self.setup["misc"]["silent"])

                # Get separations between master and current table
                i1, sep, _ = sc_file[idx_hdu].match_to_catalog_sky(sc_master_astrometry)

                # Extract position angles between master catalog and input
                # sc1 = sc_master_astrometry[i1]
                # ang = sc1.position_angle(sc_hdu)

                # Keep only those with a maximum of 0.5 arcsec
                keep = sep.arcsec < 0.5
                sep, x_hdu, y_hdu = sep[keep], xx_file[idx_hdu][keep], yy_file[idx_hdu][keep]

                # Grid value into image
                grid = grid_value_2d(x=x_hdu, y=y_hdu, value=sep.arcsec, naxis1=500, naxis2=500,
                                     nbins_x=10, nbins_y=10, conv=True, kernel_scale=0.15)

                # Append separations in arcsec
                sep_all.append(sep.arcsec)

                # Draw
                kwargs = {"vmin": 0, "vmax": 0.5, "cmap": "Spectral_r"}
                extent = [np.nanmin(x_hdu), np.nanmax(x_hdu), np.nanmin(y_hdu), np.nanmax(y_hdu)]
                im = ax_all[idx_hdu].imshow(grid, extent=extent, origin="lower", **kwargs)
                ax_all[idx_hdu].scatter(x_hdu, y_hdu, c=sep.arcsec, s=7, lw=0.5, ec="black", **kwargs)

                # Annotate detector ID
                ax_all[idx_hdu].annotate("Det.ID: {0:0d}".format(idx_hdu + 1), xy=(0.02, 1.01),
                                         xycoords="axes fraction", ha="left", va="bottom")

                # Modify axes
                if idx_hdu >= len(sc_file) - self.setup["instrument"]["layout"][0]:
                    ax_all[idx_hdu].set_xlabel("X (pix)")
                else:
                    ax_all[idx_hdu].axes.xaxis.set_ticklabels([])
                if idx_hdu % self.setup["instrument"]["layout"][0] == 0:
                    ax_all[idx_hdu].set_ylabel("Y (pix)")
                else:
                    ax_all[idx_hdu].axes.yaxis.set_ticklabels([])

                ax_all[idx_hdu].set_aspect("equal")

                # Set ticks
                ax_all[idx_hdu].xaxis.set_major_locator(MaxNLocator(5))
                ax_all[idx_hdu].xaxis.set_minor_locator(AutoMinorLocator())
                ax_all[idx_hdu].yaxis.set_major_locator(MaxNLocator(5))
                ax_all[idx_hdu].yaxis.set_minor_locator(AutoMinorLocator())

                # Left limit
                ax_all[idx_hdu].set_xlim(extent[0], extent[1])
                ax_all[idx_hdu].set_ylim(extent[2], extent[3])

            # Add colorbar
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal", label="Average separation (arcsec)")
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")

            # Print external error stats
            message_qc_astrometry(separation=flat_list(sep_all))

            # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
                fig.savefig(outpath, bbox_inches="tight")
            plt.close("all")

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])


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

    @staticmethod
    def _key2idx(key):
        """ Helper to return filter index for flags. """
        if "j" in key.lower():
            return 0
        elif "h" in key.lower():
            return 1
        elif "k" in key.lower():
            return 2
        else:
            raise ValueError("Filter '{0}' not defined".format(key))

    __qflag = None

    @property
    def _qflags(self):
        """ Reads quality flag column from catalog. """

        if self.__qflag is not None:
            return self.__qflag

        self.__qflag = self.get_columns(column_name="Qflg")
        return self.__qflag

    def qflags(self, key):
        """ Return Quality flag for given filter. """
        return [[[x[self._key2idx(key=key)] for x in y] for y in z] for z in self._qflags]

    __cflag = None

    @property
    def _cflags(self):
        """ Reads contamintation and confusion flag column from catalog. """

        if self.__cflag is not None:
            return self.__cflag

        self.__cflag = self.get_columns(column_name="Cflg")
        return self.__cflag

    def cflags(self, key):
        """ Return contamination and confusion flag for given filter. """
        return [[[x[self._key2idx(key=key)] for x in y] for y in z] for z in self._cflags]
