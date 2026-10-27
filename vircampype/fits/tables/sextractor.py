import json
import logging
import os
import pickle
import time
import warnings
from itertools import repeat

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.table import Column, Table
from astropy.table import hstack as thstack
from astropy.table import vstack as tvstack
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from joblib import Parallel, delayed
from scipy.ndimage import median as ndimage_median
from sklearn.neighbors import KernelDensity, NearestNeighbors

from vircampype.data.cube import ImageCube
from vircampype.fits.tables.sources import SourceCatalogs
from vircampype.pipeline.errors import PipelineFileNotFoundError, PipelineValueError
from vircampype.tools.astromatic import *
from vircampype.tools.fitstools import *
from vircampype.tools.imagetools import *
from vircampype.tools.mathtools import *
from vircampype.tools.messaging import *
from vircampype.tools.messaging import message_qc_astrometry
from vircampype.tools.miscellaneous import *
from vircampype.tools.photometry import *
from vircampype.tools.photometry import _bright_point_sources
from vircampype.tools.plottools import *
from vircampype.tools.productmeta import set_cards
from vircampype.tools.systemtools import *
from vircampype.tools.tabletools import *
from vircampype.tools.wcstools import astrometric_residual_rms


def _sample_statistics_grid(
    xx: np.ndarray,
    yy: np.ndarray,
    mjdeff: np.ndarray,
    exptime: np.ndarray,
    nimg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample the statistics images at the nearest covered pixel of each source.

    ``xx``/``yy`` are 0-based pixel coordinates on the statistics grid; all three
    images must be on that grid. The coarse grid can cut a coverage edge short by up
    to a pixel, so a source whose nearest pixel is uncovered (``nimg < 1``) or just
    off the grid takes the nearest covered pixel within one pixel. Sources without
    one are bad: NaN MJD, zero EXPTIME and NIMG. Holes below the grid's pixel scale
    are left to SExtractor FLAGS_WEIGHT. Returns (mjdeff, exptime, nimg, off_grid,
    bad), off_grid for the nearest pixel.
    """
    xx, yy = np.asarray(xx, dtype=float), np.asarray(yy, dtype=float)
    ny, nx = nimg.shape
    ix = np.floor(xx + 0.5).astype(int)
    iy = np.floor(yy + 0.5).astype(int)
    off = (ix < 0) | (ix >= nx) | (iy < 0) | (iy >= ny)

    # Nearest covered pixel of the 3x3 neighbourhood (the centre wins if covered)
    best = np.full(xx.shape, np.inf)
    px, py = np.zeros_like(ix), np.zeros_like(iy)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = ix + dx, iy + dy
            inside = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
            covered = np.zeros(xx.shape, dtype=bool)
            covered[inside] = nimg[cy[inside], cx[inside]] >= 1
            dist = (cx - xx) ** 2 + (cy - yy) ** 2
            take = covered & (dist < best)
            best[take], px[take], py[take] = dist[take], cx[take], cy[take]
    bad = ~np.isfinite(best)

    mjdeff_s = mjdeff[py, px].astype(float)
    exptime_s = exptime[py, px].astype(float)
    nimg_s = nimg[py, px].astype(float)
    mjdeff_s[bad], exptime_s[bad], nimg_s[bad] = np.nan, 0.0, 0.0
    return mjdeff_s, exptime_s, nimg_s, off, bad


def _strip_flxscale_from_ahead(path_ahead: str) -> None:
    """Remove SCAMP's FLXSCALE=0 cards (SOLVE_PHOTOM N): SWarp's default
    FSCALE_KEYWORD is FLXSCALE and would multiply images by zero."""
    headers = read_aheaders(path_ahead)
    for hdr in headers:
        hdr.remove("FLXSCALE", ignore_missing=True, remove_all=True)
    write_aheaders(headers, path_ahead)


def _neighbor_chunks(
    data_clean: Column,
    err_clean: np.ndarray,
    nn_idx_chunks: list[np.ndarray],
    nn_dis_chunks: list[np.ndarray],
    kernel_std_pix: float,
):
    """
    Neighbour data, weights and distances of one table chunk at a time.

    The data and the weights are gathered per chunk from the clean-table rows of the
    chunk's neighbour indices. Each chunk is element- and type-identical to gathering
    them for all sources at once and splitting, without the full-size N x 100 (x 16)
    temporaries (about 9 kB per source; 64 GB for a mosaic with 5.5 M sources).

    Parameters
    ----------
    data_clean : Column
        Clean-table column to interpolate (1-D, or 2-D for MAG_APER_COR).
    err_clean : np.ndarray
        MAGERR_AUTO of the clean table.
    nn_idx_chunks : list[np.ndarray]
        Nearest-neighbour index matrix, split at the table chunk boundaries.
    nn_dis_chunks : list[np.ndarray]
        Nearest-neighbour distance matrix, split at the same boundaries.
    kernel_std_pix : float
        Gaussian kernel width in pixels.

    Yields
    ------
    tuple[Column, np.ndarray, np.ndarray]
        Neighbour data (n, 100[, 16]), weights (n, 100) and distances (n, 100).
    """
    for idx, dis in zip(nn_idx_chunks, nn_dis_chunks):
        # Distance kernel times the neighbours' S/N (1/MAGERR rather than
        # 1/MAGERR**2 so that the few brightest stars do not dominate)
        err = err_clean[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.exp(-0.5 * (dis / kernel_std_pix) ** 2) / err
        weights[(err <= 0) | ~np.isfinite(weights)] = 0
        yield data_clean[idx], weights, dis


class SextractorCatalogs(SourceCatalogs):
    def __init__(self, setup, file_paths=None, **kwargs):
        super(SextractorCatalogs, self).__init__(
            file_paths=file_paths, setup=setup, **kwargs
        )

    @property
    def iter_data_hdu(self):
        """
        Overrides the normal table data_hdu property.

        Returns
        -------
        iterable
            List of iterators for header indices of HDUs which hold data.
        """
        return [range(2, len(hdrs), 2) for hdrs in self.headers]

    def read_from_image_headers(self, keywords, file_index=None):
        if not isinstance(keywords, list):
            raise TypeError("Keywords must be in a list!")

        if file_index is None:
            headers_image = self.image_headers[:]
        else:
            headers_image = [self.image_headers[file_index]]

        # Return values
        return [[[e[k] for e in h] for h in headers_image] for k in keywords]

    # =========================================================================== #
    # Scamp
    # =========================================================================== #
    def _scamp_header_paths(self, joined=False):
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

        names = [x.replace(".scamp.fits.tab", ".ahead") for x in self.paths_full]
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
        return " ".join(self.paths_full)

    def scamp(self):
        # Fetch log
        log = logging.getLogger(__name__)

        # Processing info
        print_header(header="SCAMP", silent=self.setup.silent)
        tstart = time.time()
        log.info(f"Running scamp on {self.n_files} files:\n{self.basenames2log}")

        # Print source count in eachdata extension to log
        for idx, fn in enumerate(self.basenames):
            log.info(f"File: {fn}")
            tables = self.file2table(file_index=idx)
            for hdu, tt in zip(self.iter_data_hdu[idx], tables):
                log.info(
                    f"Extension {hdu}: {len(tt)}/{len(tt[tt['FLAGS'] == 0])} "
                    f"total/FLAGS=0 sources"
                )

        # Construct XML path
        path_xml = f"{self.setup.folders['qc_astrometry']}scamp.xml"

        # Restore cached .ahead files if available
        ahead_paths = self._scamp_header_paths(joined=False)
        if self.setup.scamp_cache_dir is not None:
            n_restored = 0
            for ap in ahead_paths:
                cached = os.path.join(self.setup.scamp_cache_dir, os.path.basename(ap))
                if os.path.isfile(cached) and not os.path.isfile(ap):
                    rsync_file(cached, ap)
                    log.info(f"Restored cached ahead file: {os.path.basename(ap)}")
                    n_restored += 1
            if n_restored > 0:
                log.info(
                    f"Restored {n_restored} cached scamp headers "
                    f"from {self.setup.scamp_cache_dir}"
                )

        # Check for external headers
        ehdrs = [os.path.isfile(p) for p in ahead_paths]

        # If already available, skip
        if np.sum(ehdrs) == len(self):
            # Headers from pre-strip runs may still carry FLXSCALE=0
            for ap in ahead_paths:
                _strip_flxscale_from_ahead(ap)
            log.info("All scamp .ahead headers already present, skipping")
            print_elapsed(tstart, logger=log)
            return

        # Load astrometric reference
        master_astrometry = self.get_master_astrometry()
        astrefact_name = master_astrometry.paths_full[0]

        # Add to log
        log.info(f"Astrometric reference: {astrefact_name}")

        # Stored EPOCH must match exposures mean epoch.
        if self.setup.warp_gaia:
            mjd = self.read_from_image_headers([self.setup.keywords.date_mjd])
            data_epoch = Time(np.mean(mjd[0]), format="mjd").decimalyear
            if abs(master_astrometry.epoch - data_epoch) > 0.01:
                raise PipelineValueError(
                    f"MASTER-ASTROMETRY epoch ({master_astrometry.epoch:.4f} yr) does "
                    f"not match exposures epoch ({data_epoch:.4f} yr)",
                    logger=log,
                )

        # Load Scamp setup
        scs = ScampSetup(setup=self.setup)

        # Load preset
        options = yml2config(
            path_yml=scs.preset_config,
            nthreads=self.setup.n_jobs_scamp,
            checkplot_type=scs.qc_types(joined=True),
            checkplot_name=scs.qc_names(joined=True),
            skip=[
                "HEADER_NAME",
                "AHEADER_NAME",
                "ASTREF_CATALOG",
                "ASTREFCAT_NAME",
                "ASTREFCENT_KEYS",
                "ASTREFERR_KEYS",
                "ASTREFPROP_KEYS",
                "ASTREFPROPERR_KEYS",
                "ASTREFMAG_KEY",
                "ASTREFMAGERR_KEY",
                "ASTREFOBSDATE_KEY",
                "FLAGS_MASK",
                "WEIGHTFLAGS_MASK",
                "ASTR_FLAGSMASK",
                "XML_NAME",
            ],
        )

        # Construct command for scamp
        cmd = (
            f"{scs.bin} {self._scamp_catalog_paths} "
            f"-c {scs.default_config} "
            f"-HEADER_NAME {self._scamp_header_paths(joined=True)} "
            f"-ASTREF_CATALOG FILE "
            f"-ASTREFCAT_NAME {astrefact_name} "
            f"-ASTREFCENT_KEYS ra,dec "
            f"-ASTREFERR_KEYS ra_error,dec_error "
            f"-ASTREFPROP_KEYS pmra,pmdec "
            f"-ASTREFPROPERR_KEYS pmra_error,pmdec_error "
            f"-ASTREFMAG_KEY mag "
            f"-ASTREFMAGERR_KEY mag_error "
            f"-ASTREFOBSDATE_KEY obsdate "
            f"-XML_NAME {path_xml} {options}"
        )

        # Add scamp command to log
        log.info(f"Scamp command: {cmd}")

        # At most two SCAMP runs per machine (parallel tiles queue here)
        log.info("Waiting for a free scamp slot (max 2 running)...")
        wait_for_process_slot(executable="scamp", max_running=2, timeout_s=7200)

        # Run Scamp
        stdout, stderr = run_command_shell(cmd, silent=False, label="SCAMP")

        # Add stdout and stderr to log
        log.info(f"Scamp stdout:\n{stdout}")
        log.info(f"Scamp stderr:\n{stderr}")

        # Some basic QC on XML
        xml = Table.read(path_xml, format="votable", table_id=1)
        if np.max(xml["AstromSigma_Internal"].data.ravel() * 1000) > 100:
            # Remove the .ahead SCAMP just wrote, so a rerun re-solves instead of
            # silently reusing this rejected solution.
            for ap in ahead_paths:
                remove_file(ap)
            raise ValueError("Astrometric solution may be crap, please check")

        # Inject the group-level RA/Dec astrometric correlation into the .ahead headers.
        # SCAMP reports this cross-axis correlation only in scamp.xml, never in the
        # headers; writing it as ASTCORR lets it ride the solution (resampling
        # COPY_KEYWORDS propagates it into the resampled-pawprint headers, where
        # build_statistics_tile reads it as the flat-SCAMP correlation fallback).
        radec_correlation = scamp_xml_radec_correlation(path_xml)
        log.info(
            f"SCAMP RA/Dec astrometric correlation r_eff = {radec_correlation:.4f}"
        )

        # High-S/N external (vs-reference) astrometric RMS per axis (degrees). Used as
        # the per-source systematic floor in build_statistics. SCAMP reports this only in
        # scamp.xml; the per-detector header ASTRRMS1/2 is the full-sample RMS, which is
        # dominated by faint calibrators and over-estimates bright-source errors. One
        # per-pointing pair is written to every detector header. None if unavailable
        # (older SCAMP / cache-skipped run) -> build_statistics falls back to ASTRRMS1/2.
        astrrms_hsn = scamp_xml_reference_rms_highsn(path_xml)
        if astrrms_hsn is not None:
            log.info(
                "SCAMP high-S/N external RMS = "
                f"{astrrms_hsn[0] * 3_600_000:.1f}/"
                f"{astrrms_hsn[1] * 3_600_000:.1f} mas"
            )

        for ap in ahead_paths:
            if not os.path.isfile(ap):
                continue
            headers = read_aheaders(ap)
            for hdr in headers:
                # SCAMP writes FLXSCALE=0 with SOLVE_PHOTOM N; drop the trap
                hdr.remove("FLXSCALE", ignore_missing=True, remove_all=True)
                hdr.set(
                    "ASTCORR",
                    radec_correlation,
                    "RA/Dec astrometric correlation (group r_eff)",
                )
                if astrrms_hsn is not None:
                    hdr.set(
                        "ASTRMSH1",
                        astrrms_hsn[0],
                        "High-S/N external astrom. RMS axis 1 (deg)",
                    )
                    hdr.set(
                        "ASTRMSH2",
                        astrrms_hsn[1],
                        "High-S/N external astrom. RMS axis 2 (deg)",
                    )
            write_aheaders(headers, ap)

        # Cache .ahead files for future reprocessing (now carrying ASTCORR)
        if self.setup.scamp_cache_dir is not None:
            for ap in ahead_paths:
                if os.path.isfile(ap):
                    rsync_file(ap, self.setup.scamp_cache_dir)
            # Cache scamp.xml alongside the .ahead so the solution metadata (high-S/N
            # RMS floor, RA/Dec correlation) travels with the cached solution. The live
            # copy stays in qc/astrometry for diagnostics.
            if os.path.isfile(path_xml):
                rsync_file(path_xml, self.setup.scamp_cache_dir)
            log.info(
                f"Cached {len(ahead_paths)} ahead files + scamp.xml to "
                f"{self.setup.scamp_cache_dir}"
            )

        # Print time
        elapsed = time.time() - tstart
        print_message(
            message=f"\n-> Elapsed time: {elapsed:.2f}s",
            kind="okblue",
            end="\n",
            logger=log,
        )

    # =========================================================================== #
    # Other properties
    # =========================================================================== #
    @property
    def _paths_image_headers(self):
        return [x.replace(".header", ".imageheader") for x in self.paths_headers]

    _image_headers = None

    @property
    def image_headers(self):
        """
        Obtains image headers from sextractor catalogs

        Returns
        -------
        iterable
            List of lists containing the image headers for each table and each
            extension.

        """

        # Check if already determined
        if self._image_headers is not None:
            return self._image_headers

        self._image_headers = []
        for idx in range(self.n_files):
            # Try to read the database
            try:
                with open(self._paths_image_headers[idx], "rb") as f:
                    # If the file is there, load the headers...
                    self._image_headers.append(pickle.load(f))

                    # And continue with next file
                    continue

            # If not found we move on to read the headers from the fits file
            except FileNotFoundError:
                headers = sextractor2imagehdr(path=self.paths_full[idx])

                # When done for all headers dump them into the designated database
                with open(self._paths_image_headers[idx], "wb") as d:
                    pickle.dump(headers, d)

                # Append converted headers
                self._image_headers.append(headers)

        # Return
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

        self._time_obs = Time(
            [hdr[0][self.setup.keywords.date_mjd] for hdr in self.image_headers],
            scale="utc",
            format="mjd",
        )
        return self._time_obs


class AstrometricCalibratedSextractorCatalogs(SextractorCatalogs):
    def __init__(self, setup, file_paths=None):
        super(AstrometricCalibratedSextractorCatalogs, self).__init__(
            file_paths=file_paths,
            setup=setup,
            key_ra="ALPHAWIN_SKY",
            key_dec="DELTAWIN_SKY",
        )

    def build_master_illumination_correction(self):
        """Illumination correction construction method."""

        # Import
        from vircampype.fits.images.flat import MasterIlluminationCorrection

        # Processing info
        print_header(
            header="MASTER-ILLUMINATION-CORRECTION",
            right=None,
            silent=self.setup.silent,
        )
        log = logging.getLogger(__name__)
        log.info(
            f"Building illumination correction from {self.n_files} files:\n{self.basenames2log}"
        )
        tstart = time.time()

        # At the moment, this only works when there is only one passband
        if len(list(set(self.passband))) != 1:
            raise ValueError("Only one passband allowed")
        # passband = self.passband[0]

        # Split based on passband and interval
        split = self.split_keywords(keywords=[self.setup.keywords.filter_name])
        split = flat_list(
            [s.split_window(window=0.5, remove_duplicates=True) for s in split]
        )
        log.info(f"Number of illumination correction groups: {len(split)}")

        # Get master photometry catalog
        master_phot = self.get_master_photometry()

        # Now loop through separated files
        for files, idx_print in zip(split, range(1, len(split) + 1)):
            # Create master dark name
            outpath = (
                self.setup.folders["master_object"]
                + f"MASTER-ILLUMINATION-CORRECTION_{files.mjd_mean:11.5f}.fits"
            )

            # Check if the file is already there and skip if it is
            if (
                check_file_exists(file_path=outpath, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                log.info("File already exists, skipping")
                continue

            log.info(
                f"Processing illumination correction {idx_print}/{len(split)}: {outpath}"
            )

            # Grab current passband
            passband = files.passband[0]

            # Fetch clean index for current filter
            mkeep = master_phot.get_purge_index(passband=passband)

            # Fetch magnitude and coordinates for master catalog
            mag_master = master_phot.mag(passband=passband)[0][0][mkeep]
            magerr_master = master_phot.mag_err(passband=passband)[0][0][mkeep]
            skycoord_master = master_phot.skycoord[0][0][mkeep]

            data_headers, flx_scale, n_sources = [], [], []
            for idx_hdu, idx_hdr in zip(
                files.iter_data_hdu[0], range(len(files.iter_data_hdu[0]))
            ):
                # Print processing info
                message_calibration(
                    n_current=idx_print,
                    n_total=len(split),
                    name=outpath,
                    d_current=idx_hdr + 1,
                    d_total=len(files.iter_data_hdu[0]),
                    silent=self.setup.silent,
                )

                # Read current HDU for all files into a single table
                tab = tvstack(files.hdu2table(hdu_index=idx_hdu))

                # Read header of current extension in first file
                header = files.image_headers[0][idx_hdr]

                # Clean table
                tab = clean_source_table(
                    table=tab,
                    image_header=header,
                    flux_max=header["SEXSATLV"] * 0.8,
                    min_distance_to_edge=10,
                    nndis_limit=5,
                    min_fwhm=0.8,
                    max_fwhm=6.0,
                    max_ellipticity=0.25,
                    verbose=False,
                )

                # Compute zero point depending on IC mode
                if self.setup.illumination_correction_mode == "variable":
                    method = "all"
                elif self.setup.illumination_correction_mode == "constant":
                    method = "weighted"
                else:
                    raise ValueError(
                        f"IC mode must be either 'variable' or 'constant', "
                        f"not {self.setup.illumination_correction_mode}"
                    )
                try:
                    zp_all = get_zeropoint(
                        skycoord1=SkyCoord(
                            tab[self.key_ra], tab[self.key_dec], unit="deg"
                        ),
                        mag1=tab["MAG_AUTO"],
                        magerr1=tab["MAGERR_AUTO"],
                        skycoord2=skycoord_master,
                        mag2=mag_master,
                        magerr2=magerr_master,
                        # IC runs before the data-driven bright limit exists
                        mag_limits_ref=master_phot.mag_lim_setup(passband=passband),
                        method=method,
                    )
                except ValueError as e:
                    # Name the frame(s), so a bad (e.g. trailed) exposure can be
                    # moved out of the raw data by hand
                    raise PipelineValueError(
                        f"Illumination correction failed for detector {idx_hdr + 1} "
                        f"of {', '.join(files.basenames)} ({len(tab)} clean sources): "
                        f"{e} Check the frame for trailing or bad seeing.",
                        logger=log,
                    ) from e

                # Compute illumination correction
                if self.setup.illumination_correction_mode == "variable":
                    # Remove all table entries without ZP entry
                    tab, zp_all = tab[np.isfinite(zp_all)], zp_all[np.isfinite(zp_all)]

                    # Grid with NN interpolation
                    grid_zp = grid_value_2d_nn(
                        x=tab["XWIN_IMAGE"],
                        y=tab["YWIN_IMAGE"],
                        values=zp_all,
                        n_bins_x=header["NAXIS1"] // 50,
                        n_bins_y=header["NAXIS2"] // 50,
                        x_min=1,
                        y_min=1,
                        x_max=header["NAXIS1"],
                        y_max=header["NAXIS2"],
                        n_nearest_neighbors=50,
                        metric="weighted",
                        weights=1 / tab["MAGERR_AUTO"] ** 2,
                    )

                    # Resize to original image size
                    grid_zp = upscale_image(
                        grid_zp,
                        new_size=(header["NAXIS1"], header["NAXIS2"]),
                        method="PIL",
                    )
                else:
                    grid_zp = np.full(
                        (header["NAXIS2"], header["NAXIS1"]),
                        fill_value=zp_all[0],
                        dtype=np.float32,
                    )

                # Convert to flux scale
                flx_scale.append(10 ** ((grid_zp - self.setup.target_zp) / 2.5))

                # Save number of sources
                n_sources.append(len(tab))

                # Plot sources on top of illumination correction
                # import matplotlib.pyplot as plt
                #
                # fig, ax = plt.subplots(
                #     nrows=1,
                #     ncols=1,
                #     gridspec_kw=dict(top=0.98, right=0.99),
                #     **dict(figsize=(6, 5)),
                # )
                #
                # im = ax.imshow(
                #     flx_scale[-1] / np.nanmedian(flx_scale[-1]),
                #     cmap="RdYlBu_r",
                #     vmin=0.95,
                #     vmax=1.04,
                #     origin="lower",
                #     extent=[0, header["NAXIS1"], 0, header["NAXIS2"]],
                # )
                # flux = 10 ** (zp_all / 2.5)
                # ax.scatter(
                #     tab["XWIN_IMAGE"],
                #     tab["YWIN_IMAGE"],
                #     c=flux / np.nanmedian(flux),
                #     s=5,
                #     lw=0.5,
                #     cmap="RdYlBu_r",
                #     ec="black",
                #     vmin=0.95,
                #     vmax=1.04,
                # )
                # plt.colorbar(im)
                # plt.show()
                # exit()

            # Instantiate output
            illumcorr = ImageCube(setup=self.setup)

            # Loop over extensions and construct final illumination correction
            for idx_hdu, fscl, nn in zip(files.iter_data_hdu[0], flx_scale, n_sources):
                # Append to output
                illumcorr.extend(data=fscl.astype(np.float32))

                # Create empty header
                data_header = fits.Header()

                # Add data to header
                add_int_to_header(
                    header=data_header,
                    key="HIERARCH PYPE IC NSOURCES",
                    value=nn,
                    comment="Number of sources used",
                )
                add_float_to_header(
                    header=data_header,
                    key="HIERARCH PYPE IC STD",
                    value=np.nanstd(fscl),  # noqa
                    decimals=4,
                    comment="Standard deviation in relative flux",
                )

                # Append header
                data_headers.append(data_header)

            # Make primary header
            prime_cards = make_cards(
                keywords=[
                    self.setup.keywords.object,
                    self.setup.keywords.date_mjd,
                    self.setup.keywords.filter_name,
                    self.setup.keywords.date_ut,
                    "HIERARCH PYPE N_FILES",
                ],
                values=[
                    "MASTER-ILLUMINATION-CORRECTION",
                    files.mjd_mean,
                    files.passband[0],
                    files.time_obs_mean,
                    len(files),
                ],
            )
            prime_header = fits.Header(cards=prime_cards)

            # Write to disk
            illumcorr.write_mef(
                path=outpath, prime_header=prime_header, data_headers=data_headers
            )
            log.info(f"Written: {outpath}")

            # QC plot
            if self.setup.qc_plots:
                msf = MasterIlluminationCorrection(setup=self.setup, file_paths=outpath)
                msf.qc_plot2d(paths=None, axis_size=5)

        # Print time
        elapsed = time.time() - tstart
        print_message(
            message=f"\n-> Elapsed time: {elapsed:.2f}s",
            kind="okblue",
            end="\n",
            logger=log,
        )

    def resolve_reference_mag_lo(self, master_phot, table_master, passband):
        """
        Determine the data-driven bright limit of the reference stars once per run.

        Point sources of all catalogues in self (all SExtractor flags, so that the
        saturation knee stays visible) are matched to the cleaned 2MASS table; the
        largest-aperture magnitude is compared with 2MASS after removing each
        table's own zero point. Repeat measurements of a 2MASS star are combined.
        The result is written to ``master_phot.path_mag_lo_auto`` and from then
        on returned by ``master_phot.mag_lim``.
        """
        log = logging.getLogger(__name__)
        if not self.setup.reference_mag_lo_auto:
            return
        if master_phot.mag_lo_auto(passband) is not None:
            log.info(f"Reference bright limit: {master_phot.mag_lim(passband)[0]}")
            return

        mag_lo, mag_hi = master_phot.mag_lim_setup(passband)
        passband_2mass = master_phot.translate_passband(passband)
        sc_master = SkyCoord(
            table_master[master_phot.key_ra],
            table_master[master_phot.key_dec],
            unit="deg",
        )
        mag_master = np.asarray(table_master[passband_2mass], dtype=float)

        dm_all, idx_all = [], []
        for idx_file in range(self.n_files):
            for table in self.file2table(file_index=idx_file):
                mag = np.asarray(table["MAG_APER"][:, -1], dtype=float)
                ra = np.asarray(table[self.key_ra], dtype=float)
                dec = np.asarray(table[self.key_dec], dtype=float)
                use = (table["CLASS_STAR"] > 0.8) & np.isfinite(mag) & (mag < 50)
                use &= np.isfinite(ra) & np.isfinite(dec)
                if use.sum() == 0:
                    continue
                sc = SkyCoord(ra[use], dec[use], unit="deg")
                idx, d2d, _ = sc.match_to_catalog_sky(sc_master)
                m = d2d.arcsec < 1.0
                dm, idx = mag[use][m] - mag_master[idx[m]], idx[m]
                # Remove this table's zero point (median in the plateau window)
                win = (mag_master[idx] >= mag_lo + 0.5) & (
                    mag_master[idx] < min(mag_lo + 2.0, mag_hi)
                )
                if win.sum() < 5:
                    continue
                dm_all.append(dm - np.median(dm[win]))
                idx_all.append(idx)

        # One median offset per 2MASS star
        if dm_all:
            dm_all, idx_all = np.concatenate(dm_all), np.concatenate(idx_all)
            ustar, inv = np.unique(idx_all, return_inverse=True)
            dm_star = np.array(
                ndimage_median(dm_all, labels=inv, index=np.arange(len(ustar)))
            )
            mag_star = mag_master[ustar]
        else:
            dm_star, mag_star = np.array([]), np.array([])

        limit, info = reference_bright_limit(
            mag_instr=dm_star + mag_star,
            mag_ref=mag_star,
            mag_lo=mag_lo,
            mag_hi=mag_hi,
        )
        bins = " ".join(
            f"{b:.2f}:{1000 * o:+.0f}({n})" for b, n, o in info["bins"] if n
        )
        log.info(
            f"Reference bright limit: {limit:.2f} ({info['source']}; setup {mag_lo}, "
            f"{len(mag_star)} stars, plateau {info['n_plateau']}); "
            f"offsets mmag(n) per bin: {bins}"
        )
        if info["too_bright"]:
            log.warning(
                f"Stars just fainter than the setup bright limit ({mag_lo}) already "
                f"deviate from 2MASS by > 20 mmag; consider a fainter reference_mag_lo"
            )
        with open(master_phot.path_mag_lo_auto, "w") as f:
            json.dump(
                dict(
                    passband=passband_2mass,
                    mag_lim_setup=[mag_lo, mag_hi],
                    mag_lo=limit,
                    source=info["source"],
                    n_stars=len(mag_star),
                    too_bright=info["too_bright"],
                    bins=info["bins"],
                ),
                f,
                indent=1,
            )

    def calibrate_photometry(self):
        # Fetch log
        log = logging.getLogger(__name__)

        # Processing info
        print_header(header="PHOTOMETRY", silent=self.setup.silent, right=None)
        log.info(f"Calibrating photometry on {self.n_files} file(s)")
        tstart = time.time()

        # Get master photometry catalog
        master_phot = self.get_master_photometry()[0]
        mkeep = master_phot.get_purge_index(passband=self.passband[0][0])
        table_master = master_phot.file2table(file_index=0)[0][mkeep]
        log.info(f"Master photometry: {master_phot.paths_full[0]}")
        log.info(f"Master photometry: {len(table_master)} sources")

        # Data-driven bright limit (once per run, used by mag_lim from now on)
        self.resolve_reference_mag_lo(
            master_phot=master_phot,
            table_master=table_master,
            passband=self.passband[0],
        )

        # Loop over files
        for idx_file in range(self.n_files):
            # Create output path
            path_out = self.paths_full[idx_file].replace(".fits.tab", ".fits.ctab")
            log.info(f"File {idx_file + 1}/{self.n_files}; output path: {path_out}")

            # Check if the file is already there and skip if it is
            if (
                check_file_exists(file_path=path_out, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                log.info("File already exists, skipping")
                continue

            # Print processing info
            message_calibration(
                n_current=idx_file + 1,
                n_total=self.n_files,
                name=path_out,
                d_current=None,
                d_total=None,
                silent=self.setup.silent,
            )

            # Load table and passband
            tables_file = self.file2table(file_index=idx_file)
            passband = self.passband[idx_file]
            passband_2mass = master_phot.translate_passband(passband)
            log.info(f"Number of HDUs: {len(tables_file)}")
            log.info(f"Passband: {passband}")
            log.info(f"2MASS passband: {passband_2mass}")

            # Read table HDUList
            table_hdulist = fits.open(self.paths_full[idx_file], mode="readonly")

            # Loop over tables
            for tidx, tidx_hdu in enumerate(self.iter_data_hdu[idx_file]):
                # Get current table
                log.info(f"Processing table {tidx + 1}/{len(tables_file)}")
                table = tables_file[tidx]
                log.info(f"Number of sources: {len(table)}")

                # Replace bad values with NaN
                log.info("Replacing bad values with NaN")
                sextractor_nanify_bad_values(table=table)

                # Replace masked columns with regular columns and fill with NaN
                log.info("Filling masked columns with NaN")
                fill_masked_columns(table=table, fill_value=np.nan)

                # Add aperture correction to table
                log.info("Adding aperture correction to table")
                table.add_column(
                    (table["MAG_APER"][:, -1] - table["MAG_APER"].T).T,
                    name="MAG_APER_COR",
                )

                # Get first-order ZP
                log.info("Computing first-order zero point for table cleaning")
                sc_table = SkyCoord(table[self.key_ra], table[self.key_dec], unit="deg")
                sc_master = SkyCoord(
                    table_master[master_phot.key_ra],
                    table_master[master_phot.key_dec],
                    unit="deg",
                )
                zp_auto, _ = get_zeropoint(
                    skycoord1=sc_table,
                    mag1=table["MAG_AUTO"],
                    magerr1=table["MAGERR_AUTO"],
                    skycoord2=sc_master,
                    mag2=table_master[passband_2mass],
                    magerr2=table_master[f"e_{passband_2mass}"],
                    method="weighted",
                )

                # Determine flux limits from allowed magnitude range
                flux_auto_max, flux_auto_min = 10 ** (
                    -(master_phot.mag_lim(passband) - zp_auto) / 2.5
                )
                log.info(
                    f"FLUX_AUTO limits for clean catalog: "
                    f"{flux_auto_min} - {flux_auto_max}"
                )

                # Create clean source table
                log.info("Creating clean source table")
                table_clean, clean_idx = clean_source_table(
                    table=table,
                    min_distance_to_edge=25,
                    min_fwhm=0.8,
                    max_fwhm=6.0,
                    max_ellipticity=0.25,
                    nndis_limit=5,
                    flux_auto_max=flux_auto_max,
                    flux_auto_min=flux_auto_min,
                    return_filter=True,
                    finite_columns=[
                        "FWHM_WORLD",
                        "ELLIPTICITY",
                        "MAG_AUTO",
                        "MAGERR_AUTO",
                    ],
                    n_jobs=self.setup.n_jobs,
                )
                log.info(f"Created clean source table with {len(table_clean)} sources")

                # Add cleaned index to table
                table.add_column(clean_idx, name="IDX_CLEAN")

                # Compute nearest neighbors matrix
                log.info("Computing nearest neighbors matrix")
                max_dis = 1800.0
                nn_dis, nn_idx = get_nearest_neighbors(
                    x=table["XWIN_IMAGE"],
                    y=table["YWIN_IMAGE"],
                    x0=table_clean["XWIN_IMAGE"],
                    y0=table_clean["YWIN_IMAGE"],
                    max_dis=max_dis,
                    n_neighbors=100,
                    n_jobs=self.setup.n_jobs,
                    leaf_size=50,
                )

                # Determine the number of parallel chunks from the number of sources
                n_max = 200_000
                n_per_job = n_max // self.setup.n_jobs
                n_splits = int(np.ceil(len(table) / n_per_job))

                # Split table
                log.info(f"Splitting table into {n_splits} chunks")
                table_chunks = split_table(table=table, n_splits=n_splits)
                len_table_chunks = [len(chunk) for chunk in table_chunks]

                # Compute adaptive gaussian kernel width from k-th neighbor distance
                kernel_std_pix = np.median(nn_dis[:, self.setup.phot_interp_kernel_k])
                kernel_std_pix = np.clip(kernel_std_pix, 30 * 3, 5 * 60 * 3)
                log.info(
                    f"Adaptive kernel std: {kernel_std_pix:.1f} px "
                    f"({kernel_std_pix / 3 / 60:.2f} arcmin, "
                    f"k={self.setup.phot_interp_kernel_k})"
                )
                # Split the neighbour matrices at the same indices as the table; the
                # neighbour data and weights are built per chunk (bounded memory)
                err_clean = np.asarray(table_clean["MAGERR_AUTO"])
                split_at = np.cumsum(len_table_chunks)[:-1]
                nn_dis_chunks = np.split(nn_dis, split_at)
                nn_idx_chunks = np.split(nn_idx, split_at)

                # Interpolate values in loop
                for par in ["MAG_APER_COR", "ELLIPTICITY", "FWHM_WORLD"]:
                    log.info(f"Interpolating {par} values")

                    vals, vals_std, n_sources, max_dists = [], [], [], []
                    for data, weights, dis in _neighbor_chunks(
                        data_clean=table_clean[par],
                        err_clean=err_clean,
                        nn_idx_chunks=nn_idx_chunks,
                        nn_dis_chunks=nn_dis_chunks,
                        kernel_std_pix=kernel_std_pix,
                    ):
                        # Replicate weights to third dimension if required
                        if data.ndim == 3:
                            weights = np.repeat(
                                weights[:, :, np.newaxis], data.shape[2], axis=2
                            )

                        # sigma clip
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            data = sigma_clip(
                                data, axis=1, sigma=2.0, maxiters=3, masked=False
                            )
                        bad_data_mask = np.isnan(data)

                        # Set weights to 0 and data to NaN for both masks
                        data[bad_data_mask] = 0
                        weights[bad_data_mask] = 0

                        # Compute number of non-zero weights
                        n_sources.append(np.sum(weights > 0, axis=1))

                        # Compute maximum distance
                        max_dists.append(np.nanmax(dis, axis=1))

                        # Compute weighted average and standard deviation
                        # (NaN where no neighbour carries weight)
                        wsum = np.sum(weights, axis=1)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            mean = np.sum(data * weights, axis=1) / wsum
                            var = (
                                np.sum(
                                    weights * (data - mean[:, np.newaxis]) ** 2, axis=1
                                )
                                / wsum
                            )
                        vals.append(mean)
                        vals_std.append(np.sqrt(var))

                    # Flatten lists and add to table
                    table[f"{par}_INTERP"] = np.concatenate(vals).astype(np.float32)
                    table[f"{par}_INTERP_STD"] = np.concatenate(vals_std).astype(
                        np.float32
                    )
                    table[f"{par}_INTERP_N"] = np.concatenate(n_sources).astype(
                        np.int16
                    )
                    table[f"{par}_INTERP_MAXDIS"] = np.concatenate(max_dists).astype(
                        np.float32
                    )
                    log.info(f"Interpolated {par} values")

                # Match apertures and add to table
                log.info("Matching apertures")
                mag_aper_matched = table["MAG_APER"] + table["MAG_APER_COR_INTERP"]
                table.add_column(mag_aper_matched, name="MAG_APER_MATCHED")

                # Add ZP attribute to the table
                setattr(table, "zp", dict())
                setattr(table, "zperr", dict())

                # Create skycoord instances
                log.info("Creating SkyCoord instances")
                sc_table = SkyCoord(table[self.key_ra], table[self.key_dec], unit="deg")
                sc_master = SkyCoord(
                    table_master[master_phot.key_ra],
                    table_master[master_phot.key_dec],
                    unit="deg",
                )

                # Compute zero points for all magnitude columns
                for cmag, cmagerr in zip(
                    ["MAG_APER", "MAG_APER_MATCHED", "MAG_AUTO"],
                    ["MAGERR_APER", "MAGERR_APER", "MAGERR_AUTO"],
                ):
                    log.info(f"Computing zeropoint for {cmag} column")
                    zeropoint, zeropoint_err = get_zeropoint(
                        skycoord1=sc_table,
                        skycoord2=sc_master,
                        mag1=table[cmag].data,
                        mag2=table_master[passband_2mass].data,
                        magerr1=table[cmagerr].data,
                        magerr2=table_master[f"e_{passband_2mass}"].data,
                        mag_limits_ref=master_phot.mag_lim(passband),
                        method="weighted",
                    )
                    log.info(f"Computed zeropoint for {cmag} column")
                    log.info(f"zeropoint: {zeropoint}")
                    log.info(f"zeropoint_error: {zeropoint_err}")

                    # Add calibrated magnitudes to table
                    table[f"{cmag}_CAL"] = np.float32(table[cmag] + zeropoint)

                    # Write ZPs and errors into attribute
                    log.info(f"Adding calibrated magnitudes for {cmag} column")
                    hpz = "HIERARCH PYPE ZP"
                    if hasattr(zeropoint, "__len__"):
                        for idx, (zp, zperr) in enumerate(
                            zip(zeropoint, zeropoint_err)
                        ):
                            table.zp[f"{hpz} {cmag} {idx}"] = zp  # noqa
                            table.zperr[f"{hpz} ERR {cmag} {idx}"] = zperr  # noqa
                    else:
                        table.zp[f"{hpz} {cmag}"] = zeropoint  # noqa
                        table.zperr[f"{hpz} ERR {cmag}"] = zeropoint_err  # noqa

                # Replace HDU once after all ZPs are computed
                table_hdulist[tidx_hdu] = table2bintablehdu(table)

                # Record the reference magnitude range used for the ZPs
                for key, val in zip(
                    ["HIERARCH PYPE REFMAG LO", "HIERARCH PYPE REFMAG HI"],
                    master_phot.mag_lim(passband),
                ):
                    add_float_to_header(
                        header=table_hdulist[tidx_hdu].header,
                        key=key,
                        value=val,
                        comment="2MASS reference magnitude limit (mag)",
                        decimals=2,
                    )

                # Add all accumulated ZP info to header
                log.info("Adding ZP info to header")
                attr_to_comment = {
                    "zp": "Zero point (mag)",
                    "zperr": "Standard error of ZP (mag)",
                }
                for attr, comment in attr_to_comment.items():
                    for key, val in getattr(table, attr).items():
                        add_float_to_header(
                            header=table_hdulist[tidx_hdu].header,
                            key=key,
                            value=val,
                            comment=comment,
                            decimals=4,
                        )

            # Write to new output file
            log.info(f"Writing calibrated photometry to {path_out}")
            table_hdulist.writeto(path_out, overwrite=self.setup.overwrite)

            # Close original file
            table_hdulist.close()

        # Print time
        elapsed = time.time() - tstart
        print_message(
            message=f"\n-> Elapsed time: {elapsed:.2f}s",
            kind="okblue",
            end="\n",
            logger=log,
        )

    def write_crder(self, images, mjd: list[float] | None = None) -> None:
        """
        Write CRDER1/CRDER2 into the images these catalogs were extracted from.

        CRDER1/2 are the 3-sigma-clipped RMS of the per-axis residuals of the clean
        point sources with S/N > 50 against the Gaia reference, propagated to the
        epoch of each catalog (D9). An HDU with fewer than 10 matches gets no
        cards and a warning.

        Parameters
        ----------
        images : SkyImages
            Images matching these catalogs file by file (tile: its image; stacks: every
            detector).
        mjd : list[float] | None, optional
            Epoch of each catalog; default its MJD-OBS (the first exposure). A coadd
            spanning many epochs passes the mean MJD of its exposures.

        Raises
        ------
        PipelineValueError
            If an image and its catalog have a different number of data HDUs.
        """
        log = logging.getLogger(__name__)

        sc_master_raw = self.get_master_astrometry().skycoord[0][0]
        for idx, path in enumerate(self.paths_full):
            # Gaia reference at the epoch of this catalog
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                reference = sc_master_raw.apply_space_motion(
                    new_obstime=Time((mjd or self.mjd)[idx], format="mjd")
                )
            hdus_img = list(images.iter_data_hdu[idx])
            hdus_cat = list(self.iter_data_hdu[idx])
            if len(hdus_img) != len(hdus_cat):
                raise PipelineValueError(
                    f"{images.paths_full[idx]} has {len(hdus_img)} data HDUs, "
                    f"{path} has {len(hdus_cat)}"
                )
            for hdu_img, hdu_cat in zip(hdus_img, hdus_cat):
                table = Table.read(path, hdu=hdu_cat)
                skycoord = SkyCoord(table[self.key_ra], table[self.key_dec], unit="deg")
                mask = _bright_point_sources(table)
                rms = astrometric_residual_rms(skycoord, reference, mask)
                if rms is None:
                    log.warning(
                        f"CRDER: fewer than 10 matched bright point sources in "
                        f"{path} [HDU {hdu_img}]; cards omitted"
                    )
                    continue
                cards = [
                    ("CRDER1", rms[0], "Random error in RA (deg)"),
                    ("CRDER2", rms[1], "Random error in Dec (deg)"),
                ]
                set_cards(images.paths_full[idx], cards, hdu=hdu_img)
                log.info(
                    f"CRDER {images.paths_full[idx]} [HDU {hdu_img}]: "
                    f"{rms[0] * 3.6e6:.1f} / {rms[1] * 3.6e6:.1f} mas"
                )

    def plot_qc_astrometry_1d(self, axis_size=5):
        # Import
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Processing info
        print_header(header="QC ASTROMETRY 1D", silent=self.setup.silent)
        log = logging.getLogger(__name__)
        log.info(f"Generating QC astrometry 1D plots for {len(self)} files")
        tstart = time.time()

        # Get FPA layout
        fpa_layout = self.setup.fpa_layout

        # Obtain master coordinates
        sc_master_raw = self.get_master_astrometry().skycoord[0][0]

        # Apply space motion to match data obstime
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sc_master_equal = sc_master_raw.apply_space_motion(
                new_obstime=self.time_obs_mean
            )

        # Loop over files
        for idx_file in range(len(self)):
            # Generate outpath
            outpath_sep = f"{self.setup.folders['qc_astrometry']}{self.names[idx_file]}_astr_referr_sep.pdf"
            outpath_ang = f"{self.setup.folders['qc_astrometry']}{self.names[idx_file]}_astr_referr_ang.pdf"

            # Check if file already exists
            if (
                check_file_exists(file_path=outpath_sep, silent=self.setup.silent)
                and check_file_exists(file_path=outpath_ang, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                log.info("File already exists, skipping")
                continue

            log.info(
                f"Plotting astrometry 1D for file {idx_file + 1}/{len(self)}: {self.names[idx_file]}"
            )

            # Grab coordinates
            sc_file = self.skycoord[idx_file]

            # Coadd mode
            if len(self) == 1:
                fig1, ax_all1 = get_plotgrid(
                    layout=(1, 1), xsize=4 * axis_size, ysize=4 * axis_size
                )
                ax_all1 = [ax_all1]
                fig2, ax_all2 = get_plotgrid(
                    layout=(1, 1), xsize=4 * axis_size, ysize=4 * axis_size
                )
                ax_all2 = [ax_all2]
            else:
                fig1, ax_all1 = get_plotgrid(
                    layout=fpa_layout, xsize=axis_size, ysize=axis_size
                )
                ax_all1 = ax_all1.ravel()
                fig2, ax_all2 = get_plotgrid(
                    layout=fpa_layout, xsize=axis_size, ysize=axis_size
                )
                ax_all2 = ax_all2.ravel()

            # Loop over extensions
            for idx_hdu in range(len(sc_file)):
                # Print processing info
                message_calibration(
                    n_current=idx_file + 1,
                    n_total=len(self),
                    name=outpath_sep,
                    d_current=idx_hdu + 1,
                    d_total=len(sc_file),
                    silent=self.setup.silent,
                )

                # Grab data for current HDU
                sc_hdu = sc_file[idx_hdu]

                # Get separations and position angles between matched master and table
                idx, sep2d_equal = sc_hdu.match_to_catalog_sky(
                    sc_master_equal, nthneighbor=1
                )[:2]
                i1 = sep2d_equal.arcsec <= 0.5
                i2, sep2d_equal = idx[i1], sep2d_equal[i1]
                ang_equal = sc_hdu[i1].position_angle(sc_master_equal[i2])

                # Get separations and position angles between matched master and table
                idx, sep2d_raw = sc_hdu.match_to_catalog_sky(
                    sc_master_raw, nthneighbor=1
                )[:2]
                i1 = sep2d_raw.arcsec <= 0.5
                i2, sep2d_raw = idx[i1], sep2d_raw[i1]
                ang_raw = sc_hdu[i1].position_angle(sc_master_raw[i2])

                # Draw separation histograms
                kwargs_hist = dict(
                    range=(0, 100),
                    bins=20,
                    histtype="step",
                    lw=2.0,
                    ls="solid",
                    alpha=0.7,
                )
                ax_all1[idx_hdu].hist(
                    sep2d_equal.mas,
                    ec="crimson",
                    label="Equalized epoch",
                    **kwargs_hist,
                )
                ax_all1[idx_hdu].hist(
                    sep2d_raw.mas, ec="dodgerblue", label="Raw epoch", **kwargs_hist
                )
                ax_all1[idx_hdu].axvline(0, c="black", ls="dashed", lw=1)

                # Draw position angle histgrams
                kwargs_hist = dict(
                    range=(0, 360),
                    bins=20,
                    histtype="step",
                    lw=2.0,
                    ls="solid",
                    alpha=0.7,
                )
                ax_all2[idx_hdu].hist(
                    ang_equal.degree,
                    ec="crimson",
                    label="Equalized epoch",
                    **kwargs_hist,
                )
                ax_all2[idx_hdu].hist(
                    ang_raw.degree, ec="dodgerblue", label="Raw epoch", **kwargs_hist
                )

                # Modify axes
                for ax, ll in zip(
                    [ax_all1[idx_hdu], ax_all2[idx_hdu]],
                    ["Separation (mas)", "Position angle (deg)"],
                ):
                    # Annotate detector ID
                    ax.annotate(
                        f"Det.ID: {idx_hdu + 1:0d}",
                        xy=(0.02, 1.01),
                        xycoords="axes fraction",
                        ha="left",
                        va="bottom",
                    )

                    # Modify axes
                    if idx_hdu < fpa_layout[1]:
                        ax.set_xlabel(ll)
                    else:
                        ax.axes.xaxis.set_ticklabels([])
                    if (idx_hdu + 1) % fpa_layout[0] == 0:
                        ax.set_ylabel("N")
                    else:
                        ax.axes.yaxis.set_ticklabels([])

                    # Set ticks
                    ax.xaxis.set_major_locator(MaxNLocator(5))
                    ax.xaxis.set_minor_locator(AutoMinorLocator())
                    ax.yaxis.set_major_locator(MaxNLocator(5))
                    ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Set label on last iteration
            for ax in [ax_all1[-1], ax_all2[-1]]:
                ax.legend(
                    loc="lower left",
                    bbox_to_anchor=(0.01, 1.02),
                    ncol=2,
                    fancybox=False,
                    shadow=False,
                    frameon=False,
                )

            # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="tight_layout : falling back to Agg renderer"
                )
                fig1.savefig(
                    outpath_sep, bbox_inches="tight", dpi=self.setup.qc_plot_dpi
                )
                fig2.savefig(
                    outpath_ang, bbox_inches="tight", dpi=self.setup.qc_plot_dpi
                )
            plt.close("all")
            log.info(f"Written: {outpath_sep}, {outpath_ang}")

        # Print time
        elapsed = time.time() - tstart
        print_message(
            message=f"\n-> Elapsed time: {elapsed:.2f}s",
            kind="okblue",
            end="\n",
            logger=log,
        )

    def plot_qc_astrometry_2d(
        self, axis_size=5, key_x="XWIN_IMAGE", key_y="YWIN_IMAGE"
    ):
        # Import
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Processing info
        print_header(header="QC ASTROMETRY 2D", silent=self.setup.silent)
        log = logging.getLogger(__name__)
        log.info(f"Generating QC astrometry 2D plots for {len(self)} files")
        tstart = time.time()

        # Get FPA layout
        fpa_layout = self.setup.fpa_layout

        # Obtain master coordinates
        sc_master = self.get_master_astrometry().skycoord[0][0]

        # Loop over files
        for idx_file in range(len(self)):
            # Generate outpath
            outpath = f"{self.setup.folders['qc_astrometry']}{self.names[idx_file]}_astr_referror2d.pdf"

            # Check if file exists
            if (
                check_file_exists(file_path=outpath, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                log.info("File already exists, skipping")
                continue

            log.info(
                f"Plotting astrometry 2D for file {idx_file + 1}/{len(self)}: {self.names[idx_file]}"
            )

            # Grab coordinates
            xx_file = self.get_column_file(idx_file=idx_file, column_name=key_x)
            yy_file = self.get_column_file(idx_file=idx_file, column_name=key_y)
            snr_file = self.get_column_file(idx_file=idx_file, column_name="SNR_WIN")
            sc_file = self.skycoord[idx_file]

            # Apply space motion to match data obstime
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                sc_master_matched = sc_master.apply_space_motion(
                    new_obstime=self.time_obs[idx_file]
                )

            # Coadd mode
            if len(self) == 1:
                fig, ax_all = get_plotgrid(
                    layout=(1, 1), xsize=4 * axis_size, ysize=4 * axis_size
                )
                ax_all = [ax_all]
            else:
                fig, ax_all = get_plotgrid(
                    layout=fpa_layout, xsize=axis_size, ysize=axis_size
                )
                ax_all = ax_all.ravel()
            cax = fig.add_axes([0.3, 0.92, 0.4, 0.02])

            # Loop over extensions
            im, sep_all = None, []
            for idx_hdu in range(len(sc_file)):
                # Print processing info
                message_calibration(
                    n_current=idx_file + 1,
                    n_total=len(self),
                    name=outpath,
                    d_current=idx_hdu + 1,
                    d_total=len(sc_file),
                    silent=self.setup.silent,
                )

                # Read header
                header = self.image_headers[idx_file][idx_hdu]

                # Get separations between master and current table
                i1, sep, _ = sc_file[idx_hdu].match_to_catalog_sky(sc_master_matched)

                # Extract position angles between master catalog and input
                # sc1 = sc_master_astrometry[i1]
                # ang = sc1.position_angle(sc_hdu)

                # Keep only those with a maximum of 0.5 arcsec
                keep = sep.arcsec < 0.5
                sep, x_hdu, y_hdu = (
                    sep[keep],
                    xx_file[idx_hdu][keep],
                    yy_file[idx_hdu][keep],
                )
                snr_hdu = snr_file[idx_hdu][keep]

                # Determine number of bins (with given radius at least 10 sources)
                stacked = np.stack([x_hdu, y_hdu]).T
                n_neighbors = 50 if len(stacked) > 50 else len(stacked)
                dis, _ = (
                    NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
                    .fit(stacked)
                    .kneighbors(stacked)
                )
                maxdis = np.percentile(dis[:, -1], 95)
                n_bins_x, n_bins_y = (
                    int(header["NAXIS1"] / maxdis),
                    int(header["NAXIS2"] / maxdis),
                )

                # Minimum number of 3 bins
                n_bins_x = 3 if n_bins_x <= 3 else n_bins_x
                n_bins_y = 3 if n_bins_y <= 3 else n_bins_y

                # Grid value into image
                grid = grid_value_2d_nn(
                    x=x_hdu,
                    y=y_hdu,
                    values=sep.mas,
                    n_nearest_neighbors=20,
                    n_bins_x=n_bins_x,
                    n_bins_y=n_bins_y,
                    x_min=1,
                    y_min=1,
                    x_max=header["NAXIS1"],
                    y_max=header["NAXIS2"],
                    metric="weighted",
                    weights=snr_hdu,
                )

                # Save high SN separations
                sep_all.append(sep.mas[snr_hdu > np.nanpercentile(snr_hdu, 90)])

                # Draw
                kwargs = dict(vmin=0, vmax=100, cmap="Spectral_r")
                extent = [0, header["NAXIS1"], 0, header["NAXIS2"]]
                im = ax_all[idx_hdu].imshow(
                    grid, extent=extent, origin="lower", rasterized=True, **kwargs
                )
                ax_all[idx_hdu].scatter(
                    x_hdu,
                    y_hdu,
                    c=sep.mas,
                    s=5,
                    lw=0.5,
                    ec="black",
                    alpha=0.5,
                    rasterized=True,
                    **kwargs,
                )

                # Annotate detector ID
                ax_all[idx_hdu].annotate(
                    f"Det.ID: {idx_hdu + 1:0d}",
                    xy=(0.02, 1.01),
                    xycoords="axes fraction",
                    ha="left",
                    va="bottom",
                )

                # Modify axes
                if idx_hdu < fpa_layout[1]:
                    ax_all[idx_hdu].set_xlabel("X (pix)")
                else:
                    ax_all[idx_hdu].axes.xaxis.set_ticklabels([])
                if (idx_hdu + 1) % fpa_layout[0] == 0:
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
            cbar = plt.colorbar(
                im, cax=cax, orientation="horizontal", label="Average separation (mas)"
            )
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")

            # Print external error stats
            message_qc_astrometry(separation=flat_list(sep_all))

            # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="tight_layout : falling back to Agg renderer"
                )
                fig.savefig(outpath, bbox_inches="tight", dpi=self.setup.qc_plot_dpi)
            plt.close("all")
            log.info(f"Written: {outpath}")

        # Print time
        elapsed = time.time() - tstart
        print_message(
            message=f"\n-> Elapsed time: {elapsed:.2f}s",
            kind="okblue",
            end="\n",
            logger=log,
        )


class PhotometricCalibratedSextractorCatalogs(AstrometricCalibratedSextractorCatalogs):
    def __init__(self, setup, file_paths=None):
        super(PhotometricCalibratedSextractorCatalogs, self).__init__(
            file_paths=file_paths, setup=setup
        )

    def write_photometric_cards(self, images) -> None:
        """
        Write the Phase 3 photometric cards into the images these catalogs were
        extracted from (tile image or every stack detector).

        Every data HDU of an image gets PHOTZP, PHOTZPER, PHOTSYS, ABMAGLIM,
        ABMAGSAT, PSF_FWHM and ELLIPTIC from ``photometric_cards`` on the matching
        catalog HDU; the image primary gets FLUXCAL = 'ABSOLUTE'.

        Parameters
        ----------
        images : SkyImages
            Images matching these catalogs file by file.

        Raises
        ------
        PipelineValueError
            If an image and its catalog have a different number of data HDUs, or a
            card cannot be determined.
        """
        log = logging.getLogger(__name__)

        for idx, path in enumerate(self.paths_full):
            hdus_img = list(images.iter_data_hdu[idx])
            hdus_cat = list(self.iter_data_hdu[idx])
            if len(hdus_img) != len(hdus_cat):
                raise PipelineValueError(
                    f"{images.paths_full[idx]} has {len(hdus_img)} data HDUs, "
                    f"{path} has {len(hdus_cat)}"
                )
            for k, (hdu_img, hdu_cat) in enumerate(zip(hdus_img, hdus_cat)):
                table = Table.read(path, hdu=hdu_cat)
                hdr_cat = fits.getheader(path, hdu_cat)
                cards = photometric_cards(
                    table,
                    hdr_cat,
                    images.headers_data[idx][k],
                    self.setup,
                    self.passband[idx],
                )
                set_cards(images.paths_full[idx], cards, hdu=hdu_img)
                values = {key: value for key, value, _ in cards}
                log.info(
                    f"Photometric cards {images.paths_full[idx]} [HDU {hdu_img}]: "
                    f"PHOTZP={values['PHOTZP']}, ABMAGLIM={values['ABMAGLIM']}, "
                    f"PSF_FWHM={values['PSF_FWHM']}"
                )
            primary = [("FLUXCAL", "ABSOLUTE", "Flux calibration")]
            set_cards(images.paths_full[idx], primary, hdu=0)

    def _merged_table(self, clean=True):
        # Import
        from astropy import table
        from astropy.utils.metadata import MergeConflictWarning

        # Read all tables
        tables_all = flat_list(
            [self.file2table(file_index=i) for i in range(self.n_files)]
        )

        # Clean if set
        if clean:
            tables_all = [clean_source_table(t) for t in tables_all]

        # Stack all tables into a single master table
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MergeConflictWarning)
            table_master = table.vstack(tables_all)

        # Remove all sources without a match from the master table
        stacked = np.stack(
            [
                np.deg2rad(table_master[self.key_ra]),
                np.deg2rad(table_master[self.key_dec]),
            ]
        ).T
        dis, idx = (
            NearestNeighbors(n_neighbors=2, metric="haversine")
            .fit(stacked)
            .kneighbors(stacked)
        )
        dis_arsec = np.rad2deg(dis) * 3600
        dis_arcsec_nn = dis_arsec[:, 1]
        good = dis_arcsec_nn < 0.2
        table_master = table_master[good]

        # Remove duplicates
        table_master = remove_duplicates_wcs(
            table=table_master,
            sep=1,
            key_lon=self.key_ra,
            key_lat=self.key_dec,
            temp_dir=self.setup.folders["temp"],
            bin_name=self.setup.bin_stilts,
        )

        # Return
        return table_master

    def _photerr_internal_all(self):
        # Only works if there are multiple catalogs available
        if len(self) <= 1:
            raise ValueError("Internal photometric error requires multiple catalogs.")

        # Read all tables
        tables_all = flat_list(
            [self.file2table(file_index=i) for i in range(self.n_files)]
        )

        # Get merged master table
        table_master = self._merged_table(clean=True)

        # Create empty array to store all matched magnitudes
        matched_phot = np.full(
            (len(table_master), len(tables_all)), fill_value=np.nan, dtype=np.float32
        )
        matched_photerr = np.full(
            (len(table_master), len(tables_all)), fill_value=np.nan, dtype=np.float32
        )

        # Do NN search in parallel (this takes the most time in a loop)
        def __match_catalogs(t, m):
            return (
                NearestNeighbors(n_neighbors=1, metric="haversine").fit(t).kneighbors(m)
            )

        stacked_master = np.stack(
            [
                np.deg2rad(table_master[self.key_ra]),
                np.deg2rad(table_master[self.key_dec]),
            ]
        ).T
        stacked_table = [
            np.stack([np.deg2rad(tt[self.key_ra]), np.deg2rad(tt[self.key_dec])]).T
            for tt in tables_all
        ]
        # threads: sklearn KD-tree is compiled C/Cython, releases GIL
        with Parallel(n_jobs=self.setup.n_jobs, prefer="threads") as parallel:
            mp = parallel(
                delayed(__match_catalogs)(i, j)
                for i, j in zip(stacked_table, repeat(stacked_master))
            )
        dis_all, idx_all = list(zip(*mp))

        # Now loop over all individual tables and find matches
        for tidx in range(len(tables_all)):
            # Grad current match
            dis, idx = dis_all[tidx], idx_all[tidx]

            # Determine bad matches
            bad_dis = np.rad2deg(dis[:, 0]) * 3600 > 0.2

            # Write nearest neighbor photometry into matched photometry array
            matched_phot[:, tidx] = tables_all[tidx][idx[:, 0]]["MAG_AUTO_CAL"]  # noqa
            matched_photerr[:, tidx] = tables_all[tidx][idx[:, 0]][  # noqa
                "MAGERR_AUTO"
            ]

            # Mask bad matches
            matched_phot[bad_dis, tidx] = np.nan
            matched_photerr[bad_dis, tidx] = np.nan

        # Compute internal phot error
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _, phot_median, phot_err = sigma_clipped_stats(
                matched_phot, axis=1, std_ddof=1
            )
            _, photerr_median, _ = sigma_clipped_stats(matched_photerr, axis=1)

        # Return
        return phot_median, phot_err, photerr_median

    def photerr_internal(self):
        # Create pickle path
        pickle_path = f"{self.setup.folders['temp']}photerr_internal.p"

        # Try to load from file
        try:
            photerr_internal_dict = pickle.load(open(pickle_path, "rb"))

        # If not there, compute internal error
        except FileNotFoundError:
            # Print info
            print_header(
                header="INTERNAL PHOTOMETRIC ERROR",
                silent=self.setup.silent,
                left=None,
                right=None,
            )
            log = logging.getLogger(__name__)
            tstart = time.time()

            # Determine photometric statistics
            phot_median, phot_err, photerr_median = self._photerr_internal_all()

            # Get the 5% brightest sources
            mag_lo = self.get_master_photometry()[0].mag_lim_lo(
                passband=self.passband[0][0]
            )
            good = phot_median >= mag_lo
            idx_bright = phot_median[good] < np.percentile(phot_median[good], 5)

            # Determine interal photometric error
            photerr_internal = clipped_median(phot_err[good][idx_bright], sigma=2)

            # Construct dict
            photerr_internal_dict = {
                "phot_median": phot_median,
                "phot_err": phot_err,
                "photerr_median": photerr_median,
                "photerr_internal": photerr_internal,
            }

            # Dump to file
            pickle.dump(photerr_internal_dict, open(pickle_path, "wb"))

            # Print error (console headline) + machine-readable QC record (file)
            print_message(
                message=f"err = {photerr_internal_dict['photerr_internal']:0.4f} mag"
            )
            logging.getLogger(__name__).info(
                "qc photometry photerr_internal_median_mag="
                f"{photerr_internal_dict['photerr_internal']:.4f}"
            )
            elapsed = time.time() - tstart
            print_message(
                message=f"\n-> Elapsed time: {elapsed:.2f}s",
                kind="okblue",
                end="\n",
                logger=log,
            )

        # Get median error of those
        return photerr_internal_dict

    def build_statistics_tables(self, astrms_fallback, statistics_bases=None):
        """
        Write ``<catalog>.stats``: per source MJDEFF, EXPTIME, NIMG and the
        astrometric floor ASTRMS1/2/ASTRMS_CORR, one table per catalog table.

        The statistics images are ``<base>.mjd.eff.fits``, ``<base>.exptime.fits``
        and ``<base>.nimg.fits``; *statistics_bases* gives one base per catalog
        (pawprints: the statistics folder), default the catalog path without
        ``.full.fits.ctab`` (tile, stacks).
        """
        # Import
        from vircampype.fits.images.common import FitsImages

        # Processing info
        print_header(header="STATISTICS TABLES", silent=self.setup.silent, right=None)
        log = logging.getLogger(__name__)
        log.info(f"Building statistics tables for {self.n_files} files")
        tstart = time.time()

        for idx_file in range(self.n_files):
            # Create output path
            path_out = f"{self.paths_full[idx_file]}.stats"

            # Check if file already exists
            if (
                check_file_exists(file_path=path_out, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                log.info("File already exists, skipping")
                continue

            log.info(
                f"Processing statistics table {idx_file + 1}/{self.n_files}: {path_out}"
            )

            # Print processing info
            message_calibration(
                n_current=idx_file + 1,
                n_total=self.n_files,
                name=path_out,
                d_current=None,
                d_total=None,
                silent=self.setup.silent,
            )

            # Find files
            if statistics_bases is None:
                base = self.paths_full[idx_file].replace(".full.fits.ctab", "")
            else:
                base = statistics_bases[idx_file]
            path_mjd = f"{base}.mjd.eff.fits"
            path_exptime = f"{base}.exptime.fits"
            path_nimg = f"{base}.nimg.fits"

            # Per-tile flat-SCAMP floor scalar (mas, mas, correlation): the value
            # the self-calibration shrinks toward / returns where Gaia is too sparse.
            astrms1_flat, astrms2_flat, astrms_corr_flat = astrms_fallback

            # Check if files are available
            missing = [
                p for p in (path_mjd, path_exptime, path_nimg) if not os.path.isfile(p)
            ]
            if missing:
                raise PipelineFileNotFoundError(
                    f"Statistics images not found: {', '.join(missing)}", logger=log
                )

            # Instantiate
            image_mjdeff = FitsImages(file_paths=path_mjd, setup=self.setup)

            # Open current table file
            hdul_in = fits.open(self.paths_full[idx_file], mode="readonly")
            hdul_out = hdul_in.copy()

            # Catalog and statistics data HDUs pair up by position (the detector);
            # idx_hdu_position also indexes self.skycoord/self.image_headers
            if len(self.iter_data_hdu[idx_file]) != image_mjdeff.n_data_hdu[0]:
                raise PipelineValueError(
                    f"{self.paths_full[idx_file]} has "
                    f"{len(self.iter_data_hdu[idx_file])} data HDUs, {path_mjd} has "
                    f"{image_mjdeff.n_data_hdu[0]}",
                    logger=log,
                )
            for idx_hdu_position, (idx_hdu_self, idx_hdu_stats) in enumerate(
                zip(self.iter_data_hdu[idx_file], image_mjdeff.iter_data_hdu[0])
            ):
                # Read table
                table_hdu = self.filehdu2table(
                    file_index=idx_file, hdu_index=idx_hdu_self
                )

                # Read stats
                mjdeff = fits.getdata(path_mjd, idx_hdu_stats)
                exptime = fits.getdata(path_exptime, idx_hdu_stats)
                nimg = fits.getdata(path_nimg, idx_hdu_stats)

                # Obtain wcs for statistics images (they all have the same projection)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    wcs_stats = WCS(
                        header=image_mjdeff.headers_data[0][idx_hdu_position]
                    )

                # Convert to X/Y and sample; bad = off-grid or no coverage
                xx, yy = wcs_stats.wcs_world2pix(
                    table_hdu[self.key_ra], table_hdu[self.key_dec], 0
                )
                mjdeff_sources, exptime_sources, nimg_sources, off, bad = (
                    _sample_statistics_grid(xx, yy, mjdeff, exptime, nimg)
                )

                # Guard: no more than 5% of sources may fall outside the edges
                if sum(off) / len(off) > 0.05:
                    raise ValueError(
                        f"Too many sources are close to the image edge ({sum(off)}/{len(off)}). "
                        f"Please check for issues. (file: {self.paths_full[idx_file]}, TableHDU: {idx_hdu_self})"
                    )

                # Guard against OOM: the self-calibrated floor runs in a single
                # in-memory pass, so peak RAM scales with the source count; fail loud
                # before the large allocation.
                n_src = len(table_hdu)
                if n_src > self.setup.astrms_self_max_sources:
                    raise PipelineValueError(
                        f"{n_src:,} sources in {self.paths_full[idx_file]!r} exceeds "
                        f"astrms_self_max_sources "
                        f"({self.setup.astrms_self_max_sources:,}); the self-calibrated "
                        "astrometric floor is computed in a single in-memory pass and "
                        "would risk a RAM blow-up. Raise the knob if this machine has "
                        "the memory, or chunk the per-source evaluation.",
                        logger=log,
                    )
                # Flat-SCAMP floor broadcast to every source. The self-calibration
                # below overwrites it; it survives only as the sparse-Gaia fallback
                # / shrink target.
                astrms_sources1 = np.full(n_src, astrms1_flat, dtype=float)
                astrms_sources2 = np.full(n_src, astrms2_flat, dtype=float)
                astrms_corr_sources = np.full(n_src, astrms_corr_flat, dtype=float)
                astrms_sources1[bad], astrms_sources2[bad] = np.nan, np.nan
                astrms_corr_sources[bad] = np.nan

                # Self-calibrate the astrometric floor on the coadded tile vs
                # epoch-warped Gaia, overwriting the flat-SCAMP scalar. Any failure
                # (sparse Gaia, missing epoch, legacy master, ...) is logged and
                # keeps the flat-SCAMP floor -- it never aborts the tile.
                try:
                    (
                        astrms_sources1,
                        astrms_sources2,
                        astrms_corr_sources,
                    ) = self._self_calibrate_astrms(
                        idx_file=idx_file,
                        idx_hdu_position=idx_hdu_position,
                        table_hdu=table_hdu,
                        bad=bad,
                        mjdeff_sources=mjdeff_sources,
                        astrms_sources1=astrms_sources1,
                        astrms_sources2=astrms_sources2,
                        astrms_corr_sources=astrms_corr_sources,
                    )
                except Exception as exc:
                    log.warning(
                        f"Astrometric floor self-calibration failed for "
                        f"{self.paths_full[idx_file]!r} (HDU {idx_hdu_self}); "
                        f"keeping the flat-SCAMP floor. Reason: {exc}"
                    )

                # Make new columns
                new_cols = fits.ColDefs(
                    [
                        fits.Column(
                            name="MJDEFF", format="D", array=mjdeff_sources, unit="d"
                        ),
                        fits.Column(
                            name="EXPTIME",
                            format="E",
                            array=exptime_sources,
                            unit="s",
                        ),
                        fits.Column(
                            name="NIMG",
                            format="J",
                            array=np.rint(nimg_sources).astype(int),
                        ),
                        fits.Column(
                            name="ASTRMS1",
                            format="E",
                            array=astrms_sources1,
                            unit="mas",
                        ),
                        fits.Column(
                            name="ASTRMS2",
                            format="E",
                            array=astrms_sources2,
                            unit="mas",
                        ),
                        fits.Column(
                            name="ASTRMS_CORR",
                            format="E",
                            array=astrms_corr_sources,
                        ),
                    ]
                )

                hdul_out[idx_hdu_self] = fits.BinTableHDU.from_columns(
                    new_cols, header=hdul_in[idx_hdu_self].header
                )

            # Write to new output file
            hdul_out.writeto(path_out, overwrite=self.setup.overwrite)
            hdul_in.close()
            hdul_out.close()
            log.info(f"Written: {path_out}")

        # Print time
        elapsed = time.time() - tstart
        print_message(
            message=f"\n-> Elapsed time: {elapsed:.2f}s",
            kind="okblue",
            end="\n",
            logger=log,
        )

    def _self_calibrate_astrms(
        self,
        idx_file,
        idx_hdu_position,
        table_hdu,
        bad,
        mjdeff_sources,
        astrms_sources1,
        astrms_sources2,
        astrms_corr_sources,
    ):
        """
        Self-measure the astrometric position-error floor and overwrite the
        flat-SCAMP ``ASTRMS1/2/CORR`` arrays for one (tile) HDU.

        Crossmatches the coadded source catalog to epoch-warped Gaia, measures
        the per-axis excess-variance floor (and a real RA/Dec correlation) on a
        clean, bright, floor-dominated sample, and -- when the measurement is
        trustworthy -- replaces the SCAMP floor with it. A sparse-Gaia fallback
        ladder keeps the flat-SCAMP value whenever the measurement is too thin or
        non-positive, so a cloud field never yields a zero/garbage/NaN floor (the
        worst case is exactly today's behavior). When enabled, the optional
        spatial map replaces the scalar with an edge-preserving fixed-cell + 3x3
        median field that keeps the real per-detector/per-tile floor structure
        (see :meth:`_astrms_floor_map`).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The (possibly overwritten) ``(astrms1, astrms2, astrms_corr)`` arrays
            in mas, with the existing bad-source NaN mask re-applied.
        """
        log = logging.getLogger(__name__)
        setup = self.setup

        # --- Crossmatch the tile catalog to epoch-warped Gaia (mirrors
        # plot_qc_astrometry_2d) ---
        sc_master = self.get_master_astrometry().skycoord[0][0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sc_master_matched = sc_master.apply_space_motion(
                new_obstime=self.time_obs[idx_file]
            )
        sc_file = self.skycoord[idx_file][idx_hdu_position]
        i1, sep, _ = sc_file.match_to_catalog_sky(sc_master_matched)
        matched = sep.arcsec < setup.astrms_self_match_radius

        # Residuals are taken against the matched Gaia stars moved to each
        # source's own effective epoch (MJDEFF): a multi-epoch coadd position sits
        # at the weighted mean epoch, not at the first exposure's MJD-OBS. Sources
        # without a finite MJDEFF (no coverage) keep NaN residuals and so drop out
        # of the sample below.
        valid = matched & np.isfinite(mjdeff_sources)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            gaia = sc_master[i1[valid]].apply_space_motion(
                new_obstime=Time(mjdeff_sources[valid], format="mjd", scale="utc")
            )
        sc_valid = sc_file[valid]

        # Signed per-axis residuals in mas. The RA axis is the true-angle
        # coordinate alpha* = dRA * cos(dec): multiplying by cos(dec) puts it in
        # the same on-sky angular frame as the Dec axis AND as the SExtractor
        # centroid covariance (ERRAWIN/ERRBWIN are on-sky arcs), so the
        # excess-variance subtraction floor^2 = <dr^2> - <known var> is consistent
        # per axis.
        cos_dec = np.cos(np.deg2rad(sc_valid.dec.deg))
        # (x + 180) % 360 - 180: wrap-safe signed dRA for fields straddling RA=0.
        dra_deg = (sc_valid.ra.deg - gaia.ra.deg + 180.0) % 360.0 - 180.0
        dr_alpha = np.full(len(sc_file), np.nan)
        dr_delta = np.full(len(sc_file), np.nan)
        dr_alpha[valid] = dra_deg * cos_dec * 3_600_000
        dr_delta[valid] = (sc_valid.dec.deg - gaia.dec.deg) * 3_600_000

        # Per-source centroid covariance (mas^2) -- the SAME formula re-added in
        # detections_to_sources, so the subtracted and added terms cannot drift.
        cov_ra, cov_dec, cov_radec = centroid_covariance_radec_mas2(
            np.asarray(table_hdu["ERRAWIN_WORLD"], dtype=float),
            np.asarray(table_hdu["ERRBWIN_WORLD"], dtype=float),
            np.asarray(table_hdu["ERRTHETAWIN_SKY"], dtype=float),
        )

        # Gaia reference variance (mas^2); negligible at the bright end, 0 if a
        # legacy master lacks the per-source error columns.
        try:
            master = self.get_master_astrometry()
            ra_err = np.asarray(master.get_columns("ra_error")[0][0], dtype=float)
            dec_err = np.asarray(master.get_columns("dec_error")[0][0], dtype=float)
            var_gaia_ra = (ra_err[i1] * 3_600_000) ** 2
            var_gaia_dec = (dec_err[i1] * 3_600_000) ** 2
        except (KeyError, IndexError, TypeError):
            var_gaia_ra = np.zeros_like(dr_alpha)
            var_gaia_dec = np.zeros_like(dr_alpha)

        # --- Clean, bright, floor-dominated sample ---
        image_header = self.image_headers[idx_file][idx_hdu_position]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            pixscale_arcsec = (
                np.sqrt(np.abs(np.linalg.det(WCS(image_header).pixel_scale_matrix)))
                * 3600.0
            )
        clean = clean_source_table(
            table_hdu,
            image_header=image_header,
            return_filter=True,
            min_snr=setup.astrms_self_min_snr,
            max_ellipticity=setup.astrms_self_max_ellipticity,
            nndis_limit=setup.astrms_self_isolation_arcsec / pixscale_arcsec,
        )[1]
        centroid_sigma = np.sqrt(0.5 * (cov_ra + cov_dec))
        sample = (
            clean
            & matched
            & np.isfinite(dr_alpha)
            & np.isfinite(dr_delta)
            & np.isfinite(cov_ra)
            & np.isfinite(cov_dec)
            & (centroid_sigma < setup.astrms_self_max_centroid_sigma)
        )

        # --- Global scalar floor (the validated, robust core) ---
        floor1, floor2, corr, n_used = estimate_astrms_floor(
            dr_alpha[sample],
            dr_delta[sample],
            cov_ra[sample],
            cov_dec[sample],
            cov_radec[sample],
            var_gaia_ra[sample],
            var_gaia_dec[sample],
            sigma=setup.astrms_self_sigma_clip,
        )

        # Sparse-Gaia / failed-measurement rung: too few clean matches, or a
        # non-positive / non-finite floor -> keep the flat-SCAMP value for the
        # whole product. Never write a zero/tiny floor: that would UNDER-estimate
        # the public errors, the opposite and more dangerous failure mode.
        if (
            n_used < setup.astrms_self_min_n_global
            or not (np.isfinite(floor1) and np.isfinite(floor2))
            or floor1 <= 0.0
            or floor2 <= 0.0
        ):
            log.info(
                f"Astrometric floor self-calibration for {self.names[idx_file]}: "
                f"insufficient/failed measurement (n={n_used}, "
                f"floor=({floor1:.3g}, {floor2:.3g}) mas) -> keeping flat-SCAMP floor."
            )
            return astrms_sources1, astrms_sources2, astrms_corr_sources

        # Representative flat-SCAMP floor (constant per tile) used as the shrink
        # target and where the spatial map is thin.
        scamp1 = float(np.nanmedian(astrms_sources1))
        scamp2 = float(np.nanmedian(astrms_sources2))
        scamp_corr = float(np.nanmedian(astrms_corr_sources))
        scamp1 = scamp1 if np.isfinite(scamp1) else floor1
        scamp2 = scamp2 if np.isfinite(scamp2) else floor2
        scamp_corr = scamp_corr if np.isfinite(scamp_corr) else 0.0

        # Shrink a thin global measurement toward the flat-SCAMP value (continuous
        # in N: w -> 1 for large, well-sampled tiles), so a noisy 30-100 source
        # measurement is damped rather than trusted outright.
        w = n_used / (n_used + setup.astrms_self_shrink_n)
        g1 = float(np.sqrt(w * floor1**2 + (1.0 - w) * scamp1**2))
        g2 = float(np.sqrt(w * floor2**2 + (1.0 - w) * scamp2**2))
        gcorr = float(np.clip(w * corr + (1.0 - w) * scamp_corr, -1.0, 1.0))

        n_src = len(dr_alpha)
        new1 = np.full(n_src, g1, dtype=float)
        new2 = np.full(n_src, g2, dtype=float)
        new_corr = np.full(n_src, gcorr, dtype=float)

        # --- Optional spatial map: an edge-preserving fixed-cell + 3x3 median
        # floor field around the global scalar ---
        if setup.astrms_self_build_map:
            try:
                new1, new2, new_corr, diag = self._astrms_floor_map(
                    table_hdu=table_hdu,
                    sample=sample,
                    dr_alpha=dr_alpha,
                    dr_delta=dr_delta,
                    cov_ra=cov_ra,
                    cov_dec=cov_dec,
                    cov_radec=cov_radec,
                    var_gaia_ra=var_gaia_ra,
                    var_gaia_dec=var_gaia_dec,
                    g1=g1,
                    g2=g2,
                    gcorr=gcorr,
                    scamp1=scamp1,
                    scamp2=scamp2,
                    pixscale_arcsec=pixscale_arcsec,
                )
            except Exception as exc:
                log.warning(
                    f"Astrometric floor spatial map failed for "
                    f"{self.names[idx_file]}; using the global scalar floor. "
                    f"Reason: {exc}"
                )
                new1 = np.full(n_src, g1, dtype=float)
                new2 = np.full(n_src, g2, dtype=float)
                new_corr = np.full(n_src, gcorr, dtype=float)
            else:
                # QC plot of the applied floor map. The diagnostic grids are a
                # byproduct of the map actually applied, so plotting them here
                # (build time) avoids recompute drift; a plot failure must never
                # break the floor itself.
                if setup.qc_plots:
                    try:
                        self.plot_qc_astrms_floor_map(
                            name=self.names[idx_file], diag=diag
                        )
                    except Exception as exc:
                        log.warning(
                            f"Astrometric floor-map QC plot failed for "
                            f"{self.names[idx_file]}: {exc}"
                        )

        # Re-apply the pre-existing bad-source NaN mask so downstream NaN handling
        # in detections_to_sources is unchanged.
        new1[bad] = np.nan
        new2[bad] = np.nan
        new_corr[bad] = np.nan

        log.info(
            f"Astrometric floor self-calibrated for {self.names[idx_file]}: "
            f"global floor ({g1:.2f}, {g2:.2f}) mas, corr {gcorr:+.3f}, "
            f"N={n_used} clean bright matches (flat-SCAMP was "
            f"{scamp1:.2f}, {scamp2:.2f} mas"
            f"{'; + spatial map' if setup.astrms_self_build_map else ''})."
        )
        return (
            new1.astype(astrms_sources1.dtype),
            new2.astype(astrms_sources2.dtype),
            new_corr.astype(astrms_corr_sources.dtype),
        )

    def _astrms_floor_map(
        self,
        table_hdu,
        sample,
        dr_alpha,
        dr_delta,
        cov_ra,
        cov_dec,
        cov_radec,
        var_gaia_ra,
        var_gaia_dec,
        g1,
        g2,
        gcorr,
        scamp1,
        scamp2,
        pixscale_arcsec,
    ):
        """
        Build an EDGE-PRESERVING spatial floor field on fixed-size cells.

        A mosaic is a patchwork of independently SCAMP-solved tiles, so the
        astrometric floor is *discontinuous* at detector/tile edges; a smooth
        global surface (the old degree-2 polynomial) blurs those steps into a
        bland gradient. This builds the field on cells of a fixed angular size
        (``astrms_self_cell_arcmin`` ~ one VIRCAM detector), forms a robust
        per-cell floor^2 from the clean bright sample, then applies a 3x3
        ``nanmedian`` -- a nonlinear, edge-preserving smoother that denoises
        within a flat detector while keeping the steps between detectors. One
        setting works for a single 1.6 deg tile and a 6 deg mosaic alike.

        Pipeline (all in variance space until the final sqrt):
          1. fixed cells from the coadd pixel scale; per cell with
             >= ``astrms_self_cell_min_n`` clean stars, floor^2 = clipped-var of
             the (mean-subtracted) residuals minus the mean known per-source
             variance (centroid + Gaia), plus a robust cross-covariance;
          2. 3x3 nanmedian over SUPPORTED cells only (fallback/global values
             never enter the median);
          3. footprint-aware fill: interior holes close to support get a linear
             interpolation; wide holes / outside-footprint cells fall back to the
             global scalar (never bridge chip gaps or separate pawprints);
          4. conservatism guard: a below-global cell is only trusted where its 3x3
             window holds >= ``astrms_self_cell_min_neff`` stars, else it is raised
             to the global floor; variance clipped to ``[(0.5*global)^2,
             max(flat-SCAMP, global)^2]``, with well-supported cells allowed up
             to a 2x sanity ceiling (seam floors can exceed flat-SCAMP);
          5. correlation shrunk toward the global value by window support (harder
             than the variances) and clipped to ``[-0.95, 0.95]`` -- which keeps
             ``cross^2 < var1*var2`` (PSD) for the 2x2 floor covariance;
          6. each source takes its own cell's value (nearest-cell, piecewise
             constant -- NOT bilinear, which would smear the very edges we
             preserve: this is an error model, not an image product).

        Residuals are alpha* = dRA*cos(dec) and delta, in the same true-angle
        frame as the centroid/Gaia covariances (cos(dec) applied upstream).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, dict]
            ``(new1, new2, new_corr, diag)`` -- the per-source floors (mas, mas,
            correlation) and a dict of diagnostic grids for the QC plot.
        """
        from scipy.interpolate import griddata
        from scipy.ndimage import convolve, distance_transform_edt, generic_filter

        setup = self.setup
        cell_arcmin = setup.astrms_self_cell_arcmin
        min_n = setup.astrms_self_cell_min_n
        neff_min = setup.astrms_self_cell_min_neff
        # Interior holes within this many cells of real support get interpolated;
        # anything further (wide dead zones, chip gaps, gaps between pawprints) is
        # left for the global fallback. A method internal, not an operator knob.
        max_interp_cells = 2.0

        xx = np.asarray(table_hdu["XWIN_IMAGE"], dtype=float)
        yy = np.asarray(table_hdu["YWIN_IMAGE"], dtype=float)
        finite_xy = np.isfinite(xx) & np.isfinite(yy)
        if (
            not np.any(finite_xy)
            or not np.isfinite(pixscale_arcsec)
            or pixscale_arcsec <= 0
        ):
            raise ValueError(
                "astrms floor map: no finite source coordinates or bad pixel scale"
            )

        # Extent spans ALL sources so the grid covers the whole catalog footprint,
        # not just the clean sample. The sample (clean bright sources) drives the
        # per-cell estimates.
        xlo, xhi = float(np.min(xx[finite_xy])), float(np.max(xx[finite_xy]))
        ylo, yhi = float(np.min(yy[finite_xy])), float(np.max(yy[finite_xy]))
        if not (xhi > xlo and yhi > ylo):
            raise ValueError("astrms floor map: degenerate source coordinate extent")

        smask = sample & finite_xy
        sx, sy = xx[smask], yy[smask]
        # Mean-subtracted residuals + per-source known variance (centroid + Gaia)
        # on the clean sample.
        da = dr_alpha[smask] - float(np.mean(dr_alpha[smask]))
        dd = dr_delta[smask] - float(np.mean(dr_delta[smask]))
        cov_a = cov_ra[smask] + var_gaia_ra[smask]
        cov_d = cov_dec[smask] + var_gaia_dec[smask]
        cov_x = cov_radec[smask]

        # STEP 1 -- fixed-size cells from the coadd pixel scale (one cell =
        # cell_arcmin on a side ~ one detector at 10').
        cell_px = cell_arcmin * 60.0 / pixscale_arcsec
        nbx = max(3, int(np.ceil((xhi - xlo) / cell_px)))
        nby = max(3, int(np.ceil((yhi - ylo) / cell_px)))
        xe = np.linspace(xlo, xhi, nbx + 1)
        ye = np.linspace(ylo, yhi, nby + 1)

        ix = np.clip(np.digitize(sx, xe) - 1, 0, nbx - 1)
        iy = np.clip(np.digitize(sy, ye) - 1, 0, nby - 1)
        cell_var1 = np.full((nby, nbx), np.nan)
        cell_var2 = np.full((nby, nbx), np.nan)
        cell_cross = np.full((nby, nbx), np.nan)
        cell_n = np.zeros((nby, nbx))
        flat = iy * nbx + ix
        sclip = setup.astrms_self_sigma_clip
        zero = np.zeros_like(da)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for cid in np.unique(flat):
                m = flat == cid
                j, i = divmod(int(cid), nbx)
                n = int(np.sum(m))
                cell_n[j, i] = n
                if n < min_n:
                    continue
                # Per-cell floor via the SAME excess-variance estimator as the
                # global floor (estimate_astrms_floor): sigma-clip the residuals,
                # then subtract the mean known variance over the SURVIVORS only
                # (not the full cell), so high-error outliers cannot bias the floor
                # low. cov_a/cov_d already fold in the Gaia variance, so the Gaia
                # term is passed as zero. var1/var2/cross share support by
                # construction (all written together for cells with >= min_n).
                f1, f2, rho, _ = estimate_astrms_floor(
                    da[m],
                    dd[m],
                    cov_a[m],
                    cov_d[m],
                    cov_x[m],
                    zero[m],
                    zero[m],
                    sigma=sclip,
                )
                if not (np.isfinite(f1) and np.isfinite(f2)):
                    continue
                cell_var1[j, i] = f1**2
                cell_var2[j, i] = f2**2
                cell_cross[j, i] = rho * f1 * f2

        # STEP 2 -- 3x3 nanmedian over supported cells only (all-NaN windows stay
        # NaN; fallback/global values never enter the median).
        def _nanmedian3(grid):
            return generic_filter(
                grid,
                lambda w: np.nanmedian(w) if np.any(np.isfinite(w)) else np.nan,
                size=3,
                mode="constant",
                cval=np.nan,
            )

        # The median DENOISES each measured cell (pools its measured neighbours);
        # it must NOT dilate the footprint. A cell with no own measurement must
        # not become "supported" just because a single neighbour was measured --
        # that would let a below-global value escape the conservatism guard at a
        # cell with no local evidence. Restrict the median output to genuinely
        # measured cells; the rest are filled (interp / global) downstream.
        measured = np.isfinite(cell_var1)  # == isfinite(cell_var2/cell_cross)
        sm_var1 = np.where(measured, _nanmedian3(cell_var1), np.nan)
        sm_var2 = np.where(measured, _nanmedian3(cell_var2), np.nan)
        sm_cross = np.where(measured, _nanmedian3(cell_cross), np.nan)
        supported = measured  # == isfinite(sm_var1) after the restriction

        # Effective window support: sum of the clean-star counts of the MEASURED
        # cells in each 3x3 window (unmeasured cells lend no trust even if they
        # hold stars). Used by the conservatism guard and the correlation shrink.
        n_supported = np.where(measured, cell_n, 0.0)
        neff = convolve(n_supported, np.ones((3, 3)), mode="constant", cval=0.0)

        # STEP 3 -- footprint-aware fill. Linearly interpolate only interior holes
        # within max_interp_cells of real support; griddata-linear never
        # extrapolates beyond the convex hull, and the distance gate prevents
        # bridging wide holes / chip gaps / separate pawprints.
        dist = distance_transform_edt(~supported)
        interior = (~supported) & (dist <= max_interp_cells)
        gy, gx = np.mgrid[0:nby, 0:nbx]

        def _fill(grid):
            out = grid.copy()
            if int(np.sum(supported)) >= 4 and np.any(interior):
                out[interior] = griddata(
                    np.column_stack([gx[supported], gy[supported]]),
                    grid[supported],
                    np.column_stack([gx[interior], gy[interior]]),
                    method="linear",
                )
            return out

        fill_var1 = _fill(sm_var1)
        fill_var2 = _fill(sm_var2)
        fill_cross = _fill(sm_cross)

        # STEP 4 -- per-axis finalize in variance space, then sqrt.
        def _finalize_axis(fill_var, g, scamp):
            v = np.where(np.isfinite(fill_var), fill_var, g**2)  # unfilled -> global
            # A spuriously LOW cell can survive the median in sparse regions: only
            # accept a below-global floor where the window has strong support.
            weak_below = (neff < neff_min) & (v < g**2)
            v = np.where(weak_below, g**2, v)
            lo = (0.5 * g) ** 2
            # Ceiling: weak cells cap at max(scamp, global); well-supported cells
            # may exceed it (real at mosaic seams), up to a 2x sanity bound.
            hi_weak = max(scamp, g) ** 2
            hi_strong = (2.0 * max(scamp, g)) ** 2
            hi = np.where(neff >= neff_min, hi_strong, hi_weak)
            return np.sqrt(np.clip(v, lo, hi))

        grid_a1 = _finalize_axis(fill_var1, g1, scamp1)
        grid_a2 = _finalize_axis(fill_var2, g2, scamp2)

        # STEP 5 -- correlation: convert the smoothed cross-cov to a coefficient
        # via the FINAL per-cell variances, shrink toward the global value by
        # window support (harder than the variances, which are not per-cell
        # shrunk), and clip to [-0.95, 0.95] (=> |rho| < 1 => PSD covariance).
        cross = np.where(np.isfinite(fill_cross), fill_cross, gcorr * g1 * g2)
        denom = grid_a1 * grid_a2
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_local = np.where(denom > 1e-6, cross / denom, gcorr)
        w_corr = neff / (neff + neff_min)
        rho = w_corr * rho_local + (1.0 - w_corr) * gcorr
        grid_corr = np.clip(np.where(np.isfinite(rho), rho, gcorr), -0.95, 0.95)

        # STEP 6 -- nearest-cell (piecewise-constant) per-source lookup. A source
        # with a non-finite coordinate would digitize to an edge cell and pick up
        # an arbitrary value, so fall those (and any non-finite lookup) back to the
        # conservative global scalar.
        ix_all = np.clip(np.digitize(xx, xe) - 1, 0, nbx - 1)
        iy_all = np.clip(np.digitize(yy, ye) - 1, 0, nby - 1)
        new1 = grid_a1[iy_all, ix_all]
        new2 = grid_a2[iy_all, ix_all]
        new_corr = grid_corr[iy_all, ix_all]
        new1 = np.where(np.isfinite(new1) & finite_xy, new1, g1)
        new2 = np.where(np.isfinite(new2) & finite_xy, new2, g2)
        new_corr = np.where(np.isfinite(new_corr) & finite_xy, new_corr, gcorr)

        # Diagnostic grids for the QC plot. mask: 0 = global fallback,
        # 1 = interpolated, 2 = measured.
        mask = np.zeros((nby, nbx), dtype=int)
        mask[np.isfinite(fill_var1) & ~supported] = 1
        mask[supported] = 2
        diag = {
            "extent": (xlo, xhi, ylo, yhi),
            "map1": grid_a1,
            "map2": grid_a2,
            "corr": grid_corr,
            "n": cell_n,
            "raw1": np.sqrt(np.clip(cell_var1, 0.0, None)),
            "mask": mask,
            "g1": g1,
            "g2": g2,
            "gcorr": gcorr,
            "scamp1": scamp1,
            "scamp2": scamp2,
            "cell_arcmin": cell_arcmin,
        }
        return new1, new2, new_corr, diag

    def plot_qc_astrms_floor_map(self, name, diag):
        """
        QC figure for the self-calibrated astrometric floor map.

        Renders the per-cell parameter grids produced by :meth:`_astrms_floor_map`
        on the coadd pixel grid with a LOCKED aspect ratio (true-to-scale), so the
        edge-preserving structure can be inspected directly. Top row: the applied
        ASTRMS1 / ASTRMS2 / ASTRMS_CORR maps; bottom row: the per-cell clean-star
        count, the raw (pre-median) floor, and the support/interp/fallback mask.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap

        log = logging.getLogger(__name__)
        outpath = f"{self.setup.folders['qc_astrometry']}{name}_astr_floormap.pdf"
        if (
            check_file_exists(file_path=outpath, silent=self.setup.silent)
            and not self.setup.overwrite
        ):
            log.info("File already exists, skipping")
            return

        extent = diag["extent"]
        # Shared mas scale for the two axes + the raw floor, so raw vs smoothed and
        # axis 1 vs axis 2 are directly comparable.
        finite_mas = np.concatenate(
            [
                diag["map1"][np.isfinite(diag["map1"])].ravel(),
                diag["map2"][np.isfinite(diag["map2"])].ravel(),
            ]
        )
        if finite_mas.size:
            vmin_mas, vmax_mas = np.nanpercentile(finite_mas, [2, 98])
        else:
            vmin_mas, vmax_mas = 0.0, 1.0
        clim = float(max(0.05, np.nanmax(np.abs(diag["corr"]))))  # symmetric corr

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))

        def _panel(ax, img, title, cmap, vmin, vmax, label):
            im = ax.imshow(
                img,
                origin="lower",
                extent=extent,
                aspect="equal",
                rasterized=True,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(title)
            ax.set_xlabel("X (pix)")
            ax.set_ylabel("Y (pix)")
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(label)
            return im

        _panel(
            axes[0, 0],
            diag["map1"],
            f"ASTRMS1 (global {diag['g1']:.2f}, SCAMP {diag['scamp1']:.2f} mas)",
            "inferno",
            vmin_mas,
            vmax_mas,
            "mas",
        )
        _panel(
            axes[0, 1],
            diag["map2"],
            f"ASTRMS2 (global {diag['g2']:.2f}, SCAMP {diag['scamp2']:.2f} mas)",
            "inferno",
            vmin_mas,
            vmax_mas,
            "mas",
        )
        _panel(
            axes[0, 2],
            diag["corr"],
            f"ASTRMS_CORR (global {diag['gcorr']:+.3f})",
            "RdBu_r",
            -clim,
            clim,
            "correlation",
        )
        _panel(
            axes[1, 0],
            np.where(diag["n"] > 0, diag["n"], np.nan),
            "N stars / cell",
            "viridis",
            0,
            None,
            "N",
        )
        _panel(
            axes[1, 1],
            diag["raw1"],
            "raw floor1 (pre-median)",
            "inferno",
            vmin_mas,
            vmax_mas,
            "mas",
        )
        # Support/interp/fallback mask as a discrete 3-colour panel.
        mask_cmap = ListedColormap(["#d9d9d9", "#6baed6", "#08519c"])
        mask_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], mask_cmap.N)
        im_mask = axes[1, 2].imshow(
            diag["mask"],
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=mask_cmap,
            norm=mask_norm,
            rasterized=True,
        )
        axes[1, 2].set_title("cell support")
        axes[1, 2].set_xlabel("X (pix)")
        axes[1, 2].set_ylabel("Y (pix)")
        cb = plt.colorbar(
            im_mask, ax=axes[1, 2], ticks=[0, 1, 2], fraction=0.046, pad=0.04
        )
        cb.ax.set_yticklabels(["global", "interp", "measured"])

        fig.suptitle(
            f"{name}  -  astrometric floor map "
            f"({diag['cell_arcmin']:g}' cells + 3x3 median)",
            y=0.99,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="tight_layout : falling back to Agg renderer"
            )
            fig.tight_layout()
            fig.savefig(outpath, bbox_inches="tight", dpi=self.setup.qc_plot_dpi)
        plt.close("all")
        log.info(f"Written: {outpath}")

    def paths_qc_plots(self, paths, prefix=""):
        if paths is None:
            return [
                f"{self.setup.folders['qc_photometry']}{fp}.{prefix}.pdf"
                for fp in self.basenames
            ]
        else:
            return paths

    def plot_qc_photerr_internal(self):
        # Import
        import matplotlib.pyplot as plt

        # Print info
        print_header(
            header="QC INTERNAL PHOTOMETRIC ERROR",
            silent=self.setup.silent,
            left=None,
            right=None,
        )
        log = logging.getLogger(__name__)
        tstart = time.time()

        # Create output path
        outpath = (
            f"{self.setup.folders['qc_photometry']}{self.setup.name}.phot.interror.pdf"
        )

        # Check if the file is already there and skip if it is
        if (
            check_file_exists(file_path=outpath, silent=self.setup.silent)
            and not self.setup.overwrite
        ):
            print_elapsed(tstart, logger=log)
            return

        # Get internal photometric error stats
        photerr_internal_dict = self.photerr_internal()

        # Make 1D disperion histograms
        mag_ranges = [0, 14, 15, 16, 17, 18, 25]
        fig, ax_all = plt.subplots(
            nrows=2,
            ncols=3,
            gridspec_kw=dict(
                hspace=0.4, wspace=0.3, left=0.06, right=0.97, bottom=0.1, top=0.97
            ),
            **dict(figsize=(14, 8)),
        )
        ax_all = ax_all.ravel()
        for idx in range(len(mag_ranges) - 1):
            # Grab current axes and sources
            ax = ax_all[idx]
            mag_lo, mag_hi = mag_ranges[idx], mag_ranges[idx + 1]
            idx_phot = (photerr_internal_dict["phot_median"] >= mag_lo) & (
                photerr_internal_dict["phot_median"] < mag_hi
            )

            # Get median photometric error for current bin
            median_photerr_median = np.nanmedian(
                photerr_internal_dict["photerr_median"][idx_phot]
            )

            # Remove axis is no sources are present
            if np.sum(idx_phot) == 0:
                ax.remove()

            # Draw histogram
            ax.hist(
                photerr_internal_dict["phot_err"][idx_phot],
                bins=np.logspace(np.log10(0.0001), np.log10(2), 50),
                ec="black",
                histtype="step",
                lw=2,
            )
            ax.set_xscale("log")

            # Draw median
            ax.axvline(
                np.nanmedian(photerr_internal_dict["phot_err"][idx_phot]),
                c="#1f77b4",
                lw=1.5,
            )
            ax.axvline(median_photerr_median, c="crimson", lw=1.5)

            # Labels and annotations
            ax.set_xlabel("Internal photometric dispersion (mag)")
            ax.set_ylabel("Number of sources")
            ax.annotate(
                f"[{mag_lo:0.1f},{mag_hi:0.1f}) mag",
                xy=(0.02, 0.99),
                xycoords="axes fraction",
                va="top",
                ha="left",
            )
            ax.annotate(
                f"N = {np.sum(idx_phot)}/{len(idx_phot)}",
                xy=(0.98, 0.98),
                xycoords="axes fraction",
                ha="right",
                va="top",
            )
            ax.annotate(
                f"Internal photometric dispersion {np.nanmedian(photerr_internal_dict['phot_err'][idx_phot]):0.4f} mag",
                xy=(0.01, 1.01),
                xycoords="axes fraction",
                ha="left",
                va="bottom",
                c="#1f77b4",
            )
            ax.annotate(
                f"Median photometric error {median_photerr_median:0.4f} mag",
                xy=(0.01, 1.07),
                xycoords="axes fraction",
                ha="left",
                va="bottom",
                c="crimson",
            )

        # Save plot
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="tight_layout : falling back to Agg renderer"
            )
            fig.savefig(outpath, bbox_inches="tight", dpi=self.setup.qc_plot_dpi)
        plt.close("all")

        # Print time
        elapsed = time.time() - tstart
        print_message(
            message=f"\n-> Elapsed time: {elapsed:.2f}s",
            kind="okblue",
            end="\n",
            logger=log,
        )

    def plot_qc_phot_zp(self, paths=None, axis_size=5):
        """Generates ZP QC plot."""

        # Processing info
        print_header(header="QC PHOTOMETRY ZP", silent=self.setup.silent)
        log = logging.getLogger(__name__)
        log.info(f"Generating QC photometry ZP plots for {self.n_files} files")
        tstart = time.time()

        for idx_file in range(self.n_files):
            # Generate path for plot
            path_out = self.paths_qc_plots(paths=paths, prefix="zp")[idx_file]

            # Check if file already exists
            if (
                check_file_exists(file_path=path_out, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                log.info("File already exists, skipping")
                continue
            else:
                log.info(
                    f"Plotting ZP for file {idx_file + 1}/{self.n_files}: {self.names[idx_file]}"
                )
                message_calibration(
                    n_current=idx_file + 1,
                    n_total=self.n_files,
                    name=path_out,
                )

            zp_auto = self.read_from_data_headers(
                keywords=["HIERARCH PYPE ZP MAG_AUTO"]
            )[0][idx_file]
            zperr_auto = self.read_from_data_headers(
                keywords=["HIERARCH PYPE ZP ERR MAG_AUTO"]
            )[0][idx_file]

            # Machine-readable QC record: cross-detector ZP per file (file log).
            log.info(
                f"qc photometry file={self.names[idx_file]} "
                f"zp_median_mag={float(np.nanmedian(zp_auto)):.4f} "
                f"zp_scatter_mag={float(np.nanstd(zp_auto)):.4f}"
            )

            # Create plot
            plot_value_detector(
                values=zp_auto,
                errors=zperr_auto,
                path=path_out,
                ylabel="ZP AUTO (mag)",
                axis_size=axis_size,
                yrange=(np.median(zp_auto) - 0.1, np.median(zp_auto) + 0.1),
                dpi=self.setup.qc_plot_dpi,
            )
            log.info(f"Written: {path_out}")

        # Print time
        elapsed = time.time() - tstart
        print_message(
            message=f"\n-> Elapsed time: {elapsed:.2f}s",
            kind="okblue",
            end="\n",
            logger=log,
        )

    def plot_qc_phot_ref1d(self, paths=None, axis_size=5):
        # Import
        import matplotlib.pyplot as plt
        from astropy.units import Unit
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Processing info
        print_header(header="QC PHOTOMETRY REFERENCE 1D", silent=self.setup.silent)
        log = logging.getLogger(__name__)
        log.info(f"Generating QC photometry reference 1D plots for {len(self)} files")
        tstart = time.time()

        for idx_file in range(len(self)):
            # Generate path for plot
            path_out = self.paths_qc_plots(paths=paths, prefix="phot.1D")[idx_file]

            # Check if file already exists
            if (
                check_file_exists(file_path=path_out, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                log.info("File already exists, skipping")
                continue
            else:
                log.info(
                    f"Plotting phot ref 1D for file {idx_file + 1}/{len(self)}: {self.names[idx_file]}"
                )
                message_calibration(
                    n_current=idx_file + 1,
                    n_total=self.n_files,
                    name=path_out,
                )

            # Get passband
            passband = self.passband[idx_file][0]

            # Get master photometry catalog
            master_phot = self.get_master_photometry()[0]
            mkeep = master_phot.get_purge_index(passband=passband)

            # Fetch magnitude and coordinates for master catalog
            mag_master = master_phot.mag(passband=passband)[0][0][mkeep]
            master_skycoord = master_phot.skycoord[0][0][mkeep]

            # Read tables
            tab_file = self.file2table(file_index=idx_file)

            # Make plot grid
            if len(self.iter_data_hdu[idx_file]) == 1:
                fig, ax_file = plt.subplots(
                    nrows=1, ncols=1, gridspec_kw=None, **dict(figsize=(8, 5))
                )
                ax_file = [ax_file]
            else:
                fig, ax_file = get_plotgrid(
                    layout=self.setup.fpa_layout, xsize=axis_size, ysize=axis_size / 2
                )
                ax_file = ax_file.ravel()

            # Loop over extensions
            for idx_hdu in range(len(self.iter_data_hdu[idx_file])):
                # Grab axes
                ax = ax_file[idx_hdu]

                # Grab current catalog
                tab_hdu = tab_file[idx_hdu]

                # Clean table
                tab_hdu = clean_source_table(table=tab_hdu)

                # Construct skycoordinates
                sc_hdu = SkyCoord(
                    ra=tab_hdu[self.key_ra], dec=tab_hdu[self.key_dec], unit="deg"
                )

                # Xmatch science with reference
                zp_idx, zp_d2d, _ = sc_hdu.match_to_catalog_sky(master_skycoord)

                # Get good indices in reference catalog and in current field
                idx_master = zp_idx[zp_d2d < 1 * Unit("arcsec")]
                idx_final = np.arange(len(zp_idx))[zp_d2d < 1 * Unit("arcsec")]

                # Apply indices filter
                mag_hdu_match = tab_hdu["MAG_AUTO_CAL"][idx_final]
                mag_master_match = mag_master[idx_master]

                # Compute difference between reference and self
                mag_delta = mag_master_match - mag_hdu_match

                # Draw ZP
                ax.axhline(0, zorder=1, c="black", alpha=0.5)

                # KDE for ZP mag interval
                keep = (
                    mag_master_match >= master_phot.mag_lim(passband=passband)[0]
                ) & (mag_master_match <= master_phot.mag_lim(passband=passband)[1])

                # Draw photometry for all matched sources
                ax.scatter(
                    mag_master_match,
                    mag_delta,
                    c="black",
                    s=2,
                    lw=0,
                    alpha=0.4,
                    zorder=0,
                    rasterized=True,
                )

                # Draw for sources within mag limits
                ax.scatter(
                    mag_master_match[keep],
                    mag_delta[keep],
                    c="crimson",
                    s=4,
                    lw=0,
                    alpha=1.0,
                    zorder=0,
                    rasterized=True,
                )

                # Evaluate KDE
                kde = KernelDensity(
                    kernel="gaussian", bandwidth=0.1, metric="euclidean"
                )
                kde_grid = np.arange(np.floor(-1), np.ceil(1), 0.01)
                dens_zp = np.exp(
                    kde.fit((mag_delta[keep]).reshape(-1, 1)).score_samples(  # noqa
                        kde_grid.reshape(-1, 1)
                    )
                )

                # Draw KDE
                ax_kde = ax.twiny()
                ax_kde.plot(dens_zp, kde_grid, lw=1, c="black", alpha=0.8)
                ax_kde.axis("off")

                # Annotate detector ID
                ax.annotate(
                    f"Det.ID: {idx_hdu + 1:0d}",
                    xy=(0.98, 0.04),
                    xycoords="axes fraction",
                    ha="right",
                    va="bottom",
                )

                # Set limits
                ax.set_xlim(10, 18)
                ylim = (-1, 1)
                ax.set_ylim(ylim)
                ax_kde.set_ylim(ylim)

                # Modify axes
                if (idx_hdu < self.setup.fpa_layout[1]) | (len(ax_file) == 1):
                    ax.set_xlabel(
                        f"{self.setup.phot_reference_catalog.upper()} {self.passband[idx_file]} (mag)"
                    )
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if (
                    idx_hdu % self.setup.fpa_layout[0] == self.setup.fpa_layout[0] - 1
                ) | (len(ax_file) == 1):
                    ax.set_ylabel(rf"$\Delta${self.passband[idx_file]} (mag)")
                else:
                    ax.axes.yaxis.set_ticklabels([])

                # Set ticks
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="tight_layout : falling back to Agg renderer"
                )
                fig.savefig(path_out, bbox_inches="tight", dpi=self.setup.qc_plot_dpi)
            plt.close("all")
            log.info(f"Written: {path_out}")

        # Print time
        elapsed = time.time() - tstart
        print_message(
            message=f"\n-> Elapsed time: {elapsed:.2f}s",
            kind="okblue",
            end="\n",
            logger=log,
        )

    def plot_qc_phot_ref2d(self, axis_size=5):
        # Import
        import matplotlib.pyplot as plt
        from astropy.units import Unit
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Processing info
        print_header(header="QC PHOTOMETRY REFERENCE 2D", silent=self.setup.silent)
        log = logging.getLogger(__name__)
        log.info(
            f"Generating QC photometry reference 2D plots for {self.n_files} files"
        )
        tstart = time.time()

        for idx_file in range(self.n_files):
            # Generate output path
            path_out = self.paths_qc_plots(paths=None, prefix="phot.2D")[idx_file]

            # Check if file already exists
            if (
                check_file_exists(file_path=path_out, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                log.info("File already exists, skipping")
                continue
            else:
                log.info(
                    f"Plotting phot ref 2D for file {idx_file + 1}/{self.n_files}: {self.names[idx_file]}"
                )
                message_calibration(
                    n_current=idx_file + 1,
                    n_total=self.n_files,
                    name=path_out,
                )

            # Create figure for current file
            if len(self.iter_data_hdu[idx_file]) == 1:
                fig, ax = plt.subplots(
                    nrows=1, ncols=1, gridspec_kw=None, **dict(figsize=(9, 9))
                )
                ax_file = [ax]
            else:
                fig, ax_file = get_plotgrid(
                    layout=self.setup.fpa_layout, xsize=axis_size, ysize=axis_size
                )
                ax_file = ax_file.ravel()
            cax = fig.add_axes([0.25, 0.92, 0.5, 0.02])

            # Get passband
            passband = self.passband[idx_file][0]

            # Get master photometry catalog
            master_phot = self.get_master_photometry()[0]
            mkeep = master_phot.get_purge_index(passband=passband)

            # Fetch magnitude and coordinates for master catalog
            mag_master = master_phot.mag(passband=passband)[0][0][mkeep]
            skycoord_master = master_phot.skycoord[0][0][mkeep]

            # Keep only soruces within mag limit
            mag_lim = master_phot.mag_lim(self.passband[idx_file])
            keep = (mag_master >= mag_lim[0]) & (mag_master <= mag_lim[1])
            mag_master, skycoord_master = mag_master[keep], skycoord_master[keep]

            # Read sources table for current files
            table_file = self.file2table(file_index=idx_file)

            im = None
            for idx_hdu in range(len(self.iter_data_hdu[idx_file])):
                # Grab axes
                ax = ax_file[idx_hdu]

                # Get current image header
                header = self.image_headers[idx_file][idx_hdu]

                # Grab current catalog
                tab_hdu = table_file[idx_hdu]

                # Clean table
                tab_hdu = clean_source_table(table=tab_hdu)

                # Construct skycoordinates
                sc_hdu = SkyCoord(
                    ra=tab_hdu[self.key_ra], dec=tab_hdu[self.key_dec], unit="deg"
                )

                # Xmatch science with reference
                zp_idx, zp_d2d, _ = sc_hdu.match_to_catalog_sky(skycoord_master)

                # Get good indices in reference catalog and in current field
                idx_master = zp_idx[zp_d2d < 1 * Unit("arcsec")]
                idx_final = np.arange(len(zp_idx))[zp_d2d < 1 * Unit("arcsec")]

                # Apply indices filter
                mag_hdu_match = tab_hdu["MAG_AUTO_CAL"][idx_final]
                mag_master_match = mag_master[idx_master]

                # Compute difference between reference and self
                mag_delta = mag_master_match - mag_hdu_match

                # Grab X/Y coordinates
                x_hdu, y_hdu = (
                    tab_hdu["XWIN_IMAGE"][idx_final],
                    tab_hdu["YWIN_IMAGE"][idx_final],
                )

                # Grid data with nearest neighbors
                grid = grid_value_2d_nn(
                    x=x_hdu,
                    y=y_hdu,
                    values=mag_delta,
                    n_nearest_neighbors=len(x_hdu) if len(x_hdu) < 50 else 50,
                    n_bins_x=header["NAXIS1"] // 100,
                    n_bins_y=header["NAXIS2"] // 100,
                    x_min=1,
                    y_min=1,
                    x_max=header["NAXIS1"],
                    y_max=header["NAXIS2"],
                )

                # Draw
                kwargs = {"vmin": -0.1, "vmax": +0.1, "cmap": plt.get_cmap("RdBu", 20)}
                extent = [1, header["NAXIS1"], 1, header["NAXIS2"]]
                im = ax.imshow(
                    grid, extent=extent, origin="lower", rasterized=True, **kwargs
                )
                # ax.scatter(x_hdu, y_hdu, c=mag_delta, s=7,
                #            lw=0.5, ec="black", **kwargs)

                # Draw contour
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="No contour levels were found within the data range",
                    )
                    cs = ax.contour(
                        grid,
                        np.linspace(-0.1, 0.1, 21),
                        colors="k",
                        origin="lower",
                        extent=extent,
                        vmin=-0.1,
                        vmax=0.1,
                    )
                    ax.clabel(cs, inline=True, fontsize=10, fmt="%0.2f")

                # Annotate detector ID
                ax.annotate(
                    f"Det.ID: {idx_hdu + 1:0d}",
                    xy=(0.02, 1.01),
                    xycoords="axes fraction",
                    ha="left",
                    va="bottom",
                )

                # Modify axes
                if (idx_hdu < self.setup.fpa_layout[1]) | (len(ax_file) == 1):
                    ax_file[idx_hdu].set_xlabel("X (pix)")
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if (
                    idx_hdu % self.setup.fpa_layout[0] == self.setup.fpa_layout[0] - 1
                ) | (len(ax_file) == 1):
                    ax_file[idx_hdu].set_ylabel("Y (pix)")
                else:
                    ax.axes.yaxis.set_ticklabels([])

                # Equal aspect ratio
                ax.set_aspect("equal")

                # Set ticks
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator())

                # Set limits
                ax.set_xlim(1, header["NAXIS1"])
                ax.set_ylim(1, header["NAXIS2"])

            # Add colorbar
            cbar = plt.colorbar(
                im,
                cax=cax,
                orientation="horizontal",
                label="Zero Point (mag)",
                ticks=np.arange(-0.1, 0.11, 0.05),
            )
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")

            # # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="tight_layout : falling back to Agg renderer"
                )
                fig.savefig(path_out, bbox_inches="tight", dpi=self.setup.qc_plot_dpi)
            plt.close("all")
            log.info(f"Written: {path_out}")

        # Print time
        elapsed = time.time() - tstart
        print_message(
            message=f"\n-> Elapsed time: {elapsed:.2f}s",
            kind="okblue",
            end="\n",
            logger=log,
        )

    def plot_qc_psf_2d(self, axis_size=5):
        """Plot 2D spatial FWHM map per detector/extension."""

        # Import
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Processing info
        print_header(header="QC PSF 2D", silent=self.setup.silent)
        log = logging.getLogger(__name__)
        log.info(f"Generating QC PSF 2D plots for {self.n_files} files")
        tstart = time.time()

        # Get magnitude limits for source selection
        master_phot = self.get_master_photometry()[0]

        for idx_file in range(self.n_files):
            # Generate output path
            path_out = (
                f"{self.setup.folders['qc_psf']}{self.basenames[idx_file]}.psf.2D.pdf"
            )

            # Check if file already exists
            if (
                check_file_exists(file_path=path_out, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                log.info("File already exists, skipping")
                continue

            log.info(
                f"Plotting PSF 2D for file {idx_file + 1}/{self.n_files}: "
                f"{self.names[idx_file]}"
            )
            message_calibration(
                n_current=idx_file + 1,
                n_total=self.n_files,
                name=path_out,
            )

            # Get magnitude limits and read tables
            mag_lim = master_phot.mag_lim(self.passband[idx_file])
            table_file = self.file2table(file_index=idx_file)
            n_hdu = len(self.iter_data_hdu[idx_file])

            # Clean all HDUs and collect FWHM for shared colorbar range
            cleaned_tables = []
            fwhm_arrays = []
            for idx_hdu in range(n_hdu):
                tab = clean_source_table(
                    table=table_file[idx_hdu],
                    mag_cal_min=mag_lim[0],
                    mag_cal_max=mag_lim[1],
                )
                cleaned_tables.append(tab)
                fwhm_col = (
                    "FWHM_WORLD_INTERP"
                    if "FWHM_WORLD_INTERP" in tab.colnames
                    else "FWHM_WORLD"
                )
                if len(tab) > 0 and fwhm_col in tab.colnames:
                    fwhm = np.array(tab[fwhm_col], dtype=np.float64) * 3600
                    fwhm_arrays.append(fwhm[np.isfinite(fwhm)])
                else:
                    fwhm_arrays.append(np.array([]))

            # Determine shared colorbar range across all detectors
            fwhm_concat = np.concatenate([f for f in fwhm_arrays if len(f) > 0])
            if len(fwhm_concat) > 0:
                vmin = float(np.percentile(fwhm_concat, 5))
                vmax = float(np.percentile(fwhm_concat, 95))
            else:
                vmin, vmax = 0.5, 2.0

            # Create figure
            if n_hdu == 1:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
                ax_file = [ax]
            else:
                fig, ax_file = get_plotgrid(
                    layout=self.setup.fpa_layout,
                    xsize=axis_size,
                    ysize=axis_size,
                )
                ax_file = ax_file.ravel()
            cax = fig.add_axes([0.25, 0.92, 0.5, 0.02])

            # Plot each detector
            im = None
            for idx_hdu in range(n_hdu):
                ax = ax_file[idx_hdu]
                header = self.image_headers[idx_file][idx_hdu]
                tab = cleaned_tables[idx_hdu]
                fwhm_arcsec = fwhm_arrays[idx_hdu]

                if len(fwhm_arcsec) > 3:
                    x_hdu = np.array(tab["XWIN_IMAGE"], dtype=np.float64)
                    y_hdu = np.array(tab["YWIN_IMAGE"], dtype=np.float64)

                    # Keep only sources with finite FWHM
                    fwhm_col = (
                        "FWHM_WORLD_INTERP"
                        if "FWHM_WORLD_INTERP" in tab.colnames
                        else "FWHM_WORLD"
                    )
                    fwhm_raw = np.array(tab[fwhm_col], dtype=np.float64) * 3600
                    finite = np.isfinite(fwhm_raw)
                    x_hdu, y_hdu, fwhm_raw = (
                        x_hdu[finite],
                        y_hdu[finite],
                        fwhm_raw[finite],
                    )

                    # Grid data
                    grid = grid_value_2d_nn(
                        x=x_hdu,
                        y=y_hdu,
                        values=fwhm_raw,
                        n_nearest_neighbors=min(50, len(x_hdu)),
                        n_bins_x=max(3, header["NAXIS1"] // 100),
                        n_bins_y=max(3, header["NAXIS2"] // 100),
                        x_min=1,
                        y_min=1,
                        x_max=header["NAXIS1"],
                        y_max=header["NAXIS2"],
                    )

                    extent = [1, header["NAXIS1"], 1, header["NAXIS2"]]
                    im = ax.imshow(
                        grid,
                        extent=extent,
                        origin="lower",
                        rasterized=True,
                        vmin=vmin,
                        vmax=vmax,
                        cmap=plt.get_cmap("RdYlBu_r", 20),
                    )

                    # Contour overlay
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="No contour levels were found",
                        )
                        levels = np.linspace(vmin, vmax, 21)
                        cs = ax.contour(
                            grid,
                            levels,
                            colors="k",
                            origin="lower",
                            extent=extent,
                            vmin=vmin,
                            vmax=vmax,
                        )
                        ax.clabel(cs, inline=True, fontsize=10, fmt="%0.2f")

                # Annotate detector ID
                ax.annotate(
                    f"Det.ID: {idx_hdu + 1:0d}",
                    xy=(0.02, 1.01),
                    xycoords="axes fraction",
                    ha="left",
                    va="bottom",
                )

                # Annotate median FWHM
                if len(fwhm_arcsec) > 0:
                    ax.annotate(
                        f'{np.nanmedian(fwhm_arcsec):.2f}"',
                        xy=(0.98, 1.01),
                        xycoords="axes fraction",
                        ha="right",
                        va="bottom",
                    )

                # Axis labels
                if (idx_hdu < self.setup.fpa_layout[1]) | (n_hdu == 1):
                    ax.set_xlabel("X (pix)")
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if (
                    idx_hdu % self.setup.fpa_layout[0] == self.setup.fpa_layout[0] - 1
                ) | (n_hdu == 1):
                    ax.set_ylabel("Y (pix)")
                else:
                    ax.axes.yaxis.set_ticklabels([])

                ax.set_aspect("equal")
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.set_xlim(1, header["NAXIS1"])
                ax.set_ylim(1, header["NAXIS2"])

            # Colorbar
            if im is not None:
                cbar = plt.colorbar(
                    im,
                    cax=cax,
                    orientation="horizontal",
                    label="FWHM (arcsec)",
                )
                cbar.ax.xaxis.set_ticks_position("top")
                cbar.ax.xaxis.set_label_position("top")

            # Save
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="tight_layout : falling back to Agg renderer",
                )
                fig.savefig(path_out, bbox_inches="tight", dpi=self.setup.qc_plot_dpi)
            plt.close("all")
            log.info(f"Written: {path_out}")

        # Print time
        elapsed = time.time() - tstart
        print_message(
            message=f"\n-> Elapsed time: {elapsed:.2f}s",
            kind="okblue",
            end="\n",
            logger=log,
        )

    def plot_qc_psf_1d(self, axis_size=5):
        """Plot FWHM distribution histogram per detector/extension."""

        # Import
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Processing info
        print_header(header="QC PSF 1D", silent=self.setup.silent)
        log = logging.getLogger(__name__)
        log.info(f"Generating QC PSF 1D plots for {self.n_files} files")
        tstart = time.time()

        # Get magnitude limits for source selection
        master_phot = self.get_master_photometry()[0]

        for idx_file in range(self.n_files):
            # Generate output path
            path_out = (
                f"{self.setup.folders['qc_psf']}{self.basenames[idx_file]}.psf.1D.pdf"
            )

            # Check if file already exists
            if (
                check_file_exists(file_path=path_out, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                log.info("File already exists, skipping")
                continue

            log.info(
                f"Plotting PSF 1D for file {idx_file + 1}/{self.n_files}: "
                f"{self.names[idx_file]}"
            )
            message_calibration(
                n_current=idx_file + 1,
                n_total=self.n_files,
                name=path_out,
            )

            # Get magnitude limits and read tables
            mag_lim = master_phot.mag_lim(self.passband[idx_file])
            table_file = self.file2table(file_index=idx_file)
            n_hdu = len(self.iter_data_hdu[idx_file])

            # Clean all HDUs and collect FWHM for shared x-axis range
            fwhm_arrays = []
            for idx_hdu in range(n_hdu):
                tab = clean_source_table(
                    table=table_file[idx_hdu],
                    mag_cal_min=mag_lim[0],
                    mag_cal_max=mag_lim[1],
                )
                fwhm_col = (
                    "FWHM_WORLD_INTERP"
                    if "FWHM_WORLD_INTERP" in tab.colnames
                    else "FWHM_WORLD"
                )
                if len(tab) > 0 and fwhm_col in tab.colnames:
                    fwhm = np.array(tab[fwhm_col], dtype=np.float64) * 3600
                    fwhm_arrays.append(fwhm[np.isfinite(fwhm)])
                else:
                    fwhm_arrays.append(np.array([]))

            # Determine shared x-axis range across all detectors
            fwhm_concat = np.concatenate([f for f in fwhm_arrays if len(f) > 0])
            if len(fwhm_concat) > 0:
                xlim_lo = float(np.percentile(fwhm_concat, 1))
                xlim_hi = float(np.percentile(fwhm_concat, 99))
            else:
                xlim_lo, xlim_hi = 0.5, 2.0

            # Create figure
            if n_hdu == 1:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
                ax_file = [ax]
            else:
                fig, ax_file = get_plotgrid(
                    layout=self.setup.fpa_layout,
                    xsize=axis_size,
                    ysize=axis_size,
                )
                ax_file = ax_file.ravel()

            # Plot each detector
            for idx_hdu in range(n_hdu):
                ax = ax_file[idx_hdu]
                fwhm_arcsec = fwhm_arrays[idx_hdu]

                if len(fwhm_arcsec) > 0:
                    # Histogram
                    ax.hist(
                        fwhm_arcsec,
                        bins=30,
                        range=(xlim_lo, xlim_hi),
                        histtype="stepfilled",
                        fc="dodgerblue",
                        ec="black",
                        alpha=0.5,
                        lw=1.0,
                    )

                    # Median line
                    med = np.nanmedian(fwhm_arcsec)
                    ax.axvline(med, color="crimson", ls="dashed", lw=1.5)

                # Annotate detector ID
                ax.annotate(
                    f"Det.ID: {idx_hdu + 1:0d}",
                    xy=(0.02, 1.01),
                    xycoords="axes fraction",
                    ha="left",
                    va="bottom",
                )

                # Annotate median and N
                if len(fwhm_arcsec) > 0:
                    ax.annotate(
                        f'{med:.2f}" (N={len(fwhm_arcsec)})',
                        xy=(0.98, 1.01),
                        xycoords="axes fraction",
                        ha="right",
                        va="bottom",
                    )

                # Axis labels
                if (idx_hdu < self.setup.fpa_layout[1]) | (n_hdu == 1):
                    ax.set_xlabel("FWHM (arcsec)")
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if (
                    idx_hdu % self.setup.fpa_layout[0] == self.setup.fpa_layout[0] - 1
                ) | (n_hdu == 1):
                    ax.set_ylabel("N")
                else:
                    ax.axes.yaxis.set_ticklabels([])

                # Ticks
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator())

                # Shared x range
                ax.set_xlim(xlim_lo, xlim_hi)

            # Save
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="tight_layout : falling back to Agg renderer",
                )
                fig.savefig(path_out, bbox_inches="tight", dpi=self.setup.qc_plot_dpi)
            plt.close("all")
            log.info(f"Written: {path_out}")

        # Print time
        elapsed = time.time() - tstart
        print_message(
            message=f"\n-> Elapsed time: {elapsed:.2f}s",
            kind="okblue",
            end="\n",
            logger=log,
        )

    def _public_magerr_scale(self, header: fits.Header) -> tuple[np.ndarray, float]:
        """Per-aperture MAGERR inflation for the public catalog of one coadd.

        The resampling ratio is the coadd pixel scale (from ``header``, the
        tile image header) over ``setup.native_pixel_scale``.

        Returns
        -------
        tuple[np.ndarray, float]
            Inflation factor per entry of ``setup.apertures`` and the scale ratio.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            scale_out = float(np.mean(proj_plane_pixel_scales(WCS(header)))) * 3600.0
        scale_ratio = scale_out / self.setup.native_pixel_scale
        factor = aperture_error_inflation(
            diameters=self.setup.apertures,
            kernel=self.setup.resampling_kernel,
            scale_ratio=scale_ratio,
            noise_scale=self.setup.photometric_error_noise_scale,
            background_beta=self.setup.photometric_error_background_beta,
            background_dmax=self.setup.photometric_error_background_dmax,
        )
        return factor, scale_ratio

    def _public_sources(
        self,
        table,
        table_stats,
        table_class,
        data_header: fits.Header,
        photerr_internal: float,
        mag_saturation: float,
        label: str,
    ):
        """
        Source-list rows of one data array: the calibrated detections joined with
        their statistics and interpolated classification, converted with
        ``detections_to_sources``.

        Returns
        -------
        tuple[QTable, float | None]
            The sources and the correlated-noise scale ratio (None if off).
        """
        log = logging.getLogger(__name__)
        len_table_full, len_table_stats = len(table), len(table_stats)
        len_table_cls = len(table_class)
        log.info(
            f"{label}: {len_table_full} sources, {len_table_stats} statistics, "
            f"{len_table_cls} classification rows"
        )
        if min(len_table_full, len_table_cls) == 0:
            raise PipelineValueError(f"{label}: no sources", logger=log)

        # Stats and source table need to have same length
        if len_table_full != len_table_stats:
            raise PipelineValueError(
                f"{label}: source ({len_table_full}) and stats ({len_table_stats}) "
                f"table do not have same length.",
                logger=log,
            )

        # Stack source and stats tables
        table = thstack([table, table_stats], join_type="exact")

        # Length of source and classification table must be equal within 0.5%
        if np.abs(len_table_full / len_table_cls - 1) > 0.005:
            raise PipelineValueError(
                f"{label}: source ({len_table_full}) and classification "
                f"({len_table_cls}) tables do not have similar length.",
                logger=log,
            )

        # If the check passes, make a match
        xy_cs = np.column_stack((table_class["XWIN_IMAGE"], table_class["YWIN_IMAGE"]))
        xy_full = np.column_stack((table["XWIN_IMAGE"], table["YWIN_IMAGE"]))
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(xy_cs)
        distances, _ = nbrs.kneighbors(xy_full)
        len_table_final = int(np.sum(distances.ravel() < 0.1))
        log.info(f"{label}: {len_table_final} classification matches")

        # New table length must be similar to the original table length
        if np.abs(len_table_full / len_table_final - 1) > 0.005:
            raise PipelineValueError(
                f"{label}: matching of source and classification table failed. "
                f"New table length ({len_table_final}) is not similar "
                f"the original table length ({len_table_full}).",
                logger=log,
            )

        # Interpolate source classification; unmatched sources are kept with
        # NaN (-> CFLG 256) instead of being dropped
        interpolate_classification(table, table_class)

        # Correlated-noise inflation of the aperture errors (per aperture); the
        # image data header carries the WCS of the resampled data
        magerr_scale, scale_ratio = None, None
        if self.setup.photometric_error_correlated_noise:
            magerr_scale, scale_ratio = self._public_magerr_scale(data_header)
            log.info(
                f"{label}: correlated-noise MAGERR inflation "
                f"({self.setup.resampling_kernel}, scale ratio "
                f"{scale_ratio:.4f}, c={self.setup.photometric_error_noise_scale}, "
                f"beta={self.setup.photometric_error_background_beta}): "
                f"{np.array2string(magerr_scale, precision=3)}"
            )

        # Convert to the source-list format
        sources = detections_to_sources(
            table,
            photerr_internal=photerr_internal,
            apertures=self.setup.apertures,
            mag_saturation=mag_saturation,
            survey_name=self.setup.survey_name,
            magerr_scale=magerr_scale,
        )
        log.info(f"{label}: {len(sources)} sources after conversion")
        return sources, scale_ratio

    def build_public_catalog(
        self, photerr_internal: float, paths_images: list[str] | None = None
    ):
        """
        Write the source list ``<name>.sources.fits`` (SCIENCE.SRCTBL) of each image.

        The list holds the pipeline detections only (D12); 2MASS rows for bright
        stars are added by the VISIONS fold. Per data array of the image (one for a
        tile, 16 for a stack) the source table is joined with its statistics table
        and the interpolated classification (``_public_sources``); the tables are
        written with ``write_source_list`` through a local temporary file. Only
        headers of the image are read, no image data.

        Parameters
        ----------
        photerr_internal : float
            Internal photometric error floor (mag), written to PHOTIERR.
        paths_images : list[str] | None, optional
            Image of each catalog (pawprints: the pawprint products); default the
            catalog path without ``.full.fits.ctab`` (tile, stacks). The source
            list is written next to the image.

        Raises
        ------
        PipelineValueError
            If images, statistics or classification tables are missing, or the
            data arrays and tables do not match.
        """
        # Fetch log
        log = logging.getLogger(__name__)

        # Processing info
        print_header(header="PUBLIC CATALOG", silent=self.setup.silent)
        tstart = time.time()

        # Master photometry (passband translation and bright limit only)
        master_phot = self.get_master_photometry()

        # Find tile images, statistics and classification tables
        paths_tiles = paths_images or [
            x.replace(".full.fits.ctab", ".fits") for x in self.paths_full
        ]
        if len(paths_tiles) != self.n_files:
            raise PipelineValueError(
                f"{len(paths_tiles)} images for {self.n_files} catalogs", logger=log
            )
        paths_stats = [f"{p}.stats" for p in self.paths_full]
        paths_cls = [
            x.replace(".full.", ".cs.").replace(".ctab", ".tab")
            for x in self.paths_full
        ]
        for paths, kind in (
            (paths_tiles, "Images"),
            (paths_stats, "Statistics tables"),
            (paths_cls, "Classification tables"),
        ):
            if self.n_files != np.sum([os.path.exists(p) for p in paths]):
                raise PipelineValueError(f"{kind} not found", logger=log)

        # Instantiate statistics and classification tables
        statstables = SextractorCatalogs(file_paths=paths_stats, setup=self.setup)
        tables_class = SourceCatalogs(setup=self.setup, file_paths=paths_cls)

        # Loop over files
        for idx_file in range(self.n_files):
            # Log current filename
            log.info(f"Processing file: {self.paths_full[idx_file]}")

            # Create output path
            path_tile = paths_tiles[idx_file]
            path_out = path_tile.replace(".fits", ".sources.fits")
            log.info(f"Creating source list: {path_out}")

            # Check if the file is already there and skip if it is
            if (
                check_file_exists(file_path=path_out, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                continue

            # Print processing info
            if not self.setup.silent:
                message_calibration(
                    n_current=idx_file + 1,
                    n_total=self.n_files,
                    name=path_out,
                    d_current=None,
                    d_total=None,
                )

            # Grab passbands
            passband = self.passband[idx_file]
            passband_2mass = master_phot.translate_passband(passband)
            log.info(f"Passband input: {passband}; passband 2MASS: {passband_2mass}")

            # Image headers (no image data): primary and one header per data array
            with fits.open(path_tile) as hdul_image:
                image_header = hdul_image[0].header.copy()
                data_headers = [
                    h.header.copy()
                    for h in hdul_image
                    if h.is_image and h.header.get("NAXIS") == 2
                ]

            # Source, statistics and classification tables, one per data array
            tables_file = self.file2table(file_index=idx_file)
            tables_stats = statstables.file2table(file_index=idx_file)
            tables_cls = tables_class.file2table(file_index=idx_file)
            counts = {
                len(data_headers),
                len(tables_file),
                len(tables_stats),
                len(tables_cls),
            }
            if len(counts) != 1:
                raise PipelineValueError(
                    f"{path_tile}: {len(data_headers)} data arrays, "
                    f"{len(tables_file)} source, {len(tables_stats)} statistics and "
                    f"{len(tables_cls)} classification tables",
                    logger=log,
                )

            sources_file, scale_ratio = [], None
            for idx_hdu, (table, table_stats, table_class, data_header) in enumerate(
                zip(tables_file, tables_stats, tables_cls, data_headers)
            ):
                sources, scale_ratio = self._public_sources(
                    table=table,
                    table_stats=table_stats,
                    table_class=table_class,
                    data_header=data_header,
                    photerr_internal=photerr_internal,
                    mag_saturation=master_phot.mag_lim_lo(passband=passband_2mass),
                    label=f"{self.paths_full[idx_file]} [table {idx_hdu + 1}]",
                )
                sources_file.append(sources)
            hdr_cat = fits.getheader(
                self.paths_full[idx_file], self.iter_data_hdu[idx_file][0]
            )

            # Primary cards of the error model
            primary_cards = [
                (
                    "PHOTIERR",
                    round(float(photerr_internal), 4),
                    "Internal photometric error (mag)",
                )
            ]
            if self.setup.photometric_error_correlated_noise:
                primary_cards += [
                    (
                        "PHOTCNK",
                        self.setup.resampling_kernel,
                        "MAGERR correlated-noise: resampling kernel",
                    ),
                    (
                        "PHOTCNR",
                        round(float(scale_ratio), 4),
                        "MAGERR correlated-noise: out/in pixel scale",
                    ),
                    (
                        "PHOTCNC",
                        round(float(self.setup.photometric_error_noise_scale), 4),
                        "MAGERR correlated-noise: noise scale c",
                    ),
                    (
                        "PHOTCNB",
                        round(float(self.setup.photometric_error_background_beta), 5),
                        "MAGERR correlated-noise: background beta",
                    ),
                    (
                        "PHOTCND",
                        round(float(self.setup.photometric_error_background_dmax), 2),
                        "MAGERR correlated-noise: beta cap diameter (px)",
                    ),
                ]
            table_cards = [
                (
                    "REFMAGLO",
                    round(float(hdr_cat["HIERARCH PYPE REFMAG LO"]), 3),
                    "2MASS bright limit of the calibration (mag)",
                )
            ]

            # Write via a local temporary file and move into place
            path_tmp = make_path_system_tempfile(
                prefix="sources_", suffix=".fits", base_dir=self.setup.local_cache_dir
            )
            try:
                write_source_list(
                    sources_file,
                    path_tmp,
                    os.path.basename(path_tile),
                    image_header,
                    passband,
                    primary_cards,
                    table_cards,
                    data_headers=data_headers,
                )
                move_file(path_tmp, path_out)
            finally:
                remove_file(path_tmp)
            n_sources = sum(len(x) for x in sources_file)
            log.info(f"Written {n_sources} sources to {path_out}")

        # Print time
        elapsed = time.time() - tstart
        print_message(
            message=f"\n-> Elapsed time: {elapsed:.2f}s",
            kind="okblue",
            end="\n",
            logger=log,
        )
