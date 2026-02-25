import os
import pickle
import time
import warnings
from itertools import repeat

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.table import Table
from astropy.table import hstack as thstack
from astropy.table import vstack as tvstack
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity, NearestNeighbors

from vircampype.data.cube import ImageCube
from vircampype.fits.tables.sources import SourceCatalogs
from vircampype.pipeline.log import PipelineLog
from vircampype.tools.astromatic import *
from vircampype.tools.fitstools import *
from vircampype.tools.imagetools import *
from vircampype.tools.mathtools import *
from vircampype.tools.messaging import *
from vircampype.tools.messaging import message_qc_astrometry
from vircampype.tools.miscellaneous import *
from vircampype.tools.photometry import *
from vircampype.tools.plottools import *
from vircampype.tools.systemtools import *
from vircampype.tools.tabletools import *


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
        log = PipelineLog()

        # Processing info
        print_header(header="SCAMP", silent=self.setup.silent)
        tstart = time.time()
        log.info(f"Running scamp on {self.n_files} files:\n{self.basenames2log()}")

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

        # Check for external headers
        ehdrs = [os.path.isfile(p) for p in self._scamp_header_paths(joined=False)]

        # If already available, skip
        if np.sum(ehdrs) == len(self):
            print("Scamp headers already exist")
            return

        # Load astrometric reference
        astrefact_name = self.get_master_astrometry().paths_full[0]

        # Add to log
        log.info(f"Astrometric reference: {astrefact_name}")

        # Load Scamp setup
        scs = ScampSetup(setup=self.setup)

        # Load preset
        options = yml2config(
            path_yml=scs.preset_config,
            nthreads=self.setup.n_jobs,
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

        # Wait for any other scamp instances to finish
        log.info("Waiting for other scamp instances to finish...")
        wait_for_no_process(executable="scamp", poll_s=2.0, timeout_s=1800)

        # Run Scamp
        stdout, stderr = run_command_shell(cmd, silent=False)

        # Add stdout and stderr to log
        log.info(f"Scamp stdout:\n{stdout}")
        log.info(f"Scamp stderr:\n{stderr}")

        # Some basic QC on XML
        xml = Table.read(path_xml, format="votable", table_id=1)
        if np.max(xml["AstromSigma_Internal"].data.ravel() * 1000) > 100:
            raise ValueError("Astrometric solution may be crap, please check")

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
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
                continue

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
                __verbose = False
                if __verbose:
                    print(files.paths_full)
                tab = clean_source_table(
                    table=tab,
                    image_header=header,
                    flux_max=header["SEXSATLV"] * 0.8,
                    min_distance_to_edge=10,
                    nndis_limit=5,
                    min_fwhm=0.8,
                    max_fwhm=6.0,
                    max_ellipticity=0.25,
                    verbose=__verbose,
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
                zp_all = get_zeropoint(
                    skycoord1=SkyCoord(tab[self.key_ra], tab[self.key_dec], unit="deg"),
                    mag1=tab["MAG_AUTO"],
                    magerr1=tab["MAGERR_AUTO"],
                    skycoord2=skycoord_master,
                    mag2=mag_master,
                    magerr2=magerr_master,
                    mag_limits_ref=master_phot.mag_lim(passband=passband),
                    method=method,
                )

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
                        (header["NAXIS1"], header["NAXIS2"]),
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

            # QC plot
            if self.setup.qc_plots:
                msf = MasterIlluminationCorrection(setup=self.setup, file_paths=outpath)
                msf.qc_plot2d(paths=None, axis_size=5)

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def calibrate_photometry(self):
        # Fetch log
        log = PipelineLog()

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
                log.info(f"File already exists, skipping")
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

                # Compute weights with a 2 arcmin gaussian kernel
                log.info("Computing weights for nearest neighbors")
                nn_weights = (
                    np.exp(-0.5 * (nn_dis / 360) ** 2)
                    / table["MAGERR_AUTO"][:, np.newaxis] ** 2
                )
                nn_weights[np.isnan(nn_weights)] = 0

                # Split matrices at same indices as table
                nn_dis_chunks = np.split(nn_dis, np.cumsum(len_table_chunks)[:-1])
                nn_weights_chunks = np.split(
                    nn_weights, np.cumsum(len_table_chunks)[:-1]
                )

                # Interpolate values in loop
                for par in ["MAG_APER_COR", "ELLIPTICITY", "FWHM_WORLD"]:
                    log.info(f"Interpolating {par} values")

                    # Grab data for nearest neighbors
                    nn_data = table_clean[par][nn_idx]
                    nn_data_chunks = np.split(nn_data, np.cumsum(len_table_chunks)[:-1])

                    vals, vals_std, n_sources, max_dists = [], [], [], []
                    for data, weights, dis in zip(
                        nn_data_chunks, nn_weights_chunks, nn_dis_chunks
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
                        vals.append(np.average(data, axis=1, weights=weights))
                        vals_std.append(
                            np.sqrt(
                                np.average(
                                    (data - vals[-1][:, np.newaxis]) ** 2,
                                    axis=1,
                                    weights=weights,
                                )
                            )
                        )

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
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def plot_qc_astrometry_1d(self, axis_size=5):
        # Import
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Processing info
        print_header(header="QC ASTROMETRY 1D", silent=self.setup.silent)
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
                continue

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
                fig1.savefig(outpath_sep, bbox_inches="tight")
                fig2.savefig(outpath_ang, bbox_inches="tight")
            plt.close("all")

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def plot_qc_astrometry_2d(
        self, axis_size=5, key_x="XWIN_IMAGE", key_y="YWIN_IMAGE"
    ):
        # Import
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Processing info
        print_header(header="QC ASTROMETRY 2D", silent=self.setup.silent)
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
                continue

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
                    grid, extent=extent, origin="lower", **kwargs
                )
                ax_all[idx_hdu].scatter(
                    x_hdu,
                    y_hdu,
                    c=sep.mas,
                    s=5,
                    lw=0.5,
                    ec="black",
                    alpha=0.5,
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
                fig.savefig(outpath, bbox_inches="tight")
            plt.close("all")

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )


class PhotometricCalibratedSextractorCatalogs(AstrometricCalibratedSextractorCatalogs):
    def __init__(self, setup, file_paths=None):
        super(PhotometricCalibratedSextractorCatalogs, self).__init__(
            file_paths=file_paths, setup=setup
        )

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
        with Parallel(
            n_jobs=self.setup.n_jobs, prefer=self.setup.joblib_backend
        ) as parallel:
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
            _, phot_median, phot_err = sigma_clipped_stats(matched_phot, axis=1)
            _, photerr_median, _ = sigma_clipped_stats(matched_photerr, axis=1)

        # Return
        return phot_median, phot_err, photerr_median

    def photerr_internal(self):
        # Create pickle path
        pickle_path = f"{self.setup.folders['temp']}photerr_interal.p"

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
            tstart = time.time()

            # Determine photometric statistics
            phot_median, phot_err, photerr_median = self._photerr_internal_all()

            # Get the 5% brightest sources
            good = phot_median >= self.setup.reference_mag_lo
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

            # Print error
            print_message(
                message=f"err = {photerr_internal_dict['photerr_internal']:0.4f} mag"
            )
            print_message(
                message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
                kind="okblue",
                end="\n",
            )

        # Get median error of those
        return photerr_internal_dict

    def build_statistics_tables(self):
        # Import
        from vircampype.fits.images.common import FitsImages

        # Processing info
        print_header(header="STATISTICS TABLES", silent=self.setup.silent, right=None)
        tstart = time.time()

        for idx_file in range(self.n_files):
            # Create output path
            path_out = f"{self.paths_full[idx_file]}.stats"

            # Check if file already exists
            if (
                check_file_exists(file_path=path_out, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
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

            # Find files
            path_mjd = self.paths_full[idx_file].replace(
                ".full.fits.ctab", ".mjd.eff.fits"
            )
            path_exptime = self.paths_full[idx_file].replace(
                ".full.fits.ctab", ".exptime.fits"
            )
            path_nimg = self.paths_full[idx_file].replace(
                ".full.fits.ctab", ".nimg.fits"
            )
            path_astrms1 = self.paths_full[idx_file].replace(
                ".full.fits.ctab", ".astrms1.fits"
            )
            path_astrms2 = self.paths_full[idx_file].replace(
                ".full.fits.ctab", ".astrms2.fits"
            )
            path_weight = self.paths_full[idx_file].replace(
                ".full.fits.ctab", ".weight.fits"
            )

            # Check if files are available
            if not (
                os.path.isfile(path_mjd)
                and os.path.isfile(path_exptime)
                and os.path.isfile(path_nimg)
                and os.path.isfile(path_astrms1)
                and os.path.isfile(path_astrms2)
            ):
                raise ValueError("Matches for image statistics not found")

            # Instantiate
            image_mjdeff = FitsImages(file_paths=path_mjd, setup=self.setup)

            # Open current table file
            hdul_in = fits.open(self.paths_full[idx_file], mode="readonly")
            hdul_out = hdul_in.copy()

            # Loop over extensions
            # TODO: Can this be replace with image_mjdeff.iter_data_hdu[0]?
            for idx_hdu_self, idx_hdu_stats in zip(
                self.iter_data_hdu[idx_file], range(image_mjdeff.n_data_hdu[0])
            ):
                # Read table
                table_hdu = self.filehdu2table(
                    file_index=idx_file, hdu_index=idx_hdu_self
                )

                # Read stats
                try:
                    mjdeff = fits.getdata(path_mjd, idx_hdu_stats)
                    exptime = fits.getdata(path_exptime, idx_hdu_stats)
                    nimg = fits.getdata(path_nimg, idx_hdu_stats)
                    astrms1 = fits.getdata(path_astrms1, idx_hdu_stats)
                    astrms2 = fits.getdata(path_astrms2, idx_hdu_stats)
                    weight = fits.getdata(path_weight, idx_hdu_stats)
                except IndexError:
                    mjdeff = fits.getdata(path_mjd, idx_hdu_stats + 1)
                    exptime = fits.getdata(path_exptime, idx_hdu_stats + 1)
                    nimg = fits.getdata(path_nimg, idx_hdu_stats + 1)
                    astrms1 = fits.getdata(path_astrms1, idx_hdu_stats + 1)
                    astrms2 = fits.getdata(path_astrms2, idx_hdu_stats + 1)
                    weight = fits.getdata(path_weight, idx_hdu_stats + 1)

                # Renormalize weight
                weight /= np.median(weight)

                # Obtain wcs for statistics images (they all have the same projection)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    wcs_stats = WCS(header=image_mjdeff.headers_data[0][idx_hdu_stats])

                # Convert to X/Y
                xx, yy = wcs_stats.wcs_world2pix(
                    table_hdu[self.key_ra], table_hdu[self.key_dec], 0
                )
                xx_image, yy_image = xx.astype(int), yy.astype(int)

                # Mark bad data
                bad = (
                    (xx_image >= mjdeff.shape[1])
                    | (xx_image < 0)
                    | (yy_image >= mjdeff.shape[0])
                    | (yy_image < 0)
                )

                # Just to be sort of safe,
                # let's say we can't have more than 5% of sources outside the edges
                if sum(bad) / len(bad) > 0.05:
                    raise ValueError(
                        f"Too many sources are close to the image edge ({sum(bad)}/{len(bad)}). "
                        f"Please check for issues. (file: {self.paths_full[idx_file]}, TableHDU: {idx_hdu_self})"
                    )

                # Reset bad coordinates to 0/0
                xx_image[bad], yy_image[bad] = 0, 0

                # Get values for each source from data arrays
                mjdeff_sources, exptime_sources, nimg_sources = (
                    mjdeff[yy_image, xx_image],
                    exptime[yy_image, xx_image],
                    nimg[yy_image, xx_image],
                )
                astrms_sources1, astrms_sources2, weight_sources = (
                    astrms1[yy_image, xx_image],
                    astrms2[yy_image, xx_image],
                    weight[yy_image, xx_image],
                )
                # Mask bad sources
                bad &= weight_sources < 0.0001
                mjdeff_sources[bad], exptime_sources[bad] = np.nan, 0.0
                astrms_sources1[bad], astrms_sources2[bad] = np.nan, np.nan
                nimg_sources[bad] = 0

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
                    ]
                )

                hdul_out[idx_hdu_self] = fits.BinTableHDU.from_columns(
                    new_cols, header=hdul_in[idx_hdu_self].header
                )

            # Write to new output file
            hdul_out.writeto(path_out, overwrite=self.setup.overwrite)
            hdul_out.close()

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

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
            fig.savefig(outpath, bbox_inches="tight")
        plt.close("all")

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def plot_qc_phot_zp(self, paths=None, axis_size=5):
        """Generates ZP QC plot."""

        # Processing info
        print_header(header="QC PHOTOMETRY ZP", silent=self.setup.silent)
        tstart = time.time()

        for idx_file in range(self.n_files):
            # Generate path for plot
            path_out = self.paths_qc_plots(paths=paths, prefix="zp")[idx_file]

            # Check if file already exists
            if (
                check_file_exists(file_path=path_out, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                continue
            else:
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

            # Create plot
            plot_value_detector(
                values=zp_auto,
                errors=zperr_auto,
                path=path_out,
                ylabel="ZP AUTO (mag)",
                axis_size=axis_size,
                yrange=(np.median(zp_auto) - 0.1, np.median(zp_auto) + 0.1),
            )

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def plot_qc_phot_ref1d(self, paths=None, axis_size=5):
        # Import
        import matplotlib.pyplot as plt
        from astropy.units import Unit
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Processing info
        print_header(header="QC PHOTOMETRY REFERENCE 1D", silent=self.setup.silent)
        tstart = time.time()

        for idx_file in range(len(self)):
            # Generate path for plot
            path_out = self.paths_qc_plots(paths=paths, prefix="phot.1D")[idx_file]

            # Check if file already exists
            if (
                check_file_exists(file_path=path_out, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                continue
            else:
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
                fig.savefig(path_out, bbox_inches="tight")
            plt.close("all")

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def plot_qc_phot_ref2d(self, axis_size=5):
        # Import
        import matplotlib.pyplot as plt
        from astropy.units import Unit
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Processing info
        print_header(header="QC PHOTOMETRY REFERENCE 2D", silent=self.setup.silent)
        tstart = time.time()

        for idx_file in range(self.n_files):
            # Generate output path
            path_out = self.paths_qc_plots(paths=None, prefix="phot.2D")[idx_file]

            # Check if file already exists
            if (
                check_file_exists(file_path=path_out, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                continue
            else:
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
                im = ax.imshow(grid, extent=extent, origin="lower", **kwargs)
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
                fig.savefig(path_out, bbox_inches="tight")
            plt.close("all")

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def build_public_catalog(self, photerr_internal: float):
        # Fetch log
        log = PipelineLog()

        # Processing info
        print_header(header="PUBLIC CATALOG", silent=self.setup.silent)
        tstart = time.time()

        # Load master photometry catalog
        from vircampype.fits.images.common import FitsImages
        from vircampype.fits.tables.sources import MasterPhotometry2Mass

        master_phot = self.get_master_photometry()  # type: MasterPhotometry2Mass

        # Find classification tables
        paths_cls = [
            x.replace(".full.", ".cs.").replace(".ctab", ".tab")
            for x in self.paths_full
        ]

        # Check if classification tables exist
        if self.setup.source_classification:
            if self.n_files != np.sum([os.path.exists(p) for p in paths_cls]):
                raise ValueError("Classifiation tables not found")

        # Instantiate classification tables
        tables_class = None
        if self.setup.source_classification:
            tables_class = SourceCatalogs(setup=self.setup, file_paths=paths_cls)

        # Find weight images
        paths_weights = [
            x.replace(".full.fits.ctab", ".weight.fits") for x in self.paths_full
        ]

        # Check if weight images exist
        if self.n_files != np.sum([os.path.exists(p) for p in paths_weights]):
            raise ValueError("Weight images not found")

        # Find stats tables
        paths_stats = [f"{p}.stats" for p in self.paths_full]

        # Check if stats tables exist
        if self.n_files != np.sum([os.path.exists(p) for p in paths_stats]):
            raise ValueError("Weight images not found")

        # Instantiate weights
        weightimages = FitsImages(file_paths=paths_weights, setup=self.setup)

        # Instantiate statistics tables
        statstables = SextractorCatalogs(file_paths=paths_stats, setup=self.setup)

        # Loop over self and merge
        for idx_file in range(self.n_files):
            # Log current filename
            log.info(f"Processing file: {self.paths_full[idx_file]}")

            # Create output path
            path_out = self.paths_full[idx_file].replace(".ctab", ".ptab")
            if path_out == self.paths_full[idx_file]:
                raise ValueError("Can't set name of public catalog")
            log.info(f"Creating public catalog: {path_out}")

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=path_out, silent=self.setup.silent):
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

            # Load current table in HDUList
            hdulist_in = fits.open(self.paths_full[idx_file])
            log.info(f"Found {len(hdulist_in)} HDUs in current file")

            # Load source tables in current file
            tables_file = self.file2table(file_index=idx_file)
            log.info(f"Loaded {len(tables_file)} input tables")

            # Load stats tables
            tables_stats = statstables.file2table(file_index=idx_file)
            log.info(f"Loaded {len(tables_stats)} stats tables")

            # Load classification tables if set
            tables_class_file = None
            if self.setup.source_classification:
                tables_class_file = tables_class.file2table(file_index=idx_file)
                log.info(f"Loaded {len(tables_class_file)} classification tables")

            # Read master table
            table_2mass = master_phot.file2table(file_index=idx_file)[0]
            log.info(f"Loaded 2MASS table with {len(table_2mass)} sources")
            table_2mass["QFLG_PB"] = master_phot.qflags(passband=passband_2mass)[0][0]

            # Fill masked columns with NaNs
            table_2mass = fill_masked_columns(table_2mass, fill_value=np.nan)

            # Clean master table
            allowed_qflags, allowed_cflags = "ABCD", "0cd"
            log.info(
                f"Cleaning master table with qflags: {allowed_qflags}, "
                f"cflags: {allowed_cflags}"
            )
            keep_master = master_phot.get_purge_index(
                passband=passband_2mass,
                allowed_qflags=allowed_qflags,
                allowed_cflags=allowed_cflags,
            )
            table_2mass_clean = table_2mass.copy()[keep_master]
            log.info(f"Cleaned master table; sources left: {len(table_2mass_clean)}")

            # Work in each extension
            for tidx, widx in zip(
                range(len(tables_file)), weightimages.iter_data_hdu[idx_file]
            ):
                # Determine lengths
                len_table_full = len(tables_file[tidx])
                len_table_stats = len(tables_stats[tidx])

                # Log current extension
                log.info(f"Working on extension {tidx + 1}")
                log.info(f"Source table length: {len_table_full}")
                log.info(f"Stats table length: {len_table_stats}")

                # Stats and source table need to have same length
                if len_table_full != len_table_stats:
                    msg = (
                        f"Source ({len_table_full}) and stats ({len_table_stats}) "
                        f"table do not have same length."
                    )
                    log.error(msg)
                    raise ValueError(msg)

                # Stack source and stats tables
                tables_file[tidx] = thstack(
                    [tables_file[tidx], tables_stats[tidx]], join_type="exact"
                )
                log.info(f"Stacked source and stats tables")

                # Length of source and classification table must be equal within 0.1%
                if self.setup.source_classification:
                    len_table_cls = len(tables_class_file[tidx])

                    # Check length of the source and classification table
                    log.info("Checking classification table length")
                    log.info(f"Length of source table: {len_table_full}")
                    log.info(f"Length of classification table: {len_table_cls}")
                    if np.abs(len_table_full / len_table_cls - 1) > 0.005:
                        msg = (
                            f"Source ({len_table_full}) and classification "
                            f"({len_table_cls}) tables do not have similar length."
                        )
                        log.error(msg)
                        raise ValueError(msg)

                    # If the check passes, make a match
                    log.info("Matching classification and source table")
                    x_full = tables_file[tidx]["XWIN_IMAGE"]
                    y_full = tables_file[tidx]["YWIN_IMAGE"]
                    x_cs = tables_class_file[tidx]["XWIN_IMAGE"]
                    y_cs = tables_class_file[tidx]["YWIN_IMAGE"]
                    xy_cs = np.column_stack((x_cs, y_cs))
                    xy_full = np.column_stack((x_full, y_full))
                    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(xy_cs)
                    distances, indices = nbrs.kneighbors(xy_full)
                    distances, indices = distances.ravel(), indices.ravel()

                    # Apply a distance threshold to filter matches
                    distance_threshold = 0.1  # Adjust this threshold based on your data
                    log.info(f"Applying distance threshold of {distance_threshold}")
                    good_matches = distances < distance_threshold
                    good_indices = indices[good_matches]
                    len_table_final = len(good_indices)
                    log.info(f"Good matches: {len_table_final}")

                    # New table length must be similar to the original table length
                    log.info("Checking final table length")
                    log.info(f"Length of source table: {len_table_full}")
                    log.info(f"Length of matched table: {len_table_final}")
                    if np.abs(len_table_full / len_table_final - 1) > 0.005:
                        msg = (
                            f"Matching of source and classification table failed. "
                            f"New table length ({len_table_final}) is not similar "
                            f"the original table length ({len_table_full})."
                        )
                        log.error(msg)
                        raise ValueError(msg)

                    # Filter the full and cs tables based on the good matches
                    log.info("Applying match indices")
                    tables_file[tidx] = tables_file[tidx][good_matches]
                    tables_class_file[tidx] = tables_class_file[tidx][good_indices]

                    # Interpolate source classification
                    log.info(f"Interpolating source classification")
                    interpolate_classification(
                        tables_file[tidx], tables_class_file[tidx]
                    )

                # Read weight
                log.info("Reading weight")
                weight_data, weight_hdr = fits.getdata(
                    paths_weights[idx_file], widx, header=True
                )

                # Merge with 2MASS
                # TODO: Add logging here and check speed improvements
                log.info("Merging with 2MASS")
                tables_file[tidx] = merge_with_2mass(
                    table=tables_file[tidx],
                    table_2mass=table_2mass,
                    table_2mass_clean=table_2mass_clean,
                    mag_limit=master_phot.mag_lim_lo(passband=passband_2mass),
                    weight_image=weight_data,
                    weight_header=weight_hdr,
                    key_ra="ALPHAWIN_SKY",
                    key_dec="DELTAWIN_SKY",
                    key_ra_2mass="RAJ2000",
                    key_dec_2mass="DEJ2000",
                    key_mag_2mass=passband_2mass,
                    survey_name=self.setup.survey_name,
                )

                # Convert to public format
                log.info("Converting to public format")
                tables_file[tidx] = convert2public(
                    tables_file[tidx],
                    photerr_internal=photerr_internal,
                    apertures=self.setup.apertures,
                    mag_saturation=master_phot.mag_lim_lo(passband=passband_2mass),
                    survey_name=self.setup.survey_name,
                )

            # Create primary header
            log.info("Creating primary header")
            phdr = fits.Header()
            add_float_to_header(
                header=phdr,
                key="PHOTIERR",
                value=photerr_internal,
                comment="Internal photometric error (mag)",
                decimals=4,
            )
            phdr["FILTER"] = hdulist_in[0].header[self.setup.keywords.filter_name]

            # Create output file
            log.info("Creating TableHDU")
            hdulist_out = fits.HDUList(hdus=[fits.PrimaryHDU(header=phdr)])
            [hdulist_out.append(table2bintablehdu(tc)) for tc in tables_file]

            # Write to new output file
            log.info("Writing to disk")
            hdulist_out.writeto(path_out, overwrite=self.setup.overwrite)

            # Close original file
            hdulist_out.close()

        # Print time
        pmsg = f"\n-> Elapsed time: {time.time() - tstart:.2f}s"
        print_message(message=pmsg, kind="okblue", end="\n", logger=log)
