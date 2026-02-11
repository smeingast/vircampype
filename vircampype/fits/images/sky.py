import copy
import gc
import glob
import logging
import os
import shutil
import tempfile
import time
import uuid
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage.morphology import binary_closing
from skimage.draw import disk
from skimage.morphology import footprint_rectangle

from vircampype.data.cube import ImageCube
from vircampype.external.mmm import mmm
from vircampype.fits.images.common import FitsImages, MasterImages
from vircampype.miscellaneous.sourcemasks import SourceMasks
from vircampype.pipeline.errors import *
from vircampype.pipeline.log import PipelineLog
from vircampype.tools.astromatic import SextractorSetup, SwarpSetup
from vircampype.tools.fitstools import *
from vircampype.tools.imagetools import upscale_image
from vircampype.tools.mathtools import *
from vircampype.tools.messaging import *
from vircampype.tools.miscellaneous import *
from vircampype.tools.photometry import get_default_extinction
from vircampype.tools.plottools import *
from vircampype.tools.systemtools import *
from vircampype.tools.tabletools import *
from vircampype.tools.viziertools import *
from vircampype.tools.wcstools import *

logger = logging.getLogger(__name__)


class SkyImages(FitsImages):
    def __init__(self, setup, file_paths=None):
        super(SkyImages, self).__init__(setup=setup, file_paths=file_paths)

    # =========================================================================== #
    # Coordinates
    # =========================================================================== #
    _wcs = None

    @property
    def wcs(self):
        """
        Extracts WCS instances for all headers

        Returns
        -------
        List
            Nested list with WCS instances

        """

        # Check if already determined
        if self._wcs is not None:
            return self._wcs

        # Otherwise extract and return
        self._wcs = [[header2wcs(header=hdr) for hdr in h] for h in self.headers_data]
        return self._wcs

    _footprints = None

    @property
    def footprints(self):
        """Return footprints for all detectors of all files in instance."""
        if self._footprints is not None:
            return self._footprints

        self._footprints = [
            [SkyCoord(w.calc_footprint(), unit="deg") for w in ww] for ww in self.wcs
        ]
        return self._footprints

    _footprints_flat = None

    @property
    def footprints_flat(self):
        """Return flat"""
        if self._footprints_flat is not None:
            return self._footprints_flat

        self._footprints_flat = SkyCoord(flat_list(self.footprints))
        return self._footprints_flat

    def footprints_contain(self, skycoord: SkyCoord):
        """
        Test if the given skycoord is contained in the footprint of each WCS.

        Parameters
        ----------
        skycoord : SkyCoord
            The sky coordinate(s) to test.

        Returns
        -------
        List[List[bool]]
            Nested list of booleans, True if skycoord is inside the WCS footprint.
        """
        return [
            [wcs.footprint_contains(skycoord) for wcs in wcs_list]
            for wcs_list in self.wcs
        ]

    @property
    def centroid_all(self):
        return centroid_sphere(skycoord=self.footprints_flat)

    @property
    def centers_detectors(self):
        """
        Computes the centers for each detector.

        Returns
        -------
        List
            List with WCS centers.

        """

        centers = []
        for ww in self.wcs:
            temp = []
            for w in ww:
                temp.append(
                    SkyCoord(
                        *w.wcs_pix2world(w._naxis[0] / 2, w._naxis[1] / 2, 0),  # noqa
                        unit="deg",
                    )
                )
            centers.append(temp)

        return centers

    # =========================================================================== #
    # Sextractor
    # =========================================================================== #
    def paths_source_tables(self, preset=""):
        """
        Path to sextractor tables for files in instance.

        Returns
        -------
        iterable
            List with table names.
        """

        if preset is None:
            preset = ""

        return [
            x.replace(".fits", f".{preset}.fits.tab").replace("..", ".")
            for x in self.paths_full
        ]

    def sextractor(
        self,
        preset="scamp",
        silent=None,
        return_cmds=False,
        **kwargs,
    ):
        """
        Runs sextractor based on given presets.

        Parameters
        ----------
        preset : str
            Preset name.
        silent : bool, optional
            Can overrides setup on messaging.
        return_cmds : bool, optional
            Return list of sextractor shell commands instead of running them.

        """

        # Fetch log
        log = PipelineLog()

        # Load Sextractor setup
        sxs = SextractorSetup(setup=self.setup)

        if silent is None:
            silent = self.setup.silent

        # Processing info
        print_header(
            header="SOURCE DETECTION",
            left=f"Running Sextractor with preset '{preset}' on {len(self)} files",
            right=None,
            silent=silent,
            logger=log,
        )
        tstart = time.time()

        # Check for existing files
        path_tables_clean = []
        if not self.setup.overwrite:
            for pt in self.paths_source_tables(preset=preset):
                check_file_exists(file_path=pt, silent=silent)
                if not os.path.isfile(pt):
                    path_tables_clean.append(pt)

        # Set some common variables
        kwargs_yml = dict(
            path_yml=sxs.path_yml(preset=preset),
            filter_name=sxs.default_filter,
            parameters_name=sxs.path_param(preset=preset),
            gain_key=self.setup.keywords.gain,
            satur_key=self.setup.keywords.saturate,
            back_size=self.setup.sex_back_size,
            back_filtersize=self.setup.sex_back_filtersize,
        )

        # Read setup based on preset
        if preset.lower() in ["scamp", "fwhm", "psfex"]:
            ss = yml2config(skip=["catalog_name", "weight_image"], **kwargs_yml)
        elif preset == "class_star":
            ss = yml2config(
                skip=["catalog_name", "weight_image", "seeing_fwhm", "starnnw_name"],
                **kwargs_yml,
            )
        elif preset == "ic":
            ss = yml2config(
                skip=["catalog_name", "weight_image", "starnnw_name"]
                + list(kwargs.keys()),
                **kwargs_yml,
            )
        elif preset == "full":
            ss = yml2config(
                phot_apertures=",".join([str(ap) for ap in self.setup.apertures]),
                seeing_fwhm=2.5,
                skip=["catalog_name", "weight_image", "starnnw_name"]
                + list(kwargs.keys()),
                **kwargs_yml,
            )
        else:
            raise ValueError(f"Preset '{preset}' not supported")

        # Create temporary system file paths
        paths_tables_sex = [
            make_path_system_tempfile(
                prefix=f"{os.path.basename(pt)}_", suffix=".sex.cat"
            )
            for pt in path_tables_clean
        ]

        # Construct commands for source extraction
        cmds = [
            (
                f"{sxs.bin} -c {sxs.default_config} {image} "
                f"-STARNNW_NAME {sxs.default_nnw} "
                f"-CATALOG_NAME {catalog} "
                f"-WEIGHT_IMAGE {weight} {ss}"
            )
            for image, catalog, weight in zip(
                self.paths_full,
                paths_tables_sex,
                self.get_master_weight_global().paths_full,
            )
        ]

        # Check if there is a detection image available for each command
        # TODO: Check if this works
        if self.setup.sex_detection_image_path is not None:
            # Check if detection image exists
            if not os.path.isfile(self.setup.sex_detection_image_path):
                emsg = f"Detection image not found at {self.setup.sex_detection_image_path}"
                log.error(emsg)
                raise FileNotFoundError(emsg)

            # Add detection image to sextractor command
            for idx, _ in enumerate(cmds):
                cmds[idx] = cmds[idx].replace(
                    self.paths_full[idx],
                    f"{self.setup.sex_detection_image_path},{self.paths_full[idx]}",
                )
                log.info(
                    f"Double image mode\n"
                    f"Detection image '{self.setup.sex_detection_image_path}'\n"
                    f"Measurement image '{self.paths_full[idx]}'"
                )

        # Add kwargs to commands
        for key, val in kwargs.items():
            for cmd_idx, _ in enumerate(cmds):
                try:
                    cmds[cmd_idx] += f"-{key.upper()} {val[cmd_idx]}"
                except IndexError:
                    cmds[cmd_idx] += f"-{key.upper()} {val}"

        # Return commands if set
        if return_cmds:
            return cmds

        # Log all sextractor commands
        log.info("Sextractor commands:")
        for c in cmds:
            log.info(c)

        # Run Sextractor
        n_jobs_sex = (
            6 if self.setup.n_jobs > 6 else self.setup.n_jobs
        )  # max of 6 parallel jobs
        run_commands_shell_parallel(cmds=cmds, silent=True, n_jobs=n_jobs_sex)

        # Add some keywords to primary header
        for cat, img in zip(paths_tables_sex, self.paths_full):
            copy_keywords(
                path_1=cat,
                path_2=img,
                hdu_1=0,
                hdu_2=0,
                keywords=[self.setup.keywords.object, self.setup.keywords.filter_name],
            )

        # Move temporary files to final location
        for temp_path, final_path in zip(paths_tables_sex, path_tables_clean):
            shutil.move(temp_path, final_path)

        # Print time
        if not silent:
            tt_message = f"\n-> Elapsed time: {time.time() - tstart:.2f}s"
            print_message(message=tt_message, kind="okblue", end="\n", logger=log)

        # Select return class based on preset
        from vircampype.fits.tables.sextractor import (
            AstrometricCalibratedSextractorCatalogs,
            SextractorCatalogs,
        )

        if preset.lower() in ["scamp", "class_star", "fwhm", "psfex"]:
            cls = SextractorCatalogs
        elif (preset == "ic") | (preset == "full"):
            cls = AstrometricCalibratedSextractorCatalogs
        else:
            raise PipelineValueError(
                logger=log, message=f"Preset '{preset}' not supported"
            )

        # Return Table instance
        return cls(setup=self.setup, file_paths=self.paths_source_tables(preset=preset))

    def build_class_star_library(self):
        # Fetch log
        log = PipelineLog()
        log.info(f"Building class star library for {self.n_files} files")

        # Import
        from vircampype.fits.tables.sextractor import SextractorCatalogs

        # Processing info
        print_header(
            header="CLASSIFICATION", left="File", right=None, silent=self.setup.silent
        )
        tstart = time.time()

        # Run Sextractor with FWHM preset
        print_message(message="Building FWHM table...", end="")
        fwhm_catalogs = self.sextractor(preset="fwhm", silent=True)

        # Loop over files
        for idx_file in range(self.n_files):
            # Create output paths
            outpath = self.paths_full[idx_file].replace(".fits", ".cs.fits.tab")
            log.info(f"File {idx_file + 1}/{self.n_files}: {outpath}")

            # Check if the file is already there and skip if it is
            if (
                check_file_exists(file_path=outpath, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                continue

            # Print processing info
            message_calibration(
                n_current=idx_file + 1,
                n_total=self.n_files,
                name=outpath,
                d_current=None,
                d_total=None,
                silent=self.setup.silent,
            )

            # Read and clean current fwhm catalog
            fcs = [
                clean_source_table(x)
                for x in fwhm_catalogs.file2table(file_index=idx_file)
            ]

            # Get percentiles image quality measurements
            fwhms = np.array(flat_list([x["FWHM_IMAGE"] for x in fcs]))
            fwhm_percentiles = np.percentile(fwhms, [0.5, 99.5])
            fwhm_lo = round_decimals_down(fwhm_percentiles[0] / 3, decimals=2)
            fwhm_hi = round_decimals_up(fwhm_percentiles[1] / 3, decimals=2)
            log.info(f"FWHM range: {fwhm_lo} - {fwhm_hi}")

            # Determine FWHM range
            fwhm_range = np.around(
                np.arange(fwhm_lo - 0.05, fwhm_hi + 0.11, 0.05), decimals=2
            )

            # Safety check for fwhm range
            if len(fwhm_range) > 30:
                fwhm_range = np.around(np.arange(0.45, 1.91, 0.05), decimals=2)
            log.info(f"Final FWHM range: {fwhm_range}")

            # Construct sextractor commands
            cmds = [
                self.sextractor(
                    preset="class_star",
                    seeing_fwhm=ss,
                    return_cmds=True,
                    silent=True,
                )[idx_file]
                for ss in fwhm_range
            ]

            # Get catalog paths
            catalog_paths = []
            for idx in range(len(cmds)):
                cpath = cmds[idx].split("-CATALOG_NAME ")[1].split(" ")[0]
                cname = os.path.basename(cpath)
                cpath_new = cpath.replace(cname, f"FWHM{fwhm_range[idx]:0.2f}.sex.cat")
                cmds[idx] = cmds[idx].replace(cpath, cpath_new)
                catalog_paths.append(cpath_new)

            # Log catalog paths and sextractor commands
            for idx, c in enumerate(catalog_paths):
                log.info(f"Catalog path {idx + 1}/{len(cmds)}: {c}")
            for idx, c in enumerate(cmds):
                log.info(f"Sextractor command {idx + 1}/{len(cmds)}: {c}")

            # Run Sextractor
            n_jobs_sex = (
                5 if self.setup.n_jobs > 5 else self.setup.n_jobs
            )  # max of 5 parallel jobs
            log.info(f"Running {len(cmds)} sextractor commands in parallel")
            run_commands_shell_parallel(cmds=cmds, silent=True, n_jobs=n_jobs_sex)

            # Load catalogs with different input seeing
            catalogs = SextractorCatalogs(setup=self.setup, file_paths=catalog_paths)
            log.info(f"Loaded {len(catalogs)} catalogs")

            # Make output HDUList
            tables_out = [Table() for _ in self.iter_data_hdu[idx_file]]

            # Loop over files
            for idx_fwhm in range(catalogs.n_files):
                # Log current FWHM value
                log.info(f"Processing FWHM {fwhm_range[idx_fwhm]:4.2f}")
                log.info(f"Filename: {catalogs.paths_full[idx_fwhm]}")

                # Read tables for current seeing
                tables_fwhm = catalogs.file2table(file_index=idx_fwhm)
                log.info(f"Loaded {len(tables_fwhm)} FWHM tables")

                # Add classifier and coordinates for all HDUs
                for tidx in range(len(tables_fwhm)):
                    # Log current catalog and number of sources
                    log.info(f"Processing HDU {tidx + 1}/{len(tables_fwhm)}")
                    log.info(f"Number of sources: {len(tables_fwhm[tidx])}")

                    # Add coordinates only on first iteration
                    if idx_fwhm == 0:
                        tables_out[tidx]["XWIN_IMAGE"] = tables_fwhm[tidx]["XWIN_IMAGE"]
                        tables_out[tidx]["YWIN_IMAGE"] = tables_fwhm[tidx]["YWIN_IMAGE"]

                    # Add classifier
                    cs_column_name = f"CLASS_STAR_{fwhm_range[idx_fwhm]:4.2f}"
                    tables_out[tidx][cs_column_name] = tables_fwhm[tidx]["CLASS_STAR"]

            # Make FITS table
            header_prime = fits.Header()
            for fidx in range(len(fwhm_range)):
                add_float_to_header(
                    header=header_prime,
                    key=f"HIERARCH PYPE CS FWHM {fidx + 1}",
                    value=fwhm_range[fidx],
                )
            hdul = fits.HDUList(hdus=[fits.PrimaryHDU(header=header_prime)])
            for t in tables_out:
                hdul.append(fits.BinTableHDU(t))
            hdul.writeto(outpath, overwrite=True)
            log.info(f"Saved to '{outpath}'")

            # Remove sextractor catalogs
            for f in catalogs.paths_full:
                remove_file(filepath=f)
            log.info("Removed sextractor catalogs")

        # Clean up FWHM catalogs
        for f in fwhm_catalogs.paths_full:
            remove_file(filepath=f)
        log.info("Removed FWHM catalogs")

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def build_master_psf(self, preset):
        raise NotImplementedError
        # Run Sextractor with PSFEX preset
        # sources_psfex = self.sextractor(preset="psfex")

        # Run PSFEX
        # sources_psfex.psfex(preset=preset)

    def apply_illumination_correction(self):
        """Applies illumination correction to images."""

        # Fetch log
        log = PipelineLog()
        log.info(f"Applying illumination correction to {self.n_files} files")

        # Processing info
        print_header(
            header="APPLYING ILLUMINATION CORRECTION", silent=self.setup.silent
        )
        tstart = time.time()

        # Fetch illumination correction for each image
        illumcor = self.get_master_illumination_correction()
        sourcemasks = self.get_master_source_mask()
        log.info(f"Loaded {len(illumcor)} illumination corrections")
        log.info(f"Loaded {len(sourcemasks)} source masks")

        # Loop over self
        for idx_file in range(self.n_files):
            # Create output path
            outpath = f"{self.setup.folders['illumcorr']}{self.names[idx_file]}.ic.fits"
            log.info(f"File {idx_file + 1}/{self.n_files}: {outpath}")

            # Check for ahead file
            path_ahead = self.paths_full[idx_file].replace(".fits", ".ahead")
            path_ahead_sf = outpath.replace(".fits", ".ahead")
            if not os.path.isfile(path_ahead):
                log.error(f"External header not found: {path_ahead}")
                raise ValueError("External header not found")

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent):
                continue

            # Print processing info
            message_calibration(
                n_current=idx_file + 1,
                n_total=self.n_files,
                name=outpath,
                d_current=None,
                d_total=None,
                silent=self.setup.silent,
            )

            # Read data
            log.info(f"Reading data for {self.basenames[idx_file]}")
            cube_self = self.file2cube(file_index=idx_file)
            cube_flat = illumcor.file2cube(file_index=idx_file)
            cube_mask = sourcemasks.file2cube(file_index=idx_file)

            # Modify read noise, gain, and saturation keywords in headers
            data_headers = []
            for idx_hdr in range(len(self.headers_data[idx_file])):
                # Load current header
                hdr = self.headers_data[idx_file][idx_hdr]

                # Modification factor is mean for the current illumination correction
                mod = np.median(cube_flat[idx_hdr])
                log.info(f"HDU {idx_hdr + 1}: IC median factor = {mod:.4f}")

                # Add modification factor
                hdr.set(
                    "HIERARCH PYPE IC FACTOR",
                    value=np.round(mod, 3),
                    comment="Median IC modification factor",
                )

                # Adapt keywords
                keywords = [
                    self.setup.keywords.rdnoise,
                    self.setup.keywords.gain,
                    self.setup.keywords.saturate,
                ]
                mod_func = [np.divide, np.multiply, np.divide]
                for kw, func in zip(keywords, mod_func):
                    # Read comment
                    comment = self.headers_data[idx_file][idx_hdr].comments[kw]

                    # First save old keyword
                    hdr.set(
                        f"HIERARCH PYPE BACKUP {kw}",
                        value=hdr[kw],
                        comment="Value before IC",
                    )

                    # Now add new keyword and delete old one
                    hdr.set(
                        kw,
                        value=func(self.headers_data[idx_file][idx_hdr][kw], mod),
                        comment=comment,
                    )

                # Save headers
                data_headers.append(hdr)

            # Copy self for background mask
            cube_self_copy = copy.deepcopy(cube_self)
            cube_self_copy.apply_masks(sources=cube_mask)
            background = cube_self_copy.background(mesh_size=128, mesh_filtersize=3)[0]

            # Apply background
            cube_self -= background

            # Normalize
            cube_self /= cube_flat
            background /= np.median(cube_flat, axis=(1, 2))[:, np.newaxis, np.newaxis]

            # Add background back in
            cube_self += background

            # Write back to disk
            cube_self.write_mef(
                outpath,
                prime_header=self.headers_primary[idx_file],
                data_headers=data_headers,
            )
            log.info(f"Saved to '{outpath}'")

            # Copy aheader for swarping
            copy_file(path_ahead, path_ahead_sf)
            log.info(f"Copied ahead file to '{path_ahead_sf}'")

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )


class SkyImagesRaw(SkyImages):
    def __init__(self, setup, file_paths=None):
        super(SkyImagesRaw, self).__init__(setup=setup, file_paths=file_paths)

    def process_raw_basic(self):
        # Processing info
        print_header(
            header="BASIC RAW PROCESSING", right=None, silent=self.setup.silent
        )

        # Fetch log
        log = PipelineLog()
        log.info(f"Processing {self.n_files} basic raw files:\n{self.basenames2log}")
        tstart = time.time()

        # Fetch the Masterfiles
        log.info("Fetching master files")
        master_gain = self.get_master_gain()
        log.info(f"Master gain:\n{master_gain.basenames2log}")
        master_dark = self.get_master_dark(ignore_dit=True)
        log.info(f"Master dark:\n{master_dark.basenames2log}")
        master_linearity = self.get_master_linearity()
        log.info(f"Master linearity:\n{master_linearity.basenames2log}")

        # Loop over files and apply calibration
        for idx_file in range(self.n_files):
            # Create output path
            outpath = (
                f"{self.setup.folders['processed_basic']}"
                f"{self.names[idx_file]}.proc.basic.fits"
            )

            # Log processing info
            log.info(f"Processing file {idx_file + 1}/{self.n_files}:\n{outpath}")

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent):
                log.info("File already exists, skipping")
                continue

            # Print processing info
            message_calibration(
                n_current=idx_file + 1,
                n_total=self.n_files,
                name=outpath,
                d_current=None,
                d_total=None,
                silent=self.setup.silent,
            )

            # Log more info
            log.info(f"Number of extensions: {len(self.iter_data_hdu[idx_file])}")
            log.info(f"Filter name: {self.passband[idx_file]}")
            log.info(f"Target designation: {self.setup.name}")
            log.info(f"Dark file name: {master_dark.basenames[idx_file]}")
            log.info(f"Linearity file name: {master_linearity.basenames[idx_file]}")
            log.info(f"Gain: {master_gain.gain[idx_file]}")
            log.info(f"Read noise: {master_gain.rdnoise[idx_file]}")
            log.info(f"Saturation levels: {self.setup.saturation_levels}")

            # Read file into cube
            cube = self.file2cube(file_index=idx_file, hdu_index=None, dtype=np.float32)

            # Get master calibration
            dark = master_dark.file2cube(file_index=idx_file, dtype=np.float32)
            lcff = master_linearity.file2coeff(file_index=idx_file)

            # Norm to NDIT=1
            log.info(f"Normalizing to NDIT=1; using NDIT={self.ndit[idx_file]}")
            cube.normalize(norm=self.ndit[idx_file])

            # Linearize
            log.info(f"Linearizing data; using linearity coefficients:\n{lcff}")
            cube.linearize(coeff=lcff, texptime=self.texptime[idx_file])

            # Subtract with dark
            cube -= dark

            # Divide by flat if set
            if self.setup.flat_type == "twilight":
                log.info("Dividing by twilight flat")
                master_flat = self.get_master_twilight_flat()
                mflat = master_flat.file2cube(file_index=idx_file, dtype=np.float32)
                cube /= mflat

            # Add file info to main header
            phdr = self.headers_primary[idx_file].copy()
            phdr.set(
                self.setup.keywords.object,
                value=self.setup.name,
                comment="Target designation",
            )
            phdr.set(
                "DARKFILE",
                value=master_dark.basenames[idx_file],
                comment="Dark file name",
            )
            phdr.set(
                "LINFILE",
                value=master_linearity.basenames[idx_file],
                comment="Linearity file name",
            )
            phdr.set("FILTER", value=self.passband[idx_file], comment="Filter name")

            # Add setup to header
            self.setup.add_setup_to_header(header=phdr)

            # Copy data headers
            hdrs_data = [
                self.headers_data[idx_file][idx_hdu].copy()
                for idx_hdu in range(len(self.iter_data_hdu[idx_file]))
            ]

            # Fix headers
            if self.setup.fix_vircam_headers:
                log.info("Fixing headers")
                fix_vircam_headers(prime_header=phdr, data_headers=hdrs_data)

            # Add stuff to data headers
            for idx_hdu, dhdr in enumerate(hdrs_data):
                log.info(f"Modify data headers; extension {idx_hdu + 1}")
                # Grab gain and readnoise
                log.info(
                    f"Scaling gain {master_gain.gain[idx_file][idx_hdu - 1]} "
                    f"with NDIT={self.ndit[idx_file]}"
                )
                gain = (
                    master_gain.gain[idx_file][idx_hdu - 1] * self.ndit_norm[idx_file]
                )
                log.info(f"New gain: {gain}")
                rdnoise = master_gain.rdnoise[idx_file][idx_hdu - 1]
                log.info(f"Read noise: {rdnoise}")

                # Grab other parameters
                offseti, noffsets, chipid = (
                    phdr["OFFSET_I"],
                    phdr["NOFFSETS"],
                    dhdr["HIERARCH ESO DET CHIP NO"],
                )
                photstab = offseti + noffsets * (chipid - 1)
                log.info(f"Photometric stability ID: {photstab}")
                dextinct = get_default_extinction(passband=self.passband[idx_file])
                log.info(
                    f"Setting default extinction to {dextinct} "
                    f"for band {self.passband[idx_file]}"
                )

                # Add stuff to header
                dhdr.set(
                    self.setup.keywords.gain,
                    value=np.round(gain, 3),
                    comment="Gain (e-/ADU)",
                )
                dhdr.set(
                    self.setup.keywords.rdnoise,
                    value=np.round(rdnoise, 3),
                    comment="Read noise (e-)",
                )
                dhdr.set(
                    self.setup.keywords.saturate,
                    value=self.setup.saturation_levels[idx_hdu],
                    comment="Saturation level (ADU)",
                )
                dhdr.set("NOFFSETS", value=noffsets, comment="Total number of offsets")
                dhdr.set("OFFSET_I", value=offseti, comment="Current offset iteration")
                dhdr.set(
                    "NJITTER",
                    value=phdr["NJITTER"],
                    comment="Total number of jitter positions",
                )
                dhdr.set(
                    "JITTER_I",
                    value=phdr["JITTER_I"],
                    comment="Current jitter iteration",
                )
                dhdr.set("PHOTSTAB", value=photstab, comment="Photometric stability ID")
                dhdr.set(
                    "SCMPPHOT",
                    value=offseti,
                    comment="Photometric stability ID for Scamp",
                )
                dhdr.set(
                    self.setup.keywords.filter_name,
                    value=self.passband[idx_file],
                    comment="Filter name",
                )
                dhdr.set("FILTER", value=self.passband[idx_file], comment="Filter name")
                dhdr.set("DEXTINCT", value=dextinct, comment="Default extinction (mag)")

                # Add Airmass
                if self.setup.set_airmass:
                    airmass = get_airmass_from_header(
                        header=dhdr, time=dhdr[self.setup.keywords.date_ut]
                    )
                    log.info(f"Setting airmass to {airmass}")

                    dhdr.set(
                        self.setup.keywords.airmass,
                        value=airmass,
                        comment="Airmass at time of observation",
                    )

            # Write to disk
            log.info(f"Writing to disk:\n{outpath}")
            cube.write_mef(
                path=outpath, prime_header=phdr, data_headers=hdrs_data, dtype="float32"
            )

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )


class SkyImagesRawScience(SkyImagesRaw):
    def __init__(self, setup, file_paths=None):
        super(SkyImagesRawScience, self).__init__(setup=setup, file_paths=file_paths)


class SkyImagesRawOffset(SkyImagesRaw):
    def __init__(self, setup, file_paths=None):
        super(SkyImagesRawOffset, self).__init__(setup=setup, file_paths=file_paths)


class SkyImagesRawStd(SkyImagesRaw):
    def __init__(self, setup, file_paths=None):
        super(SkyImagesRawStd, self).__init__(setup=setup, file_paths=file_paths)


class SkyImagesProcessed(SkyImages):
    def __init__(self, setup, file_paths=None):
        super(SkyImagesProcessed, self).__init__(setup=setup, file_paths=file_paths)

    def __build_additional_masks(self) -> dict:
        # Fetch log
        log = PipelineLog()
        log.info(f"Building additional source masks for {self.n_files} files")

        # Create empty mask list for any additional masks
        additional_masks = dict(ra=[], dec=[], size=[])

        # Load positions and magnitudes of bright sources from master photometry
        master_phot = self.get_master_photometry()
        if self.setup.mask_2mass_sources:
            log.info(f"Adding 2MASS source masks")
            log.info(f"Lower magnitude limit: {self.setup.mask_2mass_sources_bright}")
            log.info(f"Upper magnitude limit: {self.setup.mask_2mass_sources_faint}")
            mag_master = master_phot.mag(passband=self.passband[0])[0][0]
            bright = (mag_master > self.setup.mask_2mass_sources_bright) & (
                mag_master < self.setup.mask_2mass_sources_faint
            )
            mag_bright = mag_master[bright]
            log.info(f"Adding {len(mag_bright)} bright sources to masks")
            additional_masks["ra"].extend(list(master_phot.ra[0][0][bright]))
            additional_masks["dec"].extend(list(master_phot.dec[0][0][bright]))
            additional_masks["size"].extend(
                [int(x) for x in SourceMasks.interp_2mass_size()(mag_bright)]
            )

        if self.setup.mask_bright_galaxies:
            # Load bright galaxy catalog
            log.info(f"Adding bright galaxies to source masks")
            bright_galaxies = SourceMasks.bright_galaxies()
            log.info(f"Found {len(bright_galaxies)} bright galaxies in catalog")
            sc_galaxies = SkyCoord([bg.center for bg in bright_galaxies])

            # Check which ones in footprint
            log.info("Checking which galaxies are in image footprints")
            in_footprints = self.footprints_contain(skycoord=sc_galaxies)
            in_footprints = np.array(in_footprints)
            contained_any = np.any(in_footprints, axis=(0, 1))
            indices = np.where(contained_any)[0]
            log.info(f"Found {len(indices)} galaxies in image footprints")

            """ 
            I do not check if other masks intersect the image footprints, 
            since at present, masks are applied to any part of the image they cover, 
            even if the source lies near the edge. For galaxies, this intersection 
            check is intentionally omitted here as it later takes too long.
            """

            # Save if any remain
            if len(indices) > 0:
                bright_galaxies = bright_galaxies[indices]
                additional_masks["ra"].extend(bright_galaxies.ra_deg)
                additional_masks["dec"].extend(bright_galaxies.dec_deg)
                additional_masks["size"].extend(bright_galaxies.size_deg)

        # Add manual source masks
        if self.setup.additional_source_masks is not None:
            log.info(f"Adding manual source masks")
            log.info(
                f"Number of additional masks: {len(self.setup.additional_source_masks)}"
            )
            additional_masks["ra"].extend(self.setup.additional_source_masks.ra_deg)
            additional_masks["dec"].extend(self.setup.additional_source_masks.dec_deg)
            additional_masks["size"].extend(
                self.setup.additional_source_masks.size_pix()
            )

        log.info(f"Total number of additional masks: {len(additional_masks['ra'])}")
        return additional_masks

    def build_master_source_mask(self):
        # Processing info
        print_header(
            header="MASTER-SOURCEMASK",
            right=None,
            silent=self.setup.silent,
        )
        tstart = time.time()

        self.check_compatibility(
            n_files_min=self.setup.source_masks_n_min, n_hdu_max=1, n_filter_max=1
        )

        # Fetch master BPM
        master_bpm = self.get_master_bpm()

        # Build additional source masks
        additional_masks = self.__build_additional_masks()

        # Construct final output names
        outpaths = [
            self.setup.folders["master_object"]
            + f"MASTER-SOURCE-MASK.{self.mjd[i]:0.4f}.FIL_{self.passband[i]}.fits"
            for i in range(self.n_files)
        ]

        # Check if all files exist and return if they do
        if all([check_file_exists(file_path=oo, silent=False) for oo in outpaths]):
            return

        # Start looping over detectors
        paths_temp_mask = []
        for d in self.iter_data_hdu[0]:
            # Make temp filename
            paths_temp_mask.append(
                f"{self.setup.folders['temp']}"
                f"MASK_DETECTOR_{d:02d}."
                f"MJD_{self.mjd_mean:0.4f}."
                f"FIL_{self.passband[0]}.fits"
            )

            # TODO: Fix counter
            # Print processing info
            message_calibration(
                n_current=d,
                n_total=len(self.iter_data_hdu[0]),
                name=paths_temp_mask[-1],
                d_current=None,
                d_total=None,
                silent=self.setup.silent,
            )

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=paths_temp_mask[-1], silent=True):
                continue

            # Get data
            cube_raw = self.hdu2cube(hdu_index=d, dtype=np.float32)
            bpm = master_bpm.hdu2cube(hdu_index=d, dtype=bool)

            # Instantiate empty additional and source masks
            mask_additional = ImageCube(
                cube=np.full_like(cube_raw.cube, fill_value=False, dtype=bool),
                setup=self.setup,
            )
            mask_sources = ImageCube(
                cube=np.full_like(cube_raw.cube, fill_value=False, dtype=bool),
                setup=self.setup,
            )

            # Loop over files and keep only source that appear on at least one image
            for idx_file in range(self.n_files):
                ww = self.wcs[idx_file][d - 1]
                mxx, myy = ww.wcs_world2pix(
                    additional_masks["ra"], additional_masks["dec"], 0
                )
                # remove those that are too far away from edges
                naxis1 = self.headers_data[idx_file][d - 1]["NAXIS1"]
                naxis2 = self.headers_data[idx_file][d - 1]["NAXIS2"]
                keep = (
                    (mxx > -200)
                    & (myy > -200)
                    & (mxx < naxis1 + 200)
                    & (myy < naxis2 + 200)
                )
                for sx, sy, ss in zip(
                    mxx[keep], myy[keep], np.array(additional_masks["size"])[keep]
                ):
                    aa, bb = disk((sy, sx), ss, shape=(naxis2, naxis1))
                    mask_additional.cube[idx_file][aa, bb] = True

            for i in range(self.setup.source_masks_n_iter):
                # Copy original array
                cube_masked = cube_raw.copy()
                cube_raw_temp = cube_raw.copy()

                # Apply masks
                cube_masked.apply_masks(
                    bpm=bpm,
                    sources=mask_sources,
                )

                # Apply additional masks
                cube_masked.apply_masks(sources=mask_additional)

                # Determine background level
                sky_scale, sky_std = cube_masked.background_planes()

                # Scale to same background level
                cube_masked.scale_planes(1 / sky_scale)

                # Collapse cube to median
                sky_median = cube_masked.flatten(metric=np.nanmedian, axis=0)

                # Depending on setup, use as flat or subtract background
                with warnings.catch_warnings():
                    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
                    if self.setup.flat_type == "twilight":
                        # Subtract background
                        cube_raw_temp -= sky_scale[:, np.newaxis, np.newaxis] * (
                            sky_median - 1
                        )
                    elif self.setup.flat_type == "sky":
                        # Flat-field cube
                        cube_raw_temp /= sky_median

                # Add extreme outliers to additional masks
                # TODO: Fix this for twilight mode
                mask_additional.cube += (
                    cube_raw_temp.cube
                    < sky_scale[:, np.newaxis, np.newaxis]
                    - 10 * sky_std[:, np.newaxis, np.newaxis]
                )

                # destripe on last iteration
                if (
                    i == self.setup.source_masks_n_iter - 1
                ) & self.setup.source_masks_destripe:
                    cube_raw_temp.destripe(masks=mask_additional)

                """ Destriping here is very problematic. Firstly, I can't average,
                because I have only frames of a single detector and secondly, any
                extended emission that covers a large part of the image will affect
                the destriping.
                """
                # # Destripe with mask from temporary cube
                # cube_raw_temp.destripe(
                #     masks=np.isnan(cube_masked.cube),
                #     smooth=True,
                #     combine_bad_planes=False,
                # )

                # Run source detection
                if self.setup.source_mask_method.lower() == "noisechisel":
                    mask_sources = cube_raw_temp.build_source_masks_noisechisel()
                elif self.setup.source_mask_method.lower() == "built-in":
                    mask_sources = cube_raw_temp.build_source_masks()
                else:
                    raise ValueError(
                        f"Source masking method "
                        f"'{self.setup.source_mask_method}' not supported"
                    )

                # Apply closing operation if set
                if self.setup.source_mask_closing:
                    for pidx, plane in enumerate(mask_sources):
                        mask_sources.cube[pidx] = binary_closing(
                            plane,
                            structure=footprint_rectangle(
                                (
                                    self.setup.source_masks_closing_size,
                                    self.setup.source_masks_closing_size,
                                )
                            ),
                            iterations=self.setup.source_masks_closing_iter,
                        )

                # Add additional masks to source masks
                mask_sources += mask_additional

            # Free memory
            del cube_masked
            del cube_raw_temp
            gc.collect()

            # Write to temporary master mask
            mask_sources.write_mef(
                path=paths_temp_mask[-1],
                dtype=np.uint8,
                overwrite=True,
            )

        # Load all of them into a FitsImages instance
        masks_temp = FitsImages(setup=self.setup, file_paths=paths_temp_mask)

        # Loop over output files
        for idx_file, outpath in enumerate(outpaths):
            # # Check if file exists
            if check_file_exists(file_path=outpath, silent=self.setup.silent):
                continue

            # Load all masks for this file
            masks = masks_temp.hdu2cube(hdu_index=idx_file + 1, dtype=np.uint8)

            # Create header cards
            cards = make_cards(
                keywords=[
                    self.setup.keywords.date_mjd,
                    self.setup.keywords.date_ut,
                    self.setup.keywords.object,
                ],
                values=[
                    self.mjd[idx_file],
                    self.time_obs[idx_file],
                    "MASTER-SOURCE-MASK",
                ],
            )

            # Make primary header
            prime_header = fits.Header(cards=cards)

            # Write to disk
            masks.write_mef(
                path=outpath,
                prime_header=prime_header,
                dtype=np.uint8,
                overwrite=True,
            )

        # Remove all temp files
        [remove_file(f) for f in paths_temp_mask]

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def build_master_photometry(self):
        # Processing info
        print_header(header="MASTER-PHOTOMETRY", right=None, silent=self.setup.silent)
        tstart = time.time()

        # Construct outpath
        outpath = self.setup.folders["master_object"] + "MASTER-PHOTOMETRY.fits.tab"

        # Check if the file is already there and skip if it is
        if not check_file_exists(file_path=outpath, silent=self.setup.silent):
            # Print processing info
            message_calibration(
                n_current=1,
                n_total=1,
                name=outpath,
                d_current=None,
                d_total=None,
                silent=self.setup.silent,
            )

            # Determine size to download
            radius = 1.1 * np.max(
                self.footprints_flat.separation(self.centroid_all).degree
            )

            # Download catalog
            if self.setup.phot_reference_catalog.lower() == "2mass":
                table = download_2mass(skycoord=self.centroid_all, radius=radius)
            else:
                raise ValueError(
                    f"Catalog '{self.setup.phot_reference_catalog}' not supported"
                )

            # Save catalog
            table.write(outpath, format="fits", overwrite=True)

            # Add object info to primary header
            add_key_primary_hdu(
                path=outpath, key=self.setup.keywords.object, value="MASTER-PHOTOMETRY"
            )

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def build_master_astrometry(self):
        # Processing info
        print_header(header="MASTER-ASTROMETRY", right=None, silent=self.setup.silent)
        tstart = time.time()

        # Construct outpath
        outpath = self.setup.folders["master_object"] + "MASTER-ASTROMETRY.fits.tab"

        # Check if the file is already there and skip if it is
        if not check_file_exists(file_path=outpath, silent=self.setup.silent):
            # Print processing info
            message_calibration(
                n_current=1,
                n_total=1,
                name=outpath,
                d_current=None,
                d_total=None,
                silent=self.setup.silent,
            )

            # Determine radius for download query
            radius = 1.1 * np.max(
                self.footprints_flat.separation(self.centroid_all).degree
            )

            # Download catalog
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                table = download_gaia(skycoord=self.centroid_all, radius=radius)

                # Keep only sources with valid ra/dec/pm entries
                keep = (
                    np.isfinite(table["ra"])
                    & np.isfinite(table["dec"])
                    & np.isfinite(table["pmra"])
                    & np.isfinite(table["pmdec"])
                    & np.isfinite(table["mag"])
                    & np.isfinite(table["mag_error"])
                    & (table["ruwe"] < 1.5)
                )

                # Apply cut
                table = table[keep]

                # Grab output epoch
                epoch_out = self.epoch_mean if self.setup.warp_gaia else 2016.0

                # Write to disk
                make_gaia_refcat(
                    table_in=table,
                    path_ldac_out=outpath,
                    epoch_in=2016.0,
                    epoch_out=epoch_out,
                    key_ra="ra",
                    key_ra_error="ra_error",
                    key_dec="dec",
                    key_dec_error="dec_error",
                    key_pmra="pmra",
                    key_pmra_error="pmra_error",
                    key_pmdec="pmdec",
                    key_pmdec_error="pmdec_error",
                    key_ruwe="ruwe",
                    key_gmag="mag",
                    key_gflux="flux",
                    key_gflux_error="flux_error",
                )

            # Add epoch to header
            add_key_primary_hdu(
                path=outpath, key="EPOCH", value=epoch_out, comment="Catalog epoch"
            )

            # Add object info to primary header
            add_key_primary_hdu(
                path=outpath, key=self.setup.keywords.object, value="MASTER-ASTROMETRY"
            )

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def build_master_sky(self):
        """
        Builds a sky frame from the given input data. After calibration and masking,
        the frames are normalized with their sky levels and then combined.

        """

        # Processing info
        print_header(header="MASTER-SKY", right=None, silent=self.setup.silent)
        tstart = time.time()

        # Split based on filter and interval
        split = self.split_keywords(keywords=[self.setup.keywords.filter_name])
        split = flat_list(
            [
                s.split_window(window=self.setup.sky_window, remove_duplicates=True)
                for s in split
            ]
        )

        # Remove too short entries
        split = prune_list(split, n_min=self.setup.sky_n_min)

        if len(split) == 0:
            raise ValueError("No suitable sequence found for sky images.")

        # Now loop through separated files
        for files, fidx in zip(split, range(1, len(split) + 1)):  # type: SkyImages, int
            # Check sequence (at least n files, same nHDU, same NDIT, and same filter)
            files.check_compatibility(
                n_files_min=self.setup.sky_n_min, n_hdu_max=1, n_filter_max=1
            )

            # Create master name
            outpath = (
                f"{files.setup.folders['master_object']}MASTER-SKY."
                f"MJD_{files.mjd_mean:0.4f}."
                f"FIL_{files.passband[0]}.fits"
            )

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent):
                continue

            # Fetch the Masterfiles
            master_bpm = files.get_master_bpm()
            master_mask = files.get_master_source_mask()

            # Instantiate output
            master_cube = ImageCube(setup=self.setup, cube=None)

            # Start looping over detectors
            data_headers = []
            bkg_all, bkg_std_all = [], []
            for d in files.iter_data_hdu[0]:
                # Print processing info
                message_calibration(
                    n_current=fidx,
                    n_total=len(split),
                    name=outpath,
                    d_current=d,
                    d_total=max(files.iter_data_hdu[0]),
                    silent=self.setup.silent,
                )

                # Get data
                cube = files.hdu2cube(hdu_index=d, dtype=np.float32)

                # Get masks
                sources = master_mask.hdu2cube(hdu_index=d, dtype=np.uint8)
                bpm = master_bpm.hdu2cube(hdu_index=d, dtype=bool)

                # Apply masks cube
                cube.apply_masks(sources=sources + bpm)

                # Compute sky level in each plane
                bkg, bkg_std = cube.background_planes()
                bkg_all.append(bkg)
                bkg_std_all.append(bkg_std)

                # Normalize to same flux level
                cube.normalize(norm=bkg)

                # Apply sky masks
                cube.apply_masks(
                    mask_min=self.setup.sky_mask_min,
                    mask_max=self.setup.sky_mask_max,
                    mask_below=self.setup.sky_rel_lo,
                    mask_above=self.setup.sky_rel_hi,
                    sigma_level=self.setup.sky_sigma_level,
                    sigma_iter=self.setup.sky_sigma_iter,
                )

                # Create weights if needed
                if self.setup.sky_combine_metric == "weighted":
                    metric = "weighted"
                    weights = np.empty_like(cube.cube)
                    weights[:] = (1 / bkg_std)[:, np.newaxis, np.newaxis]
                    weights[~np.isfinite(cube.cube)] = 0.0
                else:
                    metric = string2func(self.setup.flat_metric)
                    weights = None

                # Collapse cube
                collapsed = cube.flatten(metric=metric, axis=0, weights=weights)

                # Create header with sky measurements
                hdr = fits.Header()
                for cidx, bb in enumerate(bkg):
                    hdr.set(
                        f"HIERARCH PYPE SKY MEAN {cidx}",
                        value=np.round(bb, 2),
                        comment="Measured sky (ADU)",
                    )
                    hdr.set(
                        f"HIERARCH PYPE SKY NOISE {cidx}",
                        value=np.round(bkg_std[cidx], 2),
                        comment="Measured sky noise (ADU)",
                    )
                    hdr.set(
                        f"HIERARCH PYPE SKY MJD {cidx}",
                        value=np.round(files.mjd[cidx], 6),
                        comment="MJD of measured sky",
                    )

                # Append to list
                data_headers.append(hdr)

                # Collapse extensions with specified metric and append to output
                master_cube.extend(data=collapsed.astype(np.float32))

            # Compute gain harmonization
            flat_scale = np.array(bkg_all) / np.mean(bkg_all)
            flat_scale_std = np.std(flat_scale, axis=1)
            flat_scale = np.mean(flat_scale, axis=1)

            # Apply gain harmonization
            master_cube.scale_planes(flat_scale)

            # Mean flat field error
            flat_err = np.round(100.0 * np.mean(flat_scale_std), decimals=2)

            # Create primary header
            hdr_prime = fits.Header()
            hdr_prime.set(keyword=self.setup.keywords.date_mjd, value=files.mjd_mean)
            hdr_prime.set(
                keyword=self.setup.keywords.date_ut, value=files.time_obs_mean.fits
            )
            hdr_prime.set(keyword=self.setup.keywords.object, value="MASTER-SKY")
            hdr_prime.set(self.setup.keywords.dit, value=files.dit[0])
            hdr_prime.set(self.setup.keywords.ndit, value=files.ndit[0])
            hdr_prime.set(self.setup.keywords.filter_name, value=files.passband[0])
            hdr_prime.set("HIERARCH PYPE N_FILES", value=len(files))
            hdr_prime.set("HIERARCH PYPE SKY FLATERR", value=flat_err)

            # Write to disk
            master_cube.write_mef(
                path=outpath, prime_header=hdr_prime, data_headers=data_headers
            )

            # QC plot
            if self.setup.qc_plots:
                msky = MasterSky(setup=self.setup, file_paths=outpath)
                msky.qc_plot_sky(paths=None, axis_size=5)
                msky.qc_plot_sky_stability(paths=None, axis_size=5)

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def process_raw_final(self):
        """Main processing method."""

        # Processing info
        print_header(
            header="FINAL RAW PROCESSING", right=None, silent=self.setup.silent
        )
        tstart = time.time()

        # Fetch the Masterfiles
        master_sky = self.get_master_sky()
        master_source_mask = self.get_master_source_mask()

        # Loop over files and apply calibration
        for idx_file in range(self.n_files):
            # Create output path
            outpath = (
                f"{self.setup.folders['processed_final']}"
                f"{self.basenames[idx_file].replace('.proc.basic.', '.proc.final.')}"
            )

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent):
                continue

            # Print processing info
            message_calibration(
                n_current=idx_file + 1,
                n_total=self.n_files,
                name=outpath,
                d_current=None,
                d_total=None,
                silent=self.setup.silent,
            )

            # Read file into cube
            cube = self.file2cube(file_index=idx_file, hdu_index=None, dtype=np.float32)

            # Read master sky flat
            sky_norm = master_sky.file2cube(file_index=idx_file, dtype=np.float32)

            # Flat-field data or subtract background
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                if self.setup.flat_type == "twilight":
                    temp = cube.copy()
                    temp.apply_masks(
                        sources=master_source_mask.file2cube(file_index=idx_file)
                    )
                    sky_level, _ = temp.background_planes()
                    cube -= sky_norm * sky_level[:, np.newaxis, np.newaxis]
                else:
                    cube /= sky_norm

            # Destriping
            if self.setup.destripe:
                sources = master_source_mask.file2cube(file_index=idx_file, dtype=bool)
                if self.setup.qc_plots:
                    path_qc_destripe = (
                        f"{self.setup.folders['qc_sky']}"
                        f"{self.basenames[idx_file]}_striping.pdf"
                    )
                else:
                    path_qc_destripe = None
                cube.destripe(
                    masks=sources,
                    smooth=True,
                    combine_bad_planes=True,
                    path_plot=path_qc_destripe,
                )

            # Background subtraction
            if self.setup.subtract_background:
                # Load source mask
                sources = master_source_mask.file2cube(file_index=idx_file)

                # Apply mask
                temp_cube = cube.copy()
                temp_cube.apply_masks(sources=sources)

                # Compute background and sigma
                bg, bgsig = temp_cube.background(
                    mesh_size=self.setup.background_mesh_size,
                    mesh_filtersize=self.setup.background_mesh_filtersize,
                )

                # Save to temp file
                # path_temp = f"{self.setup.folders['temp']}temp_background.fits"
                # bg.write_mef(path=path_temp, overwrite=True, dtype=np.float32)

                # Free ram
                del temp_cube
                gc.collect()

                # Save sky level
                sky, skysig = bg.median(axis=(1, 2)), bgsig.median(axis=(1, 2))

                # Subtract normalized background level
                cube -= bg

            # Otherwise just calculate the sky level
            else:
                sky, skysig = cube.background_planes()

            # Bad pixel interpolation
            if self.setup.interpolate_nan:
                cube.interpolate_nan()

            # Dummy check if too many pixels where masked
            for plane in cube:
                if np.sum(np.isfinite(plane)) / plane.size < 0.2:
                    ntot, nmasked = plane.size, np.sum(~np.isfinite(plane))
                    raise ValueError(
                        f"Too many pixels masked ({nmasked / ntot * 100}%) "
                        f"for file {self.basenames[idx_file]} "
                        f"with mask '{master_source_mask.basenames[idx_file]}'"
                    )

            # Add stuff to headers
            hdrs_data = []
            for idx_hdu in range(len(self.iter_data_hdu[idx_file])):
                # Make new header for current HDU
                hdr = self.headers_data[idx_file][idx_hdu].copy()

                hdr.set(
                    "SKY",
                    value=np.round(sky[idx_hdu], 3),
                    comment="Original sky value (ADU)",
                )
                hdr.set(
                    "SKYSIG",
                    value=np.round(skysig[idx_hdu], 3),
                    comment="Standard deviation of sky value (ADU)",
                )

                # Append modified header
                hdrs_data.append(hdr)

            # Add file info to main header
            hdr_prime = self.headers_primary[idx_file].copy()
            hdr_prime.set("SKYFILE", value=master_sky.basenames[idx_file])
            hdr_prime.set("MASKFILE", value=master_source_mask.basenames[idx_file])

            # Write to disk
            cube.write_mef(
                path=outpath,
                prime_header=hdr_prime,
                data_headers=hdrs_data,
                dtype="float32",
            )

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )


class SkyImagesProcessedScience(SkyImagesProcessed):
    def __init__(self, setup, file_paths=None):
        super(SkyImagesProcessedScience, self).__init__(
            setup=setup, file_paths=file_paths
        )

    def build_master_sky_static(self):
        # Processing info
        print_header(header="MASTER-SKY-STATIC", silent=self.setup.silent)
        tstart = time.time()

        # Check compatibility
        self.check_compatibility(n_ndit_max=1, n_filter_max=1)

        # Create name
        outpath = (
            f"{self.setup.folders['master_object']}"
            f"MASTER-SKY-STATIC.MJD_{self.mjd_mean:0.4f}.fits"
        )

        # Check if the file is already there
        if (
            check_file_exists(file_path=outpath, silent=self.setup.silent)
            and not self.setup.overwrite
        ):
            return

        # Instantiate output
        master_cube = ImageCube(setup=self.setup, cube=None)

        # Looping over detectors
        data_headers = []
        for idx_hdu in self.iter_data_hdu[0]:
            # Print processing info
            if not self.setup.silent:
                message_calibration(
                    n_current=1,
                    n_total=1,
                    name=outpath,
                    d_current=idx_hdu,
                    d_total=max(self.iter_data_hdu[0]),
                )

            # Load data
            cube = self.hdu2cube(hdu_index=idx_hdu)

            # Compute sky level in each plane
            sky, sky_std = cube.background_planes()

            # Mask 3 sigma level above sky
            cube.apply_masks(mask_above=(sky + 3 * sky_std)[:, np.newaxis, np.newaxis])

            # Normalize to same flux level
            sky_scale = sky / np.mean(sky)
            cube.normalize(norm=sky_scale)

            # Collapse cube
            collapsed = cube.flatten(
                metric=np.nanmedian, axis=0, weights=None, dtype=None
            )

            # Create header with sky measurements
            hdr = fits.Header()
            for cidx in range(len(sky)):
                hdr.set(
                    f"HIERARCH PYPE SKY MEAN {cidx}",
                    value=np.round(sky[cidx], 2),
                    comment="Measured sky (ADU)",
                )
                hdr.set(
                    f"HIERARCH PYPE SKY NOISE {cidx}",
                    value=np.round(sky_std[cidx], 2),
                    comment="Measured sky noise (ADU)",
                )
                hdr.set(
                    f"HIERARCH PYPE SKY MJD {cidx}",
                    value=np.round(self.mjd[cidx], 6),
                    comment="MJD of measured sky",
                )
                hdr.set(
                    f"HIERARCH PYPE SKY SCL {cidx}",
                    value=np.round(sky_scale[cidx], 6),
                    comment="Sky scale",
                )

            # Create header for extensions
            data_headers.append(hdr)

            # Collapse extensions with specified metric and append to output
            master_cube.extend(data=collapsed.astype(np.float32))

        # Create primary header
        hdr_prime = fits.Header()
        hdr_prime.set(keyword=self.setup.keywords.date_mjd, value=self.mjd_mean)
        hdr_prime.set(
            keyword=self.setup.keywords.date_ut, value=self.time_obs_mean.fits
        )
        hdr_prime.set(keyword=self.setup.keywords.object, value="MASTER-SKY-STATIC")
        hdr_prime.set(self.setup.keywords.dit, value=self.dit[0])
        hdr_prime.set(self.setup.keywords.ndit, value=self.ndit[0])
        hdr_prime.set(self.setup.keywords.filter_name, value=self.passband[0])
        hdr_prime.set("HIERARCH PYPE N_FILES", value=len(self))

        # Write to disk
        master_cube.write_mef(
            path=outpath, prime_header=hdr_prime, data_headers=data_headers
        )

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def build_master_weight_image(self):
        """This is unfortunately necessary since sometimes detector 16 in particular
        is weird."""

        # Processing info
        print_header(header="MASTER-WEIGHT-IMAGE", silent=self.setup.silent)
        tstart = time.time()

        # Generate weight outpaths
        outpaths = [
            (
                f"{self.setup.folders['master_object']}"
                f"MASTER-WEIGHT-IMAGE.MJD_{mjd:0.5f}.fits"
            )
            for mjd in self.mjd
        ]

        # MaxiMasking
        if self.setup.build_individual_weights_maximask:
            # Build commands for MaxiMask
            bin_mm = which("maximask")
            if bin_mm is None:
                raise ValueError("MaxiMask executable not found")
            cmds = [
                f"{bin_mm} {n} --batch_size 4 --single_mask True"
                for n in self.paths_full
            ]

            # Construct output names for masks
            paths_masks = [x.replace(".fits", ".masks.fits") for x in self.paths_full]

            # Clean commands
            cmds = [
                c
                for c, n, o in zip(cmds, paths_masks, outpaths)
                if not (os.path.exists(n) | os.path.exists(o))
            ]

            if len(paths_masks) != len(self):
                raise ValueError("Something went wrong with MaxiMask")

            # Run MaxiMask in parallel
            if len(cmds) > 0:
                print_message(f"Running MaxiMask on {len(cmds)} files with 2 threads")
            run_commands_shell_parallel(cmds=cmds, n_jobs=2, silent=True)

            # Put masks into FitsImages object
            masks = FitsImages(setup=self.setup, file_paths=paths_masks)
        else:
            masks = None

        # Fetch master
        master_weights = self.get_master_weight_global()
        master_source_masks = self.get_master_source_mask()

        # Loop over files
        for idx_file in range(self.n_files):
            # Check if the file is already there and skip if it is
            if (
                check_file_exists(
                    file_path=outpaths[idx_file], silent=self.setup.silent
                )
                and not self.setup.overwrite
            ):
                continue

            # Print processing info
            message_calibration(
                n_current=idx_file + 1,
                n_total=self.n_files,
                name=outpaths[idx_file],
                d_current=None,
                d_total=None,
                silent=self.setup.silent,
            )

            # Load data
            master_weight = master_weights.file2cube(file_index=idx_file)
            master_mask = master_source_masks.file2cube(file_index=idx_file)
            cube = self.file2cube(file_index=idx_file)

            # Apply MaxiMask to image if set for better background determination
            if isinstance(masks, FitsImages):
                bpm_mm = masks.file2cube(file_index=idx_file)
                cube.cube[(bpm_mm > 1) & (bpm_mm < 128)] = np.nan

            # Get background statistics
            cube.apply_masks(sources=master_mask)
            bg_rms = cube.background(mesh_size=256, mesh_filtersize=3)[1]

            # Read and apply MaxiMask to weight
            if isinstance(masks, FitsImages):
                mask = masks.file2cube(file_index=idx_file)
                master_weight.cube[(mask > 0) & (mask < 256)] = 0

            # Mask bad pixels
            try:
                # Try to just mask detector 16 (silly workaround)
                master_weight.cube[15][bg_rms[15] > 1.5 * np.nanmedian(bg_rms)] = 0
            except IndexError:
                # Mask all detectors
                master_weight.cube[bg_rms > 1.5 * np.nanmedian(bg_rms)] = 0

            # Make primary header
            prime_header = fits.Header()
            prime_header[self.setup.keywords.object] = "MASTER-WEIGHT-IMAGE"
            prime_header["HIERARCH PYPE SETUP MAXIMASK"] = (
                self.setup.build_individual_weights_maximask
            )
            prime_header[self.setup.keywords.date_mjd] = self.headers_primary[idx_file][
                self.setup.keywords.date_mjd
            ]

            # Write to disk
            master_weight.write_mef(path=outpaths[idx_file], prime_header=prime_header)

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def build_coadd_header(self, path_header: str = None):
        # Processing info
        print_header(header="TILE-HEADER", right=None, silent=self.setup.silent)
        tstart = time.time()

        # Set default output path
        if path_header is None:
            path_header = self.setup.path_coadd_header

        # Check if header exists
        if (
            check_file_exists(file_path=path_header, silent=self.setup.silent)
            and not self.setup.overwrite
        ):
            return

        # Fix VIRCAM headers if set (necessary for building field headers from raw data)
        if self.setup.fix_vircam_headers:
            for hidx in range(len(self)):
                # if not self.setup.silent:
                # if hidx % 5 == 0 or hidx == len(self) - 1:
                #     print(f"Fixing header {hidx + 1} out of {len(self)}")
                fix_vircam_headers(
                    prime_header=self.headers_primary[hidx],
                    data_headers=self.headers_data[hidx],
                )

        # Print message
        message_calibration(
            n_current=1,
            n_total=1,
            name=path_header,
            d_current=None,
            d_total=None,
            silent=self.setup.silent,
        )

        # Construct header from projection if set
        if self.setup.projection is not None:
            # Force the header in the setup, if set
            if self.setup.projection.force_header:
                header_coadd = self.setup.projection.header

            # Otherwise construct image limits (CRPIX1/2, NAXIS1/2)
            else:
                header_coadd = self.setup.projection.subheader_from_skycoord(
                    skycoord=self.footprints_flat, enlarge=0.5
                )

        # Otherwise construct from input
        else:
            # Get optimal rotation of frame
            rotation = np.round(find_optimal_rotation(self.footprints_flat), 2)
            header_coadd = skycoord2header(
                skycoord=self.footprints_flat,
                proj_code="ZEA",
                enlarge=1.0,
                rotation=np.deg2rad(np.round(rotation, 2)),
                round_crval=True,
                cdelt=(1 / 3) / 3600,
            )

        # Dummy check
        if (header_coadd["NAXIS1"] > 250000.0) or (header_coadd["NAXIS2"] > 250000.0):
            raise ValueError(
                f"Double check if the image size is correct "
                f"({header_coadd['NAXIS1']},{header_coadd['NAXIS2']})"
            )

        # Write coadd header to disk
        header_coadd.totextfile(path_header, overwrite=True)

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def resample(self):
        """Resamples images."""

        # Processing info
        print_header(header="RESAMPLING", silent=self.setup.silent)
        tstart = time.time()

        # Load Swarp setup
        sws = SwarpSetup(setup=self.setup)

        # Read YML and override defaults
        ss = yml2config(
            path_yml=sws.preset_resampling,
            imageout_name=self.setup.path_coadd,
            weightout_name=self.setup.path_coadd_weight,
            nthreads=self.setup.n_jobs,
            resample_suffix=sws.resample_suffix,
            resampling_type=self.setup.resampling_kernel.upper(),
            gain_keyword=self.setup.keywords.gain,
            satlev_keyword=self.setup.keywords.saturate,
            fscale_keyword="FSCLSTCK",
            skip=["weight_image", "weight_thresh", "resample_dir"],
        )

        # Make system temp directory for resampling
        path_resampled_dir = make_system_tempdir()

        # Construct commands for source extraction
        cmds = [
            (
                f"{sws.bin} -c {sws.default_config} {path_image} "
                f"-WEIGHT_IMAGE {weight} "
                f"-RESAMPLE_DIR {path_resampled_dir} {ss}"
            )
            for path_image, weight in zip(
                self.paths_full, self.get_master_weight_image().paths_full
            )
        ]

        # Run for each individual image and make MEF
        for idx_file in range(self.n_files):
            # Construct output path
            outpath = (
                f"{self.setup.folders['resampled']}"
                f"{self.names[idx_file]}"
                f"{sws.resample_suffix}"
            )
            outpath_weight = outpath.replace(".fits", ".weight.fits")

            # Check if file already exists
            if (
                check_file_exists(file_path=outpath, silent=self.setup.silent)
                and not self.setup.overwrite
            ):
                continue

            # Print processing info
            message_calibration(
                n_current=idx_file + 1,
                n_total=len(self),
                name=outpath,
                d_current=None,
                d_total=None,
                silent=self.setup.silent,
            )

            # Run Swarp
            run_command_shell(cmd=cmds[idx_file], silent=True)

            # Find images generated by swarp.
            paths_images = glob.glob(
                f"{path_resampled_dir}{self.names[idx_file]}*{sws.resample_suffix}"
            )
            paths_weights = [p.replace(".fits", ".weight.fits") for p in paths_images]

            # Check if files are there
            if len(paths_images) == 0:
                raise ValueError(
                    f"No resampled images found for file {self.names[idx_file]}"
                )

            # Construct MEF from resampled detectors
            make_mef_image(
                paths_input=sorted(paths_images),
                overwrite=self.setup.overwrite,
                path_output=outpath,
                primeheader=self.headers_primary[idx_file],
            )
            make_mef_image(
                paths_input=sorted(paths_weights),
                overwrite=self.setup.overwrite,
                path_output=outpath_weight,
                primeheader=self.headers_primary[idx_file],
            )

            # Remove intermediate files
            [os.remove(x) for x in paths_images]
            [os.remove(x) for x in paths_weights]

            # Copy header entries from original file
            merge_headers(path_1=outpath, path_2=self.paths_full[idx_file])

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )


class SkyImagesProcessedOffset(SkyImagesProcessed):
    def __init__(self, setup, file_paths=None):
        super(SkyImagesProcessedOffset, self).__init__(
            setup=setup, file_paths=file_paths
        )


class SkyImagesResampled(SkyImagesProcessed):
    def __init__(self, setup, file_paths=None):
        super(SkyImagesResampled, self).__init__(setup=setup, file_paths=file_paths)

    def build_stacks(self):
        # Processing info
        print_header(
            header="CREATING STACKS", silent=self.setup.silent, left=None, right=None
        )
        tstart = time.time()

        # Split based on Offset
        split = self.split_keywords(["OFFSET_I"])

        # Sort by mean MJD (just for convenience
        sidx = np.argsort([s.mjd_mean for s in split])
        split = [split[i] for i in sidx]

        # Check sequence
        if len(split) != 6:
            raise ValueError("Sequence contains {0} offsets. Expected 6.")

        for idx_split in range(len(split)):
            # Grab files
            files = split[idx_split]

            # Get index of first frame
            idx_first, idx_last = np.argmin(files.mjd), np.argmax(files.mjd)

            # Load Swarp setup
            sws = SwarpSetup(setup=files.setup)

            # Get current OFFSET ID
            oidx = files.read_from_prime_headers(keywords=["OFFSET_I"])[0][0]

            # Construct output paths for current stack
            path_stack = (
                f"{self.setup.folders['stacks']}{self.setup.name}_{oidx:02d}.stack.fits"
            )
            path_weight = path_stack.replace(".fits", ".weight.fits")

            # Check if file already exists
            if check_file_exists(file_path=path_stack, silent=self.setup.silent):
                continue

            # Print processing info
            message_calibration(
                n_current=idx_split + 1,
                n_total=len(split),
                name=path_stack,
                d_current=None,
                d_total=None,
                silent=self.setup.silent,
            )

            # Read fits header info from input files
            cpkw = [
                "NOFFSETS",
                "OFFSET_I",
                "NJITTER",
                "DEXTINCT",
                "SKY",
                "SKYSIG",
                "MJD-OBS",
                "AIRMASS",
                "ASTIRMS1",
                "ASTIRMS2",
                "ASTRRMS1",
                "ASTRRMS2",
            ]
            cpkw_data = files.read_from_data_headers(keywords=cpkw)
            cpkw_dict = dict(zip(cpkw, cpkw_data))

            # Loop over extensions
            paths_temp_stacks, paths_temp_weights = [], []
            for idx_data_hdu, idx_iter_hdu in zip(
                files.iter_data_hdu[0], range(len(files.iter_data_hdu[0]))
            ):
                # Construct output path
                paths_temp_stacks.append(f"{path_stack}_{idx_data_hdu:02d}.fits")
                paths_temp_weights.append(
                    f"{os.path.splitext(paths_temp_stacks[-1])[0]}.weight.fits"
                )

                # Build swarp options
                ss = yml2config(
                    path_yml=sws.preset_coadd,
                    imageout_name=paths_temp_stacks[-1],
                    weightout_name=paths_temp_weights[-1],
                    fscale_keyword="FSCLSTCK",
                    gain_keyword=self.setup.keywords.gain,
                    satlev_keyword=self.setup.keywords.saturate,
                    nthreads=self.setup.n_jobs,
                    skip=["weight_thresh", "weight_image"],
                )

                # Modify file paths with current extension
                paths_full_mod = [f"{x}[{idx_data_hdu}]" for x in files.paths_full]
                cmd = (
                    f"{sws.bin} "
                    f"{' '.format(idx_data_hdu).join(paths_full_mod)} "
                    f"-c {sws.default_config} {ss}"
                )

                # Run Swarp in bash (only bash understand the [ext] options,
                # zsh does not)
                run_command_shell(cmd=cmd, shell="bash", silent=True)

                # Modify FITS header of combined image
                with fits.open(paths_temp_stacks[-1], mode="update") as hdul:
                    # Set keywords
                    hdul[0].header.set(
                        "NOFFSETS", value=cpkw_dict["NOFFSETS"][idx_first][idx_iter_hdu]
                    )
                    hdul[0].header.set(
                        "OFFSET_I", value=cpkw_dict["OFFSET_I"][idx_first][idx_iter_hdu]
                    )
                    hdul[0].header.set(
                        "NJITTER", value=cpkw_dict["NJITTER"][idx_first][idx_iter_hdu]
                    )
                    hdul[0].header.set(
                        "DEXTINCT", value=cpkw_dict["DEXTINCT"][idx_first][idx_iter_hdu]
                    )
                    sky = np.mean(list(zip(*cpkw_dict["SKY"]))[idx_iter_hdu])
                    hdul[0].header.set("OSKY", value=np.round(sky, 3))
                    skysig = np.mean(list(zip(*cpkw_dict["SKYSIG"]))[idx_iter_hdu])
                    hdul[0].header.set("OSKYSIG", value=np.round(skysig, 3))
                    backmod, backsig, backskew = mmm(hdul[0].data)
                    hdul[0].header.set(
                        "BACKMOD", value=np.round(backmod, 3), comment="Background mode"
                    )
                    hdul[0].header.set(
                        "BACKSIG",
                        value=np.round(backsig, 3),
                        comment="Background sigma",
                    )
                    hdul[0].header.set(
                        "BACKSKEW",
                        value=np.round(backskew, 3),
                        comment="Background skew",
                    )
                    hdul[0].header.set(
                        "AIRMASS", value=cpkw_dict["AIRMASS"][idx_first][idx_iter_hdu]
                    )
                    hdul[0].header.set(
                        "MJD-OBS", value=cpkw_dict["MJD-OBS"][idx_first][idx_iter_hdu]
                    )
                    mjdend = (
                        cpkw_dict["MJD-OBS"][idx_last][idx_iter_hdu]
                        + (files.dit[idx_last] * files.ndit[idx_last]) / 86400.0
                    )
                    hdul[0].header.set("MJD-END", value=mjdend, after="MJD-OBS")
                    hdul[0].header.set(
                        "DATE-OBS",
                        value=mjd2dateobs(hdul[0].header["MJD-OBS"]),
                        before="MJD-OBS",
                    )

                    hdul.flush()

            # Grab primary header of first input image
            prhdr_first, prhdr_last = (
                files.headers_primary[idx_first],
                files.headers_primary[idx_last],
            )

            # Start with new clean header for stack output
            prhdr_stk = fits.Header()
            prhdr_stk.set(
                self.setup.keywords.object, prhdr_first[self.setup.keywords.object]
            )
            prhdr_stk.set(
                self.setup.keywords.date_mjd, prhdr_first[self.setup.keywords.date_mjd]
            )
            mjdend = (
                prhdr_last[self.setup.keywords.date_mjd]
                + (
                    prhdr_last[self.setup.keywords.dit]
                    * prhdr_last[self.setup.keywords.ndit]
                )
                / 86400.0
            )
            prhdr_stk.set("MJD-END", mjdend)
            prhdr_stk.set(
                self.setup.keywords.date_ut, prhdr_first[self.setup.keywords.date_ut]
            )
            prhdr_stk.set(self.setup.keywords.dit, prhdr_first[self.setup.keywords.dit])
            prhdr_stk.set(
                self.setup.keywords.ndit, prhdr_first[self.setup.keywords.ndit]
            )
            prhdr_stk.set(
                self.setup.keywords.filter_name,
                prhdr_first[self.setup.keywords.filter_name],
            )
            prhdr_stk.set("NJITTER", prhdr_first["NJITTER"])
            prhdr_stk.set("NOFFSETS", prhdr_first["NOFFSETS"])
            prhdr_stk.set("NUSTEP", prhdr_first["NUSTEP"])
            prhdr_stk.set("NCOMBINE", len(files))
            prhdr_stk.set(
                "HIERARCH ESO OBS PROG ID", prhdr_first["HIERARCH ESO OBS PROG ID"]
            )
            prhdr_stk.set("HIERARCH ESO OBS ID", prhdr_first["HIERARCH ESO OBS ID"])
            prhdr_stk.set("HIERARCH ESO DPR TECH", prhdr_first["HIERARCH ESO DPR TECH"])
            arcnames = files.read_from_prime_headers(keywords=["ARCFILE"])[0]
            for idx in range(len(arcnames)):
                prhdr_stk.set(f"HIERARCH PYPE ARCNAME {idx:02d}", arcnames[idx])

            # Also add estimate of internal and external astrometric RMS
            astirms1_mean = np.mean(list(zip(*cpkw_dict["ASTIRMS1"]))[0])
            astirms2_mean = np.mean(list(zip(*cpkw_dict["ASTIRMS2"]))[0])
            astrrms1_mean = np.mean(list(zip(*cpkw_dict["ASTRRMS1"]))[0])
            astrrms2_mean = np.mean(list(zip(*cpkw_dict["ASTRRMS2"]))[0])
            astirms = np.sqrt(astirms1_mean**2 + astirms2_mean**2) * 3_600_000
            astrrms = np.sqrt(astrrms1_mean**2 + astrrms2_mean**2) * 3_600_000
            prhdr_stk.set(
                "ASTIRMS",
                value=np.round(astirms, 2),
                comment="Internal astr. dispersion RMS (mas)",
            )
            prhdr_stk.set(
                "ASTRRMS",
                value=np.round(astrrms, 2),
                comment="External astr. dispersion RMS (mas)",
            )

            # Construct MEF from individual detectors
            make_mef_image(
                paths_input=sorted(paths_temp_stacks),
                overwrite=self.setup.overwrite,
                path_output=path_stack,
                primeheader=prhdr_stk,
            )
            make_mef_image(
                paths_input=sorted(paths_temp_weights),
                overwrite=self.setup.overwrite,
                path_output=path_weight,
                primeheader=prhdr_stk,
            )

            # Remove intermediate files
            [os.remove(x) for x in paths_temp_stacks]
            [os.remove(x) for x in paths_temp_weights]

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    # def equalize_zero_point(self, stack_catalogs):
    #
    #     # Processing info
    #     print_header(header="EQUALIZING ZERO POINT", silent=self.setup.silent,
    #                  left=os.path.basename(self.setup.path_coadd), right=None)
    #     tstart = time.time()
    #
    #     # Get photometric stability and ZP (AUTO) from catalog headers
    #     photstab = stack_catalogs.read_from_image_headers(keywords=["PHOTSTAB"])[0]
    #     zp_auto =
    #     stack_catalogs.read_from_data_
    #     headers(keywords=["HIERARCH PYPE ZP MAG_AUTO"])[0]
    #
    #     # Flatten lists
    #     photstab_flat, zp_auto_flat = flat_list(photstab), flat_list(zp_auto)
    #
    #     # Compute relative scaling from ZPs
    #     zp_median = np.mean(zp_auto_flat)
    #     scale_zp = [zp_median - zp for zp in zp_auto_flat]
    #     scale_zp = [10**(s/2.5) for s in scale_zp]
    #
    #     # Construct dict for flxscale modifier for each photometric stability ID
    #     scale_zp_dict = dict(zip(photstab_flat, scale_zp))
    #
    #     # Loop over images in instance and write tile scale into headers
    #     for idx_file in range(len(self)):
    #
    #         # Grab current file path
    #         path_file = self.paths_full[idx_file]
    #
    #         # Check if already modified
    #         if "FSCLMOD" in self.headers_primary[idx_file]:
    #             if self.headers_primary[idx_file]["FSCLMOD"] is True:
    #                 print_message(message="{0} already modified.
    #                 ".format(os.path.basename(path_file)),
    #                               kind="warning", end=None)
    #                 continue
    #
    #         # Print processing info
    #         message_calibration(n_current=idx_file + 1,
    #         n_total=self.n_files, name=path_file,
    #                             d_current=None,
    #                             d_total=None, silent=self.setup.silent)
    #
    #         # Open file
    #         file = fits.open(path_file, mode="update")
    #
    #         # Loop over data HDUs
    #         for idx_hdu in self.iter_data_hdu[idx_file]:
    #
    #             # Read header
    #             hdr = file[idx_hdu].header
    #
    #             # Add flux scales
    #             hdr.set("FSCLZERO", value=np.round(scale_zp_dict[hdr["PHOTSTAB"]], 6),
    #                     comment="Relative flux scaling from ZP", after="FSCLSTCK")
    #             hdr.set("FSCLTILE", value=np.round(hdr["FSCLSTCK"]
    #             * hdr["FSCLZERO"], 6),
    #                     comment="Total relative flux scaling for Tile",
    #                     after="FSCLZERO")
    #
    #         # Add modification flag to primary header
    #         file[0].header["FSCLMOD"] = (True, "Tile flux modified")
    #
    #         # Flush and close file
    #         file.close()
    #
    #         # Delete extraced header for current file
    #         self.delete_headers(idx_file=idx_file)
    #
    #     # Also flush header attribute for current instance at the end so
    #     that they are regenerated when requested
    #     self._headers = None
    #
    #     # Print time
    #     print_message(message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
    #     kind="okblue", end="\n")

    def build_tile(self):
        # Fetch log
        log = PipelineLog()

        # Processing info
        print_header(
            header="CREATING TILE",
            silent=self.setup.silent,
            left=os.path.basename(self.setup.path_coadd),
            right=None,
        )
        tstart = time.time()
        log.info(f"Creating tile with {len(self)} images:\n{self.basenames2log()}")

        # Load Swarp setup
        sws = SwarpSetup(setup=self.setup)
        log.info(f"Loaded Swarp preset used: {sws.preset_coadd}")

        # Make system temp dir for swap
        swap_dir = tempfile.mkdtemp(prefix="swarp_swap_")
        log.info(f"Create temporary swap directory: {swap_dir}")

        # Generate temporary coadd and coadd weight names on local disk
        tmpdir = tempfile.gettempdir()
        path_coadd_tmp = str(f"{tmpdir}/swarp_{uuid.uuid4().hex}.fits")
        log.info(f"Saving temporary coadd image to {path_coadd_tmp}")
        path_weight_tmp = str(f"{tmpdir}/swarp_weight_{uuid.uuid4().hex}.fits")
        log.info(f"Saving temporary coadd weight to {path_weight_tmp}")
        path_coadd_tmp_hdr = path_coadd_tmp.replace(".fits", ".ahead")

        # Copy coadd header to temp location
        shutil.copyfile(self.setup.path_coadd_header, path_coadd_tmp_hdr)
        log.info(f"Copied coadd header to temporary location {path_coadd_tmp_hdr}")

        ss = yml2config(
            path_yml=sws.preset_coadd,
            imageout_name=path_coadd_tmp,
            weightout_name=path_weight_tmp,
            fscale_keyword="FSCLTILE",
            gain_keyword=self.setup.keywords.gain,
            satlev_keyword=self.setup.keywords.saturate,
            nthreads=self.setup.n_jobs,
            skip=["weight_thresh", "weight_image"],
            vmem_dir=swap_dir,
        )

        # Write all full paths into a text file in the temp folder
        path_list = f"{self.setup.folders['temp']}swarp_input_files.txt"
        with open(path_list, "w") as f:
            for item in self.paths_full:
                f.write(f"{item}\n")

        # Construct commands for swarping
        cmd = f"{sws.bin} '@{path_list}' -c '{sws.default_config}' {ss}"

        # Add swarp command to log
        log.info(f"Swarp command: {cmd}")

        # Run Swarp
        if (
            not check_file_exists(
                file_path=self.setup.path_coadd, silent=self.setup.silent
            )
            and not self.setup.overwrite
        ):
            # Run Swarp
            stdout, stderr = run_command_shell(cmd=cmd, silent=True)

            # Wait 5 seconds to ensure file is written
            time.sleep(5)

            # Add stdout and stderr to log
            log.info(f"Scamp stdout:\n{stdout}")
            log.info(f"Scamp stderr:\n{stderr}")

            # Remove temporary file list
            try:
                os.remove(path_list)
            except FileNotFoundError:
                pass

            # Compute estimate of astrometric RMS
            cpkw = ["ASTIRMS1", "ASTIRMS2", "ASTRRMS1", "ASTRRMS2"]
            astrms_data = self.read_from_data_headers(keywords=cpkw)
            astrms_dict = dict(zip(cpkw, astrms_data))
            astirms = (
                np.sqrt(
                    np.mean(astrms_dict["ASTIRMS1"]) ** 2
                    + np.mean(astrms_dict["ASTIRMS2"]) ** 2
                )
                * 3_600_000
            )
            astrrms = (
                np.sqrt(
                    np.mean(astrms_dict["ASTRRMS1"]) ** 2
                    + np.mean(astrms_dict["ASTRRMS2"]) ** 2
                )
                * 3_600_000
            )
            log.info(f"Internal astr. dispersion RMS: {astirms:.2f} mas")
            log.info(f"External astr. dispersion RMS: {astrrms:.2f} mas")

            # Copy/add primary header entries
            with (
                fits.open(path_coadd_tmp, mode="update") as hdul_tile,
                fits.open(self.paths_full[0], mode="readonly") as hdu_paw0,
            ):
                hdul_tile[0].header.set(
                    keyword=self.setup.keywords.object,
                    value=hdu_paw0[0].header[self.setup.keywords.object],
                )
                hdul_tile[0].header.set(
                    keyword=self.setup.keywords.filter_name,
                    value=hdu_paw0[0].header[self.setup.keywords.filter_name],
                )
                hdul_tile[0].header.set(
                    keyword="ASTIRMS",
                    value=np.round(astirms, 2),
                    comment="Internal astr. dispersion RMS (mas)",
                )
                hdul_tile[0].header.set(
                    keyword="ASTRRMS",
                    value=np.round(astrrms, 2),
                    comment="External astr. dispersion RMS (mas)",
                )

                # Load coadd
                coadd = hdul_tile[0].data

                # Use a max of 50_000_000 pixels for sky calculation
                if coadd.size > 50_000_000:
                    backmod, backsig, backskew = mmm(coadd[:: coadd.size // 50_000_000])
                else:
                    backmod, backsig, backskew = mmm(coadd)
                log.info(f"Background mode: {backmod:.3f}")
                log.info(f"Background sigma: {backsig:.3f}")
                log.info(f"Background skew: {backskew:.3f}")

                # Put background info into header
                hdul_tile[0].header.set(
                    "BACKMODE", value=np.round(backmod, 3), comment="Background mode"
                )
                hdul_tile[0].header.set(
                    "BACKSIG", value=np.round(backsig, 3), comment="Background sigma"
                )
                hdul_tile[0].header.set(
                    "BACKSKEW", value=np.round(backskew, 3), comment="Background skew"
                )

                # Add setup to header
                self.setup.add_setup_to_header(header=hdul_tile[0].header)

                # Add archive names of input
                log.info("Adding ARCFILE names to header:")
                arcnames = self.read_from_prime_headers(keywords=["ARCFILE"])[0]
                for idx in range(len(arcnames)):
                    hdul_tile[0].header.set(
                        f"HIERARCH PYPE ARCNAME {idx:04d}", arcnames[idx]
                    )
                    log.info(f"ARCFILE {idx:04d}: {arcnames[idx]}")

                # Flush to disk
                hdul_tile.flush()

            # Move temporary files to final location
            shutil.move(path_coadd_tmp, self.setup.path_coadd)
            shutil.move(path_weight_tmp, self.setup.path_coadd_weight)
            log.info(
                f"Moved temporary coadd to final location: {self.setup.path_coadd}"
            )
            log.info(
                f"Moved temporary coadd weight to final location: "
                f"{self.setup.path_coadd_weight}"
            )

            # Remove swap dir
            shutil.rmtree(swap_dir, ignore_errors=True)

            # Remove temporary coadd header
            remove_file(path_coadd_tmp_hdr)

        # Print time
        ttime = time.time() - tstart
        log.info(f"Elapsed time for tile creation: {ttime:.2f}s")
        print_message(
            message=f"\n-> Elapsed time: {ttime:.2f}s",
            kind="okblue",
            end="\n",
        )

    def build_statistics(self):
        # Processing info
        print_header(
            header="IMAGE STATISTICS", silent=self.setup.silent, left=None, right=None
        )
        tstart = time.time()

        # Find weights
        master_weights = self.get_master_weight_global()

        # Create temporary output paths
        folder_statistics = self.setup.folders["statistics"]
        paths_nimg = [
            folder_statistics + bn.replace(".fits", ".nimg.fits")
            for bn in self.basenames
        ]
        paths_exp = [
            folder_statistics + bn.replace(".fits", ".exptime.fits")
            for bn in self.basenames
        ]
        paths_mjd_frac = [
            folder_statistics + bn.replace(".fits", ".mjd.frac.fits")
            for bn in self.basenames
        ]
        paths_mjd_int = [
            folder_statistics + bn.replace(".fits", ".mjd.int.fits")
            for bn in self.basenames
        ]
        paths_astrms1 = [
            folder_statistics + bn.replace(".fits", ".astrms1.fits")
            for bn in self.basenames
        ]
        paths_astrms2 = [
            folder_statistics + bn.replace(".fits", ".astrms2.fits")
            for bn in self.basenames
        ]
        paths_weight = [
            folder_statistics + bn.replace(".fits", ".weight.fits")
            for bn in self.basenames
        ]

        # Loop over files
        for idx_file in range(self.n_files):
            """
            I have to cheat here to get a 64bit MJD value in the coadd Swarp only
            produces 32 bit coadds for some reason, even if all input files (including
            weights) are passed as 64 bit images and the coadd header
            includes BITPIX=-64.
            """

            # Check if the file is already there and skip if it is
            if (
                check_file_exists(
                    file_path=paths_weight[idx_file], silent=self.setup.silent
                )
                and not self.setup.overwrite
            ):
                continue

            # Print processing info
            message_calibration(
                n_current=idx_file + 1,
                n_total=len(self),
                name=self.paths_full[idx_file],
                d_current=None,
                d_total=None,
                silent=self.setup.silent,
            )

            # Make primary header
            hdr_prime = fits.Header()
            hdr_prime["OFFSET_I"] = self.read_from_prime_headers(keywords=["OFFSET_I"])[
                0
            ][idx_file]

            # Create output HDULists
            hdul_nimg = fits.HDUList(hdus=[fits.PrimaryHDU(header=hdr_prime.copy())])
            hdul_exptime = fits.HDUList(hdus=[fits.PrimaryHDU(header=hdr_prime.copy())])
            hdul_mjd_int = fits.HDUList(hdus=[fits.PrimaryHDU(header=hdr_prime.copy())])
            hdul_mjd_frac = fits.HDUList(
                hdus=[fits.PrimaryHDU(header=hdr_prime.copy())]
            )
            hdul_astrms1 = fits.HDUList(hdus=[fits.PrimaryHDU(header=hdr_prime.copy())])
            hdul_astrms2 = fits.HDUList(hdus=[fits.PrimaryHDU(header=hdr_prime.copy())])
            hdul_weights = fits.HDUList(hdus=[fits.PrimaryHDU(header=hdr_prime.copy())])

            # Loop over extensions
            for idx_hdu in range(len(self.iter_data_hdu[idx_file])):
                # Read header
                header_original = self.headers_data[idx_file][idx_hdu]

                # Resize header and convert to WCS
                wcs_resized = header2wcs(
                    resize_header(
                        header=header_original,
                        factor=self.setup.image_statistics_resize_factor,
                    )
                )

                # Get shape
                shape = wcs_resized.pixel_shape[::-1]

                # Create image statistics arrays
                arr_nimg = np.full(shape, fill_value=1, dtype=np.uint16)
                arr_exptime = np.full(
                    shape,
                    fill_value=self.dit[idx_file] * self.ndit[idx_file],
                    dtype=np.float32,
                )
                mjd_frac, mjd_int = np.modf(self.mjd[idx_file])
                arr_mjd_int = np.full(shape, fill_value=mjd_int, dtype=np.float32)
                arr_mjd_frac = np.full(shape, fill_value=mjd_frac, dtype=np.float32)
                astirms1, astirms2, astrrms1, astrrms2 = self.read_from_data_headers(
                    keywords=["ASTIRMS1", "ASTIRMS2", "ASTRRMS1", "ASTRRMS2"],
                    file_index=idx_file,
                )
                astrms1 = 3_600_000 * np.sqrt(astirms1[0][0] ** 2 + astrrms1[0][0] ** 2)
                astrms2 = 3_600_000 * np.sqrt(astirms2[0][0] ** 2 + astrrms2[0][0] ** 2)
                arr_astrms1 = np.full(shape, fill_value=astrms1, dtype=np.float32)
                arr_astrms2 = np.full(shape, fill_value=astrms2, dtype=np.float32)

                # Read weight
                weight_hdu = fits.getdata(
                    master_weights.paths_full[idx_file], idx_hdu + 1
                )

                # Resize weight
                arr_weight = upscale_image(
                    weight_hdu, new_size=wcs_resized.pixel_shape, method="pil"
                )
                arr_weight[arr_weight < 0] = 0

                # Mask arrays where weight is 0
                # w0 = arr_weight / np.nanmedian(arr_weight) <= 0.00001
                # arr_nimg[w0], arr_exptime[w0], arr_mjdeff[w0] = 0, 0, np.nan

                # Create header and modify specifically for MJD
                header_resized = wcs_resized.to_header()

                # Extend HDULists
                hdul_nimg.append(
                    fits.ImageHDU(data=arr_nimg, header=header_resized)  # noqa
                )
                hdul_exptime.append(
                    fits.ImageHDU(data=arr_exptime, header=header_resized)  # noqa
                )
                hdul_mjd_frac.append(
                    fits.ImageHDU(data=arr_mjd_frac, header=header_resized)  # noqa
                )
                hdul_mjd_int.append(
                    fits.ImageHDU(data=arr_mjd_int, header=header_resized)  # noqa
                )
                hdul_astrms1.append(
                    fits.ImageHDU(data=arr_astrms1, header=header_resized)  # noqa
                )
                hdul_astrms2.append(
                    fits.ImageHDU(data=arr_astrms2, header=header_resized)  # noqa
                )
                hdul_weights.append(
                    fits.ImageHDU(
                        data=arr_weight.astype(np.float32),  # noqa
                        header=header_resized,
                    )
                )

            # Write to disk
            hdul_nimg.writeto(paths_nimg[idx_file], overwrite=True)
            hdul_exptime.writeto(paths_exp[idx_file], overwrite=True)
            hdul_mjd_frac.writeto(paths_mjd_frac[idx_file], overwrite=True)
            hdul_mjd_int.writeto(paths_mjd_int[idx_file], overwrite=True)
            hdul_astrms1.writeto(paths_astrms1[idx_file], overwrite=True)
            hdul_astrms2.writeto(paths_astrms2[idx_file], overwrite=True)
            hdul_weights.writeto(paths_weight[idx_file], overwrite=True)

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def coadd_statistics_stacks(self, mode):
        # Processing info
        print_header(
            header=f"STACKS STATISTICS {mode.upper()}",
            silent=self.setup.silent,
            left=None,
            right=None,
        )
        tstart = time.time()

        # Split based on OFFSET ID
        split = self.split_keywords(keywords=["OFFSET_I"])

        # Loop over OFFSETs
        for idx_split in range(len(split)):
            # Grab current files
            files = split[idx_split]

            # Load Swarp setup
            sws = SwarpSetup(setup=files.setup)

            # Get current OFFSET ID
            oidx = files.read_from_prime_headers(keywords=["OFFSET_I"])[0][0]

            # Get weight paths
            weights = [p.replace(mode, "weight") for p in files.paths_full]

            # Check for existence of weight maps
            if np.sum([os.path.isfile(w) for w in weights]) != len(files):
                raise ValueError("Images and weights not synced")

            # Construct output paths for current stack
            path_stack = (
                f"{self.setup.folders['stacks']}"
                f"{self.setup.name}_{oidx:02d}.stack.{mode}.fits"
            )

            # Check if file already exists
            if check_file_exists(file_path=path_stack, silent=self.setup.silent):
                continue

            # Print processing info
            message_calibration(
                n_current=idx_split + 1,
                n_total=len(split),
                name=path_stack,
                d_current=None,
                d_total=None,
                silent=self.setup.silent,
            )

            # Generate temporary coadd and coadd weight names on local disk
            tmpdir = tempfile.gettempdir()

            # Loop over extensions
            paths_temp_stacks, paths_temp_weights = [], []
            for idx_data_hdu, idx_iter_hdu in zip(
                files.iter_data_hdu[0], range(len(files.iter_data_hdu[0]))
            ):
                # Construct output path
                paths_temp_stacks.append(str(f"{tmpdir}/swarp_{uuid.uuid4().hex}.fits"))
                paths_temp_weights.append(
                    f"{os.path.splitext(paths_temp_stacks[-1])[0]}.weight.fits"
                )

                # Modify file paths with current extension
                paths_image_mod = [f"{x}[{idx_data_hdu}]" for x in files.paths_full]
                paths_weight_mod = [f"{x}[{idx_data_hdu}]" for x in weights]

                # Build swarp options
                ss = yml2config(
                    path_yml=sws.preset_coadd,
                    imageout_name=paths_temp_stacks[-1],
                    weightout_name=paths_temp_weights[-1],
                    skip=["weight_thresh"],
                    weight_image=",".join(paths_weight_mod),
                    nthreads=self.setup.n_jobs,
                    combine_type=self.setup.image_statistics_combine_type[mode],
                )

                # Construct final command
                cmd = (
                    f"{sws.bin} {' '.format(idx_data_hdu).join(paths_image_mod)} "
                    f"-c {sws.default_config} {ss}"
                )

                # Run Swarp in bash (zsh does not understand the [ext] options)
                run_command_shell(cmd=cmd, shell="bash", silent=True)

            # Create MEF image
            make_mef_image(
                paths_input=sorted(paths_temp_stacks),
                overwrite=self.setup.overwrite,
                path_output=path_stack,
                primeheader=None,
            )

            # Convert data types
            if "nimg" in mode.lower():
                convert_bitpix_image(path=path_stack, new_type=np.uint16)
            elif "exptime" in mode.lower():
                convert_bitpix_image(path=path_stack, new_type=np.float32)
            elif "mjd" in mode.lower():
                convert_bitpix_image(path=path_stack, new_type=np.float32)
            else:
                raise ValueError

            # Remove temporary files
            [remove_file(f) for f in paths_temp_stacks]
            [remove_file(f) for f in paths_temp_weights]

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )

    def coadd_statistics_tile(self, mode):
        # Processing info
        print_header(
            header=f"TILE STATISTICS {mode.upper()}",
            silent=self.setup.silent,
            left=None,
            right=None,
        )
        tstart = time.time()

        # Construct output path
        outpath_final = self.setup.path_coadd.replace(".fits", f".{mode}.fits")
        outpath_final_weight = outpath_final.replace(".fits", ".weight.fits")

        # Check if file already exists
        if check_file_exists(file_path=outpath_final, silent=self.setup.silent):
            return

        # Generate temporary coadd and coadd weight names on local disk
        tmpdir = tempfile.gettempdir()
        outpath_temp = str(f"{tmpdir}/swarp_{uuid.uuid4().hex}.fits")
        outpath_temp_weight = outpath_temp.replace(".fits", ".weight.fits")

        # Get weight paths
        paths_weight = [p.replace(mode, "weight") for p in self.paths_full]

        if np.sum([os.path.isfile(w) for w in paths_weight]) != self.n_files:
            raise ValueError("Not all images have weights")

        # Load Swarp setup
        sws = SwarpSetup(setup=self.setup)

        # Save file lists to temporary folder
        path_temp_images = self.setup.folders["temp"] + "swarp_images.lis"
        path_temp_weights = self.setup.folders["temp"] + "swarp_weights.lis"

        with open(path_temp_images, "w") as f1, open(path_temp_weights, "w") as f2:
            f1.write("\n".join(self.paths_full))
            f2.write("\n".join(paths_weight))

        ss = yml2config(
            path_yml=sws.preset_coadd,
            weight_image=f"@{path_temp_weights}",
            imageout_name=outpath_temp,
            weightout_name=outpath_temp_weight,
            combine_type=self.setup.image_statistics_combine_type[mode],
            nthreads=self.setup.n_jobs,
            skip=["weight_thresh", "weight_suffix"],
        )

        # Construct commands for source extraction
        cmd = f"{sws.bin} @{path_temp_images} -c {sws.default_config} {ss}"

        # Run Swarp
        print_message(message=f"Coadding {os.path.basename(outpath_final)}")
        run_command_shell(cmd=cmd, silent=True)

        # Move temp to final
        shutil.move(outpath_temp, outpath_final)
        shutil.move(outpath_temp_weight, outpath_final_weight)

        # Remove temp files
        remove_file(path_temp_images)
        remove_file(path_temp_weights)

        # Convert data types
        if "nimg" in mode.lower():
            convert_bitpix_image(path=outpath_final, new_type=np.uint16)
        elif "exptime" in mode.lower():
            convert_bitpix_image(path=outpath_final, new_type=np.float32)
        elif "mjd" in mode.lower():
            convert_bitpix_image(path=outpath_final, new_type=np.float32)
        elif "astrms" in mode.lower():
            convert_bitpix_image(path=outpath_final, new_type=np.float32)
        else:
            raise ValueError

        # Print time
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
        )


class Stacks(SkyImages):
    def __init__(self, setup, file_paths=None):
        super(Stacks, self).__init__(setup=setup, file_paths=file_paths)


class Tile(SkyImages):
    def __init__(self, setup, file_paths=None):
        super(Tile, self).__init__(setup=setup, file_paths=file_paths)


class MasterSky(MasterImages):
    def __init__(self, setup, file_paths=None):
        super(MasterSky, self).__init__(setup=setup, file_paths=file_paths)

    _sky = None

    @property
    def sky(self):
        """
        Reads sky levels from headers.

        Returns
        -------
        List
            List of sky levels.

        """

        # Check if _sky determined
        if self._sky is not None:
            return self._sky

        self._sky = self._read_sequence_from_data_headers(
            keyword="HIERARCH PYPE SKY MEAN"
        )
        return self._sky

    _noise = None

    @property
    def noise(self):
        """
        Reads sky noise levels from headers.

        Returns
        -------
        List
            List of sky noise levels.

        """

        # Check if _sky determined
        if self._noise is not None:
            return self._noise

        self._noise = self._read_sequence_from_data_headers(
            keyword="HIERARCH PYPE SKY NOISE"
        )
        return self._noise

    _sky_mjd = None

    @property
    def sky_mjd(self):
        """
        Reads sky levels from headers.

        Returns
        -------
        List
            List of sky levels.

        """

        # Check if _sky determined
        if self._sky_mjd is not None:
            return self._sky_mjd

        self._sky_mjd = self._read_sequence_from_data_headers(
            keyword="HIERARCH PYPE SKY MJD"
        )
        return self._sky_mjd

    # =========================================================================== #
    # QC
    # =========================================================================== #
    def paths_qc_plots(self, paths, mode):
        """
        Generates paths for QC plots

        Parameters
        ----------
        paths : iterable
            Input paths to override internal paths
        mode : str
            QC mode. Can be either "sky" or "sky_stability".

        Returns
        -------
        iterable
            List of paths.
        """

        if paths is None:
            if mode == "sky":
                return [
                    f"{self.setup.folders['qc_sky']}{fp}.pdf" for fp in self.basenames
                ]
            elif mode == "sky_stability":
                return [
                    f"{self.setup.folders['qc_sky']}{fp}_stability.pdf"
                    for fp in self.basenames
                ]
            else:
                raise ValueError(f"Unknown QC sky plot mode: {mode}")
        else:
            return paths

    def qc_plot_sky(self, paths=None, axis_size=5):
        """
        Generates a simple QC plot for BPMs.

        Parameters
        ----------
        paths : list, optional
            Paths of the QC plot files. If None (default), use relative paths.
        axis_size : int, float, optional
            Axis size. Default is 5.

        """

        # Import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Plot paths
        paths = self.paths_qc_plots(paths=paths, mode="sky")

        for sky, noise, mjd, path in zip(self.sky, self.noise, self.sky_mjd, paths):
            # Get plot grid
            fig, axes = get_plotgrid(
                layout=self.setup.fpa_layout, xsize=axis_size, ysize=axis_size
            )
            axes = axes.ravel()

            # Helpers
            mjd_floor = np.floor(np.min(mjd))
            xmin, xmax = (
                0.999 * np.min(24 * (mjd - mjd_floor)),
                1.001 * np.max(24 * (mjd - mjd_floor)),
            )
            maxnoise = np.max([i for s in noise for i in s])
            allsky = np.array([i for s in sky for i in s])
            ymin, ymax = (
                0.98 * np.min(allsky) - maxnoise,
                1.02 * np.max(allsky) + maxnoise,
            )

            # Plot
            for idx in range(len(sky)):
                # Grab axes
                ax = axes[idx]

                # Plot sky levels
                ax.scatter(
                    24 * (mjd[idx] - mjd_floor),
                    sky[idx],
                    c="#DC143C",
                    lw=0,
                    s=40,
                    alpha=1,
                    zorder=1,
                )
                ax.errorbar(
                    24 * (mjd[idx] - mjd_floor),
                    sky[idx],
                    yerr=noise[idx],
                    ecolor="#101010",
                    fmt="none",
                    zorder=0,
                )

                # Annotate detector ID
                ax.annotate(
                    f"Det.ID: {idx + 1:0d}",
                    xy=(0.04, 0.04),
                    xycoords="axes fraction",
                    ha="left",
                    va="bottom",
                )

                # Modify axes
                if idx < self.setup.fpa_layout[1]:
                    ax.set_xlabel(f"MJD (h) + {mjd_floor:0n}d")
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if idx % self.setup.fpa_layout[0] == self.setup.fpa_layout[0] - 1:
                    ax.set_ylabel("ADU")
                else:
                    ax.axes.yaxis.set_ticklabels([])

                # Set ranges
                ax.set_xlim(
                    xmin=floor_value(data=xmin, value=0.02),
                    xmax=ceil_value(data=xmax, value=0.02),
                )
                ax.set_ylim(
                    ymin=floor_value(data=ymin, value=50),
                    ymax=ceil_value(data=ymax, value=50),
                )

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
                fig.savefig(path, bbox_inches="tight")
            plt.close("all")

    def qc_plot_sky_stability(self, paths=None, axis_size=5):
        # Import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator

        # Plot paths
        paths = self.paths_qc_plots(paths=paths, mode="sky_stability")

        for sky, mjd, path in zip(self.sky, self.sky_mjd, paths):
            # Subtract median from eavh sky level
            sky = [s - np.median(s) for s in sky]

            # Get some helper variables
            mjd_floor = np.floor(np.min(mjd))
            xmin, xmax = (
                0.999 * np.min(24 * (mjd - mjd_floor)),
                1.001 * np.max(24 * (mjd - mjd_floor)),
            )
            ymin, ymax = np.min(sky), np.max(sky)

            # Create figure
            fig, ax = plt.subplots(
                nrows=1, ncols=1, **{"figsize": [axis_size, axis_size * 0.6]}
            )

            for ss, mm in zip(sky, mjd):
                ax.plot(24 * (mm - mjd_floor), ss, alpha=0.75)

            # Labels
            ax.set_xlabel(f"MJD (h) + {mjd_floor:0n}d")
            ax.set_ylabel("Sky level - median (ADU)")

            # Set ranges
            ax.set_xlim(
                xmin=floor_value(data=xmin, value=0.02),
                xmax=ceil_value(data=xmax, value=0.02),
            )
            ax.set_ylim(
                ymin=floor_value(data=ymin, value=10),
                ymax=ceil_value(data=ymax, value=10),
            )

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
                fig.savefig(path, bbox_inches="tight")
            plt.close("all")


class MasterSourceMask(MasterImages):
    def __init__(self, setup, file_paths=None):
        super(MasterSourceMask, self).__init__(setup=setup, file_paths=file_paths)
