import functools
import glob
import json
import logging
import math
import os.path
import resource
import sys
import time

from vircampype.fits.images.common import FitsImages
from vircampype.pipeline.errors import PipelineValueError
from vircampype.pipeline.logsetup import configure_logging, get_logger
from vircampype.pipeline.setup import Setup
from vircampype.pipeline.status import PipelineStatus
from vircampype.tools.astromatic import (
    verify_sextractor_multi_seeing,
    verify_sextractor_winpos_cap,
)
from vircampype.tools.completeness import qc_completeness_tile
from vircampype.tools.fitstools import (
    build_qc_summary,
    combine_mjd_images,
    compress_images,
)
from vircampype.tools.messaging import print_end, print_stage_skip, print_start
from vircampype.tools.systemtools import *


def _max_rss_gb() -> float:
    """Peak resident set size of this process in GB (bytes on macOS, kB elsewhere)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (2**30 if sys.platform == "darwin" else 2**20)


def pipeline_step(status_attr: str, *, message: str, guard: str | None = None):
    """Decorator that wraps a Pipeline method with status-check boilerplate.

    Skips execution if the step is already done (prints a warning instead).
    Optionally skips silently if a guard attribute on ``self`` is None.
    On success, sets the status flag to True automatically.

    Parameters
    ----------
    status_attr : str
        Name of the boolean attribute on ``self.status``.
    message : str
        Label used in the "already done" warning (e.g. ``"MASTER-BPM"``).
    guard : str, optional
        If given, the method is skipped entirely when
        ``getattr(self, guard)`` is None (data not available).
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            log = logging.getLogger(__name__)
            if guard is not None and getattr(self, guard) is None:
                log.debug(f"Skipping {message}: guard '{guard}' is None")
                return
            if getattr(self.status, status_attr):
                # Expected resume skip: INFO not warning; one console line per
                # stage.
                log.info(f"Skipping {message}: already complete")
                if not self.setup.silent:
                    print_stage_skip(message)
                return
            log.info(f"START {message}")
            t0 = time.monotonic()
            method(self, *args, **kwargs)
            self.update_status(**{status_attr: True})
            log.info(
                f"DONE {message} ({time.monotonic() - t0:.0f} s, "
                f"process max RSS so far {_max_rss_gb():.1f} GB incl. mmap; status {status_attr} set)"
            )

        return wrapper

    return decorator


def _notify_on_completion(method):
    """Decorator that times a pipeline processor and sends a Pushover notification."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        t0 = print_start(obj=self.setup.name)
        try:
            method(self, *args, **kwargs)
        except Exception as e:
            # When the pipeline is driven externally (not via worker.py), this
            # is the only place the failure can reach the file log.
            logging.getLogger(__name__).exception("Pipeline processing failed")
            self._notify_pushover(f"Failed with {type(e).__name__}: {e}")
            raise
        elapsed = time.time() - t0
        print_end(tstart=t0, logger=self.log)
        self._notify_pushover(f"Processing finished in {elapsed / 60:.1f} min.")

    return wrapper


class Pipeline:
    """Top-level pipeline orchestrator for VIRCAM data processing.

    Loads a ``Setup`` from a YAML configuration file, tracks processing
    progress via ``PipelineStatus``, and exposes the full calibration and
    science reduction chain as checkpoint-guarded methods.

    Parameters
    ----------
    setup : str or dict or Setup
        Pipeline configuration (YAML path, dict, or ``Setup`` instance).
    reset_status : bool, optional
        If True, reset all pipeline progress flags on initialisation.
    """

    def __init__(self, setup, reset_status=False, **kwargs):
        self.setup = Setup.load_pipeline_setup(setup, **kwargs)

        # Start logging
        configure_logging(self.setup)
        self.log = get_logger()

        self.log.info("Initializing Pipeline")
        self.log.info(f"Pipeline setup: {json.dumps(self.setup.to_dict, indent=4)}")

        # Instantiate status
        self.log.info("Instantiating pipeline status")
        self.status = PipelineStatus()

        # Read status
        self.log.info("Try to read previous pipeline status")
        self.path_status = os.path.join(self.setup.folders["temp"], "pipeline_status.p")
        try:
            self.status.load(path=self.path_status)
            self.log.info(f"Previous pipeline status found at {self.path_status}")
        except FileNotFoundError:
            self.log.info(f"No previous pipeline status found at {self.path_status}")

        # Reset status if requested
        if reset_status:
            self.reset_status()
            self.log.info("Pipeline status has been reset.")

        # Log status
        self.log.info(
            f"Current pipeline status: {json.dumps(self.status.dict, indent=4)}"
        )

        # Verify SExtractor supports multi-seeing (classification runs for every tile)
        if self.setup.build_tile:
            self.log.info("Verifying SExtractor multi-seeing support for the tile")
            verify_sextractor_multi_seeing(setup=self.setup)
            self.log.info("SExtractor multi-seeing support verified")
        if self.setup.sex_winpos_cap:
            verify_sextractor_winpos_cap(setup=self.setup)
            self.log.info("SExtractor window-cap support verified")

    # =========================================================================== #
    def __str__(self):
        return self.status.__str__()

    def __repr__(self):
        return self.status.__repr__()

    def update_status(self, **kwargs):
        """Update pipeline status flags and persist to disk."""
        self.status.update(**kwargs)
        self.status.save(path=self.path_status)

    def reset_status(self):
        """Reset all pipeline status flags to False and persist."""
        self.status.reset()
        self.status.save(path=self.path_status)

    # =========================================================================== #
    # Raw files
    @property
    def raw(self):
        return FitsImages.from_folder(
            path=self.setup.folders["raw"],
            pattern=self.setup.pattern_data,
            setup=self.setup,
            exclude=None,
        )

    @property
    def paths_raw(self):
        return self.raw.paths_full

    _raw_split = None

    # =========================================================================== #
    # VIRCAM splitter
    @property
    def raw_split(self):
        if self._raw_split is not None:
            return self._raw_split

        self._raw_split = self.raw.split_types()
        return self._raw_split

    @property
    def raw_science(self):
        return self.raw_split["science"]

    @property
    def raw_offset(self):
        return self.raw_split["offset"]

    @property
    def raw_science_and_offset(self):
        if self.raw_offset is not None:
            return self.raw_science + self.raw_offset
        else:
            return self.raw_science

    @property
    def raw_std(self):
        return self.raw_split["std"]

    @property
    def dark_science(self):
        return self.raw_split["dark_science"]

    @property
    def dark_lin(self):
        return self.raw_split["dark_lin"]

    @property
    def dark_check(self):
        return self.raw_split["dark_check"]

    @property
    def dark_gain(self):
        return self.raw_split["dark_gain"]

    @property
    def dark_all(self):
        try:
            return self.dark_science + self.dark_lin + self.dark_gain
        except TypeError:
            return None

    @property
    def flat_twilight(self):
        return self.raw_split["flat_twilight"]

    @property
    def flat_lamp_lin(self):
        return self.raw_split["flat_lamp_lin"]

    @property
    def flat_lamp_check(self):
        return self.raw_split["flat_lamp_check"]

    @property
    def flat_lamp_gain(self):
        return self.raw_split["flat_lamp_gain"]

    # =========================================================================== #
    # Processed image data
    @property
    def _paths_processed_basic_science(self):
        return [
            (
                f"{self.setup.folders['processed_basic']}"
                f"{self.raw_science.names[i]}"
                f".proc.basic.fits"
            )
            for i in range(self.raw_science.n_files)
        ]

    @property
    def processed_basic_science(self):
        # Instantiate
        from vircampype.fits.images.sky import SkyImagesProcessedScience

        images = SkyImagesProcessedScience(
            file_paths=self._paths_processed_basic_science, setup=self.setup
        )

        # Consistency check
        if len(images) != len(self.raw_science):
            raise PipelineValueError("Raw and processed science images not matching.")

        return images

    @property
    def _paths_processed_basic_offset(self):
        return [
            (
                f"{self.setup.folders['processed_basic']}"
                f"{self.raw_offset.names[i]}"
                f".proc.basic.fits"
            )
            for i in range(self.raw_offset.n_files)
        ]

    @property
    def processed_basic_offset(self):
        # Return if no offset files
        if self.raw_offset is None:
            return None

        # Instantiate
        from vircampype.fits.images.sky import SkyImagesProcessedOffset

        images = SkyImagesProcessedOffset(
            file_paths=self._paths_processed_basic_offset, setup=self.setup
        )

        # Consistency check
        if len(images) != len(self.raw_offset):
            raise PipelineValueError("Raw and processed offset images not matching.")

        return images

    @property
    def processed_basic_science_and_offset(self):
        if self.processed_basic_offset is not None:
            return self.processed_basic_science + self.processed_basic_offset
        else:
            return self.processed_basic_science

    @property
    def _paths_processed_science_final(self):
        return [
            (
                f"{self.setup.folders['processed_final']}"
                f"{self.raw_science.names[i]}"
                f".proc.final.fits"
            )
            for i in range(self.raw_science.n_files)
        ]

    @property
    def processed_science_final(self):
        # Instantiate
        from vircampype.fits.images.sky import SkyImagesProcessedScience

        images = SkyImagesProcessedScience(
            file_paths=self._paths_processed_science_final, setup=self.setup
        )

        # Consistency check
        if len(images) != len(self.raw_science):
            raise PipelineValueError("Raw and processed science images not matching.")

        return images

    @property
    def _paths_illumination_corrected(self):
        return [
            (
                f"{self.setup.folders['illumcorr']}"
                f"{self.raw_science.names[i]}"
                f".proc.final.ic.fits"
            )
            for i in range(self.raw_science.n_files)
        ]

    @property
    def illumination_corrected(self):
        from vircampype.fits.images.sky import SkyImagesProcessedScience

        return SkyImagesProcessedScience.from_folder(
            path=self.setup.folders["illumcorr"],
            pattern="*.proc.final.ic.fits",
            setup=self.setup,
            exclude=None,
        )

    @property
    def _paths_resampled(self):
        return [
            (
                f"{self.setup.folders['resampled']}"
                f"{self.raw_science.names[i]}"
                f".proc.final.ic.resamp.fits"
            )
            for i in range(self.raw_science.n_files)
        ]

    @property
    def resampled(self):
        # Instantiate
        from vircampype.fits.images.sky import SkyImagesResampled

        images = SkyImagesResampled(file_paths=self._paths_resampled, setup=self.setup)

        # Consistency check
        if len(images) != len(self.raw_science):
            raise PipelineValueError("Raw and resampled images not matching.")

        return images

    @property
    def _paths_stacks(self):
        return sorted(glob.glob(f"{self.setup.folders['stacks']}*.stack.fits"))

    @property
    def stacks(self):
        # Instantiate
        from vircampype.fits.images.sky import Stacks

        images = Stacks(file_paths=self._paths_stacks, setup=self.setup)

        # Consistency check
        if len(images) != self.setup.n_offset_positions:
            raise PipelineValueError("Stacks incomplete")

        return images

    @property
    def pawprints(self):
        from vircampype.fits.images.sky import Pawprints

        paths = self.resampled.paths_pawprints
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise PipelineValueError(
                f"{len(missing)} pawprint products missing (run build_pawprints), "
                f"e.g. {missing[0]}"
            )
        return Pawprints(file_paths=paths, setup=self.setup)

    @property
    def tile(self):
        from vircampype.fits.images.sky import Tile

        return Tile(setup=self.setup, file_paths=self.setup.path_coadd)

    # =========================================================================== #
    # Source catalogs
    @property
    def _paths_sources_processed_final_scamp(self):
        return [
            p.replace(".fits", ".scamp.fits.tab")
            for p in self._paths_processed_science_final
        ]

    @property
    def sources_processed_final_scamp(self):
        # Instantiate
        from vircampype.fits.tables.sextractor import SextractorCatalogs

        catalogs = SextractorCatalogs(
            file_paths=self._paths_sources_processed_final_scamp, setup=self.setup
        )

        # Consistency check
        if len(catalogs) != len(self.processed_science_final):
            raise PipelineValueError("Images and catalogs not matching.")

        # Return
        return catalogs

    @property
    def _paths_sources_processed_illumcorr(self):
        return [
            p.replace(".fits", ".ic.fits.tab")
            for p in self._paths_processed_science_final
        ]

    @property
    def sources_processed_illumcorr(self):
        # Instantiate
        from vircampype.fits.tables.sextractor import (
            AstrometricCalibratedSextractorCatalogs,
        )

        catalogs = AstrometricCalibratedSextractorCatalogs(
            file_paths=self._paths_sources_processed_illumcorr, setup=self.setup
        )

        # Consistency check
        if len(catalogs) != len(self.processed_science_final):
            raise PipelineValueError("Images and catalogs not matching.")

        # Return
        return catalogs

    @property
    def _paths_sources_resampled_full(self):
        return [p.replace(".fits", ".full.fits.tab") for p in self._paths_resampled]

    @property
    def sources_resampled_full(self):
        # Instantiate
        from vircampype.fits.tables.sextractor import (
            AstrometricCalibratedSextractorCatalogs,
        )

        catalogs = AstrometricCalibratedSextractorCatalogs(
            file_paths=self._paths_sources_resampled_full, setup=self.setup
        )

        # Consistency check
        if len(catalogs) != len(self.resampled):
            raise PipelineValueError("Images and catalogs not matching.")

        # Return
        return catalogs

    @property
    def _paths_sources_stacks_full(self):
        return [p.replace(".fits", ".full.fits.tab") for p in self._paths_stacks]

    @property
    def sources_stacks_full(self):
        # Instantiate
        from vircampype.fits.tables.sextractor import (
            AstrometricCalibratedSextractorCatalogs,
        )

        catalogs = AstrometricCalibratedSextractorCatalogs(
            file_paths=self._paths_sources_stacks_full, setup=self.setup
        )

        # Consistency check
        if len(catalogs) != len(self.stacks):
            raise PipelineValueError("Images and catalogs not matching.")

        # Return
        return catalogs

    @property
    def _paths_sources_resampled_cal(self):
        return [
            p.replace(".fits.tab", ".fits.ctab")
            for p in self._paths_sources_resampled_full
        ]

    @property
    def sources_resampled_cal(self):
        # Instantiate
        from vircampype.fits.tables.sextractor import (
            PhotometricCalibratedSextractorCatalogs,
        )

        catalogs = PhotometricCalibratedSextractorCatalogs(
            file_paths=self._paths_sources_resampled_cal, setup=self.setup
        )

        # Consistency check
        if len(catalogs) != len(self.resampled):
            raise PipelineValueError("Images and catalogs not matching.")

        # Return
        return catalogs

    @property
    def _paths_sources_stacks_cal(self):
        return [
            p.replace(".fits.tab", ".fits.ctab")
            for p in self._paths_sources_stacks_full
        ]

    @property
    def sources_stacks_cal(self):
        # Instantiate
        from vircampype.fits.tables.sextractor import (
            PhotometricCalibratedSextractorCatalogs,
        )

        catalogs = PhotometricCalibratedSextractorCatalogs(
            file_paths=self._paths_sources_stacks_cal, setup=self.setup
        )

        # Consistency check
        if len(catalogs) != len(self.stacks):
            raise PipelineValueError("Images and catalogs not matching.")

        # Return
        return catalogs

    @property
    def _path_sources_tile_full(self):
        return self.setup.path_coadd.replace(".fits", ".full.fits.tab")

    @property
    def sources_tile_full(self):
        # Instantiate
        from vircampype.fits.tables.sextractor import (
            AstrometricCalibratedSextractorCatalogs,
        )

        catalog = AstrometricCalibratedSextractorCatalogs(
            file_paths=self._path_sources_tile_full, setup=self.setup
        )

        if len(catalog) != 1:
            raise PipelineValueError("Tile catalog not found")

        # Return
        return catalog

    @property
    def _path_sources_tile_cal(self):
        return self.setup.path_coadd.replace(".fits", ".full.fits.ctab")

    @property
    def sources_tile_cal(self):
        # Instantiate
        from vircampype.fits.tables.sextractor import (
            PhotometricCalibratedSextractorCatalogs,
        )

        catalog = PhotometricCalibratedSextractorCatalogs(
            file_paths=self._path_sources_tile_cal, setup=self.setup
        )

        if len(catalog) != 1:
            raise PipelineValueError("Calibrated tile catalog not found")

        # Return
        return catalog

    def _paths_resampled_statistics(self, mode):
        return [
            (
                f"{self.setup.folders['statistics']}"
                f"{self.resampled.names[i]}"
                f"."
                f"{mode}"
                f"{self.resampled.extensions[i]}"
            )
            for i in range(self.resampled.n_files)
        ]

    def resampled_statistics(self, mode):
        # Instantiate
        from vircampype.fits.images.sky import SkyImagesResampled

        images = SkyImagesResampled(
            file_paths=self._paths_resampled_statistics(mode=mode), setup=self.setup
        )

        # Consistency check
        if len(images) != len(self.resampled):
            raise PipelineValueError("Image statistics incomplete")

        # Return
        return images

    def _paths_statistics_stacks(self, mode):
        # DEAD CODE: only used by build_statistics_stacks (unused).
        return [p.replace(".fits", f".{mode}.fits") for p in self._paths_stacks]

    def _path_statistics_tile(self, mode):
        return self.setup.path_coadd.replace(".fits", f".{mode}.fits")

    # =========================================================================== #
    # Others
    @property
    def _paths_scamp_headers(self):
        return [
            p.replace(".final.fits", ".final.ahead")
            for p in self._paths_processed_science_final
        ]

    # =========================================================================== #
    # Master calibration
    @pipeline_step("master_bpm", message="MASTER-BPM", guard="flat_lamp_check")
    def build_master_bpm(self):
        """Build master bad-pixel mask from lamp-check flats and darks."""
        self.flat_lamp_check.build_master_bpm(
            darks=self.dark_check, darks_noise=self.dark_lin
        )

    @pipeline_step("master_dark", message="MASTER-DARK", guard="dark_all")
    def build_master_dark(self):
        """Combine dark frames into a master dark."""
        self.dark_all.build_master_dark()

    @pipeline_step("master_gain", message="MASTER-GAIN", guard="flat_lamp_gain")
    def build_master_gain(self):
        """Derive per-detector gain and read-noise tables."""
        self.flat_lamp_gain.build_master_gain(darks=self.dark_gain)

    @pipeline_step(
        "master_linearity", message="MASTER-LINEARITY", guard="flat_lamp_lin"
    )
    def build_master_linearity(self):
        """Derive per-detector non-linearity correction tables."""
        self.flat_lamp_lin.build_master_linearity(darks=self.dark_lin)

    @pipeline_step(
        "master_twilight_flat", message="MASTER-TWILIGHT-FLAT", guard="flat_twilight"
    )
    def build_master_twilight_flat(self):
        """Combine twilight flats into a master flat field."""
        self.flat_twilight.build_master_twilight_flat()

    @pipeline_step(
        "master_weight_global", message="MASTER-WEIGHT-GLOBAL", guard="flat_twilight"
    )
    def build_master_weight_global(self):
        """Build a global weight map from the master twilight flat."""
        self.flat_twilight.build_master_weight_global()

    @pipeline_step("master_source_mask", message="MASTER-SOURCE-MASK")
    def build_master_source_mask(self):
        """Build source masks for sky subtraction from processed images."""
        self.processed_basic_science_and_offset.build_master_source_mask()

    @pipeline_step("master_sky", message="MASTER-SKY")
    def build_master_sky(self):
        """Build master sky frame from science and/or offset images."""
        # If mixed data is requested
        if self.setup.sky_mix_science:
            self.processed_basic_science_and_offset.build_master_sky()

        # If no offset is present and mixing not requested, build from science
        elif self.processed_basic_offset is None:
            self.processed_basic_science.build_master_sky()

        # Otherwise build only from offset
        else:
            self.processed_basic_offset.build_master_sky()

    @pipeline_step(
        "master_photometry", message="MASTER-PHOTOMETRY", guard="raw_science"
    )
    def build_master_photometry(self):
        """Download and store 2MASS photometric reference catalog."""
        self.processed_basic_science.build_master_photometry()

    @pipeline_step("master_astrometry", message="MASTER-ASTROMETRY")
    def build_master_astrometry(self):
        """Download and store Gaia astrometric reference catalog."""
        self.processed_basic_science.build_master_astrometry()

    @pipeline_step("master_weight_image", message="MASTER WEIGHT IMAGE")
    def build_master_weight_image(self):
        """Build per-image weight maps for final processed science data."""
        self.processed_science_final.build_master_weight_image()

    # =========================================================================== #
    # Image processing
    @pipeline_step("processed_raw_basic", message="BASIC RAW PROCESSING")
    def process_raw_basic(self):
        """Apply dark, flat, linearity, and gain corrections to raw frames."""
        self.raw_science_and_offset.process_raw_basic()

    @pipeline_step("processed_raw_final", message="FINAL RAW PROCESSING")
    def process_science_final(self):
        """Apply sky subtraction and final corrections to science images."""
        self.processed_basic_science.process_raw_final()

    @pipeline_step("astrometry", message="ASTROMETRY")
    def calibrate_astrometry(self):
        """Run SExtractor and SCAMP for astrometric calibration."""
        self.processed_science_final.sextractor(preset="scamp")
        if self.setup.external_headers:
            nehdr = sum([os.path.isfile(p) for p in self._paths_scamp_headers])
            nfproc = sum(
                [os.path.isfile(p) for p in self._paths_processed_science_final]
            )
            if (nehdr != nfproc) or (nehdr == 0):
                raise PipelineValueError(
                    f"Not enough external headers present ({nehdr}/{nfproc})",
                    logger=self.log,
                )
        else:
            self.sources_processed_final_scamp.scamp()

    @pipeline_step("illumcorr", message="ILLUMINATION CORRECTION")
    def illumination_correction(self):
        """Derive and apply photometric illumination correction."""
        if not self.status.astrometry:
            raise PipelineValueError(
                "Astrometric calibration has to be "
                "completed before illumination correction",
                logger=self.log,
            )
        self.processed_science_final.sextractor(preset="ic")
        self.sources_processed_illumcorr.build_master_illumination_correction()
        self.processed_science_final.apply_illumination_correction()

    @pipeline_step("tile_header", message="TILE HEADER", guard="raw_science")
    def build_coadd_header(self):
        """Build the WCS header for the coadded tile."""
        self.processed_basic_science.build_coadd_header()

    @pipeline_step("resampled", message="PAWPRINT RESAMPLING")
    def resample(self):
        """Resample illumination-corrected images onto the tile grid via SWarp."""
        self.illumination_corrected.resample()

    # =========================================================================== #
    # Image assembly
    @pipeline_step("stacks", message="STACKS")
    def build_stacks(self):
        """Coadd resampled images per offset position into stacks."""
        self.resampled.build_stacks()

    @pipeline_step("pawprints", message="PAWPRINTS")
    def build_pawprints(self):
        """Single-exposure pawprint products from the resampled pawprints.

        One SCIENCE.MEFIMAGE + weight per raw exposure, with PHOTZP, PHOTZPER, PHOTSYS,
        ABMAGLIM, ABMAGSAT, PSF_FWHM and ELLIPTIC in every detector extension and
        FLUXCAL in the primary from the pawprint catalogs.
        """
        self.resampled.build_pawprints()
        self.sources_resampled_cal.write_photometric_cards(self.pawprints)

    @pipeline_step("tile", message="TILE")
    def build_tile(self):
        """Coadd all resampled images into a single deep tile."""
        self.resampled.build_tile()

    @pipeline_step("exposure_map", message="EXPOSURE MAP")
    def exposure_map_tile(self):
        """Exact exposure map of the tile; EXPTIME, EXPTMETH and SKYSQDEG in its primary header."""
        from astropy.io import fits

        from vircampype.tools.expmap import build_exposure_map
        from vircampype.tools.productmeta import set_cards

        tile = self.tile
        header = tile.headers_data[0][0]
        primary = fits.Header(
            {key: tile.headers_primary[0][key] for key in ("OBJECT", "FILTER")}
        )
        resampled = self.resampled
        exptime, area = build_exposure_map(
            paths_weights=resampled.get_master_weight_global().paths_full,
            exptimes=resampled.texptime,
            header=header,
            primary=primary,
            path_out=self.setup.path_coadd_expmap,
            cache_dir=self.setup.local_cache_dir,
        )
        set_cards(
            self.setup.path_coadd,
            [
                (
                    "EXPTIME",
                    float(exptime),
                    "Median exposure time of covered pixels (s)",
                ),
                ("EXPTMETH", "EXPMAP", "EXPTIME is the median of the exposure map"),
                ("SKYSQDEG", round(area, 6), "Covered sky area (deg2)"),
            ],
        )
        self.log.info(
            f"Exposure map: EXPTIME={exptime:.1f} s, SKYSQDEG={area:.6f} deg2, "
            f"written to {self.setup.path_coadd_expmap}"
        )

    # =========================================================================== #
    # Statistics
    @pipeline_step("statistics_resampled", message="IMAGE STATISTICS")
    def build_statistics_resampled(self):
        """Compute per-pawprint statistics images (MJD, nimg, exptime)."""
        self.resampled.build_statistics()

    @pipeline_step("statistics_stacks", message="STACKS STATISTICS")
    def build_statistics_stacks(self):
        """DEAD CODE: not called from process_science; stack statistics are unused.

        Coadd statistics images for stacks and combine MJD maps.
        """
        # Check MJDREF before coadding, so a mixed set fails before writing
        mjd_ref = self._statistics_mjd_ref()
        for mode in ["mjd.int", "mjd.frac", "nimg", "exptime"]:
            images = self.resampled_statistics(mode=mode)
            images.coadd_statistics_stacks(mode=mode)
        # Combine MJD data
        for pmi, pmf, pmc in zip(
            self._paths_statistics_stacks(mode="mjd.int"),
            self._paths_statistics_stacks(mode="mjd.frac"),
            self._paths_statistics_stacks(mode="mjd.eff"),
        ):
            combine_mjd_images(
                path_file_a=pmi,
                path_file_b=pmf,
                path_file_out=pmc,
                offset=mjd_ref,
                overwrite=True,
            )

    def _statistics_mjd_ref(self) -> float:
        """Reference MJD encoded in the per-pawprint mjd.int statistics images.

        Read back from the files actually written so the decoder follows the
        encoder; a set built with different references (or without MJDREF) fails.
        """
        try:
            refs = self.resampled_statistics(mode="mjd.int").read_from_prime_headers(
                keywords=["MJDREF"]
            )[0]
        except KeyError as exc:
            raise PipelineValueError(
                "MJDREF missing in the statistics images; rebuild them "
                "(remove the statistics folder or set overwrite)"
            ) from exc
        refs = set(float(r) for r in refs)
        if len(refs) != 1 or not math.isfinite(next(iter(refs))):
            raise PipelineValueError(
                f"Inconsistent MJDREF in statistics images: {sorted(refs)}"
            )
        return refs.pop()

    def _astrms_flat_scalar(self) -> tuple[float, float, float]:
        """Conservative flat-SCAMP astrometric floor fallback (mas, mas, corr).

        Reads the high-S/N external SCAMP RMS ``ASTRMSH1/2`` -- a per-field *group*
        scalar that ``scamp()`` writes into the ``.ahead`` and resampling propagates
        into the resampled-pawprint headers -- and returns
        ``(3.6e6 * max(ASTRMSH1), 3.6e6 * max(ASTRMSH2), 0.0)``. The max over
        contributing pawprints reproduces the old constant per-pawprint astrms map
        for a single-field tile and stays conservative across a mosaic's
        heterogeneous SCAMP groups; the correlation fallback is 0 (diagonal-only).
        This is only the fallback/shrink target -- the self-calibration overwrites
        it wherever it has enough Gaia stars. Fail-loud if ``ASTRMSH1/2`` is absent
        (solution predates the high-S/N floor, or the ``.ahead`` cache is stale).
        """
        import math

        try:
            a1, a2 = self.resampled.read_from_data_headers(["ASTRMSH1", "ASTRMSH2"])
        except KeyError as e:
            raise PipelineValueError(
                f"High-S/N external astrometric RMS {e} missing from the resampled "
                "pawprint headers; refusing to fall back to an inflated floor. "
                "Re-run SCAMP or clear the stale .ahead cache so ASTRMSH1/2 is written.",
                logger=self.log,
            ) from e
        v1 = [v for f in a1 for v in f if v is not None and math.isfinite(v)]
        v2 = [v for f in a2 for v in f if v is not None and math.isfinite(v)]
        if not v1 or not v2 or max(v1) <= 0 or max(v2) <= 0:
            raise PipelineValueError(
                "High-S/N external astrometric RMS (ASTRMSH1/2) is non-finite or "
                "non-positive in the resampled-pawprint headers; re-run SCAMP.",
                logger=self.log,
            )
        return 3_600_000 * max(v1), 3_600_000 * max(v2), 0.0

    def _coadd_statistics_tile_numpy(self, mjd_ref: float) -> None:
        """Tile statistics maps without SWarp (same output, see tools/statscoadd.py)."""
        import vircampype
        from vircampype.tools.astromatic import SwarpSetup
        from vircampype.tools.messaging import print_header, print_message
        from vircampype.tools.statscoadd import (
            STATISTICS_MODES,
            coadd_statistics_maps,
            statistics_output_paths,
        )

        print_header(header="TILE STATISTICS", silent=self.setup.silent)
        log = logging.getLogger(__name__)
        tstart = time.time()
        paths_final = statistics_output_paths(self.setup.path_coadd)
        if not self.setup.overwrite and all(
            os.path.isfile(p) for p in paths_final.values()
        ):
            log.info("Tile statistics already exist, skipping")
            return

        images = {m: self.resampled_statistics(mode=m) for m in STATISTICS_MODES}
        paths_values = {m: images[m].paths_full for m in STATISTICS_MODES}
        paths_weight = [p.replace(".nimg.", ".weight.") for p in paths_values["nimg"]]
        if not all(os.path.isfile(p) for p in paths_weight):
            raise PipelineValueError("Not all statistics images have weights")
        swarp_preset = read_yml(SwarpSetup(setup=self.setup).preset_coadd)

        # Write to local temp, then move into place
        tmpdir = make_system_tempdir(base_dir=self.setup.local_cache_dir)
        try:
            paths_tmp = {
                k: os.path.join(tmpdir, os.path.basename(p))
                for k, p in paths_final.items()
            }
            coadd_statistics_maps(
                paths_weight=paths_weight,
                paths_values=paths_values,
                paths_out=paths_tmp,
                mjd_ref=mjd_ref,
                swarp_preset=swarp_preset,
                software=("vircampype", vircampype.__version__),
                mem_max=self.setup.statistics_coadd_mem_max,
                n_threads=self.setup.n_jobs,
            )
            try:
                for k, p in paths_final.items():
                    move_file(paths_tmp[k], p)
            except Exception:
                # All nine or none: a partial set would be skipped on rerun
                for p in paths_final.values():
                    if os.path.isfile(p):
                        os.remove(p)
                raise
        finally:
            remove_directory(tmpdir)
        log.info(f"Written tile statistics: {list(paths_final.values())}")
        print_message(
            message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
            kind="okblue",
            end="\n",
            logger=log,
        )

    @pipeline_step("statistics_tile", message="TILE STATISTICS")
    def build_statistics_tile(self):
        """Coadd statistics images for the tile and build statistics tables."""
        # Check MJDREF before coadding, so a mixed set fails before writing
        mjd_ref = self._statistics_mjd_ref()
        use_swarp = self.setup.statistics_coadd == "swarp"
        if not use_swarp:
            from vircampype.tools.statscoadd import StatisticsCoaddUnsupported

            try:
                self._coadd_statistics_tile_numpy(mjd_ref=mjd_ref)
            except StatisticsCoaddUnsupported as e:
                logging.getLogger(__name__).warning(
                    f"numpy statistics coadd not applicable ({e}); using SWarp"
                )
                use_swarp = True
        if use_swarp:
            for mode in ["mjd.int", "mjd.frac", "nimg", "exptime"]:
                images = self.resampled_statistics(mode=mode)
                images.coadd_statistics_tile(mode=mode)
            # Combine MJD data
            combine_mjd_images(
                path_file_a=self._path_statistics_tile(mode="mjd.int"),
                path_file_b=self._path_statistics_tile(mode="mjd.frac"),
                path_file_out=self._path_statistics_tile(mode="mjd.eff"),
                offset=mjd_ref,
                overwrite=True,
            )

        self.sources_tile_cal.build_statistics_tables(
            astrms_fallback=self._astrms_flat_scalar()
        )

    # =========================================================================== #
    # Classification
    @pipeline_step("classification_stacks", message="STACKS CLASSIFICATION")
    def classification_stacks(self):
        """Run star/galaxy classification on stack catalogs."""
        self.stacks.build_class_star_library()

    @pipeline_step("classification_tile", message="TILE CLASSIFICATION")
    def classification_tile(self):
        """Run star/galaxy classification on the tile catalog."""
        self.tile.build_class_star_library()

    # =========================================================================== #
    # Photometry
    @pipeline_step("photometry_pawprints", message="PAWPRINT PHOTOMETRY")
    def photometry_pawprints(self):
        """Extract and photometrically calibrate pawprint source catalogs."""
        self.resampled.sextractor(preset="full")
        self.sources_resampled_full.calibrate_photometry()

    @pipeline_step("photometry_stacks", message="STACKS PHOTOMETRY")
    def photometry_stacks(self):
        """Extract and photometrically calibrate stack source catalogs.

        Writes PHOTZP, PHOTZPER, PHOTSYS, ABMAGLIM, ABMAGSAT, PSF_FWHM and ELLIPTIC
        into every stack extension and FLUXCAL into the stack primaries.
        """
        self.stacks.sextractor(preset="full")
        self.sources_stacks_full.calibrate_photometry()
        self.sources_stacks_cal.write_photometric_cards(self.stacks)

    @pipeline_step("photometry_tile", message="TILE PHOTOMETRY")
    def photometry_tile(self):
        """Extract and photometrically calibrate the tile source catalog.

        With ``sex_winpos_cap`` the WIN window is capped at the local stellar size
        (``WINPOS_SIGMAP``); the catalogue records the map (``WINPSMAP``). An existing
        catalogue made with the other setting raises: remove it (and the products
        derived from it) or set overwrite. Writes PHOTZP, PHOTZPER, PHOTSYS, ABMAGLIM,
        ABMAGSAT, PSF_FWHM and ELLIPTIC into the tile SCI and FLUXCAL and ELLIPTIC into
        its primary.
        """
        from astropy.io import fits

        path_cat = self.tile.paths_source_tables(preset="full")[0]
        reuse = os.path.isfile(path_cat) and not self.setup.overwrite
        if reuse:
            capped = "WINPSMAP" in fits.getheader(path_cat, 0)
            if capped != self.setup.sex_winpos_cap:
                raise PipelineValueError(
                    f"{path_cat} was built {'with' if capped else 'without'} the window "
                    f"cap, sex_winpos_cap is {self.setup.sex_winpos_cap}; remove it (and "
                    "the products derived from it) or set overwrite",
                    logger=self.log,
                )
        if self.setup.sex_winpos_cap and not reuse:
            path_map = self._build_winsig_map()
            self.tile.sextractor(preset="full", winpos_sigmap=[path_map])
            with fits.open(path_cat, mode="update") as hdul:
                hdul[0].header["WINPSMAP"] = (
                    os.path.basename(path_map),
                    "WIN window capped at the local stellar size",
                )
        else:
            self.tile.sextractor(preset="full")
        self.sources_tile_full.calibrate_photometry()
        self.sources_tile_cal.write_photometric_cards(self.tile, elliptic_primary=True)

    def _build_winsig_map(self) -> str:
        """SExtractor map of the local stellar window sigma for the tile (WIN cap).

        Coadds the per-detector ``WINVAR`` of the per-pawprint statistics weights (added
        where missing) and writes the ``WINPOS_SIGMAP`` file next to the coadd.
        """
        from vircampype.tools.statscoadd import (
            StatisticsCoaddUnsupported,
            coadd_window_sigma_map,
        )
        from vircampype.tools.winposmap import write_winpos_sigmap

        path_sig = self.setup.path_coadd.replace(".fits", ".winsig.fits")
        path_map = self.setup.path_coadd.replace(".fits", ".winsig.map.fits")
        paths_weight = self._paths_resampled_statistics(mode="weight")
        self.resampled.write_window_variances()  # skips files that have it
        try:
            n_valid = coadd_window_sigma_map(
                paths_weight=paths_weight,
                path_out=path_sig,
                mem_max=self.setup.statistics_coadd_mem_max,
            )
        except StatisticsCoaddUnsupported as e:
            raise PipelineValueError(
                f"Window cap needs numpy-coaddable statistics: {e}", logger=self.log
            ) from e
        if n_valid == 0:
            raise PipelineValueError(
                "Window cap: no stellar window size for any detector", logger=self.log
            )
        write_winpos_sigmap(
            path_sigma_arcsec=path_sig,
            header_image=self.tile.headers_data[0][0],
            path_out=path_map,
        )
        self.log.info(f"Window-cap map written: {path_map}")
        return path_map

    @pipeline_step("photerr_internal", message="INTERNAL PHOTOMETRIC ERROR")
    def photerr_internal(self):
        """Estimate internal photometric errors from pawprint overlaps."""
        self.sources_resampled_cal.photerr_internal()
        if self.setup.qc_plots:
            self.sources_resampled_cal.plot_qc_photerr_internal()

    # =========================================================================== #
    # QC
    @pipeline_step("qc_photometry_stacks", message="STACKS QC PHOTOMETRY")
    def qc_photometry_stacks(self):
        """Generate QC plots for stack photometric calibration."""
        self.sources_stacks_cal.plot_qc_phot_zp(axis_size=5)
        self.sources_stacks_cal.plot_qc_phot_ref1d(axis_size=5)
        self.sources_stacks_cal.plot_qc_phot_ref2d(axis_size=5)

    @pipeline_step("qc_photometry_tile", message="TILE QC PHOTOMETRY")
    def qc_photometry_tile(self):
        """Generate QC plots for tile photometric calibration."""
        self.sources_tile_cal.plot_qc_phot_zp(axis_size=5)
        self.sources_tile_cal.plot_qc_phot_ref1d(axis_size=5)
        self.sources_tile_cal.plot_qc_phot_ref2d(axis_size=5)

    @pipeline_step("qc_astrometry_stacks", message="STACKS QC ASTROMETRY")
    def qc_astrometry_stacks(self):
        """Generate QC plots for stack astrometric residuals and write CRDER1/2."""
        self.sources_stacks_cal.plot_qc_astrometry_1d()
        self.sources_stacks_cal.plot_qc_astrometry_2d()
        self.sources_stacks_cal.write_crder(self.stacks)

    @pipeline_step("qc_astrometry_pawprints", message="PAWPRINTS QC ASTROMETRY")
    def qc_astrometry_pawprints(self):
        """Write CRDER1/2 into every pawprint product detector."""
        self.sources_resampled_cal.write_crder(self.pawprints)

    @pipeline_step("qc_psf_stacks", message="STACKS QC PSF")
    def qc_psf_stacks(self):
        """Generate QC plots for stack PSF FWHM."""
        if self.setup.qc_plots:
            self.sources_stacks_cal.plot_qc_psf_2d()
            self.sources_stacks_cal.plot_qc_psf_1d()

    @pipeline_step("qc_astrometry_tile", message="TILE QC ASTROMETRY")
    def qc_astrometry_tile(self):
        """Generate QC plots for tile astrometric residuals and write CRDER1/2."""
        self.sources_tile_cal.plot_qc_astrometry_1d()
        self.sources_tile_cal.plot_qc_astrometry_2d()
        self.sources_tile_cal.write_crder(self.tile)

    @pipeline_step("qc_psf_tile", message="TILE QC PSF")
    def qc_psf_tile(self):
        """Generate QC plots for tile PSF FWHM."""
        if self.setup.qc_plots:
            self.sources_tile_cal.plot_qc_psf_2d()
            self.sources_tile_cal.plot_qc_psf_1d()

    @pipeline_step("qc_completeness_tile", message="TILE QC COMPLETENESS")
    def qc_completeness_tile(self):
        """Run completeness analysis on the tile and generate QC plots."""
        qc_completeness_tile(
            image_path=self.tile.paths_full[0],
            setup=self.setup,
        )

    # =========================================================================== #
    # QC summary
    def build_qc_summary(self):
        """Build the QC summary table from stacks, tile and pawprints."""

        if self.status.qc_summary:
            # Routine checkpoint skip: INFO + console line, not a warning.
            logging.getLogger(__name__).info(
                "Skipping QC SUMMARY TABLE: already complete"
            )
            if not self.setup.silent:
                print_stage_skip("QC SUMMARY TABLE")
            return

        stacks = tile = pawprints = None
        if self.setup.build_stacks:
            try:
                stacks = self.stacks
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"QC summary: stacks unavailable ({e}); section skipped"
                )
        if self.setup.build_tile:
            try:
                tile = self.tile
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"QC summary: tile unavailable ({e}); section skipped"
                )
        if self.setup.build_pawprints:
            try:
                pawprints = self.pawprints
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"QC summary: pawprints unavailable ({e}); section skipped"
                )

        result = build_qc_summary(
            setup=self.setup, stacks=stacks, tile=tile, pawprints=pawprints
        )
        if result is not None:
            self.update_status(qc_summary=True)

    def finalize_products(self):
        """Check and seal every Phase 3 product (PRODCATG, ASSONi, checksums).

        Not a status step: it is idempotent and runs at the end of every run, after the
        last header edit.
        """
        from vircampype.tools.productmeta import finalize_products

        logging.getLogger(__name__).info("Finalizing Phase 3 products")
        finalize_products(self)

    # =========================================================================== #
    @pipeline_step("public_catalog", message="PUBLIC CATALOG")
    def build_public_catalog(self):
        """Build the public release source catalog from the tile."""
        self.sources_tile_cal.build_public_catalog(
            photerr_internal=self.setup.photometric_error_floor,
        )

    # =========================================================================== #
    # Cleaning/archiving
    def shallow_clean(self):
        """Remove intermediate processing steps,
        but leave the most important files in the directory tree."""
        log = logging.getLogger(__name__)
        targets = [
            (self.setup.folders["illumcorr"], "*"),
            (self.setup.folders["processed_basic"], "*"),
            (self.setup.folders["processed_final"], "*.fits"),
            (self.setup.folders["processed_final"], "*.fits.tab"),
            (self.setup.folders["processed_final"], "*.ahead"),
            (self.setup.folders["resampled"], "*"),
            (self.setup.folders["statistics"], "*"),
        ]
        for path, pattern in targets:
            n_removed = clean_directory(path, pattern=pattern)
            log.info(f"Cleaned {path} (pattern '{pattern}'): {n_removed} files")

    def deep_clean(self):
        """Runs a shallow clean followed by deleting the pipeline status"""
        self.shallow_clean()
        # WARNING so the removal of the status (and the run logs in temp,
        # including the live one) is visible on the console and on record.
        logging.getLogger(__name__).warning(
            f"Deep clean: removing pipeline status and logs from "
            f"{self.setup.folders['temp']}"
        )
        clean_directory(self.setup.folders["temp"])

    @pipeline_step("archive", message="ARCHIVING")
    def archive(self):
        """Run a shallow clean and compress all remaining FITS images.

        Delivered products (primary header with PRODCATG) are never compressed.
        """
        from astropy.io import fits

        self.shallow_clean()

        # Find all remaining FITS files (recursive), except the products
        fits_files = sorted(
            glob.glob(self.setup.folders["object"] + "/**/*.fits", recursive=True)
        )
        n_all = len(fits_files)
        fits_files = [f for f in fits_files if "PRODCATG" not in fits.getheader(f, 0)]
        logging.getLogger(__name__).info(
            f"Archive: {n_all - len(fits_files)} product files left uncompressed"
        )

        if fits_files:
            n_jobs = min(self.setup.n_jobs, 2)
            compress_images(
                fits_files,
                q=self.setup.fpack_quantization_factor,
                n_jobs=n_jobs,
            )

    def unarchive(self):
        """Uncompress all compressed FITS files and reset archive status."""

        # Find all compressed files (recursive)
        fz_files = sorted(
            glob.glob(self.setup.folders["object"] + "/**/*.fz", recursive=True)
        )

        if fz_files:
            cmds = [f"funpack -F {f}" for f in fz_files]
            n_jobs = min(self.setup.n_jobs, 2)
            run_commands_shell_parallel(
                cmds=cmds, n_jobs=n_jobs, silent=True, label="Decompressing"
            )

        self.update_status(archive=False)

    # =========================================================================== #
    # Notifications
    def _notify_pushover(self, message: str) -> None:
        """Send a Pushover notification if credentials are configured."""
        if self.setup.pushover_user_key and self.setup.pushover_api_token:
            notify_pushover(
                title=f"vircampype: {self.setup.name}",
                message=message,
                user_key=self.setup.pushover_user_key,
                api_token=self.setup.pushover_api_token,
            )

    # =========================================================================== #
    # Pipeline processors
    @_notify_on_completion
    def process_calibration(self):
        """Sequentially build master calibration files."""

        # Master bad pixel mask
        self.build_master_bpm()

        # Master non-linearity tables
        self.build_master_linearity()

        # Master darks
        self.build_master_dark()

        # Master gain and read noise tables
        self.build_master_gain()

        # Master flat
        self.build_master_twilight_flat()

        # Global master weight
        self.build_master_weight_global()

    @_notify_on_completion
    def process_science(self):
        """Sequentially process science data."""

        # Basic processing
        self.process_raw_basic()

        # Build static sky, source masks, dynamic sky, and master photometry
        self.build_master_photometry()
        self.build_master_astrometry()
        self.build_master_source_mask()
        self.build_master_sky()

        # Final science data and weight maps
        self.process_science_final()

        # Build final weights
        self.build_master_weight_image()

        # Build coadd header and run scamp
        self.build_coadd_header()
        self.calibrate_astrometry()

        # Illumination correction, resampling, and image stats
        self.illumination_correction()
        self.resample()

        # Build statistics (required for mosaics/deep stacks, not for phase 3)
        self.build_statistics_resampled()

        # Single-exposure pawprint products: catalogs, files with their photometric cards,
        # CRDER
        if self.setup.build_pawprints:
            self.photometry_pawprints()
            self.build_pawprints()
            self.qc_astrometry_pawprints()
        # self.photerr_internal()

        # Build and calibrate stacks
        if self.setup.build_stacks:
            self.build_stacks()
            self.photometry_stacks()
            self.qc_photometry_stacks()
            self.qc_astrometry_stacks()
            self.qc_psf_stacks()

        # Build and calibrate tile, then its source list (photometry writes PHOTZP into
        # the SCI header before the completeness run reads it)
        if self.setup.build_tile:
            self.build_tile()
            self.exposure_map_tile()
            self.photometry_tile()
            self.qc_photometry_tile()
            self.qc_astrometry_tile()
            self.qc_psf_tile()
            if self.setup.build_completeness:
                self.qc_completeness_tile()
            self.build_statistics_tile()
            self.classification_tile()
            self.build_public_catalog()

        # Build QC summary table
        self.build_qc_summary()

        # Phase 3 check and identity cards of every product, after the last header edit
        self.finalize_products()

        if self.setup.archive:
            self.archive()
