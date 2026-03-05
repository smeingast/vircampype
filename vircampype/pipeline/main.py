import functools
import glob
import json
import os.path
import sys
import time

from astropy.table import Table

from vircampype.fits.images.common import FitsImages
from vircampype.pipeline.errors import PipelineValueError
from vircampype.pipeline.log import PipelineLog
from vircampype.pipeline.setup import Setup
from vircampype.pipeline.status import PipelineStatus
from vircampype.tools.esotools import build_phase3_stacks, build_phase3_tile
from vircampype.tools.fitstools import (
    build_qc_summary_row,
    combine_mjd_images,
    compress_images,
)
from vircampype.tools.messaging import (
    print_end,
    print_header,
    print_message,
    print_start,
)
from vircampype.tools.systemtools import *


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
            if guard is not None and getattr(self, guard) is None:
                return
            if getattr(self.status, status_attr):
                print_message(
                    message=f"{message} already done", kind="warning", end=None
                )
                return
            method(self, *args, **kwargs)
            self.update_status(**{status_attr: True})

        return wrapper

    return decorator


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
        self.log = PipelineLog(setup=self.setup)

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
        if len(images) != 6:
            raise PipelineValueError("Stacks incomplete")

        return images

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
        self.flat_lamp_check.build_master_bpm(darks=self.dark_check)

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
                    f"Not enough external headers present ({nehdr}/{nfproc})"
                )
        else:
            self.sources_processed_final_scamp.scamp()

    @pipeline_step("illumcorr", message="ILLUMINATION CORRECTION")
    def illumination_correction(self):
        """Derive and apply photometric illumination correction."""
        if not self.status.astrometry:
            raise ValueError(
                "Astrometric calibration has to be "
                "completed before illumination correction"
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

    @pipeline_step("tile", message="TILE")
    def build_tile(self):
        """Coadd all resampled images into a single deep tile."""
        self.resampled.build_tile()

    # =========================================================================== #
    # Statistics
    @pipeline_step("statistics_resampled", message="IMAGE STATISTICS")
    def build_statistics_resampled(self):
        """Compute per-pawprint statistics images (MJD, nimg, exptime)."""
        self.resampled.build_statistics()

    @pipeline_step("statistics_stacks", message="STACKS STATISTICS")
    def build_statistics_stacks(self):
        """Coadd statistics images for stacks and combine MJD maps."""
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
                path_file_a=pmi, path_file_b=pmf, path_file_out=pmc, overwrite=True
            )

    @pipeline_step("statistics_tile", message="TILE STATISTICS")
    def build_statistics_tile(self):
        """Coadd statistics images for the tile and build statistics tables."""
        for mode in [
            "mjd.int",
            "mjd.frac",
            "nimg",
            "exptime",
            "astrms1",
            "astrms2",
        ]:
            images = self.resampled_statistics(mode=mode)
            images.coadd_statistics_tile(mode=mode)
        # Combine MJD data
        combine_mjd_images(
            path_file_a=self._path_statistics_tile(mode="mjd.int"),
            path_file_b=self._path_statistics_tile(mode="mjd.frac"),
            path_file_out=self._path_statistics_tile(mode="mjd.eff"),
            overwrite=True,
        )

        self.sources_tile_cal.build_statistics_tables()

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
        """Extract and photometrically calibrate stack source catalogs."""
        self.stacks.sextractor(preset="full")
        self.sources_stacks_full.calibrate_photometry()

    @pipeline_step("photometry_tile", message="TILE PHOTOMETRY")
    def photometry_tile(self):
        """Extract and photometrically calibrate the tile source catalog."""
        self.tile.sextractor(preset="full")
        self.sources_tile_full.calibrate_photometry()

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
        """Generate QC plots for stack astrometric residuals."""
        self.sources_stacks_cal.plot_qc_astrometry_1d()
        self.sources_stacks_cal.plot_qc_astrometry_2d()

    @pipeline_step("qc_astrometry_tile", message="TILE QC ASTROMETRY")
    def qc_astrometry_tile(self):
        """Generate QC plots for tile astrometric residuals."""
        self.sources_tile_cal.plot_qc_astrometry_1d()
        self.sources_tile_cal.plot_qc_astrometry_2d()

    # =========================================================================== #
    # QC summary
    def build_qc_summary(self):
        """Build a QC summary table aggregating key metrics from stacks and tile."""

        if self.status.qc_summary:
            print_message(
                message="QC SUMMARY TABLE already done", kind="warning", end=None
            )
            return

        print_header(
            header="QC SUMMARY TABLE",
            silent=self.setup.silent,
            left=None,
            right=None,
        )
        log = PipelineLog()
        tstart = time.time()

        kw = {
            "filter_keyword": self.setup.keywords.filter_name,
            "mag_saturation": self.setup.reference_mag_lo,
        }
        rows = []

        # Collect rows from stacks
        if self.setup.build_stacks:
            try:
                stacks = self.stacks
                catalogs = self.sources_stacks_cal
                for idx in range(len(stacks)):
                    try:
                        rows.append(
                            build_qc_summary_row(
                                image_path=stacks.paths_full[idx],
                                catalog_path=catalogs.paths_full[idx],
                                product_type="stack",
                                **kw,
                            )
                        )
                    except Exception as e:
                        log.warning(f"QC summary: skipping stack {idx}: {e}")
            except Exception as e:
                log.warning(f"QC summary: cannot read stacks: {e}")

        # Collect row from tile
        if self.setup.build_tile:
            try:
                tile = self.tile
                catalog = self.sources_tile_cal
                try:
                    rows.append(
                        build_qc_summary_row(
                            image_path=tile.paths_full[0],
                            catalog_path=catalog.paths_full[0],
                            product_type="tile",
                            **kw,
                        )
                    )
                except Exception as e:
                    log.warning(f"QC summary: skipping tile: {e}")
            except Exception as e:
                log.warning(f"QC summary: cannot read tile: {e}")

        if not rows:
            print_message(
                message="No products available for QC summary",
                kind="warning",
                end=None,
            )
            return

        # Build and write table
        qc_table = Table(rows=rows)
        path_out = os.path.join(self.setup.folders["qc"], "qc_summary.ecsv")
        qc_table.write(path_out, format="ascii.ecsv", overwrite=True)

        print_message(
            message=f"\n-> QC summary written to {os.path.basename(path_out)} "
            f"({len(rows)} products, {time.time() - tstart:.1f}s)",
            kind="okblue",
            end="\n",
            logger=log,
        )

        self.update_status(qc_summary=True)

    # =========================================================================== #
    # Phase 3
    @property
    def _paths_phase3(self):
        return sorted(glob.glob(f"{self.setup.folders['phase3']}*.fits"))

    @property
    def _paths_phase3_images(self):
        return [x for x in self._paths_phase3 if "sources" not in x]

    @property
    def _paths_phase3_catalogs(self):
        return [x for x in self._paths_phase3 if "sources" in x]

    def compress_phase3_images(self):
        # Maximum of three parallel jobs
        n_jobs = 3 if self.setup.n_jobs > 3 else self.setup.n_jobs

        # Compress files
        compress_images(
            self._paths_phase3_images,
            q=self.setup.fpack_quantization_factor,
            n_jobs=n_jobs,
        )

    @pipeline_step("phase3", message="PHASE 3")
    def build_phase3(self):
        """Build ESO Phase 3 compliant image and catalog products."""
        if self.setup.build_stacks:
            build_phase3_stacks(
                stacks_images=self.stacks,
                stacks_catalogs=self.sources_stacks_cal,
                mag_saturation=self.setup.reference_mag_lo,
            )
        if self.setup.build_tile:
            build_phase3_tile(
                tile_image=self.tile,
                tile_catalog=self.sources_tile_cal,
                mag_saturation=self.setup.reference_mag_lo,
                pawprint_images=self.resampled,
            )

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

        # Clean illumination correction directory
        clean_directory(self.setup.folders["illumcorr"])

        # Clean basic processed directory
        clean_directory(self.setup.folders["processed_basic"])

        # Clean final processed directory
        clean_directory(self.setup.folders["processed_final"], pattern="*.fits")
        clean_directory(self.setup.folders["processed_final"], pattern="*.fits.tab")
        clean_directory(self.setup.folders["processed_final"], pattern="*.ahead")

        # Clean resampled pawprint images and catalogs
        clean_directory(self.setup.folders["resampled"])

        # Clean per-pawprint statistics images
        clean_directory(self.setup.folders["statistics"])

    def deep_clean(self):
        """Runs a shallow clean followed by deleting the pipeline status"""
        self.shallow_clean()
        clean_directory(self.setup.folders["temp"])

    @pipeline_step("archive", message="ARCHIVING")
    def archive(self):
        """Run a shallow clean and compress all remaining FITS images."""
        self.shallow_clean()

        # Find all remaining FITS files (recursive)
        fits_files = sorted(
            glob.glob(self.setup.folders["object"] + "/**/*.fits", recursive=True)
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
            run_commands_shell_parallel(cmds=cmds, n_jobs=n_jobs, silent=True)

        self.update_status(archive=False)

    # =========================================================================== #
    # Pipeline processors
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

    def process_science(self):
        """Sequentially process science data."""

        # Start time
        t0 = print_start(obj=self.setup.name)

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

        # Calibrate pawprints and determine internal photometric error
        if self.setup.calibrate_pawprints:
            self.photometry_pawprints()
        # self.photerr_internal()

        # Build and calibrate stacks
        if self.setup.build_stacks:
            self.build_stacks()
            self.photometry_stacks()
            self.qc_photometry_stacks()
            self.qc_astrometry_stacks()

        # Build and calibrate tile
        if self.setup.build_tile:
            self.build_tile()
            if self.setup.build_tile_only:
                print_end(tstart=t0)
                sys.exit()
            self.photometry_tile()
            self.qc_photometry_tile()
            self.qc_astrometry_tile()

        # Build QC summary table
        self.build_qc_summary()

        # Phase 3
        if self.setup.build_phase3:
            self.build_phase3()

        if self.setup.build_public_catalog:
            self.build_statistics_tile()
            if self.setup.source_classification:
                self.classification_tile()
            self.build_public_catalog()

        if self.setup.archive:
            self.archive()

        # Print finish message
        print_end(tstart=t0)
