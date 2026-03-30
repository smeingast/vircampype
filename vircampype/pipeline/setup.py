import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from astropy.io import fits
from joblib import cpu_count
from regions import Regions

from vircampype.miscellaneous.projection import Projection
from vircampype.miscellaneous.sourcemasks import *
from vircampype.pipeline.errors import PipelineValueError
from vircampype.tools.systemtools import *


@dataclass
class Setup:
    """Pipeline configuration dataclass loaded from a YAML file.

    Holds ~150 parameters controlling paths, processing flags, instrument
    settings, calibration thresholds, and external tool options.  On
    initialisation the output folder tree is created automatically.
    """

    # Pipeline setup
    name: str = None  # Name of the pipeline setup / target field
    path_data: str = None  # Path to directory containing raw FITS files
    pattern_data: str = "*.fits"  # Glob pattern for raw data file selection
    path_pype: str = None  # Root output directory for pipeline products
    path_master_common: str = None  # Path to shared master calibration files
    path_master_object: str | None = None  # Path to object-specific calibrations
    n_jobs: int = -2  # Parallel jobs: 0=all cores, -1=all-1, -2=all-2, >0=exact
    n_jobs_basic: int = 2  # Parallel jobs for basic processing steps
    n_jobs_sex: int = 5  # Parallel jobs for SExtractor runs
    n_jobs_scamp: int | None = None  # Parallel jobs for SCAMP (None = n_jobs)
    n_jobs_swarp: int | None = None  # Parallel jobs for SWarp (None = n_jobs)
    silent: bool = False  # Suppress terminal progress messages
    overwrite: bool = False  # Overwrite existing output files
    qc_plots: bool = True  # Generate quality control plots
    qc_plot_dpi: int = 300  # DPI for rasterized elements in QC plots
    log_level: str = "info"  # Logging verbosity (debug, info, warning, error)

    # What to process
    calibrate_pawprints: bool = False  # Run photometric calibration on pawprints
    build_stacks: bool = False  # Build stacked images per offset position
    build_tile: bool = True  # Build the final coadded tile
    build_tile_only: bool = False  # Skip pawprint processing, build tile only
    build_phase3: bool = False  # Create ESO Phase 3 compliant products
    build_public_catalog: bool = False  # Build public source catalog
    source_classification: bool = False  # Run star/galaxy classification
    archive: bool = False  # Archive processed data

    # Phase 3
    compress_phase3: bool = True  # Fpack-compress Phase 3 images
    fpack_quantization_factor: int = (
        16  # Fpack quantization factor (lower = less lossy)
    )

    # Optional processing steps
    destripe: bool = True  # Remove horizontal stripe pattern from images
    interpolate_nan: bool = True  # Interpolate over NaN pixels
    interpolate_max_bad_neighbors: int = (
        3  # Max bad neighbors before skipping interpolation
    )
    destripe_bad_row_fraction: float = (
        0.5  # Row is bad if masked pixel fraction exceeds this
    )
    destripe_bad_plane_fraction: float = (
        1 / 3
    )  # Plane is bad if bad-row fraction exceeds this
    interpolate_nan_kernel_sigma: float = (
        1.0  # Gaussian kernel sigma (px) for NaN interpolation
    )

    # Projection
    projection: str | Projection | None = (
        None  # Coadd projection (header file path or Projection)
    )

    # Saturation levels
    set_saturation_levels: str = (
        "default"  # Saturation level preset ('default' or 'sv')
    )

    # Astrometry
    warp_gaia: bool = True  # Propagate Gaia proper motions to observation epoch
    external_headers: bool = False  # Use external astrometric headers
    scamp_mode: Literal["fix_focalplane", "loose"] = (
        "loose"  # SCAMP astrometric calibration mode
    )
    gaia_ruwe_max: float = 1.5  # Max Gaia RUWE for astrometric reference stars
    local_gaia_catalog: str | None = (
        None  # Path to local Gaia FITS catalog (skip download)
    )
    scamp_cache_dir: str | None = None  # Directory for caching SCAMP .ahead files
    local_cache_dir: str | None = (
        None  # Local temp directory for intermediate files (default: system temp)
    )

    # Photometry
    phot_reference_catalog: Literal["2MASS"] = "2MASS"  # Photometric reference catalog
    local_2mass_catalog: str | None = (
        None  # Path to local 2MASS FITS catalog (skip download)
    )
    reference_mag_lo: int | float = None  # Bright magnitude cut for reference stars
    reference_mag_hi: int | float = None  # Faint magnitude cut for reference stars
    target_zp: float = 25.0  # Target photometric zero point (mag)
    illumination_correction_mode: Literal["variable", "constant"] = (
        "variable"  # Illumination correction mode
    )
    phot_interp_kernel_k: int = (
        10  # Number of nearest neighbors for photometric interpolation
    )
    photometric_error_floor: float = 0.005  # Minimum photometric error (mag)
    catalog_download_radius_factor: float = (
        1.1  # Enlargement factor for catalog download radius
    )
    ic_background_mesh_size: int = (
        128  # Background mesh size for illumination correction
    )

    # Master calibration lookup
    master_max_lag_bpm: int | float = 14  # Max time lag (days) for BPM lookup
    master_max_lag_dark: int | float = 14  # Max time lag (days) for dark lookup
    master_max_lag_twilight_flat: int | float = (
        14  # Max time lag (days) for twilight flat lookup
    )
    master_max_lag_sky: int | float = 60  # Max time lag (days) for sky flat lookup
    master_max_lag_gain: int | float = 14  # Max time lag (days) for gain lookup
    master_max_lag_weight: int | float = 14  # Max time lag (days) for weight lookup
    master_max_lag_linearity: int | float = (
        21  # Max time lag (days) for linearity lookup
    )

    # Bad pixel masks
    bpm_max_lag: int | float = 1  # Max time lag (days) between BPM input frames
    bpm_rel_threshold: int | float = 0.04  # Relative threshold for bad pixel detection
    bpm_frac: int | float = 0.2  # Fraction of frames a pixel must be bad to be flagged
    bpm_n_min: int = 3  # Min number of BPM frames per group

    # Darks
    dark_max_lag: int | float = 1  # Max time lag (days) between dark input frames
    dark_mask_min: bool = True  # Mask minimum value outliers in darks
    dark_mask_max: bool = True  # Mask maximum value outliers in darks
    dark_metric: Literal["mean", "median", "clipped_mean", "clipped_median"] = (
        "mean"  # Combination metric for darks
    )
    dark_n_min: int = 3  # Min number of dark frames per group
    dark_nan_plane_threshold: float = (
        0.9  # Discard dark plane if valid pixel fraction below this
    )

    # Gain
    gain_max_lag: int | float = 1  # Max time lag (days) between gain input frames

    # Linearity
    linearity_max_lag: int | float = (
        1  # Max time lag (days) between linearity input frames
    )
    linearity_reference_adu: int | float = (
        10000  # Reference flux level (ADU) for non-linearity characterization
    )

    # Flats
    flat_type: Literal["sky", "twilight"] = "twilight"  # Type of flat field frames
    flat_max_lag: int | float = 1  # Max time lag (days) between flat input frames
    flat_mask_min: bool = True  # Mask minimum value outliers in flats
    flat_mask_max: bool = True  # Mask maximum value outliers in flats
    flat_rel_lo: int | float = 0.3  # Low relative threshold for flat rejection
    flat_rel_hi: int | float = 1.7  # High relative threshold for flat rejection
    flat_sigma_level: int | float = 3  # Sigma-clipping level for flat combination
    flat_sigma_iter: int = 1  # Number of sigma-clipping iterations for flats
    flat_metric: Literal[
        "weighted", "mean", "median", "clipped_mean", "clipped_median"
    ] = "weighted"  # Combination metric for flats
    flat_n_min: int = 3  # Min number of flat frames per group
    flat_min_flux: int | float = (
        500  # Min flux (ADU) per plane; lower planes are discarded
    )

    # Weights
    weight_mask_abs_min: int | float = 0.3  # Absolute min flat value for weight masking
    weight_mask_abs_max: int | float = 1.7  # Absolute max flat value for weight masking
    weight_mask_rel_min: int | float = 0.5  # Relative min flat value for weight masking
    weight_mask_rel_max: int | float = 1.5  # Relative max flat value for weight masking
    build_individual_weights_maximask: bool = (
        False  # Use MaxiMask for individual weight images
    )
    weight_bg_rms_factor: float = (
        1.5  # Factor above median bg RMS to zero detector weight
    )
    weight_background_mesh_size: int = (
        256  # Background mesh size for weight image RMS estimation
    )

    # Source masks
    mask_2mass_sources: bool = True  # Mask bright 2MASS sources in sky frames
    mask_2mass_sources_bright: int | float = (
        1.0  # Bright magnitude limit for 2MASS source masking
    )
    mask_2mass_sources_faint: int | float = (
        10.0  # Faint magnitude limit for 2MASS source masking
    )
    mask_bright_galaxies: bool = True  # Mask bright galaxies from external catalog
    additional_source_masks: str | SourceMasks | None = (
        None  # Path to additional DS9 region masks
    )
    source_mask_method: Literal["noisechisel", "built-in"] = (
        "noisechisel"  # Source detection method for masking
    )
    source_masks_n_min: int = 5  # Min frames required to build source masks
    source_mask_closing: bool = True  # Apply morphological closing to source masks
    source_masks_closing_size: int = 5  # Kernel size for mask morphological closing
    source_masks_closing_iter: int = 2  # Number of morphological closing iterations
    source_masks_destripe: bool = True  # Destripe images before source mask detection
    source_mask_outlier_sigma: int | float = (
        10  # Sigma threshold for extreme outlier masking
    )
    min_valid_pixel_fraction: float = (
        0.2  # Min fraction of finite pixels before raising error
    )
    # Noisechisel setup for source masks
    noisechisel_qthresh: float = 0.9  # Quantile threshold for noisechisel detection
    noisechisel_erode: int = 2  # Erosion size for noisechisel segmentation
    noisechisel_detgrowquant: float = 1.0  # Detection growth quantile for noisechisel
    noisechisel_tilesize: str = (
        "32,32"  # Tile size for noisechisel background estimation
    )
    noisechisel_meanmedqdiff: float = (
        0.05  # Mean-median quantile diff threshold for noisechisel
    )
    # Built-in setup for source masks
    mask_sources_thresh: int | float = (
        3  # Detection threshold (sigma) for built-in source masking
    )
    mask_sources_min_area: int | float = 3  # Min source area (px) for built-in masking
    mask_sources_max_area: int | float = (
        250000  # Max source area (px) for built-in masking
    )
    mask_bright_sources: bool = True  # Mask saturated/bright sources in built-in mode
    mask_sources_dilate: bool = True  # Dilate source masks in built-in mode

    # Sky
    sky_n_min: int = 5  # Min number of frames for sky construction
    sky_mix_science: bool = True  # Include science frames in sky building
    sky_window: int | float = 180  # Time window (minutes) for sky frame selection
    sky_mask_min: bool = True  # Mask minimum value outliers in sky frames
    sky_mask_max: bool = True  # Mask maximum value outliers in sky frames
    sky_rel_lo: int | float = 0.3  # Low relative threshold for sky rejection
    sky_rel_hi: int | float = 1.7  # High relative threshold for sky rejection
    sky_sigma_level: int | float = 3  # Sigma-clipping level for sky combination
    sky_sigma_iter: int = 1  # Number of sigma-clipping iterations for sky
    sky_combine_metric: Literal[
        "weighted", "mean", "median", "clipped_mean", "clipped_median"
    ] = "weighted"  # Combination metric for sky frames
    sky_static_sigma: int | float = (
        3  # Sigma threshold for masking in static sky building
    )
    # Background model
    subtract_background: bool = True  # Subtract 2D background model before resampling
    background_mesh_size: int = 64  # Background mesh size for 2D model estimation
    background_mesh_filtersize: int = 3  # Background mesh filter size for smoothing

    # Statistics images
    image_statistics_resize_factor: int | float = (
        0.25  # Downsample factor for statistics images
    )

    # Sextractor
    sex_back_size: int = 64  # SExtractor background mesh size
    sex_back_filtersize: int = 3  # SExtractor background filter size
    sex_fwhm_detect_thresh: int | float = (
        50  # Detection threshold (sigma) for FWHM preset
    )
    sex_detection_image_path: str | None = (
        None  # Path to external SExtractor detection image
    )

    # Swarp
    resampling_kernel: Literal[
        "nearest", "bilinear", "lanczos2", "lanczos3", "lanczos4"
    ] = "lanczos3"  # SWarp resampling interpolation kernel
    coadd_pixel_scale: float = (
        1 / 3
    )  # Default pixel scale (arcsec) for auto-built coadd headers
    swarp_mem_max: int = 12288  # Max usable RAM for SWarp (MB)
    swarp_combine_bufsize: int = 12288  # RAM for SWarp co-addition buffer (MB)
    swarp_vmem_max: int = 524288  # Max virtual memory (disk swap) for SWarp (MB)
    swarp_post_sleep: int = 5  # Seconds to wait after SWarp for filesystem sync
    mmm_max_pixels: int = 50_000_000  # Max pixels for tile background estimation
    n_offset_positions: int = 6  # Expected number of offset positions for stacks

    # Completeness
    build_completeness: bool = False  # Enable tile completeness analysis
    completeness_tile_size_arcmin: float = 20.0  # Sub-tile size for completeness
    completeness_resolution_arcmin: float = 10.0  # Spatial resolution for results
    completeness_iterations: int = 25  # Number of injection iterations per sub-tile
    completeness_mag_lo: float = 15.0  # Bright end of magnitude range
    completeness_mag_hi: float = 23.5  # Faint end of magnitude range
    completeness_mag_bin: float = 0.25  # Magnitude bin width
    completeness_star_density: float = 3.0  # Artificial stars per arcmin² per iteration
    completeness_match_radius: float = 1.0  # Match radius in arcsec

    # Binary names
    bin_sex: str = "sex"  # SExtractor binary name
    bin_scamp: str = "scamp"  # SCAMP binary name
    bin_swarp: str = "swarp"  # SWarp binary name
    bin_psfex: str = "psfex"  # PSFEx binary name
    bin_skymaker: str = "sky"  # SkyMaker binary name
    bin_noisechisel: str = "astnoisechisel"  # Gnuastro noisechisel binary name
    bin_stilts: str = "stilts"  # STILTS binary name

    # Notifications
    pushover_user_key: str | None = None  # Pushover user key
    pushover_api_token: str | None = None  # Pushover API token

    # Miscellaneous
    survey_name: str = "VISIONS"  # Survey name for FITS headers
    fix_vircam_headers: bool = True  # Apply VIRCAM-specific header corrections

    # Folders
    folders: dict[str, str] | None = None  # Pipeline folder tree (auto-generated)

    def __post_init__(self):
        # Simple setup check
        if self.name is None:
            raise PipelineValueError("Pipeline setup needs a name")
        if self.path_data is None or not os.path.exists(self.path_data):
            raise PipelineValueError("Please provide valid path to data")
        if self.path_pype is None:
            raise PipelineValueError("Please provide valid path for pipeline output")
        if not self.path_pype.endswith("/"):
            self.path_pype += "/"
        if self.path_master_common is None:
            raise PipelineValueError(
                "Please provide path_master_common in the pipeline setup"
            )
        # Ensure trailing slashes on master paths
        if not self.path_master_common.endswith("/"):
            self.path_master_common += "/"
        if self.path_master_object is not None and not self.path_master_object.endswith(
            "/"
        ):
            self.path_master_object += "/"
        physical_cores = cpu_count(only_physical_cores=True)
        if self.n_jobs == 0:
            self.n_jobs = physical_cores
        elif self.n_jobs < 0:
            self.n_jobs = max(1, physical_cores + self.n_jobs)
        elif self.n_jobs > physical_cores:
            self.n_jobs = physical_cores
        self.n_jobs_basic = min(self.n_jobs_basic, self.n_jobs)
        self.n_jobs_sex = min(self.n_jobs_sex, self.n_jobs)
        if self.n_jobs_scamp is None:
            self.n_jobs_scamp = self.n_jobs
        else:
            self.n_jobs_scamp = min(self.n_jobs_scamp, self.n_jobs)
        if self.n_jobs_swarp is None:
            self.n_jobs_swarp = self.n_jobs
        else:
            self.n_jobs_swarp = min(self.n_jobs_swarp, self.n_jobs)

        # Validate local catalog paths
        if self.local_gaia_catalog is not None:
            if not os.path.isfile(self.local_gaia_catalog):
                raise PipelineValueError(
                    f"Local Gaia catalog not found: '{self.local_gaia_catalog}'"
                )
        if self.local_2mass_catalog is not None:
            if not os.path.isfile(self.local_2mass_catalog):
                raise PipelineValueError(
                    f"Local 2MASS catalog not found: '{self.local_2mass_catalog}'"
                )

        # Set up SCAMP cache directory
        if self.scamp_cache_dir is not None:
            self.scamp_cache_dir = os.path.join(self.scamp_cache_dir, self.name, "")
            os.makedirs(self.scamp_cache_dir, exist_ok=True)

        # Set up local cache directory for intermediate/temporary files
        if self.local_cache_dir is not None:
            self.local_cache_dir = os.path.join(self.local_cache_dir, self.name, "")
            os.makedirs(self.local_cache_dir, exist_ok=True)

        # Set keywords
        self.keywords = HeaderKeywords()

        # Folders
        if self.folders is None:
            self.folders = {}
        self.__add_folder_tree()
        self.__create_folder_tree()

        # Specific file paths
        self.path_coadd = f"{self.folders['tile']}{self.name}.fits"
        self.path_coadd_weight = self.path_coadd.replace(".fits", ".weight.fits")
        self.path_coadd_header = self.path_coadd.replace(".fits", ".ahead")

        # Other
        self.__set_projection()
        self.__set_additional_source_masks()
        self.__check_flat_type()

    # Fixed properties that users can't change
    @property
    def set_airmass(self) -> bool:
        """Whether to compute and store airmass in output headers."""
        return True

    @property
    def fpa_layout(self) -> list[int]:
        """VIRCAM focal-plane array layout as [rows, columns]."""
        return [4, 4]

    # Mutable defaults that can be changed
    @property
    def __default_saturation_levels(self) -> list[float]:
        """Saturation levels for each detector."""
        return [
            33000.0,
            32000.0,
            33000.0,
            32000.0,
            24000.0,
            24000.0,
            35000.0,
            33000.0,
            35000.0,
            35000.0,
            37000.0,
            34000.0,
            33000.0,
            35000.0,
            34000.0,
            34000.0,
        ]

    @property
    def __sv_saturation_levels(self) -> list[float]:
        """Saturation levels for each detector."""
        return [
            30000.0,
            32000.0,
            33000.0,
            32000.0,
            24000.0,
            30000.0,
            32000.0,
            33000.0,
            35000.0,
            32000.0,
            35000.0,
            32000.0,
            33000.0,
            34000.0,
            33000.0,
            35000.0,
        ]

    @property
    def saturation_levels(self) -> list[int | float]:
        """Returns the saturation level for a given detector."""
        if self.set_saturation_levels == "default":
            return self.__default_saturation_levels
        elif self.set_saturation_levels == "sv":
            return self.__sv_saturation_levels
        else:
            raise PipelineValueError(
                "Saturation levels not set correctly. Only 'default' or 'sv' allowed."
            )

    @property
    def image_statistics_combine_type(self) -> dict:
        """SWarp combination types for each statistics image mode."""
        return {
            "nimg": "SUM",
            "exptime": "SUM",
            "mjd.int": "WEIGHTED",
            "mjd.frac": "WEIGHTED",
            "astrms1": "WEIGHTED",
            "astrms2": "WEIGHTED",
        }

    @property
    def apertures(self) -> list[float]:
        """Fixed aperture diameters (pixels) for photometric extraction."""
        return [
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            12.0,
            14.0,
            16.0,
            18.0,
            21.0,
            24.0,
            27.0,
            30.0,
        ]

    def __check_flat_type(self):
        if self.flat_type.lower() not in ["sky", "twilight"]:
            raise PipelineValueError("Flat type must be either 'sky' or 'twilight'")

    def __set_projection(self):
        # If the attribute is already a Projection instance, do nothing
        if isinstance(self.projection, Projection):
            return
        # If it's None, also do nothing
        if self.projection is None:
            return
        # If specified as a string, assume it is a header text file path
        elif isinstance(self.projection, (str, Path)):
            # Convert to Path object
            if isinstance(self.projection, str):
                path_projection = Path(self.projection)
            else:
                path_projection = self.projection

            # Get filename
            name_projection = path_projection.name

            # Check if file exists
            if not path_projection.exists():
                raise PipelineValueError(
                    f"Projection header file '{self.projection}' not found"
                )

            # Load header from text file
            hdr = fits.Header.fromtextfile(path_projection)

            # Try to get name from header, otherwise set to empty string
            try:
                name = hdr["NAME"]
            except KeyError:
                name = name_projection

            # Set projection
            self.projection = Projection(header=hdr, name=name)

        else:
            raise PipelineValueError(
                "Projection must be provided as string or Projection instance"
            )

    def __set_additional_source_masks(self):
        # If given as a dict manually, do nothing
        if isinstance(self.additional_source_masks, dict):
            return

        # If given as a source mask instance also do nothing
        if isinstance(self.additional_source_masks, SourceMasks):
            return

        # If None, do nothing
        if self.additional_source_masks is None:
            return

        # If specified as string, try to load supported predefined masks
        if isinstance(self.additional_source_masks, (str, Path)):
            # Convert to Path object
            if isinstance(self.additional_source_masks, str):
                path_masks = Path(self.additional_source_masks)
            else:
                path_masks = self.additional_source_masks

            # Check if file exists
            if not path_masks.exists():
                raise PipelineValueError(f"Source mask file '{path_masks}' not found")

            # Get filename
            name_masks = path_masks.name

            # Read and set masks
            regions = Regions.read(path_masks, format="ds9")
            self.additional_source_masks = SourceMasks(regions=regions, name=name_masks)

        # Otherwise raise error
        else:
            raise ValueError("Provide valid source masks")

    @classmethod
    def load_pipeline_setup(cls, setup, **kwargs):
        """
        Load a setup instance.

        Parameters
        ----------
        setup : str, dict, Setup
            Either a string pointing to the location of a pipeline YML,
            or a dict, or a Setup instance.

        Returns
        -------
        Setup
            Setup instance.

        """

        # If given as string, load YML
        if isinstance(setup, str):
            return cls(**read_yml(path_yml=setup), **kwargs)

        # If given as Setup instance, just return it again
        if isinstance(setup, cls):
            return setup

        elif isinstance(setup, dict):
            return cls(**setup)

        # If something else was provided
        else:
            raise ValueError("Please provide a pipeline setup")

    def __add_folder_tree(self):
        """Adds pipeline folder tree to setup."""
        self.folders["pype"] = self.path_pype
        self.folders["raw"] = self.path_data
        self.folders["object"] = f"{self.path_pype}{self.name}/"
        self.folders["master_common"] = self.path_master_common
        self.folders["master_object"] = (
            self.path_master_object or f"{self.folders['object']}calibration/"
        )
        self.folders["temp"] = f"{self.folders['object']}temp/"
        self.folders["processed_basic"] = f"{self.folders['object']}processing/basic/"
        self.folders["processed_final"] = f"{self.folders['object']}processing/final/"
        self.folders["resampled"] = f"{self.folders['object']}processing/resampled/"
        self.folders["illumcorr"] = f"{self.folders['object']}processing/illumcorr/"
        self.folders["qc"] = f"{self.folders['object']}qc/"
        self.folders["qc_bpm"] = f"{self.folders['qc']}bpm/"
        self.folders["qc_dark"] = f"{self.folders['qc']}dark/"
        self.folders["qc_gain"] = f"{self.folders['qc']}gain/"
        self.folders["qc_linearity"] = f"{self.folders['qc']}linearity/"
        self.folders["qc_flat"] = f"{self.folders['qc']}flat/"
        self.folders["qc_sky"] = f"{self.folders['qc']}sky/"
        self.folders["qc_astrometry"] = f"{self.folders['qc']}astrometry/"
        self.folders["qc_photometry"] = f"{self.folders['qc']}photometry/"
        self.folders["qc_illumcorr"] = f"{self.folders['qc']}illumcorr/"
        self.folders["qc_completeness"] = f"{self.folders['qc']}completeness/"
        self.folders["qc_psf"] = f"{self.folders['qc']}psf/"
        self.folders["temp_completeness_tiles"] = (
            f"{self.folders['temp']}completeness/tiles/"
        )
        self.folders["temp_completeness_psf"] = (
            f"{self.folders['temp']}completeness/psf/"
        )
        self.folders["statistics"] = f"{self.folders['object']}processing/statistics/"
        self.folders["stacks"] = f"{self.folders['object']}products/stacks/"
        self.folders["tile"] = f"{self.folders['object']}products/tile/"
        self.folders["phase3"] = f"{self.folders['object']}products/phase3/"

    def __create_folder_tree(self):
        """Creates the folder tree for the pipeline"""

        # Common paths
        folders_common = [
            self.folders["pype"],
            self.folders["master_common"],
            self.folders["temp"],
        ]

        # calibration-specific paths
        folders_cal = [
            self.folders["qc_bpm"],
            self.folders["qc_dark"],
            self.folders["qc_gain"],
            self.folders["qc_linearity"],
            self.folders["qc_flat"],
        ]

        # Object-specific paths
        folders_object = [
            self.folders["master_object"],
            self.folders["illumcorr"],
            self.folders["qc"],
            self.folders["qc_sky"],
            self.folders["processed_basic"],
            self.folders["processed_final"],
            self.folders["qc_astrometry"],
            self.folders["resampled"],
            self.folders["statistics"],
            self.folders["tile"],
            self.folders["qc_photometry"],
            self.folders["qc_illumcorr"],
            self.folders["qc_completeness"],
            self.folders["qc_psf"],
            self.folders["temp_completeness_tiles"],
            self.folders["temp_completeness_psf"],
        ]

        # Create folder for stacks if set
        if self.build_stacks:
            folders_object += [self.folders["stacks"]]

        # Create folder for phase3 if set
        if self.build_phase3:
            folders_object += [self.folders["phase3"]]

        # Generate common paths
        for path in folders_common:
            make_folder(path)

        # Create common calibration path only if we run a calibration unit
        if "calibration" in self.name.lower():
            for path in folders_cal:
                make_folder(path=path)

        # Otherwise make object paths
        else:
            for path in folders_object:
                make_folder(path=path)

    @property
    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable dictionary of the setup parameters."""
        # Get complete dict
        dd = asdict(self)

        # Replace projection and source masks with their names
        if isinstance(dd["projection"], Projection):
            dd["projection"] = dd["projection"].name
        if isinstance(dd["additional_source_masks"], SourceMasks):
            dd["additional_source_masks"] = dd["additional_source_masks"].name

        # Remove path attributes and folder setup
        del dd["path_data"]
        del dd["path_pype"]
        del dd["folders"]

        # Return
        return dd

    def add_setup_to_header(self, header: fits.Header) -> None:
        """
        Adds setup information to a FITS header.

        Parameters
        ----------
        header : fits.Header
            The FITS header to which the setup information will be added.
        """
        for key, val in self.to_dict.items():
            if isinstance(val, tuple):
                val = str(val)
            elif isinstance(val, Projection):
                val = val.name
            elif isinstance(val, SourceMasks):
                val = val.name

            # Iteratively clip the value if it is too long
            key_upper = key.upper()
            if isinstance(val, str):
                while len(val) > 0:
                    try:
                        header.set(
                            keyword=f"HIERARCH PYPE SETUP {key_upper}",
                            value=val,
                            comment="",
                        )
                        # Loop over cards and verify
                        for card in header.cards:
                            card.verify("silentfix")
                        break
                    except ValueError as e:
                        if "too long" in str(e):
                            val = val[1:]
                        else:
                            raise
                else:
                    raise ValueError(
                        f"Value for key '{key_upper}' is too long and cannot be shortened further."
                    )
            else:
                header.set(
                    keyword=f"HIERARCH PYPE SETUP {key_upper}", value=val, comment=""
                )


class HeaderKeywords:
    """Mapping of logical header field names to ESO/VIRCAM FITS keywords."""

    def __init__(
        self,
        obj="OBJECT",
        filter_name="HIERARCH ESO INS FILT1 NAME",
        dpr_type="HIERARCH ESO DPR TYPE",
        dpr_category="HIERARCH ESO DPR CATG",
        dit="HIERARCH ESO DET DIT",
        ndit="HIERARCH ESO DET NDIT",
        date_mjd="MJD-OBS",
        date_ut="DATE-OBS",
        gain="GAIN",
        rdnoise="RDNOISE",
        saturate="SATURATE",
        airmass="AIRMASS",
    ):
        self.object = obj
        self.filter_name = filter_name
        self.type = dpr_type
        self.category = dpr_category
        self.dit = dit
        self.ndit = ndit
        self.date_mjd = date_mjd
        self.date_ut = date_ut
        self.gain = gain
        self.rdnoise = rdnoise
        self.saturate = saturate
        self.airmass = airmass
