import os

from joblib import cpu_count
from typing import Union, List, Optional
from vircampype.pipeline.errors import *
from dataclasses import dataclass, asdict
from vircampype.tools.systemtools import *
from vircampype.visions.projections import *
from vircampype.miscellaneous.sourcemasks import *
from vircampype.miscellaneous.projection import Projection


@dataclass
class Setup(dict):

    # Pipeline setup
    name: str = None
    path_data: str = None
    path_pype: str = None
    n_jobs: int = 8
    silent: bool = False
    overwrite: bool = False
    qc_plots: bool = True

    # What to process
    calibrate_pawprints: bool = False
    build_stacks: bool = False
    build_tile: bool = True
    build_tile_only: bool = False
    build_phase3: bool = False
    build_public_catalog: bool = False
    source_classification: bool = False
    archive: bool = False

    # Data setup
    pixel_scale_arcsec: Union[int, float] = 1 / 3
    maximask: bool = False
    projection: Optional[Union[str, Projection]] = None

    # Bad pixel masks
    bpm_max_lag: Union[int, float] = 1
    bpm_rel_threshold: Union[int, float] = 0.04
    bpm_frac: Union[int, float] = 0.2

    # Darks
    dark_max_lag: Union[int, float] = 1
    dark_mask_min: bool = True
    dark_mask_max: bool = True
    dark_metric: str = "mean"

    # Gain
    gain_max_lag: Union[int, float] = 1

    # Lineartiy
    linearity_max_lag: Union[int, float] = 1

    # Flats
    flat_max_lag: Union[int, float] = 1
    flat_mask_min: bool = True
    flat_mask_max: bool = True
    flat_rel_lo: Union[int, float] = 0.3
    flat_rel_hi: Union[int, float] = 1.7
    flat_sigma_level: Union[int, float] = 3
    flat_sigma_iter: int = 1
    flat_metric: str = "weighted"

    # Weights
    weight_mask_abs_max: Union[int, float] = 1.7
    weight_mask_abs_min: Union[int, float] = 0.3
    weight_mask_rel_max: Union[int, float] = 1.5
    weight_mask_rel_min: Union[int, float] = 0.5

    # Source masks
    mask_sources_thresh: Union[int, float] = 3
    mask_sources_min_area: Union[int, float] = 3
    mask_sources_max_area: Union[int, float] = 250000
    mask_2mass_sources: bool = True
    mask_bright_sources: bool = True
    additional_source_masks: Optional[Union[str, SourceMasks]] = None

    # Master calibration lookup
    master_max_lag_bpm: Union[int, float] = 14
    master_max_lag_dark: Union[int, float] = 14
    master_max_lag_sky: Union[int, float] = 60
    master_max_lag_flat: Union[int, float] = 14
    master_max_lag_gain: Union[int, float] = 14
    master_max_lag_weight: Union[int, float] = 14
    master_max_lag_linearity: Union[int, float] = 14

    # Processing
    sky_n_min: int = 5
    sky_mix_science: bool = True
    sky_window: Union[int, float] = 180
    sky_combine_metric: str = "weighted"
    subtract_background: bool = True
    background_mesh_size: int = 64
    background_mesh_filtersize: int = 3
    destripe: bool = True
    interpolate_nan: bool = True
    interpolate_max_bad_neighbors: int = 3
    image_statistics_resize_factor: Union[int, float] = 0.25

    # Binary names
    bin_sex: str = "sex"
    bin_scamp: str = "scamp"
    bin_swarp: str = "swarp"
    bin_psfex: str = "psfex"
    bin_noisechisel: str = "astnoisechisel"
    bin_stilts: str = "stilts"

    # Astromatic
    swarp_back_size: int = 384
    swarp_back_filtersize: int = 3
    sex_back_size: int = 64
    sex_back_filtersize: int = 3

    # Astrometry
    astr_reference_catalog: str = "GAIA"
    warp_gaia: bool = True
    external_headers: bool = False

    # Photometry
    phot_reference_catalog: str = "2MASS"
    reference_mag_lim: tuple = None
    target_zp: float = 25.0
    ic_mode: str = "variable"
    photerr_internal: float = 0.005

    # Folders
    folders: dict = None

    # Phase 3
    compress_phase3: bool = True
    fpack_quantization_factor: int = 16

    def __post_init__(self):

        # Simple setup check
        if self.name is None:
            raise PipelineError("Pipeline setup needs a name")
        if (self.path_data is None) | (os.path.exists(self.path_data) is False):
            raise PipelineError("Please provide valid path to data")
        if self.path_pype is None:
            raise PipelineError("Please provide valid path for pipeline output")
        if self.n_jobs > cpu_count():
            raise ValueError("More parallel jobs than available CPUs requested.")

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
        self.pixel_scale_degrees = self.pixel_scale_arcsec / 3600
        self.__set_projection()
        self.__set_additional_source_masks()

    # Fixed properties that users can't change
    @property
    def joblib_backend(self) -> str:
        return "threads"

    @property
    def fix_vircam_headers(self) -> bool:
        return True

    @property
    def set_airmass(self) -> bool:
        return True

    @property
    def fpa_layout(self) -> List[int]:
        return [4, 4]

    # Mutable defaults that can be changed
    @property
    def __default_saturation_levels(self) -> List[float]:
        """Saturation levels for each detector."""
        return [
            33000.,
            32000.,
            33000.,
            32000.,
            24000.,
            24000.,
            35000.,
            33000.,
            35000.,
            35000.,
            37000.,
            34000.,
            33000.,
            35000.,
            34000.,
            34000.,
        ]

    @property
    def saturation_levels(self) -> List[Union[int, float]]:
        """Returns the saturation level for a given detector."""
        return self.__default_saturation_levels

    @property
    def image_statistics_combine_type(self) -> dict:
        return {
            "nimg": "SUM",
            "exptime": "SUM",
            "mjd.int": "WEIGHTED",
            "mjd.frac": "WEIGHTED",
            "astrms1": "WEIGHTED",
            "astrms2": "WEIGHTED",
        }

    @property
    def apertures(self) -> List[float]:
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

    def __set_projection(self):

        # If the atrtibute is already a Projection instance, do nothing
        if isinstance(self.projection, Projection):
            return

        # If it's None, also do nothing
        if self.projection is None:
            return

        # If specified as a string, translate to a Projection instance
        elif isinstance(self.projection, str):
            # Chamaeleon
            if self.projection.lower() == "chamaeleon_wide":
                self.projection = ChamaeleonWideProjection()
            elif self.projection.lower() == "chamaeleon_deep":
                self.projection = ChamaeleonDeepProjection()
            elif self.projection.lower() == "chamaeleon_control":
                self.projection = ChamaeleonControlProjection()

            # Corona Australis
            elif self.projection.lower() == "corona_australis_wide":
                self.projection = CoronaAustralisWideProjection()
            elif self.projection.lower() == "corona_australis_deep":
                self.projection = CoronaAustralisDeepProjection()
            elif self.projection.lower() == "corona_australis_control":
                self.projection = CoronaAustralisControlProjection()

            # Lupus
            elif self.projection.lower() == "lupus_wide":
                self.projection = LupusWideProjection()
            elif self.projection.lower() == "lupus_deep_n":
                self.projection = LupusDeepNProjection()
            elif self.projection.lower() == "lupus_deep_s":
                self.projection = LupusDeepSProjection()
            elif self.projection.lower() == "lupus_control_n":
                self.projection = LupusControlNProjection()
            elif self.projection.lower() == "lupus_control_s":
                self.projection = LupusControlSProjection()

            # Ophiuchus
            elif self.projection.lower() == "ophiuchus_wide":
                self.projection = OphiuchusWideProjection()
            elif self.projection.lower() == "ophiuchus_deep":
                self.projection = OphiuchusDeepProjection()
            elif self.projection.lower() == "ophiuchus_control":
                self.projection = OphiuchusControlProjection()

            # Orion
            elif self.projection.lower() == "orion_control":
                self.projection = OrionControlProjection()

            # Pipe
            elif self.projection.lower() == "pipe_deep":
                self.projection = PipeDeepProjection()

            # Sharks
            elif self.projection.lower() == "sharksg15115":
                self.projection = SharksG15115()

            # If no match found, raise error
            else:
                print(self.projection)
                raise PipelineError(f"Projection '{self.projection}' not supported")
        else:
            raise PipelineError(
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
        if isinstance(self.additional_source_masks, str):
            if self.additional_source_masks.lower() == "chamaeleon_deep":
                self.additional_source_masks = ChamaeleonDeepSourceMasks()
            elif self.additional_source_masks.lower() == "corona_australis_deep":
                self.additional_source_masks = CoronaAustralisDeepSourceMasks()
            elif self.additional_source_masks.lower() == "corona_australis_wide":
                self.additional_source_masks = CoronaAustralisWideSourceMasks()
            elif self.additional_source_masks.lower() == "corona_australis_control":
                self.additional_source_masks = CoronaAustralisControlSourceMasks()
            elif self.additional_source_masks.lower() == "lupus_deep":
                self.additional_source_masks = LupusDeepSourceMasks()
            elif self.additional_source_masks.lower() == "ophiuchus_deep":
                self.additional_source_masks = OphiuchusDeepSourceMasks()
            elif self.additional_source_masks.lower() == "pipe_deep":
                self.additional_source_masks = PipeDeepSourceMasks()
            elif self.additional_source_masks.lower() == "orion_wide":
                self.additional_source_masks = OrionWideSourceMasks()
            else:
                raise ValueError(
                    f"Source masks '{self.additional_source_masks}' are not supported"
                )

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
        self.folders["headers"] = f"{self.folders['object']}{'headers'}/"
        self.folders["master_common"] = f"{self.path_pype}{'master'}/"
        self.folders["master_object"] = f"{self.folders['object']}{'master'}/"
        self.folders["temp"] = f"{self.folders['object']}{'temp'}/"
        self.folders[
            "processed_basic"
        ] = f"{self.folders['object']}{'processed_basic'}/"
        self.folders[
            "processed_final"
        ] = f"{self.folders['object']}{'processed_final'}/"
        self.folders["resampled"] = f"{self.folders['object']}{'resampled'}/"
        self.folders["illumcorr"] = f"{self.folders['object']}{'illumcorr'}/"
        self.folders["qc"] = f"{self.folders['object']}{'qc'}/"
        self.folders["qc_bpm"] = f"{self.folders['qc']}{'bpm'}/"
        self.folders["qc_dark"] = f"{self.folders['qc']}{'dark'}/"
        self.folders["qc_gain"] = f"{self.folders['qc']}{'gain'}/"
        self.folders["qc_linearity"] = f"{self.folders['qc']}{'linearity'}/"
        self.folders["qc_flat"] = f"{self.folders['qc']}{'flat'}/"
        self.folders["qc_sky"] = f"{self.folders['qc']}{'sky'}/"
        self.folders["qc_astrometry"] = f"{self.folders['qc']}{'astrometry'}/"
        self.folders["qc_photometry"] = f"{self.folders['qc']}{'photometry'}/"
        self.folders["qc_illumcorr"] = f"{self.folders['qc']}{'illumcorr'}/"
        self.folders["statistics"] = f"{self.folders['object']}{'statistics'}/"
        self.folders["stacks"] = f"{self.folders['object']}{'stacks'}/"
        self.folders["tile"] = f"{self.folders['object']}{'tile'}/"
        self.folders["phase3"] = f"{self.path_pype}{'phase3/'}{self.name}/"

    def __create_folder_tree(self):
        """Creates the folder tree for the pipeline"""

        # Common paths
        folders_common = [
            self.folders["pype"],
            self.folders["headers"],
            self.folders["master_common"],
            self.folders["temp"],
        ]

        if self.build_phase3:
            folders_common += [self.folders["phase3"]]

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
            self.folders["illumcorr"],
        ]

        # Create folder for stacks if set
        if self.build_stacks:
            folders_object += [self.folders["stacks"]]

        # Generate common paths
        for path in folders_common:
            make_folder(path)

        # Create common calibration path only if we run a calibration unit
        if "calibration" in self.name.lower():
            for path in folders_cal:
                make_folder(path=path)

        # Other wise make object paths
        else:
            for path in folders_object:
                make_folder(path=path)

    @property
    def dict(self):

        # Get complete dict
        dd = asdict(self)

        # Remove path attributes and folder setup
        del dd["path_data"]
        del dd["path_pype"]
        del dd["folders"]

        # Return
        return dd


class HeaderKeywords:
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
