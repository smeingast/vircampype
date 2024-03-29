import pickle


class PipelineStatus:
    def __init__(
        self,
        master_bpm: bool = False,
        master_linearity: bool = False,
        master_dark: bool = False,
        master_gain: bool = False,
        master_flat: bool = False,
        master_weight_global: bool = False,
        processed_raw_basic: bool = False,
        master_sky_static: bool = False,
        master_source_mask: bool = False,
        master_sky_dynamic: bool = False,
        master_photometry: bool = False,
        master_astrometry: bool = False,
        processed_raw_final: bool = False,
        master_weight_image: bool = False,
        tile_header: bool = False,
        astrometry: bool = False,
        illumcorr: bool = False,
        resampled: bool = False,
        statistics_resampled: bool = False,
        photometry_pawprints: bool = False,
        photerr_internal: bool = False,
        stacks: bool = False,
        statistics_stacks: bool = False,
        classification_stacks: bool = False,
        photometry_stacks: bool = False,
        qc_photometry_stacks: bool = False,
        qc_astrometry_stacks: bool = False,
        tile: bool = False,
        statistics_tile: bool = False,
        classification_tile: bool = False,
        photometry_tile: bool = False,
        qc_photometry_tile: bool = False,
        qc_astrometry_tile: bool = False,
        phase3: bool = False,
        public_catalog: bool = False,
        archive: bool = False,
    ):

        # Set status calibration attributes
        self.master_bpm = master_bpm
        self.master_linearity = master_linearity
        self.master_dark = master_dark
        self.master_gain = master_gain
        self.master_flat = master_flat
        self.master_weight_global = master_weight_global

        # Set science attributes
        self.processed_raw_basic = processed_raw_basic
        self.master_sky_static = master_sky_static
        self.master_source_mask = master_source_mask
        self.master_sky_dynamic = master_sky_dynamic
        self.master_photometry = master_photometry
        self.master_astrometry = master_astrometry
        self.processed_raw_final = processed_raw_final
        self.master_weight_image = master_weight_image
        self.tile_header = tile_header
        self.astrometry = astrometry
        self.illumcorr = illumcorr
        self.resampled = resampled
        self.statistics_resampled = statistics_resampled
        self.photometry_pawprints = photometry_pawprints
        self.photerr_internal = photerr_internal
        self.stacks = stacks
        self.statistics_stacks = statistics_stacks
        self.classification_stacks = classification_stacks
        self.photometry_stacks = photometry_stacks
        self.qc_photometry_stacks = qc_photometry_stacks
        self.qc_astrometry_stacks = qc_astrometry_stacks
        self.tile = tile
        self.statistics_tile = statistics_tile
        self.classification_tile = classification_tile
        self.photometry_tile = photometry_tile
        self.qc_photometry_tile = qc_photometry_tile
        self.qc_astrometry_tile = qc_astrometry_tile
        self.phase3 = phase3
        self.public_catalog = public_catalog
        self.archive = archive

    def __str__(self):
        return self.status_dict.__str__()

    def __repr__(self):
        return self.status_dict.__repr__()

    @staticmethod
    def __attributes():
        return [
            "master_bpm",
            "master_linearity",
            "master_dark",
            "master_gain",
            "master_flat",
            "master_weight_global",
            "processed_raw_basic",
            "master_sky_static",
            "master_source_mask",
            "master_sky_dynamic",
            "master_photometry",
            "master_astrometry",
            "processed_raw_final",
            "master_weight_image",
            "tile_header",
            "astrometry",
            "illumcorr",
            "resampled",
            "statistics_resampled",
            "photometry_pawprints",
            "stacks",
            "statistics_stacks",
            "classification_stacks",
            "photometry_stacks",
            "qc_photometry_stacks",
            "qc_astrometry_stacks",
            "tile",
            "statistics_tile",
            "classification_tile",
            "photometry_tile",
            "qc_photometry_tile",
            "qc_astrometry_tile",
            "phase3",
            "public_catalog",
            "archive",
        ]

    @property
    def status_dict(self):
        return {attr: getattr(self, attr) for attr in self.__attributes()}

    def update(self, **kwargs):
        for key, val in kwargs.items():
            if key not in self.__attributes():
                raise ValueError(
                    "Cannot set pipeline status for attribute '{0}'".format(key)
                )
            else:
                setattr(self, key, val)

    def reset(self):
        """Set all status attributes to False"""
        for attr in self.__attributes():
            self.update(**{attr: False})

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.status_dict, f)

    def read(self, path: str):
        with open(path, "rb") as f:
            status = pickle.load(f)
            for key, val in status.items():
                setattr(self, key, val)
