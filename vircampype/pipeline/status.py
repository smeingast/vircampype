import pickle


class PipelineStatus:
    def __init__(
        self,
        master_bpm=False,
        master_linearity=False,
        master_dark=False,
        master_gain=False,
        master_flat=False,
        master_weight_global=False,
        processed_raw_basic=False,
        master_sky_static=False,
        master_source_mask=False,
        master_sky_dynamic=False,
        master_photometry=False,
        master_astrometry=False,
        processed_raw_final=False,
        master_weight_image=False,
        tile_header=False,
        astrometry=False,
        illumcorr=False,
        resampled=False,
        statistics_resampled=False,
        photometry_pawprints=False,
        photerr_internal=False,
        stacks=False,
        statistics_stacks=False,
        classification_stacks=False,
        photometry_stacks=False,
        qc_photometry_stacks=False,
        qc_astrometry_stacks=False,
        tile=False,
        statistics_tile=False,
        classification_tile=False,
        photometry_tile=False,
        qc_photometry_tile=False,
        qc_astrometry_tile=False,
        phase3=False,
        archive=False,
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

    def save(self, path):
        pickle.dump(self.status_dict, open(path, "wb"))

    def read(self, path):
        status = pickle.load(open(path, "rb"))
        for key, val in status.items():
            setattr(self, key, val)
