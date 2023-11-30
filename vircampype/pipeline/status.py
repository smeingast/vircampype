import pickle
import dataclasses


@dataclasses.dataclass
class PipelineStatus:
    """
    Class to manage and store the status of a pipeline.

    Attributes
    ----------
    master_bpm : bool
        Status of Master BPM, defaults to False.
    master_linearity : bool
        Status of Master Linearity, defaults to False.
    master_dark : bool
        Status of Master Dark, defaults to False.
    master_gain : bool
        Status of Master Gain, defaults to False.
    master_twilight_flat : bool
        Status of Master Twilight Flat, defaults to False.
    master_weight_global : bool
        Status of Master Weight Global, defaults to False.
    processed_raw_basic : bool
        Status of Processed Raw Basic, defaults to False.
    master_source_mask : bool
        Status of Master Source Mask, defaults to False.
    master_sky : bool
        Status of Master Sky, defaults to False.
    master_photometry : bool
        Status of Master Photometry, defaults to False.
    master_astrometry : bool
        Status of Master Astrometry, defaults to False.
    processed_raw_final : bool
        Status of Processed Raw Final, defaults to False.
    master_weight_image : bool
        Status of Master Weight Image, defaults to False.
    tile_header : bool
        Status of Tile Header, defaults to False.
    astrometry : bool
        Status of Astrometry, defaults to False.
    illumcorr : bool
        Status of Illumination Correction, defaults to False.
    resampled : bool
        Status of Resampled, defaults to False.
    statistics_resampled : bool
        Status of Statistics for Resampled, defaults to False.
    photometry_pawprints : bool
        Status of Photometry Pawprints, defaults to False.
    photerr_internal : bool
        Status of Photometry Error Internal, defaults to False.
    stacks : bool
        Status of Stacks, defaults to False.
    statistics_stacks : bool
        Status of Statistics Stacks, defaults to False.
    classification_stacks : bool
        Status of Classification Stacks, defaults to False.
    photometry_stacks : bool
        Status of Photometry Stacks, defaults to False.
    qc_photometry_stacks : bool
        Status of Quality Control Photometry Stacks, defaults to False.
    qc_astrometry_stacks : bool

    Methods
    -------
    update(**kwargs):
        Updates the statuses of provided attributes.
    reset():
        Resets all status attributes to False.
    save(path: str):
        Saves the current instance's status dictionary to a file.
    load(path: str):
        Loads an instance's status dictionary from a file.

    """

    master_bpm: bool = False
    master_linearity: bool = False
    master_dark: bool = False
    master_gain: bool = False
    master_twilight_flat: bool = False
    master_weight_global: bool = False
    processed_raw_basic: bool = False
    master_source_mask: bool = False
    master_sky: bool = False
    master_photometry: bool = False
    master_astrometry: bool = False
    processed_raw_final: bool = False
    master_weight_image: bool = False
    tile_header: bool = False
    astrometry: bool = False
    illumcorr: bool = False
    resampled: bool = False
    statistics_resampled: bool = False
    photometry_pawprints: bool = False
    photerr_internal: bool = False
    stacks: bool = False
    statistics_stacks: bool = False
    classification_stacks: bool = False
    photometry_stacks: bool = False
    qc_photometry_stacks: bool = False
    qc_astrometry_stacks: bool = False
    tile: bool = False
    statistics_tile: bool = False
    classification_tile: bool = False
    photometry_tile: bool = False
    qc_photometry_tile: bool = False
    qc_astrometry_tile: bool = False
    phase3: bool = False
    public_catalog: bool = False
    archive: bool = False

    def update(self, **kwargs):
        """
        Updates the statuses of provided attributes.

        Parameters
        ----------
        **kwargs : kwargs
            Keyword arguments representing the statuses to update.

        Raises
        ------
        ValueError
            If provided status attribute does not exist.

        """
        for key, value in kwargs.items():
            if key not in dataclasses.asdict(self):
                raise ValueError(f"Cannot set pipeline status for attribute '{key}'")
            else:
                setattr(self, key, value)

    def reset(self):
        """
        Resets all status attributes to False.

        """
        for field in dataclasses.fields(self):
            setattr(self, field.name, False)

    def save(self, path: str):
        """
        Saves the current instance's status dictionary to a file.

        Parameters
        ----------
        path : str
            The path of the file to save to.

        """
        with open(path, "wb") as f:
            pickle.dump(dataclasses.asdict(self), f)

    def load(self, path: str):
        """
        Loads an instance's status dictionary from a file.

        Parameters
        ----------
        path : str
            The path of the file to load from.

        Raises
        ------
        ValueError
            If loaded file contains keys that are not in the dataclass fields.

        """
        with open(path, "rb") as f:
            status = pickle.load(f)
            self.update(**status)

    @property
    def dict(self) -> dict:
        """
        Returns the status dictionary.

        Returns
        -------
        dict
            The status dictionary.

        """
        return dataclasses.asdict(self)
