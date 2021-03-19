import glob
import pickle

from vircampype.tools.messaging import *
from vircampype.pipeline.setup import Setup
from vircampype.fits.images.common import FitsImages
from vircampype.tools.systemtools import run_commands_shell_parallel
from vircampype.tools.systemtools import clean_directory, remove_file


class Pipeline:

    def __init__(self, setup, **kwargs):
        self.setup = Setup.load_pipeline_setup(setup, **kwargs)

        # Read status
        self.path_status = "{0}{1}".format(self.setup.folders["temp"], "pipeline_status.p")
        self.status = PipelineStatus()
        try:
            self.status.read(path=self.path_status)
        except FileNotFoundError:
            pass

    # =========================================================================== #
    def __str__(self):
        return self.status.__str__()

    def __repr__(self):
        return self.status.__repr__()

    # =========================================================================== #
    def update_status(self, path, **kwargs):
        self.status.update(**kwargs)
        self.status.save(path=path)

    # =========================================================================== #
    @property
    def raw(self):
        return FitsImages.from_folder(path=self.setup.folders["raw"],  pattern="*.fits",
                                      setup=self.setup, exclude=None)

    @property
    def paths_raw(self):
        return self.raw.paths_full

    _raw_split = None

    @property
    def raw_split(self):

        if self._raw_split is not None:
            return self._raw_split

        self._raw_split = self.raw.split_types()
        return self._raw_split

    @property
    def processed(self):
        from vircampype.fits.images.sky import ProcessedScienceImages
        return ProcessedScienceImages.from_folder(path=self.setup.folders["processed"],
                                                  pattern="*.proc.fits", setup=self.setup, exclude=None)

    @property
    def processed_sources_scamp(self):
        from vircampype.fits.tables.sextractor import SextractorCatalogs
        return SextractorCatalogs.from_folder(path=self.setup.folders["processed"],
                                              pattern="*.proc.scamp.fits.tab", setup=self.setup, exclude=None)

    @property
    def processed_sources_superflat(self):
        from vircampype.fits.tables.sextractor import AstrometricCalibratedSextractorCatalogs
        return AstrometricCalibratedSextractorCatalogs.from_folder(path=self.setup.folders["processed"],
                                                                   pattern="*.superflat.fits.tab",
                                                                   setup=self.setup, exclude=None)

    @property
    def superflatted(self):
        from vircampype.fits.images.sky import ProcessedScienceImages
        return ProcessedScienceImages.from_folder(path=self.setup.folders["superflat"],
                                                  pattern="*.proc.sf.fits", setup=self.setup, exclude=None)

    @property
    def resampled(self):
        from vircampype.fits.images.sky import ResampledScienceImages
        return ResampledScienceImages.from_folder(path=self.setup.folders["resampled"],
                                                  pattern="*.resamp.fits", setup=self.setup, exclude=None)

    @property
    def resampled_sources_full(self):
        from vircampype.fits.tables.sextractor import AstrometricCalibratedSextractorCatalogs
        return AstrometricCalibratedSextractorCatalogs.from_folder(path=self.setup.folders["resampled"],
                                                                   pattern="*full.fits.tab",
                                                                   setup=self.setup, exclude=None)

    @property
    def resampled_sources_crunched(self):
        from vircampype.fits.tables.sextractor import PhotometricCalibratedSextractorCatalogs
        return PhotometricCalibratedSextractorCatalogs.from_folder(path=self.setup.folders["resampled"],
                                                                   pattern="*full.fits.ctab",
                                                                   setup=self.setup, exclude=None)

    @property
    def tile(self):
        from vircampype.fits.images.sky import Tile
        return Tile(setup=self.setup, file_paths=self.setup.path_coadd)

    @property
    def tile_sources_full(self):
        from vircampype.fits.tables.sextractor import AstrometricCalibratedSextractorCatalogs
        return AstrometricCalibratedSextractorCatalogs.from_folder(path=self.setup.folders["tile"],
                                                                   pattern="*full.fits.tab",
                                                                   setup=self.setup, exclude=None)

    @property
    def tile_sources_crunched(self):
        from vircampype.fits.tables.sextractor import PhotometricCalibratedSextractorCatalogs
        return PhotometricCalibratedSextractorCatalogs.from_folder(path=self.setup.folders["tile"],
                                                                   pattern="*full.fits.ctab",
                                                                   setup=self.setup, exclude=None)

    # =========================================================================== #
    @property
    def science_raw(self):
        return self.raw_split["science"]

    @property
    def offset_raw(self):
        return self.raw_split["offset"]

    @property
    def std_raw(self):
        return self.raw_split["std"]

    @property
    def dark_science(self):
        return self.raw_split["dark_science"]
    
    @property
    def dark_lin(self):
        return self.raw_split["dark_lin"]

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
    def build_master_bpm(self):
        if self.flat_lamp_check is not None:
            if not self.status.master_bpm:
                self.flat_lamp_check.build_master_bpm()
                self.update_status(path=self.path_status, master_bpm=True)
            else:
                print_message(message="MASTER-BPM already created", kind="warning", end=None)

    def build_master_dark(self):
        if self.dark_all is not None:
            if not self.status.master_dark:
                self.dark_all.build_master_dark()
                self.update_status(path=self.path_status, master_dark=True)
            else:
                print_message(message="MASTER-DARK already created", kind="warning", end=None)

    def build_master_gain(self):
        if self.flat_lamp_gain is not None:
            if not self.status.master_gain:
                self.flat_lamp_gain.build_master_gain(darks=self.dark_gain)
                self.update_status(path=self.path_status, master_gain=True)
            else:
                print_message(message="MASTER-GAIN already created", kind="warning", end=None)

    def build_master_linearity(self):
        if self.flat_lamp_lin is not None:
            if not self.status.master_linearity:
                self.flat_lamp_lin.build_master_linearity()
                self.update_status(path=self.path_status, master_linearity=True)
            else:
                print_message(message="MASTER-LINEARITY already created", kind="warning", end=None)

    def build_master_flat(self):
        if self.flat_twilight is not None:
            if not self.status.master_flat:
                self.flat_twilight.build_master_flat()
                self.update_status(path=self.path_status, master_flat=True)
            else:
                print_message(message="MASTER-FLAT already created", kind="warning", end=None)

    def build_master_weight_global(self):
        if self.flat_twilight is not None:
            if not self.status.master_weight_global:
                self.flat_twilight.build_master_weight_global()
                self.update_status(path=self.path_status, master_weight_global=True)
            else:
                print_message(message="MASTER-WEIGHT-GLOBAL already created", kind="warning", end=None)

    def build_master_source_mask(self):
        if self.science_raw is not None:
            if not self.status.master_source_mask:
                self.science_raw.build_master_source_mask()
                self.update_status(path=self.path_status, master_source_mask=True)
            else:
                print_message(message="MASTER-SOURCE-MASK already created", kind="warning", end=None)

    def build_master_sky(self):
        if self.science_raw is not None:
            if not self.status.master_sky:
                # Mix offset frames with science frames if set
                if self.setup.sky_mix_science and (self.offset_raw is not None):
                    mixed = self.science_raw + self.offset_raw
                    mixed.build_master_sky()

                # If no offset is given, build from science frames
                elif self.offset_raw is None:
                    self.science_raw.build_master_sky()

                # Otherwise build only from Offset frames
                else:
                    self.offset_raw.build_master_sky()

                # Update pipeline status
                self.update_status(path=self.path_status, master_sky=True)

            else:
                print_message(message="MASTER-SKY already created", kind="warning", end=None)

    def build_master_photometry(self):
        if self.science_raw is not None:
            if not self.status.master_photometry:
                self.science_raw.build_master_photometry()
                self.update_status(path=self.path_status, master_photometry=True)
            else:
                print_message(message="MASTER-PHOTOMETRY already created", kind="warning", end=None)

    def build_coadd_header(self):
        if self.science_raw is not None:
            if not self.status.tile_header:
                self.science_raw.build_coadd_header()
                self.update_status(path=self.path_status, tile_header=True)
            else:
                print_message(message="TILE HEADER already built", kind="warning", end=None)

    def process_raw_science(self):
        if self.science_raw is not None:
            if not self.status.processed_raw:
                self.science_raw.process_raw()
                self.update_status(path=self.path_status, processed_raw=True)
            else:
                print_message(message="RAW PROCESSING already done", kind="warning", end=None)

    def calibrate_astrometry(self):
        if not self.status.astrometry:
            self.processed.sextractor(preset="scamp")
            self.processed_sources_scamp.scamp()
            self.update_status(path=self.path_status, astrometry=True)
        else:
            print_message(message="ASTROMETRY already calibrated", kind="warning", end=None)

    def superflat(self):
        if not self.status.superflat:
            if not self.status.astrometry:
                raise ValueError("Astrometric calibration has to be completed before superflat")
            self.processed.sextractor(preset="superflat")
            self.processed_sources_superflat.build_master_superflat()
            self.processed.apply_superflat()
            self.update_status(path=self.path_status, superflat=True)
        else:
            print_message(message="SUPERFLAT already applied", kind="warning", end=None)

    def build_master_weight_image(self):
        if not self.status.master_weight_image:
            self.superflatted.build_master_weight_image()
            self.update_status(path=self.path_status, master_weight_image=True)
        else:
            print_message(message="MASTER WEIGHT IMAGE already built", kind="warning", end=None)

    def resample(self):
        if not self.status.resampled:
            self.superflatted.resample()
            self.update_status(path=self.path_status, resampled=True)
        else:
            print_message(message="PAWPRINT RESAMPLING already done", kind="warning", end=None)

    def build_stacks(self):
        if not self.status.stacks:
            self.resampled.build_stacks()
            self.update_status(path=self.path_status, stacks=True)
        else:
            print_message(message="STACKS already built", kind="warning", end=None)

    def build_master_psf_pawprints(self):
        if not self.status.master_psf_pawprints:
            self.resampled.build_master_psf(preset="pawprints")
            self.update_status(path=self.path_status, master_psf_pawprints=True)
        else:
            print_message(message="PAWPRINT MASTER PSF already built", kind="warning", end=None)

    def classification_pawprints(self):
        if not self.status.classification_pawprints:
            self.resampled.build_class_star_library()
            self.update_status(path=self.path_status, classification_pawprints=True)
        else:
            print_message(message="PAWPRINT CLASSIFICATION library for already built", kind="warning", end=None)

    def classification_tile(self):
        if not self.status.classification_tile:
            self.tile.build_class_star_library()
            self.update_status(path=self.path_status, classification_tile=True)
        else:
            print_message(message="TILE CLASSIFICATION library for already built", kind="warning", end=None)

    def photometry_pawprints(self):
        if not self.status.photometry_pawprints:
            self.resampled.sextractor(preset="full")
            self.resampled_sources_full.crunch_source_catalogs()
            self.update_status(path=self.path_status, photometry_pawprints=True)
        else:
            print_message(message="PAWPRINT PHOTOMETRY already done", kind="warning", end=None)

    def build_tile(self):
        if not self.status.tile:
            self.resampled_sources_crunched.write_coadd_flux_scale()
            self.resampled.coadd_pawprints()
            self.update_status(path=self.path_status, tile=True)
        else:
            print_message(message="TILE already created", kind="warning", end=None)

    def build_tile_statistics(self):
        if not self.status.tile_statistics:
            self.resampled.build_tile_statistics()
            self.update_status(path=self.path_status, tile_statistics=True)
        else:
            print_message(message="TILE STATISTICS already created", kind="warning", end=None)

    def photometry_tile(self):
        if not self.status.photometry_tile:
            self.tile.sextractor(preset="full")
            self.tile_sources_full.add_statistics()
            self.tile_sources_full.crunch_source_catalogs()
            self.update_status(path=self.path_status, photometry_tile=True)
        else:
            print_message(message="TILE PHOTOMETRY already done", kind="warning", end=None)

    # =========================================================================== #
    def shallow_clean(self):
        """ Remove intermediate processing steps, but leave the most important files in the directory tree. """

        # Remove all extracted headers
        clean_directory(self.setup.folders["headers"])

        # Remove processed raw frames, their source catalogs, and the astrometric solution
        clean_directory(self.setup.folders["processed"])

        # Remove all data except maximasks from the superflat directory
        clean_directory(self.setup.folders["superflat"], pattern="*sf.ahead")
        clean_directory(self.setup.folders["superflat"], pattern="*sf.fits")

        # Remove intermediate source catalogs and coadd header
        clean_directory(self.setup.folders["resampled"], pattern="*.tab")
        clean_directory(self.setup.folders["resampled"], pattern="*ahead")

        # Remove temporary tables and additional headers from tile directory
        clean_directory(self.setup.folders["tile"], pattern="*.tab")
        clean_directory(self.setup.folders["tile"], pattern="*.ahead")

    def deepclean(self):
        """ Runs a shallow clean followed by deleting also the pipeline status"""
        self.shallow_clean()
        remove_file(filepath=self.path_status)

    def archive(self):
        """ First runs a deep clean and then compresses all remaining FITS images. """

        # Deep clean
        self.deepclean()

        # Find all remaining files
        fits_files = sorted(glob.glob(self.setup.folders["object"] + "/**/*.fits"))

        # Construct compression commands
        cmds = ["fpack -D -Y -q {0} {1}".format(self.setup.fpack_quantization_factor, f) for f in fits_files]

        # Run in parallel (maximum of 2 at a time)
        n_jobs = 1 if self.setup.n_jobs == 1 else 2
        run_commands_shell_parallel(cmds=cmds, n_jobs=n_jobs, silent=True)

    def unarchive(self):
        """ Uncompresses all compressed files. """

        # Find all compressed files
        fz_files = sorted(glob.glob(self.setup.folders["object"] + "/**/*.fz"))

        # Construct compression commands
        cmds = ["funpack -F {0}".format(f) for f in fz_files]

        # Run in parallel (maximum of 2 at a time)
        n_jobs = 1 if self.setup.n_jobs == 1 else 2
        run_commands_shell_parallel(cmds=cmds, n_jobs=n_jobs, silent=True)

    # =========================================================================== #
    def build_master_calibration(self):
        """ Sequentially build master calibration files. """
        self.build_master_bpm()
        self.build_master_dark()
        self.build_master_gain()
        self.build_master_linearity()
        self.build_master_flat()
        self.build_master_weight_global()
        self.build_master_source_mask()
        self.build_master_sky()
        self.build_master_photometry()

    def process_science(self):
        """ Sequentially process science data. """
        t0 = print_start(obj=self.setup.name)
        self.build_master_calibration()
        self.process_raw_science()
        self.calibrate_astrometry()
        self.superflat()
        self.build_master_weight_image()
        self.build_coadd_header()
        self.resample()
        self.build_stacks()
        self.classification_pawprints()
        self.photometry_pawprints()
        self.build_tile()
        self.build_tile_statistics()
        self.classification_tile()
        self.photometry_tile()
        print_end(tstart=t0)

    def phase3(self):

        # Import
        from vircampype.tools.esotools import make_phase3_pawprints, make_phase3_tile

        # Generate phase 3 comliant pawprints
        make_phase3_pawprints(pawprint_images=self.resampled, pawprint_catalogs=self.resampled_sources_crunched)
        make_phase3_tile(tile_image=self.tile, tile_catalog=self.tile_sources_crunched, pawprint_images=self.resampled)


class PipelineStatus:
    def __init__(self, master_bpm=False, master_dark=False, master_gain=False, master_linearity=False,
                 master_flat=False, master_weight_global=False, master_source_mask=False, master_sky=False,
                 processed_raw=False, astrometry=False, master_photometry=False, tile_header=False, superflat=False,
                 master_weight_image=False, resampled=False, stacks=False, master_psf_pawprints=False,
                 classification_pawprints=False, photometry_pawprints=False, classification_tile=False,
                 tile_statistics=False, tile=False, photometry_tile=False):

        # Set status attributes
        self.master_bpm = master_bpm
        self.master_dark = master_dark
        self.master_gain = master_gain
        self.master_linearity = master_linearity
        self.master_flat = master_flat
        self.master_weight_global = master_weight_global
        self.master_source_mask = master_source_mask
        self.master_sky = master_sky
        self.master_photometry = master_photometry
        self.tile_header = tile_header
        self.processed_raw = processed_raw
        self.astrometry = astrometry
        self.superflat = superflat
        self.master_weight_image = master_weight_image
        self.resampled = resampled
        self.stacks = stacks
        self.master_psf_pawprints = master_psf_pawprints
        self.classification_pawprints = classification_pawprints
        self.photometry_pawprints = photometry_pawprints
        self.classification_tile = classification_tile
        self.tile_statistics = tile_statistics
        self.tile = tile
        self.photometry_tile = photometry_tile

    def __str__(self):
        return self.status_dict.__str__()

    def __repr__(self):
        return self.status_dict.__repr__()

    @staticmethod
    def __attributes():
        return ["master_bpm", "master_dark", "master_gain", "master_linearity", "master_flat", "master_weight_global",
                "master_source_mask", "master_sky", "master_photometry", "tile_header", "processed_raw", "astrometry",
                "superflat", "master_weight_image", "resampled", "stacks", "master_psf_pawprints",
                "classification_pawprints", "photometry_pawprints", "classification_tile", "tile", "tile_statistics",
                "photometry_tile"]

    @property
    def status_dict(self):
        return {attr: getattr(self, attr) for attr in self.__attributes()}

    def update(self, **kwargs):
        for key, val in kwargs.items():
            if key not in self.__attributes():
                raise ValueError("Cannot set pipeline status for attribute '{0}'".format(key))
            else:
                setattr(self, key, val)

    def save(self, path):
        pickle.dump(self.status_dict, open(path, "wb"))

    def read(self, path):
        status = pickle.load(open(path, "rb"))
        for key, val in status.items():
            setattr(self, key, val)
