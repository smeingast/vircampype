import glob
import pickle

from vircampype.tools.messaging import *
from vircampype.pipeline.setup import Setup
from vircampype.pipeline.errors import PipelineError
from vircampype.fits.images.common import FitsImages
from vircampype.tools.fitstools import compress_images
from vircampype.tools.esotools import build_phase3_stacks, make_phase3_tile
from vircampype.tools.systemtools import run_commands_shell_parallel, clean_directory, remove_file


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

    def update_status(self, **kwargs):
        self.status.update(**kwargs)
        self.status.save(path=self.path_status)

    # =========================================================================== #
    # Raw files
    @property
    def raw(self):
        return FitsImages.from_folder(path=self.setup.folders["raw"],  pattern="*.fits",
                                      setup=self.setup, exclude=None)

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
        return ["{0}{1}.proc.basic{2}".format(self.setup.folders["processed_basic"], self.raw_science.names[i],
                                              self.raw_science.extensions[i]) for i in range(self.raw_science.n_files)]

    @property
    def processed_basic_science(self):

        # Instantiate
        from vircampype.fits.images.sky import SkyImagesProcessedScience
        images = SkyImagesProcessedScience(file_paths=self._paths_processed_basic_science, setup=self.setup)

        # Consistency check
        if len(images) != len(self.raw_science):
            raise PipelineError("Raw and processed science images not matching.")

        return images

    @property
    def _paths_processed_basic_offset(self):
        return ["{0}{1}.proc.basic{2}".format(self.setup.folders["processed_basic"], self.raw_offset.names[i],
                                              self.raw_offset.extensions[i]) for i in range(self.raw_offset.n_files)]

    @property
    def processed_basic_offset(self):

        # Return if no offset files
        if self.raw_offset is None:
            return None

        # Instantiate
        from vircampype.fits.images.sky import SkyImagesProcessedOffset
        images = SkyImagesProcessedOffset(file_paths=self._paths_processed_basic_offset, setup=self.setup)

        # Consistency check
        if len(images) != len(self.raw_offset):
            raise PipelineError("Raw and processed offset images not matching.")

        return images

    @property
    def processed_basic_science_and_offset(self):
        if self.processed_basic_offset is not None:
            return self.processed_basic_science + self.processed_basic_offset
        else:
            return self.processed_basic_science

    @property
    def _paths_processed_science_final(self):
        return ["{0}{1}.proc.final{2}".format(self.setup.folders["processed_final"], self.raw_science.names[i],
                                              self.raw_science.extensions[i]) for i in range(self.raw_science.n_files)]

    @property
    def processed_science_final(self):

        # Instantiate
        from vircampype.fits.images.sky import SkyImagesProcessedScience
        images = SkyImagesProcessedScience(file_paths=self._paths_processed_science_final, setup=self.setup)

        # Consistency check
        if len(images) != len(self.raw_science):
            raise PipelineError("Raw and processed science images not matching.")

        return images

    @property
    def _paths_illumination_corrected(self):
        return ["{0}{1}.proc.final.ic{2}".format(self.setup.folders["illumcorr"], self.raw_science.names[i],
                                                 self.raw_science.extensions[i])
                for i in range(self.raw_science.n_files)]

    @property
    def illumination_corrected(self):
        from vircampype.fits.images.sky import SkyImagesProcessedScience
        return SkyImagesProcessedScience.from_folder(path=self.setup.folders["illumcorr"],
                                                     pattern="*.proc.final.ic.fits", setup=self.setup, exclude=None)

    @property
    def _paths_resampled(self):
        return ["{0}{1}.proc.final.ic.resamp{2}"
                "".format(self.setup.folders["resampled"], self.raw_science.names[i],
                          self.raw_science.extensions[i]) for i in range(self.raw_science.n_files)]

    @property
    def resampled(self):

        # Instantiate
        from vircampype.fits.images.sky import SkyImagesResampled
        images = SkyImagesResampled(file_paths=self._paths_resampled, setup=self.setup)

        # Consistency check
        if len(images) != len(self.raw_science):
            raise PipelineError("Raw and resampled images not matching.")

        return images

    @property
    def _paths_stacks(self):
        return sorted(glob.glob("{0}*.stack.fits".format(self.setup.folders["stacks"])))

    @property
    def stacks(self):

        # Instantiate
        from vircampype.fits.images.sky import Stacks
        images = Stacks(file_paths=self._paths_stacks, setup=self.setup)

        # Consistency check
        if len(images) != 6:
            raise PipelineError("Stacks incomplete")

        return images

    @property
    def tile(self):
        from vircampype.fits.images.sky import Tile
        return Tile(setup=self.setup, file_paths=self.setup.path_coadd)

    # =========================================================================== #
    # Source catalogs
    @property
    def _paths_sources_processed_final_scamp(self):
        return [p.replace(".fits", ".scamp.fits.tab") for p in self._paths_processed_science_final]

    @property
    def sources_processed_final_scamp(self):

        # Instantiate
        from vircampype.fits.tables.sextractor import SextractorCatalogs
        catalogs = SextractorCatalogs(file_paths=self._paths_sources_processed_final_scamp, setup=self.setup)

        # Consistency check
        if len(catalogs) != len(self.processed_science_final):
            raise PipelineError("Images and catalogs not matching.")

        # Return
        return catalogs

    @property
    def _paths_sources_processed_illumcorr(self):
        return [p.replace(".fits", ".ic.fits.tab") for p in self._paths_processed_science_final]

    @property
    def sources_processed_illumcorr(self):

        # Insantiate
        from vircampype.fits.tables.sextractor import AstrometricCalibratedSextractorCatalogs
        catalogs = AstrometricCalibratedSextractorCatalogs(file_paths=self._paths_sources_processed_illumcorr,
                                                           setup=self.setup)

        # Consistency check
        if len(catalogs) != len(self.processed_science_final):
            raise PipelineError("Images and catalogs not matching.")

        # Return
        return catalogs

    @property
    def _paths_sources_stacks_full(self):
        return [p.replace(".fits", ".full.fits.tab") for p in self._paths_stacks]

    @property
    def sources_stacks_full(self):

        # Instantiate
        from vircampype.fits.tables.sextractor import AstrometricCalibratedSextractorCatalogs
        catalogs = AstrometricCalibratedSextractorCatalogs(file_paths=self._paths_sources_stacks_full, setup=self.setup)

        # Consistency check
        if len(catalogs) != len(self.stacks):
            raise PipelineError("Images and catalogs not matching.")

        # Return
        return catalogs

    @property
    def _paths_sources_stacks_crunched(self):
        return [p.replace(".fits.tab", ".fits.ctab") for p in self._paths_sources_stacks_full]

    @property
    def sources_stacks_crunched(self):

        # Instantiate
        from vircampype.fits.tables.sextractor import PhotometricCalibratedSextractorCatalogs
        catalogs = PhotometricCalibratedSextractorCatalogs(file_paths=self._paths_sources_stacks_crunched,
                                                           setup=self.setup)

        # Consistency check
        if len(catalogs) != len(self.stacks):
            raise PipelineError("Images and catalogs not matching.")

        # Return
        return catalogs

    @property
    def _path_sources_tile_full(self):
        return self.setup.path_coadd.replace(".fits", ".full.fits.tab")

    @property
    def sources_tile_full(self):

        # Instantiate
        from vircampype.fits.tables.sextractor import AstrometricCalibratedSextractorCatalogs
        catalog = AstrometricCalibratedSextractorCatalogs(file_paths=self._path_sources_tile_full, setup=self.setup)

        if len(catalog) != 1:
            raise PipelineError("Tile catalog not found")

        # Return
        return catalog

    @property
    def _path_sources_tile_crunched(self):
        return self.setup.path_coadd.replace(".fits", ".full.fits.ctab")

    @property
    def sources_tile_crunched(self):

        # Instantiate
        from vircampype.fits.tables.sextractor import PhotometricCalibratedSextractorCatalogs
        catalog = PhotometricCalibratedSextractorCatalogs(file_paths=self._path_sources_tile_crunched, setup=self.setup)

        if len(catalog) != 1:
            raise PipelineError("Crunched tile catalog not found")

        # Return
        return catalog

    def _paths_resampled_statistics(self, mode):
        return ["{0}{1}.{2}{3}".format(self.setup.folders["statistics"], self.resampled.names[i], mode,
                                       self.resampled.extensions[i]) for i in range(self.resampled.n_files)]

    def resampled_statistics(self, mode):

        # Instantiate
        from vircampype.fits.images.sky import SkyImagesResampled
        images = SkyImagesResampled(file_paths=self._paths_resampled_statistics(mode=mode), setup=self.setup)

        # Consistency check
        if len(images) != len(self.resampled):
            raise PipelineError("Image statistics incomplete")

        # Return
        return images

    # =========================================================================== #
    # Master calibration
    def build_master_bpm(self):
        if self.flat_lamp_check is not None:
            if not self.status.master_bpm:
                self.flat_lamp_check.build_master_bpm(darks=self.dark_check)
                self.update_status(master_bpm=True)
            else:
                print_message(message="MASTER-BPM already created", kind="warning", end=None)

    def build_master_dark(self):
        if self.dark_all is not None:
            if not self.status.master_dark:
                self.dark_all.build_master_dark()
                self.update_status(master_dark=True)
            else:
                print_message(message="MASTER-DARK already created", kind="warning", end=None)

    def build_master_gain(self):
        if self.flat_lamp_gain is not None:
            if not self.status.master_gain:
                self.flat_lamp_gain.build_master_gain(darks=self.dark_gain)
                self.update_status(master_gain=True)
            else:
                print_message(message="MASTER-GAIN already created", kind="warning", end=None)

    def build_master_linearity(self):
        if self.flat_lamp_lin is not None:
            if not self.status.master_linearity:
                self.flat_lamp_lin.build_master_linearity(darks=self.dark_lin)
                self.update_status(master_linearity=True)
            else:
                print_message(message="MASTER-LINEARITY already created", kind="warning", end=None)

    def build_master_flat(self):
        if self.flat_twilight is not None:
            if not self.status.master_flat:
                self.flat_twilight.build_master_flat()
                self.update_status(master_flat=True)
            else:
                print_message(message="MASTER-FLAT already created", kind="warning", end=None)

    def build_master_weight_global(self):
        if self.flat_twilight is not None:
            if not self.status.master_weight_global:
                self.flat_twilight.build_master_weight_global()
                self.update_status(master_weight_global=True)
            else:
                print_message(message="MASTER-WEIGHT-GLOBAL already created", kind="warning", end=None)

    def build_master_source_mask(self):
        if not self.status.master_source_mask:
            self.processed_basic_science_and_offset.build_master_source_mask()
            self.update_status(master_source_mask=True)
        else:
            print_message(message="MASTER-SOURCE-MASK already created", kind="warning", end=None)

    def build_master_sky_static(self):
        if not self.status.master_sky_static:
            self.processed_basic_science_and_offset.build_master_sky_static()
            self.update_status(master_sky_static=True)
        else:
            print_message(message="MASTER-SKY-STATIC already created", kind="warning", end=None)

    def build_master_sky_dynamic(self):

        if not self.status.master_sky_dynamic:

            # If mixed data is requested
            if self.setup.sky_mix_science:
                self.processed_basic_science_and_offset.build_master_sky_dynamic()

            # If no offset is present and mixing not requested, build from science
            elif self.processed_basic_offset is None:
                self.processed_basic_science.build_master_sky_dynamic()

            # Otherwise build only from offset
            else:
                self.processed_basic_offset.build_master_sky_dynamic()

            self.update_status(master_sky_dynamic=True)
        else:
            print_message(message="MASTER-SKY-DYNAMIC already created", kind="warning", end=None)

    def build_master_photometry(self):
        if self.raw_science is not None:
            if not self.status.master_photometry:
                self.processed_basic_science.build_master_photometry()
                self.update_status(master_photometry=True)
            else:
                print_message(message="MASTER-PHOTOMETRY already created", kind="warning", end=None)

    def build_master_astrometry(self):
        if not self.status.master_astrometry:
            self.processed_basic_science.build_master_astrometry()
            self.update_status(master_astrometry=True)
        else:
            print_message(message="MASTER-ASTROMETRY already created", kind="warning", end=None)

    def build_master_weight_image(self):
        if not self.status.master_weight_image:
            self.processed_science_final.build_master_weight_image()
            self.update_status(master_weight_image=True)
        else:
            print_message(message="MASTER WEIGHT IMAGE already built", kind="warning", end=None)

    # =========================================================================== #
    # Image processing
    def process_raw_basic(self):
        if not self.status.processed_raw_basic:
            self.raw_science_and_offset.process_raw_basic()
            self.update_status(processed_raw_basic=True)
        else:
            print_message(message="BASIC RAW PROCESSING already done", kind="warning", end=None)

    def process_science_final(self):
        if not self.status.processed_raw_final:
            self.processed_basic_science.process_raw_final()
            self.update_status(processed_raw_final=True)
        else:
            print_message(message="FINAL RAW PROCESSING already done", kind="warning", end=None)

    def calibrate_astrometry(self):
        if not self.status.astrometry:
            self.processed_science_final.sextractor(preset="scamp")
            # exit()
            self.sources_processed_final_scamp.scamp()
            self.update_status(astrometry=True)
        else:
            print_message(message="ASTROMETRY already calibrated", kind="warning", end=None)

    def illumination_correction(self):
        if not self.status.illumcorr:
            if not self.status.astrometry:
                raise ValueError("Astrometric calibration has to be completed before illumination correction")
            self.processed_science_final.sextractor(preset="ic")
            self.sources_processed_illumcorr.build_master_illumination_correction()
            self.processed_science_final.apply_illumination_correction()
            self.update_status(illumcorr=True)
        else:
            print_message(message="ILLUMINATION CORRECTION already applied", kind="warning", end=None)

    def build_coadd_header(self):
        if self.raw_science is not None:
            if not self.status.tile_header:
                self.processed_basic_science.build_coadd_header()
                self.update_status(tile_header=True)
            else:
                print_message(message="TILE HEADER already built", kind="warning", end=None)

    def resample(self):
        if not self.status.resampled:
            self.illumination_corrected.resample()
            self.update_status(resampled=True)
        else:
            print_message(message="PAWPRINT RESAMPLING already done", kind="warning", end=None)

    # =========================================================================== #
    # Image assembly
    def build_stacks(self):
        if not self.status.stacks:
            self.resampled.build_stacks()
            self.update_status(stacks=True)
        else:
            print_message(message="STACKS already built", kind="warning", end=None)

    def build_tile(self):
        if not self.status.tile:
            # self.resampled.equalize_zero_point(stack_catalogs=self.sources_stacks_crunched)
            self.resampled.build_tile()
            self.update_status(tile=True)
        else:
            print_message(message="TILE already created", kind="warning", end=None)

    # =========================================================================== #
    # Statistics
    def build_statistics_resampled(self):
        if not self.status.build_statistics:
            self.resampled.build_statistics()
            self.update_status(build_statistics=True)
        else:
            print_message(message="IMAGE STATISTICS already built", kind="warning", end=None)

    def build_statistics_stacks(self):
        if not self.status.statistics_stacks:
            for mode in ["mjdeff", "nimg", "exptime"]:
                images = self.resampled_statistics(mode=mode)
                images.coadd_statistics_stacks(mode=mode)
            self.update_status(statistics_stacks=True)
        else:
            print_message(message="STACKS STATISTICS already created", kind="warning", end=None)

    def build_statistics_tile(self):
        if not self.status.statistics_tile:
            for mode in ["mjdeff", "nimg", "exptime"]:
                images = self.resampled_statistics(mode=mode)
                images.coadd_statistics_tile(mode=mode)
            self.update_status(statistics_tile=True)
        else:
            print_message(message="TILE STATISTICS already created", kind="warning", end=None)

    # =========================================================================== #
    # Classification
    def classification_stacks(self):
        if not self.status.classification_stacks:
            self.stacks.build_class_star_library()
            self.update_status(classification_stacks=True)
        else:
            print_message(message="STACKS CLASSIFICATION library for already built", kind="warning", end=None)

    def classification_tile(self):
        if not self.status.classification_tile:
            self.tile.build_class_star_library()
            self.update_status(classification_tile=True)
        else:
            print_message(message="TILE CLASSIFICATION library for already built", kind="warning", end=None)

    # =========================================================================== #
    # Photometry
    def photometry_stacks(self):
        if not self.status.photometry_stacks:
            self.stacks.sextractor(preset="full")
            self.sources_stacks_full.add_statistics()
            self.sources_stacks_full.crunch_source_catalogs()
            self.update_status(photometry_stacks=True)
        else:
            print_message(message="STACKS PHOTOMETRY already done", kind="warning", end=None)

    def photometry_tile(self):
        if not self.status.photometry_tile:
            self.tile.sextractor(preset="full")
            self.sources_tile_full.add_statistics()
            self.sources_tile_full.crunch_source_catalogs()
            self.update_status(photometry_tile=True)
        else:
            print_message(message="TILE PHOTOMETRY already done", kind="warning", end=None)

    # =========================================================================== #
    # QC astrometry
    def qc_astrometry_stacks(self):
        if not self.status.qc_astrometry_stacks:
            self.sources_stacks_full.plot_qc_astrometry()
            self.update_status(qc_astrometry_stacks=True)
        else:
            print_message(message="QC ASTROMETRY STACKS already done", kind="warning", end=None)

    def qc_astrometry_tile(self):
        if not self.status.qc_astrometry_tile:
            self.sources_tile_full.plot_qc_astrometry()
            self.update_status(qc_astrometry_tile=True)
        else:
            print_message(message="QC ASTROMETRY TILE already done", kind="warning", end=None)

    # =========================================================================== #
    # Phase 3
    @property
    def _paths_phase3(self):
        return sorted(glob.glob("{0}*.fits".format(self.setup.folders["phase3"])))

    @property
    def _paths_phase3_images(self):
        return [x for x in self._paths_phase3 if "sources" not in x]

    def compress_phase3_images(self):

        # Maximum of three parallel jobs
        n_jobs = 3 if self.setup.n_jobs > 3 else self.setup.n_jobs

        # Compress files
        compress_images(self._paths_phase3_images, q=self.setup.fpack_quantization_factor, n_jobs=n_jobs)

    def phase3(self):
        if not self.status.phase3:
            photerr_internal = self.sources_stacks_crunched.photerr_internal()
            if self.setup.build_stacks:
                build_phase3_stacks(stacks_images=self.stacks, stacks_catalogs=self.sources_stacks_crunched,
                                    photerr_internal=photerr_internal)
            if self.setup.build_tile:
                make_phase3_tile(tile_image=self.tile, tile_catalog=self.sources_tile_crunched,
                                 pawprint_images=self.resampled, photerr_internal=photerr_internal)
            # if self.setup.compress_phase3:
            #     self.compress_phase3_images()
            self.update_status(phase3=True)
        else:
            print_message(message="PHASE 3 compliance already done", kind="warning", end=None)

    # =========================================================================== #
    # Cleaning/archiving
    def shallow_clean(self):
        """ Remove intermediate processing steps, but leave the most important files in the directory tree. """

        # Remove all extracted headers
        clean_directory(self.setup.folders["headers"])

        # Remove processed raw frames, their source catalogs, and the astrometric solution
        clean_directory(self.setup.folders["processed"])

        # Remove all data except maximasks from the illumination correction directory
        clean_directory(self.setup.folders["illumcorr"], pattern="*ic.ahead")
        clean_directory(self.setup.folders["illumcorr"], pattern="*ic.fits")

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
    # Pipeline processors
    def process_calibration(self):
        """ Sequentially build master calibration files. """

        # Master bad pixel mask
        self.build_master_bpm()

        # Master non-linearity tables
        self.build_master_linearity()

        # Master darks
        self.build_master_dark()

        # Master gain and read noise tables
        self.build_master_gain()

        # Master flat
        self.build_master_flat()

        # Global master weight
        self.build_master_weight_global()

    def process_science(self):
        """ Sequentially process science data. """

        # Start time
        t0 = print_start(obj=self.setup.name)

        # Basic processing
        self.process_raw_basic()

        # Build static sky, source masks, dynamic sky, and master photometry
        self.build_master_photometry()
        self.build_master_astrometry()
        self.build_master_sky_static()
        self.build_master_source_mask()
        self.build_master_sky_dynamic()

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
        self.build_statistics_resampled()

        # Build and calibrate stacks
        if self.setup.build_stacks:
            self.build_stacks()
            self.build_statistics_stacks()
            self.classification_stacks()
            self.photometry_stacks()
            self.qc_astrometry_stacks()

        # Build and calibrate tile
        if self.setup.build_tile:
            self.build_tile()
            self.build_statistics_tile()
            self.classification_tile()
            self.photometry_tile()
            self.qc_astrometry_tile()

        # Phase 3
        if self.setup.build_phase3:
            self.phase3()

        # Print finish message
        print_end(tstart=t0)


class PipelineStatus:
    def __init__(self, master_bpm=False, master_linearity=False, master_dark=False, master_gain=False,
                 master_flat=False, master_weight_global=False, processed_raw_basic=False, master_sky_static=False,
                 master_source_mask=False, master_sky_dynamic=False, master_photometry=False, master_astrometry=False,
                 processed_raw_final=False, master_weight_image=False, tile_header=False, astrometry=False,
                 illumcorr=False, resampled=False, build_statistics=False, stacks=False, statistics_stacks=False,
                 classification_stacks=False, photometry_stacks=False, qc_astrometry_stacks=False, tile=False,
                 statistics_tile=False, classification_tile=False, photometry_tile=False, qc_astrometry_tile=False,
                 phase3=False):

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
        self.build_statistics = build_statistics
        self.stacks = stacks
        self.statistics_stacks = statistics_stacks
        self.classification_stacks = classification_stacks
        self.photometry_stacks = photometry_stacks
        self.qc_astrometry_stacks = qc_astrometry_stacks
        self.tile = tile
        self.statistics_tile = statistics_tile
        self.classification_tile = classification_tile
        self.photometry_tile = photometry_tile
        self.qc_astrometry_tile = qc_astrometry_tile
        self.phase3 = phase3

    def __str__(self):
        return self.status_dict.__str__()

    def __repr__(self):
        return self.status_dict.__repr__()

    @staticmethod
    def __attributes():
        return ["master_bpm", "master_linearity", "master_dark", "master_gain", "master_flat", "master_weight_global",
                "processed_raw_basic", "master_sky_static", "master_source_mask", "master_sky_dynamic",
                "master_photometry", "master_astrometry", "processed_raw_final", "master_weight_image", "tile_header",
                "astrometry", "illumcorr", "resampled", "build_statistics", "stacks", "statistics_stacks",
                "classification_stacks", "photometry_stacks", "qc_astrometry_stacks", "tile", "statistics_tile",
                "classification_tile", "photometry_tile", "qc_astrometry_tile", "phase3"]

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
