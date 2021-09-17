import os
import glob

from vircampype.pipeline.main import Pipeline
from vircampype.tools.systemtools import read_yml
from vircampype.tools.miscellaneous import flat_list
from vircampype.tools.systemtools import make_folder, make_symlinks, copy_file


def build_mosaic(path_master_astro_photo, path_scripts, name, path_data, path_pype, n_jobs, projection,
                 additional_source_masks, phase3_photerr_internal, reference_mag_lim,
                 resize_header_before_resampling=False):

    # Check if master catalogs are available
    path_master_astro = "{0}MASTER-ASTROMETRY.fits.tab".format(path_master_astro_photo)
    path_master_photo = "{0}MASTER-PHOTOMETRY.fits.tab".format(path_master_astro_photo)
    if os.path.isfile(path_master_astro) * os.path.isfile(path_master_photo) != 1:
        raise ValueError("Master tables not found")

    # Print info
    paths_scripts = sorted(glob.glob(path_scripts + "*yml"))
    print("Found {0:2d} scripts".format(len(paths_scripts)))

    # Loop over scripts
    field_names, paths_folders_raw, paths_folders_proc, paths_folders_master = [], [], [], []
    for ss in paths_scripts:
        yml = read_yml(ss)
        paths_folders_raw.append(yml["path_data"])
        paths_folders_proc.append(yml["path_pype"] + yml["name"] + "/processed_final/")
        paths_folders_master.append(yml["path_pype"] + yml["name"] + "/master/")
        field_names.append(yml["name"])

    # Pipeline setup
    setup = dict(name=name, path_data=path_data, path_pype=path_pype, n_jobs=n_jobs, projection=projection,
                 additional_source_masks=additional_source_masks, phase3_photerr_internal=phase3_photerr_internal,
                 reference_mag_lim=reference_mag_lim, external_headers=True, build_stacks=False, build_tile=True,
                 build_phase3=True, archive=False, qc_plots=True,
                 resize_header_before_resampling=resize_header_before_resampling)

    # Find all raw input files
    paths_data_raw = flat_list([glob.glob(p + "*.fits") for p in paths_folders_raw])
    links_data_raw = ["{0}{1}".format(path_data, os.path.basename(p)) for p in paths_data_raw]
    print("Found {0:4d} raw images".format(len(paths_data_raw)))

    # Find processed images
    paths_data_proc = flat_list([glob.glob(p + "*.fits") for p in paths_folders_proc])
    print("Found {0:4d} processed images".format(len(paths_data_proc)))

    # Find Source masks
    paths_source_masks = flat_list([glob.glob(p + "MASTER-SOURCE-MASK*.fits") for p in paths_folders_master])
    print("Found {0:4d} master source masks".format(len(paths_source_masks)))

    # Find master weights
    paths_weights = flat_list([glob.glob(p + "MASTER-WEIGHT*.fits") for p in paths_folders_master])
    print("Found {0:4d} master weights".format(len(paths_weights)))

    # Safety check
    if len(paths_data_raw) != len(paths_data_proc) != len(paths_source_masks) != len(paths_weights):
        raise ValueError("Raw and processed images not matching")

    # Require input to continue
    if (input("Continue (Y/n)") or "Y") != "Y":
        exit()

    # Initialize Pipeline (to build folder tree)
    print("Initializing pipeline")
    make_folder(path_data)
    pipe = Pipeline(setup=setup)

    # Force header
    pipe.setup.projection.force_header = True

    # Copy master tables
    print("Copying master tables")
    copy_file(path_master_astro, "{0}{1}/master/".format(path_pype, name))
    copy_file(path_master_photo, "{0}{1}/master/".format(path_pype, name))

    # Create symbolic links for raw files
    print("Creating symbolic links")
    make_symlinks(paths_data_raw, links_data_raw)

    # Create symbolic links for processed data
    for fn in field_names:

        # All final processed files
        ff = glob.glob(path_pype + fn + "/processed_final/*")
        fl = ["{0}{1}/processed_final/{2}".format(path_pype, name, os.path.basename(f)) for f in ff]
        make_symlinks(ff, fl)

        # Source masks
        ff = glob.glob(path_pype + fn + "/master/MASTER-SOURCE-MASK*")
        fl = ["{0}{1}/master/{2}".format(path_pype, name, os.path.basename(f)) for f in ff]
        make_symlinks(ff, fl)

        # Weights
        ff = glob.glob(path_pype + fn + "/master/MASTER-WEIGHT-IMAGE*")
        fl = ["{0}{1}/master/{2}".format(path_pype, name, os.path.basename(f)) for f in ff]
        make_symlinks(ff, fl)

    # Set early processing steps to completed
    print("Setting pipeline status")
    pipe.status.processed_raw_basic = True
    pipe.status.master_photometry = True
    pipe.status.master_astrometry = True
    pipe.status.master_sky_static = True
    pipe.status.master_source_mask = True
    pipe.status.master_sky_dynamic = True
    pipe.status.processed_raw_final = True
    pipe.status.master_weight_image = True
    pipe.status.astrometry = True

    # Start processing
    print("Launching pipeline")
    pipe.process_science()
