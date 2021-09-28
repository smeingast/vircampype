import os
import glob

from vircampype.pipeline.main import Pipeline
from vircampype.tools.systemtools import read_yml
from vircampype.tools.miscellaneous import flat_list
from vircampype.tools.systemtools import make_folder, make_symlinks, copy_file


def build_mosaic(name, path_scripts, path_data, path_pype, path_master_astro_photo, n_jobs, additional_source_masks,
                 reference_mag_lim, projection, phase3_photerr_internal, **kwargs):

    # Check if master catalogs are available
    path_master_astro = "{0}MASTER-ASTROMETRY.fits.tab".format(path_master_astro_photo)
    path_master_photo = "{0}MASTER-PHOTOMETRY.fits.tab".format(path_master_astro_photo)
    if os.path.isfile(path_master_astro) * os.path.isfile(path_master_photo) != 1:
        raise ValueError("Master tables not found")

    # Print info
    paths_scripts = sorted(glob.glob(path_scripts + "*yml"))
    print("Found {0:2d} scripts".format(len(paths_scripts)))

    # Loop over scripts
    field_names, paths_folders_raw, paths_folders_resampled, paths_folders_statistics = [], [], [], []
    for ss in paths_scripts:
        yml = read_yml(ss)
        paths_folders_raw.append(yml["path_data"])
        paths_folders_resampled.append(yml["path_pype"] + yml["name"] + "/resampled/")
        paths_folders_statistics.append(yml["path_pype"] + yml["name"] + "/statistics/")
        field_names.append(yml["name"])

    # Pipeline setup
    setup = dict(name=name, path_data=path_data, path_pype=path_pype, n_jobs=n_jobs, projection=projection,
                 additional_source_masks=additional_source_masks, phase3_photerr_internal=phase3_photerr_internal,
                 reference_mag_lim=reference_mag_lim, external_headers=True, build_stacks=False, build_tile=True,
                 build_phase3=True, archive=False, qc_plots=True, **kwargs)

    # Find resampled images and create link paths
    paths_resampled = flat_list([glob.glob(p + "*resamp.fits") for p in paths_folders_resampled])
    links_resampled = ["{0}{1}/resampled/{2}".format(path_pype, name, os.path.basename(f)) for f in paths_resampled]
    print("Found {0:4d} resampled images".format(len(paths_resampled)))

    # Find resampled weights and create link paths
    paths_resampled_weights = flat_list([glob.glob(p + "*resamp.weight.fits") for p in paths_folders_resampled])
    links_resampled_weights = ["{0}{1}/resampled/{2}".format(path_pype, name, os.path.basename(f))
                               for f in paths_resampled_weights]
    print("Found {0:4d} resampled weights".format(len(paths_resampled_weights)))

    # Find statistics fikes
    paths_statistics = flat_list([glob.glob(p + "*.fits") for p in paths_folders_statistics])
    links_statistics = ["{0}{1}/statistics/{2}".format(path_pype, name, os.path.basename(f)) for f in paths_statistics]
    print("Found {0:4d} statistics files".format(len(paths_statistics)))

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
    make_symlinks(paths_resampled, links_resampled)
    make_symlinks(paths_resampled_weights, links_resampled_weights)
    make_symlinks(paths_statistics, links_statistics)

    # Set pipeline processing steps to completed
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
    pipe.status.illumcorr = True
    pipe.status.resampled = True
    pipe.status.build_statistics = True

    # Launch pipeline
    print("Launching pipeline")
    pipe.process_science()
