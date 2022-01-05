import os

from glob import glob
from astropy.io import fits
from vircampype.pipeline.main import Pipeline
from vircampype.tools.systemtools import read_yml
from vircampype.tools.miscellaneous import flat_list
from vircampype.tools.systemtools import make_folder, make_symlinks, copy_file


def build_mosaic(
    name,
    paths_scripts,
    path_data,
    path_pype,
    path_master_astro_photo,
    n_jobs,
    reference_mag_lim,
    projection,
    phase3_photerr_internal,
    build_phase3=True,
    **kwargs,
):

    # Check if master catalogs are available
    path_master_astro = f"{path_master_astro_photo}MASTER-ASTROMETRY.fits.tab"
    path_master_photo = f"{path_master_astro_photo}MASTER-PHOTOMETRY.fits.tab"
    if os.path.isfile(path_master_astro) * os.path.isfile(path_master_photo) != 1:
        raise ValueError("Master tables not found")

    # Check if scripts are there
    for ps in paths_scripts:
        if not os.path.isfile(ps):
            raise ValueError(f"Script '{os.path.basename(ps)}' does not exist")

    # Print info
    print(f"Found {len(paths_scripts):2d} scripts")

    # Loop over scripts
    (
        field_names,
        paths_folders_raw,
        paths_folders_resampled,
        paths_folders_statistics,
    ) = ([], [], [], [])
    for ss in paths_scripts:
        yml = read_yml(ss)
        paths_folders_raw.append(yml["path_data"])
        paths_folders_resampled.append(yml["path_pype"] + yml["name"] + "/resampled/")
        paths_folders_statistics.append(yml["path_pype"] + yml["name"] + "/statistics/")
        field_names.append(yml["name"])

    # Pipeline setup
    setup = dict(
        name=name,
        path_data=path_data,
        path_pype=path_pype,
        n_jobs=n_jobs,
        projection=projection,
        phase3_photerr_internal=phase3_photerr_internal,
        reference_mag_lim=reference_mag_lim,
        external_headers=True,
        build_stacks=False,
        build_tile=True,
        build_phase3=build_phase3,
        archive=False,
        qc_plots=True,
        **kwargs,
    )

    # Find raw images and keep only science frames
    paths_raw = flat_list(sorted([glob(p + "*.fits") for p in paths_folders_raw]))
    temp = [
        "SKY" not in fits.getheader(pr)["HIERARCH ESO DPR TYPE"] for pr in paths_raw
    ]
    paths_raw = [pr for pr, tt in zip(paths_raw, temp) if tt]
    links_raw = [f"{path_data}/{os.path.basename(f)}" for f in paths_raw]
    print(f"Found {len(paths_raw):4d} raw images")

    # Find resampled images and create link paths
    paths_resampled = flat_list(
        sorted([glob(p + "*resamp.fits") for p in paths_folders_resampled])
    )
    links_resampled = [
        f"{path_pype}{name}/resampled/{os.path.basename(f)}" for f in paths_resampled
    ]
    print(f"Found {len(paths_resampled):4d} resampled images")

    # Dummy check
    if len(paths_raw) != len(paths_resampled):
        raise ValueError("Raw and resampled images not matching")

    # Find resampled weights and create link paths
    paths_resampled_weights = flat_list(
        sorted([glob(p + "*resamp.weight.fits") for p in paths_folders_resampled])
    )
    links_resampled_weights = [
        f"{path_pype}{name}/resampled/{os.path.basename(f)}"
        for f in paths_resampled_weights
    ]
    print(f"Found {len(paths_resampled_weights):4d} resampled weights")

    # Find statistics files
    paths_statistics = flat_list(
        sorted([glob(p + "*.fits") for p in paths_folders_statistics])
    )
    links_statistics = [
        f"{path_pype}{name}/statistics/{os.path.basename(f)}" for f in paths_statistics
    ]
    print(f"Found {len(paths_statistics):4d} statistics files")

    # Dummy check
    if len(paths_statistics) != 5 * len(paths_resampled):
        raise ValueError("Statistics and image files not matching")

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
    copy_file(path_master_astro, f"{path_pype}{name}/master/")
    copy_file(path_master_photo, f"{path_pype}{name}/master/")

    # Create symbolic links
    print("Creating symbolic links")
    make_symlinks(paths_raw, links_raw)
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
