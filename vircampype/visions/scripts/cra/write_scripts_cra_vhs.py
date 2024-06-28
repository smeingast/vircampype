import os
import glob
import yaml
from vircampype.visions.locations import path_data_pleiades, path_visions_proj

# Define paths
path_data = f"{path_data_pleiades}/VHS/CrA/data_vhs/"
path_pype = f"{path_data_pleiades}/VHS/CrA/vircampype/"
path_scripts_j = f"{path_visions_proj}/Pipeline/scripts/CrA/VHS_J/"
path_scripts_ks = f"{path_visions_proj}/Pipeline/scripts/CrA/VHS_Ks/"

# Find all files recursively
files_j = glob.glob(path_data + "**/J/*fits")
files_ks = glob.glob(path_data + "**/Ks/*fits")

# Get data directories
unique_directories_j = sorted(list(set([os.path.dirname(x) + "/" for x in files_j])))
unique_directories_ks = sorted(list(set([os.path.dirname(x) + "/" for x in files_ks])))

# Common kwargs
kwargs = dict(
    additional_source_masks="Corona_Australis_wide",
    archive=False,
    build_phase3=False,
    build_public_catalog=True,
    build_stacks=False,
    build_tile=True,
    build_tile_only=False,
    external_headers=True,
    n_jobs=16,
    projection="Corona_Australis_wide",
    source_classification=True,
)

# J
for udj in unique_directories_j:
    name = "{0}_{1}".format(udj.split("data_vhs/")[1].split("/")[0], "J")
    setup = dict(
        name=name,
        path_data=udj,
        path_pype=path_pype,
        reference_mag_lo=13.0,
        reference_mag_hi=15.5,
        **kwargs
    )

    # Write YML
    path_yml = f"{path_scripts_j}{name}.yml"
    file = open(path_yml, "w")
    yaml.dump(setup, file)
    file.close()

# Ks
for udks in unique_directories_ks:
    name = f"{udks.split('data_vhs/')[1].split('/')[0]}_{'Ks'}"
    setup = dict(
        name=name,
        path_data=udks,
        path_pype=path_pype,
        reference_mag_lo=12.0,
        reference_mag_hi=14.5,
        **kwargs
    )

    # Write YML
    path_yml = f"{path_scripts_ks}{name}.yml"
    file = open(path_yml, "w")
    yaml.dump(setup, file)
    file.close()
