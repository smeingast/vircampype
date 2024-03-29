import os
import glob
import yaml

# Define base path
path_data = "/Volumes/Data/VHS/CrA/data_vhs/"
path_pype = "/Volumes/Data/VHS/CrA/vircampype/"
path_scripts_j = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/CrA/VHS_J/"
path_scripts_ks = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/CrA/VHS_Ks/"

# Find all files recursively
files_j = glob.glob(path_data + "**/J/*fits")
files_ks = glob.glob(path_data + "**/Ks/*fits")

# Get data directories
unique_directories_j = sorted(list(set([os.path.dirname(x) + "/" for x in files_j])))
unique_directories_ks = sorted(list(set([os.path.dirname(x) + "/" for x in files_ks])))

# Common kwargs
kwargs = dict(
    n_jobs=16,
    projection="Corona_Australis_wide",
    additional_source_masks="Corona_Australis_wide",
    archive=False,
    external_headers=True,
    build_stacks=False,
    build_tile=True,
    build_tile_only=True,
    build_phase3=False,
    build_class_star_library=False,
)

# J
for udj in unique_directories_j:
    name = "{0}_{1}".format(udj.split("data_vhs/")[1].split("/")[0], "J")
    setup = dict(
        name=name,
        path_data=udj,
        path_pype=path_pype,
        reference_mag_lim=(13.0, 15.5),
        **kwargs
    )

    # Write YML
    path_yml = "{0}{1}.yml".format(path_scripts_j, name)
    file = open(path_yml, "w")
    yaml.dump(setup, file)
    file.close()

# Ks
for udks in unique_directories_ks:
    name = "{0}_{1}".format(udks.split("data_vhs/")[1].split("/")[0], "Ks")
    setup = dict(
        name=name,
        path_data=udks,
        path_pype=path_pype,
        reference_mag_lim=(12.0, 14.5),
        **kwargs
    )

    # Write YML
    path_yml = "{0}{1}.yml".format(path_scripts_ks, name)
    file = open(path_yml, "w")
    yaml.dump(setup, file)
    file.close()
