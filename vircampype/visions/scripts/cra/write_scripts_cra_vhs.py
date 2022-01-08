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

# Reference limits
reference_mag_lim_j = 13.0, 15.5
reference_mag_lim_ks = 12.0, 14.5

# Numer of parallel jobs
n_jobs = 16

# Projection
projection = "Corona_Australis_wide"
additional_source_masks = "Corona_Australis_wide"

# Archive
archive = False

# Headers
external_headers = True

# No stacks, phase 3, or classification
build_stacks = False
build_phase3 = False
build_class_star_library = False

# Generate scripts and write them to disk
for udj in unique_directories_j:
    name = "{0}_{1}".format(udj.split("data_vhs/")[1].split("/")[0], "J")
    setup = dict(
        name=name,
        path_data=udj,
        path_pype=path_pype,
        n_jobs=n_jobs,
        reference_mag_lim=reference_mag_lim_j,
        projection=projection,
        additional_source_masks=additional_source_masks,
        archive=archive,
        external_headers=external_headers,
        build_stacks=build_stacks,
        build_phase3=build_phase3,
        build_class_star_library=build_class_star_library,
    )

    # Write YML
    path_yml = "{0}{1}.yml".format(path_scripts_j, name)
    file = open(path_yml, "w")
    yaml.dump(setup, file)
    file.close()

for udks in unique_directories_ks:
    name = "{0}_{1}".format(udks.split("data_vhs/")[1].split("/")[0], "Ks")
    setup = dict(
        name=name,
        path_data=udks,
        path_pype=path_pype,
        n_jobs=n_jobs,
        reference_mag_lim=reference_mag_lim_ks,
        projection=projection,
        additional_source_masks=additional_source_masks,
        archive=archive,
        external_headers=external_headers,
        build_stacks=build_stacks,
        build_phase3=build_phase3,
        build_class_star_library=build_class_star_library,
    )

    # Write YML
    path_yml = "{0}{1}.yml".format(path_scripts_ks, name)
    file = open(path_yml, "w")
    yaml.dump(setup, file)
    file.close()
