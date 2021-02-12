import os
import glob
import yaml

# Define base path
path_data = "/Volumes/Data/VISIONS/VHS/CrA/data_vhs/"
path_scripts = "/Volumes/Data/VISIONS/VHS/CrA/scripts/"

# Find all files recursively
files_j = glob.glob(path_data + "**/J/*fits")
files_ks = glob.glob(path_data + "**/Ks/*fits")

# Get data directories
unique_directories_j = sorted(list(set([os.path.dirname(x) + "/" for x in files_j])))
unique_directories_ks = sorted(list(set([os.path.dirname(x) + "/" for x in files_ks])))

# Reference limits
reference_mag_lim_j = 13.0, 16.0
reference_mag_lim_ks = 12.0, 15.0

path_pype = "/Volumes/Data/VISIONS/VHS/CrA/vircampype/"

# Generate setups and write them to disk
for udj in unique_directories_j:
    name = "{0}_{1}".format(udj.split("data_vhs/")[1].split("/")[0], "J")
    setup = dict(name=name, path_data=udj, path_pype=path_pype, reference_mag_lim=reference_mag_lim_j)

    # Write YML
    path_yml = "{0}{1}.yml".format(path_scripts, name)
    file = open(path_yml, "w")
    yaml.dump(setup, file)
    file.close()

for udks in unique_directories_ks:
    name = "{0}_{1}".format(udks.split("data_vhs/")[1].split("/")[0], "Ks")
    setup = dict(name=name, path_data=udks, path_pype=path_pype, reference_mag_lim=reference_mag_lim_ks)

    # Write YML
    path_yml = "{0}{1}.yml".format(path_scripts, name)
    file = open(path_yml, "w")
    yaml.dump(setup, file)
    file.close()
