import os
import glob
import yaml

# Define base path
path_data = "/Volumes/Data/VISIONS/198C-2009A/data_wide/"
path_scripts = "/Volumes/Data/VISIONS/198C-2009A/scripts/"
path_pype = "/Volumes/Data/VISIONS/198C-2009A/vircampype/"

# Find all files recursively
files = glob.glob(path_data + "CrA*/**/*.fits")

# Get data directories
unique_directories = sorted(list(set([os.path.dirname(x) + "/" for x in files])))

# Reference limits
reference_mag_lim = 11.5, 14.5
n_jobs = 12

# Generate setups and write them to disk
for udj in unique_directories:
    name = "{0}_{1}".format(udj.split("data_wide/")[1].split("/")[0], udj[-2])
    setup = dict(name=name, path_data=udj, path_pype=path_pype, n_jobs=n_jobs,
                 reference_mag_lim=reference_mag_lim, projection="Corona_Australis_wide")

    # Write YML
    path_yml = "{0}{1}.yml".format(path_scripts, name)
    file = open(path_yml, "w")
    yaml.dump(setup, file)
    file.close()
