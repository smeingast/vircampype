import os
import glob
import yaml

# Define base path
path_data = "/Volumes/Data/VISIONS/198C-2009E/data_deep/"
path_scripts = "/Volumes/Data/VISIONS/198C-2009E/scripts/"
path_pype = "/Volumes/Data/VISIONS/198C-2009E/vircampype/"

# Find all files recursively
files = glob.glob(path_data + "CrA*/*.fits")

# Get data directories
unique_directories = sorted(list(set([os.path.dirname(x) + "/" for x in files])))

# Reference limits
reference_mag_lim = dict(j=(12.0, 15.0), h=(11.5, 14.5), ks=(11.0, 14.0))

# Number of parallel jobs
n_jobs = 6

# Projection
projection = "Corona_Australis_deep"

# Generate setups and write them to disk
for udj in unique_directories:

    if "deep_j" in udj.lower():
        ref_lim = reference_mag_lim["j"]
    elif "deep_h" in udj.lower():
        ref_lim = reference_mag_lim["h"]
    elif "deep_k" in udj.lower():
        ref_lim = reference_mag_lim["ks"]
    else:
        raise ValueError

    name = udj.split("data_deep/")[1].split("/")[0]
    setup = dict(name=name, path_data=udj, path_pype=path_pype, n_jobs=n_jobs,
                 reference_mag_lim=ref_lim, projection=projection)

    # Write YML
    path_yml = "{0}{1}.yml".format(path_scripts, name)
    file = open(path_yml, "w")
    yaml.dump(setup, file)
    file.close()
