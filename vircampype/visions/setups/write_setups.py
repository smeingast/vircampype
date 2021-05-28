import os
import glob
import yaml
from astropy.io import fits

# =========================================================================== #
# =========================================================================== #
# Setup
path_data = "/Volumes/Data/VISIONS/198C-2009A/data_control/"
path_scripts = "/Volumes/Data/VISIONS/198C-2009A/scripts/"
path_pype = "/Volumes/Data/VISIONS/198C-2009A/vircampype/"

files = glob.glob(path_data + "Oph*/*.fits")
projection = "Ophiuchus_Control"

reference_mag_lim = dict(J=(12.0, 15.0), H=(11.5, 14.5), Ks=(11.0, 14.0))
n_jobs = 12

# =========================================================================== #
# =========================================================================== #


# Get data directories
unique_directories = sorted(list(set([os.path.dirname(x) + "/" for x in files])))

# Generate setups and write them to disk
for udj in unique_directories:

    # Find passband in first file
    first_file = glob.glob("{0}*fits".format(udj))[0]
    passband = fits.getheader(first_file, 0)["HIERARCH ESO INS FILT1 NAME"]

    # Get Object name
    name = fits.getheader(first_file, 0)["OBJECT"]

    # Make setup
    setup = dict(name=name, path_data=udj, path_pype=path_pype, n_jobs=n_jobs,
                 reference_mag_lim=reference_mag_lim[passband], projection=projection)

    # Write YML
    path_yml = "{0}{1}.yml".format(path_scripts, name)
    file = open(path_yml, "w")
    yaml.dump(setup, file)
    file.close()
