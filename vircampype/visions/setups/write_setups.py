import os
import glob
import yaml
from astropy.io import fits

# =========================================================================== #
# =========================================================================== #
# Setup
path_data = "/Volumes/Data/VISIONS/198C-2009E/data_control/"
path_scripts = "/Volumes/Data/VISIONS/198C-2009E/scripts/"
path_pype = "/Volumes/Data/VISIONS/198C-2009E/vircampype/"

files = glob.glob(path_data + "CrA*/*.fits")
projection = "Corona_Australis_control"
additional_source_masks = "Corona_Australis_control"
# additional_source_masks = None

reference_mag_lim = dict(J=(12.0, 15.5), H=(11.5, 15.0), Ks=(11.0, 14.5))
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
                 reference_mag_lim=reference_mag_lim[passband], projection=projection,
                 additional_source_masks=additional_source_masks)

    # Write YML
    path_yml = "{0}{1}.yml".format(path_scripts, name)
    file = open(path_yml, "w")
    yaml.dump(setup, file)
    file.close()
