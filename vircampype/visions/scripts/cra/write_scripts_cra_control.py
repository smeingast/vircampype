from glob import glob
from vircampype.visions.scripts.write_scripts import write_scripts

# Define paths
path_scripts = "/Users/stefan/Dropbox/Projects/VISIONS/scripts/VISIONS/CrA/CrA_control/"
path_data = "/Volumes/Data/VISIONS/198C-2009E/data_control/"
path_pype = "/Volumes/Data/VISIONS/198C-2009E/vircampype/"

# Search for files
paths_files = sorted(glob(path_data + "CrA_control*/*.fits"))

# Write scripts
write_scripts(paths_files=paths_files, path_pype=path_pype, path_scripts=path_scripts, archive=True,
              projection="Corona_Australis_control", additional_source_masks="Corona_Australis_control", n_jobs=18,
              external_headers=True, reference_mag_lim=(12.5, 15.5), phase3_photerr_internal=0.005,
              name_suffix=None, build_stacks=True, build_tile=True, build_phase3=True)
