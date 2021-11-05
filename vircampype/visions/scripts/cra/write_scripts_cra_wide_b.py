from glob import glob
from vircampype.visions.scripts.write_scripts import write_scripts

# Define paths
path_data = "/Volumes/Data/VISIONS/198C-2009A/data_wide/"
path_pype = "/Volumes/Data/VISIONS/198C-2009A/vircampype/"
path_scripts = "/Users/stefan/Dropbox/Projects/VISIONS/scripts/VISIONS/CrA/CrA_wide_B/"

# Search for files
paths_files = sorted(glob(path_data + "CrA*/B/*.fits"))

# Write scripts
write_scripts(paths_files=paths_files, path_pype=path_pype, path_scripts=path_scripts, archive=False,
              projection="Corona_Australis_wide", additional_source_masks="Corona_Australis_wide", n_jobs=18,
              external_headers=True, reference_mag_lim=(12.0, 15.0), phase3_photerr_internal=0.005,
              name_suffix="_B", build_stacks=True, build_tile=True, build_phase3=True, build_class_star_library=False)

# Define paths again for compensation run
path_data = "/Volumes/Data/VISIONS/198C-2009B/data_wide/"
path_pype = "/Volumes/Data/VISIONS/198C-2009B/vircampype/"

# Search for files
paths_files = sorted(glob(path_data + "CrA*/B/*.fits"))

# Write scripts
write_scripts(paths_files=paths_files, path_pype=path_pype, path_scripts=path_scripts, archive=False,
              projection="Corona_Australis_wide", additional_source_masks="Corona_Australis_wide", n_jobs=18,
              external_headers=True, reference_mag_lim=(12.0, 15.0), phase3_photerr_internal=0.005,
              name_suffix="_B", build_stacks=True, build_tile=True, build_phase3=True, build_class_star_library=False)

