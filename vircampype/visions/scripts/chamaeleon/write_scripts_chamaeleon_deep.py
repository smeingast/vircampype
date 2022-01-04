from glob import glob
from vircampype.visions.scripts.write_scripts import write_scripts

# Define paths
path_scripts = (
    "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/Chamaeleon/deep/"
)
path_data = "/Volumes/Data/VISIONS/198C-2009H/data_deep/"
path_pype = "/Volumes/Data/VISIONS/198C-2009H/vircampype/"

# Search for files
paths_files_j = sorted(glob(path_data + "Chamaeleon_deep_J*/*.fits"))
paths_files_h = sorted(glob(path_data + "Chamaeleon_deep_H*/*.fits"))
paths_files_k = sorted(glob(path_data + "Chamaeleon_deep_K*/*.fits"))

# Write scripts for J
write_scripts(
    paths_files=paths_files_j,
    path_pype=path_pype,
    path_scripts=path_scripts,
    archive=False,
    projection="Chamaeleon_deep",
    additional_source_masks=None,
    n_jobs=18,
    external_headers=True,
    reference_mag_lim=(12.5, 15.5),
    phase3_photerr_internal=0.005,
    name_suffix=None,
    build_stacks=False,
    build_tile=True,
    build_phase3=False,
    build_class_star_library=False,
    name_from_directory=False,
)

# Write scripts for H
write_scripts(
    paths_files=paths_files_h,
    path_pype=path_pype,
    path_scripts=path_scripts,
    archive=False,
    projection="Chamaeleon_deep",
    additional_source_masks=None,
    n_jobs=18,
    external_headers=True,
    reference_mag_lim=(12.0, 15.0),
    phase3_photerr_internal=0.005,
    name_suffix=None,
    build_stacks=False,
    build_tile=True,
    build_phase3=False,
    build_class_star_library=False,
    name_from_directory=False,
)

# Write scripts for Ks
write_scripts(
    paths_files=paths_files_k,
    path_pype=path_pype,
    path_scripts=path_scripts,
    archive=False,
    projection="Chamaeleon_deep",
    additional_source_masks=None,
    n_jobs=18,
    external_headers=True,
    reference_mag_lim=(11.5, 14.5),
    phase3_photerr_internal=0.005,
    name_suffix=None,
    build_stacks=False,
    build_tile=True,
    build_phase3=False,
    build_class_star_library=False,
    name_from_directory=False,
)
