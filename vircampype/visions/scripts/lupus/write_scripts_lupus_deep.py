from glob import glob
from vircampype.visions.scripts.write_scripts import write_scripts

# Define paths
path_scripts = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/Lupus/deep/"
path_data = "/Volumes/Data/VISIONS/198C-2009E/data_deep/"
path_pype = "/Volumes/Data/VISIONS/198C-2009E/vircampype/"

# Search for files
paths_files_j_n = sorted(glob(path_data + "Lupus_deep_J_N*/*.fits"))
paths_files_j_s = sorted(glob(path_data + "Lupus_deep_J_S*/*.fits"))
paths_files_h_n = sorted(glob(path_data + "Lupus_deep_H_N*/*.fits"))
paths_files_h_s = sorted(glob(path_data + "Lupus_deep_H_S*/*.fits"))
paths_files_k_n = sorted(glob(path_data + "Lupus_deep_Ks_N*/*.fits"))
paths_files_k_s = sorted(glob(path_data + "Lupus_deep_Ks_S*/*.fits"))

# Common kwargs
kwargs = dict(
    path_pype=path_pype,
    path_scripts=path_scripts,
    archive=False,
    additional_source_masks="Lupus_deep",
    n_jobs=18,
    external_headers=True,
    phase3_photerr_internal=0.005,
    name_suffix=None,
    build_stacks=False,
    build_tile=True,
    build_phase3=False,
    build_class_star_library=False,
    name_from_directory=False,
)

# Write scripts for J North
write_scripts(
    paths_files=paths_files_j_n,
    projection="Lupus_deep_n",
    reference_mag_lim=(12.5, 15.5),
    **kwargs
)

# Write scripts for J South
write_scripts(
    paths_files=paths_files_j_s,
    projection="Lupus_deep_s",
    reference_mag_lim=(12.5, 15.5),
    **kwargs
)

# Write scripts for H North
write_scripts(
    paths_files=paths_files_h_n,
    projection="Lupus_deep_n",
    reference_mag_lim=(12.0, 15.0),
    **kwargs
)

# Write scripts for H South
write_scripts(
    paths_files=paths_files_h_s,
    projection="Lupus_deep_s",
    reference_mag_lim=(12.0, 15.0),
    **kwargs
)

# Write scripts for Ks North
write_scripts(
    paths_files=paths_files_k_n,
    projection="Lupus_deep_n",
    reference_mag_lim=(11.5, 14.5),
    **kwargs
)

# Write scripts for Ks South
write_scripts(
    paths_files=paths_files_k_s,
    projection="Lupus_deep_s",
    reference_mag_lim=(11.5, 14.5),
    **kwargs
)
