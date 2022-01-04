from glob import glob
from vircampype.visions.scripts.write_scripts import write_scripts

# Define paths
path_scripts = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/CrA/CrA_deep/"
path_data = "/Volumes/Data/VISIONS/198C-2009E/data_deep/"
path_pype = "/Volumes/Data/VISIONS/198C-2009E/vircampype/"

# Write scripts for J
paths_files = sorted(glob(path_data + "CrA_deep_J*/*.fits"))
write_scripts(
    paths_files=paths_files,
    path_pype=path_pype,
    path_scripts=path_scripts,
    archive=True,
    projection="Corona_Australis_deep",
    additional_source_masks="Corona_Australis_deep",
    n_jobs=18,
    external_headers=True,
    reference_mag_lim=(12.5, 15.5),
    phase3_photerr_internal=0.005,
    name_suffix=None,
    build_stacks=True,
    build_tile=True,
    build_phase3=True,
)

# Write scripts for H
paths_files = sorted(glob(path_data + "CrA_deep_H*/*.fits"))
write_scripts(
    paths_files=paths_files,
    path_pype=path_pype,
    path_scripts=path_scripts,
    archive=True,
    projection="Corona_Australis_deep",
    additional_source_masks="Corona_Australis_deep",
    n_jobs=18,
    external_headers=True,
    reference_mag_lim=(12.0, 15.0),
    phase3_photerr_internal=0.005,
    name_suffix=None,
    build_stacks=True,
    build_tile=True,
    build_phase3=True,
)

# Write scripts for Ks
paths_files = sorted(glob(path_data + "CrA_deep_Ks*/*.fits"))
write_scripts(
    paths_files=paths_files,
    path_pype=path_pype,
    path_scripts=path_scripts,
    archive=True,
    projection="Corona_Australis_deep",
    additional_source_masks="Corona_Australis_deep",
    n_jobs=18,
    external_headers=True,
    reference_mag_lim=(11.5, 14.5),
    phase3_photerr_internal=0.005,
    name_suffix=None,
    build_stacks=True,
    build_tile=True,
    build_phase3=True,
)
