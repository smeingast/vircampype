from glob import glob
from vircampype.visions.scripts.write_scripts import write_scripts

# Define paths
path_scripts = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/CrA/deep/"
path_data = "/Volumes/Data/VISIONS/198C-2009E/data_deep/"
path_pype = "/Volumes/Data/VISIONS/198C-2009E/vircampype/"

# Find files
paths_files_j = sorted(glob(path_data + "CrA_deep_J*/*.fits"))
paths_files_h = sorted(glob(path_data + "CrA_deep_H*/*.fits"))
paths_files_ks = sorted(glob(path_data + "CrA_deep_Ks*/*.fits"))

# Common kwargs
kwargs = dict(
    path_pype=path_pype,
    path_scripts=path_scripts,
    archive=False,
    projection="Corona_Australis_deep",
    additional_source_masks="Corona_Australis_deep",
    n_jobs=18,
    external_headers=True,
    phase3_photerr_internal=0.005,
    name_suffix=None,
    build_stacks=True,
    build_tile=True,
    build_phase3=True,
    build_public_catalog=False,
)

# Write scripts
write_scripts(paths_files=paths_files_j, reference_mag_lim=(12.5, 15.5), **kwargs)
write_scripts(paths_files=paths_files_h, reference_mag_lim=(12.0, 15.0), **kwargs)
write_scripts(paths_files=paths_files_ks, reference_mag_lim=(11.5, 14.5), **kwargs)
