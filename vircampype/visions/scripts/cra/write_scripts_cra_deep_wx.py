from glob import glob
from vircampype.visions.scripts.write_scripts import write_scripts

# Define paths
path_scripts = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/CrA/deep_wx/"
path_data = "/Volumes/Data/VISIONS/198C-2009M_CrA/CrA_deep_wx_1/"
path_pype = "/Volumes/Data/VISIONS/198C-2009M_CrA/vircampype/"

# Search for files
paths_files = sorted(glob(path_data + "*.fits"))

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
write_scripts(paths_files=paths_files, reference_mag_lim=(12.0, 15.0), **kwargs)
