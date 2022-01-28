from glob import glob
from vircampype.visions.scripts.write_scripts import write_scripts

# Define paths
path_scripts = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/CrA/wide_F/"
path_data = "/Volumes/Data/VISIONS/198C-2009H/data_wide/"

# Search for files
paths_files_1 = sorted(glob(path_data + "CrA*/B/*.fits"))

# Common kwargs
kwargs = dict(
    path_pype="/Volumes/Data/VISIONS/198C-2009H/vircampype/",
    path_scripts=path_scripts,
    archive=False,
    projection="Corona_Australis_wide",
    additional_source_masks="Corona_Australis_wide",
    n_jobs=18,
    external_headers=True,
    reference_mag_lim=(12.0, 15.0),
    phase3_photerr_internal=0.005,
    name_suffix="_F",
    build_stacks=True,
    build_tile=True,
    build_phase3=True,
    build_public_catalog=False,
)

# Write scripts
write_scripts(paths_files=paths_files_1, **kwargs)

""" CrA_wide_1_5_3/B has grade C. Was repeated a few days later in run I """
# Define paths
path_data = "/Volumes/Data/VISIONS/198C-2009I/data_wide/"
kwargs["path_pype"] = "/Volumes/Data/VISIONS/198C-2009I/vircampype/"

# Search for files
paths_files_2 = sorted(glob(path_data + "CrA*/*.fits"))

# Write scripts
write_scripts(paths_files=paths_files_2, **kwargs)
