import os
from glob import glob
from vircampype.visions.scripts.write_scripts import write_scripts

# Define paths
cloud_home = os.environ["CLOUD_HOME"]
path_scripts = f"{cloud_home}/Projects/VISIONS/Pipeline/scripts/CrA/deep_wx/"
path_data = "/Volumes/Data/VISIONS/198C-2009M_CrA/CrA_deep_wx_1/"
path_pype = "/Volumes/Data/VISIONS/198C-2009M_CrA/vircampype/"

# Search for files
paths_files = sorted(glob(path_data + "*.fits"))

# Write scripts
write_scripts(
    paths_files=paths_files,
    additional_source_masks="Corona_Australis_wide",
    archive=False,
    build_phase3=False,
    build_public_catalog=True,
    build_stacks=False,
    build_tile=True,
    build_tile_only=False,
    path_pype=path_pype,
    path_scripts=path_scripts,
    projection="Corona_Australis_deep",
    n_jobs=18,
    external_headers=True,
    reference_mag_lo=12.0,
    reference_mag_hi=15.0,
    name_suffix=None,
    source_classification=True,
)
