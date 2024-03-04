from glob import glob
from vircampype.visions.scripts.write_scripts import write_scripts

# Define paths
path_data = "/Volumes/Data/VISIONS/198C-2009A/data_wide/"
path_pype = "/Volumes/Data/VISIONS/198C-2009A/vircampype/"
path_scripts = "/Users/stefan/iCloud/Projects/VISIONS/Pipeline/scripts/CrA/wide_A/"

# Search for files
paths_files = sorted(glob(path_data + "CrA*/A/*.fits"))

# Write scripts
write_scripts(
    paths_files=paths_files,
    additional_source_masks="Corona_Australis_wide",
    archive=False,
    build_phase3=False,
    build_public_catalog=False,
    build_stacks=False,
    build_tile=False,
    build_tile_only=False,
    path_pype=path_pype,
    path_scripts=path_scripts,
    projection="Corona_Australis_wide",
    n_jobs=18,
    external_headers=True,
    reference_mag_lo=12.0,
    reference_mag_hi=15.0,
    name_suffix="_A",
    source_classification=False,
)
