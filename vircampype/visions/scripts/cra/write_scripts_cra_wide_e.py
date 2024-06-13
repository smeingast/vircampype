from glob import glob
from vircampype.visions.scripts.write_scripts import write_scripts
from vircampype.visions.locations import path_data_pleiades, path_visions_proj

# Define paths
path_data = f"{path_data_pleiades}VISIONS/198C-2009H/data_wide/"
path_pype = f"{path_data_pleiades}VISIONS/198C-2009H/vircampype/"
path_scripts = f"{path_visions_proj}Pipeline/scripts/CrA/wide_E/"

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
    name_suffix="_E",
    source_classification=False,
)

""" Four tiles are missing from this run. These were done later in run L. """

# Define paths
path_data = f"{path_data_pleiades}VISIONS/198C-2009L/data_wide/"
path_pype = f"{path_data_pleiades}VISIONS/198C-2009L/vircampype/"

# Search for files
paths_files = sorted(glob(path_data + "CrA*/*.fits"))

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
    name_suffix="_E",
    source_classification=False,
)
