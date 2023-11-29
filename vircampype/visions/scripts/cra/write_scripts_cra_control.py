from glob import glob
from vircampype.visions.scripts.write_scripts import write_scripts

# Define paths
path_scripts = "/Users/stefan/MyCloud/Projects/VISIONS/Pipeline/scripts/CrA/control/"
path_data = "/Volumes/Data/VISIONS/198C-2009E/data_control/"
path_pype = "/Volumes/Data/VISIONS/198C-2009E/vircampype/"

# Search for files
paths_files_j = sorted(glob(path_data + "CrA_control_J/*.fits"))
paths_files_h = sorted(glob(path_data + "CrA_control_H/*.fits"))
paths_files_ks = sorted(glob(path_data + "CrA_control_Ks/*.fits"))

# Common kwargs
kwargs = dict(
    path_pype=path_pype,
    path_scripts=path_scripts,
    archive=False,
    projection="Corona_Australis_control",
    additional_source_masks="Corona_Australis_control",
    n_jobs=18,
    external_headers=True,
    photometric_error_floor=0.005,
    name_suffix=None,
    build_stacks=True,
    build_tile=True,
    build_phase3=True,
    build_public_catalog=False,
)

# Write scripts
write_scripts(
    paths_files=paths_files_j, reference_mag_lo=12.5, reference_mag_hi=15.5, **kwargs
)
write_scripts(
    paths_files=paths_files_h, reference_mag_lo=12.0, reference_mag_hi=15.0, **kwargs
)
write_scripts(
    paths_files=paths_files_ks, reference_mag_lo=11.5, reference_mag_hi=14.5, **kwargs
)
