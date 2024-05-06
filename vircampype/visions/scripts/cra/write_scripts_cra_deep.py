import os
from glob import glob
from vircampype.visions.scripts.write_scripts import write_scripts

# Define paths
cloud_home = os.getenv("CLOUD_HOME")
path_scripts = f"{cloud_home}/Projects/VISIONS/Pipeline/scripts/CrA/deep/"
path_data = "/Volumes/Data/VISIONS/198C-2009E/data_deep/"
path_pype = "/Volumes/Data/VISIONS/198C-2009E/vircampype/"

# Find files
paths_files_j = sorted(glob(path_data + "CrA_deep_J*/*.fits"))
paths_files_h = sorted(glob(path_data + "CrA_deep_H*/*.fits"))
paths_files_ks = sorted(glob(path_data + "CrA_deep_Ks*/*.fits"))

# Common kwargs
kwargs = dict(
    additional_source_masks="Corona_Australis_deep",
    archive=False,
    build_phase3=False,
    build_public_catalog=True,
    build_stacks=False,
    build_tile=True,
    build_tile_only=False,
    path_pype=path_pype,
    path_scripts=path_scripts,
    projection="Corona_Australis_deep",
    n_jobs=10,
    external_headers=True,
    name_suffix=None,
    photometric_error_floor=0.005

)

# Write scripts
write_scripts(paths_files=paths_files_j, reference_mag_lo=12.5, reference_mag_hi=15.5,
              **kwargs)
write_scripts(paths_files=paths_files_h, reference_mag_lo=12.0, reference_mag_hi=15.0,
              **kwargs)
write_scripts(paths_files=paths_files_ks, reference_mag_lo=11.5, reference_mag_hi=14.5,
              **kwargs)
