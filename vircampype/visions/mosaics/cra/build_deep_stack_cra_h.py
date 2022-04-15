import glob
from vircampype.visions.mosaics.build_mosaic import build_mosaic

# Setup for pipeline
name = "CrA_deep_H"
path_scripts = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/CrA/deep/"
path_master_astro_photo = (
    "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/other/master-astro-photo/CrA_deep/"
)
path_data = f"/Volumes/Data/Mosaics/CrA/{name}/"
path_pype = f"/Volumes/Data/Mosaics/CrA/vircampype/"

# Get script paths
paths_scripts = sorted(glob.glob(f"{path_scripts}*_H_*.yml"))

# Build mosaic
build_mosaic(
    name=name,
    paths_scripts=paths_scripts,
    path_data=path_data,
    path_pype=path_pype,
    path_master_astro_photo=path_master_astro_photo,
    n_jobs=10,
    reference_mag_lim=(12.0, 15.0),
    projection="Corona_Australis_deep",
    photerr_internal=0.005,
    build_public_catalog=True,
)
