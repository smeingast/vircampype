import glob
from vircampype.visions.mosaics.build_mosaic import build_mosaic

# Setup for pipeline
name = "CrA_mosaic_H_F"
path_visions = "/Users/stefan/iCloud/Projects/VISIONS/"
path_scripts = f"{path_visions}Pipeline/scripts/CrA/wide_F/"
path_master_astro_photo = f"{path_visions}Pipeline/master-astro-photo/CrA_wide/"
path_data = f"/Volumes/Data/Mosaics/CrA/{name}/"
path_pype = f"/Volumes/Data/Mosaics/CrA/vircampype/"
projection = "Corona_Australis_wide"
reference_mag_lim = (12.0, 15.0)  # H wide
photerr_internal = 0.005
n_jobs = 10

# Get script paths
paths_scripts = sorted(glob.glob(f"{path_scripts}*.yml"))

build_mosaic(
    name=name,
    paths_scripts=paths_scripts,
    path_data=path_data,
    path_pype=path_pype,
    n_jobs=n_jobs,
    path_master_astro_photo=path_master_astro_photo,
    reference_mag_lim=reference_mag_lim,
    projection=projection,
    photerr_internal=photerr_internal,
    build_public_catalog=True,
    source_classification=True,
)
