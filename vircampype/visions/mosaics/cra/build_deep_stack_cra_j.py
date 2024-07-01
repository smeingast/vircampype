import glob
from vircampype.visions.mosaics.build_mosaic import build_mosaic
from vircampype.visions.locations import (
    path_data_pleiades,
    path_visions_proj,
    path_data_betelgeuse,
)

# Setup for pipeline
name = "CrA_deep_J"
dir_scripts = f"{path_visions_proj}Pipeline/scripts/CrA/deep/"
dir_master_astro_photo = f"{path_visions_proj}/Pipeline/master-astro-photo/CrA_deep/"
path_master_astro = f"{dir_master_astro_photo}MASTER-ASTROMETRY.fits.tab"
path_master_photo = f"{dir_master_astro_photo}MASTER-PHOTOMETRY.fits.tab"
path_data = f"{path_data_pleiades}/Mosaics/CrA/{name}/"
path_pype = f"{path_data_pleiades}/Mosaics/CrA/vircampype/"
projection = "Corona_Australis_deep"
reference_mag_lo = 12.5
reference_mag_hi = 15.5
photometric_error_floor = 0.005
n_jobs = 10
detection_image_path = f"{path_data_betelgeuse}/CrA/Images_DeNeb/CrA_deep_J.clean.fits"

# Get script paths
paths_scripts = sorted(glob.glob(f"{dir_scripts}*_J_*.yml"))

# Build mosaic
build_mosaic(
    name=name,
    paths_scripts=paths_scripts,
    path_data=path_data,
    path_pype=path_pype,
    path_master_astro=path_master_astro,
    path_master_photo=path_master_photo,
    n_jobs=n_jobs,
    reference_mag_lo=reference_mag_lo,
    reference_mag_hi=reference_mag_hi,
    projection=projection,
    photometric_error_floor=photometric_error_floor,
    build_public_catalog=True,
    source_classification=True,
    sex_detection_image_path=detection_image_path,
)
