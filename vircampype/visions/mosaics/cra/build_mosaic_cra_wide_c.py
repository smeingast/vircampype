import glob
from vircampype.visions.mosaics.build_mosaic import build_mosaic
from vircampype.visions.locations import path_data_pleiades, path_visions_proj

# Setup for pipeline
name = "CrA_mosaic_H_C"
path_scripts = f"{path_visions_proj}Pipeline/scripts/CrA/wide_C/"
path_master_astro_photo = f"{path_visions_proj}Pipeline/master-astro-photo/CrA_wide/"
path_master_astro = f"{path_master_astro_photo}MASTER-ASTROMETRY.fits.tab"
path_master_photo = f"{path_master_astro_photo}MASTER-PHOTOMETRY.fits.tab"
path_data = f"{path_data_pleiades}Mosaics/CrA/{name}/"
path_pype = f"{path_data_pleiades}Mosaics/CrA/vircampype/"
projection = "Corona_Australis_wide"
reference_mag_lo = 12.0
reference_mag_hi = 15.0
photometric_error_floor = 0.005
n_jobs = 10

# Get script paths
paths_scripts = sorted(glob.glob(f"{path_scripts}*.yml"))

# Build mosaic
build_mosaic(
    name=name,
    paths_scripts=paths_scripts,
    path_data=path_data,
    path_pype=path_pype,
    n_jobs=n_jobs,
    path_master_astro=path_master_astro,
    path_master_photo=path_master_photo,
    reference_mag_lo=reference_mag_lo,
    reference_mag_hi=reference_mag_hi,
    projection=projection,
    photometric_error_floor=photometric_error_floor,
    build_public_catalog=True,
    source_classification=True,
)
