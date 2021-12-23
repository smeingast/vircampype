import glob
from vircampype.visions.mosaics.build_mosaic import build_mosaic

# Setup for pipeline
name = "Lupus_deep_J"
path_scripts = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/Lupus/deep/"
path_master_astro_photo = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/other/master-astro-photo/Lupus_deep/"
path_data = f"/Volumes/Data/Mosaics/Lupus/{name}/"
path_pype = f"/Volumes/Data/Mosaics/Lupus/vircampype/"
reference_mag_lim = (11.5, 14.5)
phase3_photerr_internal = 0.005
n_jobs = 10

"""
FIRST BUILD DS9 REGION MASKS BASED ON K BAND DATA, THEN RUN ALL TILES AGAIN, THEN BUILD MOSAIC
"""

# Get script paths
paths_scripts_n = sorted(glob.glob(f"{path_scripts}*_Ks_N_*.yml"))
paths_scripts_s = sorted(glob.glob(f"{path_scripts}*_Ks_S_*.yml"))

# Build mosaic for North field
build_mosaic(name=name, paths_scripts=paths_scripts_n, path_data=path_data, path_pype=path_pype, n_jobs=n_jobs,
             path_master_astro_photo=path_master_astro_photo, reference_mag_lim=reference_mag_lim,
             projection="Lupus_deep_n", phase3_photerr_internal=phase3_photerr_internal,
             build_class_star_library=False)

# Build mosaic for North field
build_mosaic(name=name, paths_scripts=paths_scripts_s, path_data=path_data, path_pype=path_pype, n_jobs=n_jobs,
             path_master_astro_photo=path_master_astro_photo, reference_mag_lim=reference_mag_lim,
             projection="Lupus_deep_s", phase3_photerr_internal=phase3_photerr_internal,
             build_class_star_library=False)
