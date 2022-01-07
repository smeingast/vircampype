import glob
from vircampype.visions.mosaics.build_mosaic import build_mosaic

# Setup for pipeline
name = "Chamaeleon_deep_J"
path_scripts = (
    "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/Chamaeleon/deep/"
)
path_master_astro_photo = (
    "/Users/stefan/Dropbox/Projects/VISIONS/"
    "Pipeline/other/master-astro-photo/Chamaeleon_deep/"
)
path_data = f"/Volumes/Data/Mosaics/Chamaeleon/{name}/"
path_pype = f"/Volumes/Data/Mosaics/Chamaeleon/vircampype/"
reference_mag_lim = (12.5, 15.5)
phase3_photerr_internal = 0.005
n_jobs = 10

# Get script paths
paths_scripts = sorted(glob.glob(f"{path_scripts}*_J_*.yml"))

# Build mosaic
build_mosaic(
    name=name,
    paths_scripts=paths_scripts,
    path_data=path_data,
    projection="Chamaeleon_deep",
    path_pype=path_pype,
    n_jobs=n_jobs,
    path_master_astro_photo=path_master_astro_photo,
    reference_mag_lim=reference_mag_lim,
    phase3_photerr_internal=phase3_photerr_internal,
    build_class_star_library=False,
    build_phase3=False,
)
