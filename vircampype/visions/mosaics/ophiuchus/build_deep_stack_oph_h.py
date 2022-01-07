import glob
from vircampype.visions.mosaics.build_mosaic import build_mosaic

# Setup for pipeline
name = "Ophiuchus_deep_H"
path_pipe = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/"
path_scripts = f"{path_pipe}scripts/Ophiuchus/deep/"
path_master_astro_photo = f"{path_pipe}other/master-astro-photo/Ophiuchus_deep/"
path_data = f"/Volumes/Data/Mosaics/Ophiuchus/{name}/"
path_pype = f"/Volumes/Data/Mosaics/Ophiuchus/vircampype/"
projection = "Ophiuchus_deep"
reference_mag_lim = (12.0, 15.0)
phase3_photerr_internal = 0.005
n_jobs = 10

# Get script paths
paths_scripts = sorted(glob.glob(f"{path_scripts}*_H_*.yml"))

# Build mosaic
build_mosaic(
    name=name,
    paths_scripts=paths_scripts,
    path_data=path_data,
    path_pype=path_pype,
    n_jobs=n_jobs,
    path_master_astro_photo=path_master_astro_photo,
    reference_mag_lim=reference_mag_lim,
    projection=projection,
    phase3_photerr_internal=phase3_photerr_internal,
    build_class_star_library=False,
    build_phase3=False,
)
