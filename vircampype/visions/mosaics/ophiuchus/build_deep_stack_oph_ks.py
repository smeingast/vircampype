import glob
from vircampype.visions.mosaics.build_mosaic import build_mosaic

# Setup for pipeline
name = "Ophiuchus_deep_Ks"
path_pipe = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/"
path_scripts = f"{path_pipe}scripts/Ophiuchus/deep/"
path_master_astro_photo = f"{path_pipe}other/master-astro-photo/Ophiuchus_deep/"
path_data = f"/Volumes/Data/Mosaics/Ophiuchus/{name}/"
path_pype = f"/Volumes/Data/Mosaics/Ophiuchus/vircampype/"

# Get script paths
paths_scripts = sorted(glob.glob(f"{path_scripts}*_K_*.yml"))

# Build mosaic
build_mosaic(
    name=name,
    paths_scripts=paths_scripts,
    path_data=path_data,
    path_pype=path_pype,
    path_master_astro_photo=path_master_astro_photo,
    n_jobs=10,
    reference_mag_lim=(11.5, 14.5),
    projection="Ophiuchus_deep",
    photerr_internal=0.005,
    build_public_catalog=True,
)
