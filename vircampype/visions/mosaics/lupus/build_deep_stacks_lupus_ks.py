import glob
from vircampype.visions.mosaics.build_mosaic import build_mosaic

# Setup for pipeline
path_scripts = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/Lupus/deep/"
path_master_astro_photo = "/Users/stefan/Dropbox/Projects/VISIONS/" \
                          "Pipeline/other/master-astro-photo/Lupus_deep/"
path_pype = f"/Volumes/Data/Mosaics/Lupus/vircampype/"
reference_mag_lim = (11.5, 14.5)
phase3_photerr_internal = 0.005
n_jobs = 10

# Get script paths
paths_scripts_n = sorted(glob.glob(f"{path_scripts}*_Ks_N_*.yml"))
paths_scripts_s = sorted(glob.glob(f"{path_scripts}*_Ks_S_*.yml"))

# Common kwargs
kwargs = dict(
    path_pype=path_pype,
    n_jobs=n_jobs,
    path_master_astro_photo=path_master_astro_photo,
    reference_mag_lim=reference_mag_lim,
    phase3_photerr_internal=phase3_photerr_internal,
    build_class_star_library=False,
    build_phase3=False,
)

# Build mosaic for North field
name = "Lupus_deep_N_Ks"
path_data = f"/Volumes/Data/Mosaics/Lupus/{name}/"
build_mosaic(
    name=name,
    paths_scripts=paths_scripts_n,
    path_data=path_data,
    projection="Lupus_deep_n",
    **kwargs,
)

# Build mosaic for South field
name = "Lupus_deep_S_Ks"
path_data = f"/Volumes/Data/Mosaics/Lupus/{name}/"
build_mosaic(
    name=name,
    paths_scripts=paths_scripts_s,
    path_data=path_data,
    projection="Lupus_deep_s",
    **kwargs,
)
