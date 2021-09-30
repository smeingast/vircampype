from vircampype.visions.mosaics.build_mosaic import build_mosaic

# Setup for pipeline
name = "CrA_mosaic_H_C"
path_scripts = "/Users/stefan/Dropbox/Projects/VISIONS/scripts/VISIONS/198C-2009E/CrA/CrA_wide_C/"
path_master_astro_photo = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/CrA/"
path_data = f"/Volumes/Data/VISIONS/198C-2009E/data_mosaic/{name}/"
path_pype = "/Volumes/Data/VISIONS/198C-2009E/vircampype/"
projection = "Corona_Australis_wide"
reference_mag_lim = (12.0, 15.0)  # H wide
phase3_photerr_internal = 0.005
n_jobs = 18

build_mosaic(name=name, path_scripts=path_scripts, path_data=path_data, path_pype=path_pype, n_jobs=n_jobs,
             path_master_astro_photo=path_master_astro_photo, reference_mag_lim=reference_mag_lim,
             projection=projection, phase3_photerr_internal=phase3_photerr_internal, build_class_star_library=True)
