from vircampype.visions.mosaics.build_mosaic import build_mosaic

# Setup for pipeline
name = "CrA_mosaic_H_C"
path_scripts = "/Users/stefan/Dropbox/Projects/VISIONS/scripts/VISIONS/CrA/CrA_wide_C/"
path_master_astro_photo = "/Users/stefan/Dropbox/Data/VISIONS/Master/CrA/Mosaic/"
path_data = f"/Volumes/Data/Mosaics/CrA/{name}/"
path_pype = f"/Volumes/Data/Mosaics/CrA/vircampype/"
projection = "Corona_Australis_wide"
reference_mag_lim = (12.0, 15.0)  # H wide
phase3_photerr_internal = 0.005
n_jobs = 10

build_mosaic(name=name, path_scripts=path_scripts, path_data=path_data, path_pype=path_pype, n_jobs=n_jobs,
             path_master_astro_photo=path_master_astro_photo, reference_mag_lim=reference_mag_lim,
             projection=projection, phase3_photerr_internal=phase3_photerr_internal, build_class_star_library=True)
