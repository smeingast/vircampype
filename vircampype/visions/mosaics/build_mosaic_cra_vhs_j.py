from vircampype.visions.mosaics.build_mosaic import build_mosaic

# Setup for pipeline
name = "CrA_mosaic_J"
path_scripts = "/Users/stefan/Dropbox/Projects/VISIONS/scripts/VHS/CrA/J/"
path_master_astro_photo = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/CrA/"
path_data = f"/Volumes/Data/VHS/CrA/data_mosaic/{name}/"
path_pype = "/Volumes/Data/VHS/CrA/vircampype/"
projection = "Corona_Australis_wide"
additional_source_masks = "Corona_Australis_wide"
reference_mag_lim = (13.0, 15.5)  # J VHS
phase3_photerr_internal = 0.005
n_jobs = 18

# Launch mosaic builder
build_mosaic(name=name, path_scripts=path_scripts, path_data=path_data, path_pype=path_pype, n_jobs=n_jobs,
             path_master_astro_photo=path_master_astro_photo, additional_source_masks=additional_source_masks,
             reference_mag_lim=reference_mag_lim, projection=projection, build_class_star_library=True,
             phase3_photerr_internal=phase3_photerr_internal)
