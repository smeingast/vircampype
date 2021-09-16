from vircampype.visions.mosaics.main import build_mosaic

# Setup for pipeline
name = "CrA_mosaic_J"
path_scripts = "/Volumes/Data/VHS/CrA/scripts/J/"
path_master_astro_photo = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/CrA/"
path_data = f"/Volumes/Data/VHS/CrA/data_mosaic/{name}/"
path_pype = "/Volumes/Data/VHS/CrA/vircampype/"
projection = "Corona_Australis_wide"
additional_source_masks = "Corona_Australis_wide"
reference_mag_lim = (13.0, 15.5)  # J VHS
phase3_photerr_internal = 0.005
n_jobs = 18

build_mosaic(path_master_astro_photo=path_master_astro_photo, path_pype=path_pype, path_data=path_data, n_jobs=n_jobs,
             projection=projection, additional_source_masks=additional_source_masks, path_scripts=path_scripts,
             reference_mag_lim=reference_mag_lim, phase3_photerr_internal=phase3_photerr_internal, name=name)
