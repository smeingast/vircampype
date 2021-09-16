from vircampype.visions.mosaics.main import build_mosaic

# Setup for pipeline
name = "CrA_mosaic_H_C"
path_scripts = "/Volumes/Data/VISIONS/198C-2009E/scripts/CrA_wide_C/"
path_master_astro_photo = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/CrA/"
path_data = "/Volumes/Data/VISIONS/198C-2009E/data_mosaic/{0}/".format(name)
path_pype = "/Volumes/Data/VISIONS/198C-2009E/vircampype/"
projection = "Corona_Australis_wide"
additional_source_masks = "Corona_Australis_wide"
reference_mag_lim = (12.0, 15.0)  # H wide
phase3_photerr_internal = 0.005
n_jobs = 18

build_mosaic(path_master_astro_photo=path_master_astro_photo, path_pype=path_pype, path_data=path_data, n_jobs=n_jobs,
             projection=projection, additional_source_masks=additional_source_masks, path_scripts=path_scripts,
             reference_mag_lim=reference_mag_lim, phase3_photerr_internal=phase3_photerr_internal, name=name)
