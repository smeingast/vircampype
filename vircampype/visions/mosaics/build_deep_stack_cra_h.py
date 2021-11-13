from vircampype.visions.mosaics.build_mosaic import build_mosaic

# Setup for pipeline
name = "CrA_deep_H"
path_scripts = "/Users/stefan/Dropbox/Projects/VISIONS/scripts/VISIONS/CrA/CrA_deep_H/"
path_master_astro_photo = "/Volumes/Data/VISIONS/198C-2009E/vircampype/CrA_deep_H_1/master/"
path_data = f"/Volumes/Data/Mosaics/CrA/{name}/"
path_pype = f"/Volumes/Data/Mosaics/CrA/vircampype/"
projection = "Corona_Australis_deep"
reference_mag_lim = (12.0, 15.0)
phase3_photerr_internal = 0.005
n_jobs = 10

build_mosaic(name=name, path_scripts=path_scripts, path_data=path_data, path_pype=path_pype, n_jobs=n_jobs,
             path_master_astro_photo=path_master_astro_photo, reference_mag_lim=reference_mag_lim,
             projection=projection, phase3_photerr_internal=phase3_photerr_internal, build_class_star_library=True)
