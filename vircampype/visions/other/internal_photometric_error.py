from glob import glob
from vircampype.fits.tables.sextractor import PhotometricCalibratedSextractorCatalogs

path_base = "/Volumes/Data/VISIONS/198C-2009E/vircampype/"
file_paths = sorted(glob(path_base + "CrA_wide*/tile/*ctab"))

setup = dict(name="dummy", path_data=path_base, path_pype=path_base, n_jobs=18,
             projection=None, additional_source_masks=None,
             reference_mag_lim=(12.0, 15.0))

images = PhotometricCalibratedSextractorCatalogs(setup=setup, file_paths=file_paths)
photerr_internal = images.photerr_internal()
images.plot_qc_photerr_internal()
print(photerr_internal)
