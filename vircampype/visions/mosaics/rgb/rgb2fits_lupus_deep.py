from pathlib import Path
from vircampype.visions.mosaics.rgb.utils import jpg2fits

# File paths
path_base = Path("/Users/stefan/Dropbox/Projects/VISIONS/RGB/Lupus")
path_jpg_n = path_base / "Lupus_deep_N.jpg"
path_fits_n = path_base / "Lupus_deep_J_N.fits"
path_jpg_s = path_base / "Lupus_deep_S.jpg"
path_fits_s = path_base / "Lupus_deep_J_S.fits"

# Weight paths
paths_weights_n = path_base.glob("*N.weight.fits")
paths_weights_s = path_base.glob("*S.weight.fits")

# Convert channels to fits
jpg2fits(path_jpg=path_jpg_n, path_fits=path_fits_n, paths_weights=paths_weights_n)
jpg2fits(path_jpg=path_jpg_s, path_fits=path_fits_s, paths_weights=paths_weights_s)
