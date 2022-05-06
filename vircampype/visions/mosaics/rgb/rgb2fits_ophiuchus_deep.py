from pathlib import Path
from vircampype.visions.mosaics.rgb.utils import jpg2fits

# File paths
path_base = Path("/Users/stefan/Dropbox/Projects/VISIONS/RGB/Ophiuchus")
path_jpg = path_base / "Ophiuchus_deep_RGB.jpg"
path_fits = path_base / "Ophiuchus_deep_J.fits"

# Weight paths
paths_weights = path_base.glob("*.weight.fits")

# Convert channels to fits
jpg2fits(path_jpg=path_jpg, path_fits=path_fits, paths_weights=paths_weights)
