from pathlib import Path
from vircampype.visions.mosaics.rgb.utils import jpg2fits

# File paths
path_base = Path("/Users/stefan/Dropbox/Projects/VISIONS/RGB/CrA")
path_jpg = path_base / "CrA_control.jpg"
path_fits = path_base / "CrA_control_J.fits"

# Weight paths
paths_weights = path_base.glob("*control*.weight.fits")

# Convert channels to fits
jpg2fits(path_jpg=path_jpg, path_fits=path_fits, paths_weights=paths_weights)
