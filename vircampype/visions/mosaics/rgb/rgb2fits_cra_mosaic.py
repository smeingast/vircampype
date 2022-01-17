import os
from glob import glob
from pathlib import Path
from vircampype.visions.mosaics.rgb.utils import jpg2fits

# File paths
path_base = "/Volumes/Data/RGB/CrA/mosaic/data/"

# Find files
paths_jpg = sorted(glob(f"{path_base}CrA_RGB_*.jpg"))
paths_fits = sorted(glob(f"{path_base}CrA_RGB_J_*.fits"))

if len(paths_jpg) != len(paths_fits):
    raise ValueError("JPG and FITS files not matching")

# Check existense of files
for pj, pf in zip(paths_jpg, paths_fits):
    if not os.path.exists(pj):
        raise ValueError

for pj, pf in zip(paths_jpg, paths_fits):

    # Weight paths
    paths_weights = None

    # Convert channels to fits
    jpg2fits(path_jpg=Path(pj), path_fits=Path(pf), paths_weights=paths_weights)
