from glob import glob
from vircampype.visions.mosaics.build_mosaic_rgb import build_mosaic_rgb

name_mosaic = "CrA_RGB_H"
path_mosaic = "/Volumes/Data/Mosaics/CrA/RGB_H/"
paths_tiles_wide = sorted(
    glob("/Volumes/Data/VISIONS/198C-2009A/vircampype/CrA_wide*_A/tile/*A.fits")
)
path_deep_stack = "/Volumes/Data/Mosaics/CrA/vircampype/CrA_deep_H/tile/CrA_deep_H.fits"

build_mosaic_rgb(
    paths_tiles_wide=paths_tiles_wide,
    path_deep_stack=path_deep_stack,
    path_mosaic=path_mosaic,
    name_mosaic=name_mosaic,
    field_name="Corona_Australis_wide",
)
