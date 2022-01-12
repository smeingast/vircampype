from glob import glob
from vircampype.visions.mosaics.build_mosaic_rgb import build_mosaic_rgb

name_mosaic = "CrA_RGB_J"
path_mosaic = "/Volumes/Data/Mosaics/CrA/RGB_J/"
paths_tiles_wide = sorted(glob("/Volumes/Data/VHS/CrA/vircampype/*GPS*_J/tile/*J.fits"))
path_deep_stack = "/Volumes/Data/Mosaics/CrA/vircampype/CrA_deep_J/tile/CrA_deep_J.fits"

build_mosaic_rgb(
    paths_tiles_wide=paths_tiles_wide,
    path_deep_stack=path_deep_stack,
    path_mosaic=path_mosaic,
    name_mosaic=name_mosaic,
    field_name="Corona_Australis_wide",
)
