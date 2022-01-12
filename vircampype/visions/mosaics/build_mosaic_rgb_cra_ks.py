from glob import glob
from vircampype.visions.mosaics.build_mosaic_rgb import build_mosaic_rgb

name_mosaic = "CrA_RGB_Ks"
path_mosaic = "/Volumes/Data/Mosaics/CrA/RGB_Ks/"
paths_tiles_wide = sorted(
    glob("/Volumes/Data/VHS/CrA/vircampype/*GPS*_Ks/tile/*Ks.fits")
)
path_deep_stack = (
    "/Volumes/Data/Mosaics/CrA/vircampype/CrA_deep_Ks/tile/CrA_deep_ks.fits"
)

build_mosaic_rgb(
    paths_tiles_wide=paths_tiles_wide,
    path_deep_stack=path_deep_stack,
    path_mosaic=path_mosaic,
    name_mosaic=name_mosaic,
    field_name="Corona_Australis_wide",
)
