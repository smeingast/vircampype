from glob import glob
from vircampype.visions.mosaics.build_mosaic_rgb import build_mosaic_rgb

name_mosaic = "CrA_mosaic_RGB_Ks"
path_mosaic = "/Volumes/Data/Mosaics/CrA/RGB_Ks/"
paths_tiles_wide = sorted(
    glob("/Volumes/Data/VHS/CrA/vircampype/*GPS*_Ks/tile/*Ks.fits")
)
paths_tiles_deep = sorted(
    glob("/Volumes/Data/VISIONS/198C-2009E/vircampype/CrA_deep_Ks_*/tile/*[1-2].fits")
)

build_mosaic_rgb(
    paths_tiles_wide=paths_tiles_wide,
    paths_tiles_deep=paths_tiles_deep,
    path_mosaic=path_mosaic,
    name_mosaic=name_mosaic,
    field_name="Corona_Australis_wide",
)
