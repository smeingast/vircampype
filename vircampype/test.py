from vircampype.fits.images.vircam import VircamImages

# Path
path = "/Users/stefan/Dropbox/Data/VISIONS/DR1/raw/"

# Sort files
# files = VircamImages.from_folder(path=path, substring="*.fits")
# files.move2subdirectories()

# Build master calibration
calibration = VircamImages.from_folder(path=path + "calibration/", substring="*.fits")
calibration.build_mastercalibration()

# print(test)
# print(len(test))
