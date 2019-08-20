# =========================================================================== #
# Import
import shutil

from vircampype.utils.miscellaneous import *
from vircampype.fits.images.dark import DarkImages
from vircampype.fits.images.flat import FlatImages
from vircampype.fits.images.common import FitsImages
from vircampype.fits.images.sky import OffsetImages, ScienceImages, StdImages


class VircamImages(FitsImages):

    def __init__(self, file_paths):
        """
        Class for Vircam images.

        Parameters
        ----------
        file_paths : list
            List of file paths.

        """

        # Initialize files
        super(VircamImages, self).__init__(file_paths=file_paths)

    def split_type(self):
        """
        Basic file splitting routine for VIRCAM data

        Returns
        -------
        ImageList
            ImageList of FitsImages for the various data types

        """

        # Get the type, and category from the primary header
        types, category = self.primeheaders_get_keys([self.setup["keywords"]["type"],
                                                      self.setup["keywords"]["category"]])

        # Extract the various data types for VIRCAM
        science_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                         c == "SCIENCE" and "OBJECT" in t]
        science = None if len(science_index) < 1 else \
            VircamScienceImages([self.full_paths[i] for i in science_index])

        offset_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                        c == "SCIENCE" and "SKY" in t]
        offset = None if len(offset_index) < 1 else \
            VircamOffsetImages([self.full_paths[i] for i in offset_index])

        dark_science_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                              c == "CALIB" and t == "DARK"]
        dark_science = None if len(dark_science_index) < 1 else \
            VircamDarkImages([self.full_paths[i] for i in dark_science_index])

        flat_twilight_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                               c == "CALIB" and t == "FLAT,TWILIGHT"]
        flat_twilight = None if len(flat_twilight_index) < 1 else \
            VircamFlatTwilight([self.full_paths[i] for i in flat_twilight_index])

        dark_lin_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                          c == "CALIB" and t == "DARK,LINEARITY"]
        dark_lin = None if len(dark_lin_index) < 1 else \
            VircamDarkImages([self.full_paths[i] for i in dark_lin_index])

        flat_lamp_lin_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                               c == "CALIB" and t == "FLAT,LAMP,LINEARITY"]
        flat_lamp_lin = None if len(flat_lamp_lin_index) < 1 else \
            VircamFlatLampLin([self.full_paths[i] for i in flat_lamp_lin_index])

        flat_lamp_check_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                                 c == "CALIB" and t == "FLAT,LAMP,CHECK"]
        flat_lamp_check = None if len(flat_lamp_check_index) < 1 else \
            VircamFlatLampCheck([self.full_paths[i] for i in flat_lamp_check_index])

        dark_gain_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                           c == "CALIB" and t == "DARK,GAIN"]
        dark_gain = None if len(dark_gain_index) < 1 else \
            VircamDarkImages([self.full_paths[i] for i in dark_gain_index])

        flat_lamp_gain_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                                c == "CALIB" and t == "FLAT,LAMP,GAIN"]
        flat_lamp_gain = None if len(flat_lamp_gain_index) < 1 else \
            VircamFlatLampGain([self.full_paths[i] for i in flat_lamp_gain_index])

        std_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                     c == "CALIB" and t == "STD,FLUX"]
        std = None if len(std_index) < 1 else \
            VircamStdImages([self.full_paths[i] for i in std_index])

        return {"science": science, "offset": offset, "std": std,
                "dark_science": dark_science, "dark_lin": dark_lin, "dark_gain": dark_gain,
                "flat_twilight": flat_twilight, "flat_lamp_lin": flat_lamp_lin,
                "flat_lamp_check": flat_lamp_check, "flat_lamp_gain": flat_lamp_gain}

    def move2subdirectories(self):

        types = self.split_type()

        for otype, images in types.items():

            if images is None:
                continue

            # Loop over images
            for idx in range(len(images)):

                # Make subfolders for science files
                if otype == "science":
                    mdir = images.file_directories[idx] + "science/"
                    sname = images.primeheaders_get_keys(["HIERARCH ESO OBS NAME"])[0][idx]
                    mdir += sname + "/"
                # TODO: Check how this sorts offset images. Perhaps easiest when they go into the science path
                elif otype == "offset":
                    continue
                else:
                    mdir = images.file_directories[idx] + "calibration/"

                # Make directory
                make_folder(path=mdir)

                # Move current file into this directory
                shutil.move(images.full_paths[idx], mdir)

    def build_mastercalibration(self):
        """
        Builds masterdarks, masterbpms, masterlinearity tables, masterflats/weights, and mastersky images from all given
        input files. The setup must be modified before calling this method.

        Build order:
        1. BPM, 2. Darks, 3. Linearity, 4. Flat

        Returns
        -------
        tuple
            Tuple holding (dark_science, dark_linearity, dark_gain, bad pixel mask, lineartiy, flat, weight, sky, ref)
            master instances.

        """

        # Split into categories
        split = self.split_type()

        # MasterBPM needs to build first
        split["flat_lamp_check"].build_master_bpm()
        exit()

        # Next are darks
        for ot in ["dark_science", "dark_lin", "dark_gain"]:
            split[ot].build_master_dark()


class VircamDarkImages(DarkImages):

    def __init__(self, file_paths=None):
        super(VircamDarkImages, self).__init__(file_paths=file_paths)


class VircamFlatImages(FlatImages):

    def __init__(self, file_paths=None):
        super(VircamFlatImages, self).__init__(file_paths=file_paths)


class VircamFlatTwilight(VircamFlatImages):

    def __init__(self, file_paths=None):
        super(VircamFlatTwilight, self).__init__(file_paths=file_paths)


class VircamFlatLampLin(VircamFlatImages):
    def __init__(self, file_paths=None):
        super(VircamFlatLampLin, self).__init__(file_paths=file_paths)


class VircamFlatLampGain(VircamFlatImages):
    def __init__(self, file_paths=None):
        super(VircamFlatLampGain, self).__init__(file_paths=file_paths)


class VircamFlatLampCheck(VircamFlatImages):

    def __init__(self, file_paths=None):
        super(VircamFlatLampCheck, self).__init__(file_paths=file_paths)


class VircamOffsetImages(OffsetImages):
    def __init__(self, file_paths=None):
        super(VircamOffsetImages, self).__init__(file_paths=file_paths)


class VircamScienceImages(ScienceImages):
    def __init__(self, file_paths=None):
        super(VircamScienceImages, self).__init__(file_paths=file_paths)


class VircamStdImages(StdImages):
    def __init__(self, file_paths=None):
        super(VircamStdImages, self).__init__(file_paths=file_paths)
