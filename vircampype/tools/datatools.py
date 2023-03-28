import os
import glob
import shutil

from astropy.io import fits
from vircampype.tools.systemtools import make_folder

__all__ = ["sort_vircam_calibration", "sort_vircam_science", "sort_by_passband"]


def sort_vircam_calibration(path_all, path_calibration, extension=".fits"):

    # Add '/' if necessary
    if not path_all.endswith("/"):
        path_all += "/"
    if not path_calibration.endswith("/"):
        path_calibration += "/"

    # Make calibration folder
    make_folder(path=path_calibration)

    # Find files
    paths_all = glob.glob(pathname="{0}*{1}".format(path_all, extension))

    # Get category
    catg_all = [fits.getheader(filename=f)["HIERARCH ESO DPR CATG"] for f in paths_all]

    idx_calib = [i for i, j in enumerate(catg_all) if j == "CALIB"]
    idx_science = [i for i, j in enumerate(catg_all) if j == "SCIENCE"]

    # Dummy check
    if len(idx_calib) + len(idx_science) != len(paths_all):
        raise ValueError("Input and output not matching")

    # Get paths for calibration files
    paths_calib = [paths_all[i] for i in idx_calib]

    # Move files to calibration directory
    for p in paths_calib:

        # If file exists, remove and continue
        if os.path.exists(path_calibration + os.path.basename(p)):
            os.remove(p)

        # Otherwise move
        else:
            shutil.move(p, path_calibration)


def sort_vircam_science(path, extension=".fits"):

    # Add '/' if necessary
    if not path.endswith("/"):
        path += "/"

    # Find files
    paths_orig = glob.glob(pathname="{0}*{1}".format(path, extension))
    file_names = [os.path.basename(f) for f in paths_orig]
    paths_dirs = ["{0}/".format(os.path.dirname(f)) for f in paths_orig]

    # Get Object Name
    obj = [
        fits.getheader(filename=f)["HIERARCH ESO OBS NAME"].replace("VISIONS_", "")
        for f in paths_orig
    ]

    # Identify unique objects
    uobj = sorted(list(set(obj)))

    # Make folders
    for uo in uobj:
        make_folder(path=path + uo)

    # Construct output paths
    paths_move = [
        "{0}{1}/{2}".format(d, o, f) for d, o, f in zip(paths_dirs, obj, file_names)
    ]

    # Move files to folders
    for po, pm in zip(paths_orig, paths_move):
        shutil.move(po, pm)


def sort_by_passband(paths):

    # Get base directories of files
    directories = [f"{os.path.dirname(p)}/" for p in paths]

    # Get passbands
    passbands = [
        fits.getheader(filename=p)["HIERARCH ESO INS FILT1 NAME"] for p in paths
    ]

    # Loop over files and sort into subdirectories
    for pp, dd, pb in zip(paths, directories, passbands):

        # Make directories
        make_folder(f"{dd}{pb}")

        # Move files
        outname = f"{dd}{pb}/{os.path.basename(pp)}"
        shutil.move(pp, outname)
