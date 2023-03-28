import os
import shutil

from astropy.io import fits
from typing import List, Optional
from vircampype.tools.systemtools import make_folder

__all__ = [
    "split_in_science_and_calibration",
    "sort_vircam_calibration",
    "sort_vircam_science",
    "sort_by_passband",
]


def split_in_science_and_calibration(paths_files: List) -> (List, List):
    # Check that at least one file is there
    if len(paths_files) == 0:
        raise ValueError("No files found!")

    # Get absolute filepaths
    paths_files = [os.path.abspath(p) for p in paths_files]

    # Get category
    catg_all = [
        fits.getheader(filename=f)["HIERARCH ESO DPR CATG"] for f in paths_files
    ]

    # Grab science and calibration file indices
    idx_science = [i for i, j in enumerate(catg_all) if j == "SCIENCE"]
    idx_calib = [i for i, j in enumerate(catg_all) if j == "CALIB"]

    # Dummy check
    if len(idx_calib) + len(idx_science) != len(paths_files):
        raise ValueError("Input and output not matching")

    # Get paths for calibration files
    paths_science = [paths_files[i] for i in idx_science]
    paths_calib = [paths_files[i] for i in idx_calib]

    # Return
    return paths_science, paths_calib


def sort_vircam_calibration(paths_calib: List) -> Optional[List]:
    # Check that at least one file is there
    if len(paths_calib) == 0:
        return

    # Get absolute filepaths
    paths_calib = [os.path.abspath(p) for p in paths_calib]

    # Get current working directory and make calibration folder
    path_folder = f"{os.getcwd()}/Calibration/"
    make_folder(path=path_folder)

    # Move files to calibration directory
    for p in paths_calib:
        shutil.move(p, path_folder)

    return [f"{path_folder}{p}" for p in paths_calib]


def sort_vircam_science(paths_science: List) -> Optional[List]:
    # Check that files are actually provide
    if len(paths_science) == 0:
        return

    # Get absolute filepaths
    paths_science = [os.path.abspath(p) for p in paths_science]

    # Get directories and filenames
    paths_dirs = [f"{os.path.dirname(f)}/" for f in paths_science]
    file_names = [os.path.basename(f) for f in paths_science]

    # Get Object Name
    obj = [fits.getheader(filename=f)["HIERARCH ESO OBS NAME"] for f in paths_science]

    # Identify unique objects
    uobj = sorted(list(set(obj)))

    # Get current working directory
    cwd = f"{os.getcwd()}/"

    # Make folders
    for uo in uobj:
        make_folder(path=f"{cwd}{uo}")

    # Construct output paths
    paths_move = [f"{d}{o}/{f}" for d, o, f in zip(paths_dirs, obj, file_names)]

    # Move files to folders
    for po, pm in zip(paths_science, paths_move):
        shutil.move(po, pm)

    return paths_move


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
