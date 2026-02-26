import os
import shutil
from typing import List, Optional, Tuple

from astropy.io import fits

from vircampype.tools.systemtools import make_folder

__all__ = [
    "split_in_science_and_calibration",
    "sort_vircam_calibration",
    "sort_vircam_science",
    "sort_by_passband",
]


def split_in_science_and_calibration(
    paths_files: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Split a list of VIRCAM FITS files into science and calibration subsets.

    Reads the ``HIERARCH ESO DPR CATG`` keyword from each file header and
    classifies files as either ``SCIENCE`` or ``CALIB``.

    Parameters
    ----------
    paths_files : List[str]
        List of paths to raw VIRCAM FITS files.

    Returns
    -------
    paths_science : List[str]
        Paths of files classified as ``SCIENCE``.
    paths_calib : List[str]
        Paths of files classified as ``CALIB``.

    Raises
    ------
    ValueError
        If no files are supplied or if the combined count of science and
        calibration files does not equal the total number of input files.
    """
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


def sort_vircam_calibration(paths_calib: List[str]) -> Optional[List[str]]:
    """
    Move VIRCAM calibration files into a ``Calibration/`` subdirectory.

    Creates the subdirectory under the current working directory and moves
    each file into it.

    Parameters
    ----------
    paths_calib : List[str]
        Paths to calibration FITS files.

    Returns
    -------
    List[str] or None
        New file paths after moving, or ``None`` if *paths_calib* is empty.
    """
    # Check that at least one file is there
    if len(paths_calib) == 0:
        return None

    # Get absolute filepaths
    paths_calib = [os.path.abspath(p) for p in paths_calib]

    # Get current working directory and make calibration folder
    path_folder = f"{os.getcwd()}/Calibration/"
    make_folder(path=path_folder)

    # Move files to calibration directory
    for p in paths_calib:
        shutil.move(p, path_folder)

    return [f"{path_folder}{os.path.basename(p)}" for p in paths_calib]


def sort_vircam_science(paths_science: List[str]) -> Optional[List[str]]:
    """
    Sort VIRCAM science files into per-object subdirectories.

    Reads ``HIERARCH ESO OBS NAME`` from each file header and moves the file
    into a subdirectory named after that value.

    Parameters
    ----------
    paths_science : List[str]
        Paths to science FITS files.

    Returns
    -------
    List[str] or None
        New file paths after moving, or ``None`` if *paths_science* is empty.
    """
    # Check that files are actually provided
    if len(paths_science) == 0:
        return None

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


def sort_by_passband(paths: List[str]) -> None:
    """
    Sort FITS files into per-passband subdirectories.

    Reads ``HIERARCH ESO INS FILT1 NAME`` from each file header and moves
    the file into a subdirectory named after the passband, created inside the
    file's current directory.

    Parameters
    ----------
    paths : List[str]
        Paths to FITS files to sort.

    Returns
    -------
    None
    """
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
