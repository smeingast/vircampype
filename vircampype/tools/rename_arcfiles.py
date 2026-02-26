import re
import sys
from pathlib import Path
from typing import Generator

from astropy.io import fits

# Set to True for a dry run (no actual renaming), False to perform renaming
DRY_RUN = False

if not DRY_RUN:
    input(
        "WARNING: DRY_RUN is set to False. This will rename files. "
        "Press Enter to continue or Ctrl+C to abort."
    )


def find_fits_fz_files(root_dir: str) -> Generator[Path, None, None]:
    """
    Recursively yield ``.FZ`` files with exactly 11-character names.

    Walks the directory tree rooted at *root_dir* in alphabetical order and
    yields ``Path`` objects for files whose names end in ``.FZ`` and are
    exactly 11 characters long.

    Parameters
    ----------
    root_dir : str
        Root directory to search.

    Yields
    ------
    Path
        Path to each matching ``.FZ`` file.
    """
    # Walk the directory tree alphabetically
    for dirpath, dirnames, filenames in sorted_walk(root_dir):
        for fname in sorted(filenames):
            if fname.endswith(".FZ") and len(fname) == 11:
                yield Path(dirpath) / fname


def sorted_walk(root_dir: str) -> Generator[tuple, None, None]:
    """
    Walk a directory tree alphabetically, yielding ``(dirpath, dirnames, filenames)``.

    Equivalent to ``os.walk`` but with directory and file names sorted
    alphabetically at every level.

    Parameters
    ----------
    root_dir : str
        Root directory from which to start walking.

    Yields
    ------
    tuple
        A 3-tuple ``(dirpath, dirnames, filenames)`` where *dirpath* is a
        string, and *dirnames* / *filenames* are sorted lists of names.
    """
    # Generator that walks the directory tree alphabetically
    root = Path(root_dir)
    dirs = [root]
    while dirs:
        current_dir = dirs.pop(0)
        dirnames = sorted([d.name for d in current_dir.iterdir() if d.is_dir()])
        filenames = sorted([f.name for f in current_dir.iterdir() if f.is_file()])
        yield str(current_dir), dirnames, filenames
        # Add subdirectories to the list to walk them in alphabetical order
        dirs = [current_dir / d for d in dirnames] + dirs


def parse_arcfile(arcfile_value: str) -> str:
    """
    Convert an ESO ARCFILE keyword value to a filesystem-friendly filename.

    Parses a string of the form ``VCAM.YYYY-MM-DDTHH:MM:SS.mmm.fits`` and
    returns a sanitised filename ``VCAM_YYYY-MM-DD_HH-MM-SS-mmm.fits``.

    Parameters
    ----------
    arcfile_value : str
        Value of the ``ARCFILE`` FITS keyword, e.g.
        ``"VCAM.2009-10-24T08:18:33.891.fits"``.

    Returns
    -------
    str
        Reformatted filename without colons or dots in the timestamp part.

    Raises
    ------
    ValueError
        If *arcfile_value* does not match the expected ARCFILE pattern.
    """
    # Example: VCAM.2009-10-24T08:18:33.891.fits
    m = re.match(
        r"^(?P<prefix>VCAM)\.(?P<date>\d{4}-\d{2}-\d{2})T"
        r"(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\.fits$",
        arcfile_value,
    )
    if not m:
        raise ValueError(f"Invalid ARCFILE format: {arcfile_value}")
    prefix = m.group("prefix")
    date = m.group("date")
    time = m.group("time").replace(":", "-").replace(".", "-")
    return f"{prefix}_{date}_{time}.fits"


def main(root_dir: str) -> None:
    """
    Rename all ``.FZ`` VIRCAM archive files under *root_dir*.

    Iterates over every 11-character ``.FZ`` file found recursively under
    *root_dir*, reads the ``ARCFILE`` keyword from its primary FITS header,
    and renames the file to the sanitised form produced by
    :func:`parse_arcfile` (with a ``.fz`` suffix appended).

    When ``DRY_RUN`` is ``True`` the function only prints planned renames
    without touching the file system.

    Parameters
    ----------
    root_dir : str
        Root directory to search for files.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a target filename already exists at the destination path.
    """
    for path in find_fits_fz_files(root_dir):
        # Read the FITS header to get the ARCFILE value
        arcfile = fits.getheader(path)["ARCFILE"]

        # Parse the ARCFILE value to create the new filename
        new_fname = parse_arcfile(arcfile)

        # Add .fz extension back
        new_fname += ".fz"

        # Create the new path with the same directory
        new_path = path.with_name(new_fname)
        if path.resolve() != new_path.resolve():
            print(f"Renaming:\n  {path}\n  -> {new_path}")
            if new_path.exists():
                raise ValueError(
                    f"ERROR:\n{new_path} already exists. Skipping rename for\n{path}."
                )
            else:
                if not DRY_RUN:
                    path.rename(new_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_fits.py <directory>")
        sys.exit(1)
    main(sys.argv[1])
