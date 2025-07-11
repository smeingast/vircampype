import re
import sys
from pathlib import Path

from astropy.io import fits

# Set to True for a dry run (no actual renaming), False to perform renaming
DRY_RUN = False

if not DRY_RUN:
    input(
        "WARNING: DRY_RUN is set to False. This will rename files. "
        "Press Enter to continue or Ctrl+C to abort."
    )


def find_fits_fz_files(root_dir):
    # Walk the directory tree alphabetically
    for dirpath, dirnames, filenames in sorted_walk(root_dir):
        for fname in sorted(filenames):
            if fname.endswith(".FZ") and len(fname) == 11:
                yield Path(dirpath) / fname


def sorted_walk(root_dir):
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


def parse_arcfile(arcfile_value):
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


def main(root_dir):
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
