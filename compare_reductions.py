#!/usr/bin/env python3
"""Golden-output diff for the logging overhaul.

Compares two pipeline output trees (an OLD reference reduction and a NEW one
produced by the logging-overhaul branch) and verifies the science products are
bit-for-bit identical. The logging refactor changes only output channels, so
the expectation is EXACT equality of all image/table data, and identical
headers apart from provenance/timestamp cards.

Usage:
    python compare_reductions.py [--old DIR] [--new DIR] [--strict-headers]

Exit code 0 means no science difference; 1 means a real data difference was
found (investigate); 2 means structural issues (missing files, read errors).

The pipeline log, status pickle, and QC plots legitimately differ and are
skipped. If the only differences are timestamp/provenance header cards, add
them to IGNORE_KEYWORDS (or run with the defaults, which already cover the
common ones).
"""

import argparse
import os
import sys

from astropy.io.fits import FITSDiff

OLD_DEFAULT = (
    "/Volumes/visions/pipeline/results/CoronaAustralis/vhs/"
    "P87A_Str28_GPS_RA_265_300_1_1_12_J_old"
)
NEW_DEFAULT = (
    "/Volumes/visions/pipeline/results/CoronaAustralis/vhs/"
    "P87A_Str28_GPS_RA_265_300_1_1_12_J"
)

# Cards that legitimately vary between two runs (creation/processing time,
# checksums, tool execution stamps, free-text history). NOT science.
IGNORE_KEYWORDS = [
    "DATE",  # file write time (NB: NOT DATE-OBS, which is science and kept)
    "CHECKSUM",
    "DATASUM",
    "HISTORY",
    "COMMENT",
    "SOFTNAME",
    "SOFTVERS",
    "SWDATE",
    "AUTHOR",
    "ORIGFILE",
    # Completeness QC is stochastic (random artificial-star injection), so its
    # derived header metrics legitimately vary run-to-run.
    "PYPE COMP50 MAX",
    "PYPE COMP50 MED",
    "PYPE COMP50 MIN",
    "PYPE COMP90 MAX",
    "PYPE COMP90 MED",
    "PYPE COMP90 MIN",
]

# Files / directories that are not science and are expected to differ.
SKIP_DIRS = {"temp", "headers"}
SKIP_SUFFIXES = (".log", ".p", ".png", ".pdf", ".jpg", ".txt", ".xml", ".ahead")
FITS_SUFFIXES = (".fits", ".fits.fz", ".fz", ".cat")


def _science_files(root: str) -> set[str]:
    """Relative paths of candidate science FITS files under ``root``."""
    found: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(SKIP_SUFFIXES):
                continue
            if name.endswith(FITS_SUFFIXES):
                rel = os.path.relpath(os.path.join(dirpath, name), root)
                found.add(rel)
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default=OLD_DEFAULT, help="Reference (old) output dir")
    ap.add_argument("--new", default=NEW_DEFAULT, help="New output dir")
    ap.add_argument(
        "--strict-headers",
        action="store_true",
        help="Do not ignore provenance keywords; compare every header card.",
    )
    args = ap.parse_args(argv)

    if not os.path.isdir(args.old) or not os.path.isdir(args.new):
        print(f"ERROR: missing directory: old={args.old} new={args.new}")
        return 2

    old_files = _science_files(args.old)
    new_files = _science_files(args.new)

    only_old = sorted(old_files - new_files)
    only_new = sorted(new_files - old_files)
    common = sorted(old_files & new_files)

    ignore = [] if args.strict_headers else IGNORE_KEYWORDS

    data_diffs: list[str] = []
    header_diffs: list[str] = []
    read_errors: list[str] = []
    identical = 0

    for rel in common:
        old_p = os.path.join(args.old, rel)
        new_p = os.path.join(args.new, rel)
        try:
            diff = FITSDiff(
                old_p,
                new_p,
                ignore_keywords=ignore,
                ignore_comments=["*"],
                ignore_blank_cards=True,
                rtol=0.0,
                atol=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            read_errors.append(f"{rel}: {type(exc).__name__}: {exc}")
            continue

        if diff.identical:
            identical += 1
            continue

        # Classify from the report text: a real science diff has differing
        # pixel/table data; otherwise it is a header-only difference.
        report = diff.report()
        if "Data contains differences" in report:
            data_diffs.append(rel)
            print("=" * 78)
            print(f"DATA DIFFERENCE: {rel}")
            print(report)
        else:
            header_diffs.append(rel)

    # ----- summary -----
    print("\n" + "#" * 78)
    print("GOLDEN-OUTPUT DIFF SUMMARY")
    print(f"  old: {args.old}")
    print(f"  new: {args.new}")
    print(f"  science FITS files compared : {len(common)}")
    print(f"  identical                   : {identical}")
    print(f"  DATA differences            : {len(data_diffs)}")
    print(f"  header-only differences     : {len(header_diffs)}")
    print(f"  only in OLD                 : {len(only_old)}")
    print(f"  only in NEW                 : {len(only_new)}")
    print(f"  read errors                 : {len(read_errors)}")

    if only_old:
        print("\n  Files only in OLD:")
        for r in only_old:
            print(f"    - {r}")
    if only_new:
        print("\n  Files only in NEW:")
        for r in only_new:
            print(f"    - {r}")
    if header_diffs:
        print("\n  Header-only differences (likely provenance; inspect to confirm):")
        for r in header_diffs:
            print(f"    - {r}")
    if read_errors:
        print("\n  Read errors:")
        for r in read_errors:
            print(f"    - {r}")

    if data_diffs:
        print("\nRESULT: FAIL - real data differences found (see reports above).")
        return 1
    if only_old or only_new or read_errors:
        print(
            "\nRESULT: INCOMPLETE - structural differences (missing files / read "
            "errors). No data differences among comparable files."
        )
        return 2
    if header_diffs:
        print(
            "\nRESULT: PASS for data; header-only diffs remain. Inspect them: if "
            "they are only timestamp/provenance cards, add the keywords to "
            "IGNORE_KEYWORDS (or trust this result)."
        )
        return 0
    print("\nRESULT: PASS - all science products are bit-for-bit identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
