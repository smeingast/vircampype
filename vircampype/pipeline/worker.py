#!/usr/bin/env python
"""
Either install vircampype or add root to PYTHONPATH via e.g.
export PYTHONPATH="/Users/stefan/Dropbox/Projects/vircampype/":$PYTHONPATH}
"""
import sys
import argparse
from vircampype.tools.datatools import *
from vircampype.pipeline.main import Pipeline


def main():
    # Setup parser
    parser = argparse.ArgumentParser(description="Pipeline for VIRCAM images.")
    parser.add_argument("-s", "--setup", help="Input setup file", type=list)
    parser.add_argument(
        "--sort",
        help="Sort files by passing their paths (e.g. /path/to/files/*fits)",
        nargs="+",
        default=None,
    )

    # Parse arguments
    args = parser.parse_args()

    if isinstance(args.sort, list):
        paths_science, paths_calib = split_in_science_and_calibration(
            paths_files=args.sort
        )
        sort_vircam_calibration(paths_calib=paths_calib)
        sort_vircam_science(paths_science=paths_science)
        return

    # Initialize pipeline
    pipeline = Pipeline(setup=args.setup)

    # Set console title
    sys.stdout.write("\x1b]2;{0}\x07".format(pipeline.setup.name))

    # Run pipeline
    if "calibration" in pipeline.setup.name.lower():
        pipeline.process_calibration()
    else:
        pipeline.process_science()


if __name__ == "__main__":
    main()
