#!/usr/bin/env python
import sys
import argparse
sys.path.append("/Users/stefan/Dropbox/Projects/vircampype")
from vircampype.pipeline.main import Pipeline

# Setup parser
parser = argparse.ArgumentParser(description="Pipeline for VIRCAM images.")
parser.add_argument("-s", "--setup", help="Input setup file", type=str, required=True)

# Parse arguments
args = parser.parse_args()

# Initialize pipeline
pipeline = Pipeline(setup=args.setup)

# Set console title
sys.stdout.write("\x1b]2;{0}\x07".format(pipeline.setup.name))

# Run pipeline
pipeline.build_master_calibration()
pipeline.process_science()
