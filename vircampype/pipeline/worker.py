#!/usr/bin/env python
"""
Either install vircampype or add root to PYTHONPATH via
export PYTHONPATH="/Users/stefan/Dropbox/Projects/vircampype/":$PYTHONPATH}
"""
import sys
import argparse
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
if "calibration" in pipeline.setup.name.lower():
    pipeline.build_master_calibration()
else:
    pipeline.process_science()
