import glob
from vircampype.visions.scripts.scamp.group_scamp_headers import group_scamp_headers

# Find scripts for VISIONS
path_base = "/Users/stefan/Dropbox/Projects/VISIONS/scripts/VISIONS/CrA/CrA_control/"
paths_scripts = sorted(glob.glob(path_base + "/*yml"))

# Run grouping function
folder = "/Users/stefan/Dropbox/Projects/VISIONS/Scamp/CrA_control/"
group_scamp_headers(paths_scripts=paths_scripts, folder=folder, prepare_scamp=True)
