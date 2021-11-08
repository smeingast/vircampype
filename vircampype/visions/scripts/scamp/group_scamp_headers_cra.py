import glob
from vircampype.visions.scripts.scamp.group_scamp_headers import group_scamp_headers

# Find scripts for VISIONS
""" THIS WAS RUN BEFORE THE CONTROL FIELD SCRIPTS WERE THERE. THESE SHOULD BE EXCLUDED """
path_base = "/Users/stefan/Dropbox/Projects/VISIONS/scripts/VISIONS/CrA/"
paths_scripts = sorted(glob.glob(path_base + "/**/*yml"))

# Find scripts for VHS and append
path_base = "/Users/stefan/Dropbox/Projects/VISIONS/scripts/VHS/CrA/"
paths_scripts += sorted(glob.glob(path_base + "/**/*yml"))

# Run grouping function
folder = "/Users/stefan/Dropbox/Projects/VISIONS/Scamp/CrA/"
group_scamp_headers(paths_scripts=paths_scripts, folder=folder, prepare_scamp=True)
