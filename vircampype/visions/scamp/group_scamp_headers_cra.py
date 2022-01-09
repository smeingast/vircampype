import os
import glob
from visions.scamp.group_scamp_headers import group_scamp_headers

# Find scripts for VISIONS
path_pipe = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/"
path_base = f"{path_pipe}scripts/CrA/"
path_gaia_raw = f"{path_pipe}scamp/CrA/gaia_edr3_raw.fits"
paths_scripts = sorted(glob.glob(path_base + "/**/*yml"))
# Remove symbolic links
paths_scripts = [ps for ps in paths_scripts if not os.path.islink(ps)]

# Run grouping function
folder = f"{path_pipe}scamp/CrA/"
group_scamp_headers(
    paths_scripts=paths_scripts,
    folder=folder,
    path_gaia_raw=path_gaia_raw,
    prepare_scamp=True,
)
