import glob
from visions.scamp.group_scamp_headers import group_scamp_headers

# Find scripts for VISIONS
path_scripts = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/Pipe/deep/"
paths_scripts = sorted(glob.glob(path_scripts + "/*yml"))

# Gaia raw catalog
path_gaia_raw = (
    "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scamp/Pipe/gaia_edr3_raw.fits"
)

# Run grouping function
folder = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scamp/Pipe/deep/"
group_scamp_headers(
    paths_scripts=paths_scripts,
    path_gaia_raw=path_gaia_raw,
    folder=folder,
    prepare_scamp=True,
)
