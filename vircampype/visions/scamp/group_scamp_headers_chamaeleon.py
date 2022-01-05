import glob
from visions.scamp.group_scamp_headers import group_scamp_headers

# Find scripts for VISIONS
path_pipe = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/"
path_scripts = f"{path_pipe}scripts/Chamaeleon/"
paths_scripts = sorted(glob.glob(path_scripts + "/**/*yml"))

# Run grouping function
path_gaia_raw = f"{path_pipe}scamp/Chamaeleon/gaia_edr3_raw_only_deep.fits"
folder = f"{path_pipe}scamp/Chamaeleon/"
group_scamp_headers(
    paths_scripts=paths_scripts,
    folder=folder,
    path_gaia_raw=path_gaia_raw,
    prepare_scamp=True,
)
