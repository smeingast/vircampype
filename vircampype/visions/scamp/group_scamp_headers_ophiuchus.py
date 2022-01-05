import glob
from visions.scamp.group_scamp_headers import group_scamp_headers

# Find scripts for VISIONS
path_base = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/Ophiuchus/"
paths_scripts = sorted(glob.glob(path_base + "/**/*yml"))

# Run grouping function
path_gaia_raw = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scamp/Ophiuchus/gaia_edr3_raw_only_deep.fits"
folder = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scamp/Ophiuchus/"
group_scamp_headers(paths_scripts=paths_scripts, folder=folder,
                    path_gaia_raw=path_gaia_raw, prepare_scamp=True)
