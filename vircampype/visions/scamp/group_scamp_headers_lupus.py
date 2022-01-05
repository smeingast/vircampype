import glob
from visions.scamp.group_scamp_headers import group_scamp_headers

# Find scripts for VISIONS
path_base = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/Lupus/"
paths_scripts = sorted(glob.glob(path_base + "/**/*yml"))

# Run grouping function
path_gaia_raw = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/" \
                "scamp/Lupus/gaia_edr3_raw_only_deep.fits"
group_scamp_headers(
    paths_scripts=paths_scripts,
    path_gaia_raw=path_gaia_raw,
    folder="/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scamp/Lupus/",
    prepare_scamp=True,
)
