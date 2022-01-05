# Import
import os
import glob

from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.gaia import Gaia
from joblib import Parallel, delayed
from astropy.coordinates import SkyCoord
from vircampype.tools.systemtools import read_yml
from astropy.wcs.utils import wcs_to_celestial_frame

# Login to Gaia archin
Gaia.login(user="smeing01", password="1hGFjW?vjKxJpEL")

# Setup
out_path = "/Users/stefan/Dropbox/Data/Gaia/EDR3/Download/"

# Change to directory
os.chdir(out_path)

# Find scripts
path_base = "/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scripts/Ophiuchus/"
paths_scripts = sorted(glob.glob(path_base + "/**/*yml"))

# Find tile centers
sc_tile_centers = []
for ps in paths_scripts:
    yml = read_yml(ps)
    th = glob.glob(f"{yml['path_pype']}{yml['name']}/tile/*.ahead")
    if len(th) != 1:
        raise ValueError(f"Tile header not found for field '{yml['name']}'")
    th = th[0]
    hdr = fits.Header.fromtextfile(th)
    ww = WCS(hdr)
    frame = wcs_to_celestial_frame(ww)
    sc_tile_centers.append(
        SkyCoord(
            *ww.wcs_pix2world(hdr["NAXIS1"] / 2, hdr["NAXIS2"] / 2, 1),
            frame=frame,
            unit="deg",
        )
    )


# Merge into single SkyCoord instance
sc_tile_centers = SkyCoord(sc_tile_centers)

# Make lon/lat arrays
cra_all, cdec_all = sc_tile_centers.icrs.ra.degree, sc_tile_centers.icrs.dec.degree

# Number of parallel jobs
n_jobs = 12


# Build queries
queries, out_files = [], []
for idx in range(len(cra_all)):

    # Get limits
    cra, cdec = cra_all[idx], cdec_all[idx]

    out_file = out_path + f"{idx:05d}_ra{cra:0>6.2f}_dec{cdec:0>6.2f}.fits"

    # Skip if file exists already
    if os.path.isfile(out_file):
        continue

    # Build query
    queries.append(
        f"SELECT * FROM gaiaedr3.gaia_source WHERE "  # noqa
        f"1=CONTAINS(POINT('ICRS',{cra},{cdec}), CIRCLE('ICRS',ra,dec, 1.5)))"
    )

    # Build output filename
    out_files.append(out_file)

    # Run query
    # Gaia.launch_job_async(
    #     queries[-1], dump_to_file=True, output_format="fits", output_file=out_file
    # )
    # exit()


def _mp_query(query, output_file):
    """parallelization function to query database"""
    Gaia.launch_job_async(
        query, dump_to_file=True, output_format="fits", output_file=output_file
    )


# Run jobs in parallel
with Parallel(n_jobs=n_jobs) as parallel:
    mp = parallel(
        delayed(_mp_query)(q, f) for q, f, _ in zip(queries, out_files, tqdm(queries))
    )


# Run jobs in a loop
# for q, f in zip(queries, out_files):
#     _mp_query(query=q, output_file=f)
