import os
import shutil
import numpy as np

from glob import glob
from astropy.io import fits
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from vircampype.external.mmm import mmm
from astropy.convolution import convolve, Gaussian2DKernel
from vircampype.tools.systemtools import get_resource_path, which
from scipy.ndimage import binary_closing, distance_transform_cdt, generate_binary_structure


# Define helper methods
def find_convolution_scale(data, target_rms):
    test_fwhm = np.linspace(0.1, 1, 10)
    # rms_conv = [mmm(convolve(data, Gaussian2DKernel(fwhm)))[1] for fwhm in test_fwhm]

    def get_rms(scale):
        return mmm(convolve(data, Gaussian2DKernel(scale), boundary="extend"))[1]

    # Run convolution jobs in parallel
    with Parallel(n_jobs=10) as parallel:
        rms_conv = parallel(delayed(get_rms)(i) for i in test_fwhm)

    # Interpolate at target RMS
    interp = interp1d(rms_conv, test_fwhm, kind="linear")
    return float(interp(target_rms))


def distance_from_edge(img):
    img = np.pad(img, 1, mode="constant")
    dist = distance_transform_cdt(img, metric="taxicab")
    return dist[1:-1, 1:-1]


# Setup
# weight_edge_gradient = 1000
weight_edge_gradient = None
match_rms = False

# Swarp setup
bin_swarp = which("swarp")
path_config = get_resource_path(package="vircampype.resources.astromatic.swarp.presets", resource="tiles.config")

# Find tiles to swarp
paths_tiles_wide = sorted(glob("/Volumes/Data/VHS/CrA/vircampype/phase3/*_Ks/*tl.fits"))
paths_weights_wide = [pt.replace(".fits", ".weight.fits") for pt in paths_tiles_wide]

# Find deep tiles
# paths_tiles_deep = sorted(glob("/Volumes/Data/Mosaics/CrA/temp/*tl.resamp.fits"))
paths_tiles_deep = []
paths_weights_deep = [pt.replace(".fits", ".weight.fits") for pt in paths_tiles_deep]

paths_tiles_all = paths_tiles_wide + paths_tiles_deep
paths_weights_all = paths_weights_wide + paths_weights_deep

# Generate output paths for all files
paths_tiles_all_out = ["/tmp/{0}".format(os.path.basename(p)) for p in paths_tiles_all]
paths_weights_all_out = ["/tmp/{0}".format(os.path.basename(p)) for p in paths_weights_all]

# Check if weights are there
if np.sum([os.path.isfile(pw) for pw in paths_weights_all]) != len(paths_tiles_all):
    raise ValueError("Not all files have weights")

# Set scale zero points to sun-like stars
jh_sun = 0.286
hk_sun = 0.076
zp_target_j = 25.0
zp_target_h = zp_target_j - jh_sun
zp_target_k = zp_target_h - hk_sun

# Read passbands
passbands = [fits.getheader(pp)["FILTER"] for pp in paths_tiles_all]
if len(set(passbands)) != 1:
    raise ValueError("Mixed filter")
else:
    passband = passbands[0]

# Prepare weights
for pw, pwo in zip(paths_weights_all, paths_weights_all_out):

    if os.path.isfile(pwo):
        print("File '{0}' already exists".format(os.path.basename(pwo)))
        continue

    # Print current filename
    print(os.path.basename(pwo))

    # Just copy file if no edge gradient is requested
    if weight_edge_gradient is None:
        shutil.copyfile(pw, pwo)
        continue

    # Read weight and header
    dw, hw = fits.getdata(pw, header=True)

    # Create binary weight
    dw_flat = np.full_like(dw, fill_value=0, dtype=int)
    dw_flat[dw > 0] = 1
    dw_flat = binary_closing(dw_flat, structure=generate_binary_structure(2, 2), iterations=50)

    # Get distance from edge
    disarr = distance_from_edge(dw_flat)

    # Create distance weights
    dwmod = disarr / weight_edge_gradient
    dwmod[dwmod > 1] = 1

    # Modify original weight
    dwe = dw * dwmod

    # Write new weight
    phdu = fits.PrimaryHDU(data=dwe.astype(np.float32), header=hw)
    phdu.writeto(pwo, overwrite=True)

# Select target ZP
if "j" == passband.lower():
    zp_target = zp_target_j
elif "h" == passband.lower():
    zp_target = zp_target_h
elif "ks" == passband.lower():
    zp_target = zp_target_k
else:
    raise ValueError

# Read ZPs from headers
zp_tiles = [fits.getheader(pp)["AUTOZP"] for pp in paths_tiles_all]

# Compute flux scale for each image
scale_zp = [zp_target - zp for zp in zp_tiles]
flxscl = [10 ** (s / 2.5) for s in scale_zp]
flxscl_swarp = ",".join(["{0:0.5f}".format(s) for s in flxscl])

# Read background sigma and find target RMS
noise_orig = [fits.getheader(f)["BACKSIG"] for f in paths_tiles_wide]
noise_scaled = np.array([a*b for a, b in zip(flxscl, noise_orig)])
rms_target = np.min(noise_scaled)

# Compute and apply convolution scales to wide data
for pt, ptt, sc in zip(paths_tiles_all, paths_tiles_all_out, flxscl):

    # Check if output file already exists
    if os.path.isfile(ptt):
        print("File '{0}' already exists".format(os.path.basename(ptt)))
        continue

    # Just copy deep tiles
    if ("deep" in pt.lower()) | (match_rms is False):
        print("Skipping convolution for file {0}".format(os.path.basename(pt)))
        shutil.copyfile(pt, ptt)
        continue

    # Print current filename
    print(os.path.basename(ptt))

    # Read data and header
    dt, dt_hdr = fits.getdata(pt, header=True)

    # Find convolution scale based on background measurements
    csc = find_convolution_scale(data=dt[1500:-1500:10, 1500:-1500:10] * sc, target_rms=rms_target)
    print("Convolution scale = {0:0.2f}".format(csc))

    # Apply convolution with determined scale
    dt = convolve(dt, kernel=Gaussian2DKernel(csc), boundary="extend")

    # Write new file
    dt_hdr["CSCALE"] = (csc, "Size of convolution kernel for tile matching.")
    phdu = fits.PrimaryHDU(data=dt.astype(np.float32), header=dt_hdr)  # noqa
    phdu.writeto(ptt, overwrite=True)

# Construct swarp command
files = " ".join(paths_tiles_all_out)
weights = ",".join(["{0}".format(pw) for pw in paths_weights_all_out])
cmd = "{0} -c {1} -FSCALE_DEFAULT {2} " \
      "-WEIGHT_IMAGE {3} " \
      "-IMAGEOUT_NAME /Volumes/Data/Mosaics/CrA/coadd.fits " \
      "-WEIGHTOUT_NAME /Volumes/Data/Mosaics/CrA/coadd.weight.fits" \
      " {4}".format(bin_swarp, path_config, flxscl_swarp, weights, files)

print(cmd)
