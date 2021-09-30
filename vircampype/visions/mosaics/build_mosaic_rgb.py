import os
import shutil
import numpy as np

from typing import List
from astropy.io import fits
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from vircampype.external.mmm import mmm
from vircampype.tools.systemtools import make_folder
from vircampype.tools.systemtools import run_command_shell
from astropy.convolution import convolve, Gaussian2DKernel
from vircampype.tools.systemtools import get_resource_path, which
from scipy.ndimage import binary_closing, distance_transform_cdt, generate_binary_structure

__all__ = ["find_convolution_scale", "distance_from_edge", "add_edge_gradient", "build_mosaic_rgb"]


# Define helper methods
def find_convolution_scale(data, target_rms):

    # FWHM probe range
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


def add_edge_gradient(img, npix):

    # Create binary weight and fill holes
    img_flat = np.full_like(img, fill_value=0, dtype=int)
    img_flat[img > 0] = 1
    img_flat = binary_closing(img_flat, structure=generate_binary_structure(2, 2), iterations=50)

    # Get distance from edge
    disarr = distance_from_edge(img_flat)

    # Create distance weights
    dwmod = disarr / npix
    dwmod[dwmod > 1] = 1

    # Modify original weight
    return img * dwmod


def build_mosaic_rgb(paths_tiles_wide: List, paths_tiles_deep: List, path_mosaic: str, name_mosaic: str,
                     field_name: str, weight_edge_gradient: (int, float) = 1000):

    # Check and make mosaic path
    if not path_mosaic.endswith("/"):
        path_mosaic += "/"
    make_folder(path_mosaic)

    # Find Swarp exe
    bin_swarp = which("swarp")

    # Load mosaic header from pipeline package
    path_config = get_resource_path(package="vircampype.resources.astromatic.swarp.presets", resource="tiles.config")
    path_header_mosaic = get_resource_path(package="visions.headers", resource=f"{field_name}.header")

    # Check header
    if isinstance(fits.Header.fromtextfile(path_header_mosaic), fits.Header):
        pass

    # Find wide weights and source catalogs
    paths_weights_wide = [pt.replace(".fits", ".weight.fits") for pt in paths_tiles_wide]
    paths_catalogs_wide = [pt.replace(".fits", ".full.fits.ctab") for pt in paths_tiles_wide]

    # Check if wide weights and catalogs exist
    if len(paths_tiles_wide) != len([os.path.isfile(p) for p in paths_weights_wide]):
        raise ValueError("Tiles and weights for wide fields do not match")
    if len(paths_tiles_wide) != len([os.path.isfile(p) for p in paths_catalogs_wide]):
        raise ValueError("Catalogs for wide fields do not match tiles")

    # Find deep weights and source catalogs
    paths_weights_deep = [pt.replace(".fits", ".weight.fits") for pt in paths_tiles_deep]
    paths_catalogs_deep = [pt.replace(".fits", ".full.fits.ctab") for pt in paths_tiles_deep]

    # Check if deep weights and catalogs exist
    if len(paths_tiles_deep) != len([os.path.isfile(p) for p in paths_tiles_deep]):
        raise ValueError("Tiles and weights for deep fields do not match")
    if len(paths_tiles_deep) != len([os.path.isfile(p) for p in paths_catalogs_deep]):
        raise ValueError("Catalogs for wide fields do not match tiles")

    # Require input to continue
    print(f"Found {len(paths_tiles_wide)} wide tiles")
    print(f"Found {len(paths_tiles_deep)} deep tiles")
    if (input("Continue (Y/n)") or "Y") != "Y":
        exit()

    # Combine wide and deep paths
    paths_tiles_all = paths_tiles_wide + paths_tiles_deep
    paths_weights_all = paths_weights_wide + paths_weights_deep
    paths_catalogs_all = paths_catalogs_wide + paths_catalogs_deep

    # Generate output paths for all files
    paths_tiles_all_out = [f"{path_mosaic}{os.path.basename(p)}" for p in paths_tiles_all]
    paths_weights_all_out = [f"{path_mosaic}{os.path.basename(p)}" for p in paths_weights_all]

    # Check if weights are there
    if np.sum([os.path.isfile(pw) for pw in paths_weights_all]) != len(paths_tiles_all):
        raise ValueError("Not all files have weights")

    # Set zero point target scale sun-like stars
    zp_target_h, jh_sun, hk_sun = 25.0, 0.286, 0.076
    zp_target = {"J": jh_sun + zp_target_h, "H": zp_target_h, "Ks": zp_target_h - hk_sun}

    # Read passbands from tiles
    passbands = [fits.getheader(pp)["HIERARCH ESO INS FILT1 NAME"] for pp in paths_tiles_all]

    # There can be only 1 passband
    if len(set(passbands)) != 1:
        raise ValueError("Mixed filter")
    else:
        passband = passbands[0]

    # Prepare weights
    print("\nPreparing weights")
    for pw, pwo in zip(paths_weights_all, paths_weights_all_out):

        # Check if weight already exists
        if os.path.isfile(pwo):
            print(f"Weight '{os.path.basename(pwo)}' already exists")
            continue

        # Print current filename
        print(os.path.basename(pwo))

        # Just copy weight file if no edge gradient is requested
        if weight_edge_gradient is None:
            shutil.copyfile(pw, pwo)
            continue

        # Read weight and header
        dw, hw = fits.getdata(pw, header=True)

        # Add edge gradient
        dwe = add_edge_gradient(img=dw, npix=weight_edge_gradient)

        # Write new weight
        phdu = fits.PrimaryHDU(data=dwe.astype(np.float32), header=hw)
        phdu.writeto(pwo, overwrite=True)

    # Read ZP for all tiles from source catalogs
    zp_tiles = [fits.getheader(pp, 2)["HIERARCH PYPE ZP MAG_AUTO"] for pp in paths_catalogs_all]

    # Compute flux scale for each image
    print(f"Target ZP = {zp_target[passband]:0.3f} mag")
    scale_zp = [zp_target[passband] - zp for zp in zp_tiles]
    flxscl = [10**(s / 2.5) for s in scale_zp]
    # flxscl_swarp = ",".join([f"{s:0.5f}" for s in flxscl])

    # Read background sigma and find target RMS (only for wide tiles)
    backsig_orig = [fits.getheader(f)["BACKSIG"] for f in paths_tiles_wide]
    backsig_scaled = np.array([a*b for a, b in zip(flxscl, backsig_orig)])
    backsig_target = np.min(backsig_scaled)

    # Compute and apply convolution scales to wide data
    print("\nNoise equalization")
    for pt, ptt, fscl in zip(paths_tiles_all, paths_tiles_all_out, flxscl):

        # Check if output file already exists
        if os.path.isfile(ptt):
            print(f"File '{os.path.basename(ptt)}' already exists")
            continue

        # Print current filename
        print(f"\n{os.path.basename(ptt)}")
        print(f"Flux scale = {fscl:0.3f}")

        # Read data and header for current tile
        data_tile, hdr_tile = fits.getdata(pt, header=True)

        # Apply flux scale from ZP scaling
        data_tile *= fscl

        # Get a smaller view of the data array for statistics
        if data_tile.size > 50_000_000:  # noqa
            data_tile_s = data_tile[::data_tile.size // 50_000_000]
        else:
            data_tile_s = data_tile

        # Skip convolution for deep tiles
        if "deep" in pt.lower():
            print(f"Skipping convolution for file {os.path.basename(pt)}")
            cscl = 0.0

        # Otherwise find convolution scale based on background measurements
        else:
            cscl = find_convolution_scale(data=data_tile_s, target_rms=backsig_target)
            print(f"Convolution scale = {cscl:0.3f}")

            # Apply convolution with determined scale
            data_tile = convolve(data_tile, kernel=Gaussian2DKernel(cscl), boundary="extend")

            # Recompute background level
            if data_tile.size > 50_000_000:  # noqa
                data_tile_s = data_tile[::data_tile.size // 50_000_000]  # noqa
            else:
                data_tile_s = data_tile

        # Determine background statistics
        backmod, backsig, backskew = mmm(data_tile_s)

        # Subtract overall background level
        data_tile -= backmod

        # Update header
        hdr_tile["NBACKMOD"] = (0.0, "Background mode")
        hdr_tile["NBACKSIG"] = (backsig, "Background sigma")
        hdr_tile["TBACKSIG"] = (backsig_target, "Target background sigma")
        hdr_tile["NBACKSKE"] = (backskew, "Background skew")
        hdr_tile["CSCL"] = (cscl, "Size of convolution kernel for tile matching")
        hdr_tile["FSCL"] = (fscl, "Flux scaling for ZP equalization")

        # Write new file
        phdu = fits.PrimaryHDU(data=data_tile.astype(np.float32), header=hdr_tile)  # noqa
        phdu.writeto(ptt, overwrite=True)

    # Construct swarp command
    files = " ".join(paths_tiles_all_out)
    weights = ",".join([f"{pw}" for pw in paths_weights_all_out])
    cmd = f"{bin_swarp} -c {path_config} " \
          f"-IMAGEOUT_NAME {path_mosaic}{name_mosaic}.fits " \
          f"-WEIGHTOUT_NAME {path_mosaic}{name_mosaic}.weight.fits " \
          f"-HEADEROUT_NAME {path_header_mosaic} " \
          f"-WEIGHT_IMAGE {weights} " \
          f"{files}"

    # Run Swarp
    if (input("Run Swarp (Y/n)") or "Y") == "Y":
        run_command_shell(cmd=cmd, silent=False)
    else:
        print(cmd)

    # Return command
    return cmd
