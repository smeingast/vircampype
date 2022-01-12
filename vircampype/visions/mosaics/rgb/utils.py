import PIL
import numpy as np
from PIL import Image
from astropy.io import fits
from pathlib import Path
from typing import Generator, List, Union


# Target ZPs
def flux_scale_rgb(zeropoints_rgb):
    zp_target_h, jh_sun, hk_sun = 25.0, 0.286, 0.076
    zp_target = [zp_target_h - hk_sun, zp_target_h, jh_sun + zp_target_h]
    scale_zp = [zpt - zp for zpt, zp in zip(zp_target, zeropoints_rgb)]
    return [10 ** (s / 2.5) for s in scale_zp]


def jpg2fits(
    path_jpg: Path,
    path_fits: Path,
    paths_weights: Union[List, Generator] = None,
    overwrite: bool = False,
):

    # Increase size limit for PIL
    PIL.Image.MAX_IMAGE_PIXELS = 10_000_000_000

    # Read original header
    hdr = fits.getheader(path_fits.absolute())

    # Read RGB and convert to float (necessary to support NaN)
    im_r, im_g, im_b = np.array(Image.open(path_jpg.absolute())).T  # noqa
    im_r = np.flipud(im_r.T).astype(np.float32)
    im_g = np.flipud(im_g.T).astype(np.float32)
    im_b = np.flipud(im_b.T).astype(np.float32)

    # Check dimensions
    shape_fits = (hdr["NAXIS2"], hdr["NAXIS1"])
    if not im_r.shape == shape_fits:
        raise ValueError(
            f"JPG ({im_r.shape}) and FITS ({shape_fits}) dimensions not matching"
        )

    # Find bad values on weights if given
    if paths_weights is not None:

        # Read weights
        weights = [fits.getdata(pw) for pw in paths_weights]

        # Make master weight
        mweight = np.full_like(weights[0], fill_value=1, dtype=int)
        for ww in weights:
            mweight[ww < 0.0001] = 0

        # Mask bad values
        im_r[mweight == 0] = np.nan
        im_g[mweight == 0] = np.nan
        im_b[mweight == 0] = np.nan

    # Save RGB files
    suffix = path_jpg.suffix
    hdu_r = fits.PrimaryHDU(data=im_r, header=hdr)
    hdu_r.writeto(str(path_jpg).replace(suffix, "_R.fits"), overwrite=overwrite)

    hdu_g = fits.PrimaryHDU(data=im_g, header=hdr)
    hdu_g.writeto(str(path_jpg).replace(suffix, "_G.fits"), overwrite=overwrite)

    hdu_b = fits.PrimaryHDU(data=im_b, header=hdr)
    hdu_b.writeto(str(path_jpg).replace(suffix, "_B.fits"), overwrite=overwrite)


def prepare_rgb(path_r, path_g, path_b, zeropoints_rgb=None):

    if zeropoints_rgb is None:
        zeropoints_rgb = [25, 25, 25]

    # Read data
    dr, hr = fits.getdata(path_r, header=True)
    dg, hg = fits.getdata(path_g, header=True)
    db, hb = fits.getdata(path_b, header=True)

    # Read weights
    wr = fits.getdata(path_r.replace(".fits", ".weight.fits"))
    wg = fits.getdata(path_g.replace(".fits", ".weight.fits"))
    wb = fits.getdata(path_b.replace(".fits", ".weight.fits"))

    # Set 0 weight pixels to invalid
    dr[wr < 0.0001] = np.nan
    dg[wg < 0.0001] = np.nan
    db[wb < 0.0001] = np.nan

    # Scale data to match sun-like ZP
    fscl = flux_scale_rgb(zeropoints_rgb=zeropoints_rgb)
    dr *= fscl[0]
    dg *= fscl[1]
    db *= fscl[2]

    for pp, dd, hh in zip([path_r, path_g, path_b], [dr, dg, db], [hr, hg, hb]):
        phdu = fits.PrimaryHDU(data=dd, header=hh)
        phdu.writeto(pp.replace(".fits", "_scaled.fits"), overwrite=False)


def tile_image(path, direction="y", npieces=2):
    import os
    from tifffile import imread, imwrite
    from vircampype.tools.imagetools import chop_image
    img = imread(path)

    # Get name and extension
    fname, fext = os.path.splitext(path)

    if direction == "y":
        axis = 0
    elif direction == "x":
        axis = 1
    else:
        raise ValueError
    chopped = chop_image(img, axis=axis, npieces=npieces)
    for idx in range(len(chopped)):
        imwrite(f"{fname}_{idx}{fext}", data=chopped[idx])


if __name__ == "__main__":
    tile_image(path="/Volumes/Data/VISIONS/RGB/CrA/mosaic/L.tif",
               direction="y", npieces=3)
