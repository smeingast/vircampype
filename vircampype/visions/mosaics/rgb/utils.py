import PIL
import numpy as np
from PIL import Image
from astropy.io import fits
from pathlib import Path
from typing import Generator, List, Union


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
