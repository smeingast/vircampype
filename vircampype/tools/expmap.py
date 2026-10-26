"""Exact full-resolution exposure map (ANCILLARY.EXPMAP) of a tile."""

import logging
import os

import numpy as np
from astropy.io import fits

from vircampype.pipeline.errors import PipelineValueError
from vircampype.tools.sdpcheck import median_from_counts, scan_expmap
from vircampype.tools.systemtools import (
    make_path_system_tempfile,
    move_file,
    remove_file,
)
from vircampype.tools.wcstools import integer_wcs_offset

log = logging.getLogger(__name__)

__all__ = ["build_exposure_map"]

_WCS_KEYS = ["CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CRVAL1", "CRVAL2"]
_WCS_KEYS += ["CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"]
_WCS_KEYS += ["EQUINOX", "RADESYS"]


def build_exposure_map(
    paths_weights: list[str],
    exptimes: list[float],
    header: fits.Header,
    primary: fits.Header,
    path_out: str,
    cache_dir: str | None = None,
) -> tuple[float, float]:
    """
    Build the exact exposure map of a tile: the sum of DIT·NDIT over all pawprints
    whose resampled weight is > 0 in a pixel. The map is accumulated in place in a
    memmap of the output file on the cache and moved to ``path_out`` at the end.

    Parameters
    ----------
    paths_weights : list[str]
        Weight MEFs of the resampled pawprints. Every image extension is a sub-raster
        of the tile grid (same CRVAL/CD/CTYPE, integer CRPIX offset).
    exptimes : list[float]
        DIT·NDIT of each pawprint in seconds.
    header : astropy.io.fits.Header
        SCI header of the tile (grid and WCS).
    primary : astropy.io.fits.Header
        Primary header of the exposure map (OBJECT, FILTER).
    path_out : str
        Output path of the exposure map.
    cache_dir : str, optional
        Directory of the temporary file; the system temp directory if None.

    Returns
    -------
    tuple[float, float]
        Median exposure time of the covered pixels (s) and covered sky area (deg²).

    Raises
    ------
    PipelineValueError
        If an extension is not an integer sub-raster of the grid, or nothing is covered.

    """
    # int16 iff every value a pixel can reach is an integer within the int16 range
    integral = all(float(e).is_integer() for e in exptimes)
    dtype = np.int16 if integral and sum(exptimes) <= 32767 else np.float32
    nx, ny = header["NAXIS1"], header["NAXIS2"]

    # Headers: empty primary + EXPMAP extension on the SCI grid
    extension = fits.ImageHDU(data=np.zeros((1, 1), dtype)).header
    extension["NAXIS1"], extension["NAXIS2"] = nx, ny
    extension["EXTNAME"] = "EXPMAP"
    extension["BUNIT"] = ("s", "Exposure time")
    for key in _WCS_KEYS:
        if key in header:
            extension[key] = (header[key], header.comments[key])
    # EXTEND by hand: update_extend() only acts on a list with extensions, and the
    # update-mode close would otherwise insert it into one of the blank cards
    primary = fits.PrimaryHDU(header=primary.copy()).header
    primary.insert("NAXIS", ("EXTEND", True), after=True)
    for hdr in (primary, extension):
        for _ in range(36):
            hdr.append(fits.Card(), end=True)

    # Create the file without allocating its data (sparse zero block)
    path_tmp = make_path_system_tempfile(
        prefix="expmap_", suffix=".fits", base_dir=cache_dir
    )
    try:
        with open(path_tmp, "wb") as f:
            f.write(primary.tostring().encode("ascii"))
            f.write(extension.tostring().encode("ascii"))
            nbytes = nx * ny * np.dtype(dtype).itemsize
            f.truncate(f.tell() + ((nbytes + 2879) // 2880) * 2880)

        with fits.open(path_tmp, mode="update", memmap=True) as hdul:
            data = hdul[1].data
            for path, exptime in zip(paths_weights, exptimes, strict=True):
                value = int(exptime) if dtype is np.int16 else np.float32(exptime)
                with fits.open(path, memmap=True) as weights:
                    for idx, hdu in enumerate(weights[1:], start=1):
                        if not isinstance(hdu, fits.ImageHDU):
                            continue
                        x0, y0 = integer_wcs_offset(header, hdu.header)
                        nyy, nxx = hdu.data.shape
                        if x0 < 0 or y0 < 0 or x0 + nxx > nx or y0 + nyy > ny:
                            raise PipelineValueError(
                                f"{os.path.basename(path)}[{idx}] lies outside the "
                                f"tile grid (offset x0={x0}, y0={y0})"
                            )
                        view = data[y0 : y0 + nyy, x0 : x0 + nxx]
                        view[hdu.data > 0] += value
                log.debug(f"Exposure map: added {os.path.basename(path)} ({exptime} s)")

            # Strip pass: median and covered area
            counts = scan_expmap(data)[2]
        covered = sum(counts.values())
        if covered == 0:
            raise PipelineValueError("exposure map has no covered pixels")
        median = median_from_counts(counts)
        cd = header["CD1_1"] * header["CD2_2"] - header["CD1_2"] * header["CD2_1"]
        area = covered * abs(cd)
        move_file(path_tmp, path_out)
    except Exception:
        remove_file(path_tmp)
        raise
    remove_file(path_tmp)  # left behind only by the cross-filesystem copy
    return float(median), float(area)
