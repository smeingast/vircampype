"""Saturation maps: how many input exposures were saturated at each pixel.

A coadd has one SATURATE (SWarp writes the lowest of its inputs), far below where most
inputs saturate, and the coadd peak of a star is diluted by seeing and jitter, so no
single level flags saturated stars correctly. Instead every resampled pawprint gets a
0/1 map of its pixels at or above SATURATION_FRACTION x its own SATURATE (weight > 0),
plus the hollow cores enclosed by them, and the maps of stacks and tiles count those
maps on their grid. SExtractor reads the count map as FLAG_IMAGE (FLAG_TYPE MAX):
IMAFLAGS_ISO is the largest number of saturated inputs under a source.
"""

import logging
import os

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_fill_holes

from vircampype.pipeline.errors import PipelineValueError
from vircampype.tools.systemtools import (
    make_path_system_tempfile,
    move_file,
    remove_file,
)
from vircampype.tools.wcstools import integer_wcs_offset

log = logging.getLogger(__name__)

__all__ = [
    "SATURATION_FRACTION",
    "add_saturation_counts",
    "build_pawprint_saturation_map",
    "build_saturation_count_map",
    "map_header",
    "saturation_map_path",
]

# Pixels at or above this fraction of SATURATE count as saturated: peaks of
# marginally saturated stars lose a few percent in the resampling (Lanczos)
SATURATION_FRACTION = 0.95

_WCS_KEYS = ["CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CRVAL1", "CRVAL2"]
_WCS_KEYS += ["CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"]
_WCS_KEYS += ["EQUINOX", "RADESYS"]


def saturation_map_path(path_image: str) -> str:
    """Saturation map of a stack or tile image: ``<name>.satmap.fits`` next to it."""
    return path_image.removesuffix(".fits") + ".satmap.fits"


def map_header(header: fits.Header, data_shape: tuple[int, int]) -> fits.Header:
    """Header of a uint8 map on the grid of ``header``."""
    hdr = fits.ImageHDU(data=np.zeros((1, 1), np.uint8)).header
    hdr["NAXIS1"], hdr["NAXIS2"] = data_shape[1], data_shape[0]
    for key in _WCS_KEYS:
        if key in header:
            hdr[key] = (header[key], header.comments[key])
    hdr["BUNIT"] = ("", "Number of saturated input exposures")
    return hdr


def _write_via_cache(hdul: fits.HDUList, path_out: str, cache_dir: str | None):
    """Write through a local temporary file and move into place."""
    path_tmp = make_path_system_tempfile(
        prefix="satmap_", suffix=".fits", base_dir=cache_dir
    )
    try:
        hdul.writeto(path_tmp, overwrite=True)
        move_file(path_tmp, path_out)
    finally:
        remove_file(path_tmp)


def build_pawprint_saturation_map(
    path_image: str, path_weight: str, path_out: str, cache_dir: str | None = None
) -> int:
    """
    0/1 map of one resampled pawprint: pixels at or above SATURATION_FRACTION x the
    SATURATE of their extension plus the pixels they enclose, with weight > 0. One uint8
    extension per detector on the detector's grid.

    Returns
    -------
    int
        Number of flagged pixels (saturated and enclosed).

    Raises
    ------
    PipelineValueError
        If an image extension has no SATURATE or image and weight differ in layout.
    """
    out = fits.HDUList([fits.PrimaryHDU()])
    n_sat = 0
    with (
        fits.open(path_image, memmap=True) as hdul,
        fits.open(path_weight, memmap=True) as hdul_w,
    ):
        if len(hdul) != len(hdul_w):
            raise PipelineValueError(
                f"{os.path.basename(path_image)} and its weight differ in extensions"
            )
        for idx in range(1, len(hdul)):
            hdr, data, weight = hdul[idx].header, hdul[idx].data, hdul_w[idx].data
            if data.shape != weight.shape:
                raise PipelineValueError(
                    f"{os.path.basename(path_image)}[{idx}]: image {data.shape}, "
                    f"weight {weight.shape}"
                )
            if "SATURATE" not in hdr:
                raise PipelineValueError(
                    f"{os.path.basename(path_image)}[{idx}] has no SATURATE"
                )
            level = SATURATION_FRACTION * float(hdr["SATURATE"])
            # Correlated double sampling: the cores of the brightest stars are already
            # saturated at the first read and fall below the level (down to negative
            # values) inside a ring of saturated pixels. Fill the holes enclosed by the
            # ring; masked or non-finite pixels may close a ring but are never flagged.
            mask = binary_fill_holes(data >= level) & np.isfinite(data) & (weight > 0)
            n_sat += int(mask.sum())
            hdr_map = map_header(hdr, mask.shape)
            hdr_map["SATLEVEL"] = (round(level, 3), "Level of a saturated pixel (ADU)")
            out.append(fits.ImageHDU(mask.astype(np.uint8), header=hdr_map))
    _write_via_cache(out, path_out, cache_dir)
    return n_sat


def add_saturation_counts(
    counts: np.ndarray,
    header: fits.Header,
    paths_maps: list[str],
    hdu_index: int | None = None,
) -> None:
    """
    Add pawprint saturation maps onto a count array on the grid of ``header`` (in
    place, capped at 255). Every map extension lies on that grid at an integer offset;
    parts beyond the edge are clipped and extensions wholly outside are skipped.

    Parameters
    ----------
    counts : np.ndarray
        uint8 count array (may be a memmap) with the shape of the grid.
    header : astropy.io.fits.Header
        Header of the grid (tile image or stack extension).
    paths_maps : list[str]
        Pawprint saturation maps.
    hdu_index : int | None, optional
        Only this extension of every map (stacks: the detector of the stack
        extension); None uses all extensions (tile).
    """
    ny, nx = counts.shape
    for path in paths_maps:
        with fits.open(path, memmap=True) as hdul:
            indices = range(1, len(hdul)) if hdu_index is None else [hdu_index]
            for idx in indices:
                hdu = hdul[idx]
                x0, y0 = integer_wcs_offset(header, hdu.header)
                nyy, nxx = hdu.data.shape
                xa, ya = max(x0, 0), max(y0, 0)
                xb, yb = min(x0 + nxx, nx), min(y0 + nyy, ny)
                if xa >= xb or ya >= yb:
                    continue
                sat = hdu.data[ya - y0 : yb - y0, xa - x0 : xb - x0] > 0
                if not sat.any():
                    continue
                view = counts[ya:yb, xa:xb]
                view[sat & (view < 255)] += 1


def build_saturation_count_map(
    paths_maps: list[str],
    header: fits.Header,
    path_out: str,
    cache_dir: str | None = None,
) -> int:
    """
    Count map of a tile (or mosaic): the number of pawprints saturated in each pixel,
    one uint8 primary HDU on the grid of the tile image, accumulated in a memmap of
    the output file on the cache and moved into place.

    Returns
    -------
    int
        Number of pixels saturated in at least one pawprint.
    """
    nx, ny = header["NAXIS1"], header["NAXIS2"]
    hdr = fits.PrimaryHDU(data=np.zeros((1, 1), np.uint8)).header
    hdr_map = map_header(header, (ny, nx))
    for card in hdr_map.cards:
        if card.keyword not in ("XTENSION", "PCOUNT", "GCOUNT"):
            hdr.set(card.keyword, card.value, card.comment)
    hdr["NAXIS1"], hdr["NAXIS2"] = nx, ny
    path_tmp = make_path_system_tempfile(
        prefix="satmap_", suffix=".fits", base_dir=cache_dir
    )
    try:
        with open(path_tmp, "wb") as f:  # sparse zero block, no data allocated
            f.write(hdr.tostring().encode("ascii"))
            f.truncate(f.tell() + ((nx * ny + 2879) // 2880) * 2880)
        with fits.open(path_tmp, mode="update", memmap=True) as hdul:
            counts = hdul[0].data
            add_saturation_counts(counts, header, paths_maps)
            n_flagged = int(
                sum((counts[r : r + 1024] > 0).sum() for r in range(0, ny, 1024))
            )
        move_file(path_tmp, path_out)
    except Exception:
        remove_file(path_tmp)
        raise
    remove_file(path_tmp)  # left behind only by the cross-filesystem copy
    log.info(f"Saturation map {os.path.basename(path_out)}: {n_flagged} pixels")
    return n_flagged
