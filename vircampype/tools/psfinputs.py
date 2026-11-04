"""Inputs of the tile PSF photometry (4.0): the tile, its weight, saturation map and
'full' catalog, and the detector stacks the PSF models are built from.

Every stack extension must lie on the tile grid at an integer pixel offset, carry its
own SATURATE and GAIN, and have the band of the tile. Anything missing or inconsistent
raises, naming the file.
"""

import os
from dataclasses import dataclass

import numpy as np
from astropy.io import fits

from vircampype.pipeline.errors import PipelineFileNotFoundError, PipelineValueError
from vircampype.tools.satmap import saturation_map_path
from vircampype.tools.wcstools import integer_wcs_offset

__all__ = ["BANDS", "PsfInputs", "StackHdu", "psf_inputs", "read_band", "stack_hdus"]

BANDS = ("J", "H", "Ks")

# Catalog columns every later step relies on (others are checked where they are used)
CATALOG_COLUMNS = ("NUMBER", "XWIN_IMAGE", "YWIN_IMAGE")


@dataclass(frozen=True)
class StackHdu:
    """One detector of one stack, placed on the tile grid."""

    path: str
    hdu: int
    path_weight: str
    path_satmap: str | None
    offset: int
    detector: int
    x0: int  # 0-based tile column of the stack pixel (1, 1)
    y0: int  # 0-based tile row of the stack pixel (1, 1)
    shape: tuple[int, int]  # (rows, columns)
    saturate: float
    gain: float


@dataclass(frozen=True)
class PsfInputs:
    """Everything the PSF photometry of one tile reads, checked for consistency."""

    band: str
    path_tile: str
    path_weight: str
    path_satmap: str | None
    path_catalog: str
    shape: tuple[int, int]  # (rows, columns)
    saturate: float
    gain: float
    n_sources: int
    stacks: tuple[StackHdu, ...]


def read_band(header: fits.Header, name: str) -> str:
    """Band from FILTER and/or HIERARCH ESO INS FILT1 NAME (J, H or Ks)."""
    keys = ("FILTER", "HIERARCH ESO INS FILT1 NAME")
    values = {str(header[k]).strip() for k in keys if header.get(k)}
    lookup = {b.upper(): b for b in BANDS}
    bands = {lookup.get(v.upper()) for v in values}
    if len(bands) != 1 or None in bands:
        raise PipelineValueError(f"{name}: no unique J/H/Ks band in {sorted(values)}")
    return bands.pop()


def _positive(header: fits.Header, key: str, name: str) -> float:
    try:
        value = float(header[key])
    except (KeyError, TypeError, ValueError):
        raise PipelineValueError(f"{name}: {key} missing or not a number") from None
    if not (np.isfinite(value) and value > 0):
        raise PipelineValueError(f"{name}: {key} = {value} is not positive")
    return value


def _require(path: str) -> str:
    if not os.path.isfile(path):
        raise PipelineFileNotFoundError(f"PSF photometry input not found: {path}")
    return path


def _offset(reference: fits.Header, header: fits.Header, name: str) -> tuple[int, int]:
    """integer_wcs_offset with the file named in the error."""
    try:
        return integer_wcs_offset(reference, header)
    except PipelineValueError as e:
        raise PipelineValueError(f"{name}: {e}") from None


def _same_grid(reference: fits.Header, header: fits.Header, name: str) -> None:
    """Same shape and WCS as the reference (offset 0)."""
    shape = (header["NAXIS2"], header["NAXIS1"])
    if shape != (reference["NAXIS2"], reference["NAXIS1"]):
        raise PipelineValueError(f"{name}: shape {shape} differs from its image")
    if _offset(reference, header, name) != (0, 0):
        raise PipelineValueError(f"{name}: not on the grid of its image")


def stack_hdus(
    paths_stacks: list[str], tile_header: fits.Header, band: str, satmaps: bool = True
) -> tuple[StackHdu, ...]:
    """
    All detector extensions of the stack MEFs (``<name>_NN.stack.fits``, NN = offset;
    extension k = detector k) with their weights (``.weight.fits``) and, if
    ``satmaps``, saturation maps (``.satmap.fits``), placed on the tile grid.
    """
    out = []
    for path in paths_stacks:
        name = os.path.basename(path)
        try:
            offset = int(name.removesuffix(".stack.fits").rsplit("_", 1)[1])
        except (IndexError, ValueError):
            raise PipelineValueError(f"{name}: no offset index in the name") from None
        path_weight = _require(path.removesuffix(".fits") + ".weight.fits")
        path_satmap = _require(saturation_map_path(path)) if satmaps else None
        with fits.open(_require(path), memmap=True) as hdul:
            headers = [h.header for h in hdul[1:]]
            if read_band(hdul[0].header, name) != band:
                raise PipelineValueError(f"{name}: band differs from the tile ({band})")
        others = [(path_weight, "weight")]
        if path_satmap is not None:
            others.append((path_satmap, "satmap"))
        for path_other, kind in others:
            with fits.open(path_other, memmap=True) as hdul:
                if len(hdul) - 1 != len(headers):
                    raise PipelineValueError(f"{name}: {kind} has another layout")
                for idx, header in enumerate(headers, start=1):
                    _same_grid(header, hdul[idx].header, f"{name} {kind}[{idx}]")
        for idx, header in enumerate(headers, start=1):
            label = f"{name}[{idx}]"
            if header.get("OFFSET_I") != offset:
                raise PipelineValueError(f"{label}: OFFSET_I differs from the name")
            x0, y0 = _offset(tile_header, header, label)
            out.append(
                StackHdu(
                    path=path,
                    hdu=idx,
                    path_weight=path_weight,
                    path_satmap=path_satmap,
                    offset=offset,
                    detector=idx,
                    x0=x0,
                    y0=y0,
                    shape=(header["NAXIS2"], header["NAXIS1"]),
                    saturate=_positive(header, "SATURATE", label),
                    gain=_positive(header, "GAIN", label),
                )
            )
    if len({s.offset for s in out}) != len(paths_stacks):
        raise PipelineValueError("Stacks: one file per offset expected")
    return tuple(out)


def psf_inputs(
    path_tile: str,
    paths_stacks: list[str],
    satmaps: bool = True,
    stacks: tuple[StackHdu, ...] | None = None,
) -> PsfInputs:
    """
    Collect and check the inputs of one tile: ``<tile>.fits`` with ``.weight.fits``,
    ``.satmap.fits`` (if ``satmaps``) and ``.full.fits.ctab`` next to it, and the
    stacks. ``stacks`` replaces the stack MEFs by an explicit list (development runs
    on stacks built outside the pipeline).
    """
    name = os.path.basename(path_tile)
    header = fits.getheader(_require(path_tile), 0)
    band = read_band(header, name)
    path_weight = _require(path_tile.removesuffix(".fits") + ".weight.fits")
    _same_grid(header, fits.getheader(path_weight, 0), f"{name} weight")
    path_satmap = None
    if satmaps:
        path_satmap = _require(saturation_map_path(path_tile))
        _same_grid(header, fits.getheader(path_satmap, 0), f"{name} satmap")
    path_catalog = _require(path_tile.removesuffix(".fits") + ".full.fits.ctab")
    cat_header = fits.getheader(path_catalog, 2)
    columns = {cat_header[f"TTYPE{i}"] for i in range(1, cat_header["TFIELDS"] + 1)}
    required = CATALOG_COLUMNS + (("IMAFLAGS_ISO",) if satmaps else ())
    missing = [c for c in required if c not in columns]
    if missing:
        raise PipelineValueError(f"{os.path.basename(path_catalog)}: no {missing}")
    if stacks is not None and paths_stacks:
        raise PipelineValueError("Give either stack files or a stack list, not both")
    if stacks is None:
        stacks = stack_hdus(paths_stacks, header, band, satmaps=satmaps)
    if not stacks:
        raise PipelineValueError(f"{name}: no stacks")
    return PsfInputs(
        band=band,
        path_tile=path_tile,
        path_weight=path_weight,
        path_satmap=path_satmap,
        path_catalog=path_catalog,
        shape=(header["NAXIS2"], header["NAXIS1"]),
        saturate=_positive(header, "SATURATE", name),
        gain=_positive(header, "GAIN", name),
        n_sources=int(cat_header["NAXIS2"]),
        stacks=stacks,
    )
