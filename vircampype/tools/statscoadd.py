"""Tile statistics coadd without SWarp.

The per-pawprint statistics images (``build_statistics``) hold one constant value per
detector (nimg = 1, exptime = DIT * NDIT, the integer and fractional MJD parts) plus a
resized weight map. SWarp combines them with ``RESAMPLE=N``, ``CENTER_TYPE=ALL``: each
input is shifted by whole pixels onto the output grid, then SUM (nimg, exptime) or
WEIGHTED (MJD planes). This module reproduces SWarp 2.41.6 bit for bit with numpy,
reading only the weight maps, and works through the output in row strips, so RAM stays
bounded for any coadd size. The output header comes from a SWarp ``HEADER_ONLY`` run.

SWarp rules reproduced (src/coadd.c, field.c, weight.c):
- output CRPIX = max(input CRPIX) per axis (field.c, CENTER_ALL); input pixel 1 lands on
  output pixel floor(CRPIX_out - CRPIX_in + 1) (coadd.c:344)
- input variance = float32(1 / w) for w > 0; valid if variance < BIG (1e30)
- SUM: sum of valid values (double), stored float32; output variance = sum of variances
- WEIGHTED: val * (1 / sum(1/var)) (double), stored float32; output variance 1/sum(1/var)
- output weight = float32(1 / float32 variance), 0 where nothing is valid
- no valid input: SUM gives 0; WEIGHTED gives the value of the first covering input
  (only the first invalid sample enters SWarp's pixel stack, coadd_movewdata); no input: 0
"""

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits

__all__ = [
    "STATISTICS_MODES",
    "StatisticsCoaddUnsupported",
    "coadd_statistics_maps",
    "statistics_output_paths",
]

STATISTICS_MODES = ("nimg", "exptime", "mjd.int", "mjd.frac")
_BIG = np.float32(1e30)  # SWarp's BIG: variance of a bad pixel and the coadd threshold
# Live float64-equivalent bytes per output pixel in a strip: accumulators (6 x 8),
# first-value planes and flags (9), and output temporaries (~70)
_BYTES_PER_PIXEL = 128


class StatisticsCoaddUnsupported(ValueError):
    """The inputs do not match what the numpy coadd reproduces; use SWarp instead."""


@dataclass
class _Input:
    """One detector (HDU) of a per-pawprint statistics image."""

    path_weight: str
    data_offset: int  # byte offset of the weight data in its file
    ny: int
    nx: int
    crpix1: float
    crpix2: float
    values: dict = field(default_factory=dict)  # mode -> constant value
    y0: int = 0  # 0-based output row/column of the input's first pixel
    x0: int = 0


def _scan_pawprint(path_weight, paths_values, header_out):
    """Records of all detectors of one pawprint (geometry + constant values)."""
    records = []
    with fits.open(path_weight, memmap=False, lazy_load_hdus=False) as hw:
        for k in range(1, len(hw)):
            h = hw[k].header
            if h["BITPIX"] != -32 or h["NAXIS"] != 2:
                raise StatisticsCoaddUnsupported(f"{path_weight}[{k}]: not 2D float32")
            for key in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2"):
                if h[key] != header_out[key]:
                    raise StatisticsCoaddUnsupported(
                        f"{path_weight}[{k}]: {key} differs"
                    )
            records.append(
                _Input(
                    path_weight=path_weight,
                    data_offset=hw.fileinfo(k)["datLoc"],
                    ny=h["NAXIS2"],
                    nx=h["NAXIS1"],
                    crpix1=h["CRPIX1"],
                    crpix2=h["CRPIX2"],
                )
            )
    for mode, path in paths_values.items():
        with fits.open(path, memmap=False) as hv:
            if len(hv) - 1 != len(records):
                raise StatisticsCoaddUnsupported(f"{path}: HDU count differs")
            for rec, hdu in zip(records, hv[1:]):
                if (hdu.header["NAXIS2"], hdu.header["NAXIS1"]) != (rec.ny, rec.nx):
                    raise StatisticsCoaddUnsupported(f"{path}: shape differs")
                first, last = hdu.section[0, 0], hdu.section[rec.ny - 1, rec.nx - 1]
                if first != last:
                    raise StatisticsCoaddUnsupported(f"{path}: image is not constant")
                rec.values[mode] = np.float32(first)
    return records


def _place(inputs, header_out):
    """Integer placement of every input on the output grid (SWarp, CENTER_ALL)."""
    for axis in (1, 2):
        crpix_out = max(getattr(r, f"crpix{axis}") for r in inputs)
        crpix_min = min(
            getattr(r, f"crpix{axis}") - getattr(r, f"n{'xy'[axis - 1]}") + 1
            for r in inputs
        )
        # The header stores CRPIX with 13 significant digits (%19.12E)
        if abs(crpix_out - header_out[f"CRPIX{axis}"]) > 1e-11 * max(
            1.0, abs(crpix_out)
        ):
            raise StatisticsCoaddUnsupported(f"CRPIX{axis} is not max(input CRPIX)")
        if int(crpix_out - crpix_min + 1.01) != header_out[f"NAXIS{axis}"]:
            raise StatisticsCoaddUnsupported(
                f"NAXIS{axis} differs from the input extent"
            )
        for r in inputs:
            o = math.floor(crpix_out - getattr(r, f"crpix{axis}") + 1.0) - 1
            if axis == 1:
                r.x0 = o
            else:
                r.y0 = o


def _read_rows(rec, r0, r1, fh):
    """Weight rows r0..r1 (input coordinates) as native float32."""
    fh.seek(rec.data_offset + r0 * rec.nx * 4)
    buf = fh.read((r1 - r0) * rec.nx * 4)
    return np.frombuffer(buf, dtype=">f4").reshape(r1 - r0, rec.nx).astype(np.float32)


class _FitsWriter:
    """Single-HDU FITS file written row block by row block.

    Big-endian data, padded to 2880 bytes. ``bzero`` stores unsigned 16-bit values as
    signed ``BITPIX=16`` with ``BZERO=32768``, as astropy writes uint16.
    """

    def __init__(self, path, header, dtype, bzero=None):
        self.path, self.dtype, self.bzero = path, np.dtype(dtype), bzero
        self.nx, self.ny = header["NAXIS1"], header["NAXIS2"]
        hdr = header.copy()
        hdr["BITPIX"] = {"i2": 16, "f4": -32, "f8": -64}[self.dtype.str[1:]]
        if bzero is not None:
            hdr["BSCALE"], hdr["BZERO"] = 1, bzero
        raw = hdr.tostring().encode("ascii")
        self.hdr_len = len(raw)
        n_data = self.nx * self.ny * self.dtype.itemsize
        with open(path, "wb") as f:
            f.write(raw)
            f.truncate(self.hdr_len + -(-n_data // 2880) * 2880)

    def write_rows(self, y0, rows):
        if self.bzero is not None:
            rows = rows.astype(np.int32) - self.bzero
        data = np.ascontiguousarray(rows, dtype=self.dtype.newbyteorder(">"))
        with open(self.path, "r+b") as f:
            f.seek(self.hdr_len + y0 * self.nx * self.dtype.itemsize)
            f.write(data.data)


def coadd_statistics_maps(
    paths_weight: list[str],
    paths_values: dict[str, list[str]],
    header_out: fits.Header,
    paths_out: dict[str, str],
    mjd_ref: float,
    mem_max: float = 2048,
    n_threads: int = 8,
) -> None:
    """Coadd per-pawprint statistics images into tile statistics maps.

    Parameters
    ----------
    paths_weight : list[str]
        Per-pawprint statistics weight images (``*.weight.fits``), in coadd order.
    paths_values : dict[str, list[str]]
        For each mode in ``STATISTICS_MODES``, the per-pawprint statistics images in
        the same order as ``paths_weight``.
    header_out : fits.Header
        Output header (SWarp ``HEADER_ONLY``); used for every output file.
    paths_out : dict[str, str]
        Output path for each mode, each ``"<mode>.weight"`` and ``"mjd.eff"``
        (see ``statistics_output_paths``).
    mjd_ref : float
        Reference MJD added back to the MJD planes for ``mjd.eff``.
    mem_max : int, optional
        RAM budget (MB) for one output strip.
    n_threads : int, optional
        Threads for reading headers and weights.

    Raises
    ------
    StatisticsCoaddUnsupported
        If the inputs are not what this coadd reproduces (caller falls back to SWarp).
    """
    if set(paths_values) != set(STATISTICS_MODES):
        raise ValueError(f"paths_values must have the modes {STATISTICS_MODES}")
    for mode in STATISTICS_MODES:
        if len(paths_values[mode]) != len(paths_weight):
            raise ValueError(f"{mode}: number of images differs from the weights")

    # Geometry and constants of every detector; map() keeps file-then-HDU order
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        per_file = pool.map(
            lambda i: _scan_pawprint(
                paths_weight[i],
                {m: paths_values[m][i] for m in STATISTICS_MODES},
                header_out,
            ),
            range(len(paths_weight)),
        )
        inputs = [rec for recs in per_file for rec in recs]
    _place(inputs, header_out)

    # Output files in their final types (as convert_bitpix_image/combine_mjd_images)
    ny, nx = header_out["NAXIS2"], header_out["NAXIS1"]
    combinet = {
        "nimg": "SUM",
        "exptime": "SUM",
        "mjd.int": "WEIGHTED",
        "mjd.frac": "WEIGHTED",
    }
    writers = {}
    for mode in STATISTICS_MODES:
        hdr = header_out.copy()
        hdr["COMBINET"] = combinet[mode]
        bzero = 32768 if mode == "nimg" else None
        writers[mode] = _FitsWriter(
            paths_out[mode], hdr, "i2" if bzero else "f4", bzero
        )
        hdr_w = hdr.copy()
        hdr_w["SATURATE"] = 0.0  # SWarp's weight maps carry no saturation level
        writers[f"{mode}.weight"] = _FitsWriter(
            paths_out[f"{mode}.weight"], hdr_w, "f4"
        )
        if mode == "mjd.int":
            hdr_eff = hdr.copy()
            hdr_eff.remove(
                "EXTEND", ignore_missing=True
            )  # as combine_mjd_images writes
            hdr_eff["MJDREF"] = (mjd_ref, "Reference MJD added to the planes")
            writers["mjd.eff"] = _FitsWriter(paths_out["mjd.eff"], hdr_eff, "f8")

    # Row strips sized from the RAM budget
    h_strip = max(1, min(ny, int(mem_max * 2**20) // (nx * _BYTES_PER_PIXEL)))
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        for s0 in range(0, ny, h_strip):
            s1 = min(ny, s0 + h_strip)
            _coadd_strip(inputs, s0, s1, nx, writers, mjd_ref, pool, n_threads)


def _coadd_strip(inputs, s0, s1, nx, writers, mjd_ref, pool, n_threads):
    """Accumulate and write output rows s0..s1."""
    shape = (s1 - s0, nx)
    s_inv, s_var = np.zeros(shape), np.zeros(shape)  # sum 1/var, sum var (valid)
    s_nimg, s_exp = np.zeros(shape), np.zeros(shape)  # SUM planes
    s_mi, s_mf = np.zeros(shape), np.zeros(shape)  # sum value/var (WEIGHTED planes)
    seen = np.zeros(shape, bool)  # covered by any input (also zero weight)
    first_mi = np.zeros(shape, np.float32)  # value of the first covering input
    first_mf = np.zeros(shape, np.float32)

    # Overlapping inputs grouped by file, in input order (SWarp stack order)
    groups, last = [], None
    for rec in inputs:
        if (
            rec.y0 >= s1
            or rec.y0 + rec.ny <= s0
            or rec.x0 >= nx
            or rec.x0 + rec.nx <= 0
        ):
            continue
        if rec.path_weight != last:
            groups.append([])
            last = rec.path_weight
        groups[-1].append(rec)

    def read_group(recs):
        with open(recs[0].path_weight, "rb") as fh:
            return [
                (
                    rec,
                    _read_rows(
                        rec,
                        max(s0, rec.y0) - rec.y0,
                        min(s1, rec.y0 + rec.ny) - rec.y0,
                        fh,
                    ),
                )
                for rec in recs
            ]

    window = 4 * n_threads  # bounded read-ahead
    for i in range(0, len(groups), window):
        for blocks in pool.map(read_group, groups[i : i + window]):
            for rec, w in blocks:
                ya, xa = max(s0, rec.y0), max(0, rec.x0)
                yb, xb = ya + w.shape[0], min(nx, rec.x0 + rec.nx)
                w = w[:, xa - rec.x0 : xb - rec.x0]
                sl = (slice(ya - s0, yb - s0), slice(xa, xb))
                with np.errstate(divide="ignore", over="ignore"):
                    var = np.where(w > 0, np.float32(1) / w, _BIG)
                valid = var < _BIG
                inv = np.where(
                    valid, 1.0 / np.where(valid, var, 1).astype(np.float64), 0.0
                )
                s_inv[sl] += inv
                s_var[sl] += np.where(valid, var, 0).astype(np.float64)
                s_nimg[sl] += valid * np.float64(rec.values["nimg"])
                s_exp[sl] += valid * np.float64(rec.values["exptime"])
                s_mi[sl] += inv * np.float64(rec.values["mjd.int"])
                s_mf[sl] += inv * np.float64(rec.values["mjd.frac"])
                new = ~seen[sl]
                first_mi[sl][new] = rec.values["mjd.int"]
                first_mf[sl][new] = rec.values["mjd.frac"]
                seen[sl] = True

    cov = s_inv > 0
    wval = np.zeros(shape)
    np.divide(1.0, s_inv, out=wval, where=cov)
    mjd_int = np.where(cov, s_mi * wval, first_mi).astype(np.float32)
    mjd_frac = np.where(cov, s_mf * wval, first_mf).astype(np.float32)
    del s_mi, s_mf, first_mi, first_mf, seen
    writers["mjd.int"].write_rows(s0, mjd_int)
    writers["mjd.frac"].write_rows(s0, mjd_frac)
    mjd_eff = mjd_int.astype(np.float64) + mjd_frac.astype(np.float64) + mjd_ref
    mjd_eff[(mjd_int == 0) & (mjd_frac == 0)] = 0.0  # combine_mjd_images
    writers["mjd.eff"].write_rows(s0, mjd_eff)
    del mjd_eff, mjd_int, mjd_frac

    writers["nimg"].write_rows(s0, np.rint(s_nimg.astype(np.float32)))
    writers["exptime"].write_rows(s0, s_exp.astype(np.float32))
    del s_nimg, s_exp

    # Output weights: float32 variance, written as float32(1 / variance)
    for modes, var_out in (
        (("mjd.int", "mjd.frac"), wval),
        (("nimg", "exptime"), s_var),
    ):
        v = np.where(cov, var_out, _BIG).astype(np.float32)
        w_out = np.zeros(shape, np.float32)
        np.divide(np.float32(1), v, out=w_out, where=v < _BIG)
        for mode in modes:
            writers[f"{mode}.weight"].write_rows(s0, w_out)


def statistics_output_paths(path_coadd: str) -> dict[str, str]:
    """Paths of the tile statistics maps next to the coadd."""
    paths = {}
    for mode in STATISTICS_MODES:
        paths[mode] = path_coadd.replace(".fits", f".{mode}.fits")
        paths[f"{mode}.weight"] = path_coadd.replace(".fits", f".{mode}.weight.fits")
    paths["mjd.eff"] = path_coadd.replace(".fits", ".mjd.eff.fits")
    return paths
