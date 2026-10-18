"""Tile statistics coadd without SWarp.

The per-pawprint statistics images (``build_statistics``) hold one constant value per
detector (nimg = 1, exptime = DIT * NDIT, the integer and fractional MJD parts) plus a
resized weight map. SWarp combines them with ``RESAMPLE=N``, ``CENTER_TYPE=ALL``: each
input is shifted by whole pixels onto the output grid, then SUM (nimg, exptime) or
WEIGHTED (MJD planes). This module reproduces SWarp 2.41.6 bit for bit with numpy,
reading only the weight maps, and works through the output in row strips, so RAM stays
bounded for any coadd size. The output header is built as SWarp writes it (header.c,
fitswcs.c), except SOFTNAME/SOFTVERS, which name vircampype.

SWarp rules reproduced (src/coadd.c, field.c, weight.c):
- output WCS = the first input's; CRPIX = max(input CRPIX) and NAXIS =
  int(max - min + 1.01) per axis (field.c, CENTER_ALL); MJD-OBS = earliest input
- input pixel 1 lands on output pixel floor(CRPIX_out - CRPIX_in + 1) (coadd.c:344)
- input variance = float32(1 / w) for w > 0; valid if variance < BIG (1e30)
- SUM: sum of valid values (double), stored float32; output variance = sum of variances
- WEIGHTED: val * (1 / sum(1/var)) (double), stored float32; output variance 1/sum(1/var)
- output weight = float32(1 / float32 variance), 0 where nothing is valid
- no valid input: SUM gives 0; WEIGHTED gives the value of the first covering input
  (only the first invalid sample enters SWarp's pixel stack, coadd_movewdata); no input: 0
"""

import math
import os
import socket
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits

__all__ = [
    "STATISTICS_MODES",
    "StatisticsCoaddUnsupported",
    "coadd_psf_sigma_map",
    "coadd_statistics_maps",
    "statistics_output_paths",
]

STATISTICS_MODES = ("nimg", "exptime", "mjd.int", "mjd.frac")
_BIG = np.float32(1e30)  # SWarp's BIG: variance of a bad pixel and the coadd threshold
# Live float64-equivalent bytes per output pixel in a strip: accumulators (6 x 8),
# first-value planes and flags (9), and output temporaries (~70)
_BYTES_PER_PIXEL = 128
_MJD2000 = 51544.5
# SWarp settings the reproduced rules depend on (coadd.yml preset, as validated)
_PRESET_REQUIRED = {
    "RESAMPLE": "N",
    "CENTER_TYPE": "ALL",
    "IMAGE_SIZE": 0,
    "CELESTIAL_TYPE": "NATIVE",
    "WEIGHT_TYPE": "MAP_WEIGHT",
    "RESCALE_WEIGHTS": "Y",
    "WEIGHT_THRESH": "",
    "BLANK_BADPIXELS": "N",
    "SUBTRACT_BACK": "N",
    "FSCALE_KEYWORD": "FLXSCALE",
    "FSCALE_DEFAULT": 1.0,
    "COPY_KEYWORDS": "OBJECT",
    "WRITE_FILEINFO": "N",
}
# Input keywords that would change SWarp's output; not reproduced
_KEYS_UNSUPPORTED = (
    "EXPTIME",
    "GAIN",
    "SATURATE",
    "OBJECT",
    "MJD-END",
    "MJDSTART",
    "MJDSTOP",
    "DATE-END",
    "CROTA2",
)


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
    mjd_obs: float
    values: dict = field(default_factory=dict)  # mode -> constant value
    wcs: dict | None = None  # WCS keywords (first input only, for the header)
    y0: int = 0  # 0-based output row/column of the input's first pixel
    x0: int = 0


def _wcs_keys(h):
    """WCS keywords SWarp copies from the first input (fitswcs.c read_wcs)."""
    if any(k in h for k in _KEYS_UNSUPPORTED) or any(k.startswith("PV") for k in h):
        raise StatisticsCoaddUnsupported("input header has keywords not reproduced")
    if h.get("FLXSCALE", 1.0) != 1.0:  # SWarp scales values and weights by it
        raise StatisticsCoaddUnsupported("input FLXSCALE != 1 not reproduced")
    if "MJD-OBS" not in h:  # SWarp would fall back to DATE-OBS/MJDSTART
        raise StatisticsCoaddUnsupported("input without MJD-OBS not reproduced")
    if h.get("RADESYS") not in ("ICRS", "FK5"):
        raise StatisticsCoaddUnsupported("input RADESYS not reproduced")
    w = {k: h[k] for k in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2")}
    w["CUNIT1"], w["CUNIT2"] = h.get("CUNIT1", "deg"), h.get("CUNIT2", "deg")
    w["EQUINOX"], w["RADESYS"] = h.get("EQUINOX", 2000.0), h["RADESYS"]
    for i in (1, 2):
        for j in (1, 2):
            if "CD1_1" in h:
                w[f"CD{i}_{j}"] = h.get(f"CD{i}_{j}", 1.0 if i == j else 0.0)
            else:  # CD = CDELT_i * PC_ij
                pc = h.get(f"PC{i}_{j}", 1.0 if i == j else 0.0)
                w[f"CD{i}_{j}"] = pc * h.get(f"CDELT{i}", 1.0)
    return w


def _scan_pawprint(path_weight, paths_values):
    """Records of all detectors of one pawprint (geometry + constant values)."""
    records = []
    with fits.open(path_weight, memmap=False, lazy_load_hdus=False) as hw:
        for k in range(1, len(hw)):
            h = hw[k].header
            if (
                isinstance(hw[k], fits.CompImageHDU)
                or h["BITPIX"] != -32
                or h["NAXIS"] != 2
                or h.get("BSCALE", 1.0) != 1.0
                or h.get("BZERO", 0.0) != 0.0
            ):  # weights are read as raw, uncompressed float32
                raise StatisticsCoaddUnsupported(f"{path_weight}[{k}]: not 2D float32")
            records.append(
                _Input(
                    path_weight=path_weight,
                    data_offset=hw.fileinfo(k)["datLoc"],
                    ny=h["NAXIS2"],
                    nx=h["NAXIS1"],
                    crpix1=h["CRPIX1"],
                    crpix2=h["CRPIX2"],
                    mjd_obs=h.get("MJD-OBS", 0.0),
                    wcs=_wcs_keys(h),
                )
            )
    for mode, path in paths_values.items():
        with fits.open(path, memmap=False) as hv:
            if len(hv) - 1 != len(records):
                raise StatisticsCoaddUnsupported(f"{path}: HDU count differs")
            for rec, hdu in zip(records, hv[1:]):
                # SWarp reads the WCS from the images; they must match the weights
                g = (rec.ny, rec.nx, rec.crpix1, rec.crpix2, rec.mjd_obs)
                hh = hdu.header
                if (
                    hh["NAXIS2"],
                    hh["NAXIS1"],
                    hh["CRPIX1"],
                    hh["CRPIX2"],
                    hh.get("MJD-OBS", 0.0),
                ) != g:
                    raise StatisticsCoaddUnsupported(
                        f"{path}: geometry differs from weight"
                    )
                if _wcs_keys(hh) != rec.wcs:
                    raise StatisticsCoaddUnsupported(f"{path}: WCS differs from weight")
                # build_statistics writes constants (np.full); spot-check corners and
                # centre (a full read would multiply the input I/O by five)
                ny, nx = rec.ny - 1, rec.nx - 1
                first = hdu.section[0, 0]
                for y, x in ((0, nx), (ny, 0), (ny, nx), (ny // 2, nx // 2)):
                    if hdu.section[y, x] != first:
                        raise StatisticsCoaddUnsupported(f"{path}: image not constant")
                rec.values[mode] = np.float32(first)
    return records


def _place(inputs):
    """Output grid and integer placement of every input (SWarp, CENTER_ALL).

    Returns the output CRPIX and NAXIS per axis.
    """
    first = inputs[0].wcs
    for r in inputs:  # SWarp copies the first input's WCS: all must share the frame
        for key in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2"):
            if r.wcs[key] != first[key]:
                raise StatisticsCoaddUnsupported(f"{key} differs between inputs")
    crpix, naxis = {}, {}
    for axis in (1, 2):
        cmax = max(getattr(r, f"crpix{axis}") for r in inputs)
        cmin = min(
            getattr(r, f"crpix{axis}") - getattr(r, ("nx", "ny")[axis - 1]) + 1
            for r in inputs
        )
        crpix[axis], naxis[axis] = cmax, max(1, int(cmax - cmin + 1.01))
        for r in inputs:
            o = math.floor(cmax - getattr(r, f"crpix{axis}") + 1.0) - 1
            if axis == 1:
                r.x0 = o
            else:
                r.y0 = o
    return crpix, naxis


def _card(key, value="", comment="", fmt=None):
    """80-char card image as SWarp's fitswrite formats it."""
    if key == "COMMENT":
        return f"COMMENT {value}".ljust(80)[:80]
    if isinstance(value, str):
        v = f"'{value:<8}'".ljust(20)
    elif isinstance(value, bool):
        v = f"{'T' if value else 'F':>20}"
    elif fmt == "expo":
        v = f"{value:20.12E}"
    elif fmt == "float":
        v = f"{value:20.8f}"
    else:
        v = f"{value:>20}"
    return f"{key:<8}= {v} / {comment}".ljust(80)[:80]


def _output_header(inputs, crpix, naxis, preset, combinet, saturate, software):
    """Output header as SWarp 2.41.6 writes it (RESAMPLE=N, CENTER_TYPE=ALL)."""
    w = inputs[0].wcs
    years = [2000.0 - (_MJD2000 - r.mjd_obs) / 365.25 for r in inputs if r.mjd_obs > 0]
    cards = [
        _card("SIMPLE", True, "This is a FITS file"),
        _card("BITPIX", -32),
        _card("NAXIS", 2),
        _card("NAXIS1", naxis[1], "Number of pixels along this axis"),
        _card("NAXIS2", naxis[2], "Number of pixels along this axis"),
        _card("EXTEND", True, "This file may contain FITS extensions"),
        _card("EQUINOX", float(w["EQUINOX"]), "Mean equinox", "float"),
    ]
    if years:
        mjd = (min(years) - 2000.0) * 365.25 + _MJD2000
        cards.append(_card("MJD-OBS", mjd, "Modified Julian date at start", "float"))
    cards.append(_card("RADESYS", w["RADESYS"], "Astrometric system"))
    for i in (1, 2):
        cards += [
            _card(f"CTYPE{i}", w[f"CTYPE{i}"], "WCS projection type for this axis"),
            _card(f"CUNIT{i}", w[f"CUNIT{i}"], "Axis unit"),
            _card(f"CRVAL{i}", w[f"CRVAL{i}"], "World coordinate on this axis", "expo"),
            _card(f"CRPIX{i}", crpix[i], "Reference pixel on this axis", "expo"),
        ]
        for j in (1, 2):
            cards.append(
                _card(f"CD{i}_{j}", w[f"CD{i}_{j}"], "Linear projection matrix", "expo")
            )
    user = os.environ.get("USERNAME") or os.environ.get("LOGNAME") or "unknown"
    date = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    cards += [
        _card("EXPTIME", 0.0, "Maximum equivalent exposure time (s)", "expo"),
        _card("GAIN", 0.0, "Maximum equivalent gain (e-/ADU)", "expo"),
        _card("SATURATE", float(saturate), "Saturation Level (ADU)", "expo"),
        _card("COMMENT"),
        _card("SOFTNAME", software[0], "The software that processed those data"),
        _card("SOFTVERS", software[1], "Version of the software"),
        _card("COMMENT"),
        _card("AUTHOR", user, "Who ran the software"),
        _card("ORIGIN", socket.gethostname(), "Where it was done"),
        _card("DATE", date, "When it was started (GMT)"),
        _card("COMBINET", combinet, "COMBINE_TYPE config parameter for SWarp"),
        _card("COMMENT"),
        _card("COMMENT", " Propagated FITS keywords"),
        _card("COMMENT"),
        _card("COMMENT", " Axis-dependent config parameters"),
    ]
    for i in (1, 2):
        cards += [
            _card(
                f"RESAMPT{i}",
                preset["RESAMPLING_TYPE"],
                "RESAMPLING_TYPE config parameter",
            ),
            _card(f"CENTERT{i}", preset["CENTER_TYPE"], "CENTER_TYPE config parameter"),
            _card(
                f"PSCALET{i}",
                preset["PIXELSCALE_TYPE"],
                "PIXELSCALE_TYPE config parameter",
            ),
        ]
    return fits.Header.fromstring("".join(cards))


def _read_rows(rec, r0, r1, fh):
    """Weight rows r0..r1 (input coordinates) as native float32."""
    fh.seek(rec.data_offset + r0 * rec.nx * 4)
    buf = bytearray((r1 - r0) * rec.nx * 4)  # read once, byteswap in place
    if fh.readinto(buf) != len(buf):
        raise OSError(f"{rec.path_weight}: truncated weight data")
    w = np.frombuffer(buf, dtype=">f4").reshape(r1 - r0, rec.nx)
    w.byteswap(inplace=True)
    return w.view(w.dtype.newbyteorder())


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
    paths_out: dict[str, str],
    mjd_ref: float,
    swarp_preset: dict,
    software: tuple[str, str] = ("vircampype", ""),
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
    paths_out : dict[str, str]
        Output path for each mode, each ``"<mode>.weight"`` and ``"mjd.eff"``
        (see ``statistics_output_paths``).
    mjd_ref : float
        Reference MJD added back to the MJD planes for ``mjd.eff``.
    swarp_preset : dict
        The SWarp coadd preset (coadd.yml) whose behaviour is reproduced.
    software : tuple[str, str], optional
        SOFTNAME and SOFTVERS for the output headers.
    mem_max : int, optional
        RAM budget (MB) for one output strip.
    n_threads : int, optional
        Threads for reading headers and weights.

    Raises
    ------
    StatisticsCoaddUnsupported
        If the inputs are not what this coadd reproduces (caller falls back to SWarp).
    """
    for key, val in _PRESET_REQUIRED.items():
        if swarp_preset.get(key) != val:
            raise StatisticsCoaddUnsupported(f"SWarp preset {key} != {val}")
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
            ),
            range(len(paths_weight)),
        )
        inputs = [rec for recs in per_file for rec in recs]
    if not inputs:
        raise ValueError("No statistics images")
    crpix, naxis = _place(inputs)

    # Output files in their final types (as convert_bitpix_image/combine_mjd_images)
    ny, nx = naxis[2], naxis[1]
    saturate = swarp_preset["SATLEV_DEFAULT"]
    combinet = {
        "nimg": "SUM",
        "exptime": "SUM",
        "mjd.int": "WEIGHTED",
        "mjd.frac": "WEIGHTED",
    }
    writers = {}
    for mode in STATISTICS_MODES:
        args = (inputs, crpix, naxis, swarp_preset, combinet[mode])
        hdr = _output_header(*args, saturate, software)
        bzero = 32768 if mode == "nimg" else None
        writers[mode] = _FitsWriter(
            paths_out[mode], hdr, "i2" if bzero else "f4", bzero
        )
        hdr_w = _output_header(*args, 0.0, software)  # weights: no saturation level
        writers[f"{mode}.weight"] = _FitsWriter(
            paths_out[f"{mode}.weight"], hdr_w, "f4"
        )
        if mode == "mjd.int":
            # combine_mjd_images saves via astropy: standard first cards, no EXTEND
            hdr_eff = hdr.copy()
            hdr_eff.remove("EXTEND", ignore_missing=True)
            for key, comment in (
                ("SIMPLE", "conforms to FITS standard"),
                ("BITPIX", "array data type"),
                ("NAXIS", "number of array dimensions"),
                ("NAXIS1", ""),
                ("NAXIS2", ""),
            ):
                hdr_eff.comments[key] = comment
            hdr_eff["MJDREF"] = (mjd_ref, "Reference MJD added to the planes")
            writers["mjd.eff"] = _FitsWriter(paths_out["mjd.eff"], hdr_eff, "f8")

    # Row strips sized from the RAM budget; read-ahead holds at most 4 * n_threads
    # bytes per output pixel, plus the block in use (<= 4: detectors don't overlap)
    bytes_px = _BYTES_PER_PIXEL + 4 * (n_threads + 1)
    h_strip = max(1, min(ny, int(mem_max * 2**20) // (nx * bytes_px)))
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

    def nbytes(recs):
        return sum((min(s1, r.y0 + r.ny) - max(s0, r.y0)) * r.nx * 4 for r in recs)

    def blocks_in_order():  # in input order, read-ahead bounded in bytes
        limit, total, pending = 4 * n_threads * shape[0] * nx, 0, deque()
        for g in groups:
            pending.append((pool.submit(read_group, g), nbytes(g)))
            total += pending[-1][1]
            while total > limit:
                future, nb = pending.popleft()
                total -= nb
                yield future.result()
        while pending:
            yield pending.popleft()[0].result()

    for blocks in blocks_in_order():
        for rec, w in blocks:
            ya, xa = max(s0, rec.y0), max(0, rec.x0)
            yb, xb = ya + w.shape[0], min(nx, rec.x0 + rec.nx)
            w = w[:, xa - rec.x0 : xb - rec.x0]
            sl = (slice(ya - s0, yb - s0), slice(xa, xb))
            with np.errstate(divide="ignore", over="ignore"):
                var = np.where(w > 0, np.float32(1) / w, _BIG)
            valid = var < _BIG
            inv = np.where(valid, 1.0 / np.where(valid, var, 1).astype(np.float64), 0.0)
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


def coadd_psf_sigma_map(
    paths_weight: list[str],
    paths_psfvar: list[str],
    path_out: str,
    mem_max: float = 2048,
    n_threads: int = 8,
) -> fits.Header:
    """Coadd per-pawprint stellar window variances into a local window-sigma map.

    Each per-pawprint ``psfvar`` image holds one constant per detector: the squared
    stellar window sigma (arcsec^2), or 0 where it could not be measured. The map is
    the inverse-variance WEIGHTED mean of the valid (> 0) variances on the statistics
    grid (same placement as ``coadd_statistics_maps``), square-rooted: the local
    stellar window sigma of the coadd in arcsec. Pixels without a valid input are 0.

    Returns
    -------
    fits.Header
        Header of the written map (statistics-grid WCS).
    """
    if len(paths_psfvar) != len(paths_weight):
        raise ValueError("psfvar: number of images differs from the weights")
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        per_file = pool.map(
            lambda i: _scan_pawprint(paths_weight[i], {"psfvar": paths_psfvar[i]}),
            range(len(paths_weight)),
        )
        inputs = [rec for recs in per_file for rec in recs]
    if not inputs:
        raise ValueError("No statistics images")
    crpix, naxis = _place(inputs)
    ny, nx = naxis[2], naxis[1]
    hdr = _output_header(
        inputs,
        crpix,
        naxis,
        {
            "RESAMPLING_TYPE": "NEAREST",
            "CENTER_TYPE": "ALL",
            "PIXELSCALE_TYPE": "MEDIAN",
        },
        "WEIGHTED",
        0.0,
        ("vircampype", ""),
    )
    hdr["BUNIT"] = ("arcsec", "Local stellar window sigma (WEIGHTED, sqrt)")
    writer = _FitsWriter(path_out, hdr, "f4")

    bytes_px = 8 * 3 + 4 * (n_threads + 1)
    h_strip = max(1, min(ny, int(mem_max * 2**20) // (nx * bytes_px)))
    for s0 in range(0, ny, h_strip):
        s1 = min(ny, s0 + h_strip)
        shape = (s1 - s0, nx)
        s_inv, s_val = np.zeros(shape), np.zeros(shape)
        for rec in inputs:
            if (
                rec.y0 >= s1
                or rec.y0 + rec.ny <= s0
                or rec.x0 >= nx
                or rec.x0 + rec.nx <= 0
            ):
                continue
            val = np.float64(rec.values["psfvar"])
            if not (val > 0 and np.isfinite(val)):
                continue  # no stellar size for this detector: does not vote
            with open(rec.path_weight, "rb") as fh:
                w = _read_rows(
                    rec, max(s0, rec.y0) - rec.y0, min(s1, rec.y0 + rec.ny) - rec.y0, fh
                )
            ya, xa = max(s0, rec.y0), max(0, rec.x0)
            yb, xb = ya + w.shape[0], min(nx, rec.x0 + rec.nx)
            w = w[:, xa - rec.x0 : xb - rec.x0]
            sl = (slice(ya - s0, yb - s0), slice(xa, xb))
            with np.errstate(divide="ignore", over="ignore"):
                var = np.where(w > 0, np.float32(1) / w, _BIG)
            valid = var < _BIG
            inv = np.where(valid, 1.0 / np.where(valid, var, 1).astype(np.float64), 0.0)
            s_inv[sl] += inv
            s_val[sl] += inv * val
        out = np.zeros(shape, np.float32)
        cov = s_inv > 0
        out[cov] = np.sqrt(s_val[cov] / s_inv[cov])
        writer.write_rows(s0, out)
    return hdr


def statistics_output_paths(path_coadd: str) -> dict[str, str]:
    """Paths of the tile statistics maps next to the coadd."""
    paths = {}
    for mode in STATISTICS_MODES:
        paths[mode] = path_coadd.replace(".fits", f".{mode}.fits")
        paths[f"{mode}.weight"] = path_coadd.replace(".fits", f".{mode}.weight.fits")
    paths["mjd.eff"] = path_coadd.replace(".fits", ".mjd.eff.fits")
    return paths
