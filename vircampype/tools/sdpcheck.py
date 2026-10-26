"""ESO Phase 3 (SDP standard v8) checks for the vircampype products.

The only validation authority of v3.0: one explicit check per product shape (T1 MEF
tile, T3 exposure map, T4 source list, S1 stack, S2 weight), plus batch and catalog
family checks. Findings are strings ``"<file>[<hdu>]: <problem>"``; [] means compliant.

    python -m vircampype.tools.sdpcheck [--family] file.fits ...
"""

import argparse
import math
import os
import re

import numpy as np
from astropy.io import fits
from astropy.io.votable.ucd import check_ucd

from vircampype.pipeline.errors import PipelineValueError

__all__ = [
    "RESERVED_COLUMN_NAMES",
    "check_batch",
    "check_family",
    "check_product",
    "main",
    "median_from_counts",
    "scan_expmap",
]

# Sybase IQ reserved words, forbidden as column names (SDP standard App. B, §12.2.3)
RESERVED_COLUMN_NAMES = frozenset(
    """
    active add all algorithm alter and any append as asc auto backup begin between
    bigint binary bit bottom break by calibrate calibration call cancel capability
    cascade case cast certificate char char_convert character check checkpoint checksum
    clientport close columns comment commit committed comparisons computes conflict
    connect constraint contains continue convert create cross cube current
    current_timestamp current_user cursor date dbspace dbspacename deallocate debug dec
    decimal declare decoupled decrypted default delay delete deleting density desc
    deterministic disable distinct do double drop dynamic elements else elseif enable
    encapsulated encrypted end endif escape except exception exclude exec execute
    existing exists explicit express externlogin fastfirstrow fetch first float
    following for force foreign forward from full gb goto grant group grouping having
    hidden history holdlock identified if in inactive index index_lparen inner inout
    input insensitive insert inserting install instead int integer integrated intersect
    into iq is isolation jdk join kb key lateral left like lock logging login long mb
    match membership message mode modify namespace natural new no noholdlock nolock not
    notify null numeric of off on open optimization option options or order others out
    outer over pages paglock partial partition passthrough password plan preceding
    precision prepare primary print privileges proc procedure proxy publication
    raiserror range raw readcommitted readonly readpast readtext readuncommitted
    readwrite real recursive reference references release relocate remote remove rename
    reorganize repeatable repeatableread reserve resizing resource restore restrict
    return revoke right rollback rollup root row rowlock rows save savepoint schedule
    scroll secure select sensitive serializable service session set setuser share
    smallint soapaction some space sqlcode sqlstate start stop subtrans subtransaction
    synchronize syntax_error table tablock tablockx tb temporary then ties time
    timestamp tinyint to top tran transaction transactional transfer tries trigger
    truncate tsequal unbounded uncommitted union unique uniqueidentifier unknown
    unsigned update updating updlock url user utc using validate values varbinary
    varchar variable varying virtual view wait waitfor web when where while window with
    withauto with_cube with_lparen with_rollup within word work writeserver writetext
    xlock xml
    """.split()  # noqa: SIM905
)

# Primary observation block (Table 8); RADESYS and FLUXCAL are checked by value
_OBS_STRINGS = ("ORIGIN", "TELESCOP", "INSTRUME", "FILTER", "OBJECT", "PROG_ID")
_OBS_STRINGS += ("OBSTECH", "PROCSOFT", "REFERENC")
_OBS_FLOATS = ("RA", "DEC", "EXPTIME", "TEXPTIME", "MJD-OBS", "MJD-END")
# Indexed lists (TELESC1.., ...) exist iff the parent value is 'MULTI' (§5.1)
_MULTI = {"TELESCOP": "TELESC", "INSTRUME": "INSTR", "FILTER": "FILTER"}
_MULTI |= {"PROG_ID": "PROGID"}
_PHOT_NUMBERS = ("PHOTZP", "PHOTZPER", "ABMAGLIM", "ABMAGSAT", "PSF_FWHM", "ELLIPTIC")
# Photometric cards a source list shares with its image (§9.2.1)
_PHOT_KEYS = ("PHOTSYS",) + _PHOT_NUMBERS
_WCS_STRINGS = ("CTYPE1", "CTYPE2")
_WCS_NUMBERS = ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2")
_WCS_NUMBERS += ("CD1_1", "CD1_2", "CD2_1", "CD2_2")
# Pixel grid: CONF and EXPMAP must equal SCI here
_GRID_KEYS = ("NAXIS1", "NAXIS2", "CUNIT1", "CUNIT2") + _WCS_STRINGS + _WCS_NUMBERS
# Other WCS conventions and alternate WCS sets (suffix A-Z) are not allowed (§5.4)
_WCS_FORBIDDEN = r"(PC\d+_\d+|PC\d{6}|CDELT\d+|CROTA\d+)[A-Z]?"
_WCS_FORBIDDEN += r"|(CD\d+_\d+|CRVAL\d+|CRPIX\d+|CTYPE\d+|CUNIT\d+)[A-Z]"
_HDUCLASS = {"HDUCLASS": "ESO", "HDUDOC": "SDP"}
_HDUCLASS |= {"HDUVERS": "SDP version 8", "HDUCLAS1": "IMAGE"}
_MAX_STRING = 68
# Tolerance for TEXPTIME <= (MJD-END - MJD-OBS) * 86400: MJD values written with
# 8 decimals are rounded to 0.43 ms each
_SPAN_TOLERANCE = 1e-3


# Public API
def check_product(
    path: str, prodcatg: str | None = None, final: bool = True
) -> list[str]:
    """
    Check one product file; returns the findings ([] = compliant).

    ``prodcatg`` defaults to the primary PRODCATG. ``final=False`` (requires
    ``prodcatg``, else PipelineValueError) skips what the finalizer writes: PRODCATG,
    DATE, ASSONi with its cross-file checks, and CHECKSUM/DATASUM.
    """
    if not final and prodcatg is None:
        raise PipelineValueError("check_product: prodcatg is required with final=False")
    name = os.path.basename(path)
    problems = _check_name(name)
    try:
        hdul = fits.open(path, memmap=True)
    except OSError as error:
        return problems + [f"{name}: cannot be read as FITS ({error})"]

    with hdul:
        primary = hdul[0].header
        category = prodcatg if prodcatg is not None else primary.get("PRODCATG")
        if final:
            if "PRODCATG" not in primary:
                problems.append(f"{name}[0]: missing PRODCATG")
            elif primary["PRODCATG"] != category:
                problems.append(
                    f"{name}[0]: PRODCATG = {primary['PRODCATG']!r}, "
                    f"expected {category!r}"
                )
            # Not in Table 8, but ESO's validator requires it (batch 23808)
            problems += _require(primary, ("DATE",), f"{name}[0]")
        if category is None:
            return problems

        if category == "SCIENCE.IMAGE":
            problems += _check_image_mef(path, hdul, final)
        elif category == "ANCILLARY.EXPMAP":
            problems += _check_expmap(path, hdul)
        elif category == "SCIENCE.SRCTBL":
            problems += _check_srctbl(path, hdul, final)
        elif category == "SCIENCE.MEFIMAGE":
            problems += _check_mefimage(path, hdul, final)
        elif category == "ANCILLARY.WEIGHTMAP":
            problems += _check_weightmap(path, hdul)
        else:
            problems.append(f"{name}[0]: unknown product category {category!r}")
            return problems

        for index, hdu in enumerate(hdul):
            where = f"{name}[{index}]"
            problems += _check_values(hdu.header, where)
            if final:
                problems += _require(hdu.header, ("CHECKSUM", "DATASUM"), where)
    return problems


def check_batch(paths: list[str]) -> list[str]:
    """
    Check one batch in submission order; returns the findings ([] = consistent).

    Unique file names of at most 68 characters; every ASSONi in the batch; every
    ``.fits`` provenance entry (ORIGFILE) listed earlier. Raw ``VCAM.`` ARCFILE
    references are not resolved.
    """
    names = [os.path.basename(path) for path in paths]
    problems = []
    first_path = {}
    for path, name in zip(paths, names):
        problems += _check_name(name)
        if name in first_path:
            problems.append(
                f"{name}: duplicate file name in the batch ({first_path[name]}, {path})"
            )
        else:
            first_path[name] = path

    for position, (path, name) in enumerate(zip(paths, names)):
        try:
            hdul = fits.open(path, memmap=True)
        except OSError as error:
            problems.append(f"{name}: cannot be read as FITS ({error})")
            continue
        with hdul:
            primary = hdul[0].header
            for key in _indexed_keys(primary, "ASSON"):
                if primary[key] not in first_path:
                    problems.append(
                        f"{name}[0]: {key} {primary[key]!r} is not in the batch"
                    )
            for label, value in _prov_entries(hdul):
                if not isinstance(value, str) or not value.endswith(".fits"):
                    continue
                if value not in first_path:
                    problems.append(f"{name}[0]: {label} {value!r} is not in the batch")
                elif value not in names[:position]:
                    problems.append(
                        f"{name}[0]: {label} {value!r} must be listed before this file"
                    )
    return problems


def check_family(paths: list[str]) -> list[str]:
    """
    Check one catalog family (any order); returns the findings ([] = consistent).

    One MCATALOG (empty PHASE3CATALOG, PHASE3FILELIST naming exactly the tiles,
    FPRA1/FPDE1) and >= 2 CATALOGTILEs. The MCATALOG table is the reference: it has
    TTYPE/TFORM/TUNIT/TUCD for every column, and every tile has its FILTER, TFIELDS and
    those column cards, plus TCOMMi. Column-level rules are not repeated here.
    """
    problems = []
    files = {}  # name -> (primary, table header or None, ORIGFILE list or None)
    for path in paths:
        name = os.path.basename(path)
        try:
            hdul = fits.open(path, memmap=True)
        except OSError as error:
            problems.append(f"{name}: cannot be read as FITS ({error})")
            continue
        with hdul:
            primary, table, listed = hdul[0].header.copy(), None, None
            problems += _check_no_data(primary, f"{name}[0]")
            if len(hdul) < 2 or type(hdul[1]) is not fits.BinTableHDU:
                problems.append(f"{name}[1]: expected a BINTABLE extension")
            else:
                table = hdul[1].header.copy()
                if hdul[1].name != "PHASE3CATALOG":
                    extname = hdul[1].name
                    problems.append(
                        f"{name}[1]: EXTNAME {extname!r}, not PHASE3CATALOG"
                    )
                if primary.get("PRODCATG") == "SCIENCE.MCATALOG":
                    found, listed = _check_master(name, hdul)
                    problems += found
            files[name] = (primary, table, listed)

    category = {name: file[0].get("PRODCATG") for name, file in files.items()}
    masters = [n for n, c in category.items() if c == "SCIENCE.MCATALOG"]
    tiles = [n for n, c in category.items() if c == "SCIENCE.CATALOGTILE"]
    for name, prodcatg in category.items():
        if name not in masters + tiles:
            problems.append(f"{name}[0]: PRODCATG = {prodcatg!r} is not in a family")
    if len(tiles) < 2:
        problems.append(
            f"family: {len(tiles)} SCIENCE.CATALOGTILE files, expected >= 2"
        )
    if len(masters) != 1:
        problems.append(f"family: {len(masters)} SCIENCE.MCATALOG files, expected 1")
        return problems
    master = masters[0]
    master_primary, reference, listed = files[master]

    # File list and footprint area (§12.2.2.1, §5.16)
    if listed is not None:
        where = f"{master}[2]"
        for name in sorted({n for n in listed if listed.count(n) > 1}):
            problems.append(f"{where}: ORIGFILE {name!r} is listed twice")
        for name in sorted(set(listed) - set(tiles)):
            problems.append(f"{where}: ORIGFILE {name!r} is not a tile of the family")
        for name in sorted(set(tiles) - set(listed)):
            problems.append(f"{where}: tile {name!r} is missing from ORIGFILE")
    area = master_primary.get("SKYSQDEG")
    tile_areas = [files[name][0].get("SKYSQDEG") for name in tiles]
    numbers = _is_number(area) and tiles and all(_is_number(a) for a in tile_areas)
    if numbers and area > sum(tile_areas) + 1e-9:
        total = sum(tile_areas)
        problems.append(f"{master}[0]: SKYSQDEG = {area} exceeds the tiles' {total}")

    # Every tile against the MCATALOG table
    if reference is None:
        return problems
    tfields = reference.get("TFIELDS")
    if not isinstance(tfields, int):
        return problems + [f"{master}[1]: missing TFIELDS"]
    columns = range(1, tfields + 1)
    keys = tuple(
        f"{k}{i}" for i in columns for k in ("TTYPE", "TFORM", "TUNIT", "TUCD")
    )
    problems += [f"{master}[1]: missing {key}" for key in keys if key not in reference]
    for name in [master] + tiles:
        primary, table, _ = files[name]
        if table is None:
            continue
        where = f"{name}[1]"
        problems += [
            f"{where}: missing TCOMM{i}" for i in columns if f"TCOMM{i}" not in table
        ]
        if name != master:
            problems += _compare(
                primary, master_primary, ("FILTER",), f"{name}[0]", master
            )
            problems += _compare(table, reference, ("TFIELDS",) + keys, where, master)
    return problems


def _check_master(name: str, hdul: fits.HDUList) -> tuple[list[str], list[str] | None]:
    """MCATALOG footprint, empty table and file list; returns findings, ORIGFILEs."""
    problems = _require_numbers(hdul[0].header, ("FPRA1", "FPDE1"), f"{name}[0]")
    if hdul[1].header.get("NAXIS2") != 0:
        problems.append(f"{name}[1]: {hdul[1].header.get('NAXIS2')} rows, expected 0")
    if len(hdul) < 3 or hdul[2].name != "PHASE3FILELIST":
        problems.append(f"{name}[2]: expected a PHASE3FILELIST extension")
        return problems, None
    if type(hdul[2]) is not fits.BinTableHDU or "ORIGFILE" not in hdul[2].columns.names:
        problems.append(f"{name}[2]: PHASE3FILELIST without an ORIGFILE column")
        return problems, None
    form = str(hdul[2].columns["ORIGFILE"].format)
    if not re.fullmatch(r"\d+A", form):
        problems.append(f"{name}[2]: ORIGFILE format {form!r}, expected 'nA'")
    return problems, [str(value).strip() for value in hdul[2].data["ORIGFILE"]]


def main(argv: list[str] | None = None) -> int:
    """Check product files and their batch; print one finding per line."""
    parser = argparse.ArgumentParser(
        prog="sdpcheck",
        description="Check vircampype products against the ESO Phase 3 standard.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="product files in submission order (images before their source lists)",
    )
    parser.add_argument(
        "--family",
        action="store_true",
        help="check the files as one catalog family (MCATALOG + CATALOGTILEs)",
    )
    args = parser.parse_args(argv)

    if args.family:
        problems = check_family(args.files)
    else:
        problems = []
        for path in args.files:
            problems += check_product(path)
        problems += check_batch(args.files)
    # check_product and check_batch both report over-long file names
    problems = list(dict.fromkeys(problems))
    for problem in problems:
        print(problem)
    return 1 if problems else 0


# Product shapes
def _check_image_mef(path: str, hdul: fits.HDUList, final: bool) -> list[str]:
    """T1 SCIENCE.IMAGE in MEF format: empty primary, SCI, CONF (§6.2.1.2, §2.1)."""
    name = os.path.basename(path)
    primary = hdul[0].header
    where = f"{name}[0]"
    problems = _check_no_data(primary, where)
    problems += _check_observation(primary, where)
    problems += _expect(primary, "FLUXCAL", ("ABSOLUTE",), where)
    problems += _check_prov(hdul, where)

    extnames = [hdu.name for hdu in hdul[1:]]
    layout = extnames[:2] == ["SCI", "CONF"] and extnames[2:] in (
        [],
        ["PHASE3PROVENANCE"],
    )
    if not layout or any(type(hdu) is not fits.ImageHDU for hdu in hdul[1:3]):
        problems.append(
            f"{name}: extensions {extnames}, expected image extensions SCI and CONF "
            "(and optionally PHASE3PROVENANCE)"
        )
        return problems

    sci, conf = hdul[1].header, hdul[2].header
    where_sci, where_conf = f"{name}[1]", f"{name}[2]"
    for key, value in _HDUCLASS.items():
        problems += _expect(sci, key, (value,), where_sci)
        problems += _expect(conf, key, (value,), where_conf)

    problems += _expect(sci, "HDUCLAS2", ("DATA",), where_sci)
    problems += _expect(sci, "CONFDATA", ("CONF",), where_sci)
    problems += _forbid(sci, ("SCIDATA",), where_sci)
    problems += _require(sci, ("BUNIT",), where_sci)
    problems += _require_numbers(sci, ("MJD-OBS",), where_sci)
    problems += _check_wcs(sci, where_sci)
    problems += _check_photometry(sci, where_sci)

    # ESO's validator requires ELLIPTIC in the primary of the tile (batch 23808)
    problems += _require_numbers(primary, ("ELLIPTIC",), where)
    problems += _compare(primary, sci, ("ELLIPTIC",), where, "SCI")

    problems += _expect(conf, "HDUCLAS2", ("CONF",), where_conf)
    problems += _expect(conf, "SCIDATA", ("SCI",), where_conf)
    problems += _forbid(conf, ("CONFDATA",), where_conf)
    problems += _compare(conf, sci, _GRID_KEYS, where_conf, "SCI")

    if final:
        found, headers = _associated(path, primary, "ANCILLARY.EXPMAP", where)
        problems += found
        if headers is not None:
            expmaps = [
                header for header in headers[1:] if header.get("EXTNAME") == "EXPMAP"
            ]
            if not expmaps:
                problems.append(f"{where}: ASSON1 {primary['ASSON1']!r} has no EXPMAP")
            else:
                label = f"{where}: ASSON1 {primary['ASSON1']} EXPMAP"
                problems += _compare(expmaps[0], sci, _GRID_KEYS, label, "SCI")
                expmap = os.path.join(os.path.dirname(path), primary["ASSON1"])
                problems += _check_exptime(primary, expmap, where)
    return problems


def _check_expmap(path: str, hdul: fits.HDUList) -> list[str]:
    """T3 ANCILLARY.EXPMAP: empty primary and one EXPMAP extension (§4.3.1, §2.3)."""
    name = os.path.basename(path)
    primary = hdul[0].header
    where = f"{name}[0]"
    problems = _check_no_data(primary, where)
    problems += _require(primary, ("OBJECT", "FILTER"), where)
    problems += _forbid(primary, _indexed_keys(primary, "ASSON"), where)

    extnames = [hdu.name for hdu in hdul[1:]]
    if extnames != ["EXPMAP"] or type(hdul[1]) is not fits.ImageHDU:
        problems.append(f"{name}: extensions {extnames}, expected one image EXPMAP")
        return problems

    header = hdul[1].header
    where = f"{name}[1]"
    problems += _expect(header, "BUNIT", ("s",), where)
    problems += _expect(header, "BITPIX", (16, -32), where)
    problems += _forbid(header, ("BLANK",), where)
    problems += _check_wcs(header, where)
    data = hdul[1].data
    if data is not None and data.ndim == 2:
        not_finite, negative, _ = scan_expmap(data)
        if not_finite:
            problems.append(f"{where}: EXPMAP contains NaN or Inf (void pixels are 0)")
        if negative:
            problems.append(f"{where}: EXPMAP contains negative values")
    return problems


def scan_expmap(data: np.ndarray) -> tuple[bool, bool, dict]:
    """
    One pass over an EXPMAP in 1024-row memmap chunks: (has NaN/Inf, has negative
    values, counts of the positive values). Exposure maps hold few distinct values.
    """
    not_finite = negative = False
    counts = {}
    for start in range(0, data.shape[0], 1024):
        chunk = data[start : start + 1024]
        not_finite = not_finite or not np.isfinite(chunk).all()
        negative = negative or bool((chunk < 0).any())
        values, numbers = np.unique(chunk[chunk > 0], return_counts=True)
        for value, number in zip(values.tolist(), numbers.tolist()):
            counts[value] = counts.get(value, 0) + number
    return not_finite, negative, counts


def median_from_counts(counts: dict) -> float:
    """
    Exact median of the values whose counts are given: mean of the elements at 0-based
    positions (total - 1) // 2 and total // 2.
    """
    total, seen, lower = sum(counts.values()), 0, None
    for value in sorted(counts):
        seen += counts[value]
        if lower is None and seen > (total - 1) // 2:
            lower = value
        if seen > total // 2:
            return (lower + value) / 2


def _check_exptime(primary: fits.Header, expmap: str, where: str) -> list[str]:
    """EXPTIME = exact median of the nonzero EXPMAP pixels, within 1 % (§4.3.1)."""
    with fits.open(expmap, memmap=True) as hdul:
        data = hdul["EXPMAP"].data
        if data is None or data.ndim != 2:
            return []  # reported by the EXPMAP check
        counts = scan_expmap(data)[2]
    if not counts:
        return [f"{where}: the EXPMAP has no covered (nonzero) pixels"]
    median = median_from_counts(counts)
    exptime = primary.get("EXPTIME")
    if _is_number(exptime) and abs(exptime - median) > 0.01 * median:
        return [f"{where}: EXPTIME = {exptime}, EXPMAP median {median:g} (> 1 %)"]
    return []


def _check_srctbl(path: str, hdul: fits.HDUList, final: bool) -> list[str]:
    """T4 SCIENCE.SRCTBL: empty primary and one binary table (§9.2, §2.4, §3.3)."""
    name = os.path.basename(path)
    primary = hdul[0].header
    where = f"{name}[0]"
    problems = _check_no_data(primary, where)
    problems += _check_observation(primary, where)
    # ELLIPTIC as often as in the tile, primary and table (ESO validator, batch 23808)
    problems += _require_numbers(primary, ("EQUINOX", "ELLIPTIC"), where)
    problems += _forbid(primary, _indexed_keys(primary, "ASSON"), where)
    problems += _check_prov(hdul, where, origfile=True)
    if _indexed_keys(primary, "PROV") != ["PROV1"]:
        problems.append(
            f"{where}: provenance must be exactly PROV1 (the tile file name)"
        )

    if len(hdul) != 2 or type(hdul[1]) is not fits.BinTableHDU:
        problems.append(f"{name}: expected exactly one binary table extension")
        return problems
    table = hdul[1]
    where_table = f"{name}[1]"
    problems += _require(table.header, ("EXTNAME",), where_table)
    problems += _check_photometry(table.header, where_table)
    problems += _check_table(table, where_table)

    tile = primary.get("PROV1")
    if isinstance(tile, str) and tile.endswith(".fits"):
        headers = _read_headers(os.path.join(os.path.dirname(path), tile))
        if headers is None:
            if final:
                problems.append(
                    f"{where}: PROV1 file {tile!r} not found next to this file"
                )
        else:
            # Primary cards copied from the image (§9.2.1, plan §2.4)
            prefixes = (*_MULTI.values(), "OBID")
            indexed = [k for p in prefixes for k in _indexed_keys(headers[0], p)]
            keys = _OBS_STRINGS + _OBS_FLOATS + ("NCOMBINE", "EQUINOX", "RADESYS")
            keys += ("ELLIPTIC",)
            label = f"{tile} primary"
            problems += _compare(
                primary, headers[0], keys + tuple(indexed), where, label
            )
            scis = [header for header in headers[1:] if header.get("EXTNAME") == "SCI"]
            if not scis:
                problems.append(f"{where}: PROV1 file {tile!r} has no SCI extension")
            else:
                label = f"{tile} SCI"
                problems += _compare(
                    table.header, scis[0], _PHOT_KEYS, where_table, label
                )
    return problems


def _check_mefimage(path: str, hdul: fits.HDUList, final: bool) -> list[str]:
    """S1 SCIENCE.MEFIMAGE: empty primary and 16 detector extensions (§6.2.2)."""
    name = os.path.basename(path)
    primary = hdul[0].header
    where = f"{name}[0]"
    problems = _check_no_data(primary, where)
    problems += _check_observation(primary, where)
    problems += _expect(primary, "FLUXCAL", ("ABSOLUTE",), where)
    problems += _check_prov(hdul, where)
    problems += _forbid(primary, _hduclass_keys(primary), where)

    images = [hdu for hdu in hdul[1:] if hdu.name != "PHASE3PROVENANCE"]
    problems += _check_detectors(name, images)
    for index, hdu in enumerate(images, start=1):
        header = hdu.header
        where_ext = f"{name}[{index}]"
        problems += _forbid(header, _hduclass_keys(header), where_ext)
        problems += _require(header, ("BUNIT",), where_ext)
        problems += _check_photometry(header, where_ext, unmeasured_psf=True)

    if final:
        found, headers = _associated(path, primary, "ANCILLARY.WEIGHTMAP", where)
        problems += found
        if headers is not None:
            weight = primary["ASSON1"]
            if len(headers) - 1 != len(images):
                problems.append(
                    f"{where}: ASSON1 {weight!r} has {len(headers) - 1} extensions, "
                    f"this file {len(images)}"
                )
            else:
                for index, hdu in enumerate(images, start=1):
                    label = f"{where}: ASSON1 {weight}[{index}]"
                    keys = ("NAXIS1", "NAXIS2")
                    problems += _compare(headers[index], hdu.header, keys, label, name)
    return problems


def _check_weightmap(path: str, hdul: fits.HDUList) -> list[str]:
    """S2 ANCILLARY.WEIGHTMAP: empty primary and 16 extensions (§4.3.4, §2.3)."""
    name = os.path.basename(path)
    primary = hdul[0].header
    where = f"{name}[0]"
    problems = _check_no_data(primary, where)
    problems += _require(primary, ("OBJECT", "FILTER"), where)
    problems += _forbid(primary, _indexed_keys(primary, "ASSON"), where)

    problems += _check_detectors(name, hdul[1:])
    return problems


def _check_detectors(name: str, images: list) -> list[str]:
    """16 image extensions with unique EXTNAMEs, 2-D data and a WCS (§6.2.2)."""
    problems = []
    if len(images) != 16 or any(type(hdu) is not fits.ImageHDU for hdu in images):
        problems.append(
            f"{name}: {len(images)} extensions, expected 16 image extensions"
        )
    extnames = []
    for index, hdu in enumerate(images, start=1):
        header = hdu.header
        where = f"{name}[{index}]"
        problems += _require(header, ("EXTNAME",), where)
        if header.get("EXTNAME") in extnames:
            problems.append(f"{where}: EXTNAME {header['EXTNAME']!r} is not unique")
        extnames.append(header.get("EXTNAME"))
        shape = [header.get(key) for key in ("NAXIS1", "NAXIS2")]
        if header.get("NAXIS") != 2 or not all(_is_number(n) and n > 0 for n in shape):
            problems.append(f"{where}: expected 2-D data (NAXIS = 2, NAXIS1/2 > 0)")
        problems += _check_wcs(header, where)
    return problems


# Shared rules
def _check_name(name: str) -> list[str]:
    """File names: at most 68 characters, suffix .fits (§3.4.1)."""
    problems = []
    if len(name) > _MAX_STRING:
        problems.append(f"{name}: file name has {len(name)} characters, more than 68")
    if not name.endswith(".fits"):
        problems.append(f"{name}: file name does not end with .fits")
    return problems


def _check_values(header: fits.Header, where: str) -> list[str]:
    """Upper-case keywords, strings <= 68 characters, finite numbers, no RA/DEC_ERR."""
    problems = []
    for card in header.cards:
        key = card.keyword
        if key in ("", "COMMENT", "HISTORY"):
            continue
        if key != key.upper():
            problems.append(f"{where}: keyword {key!r} is not upper case")
        value = card.value
        if isinstance(value, str) and len(value) > _MAX_STRING:
            problems.append(f"{where}: {key} value has {len(value)} characters, max 68")
        if isinstance(value, float) and not math.isfinite(value):
            problems.append(f"{where}: {key} = {value} is not finite")
        if key in ("RA_ERR", "DEC_ERR"):
            problems.append(f"{where}: {key} is not allowed")
    return problems


def _check_observation(header: fits.Header, where: str) -> list[str]:
    """Primary observation block of images and source lists (§5.1, Table 8)."""
    problems = _require(header, _OBS_STRINGS, where)
    problems += _require_numbers(header, _OBS_FLOATS, where)
    obids = _indexed_keys(header, "OBID") or ["OBID1"]
    problems += _require_numbers(header, ("NCOMBINE", *obids), where, integer=True)
    problems += _expect(header, "RADESYS", ("ICRS", "FK5"), where)
    if "EQUINOX" in header and header["EQUINOX"] != 2000.0:
        problems.append(f"{where}: EQUINOX = {header['EQUINOX']!r}, expected 2000.0")
    elif "EQUINOX" not in header and header.get("RADESYS") == "FK5":
        problems.append(f"{where}: RADESYS = 'FK5' requires EQUINOX = 2000.0")
    if "TIMESYS" in header:
        problems += _expect(header, "TIMESYS", ("UTC", "TAI"), where)
    # ESO's validator compares TELESCOP with the archived raw value (batch 23808)
    for key in ("TELESCOP", *_indexed_keys(header, "TELESC")):
        if header.get(key) == "VISTA":
            problems.append(f"{where}: {key} = 'VISTA', expected 'ESO-VISTA'")

    exptime = header.get("EXPTIME")
    if _is_number(exptime) and exptime <= 0:
        problems.append(f"{where}: EXPTIME = {exptime}, must be > 0")
    texptime = header.get("TEXPTIME")
    start, end = header.get("MJD-OBS"), header.get("MJD-END")
    if all(_is_number(value) for value in (texptime, start, end)):
        span = (end - start) * 86400
        if texptime > span + _SPAN_TOLERANCE:
            problems.append(
                f"{where}: TEXPTIME = {texptime} exceeds "
                f"(MJD-END - MJD-OBS) * 86400 = {span:.3f}"
            )

    for key, prefix in _MULTI.items():
        indexed = _indexed_keys(header, prefix)
        if header.get(key) == "MULTI" and not indexed:
            problems.append(f"{where}: {key} = 'MULTI' without {prefix}1..")
        if header.get(key) != "MULTI" and indexed:
            problems.append(f"{where}: {prefix}i only allowed with {key} = 'MULTI'")
        problems += _require(header, tuple(indexed), where)
        problems += _check_sequence(header, prefix, where)
    problems += _check_sequence(header, "OBID", where, distinct=True)
    return problems


def _check_sequence(
    header: fits.Header, prefix: str, where: str, distinct: bool = False
) -> list[str]:
    """Indexed keywords run PREFIX1..PREFIXn without gaps or leading zeros (§5.1)."""
    keys = _indexed_keys(header, prefix)
    problems = []
    if keys != [f"{prefix}{index}" for index in range(1, len(keys) + 1)]:
        problems.append(
            f"{where}: {prefix}i must run {prefix}1..{prefix}{len(keys)} without gaps, "
            f"found {', '.join(keys)}"
        )
    values = [header[key] for key in keys]
    if distinct and len(set(values)) < len(values):
        problems.append(f"{where}: {prefix}i values are not distinct")
    return problems


def _check_photometry(
    header: fits.Header, where: str, unmeasured_psf: bool = False
) -> list[str]:
    """Photometric and PSF cards; PSF_FWHM = -1 only for a MEFIMAGE chip (§5.11)."""
    problems = _require_numbers(header, _PHOT_NUMBERS, where)
    problems += _expect(header, "PHOTSYS", ("VEGA", "AB"), where)
    psf = header.get("PSF_FWHM")
    if _is_number(psf) and psf <= 0 and not (unmeasured_psf and psf == -1):
        allowed = "> 0 or -1" if unmeasured_psf else "> 0"
        problems.append(f"{where}: PSF_FWHM = {psf}, must be {allowed}")
    return problems


def _check_wcs(header: fits.Header, where: str) -> list[str]:
    """One CD-matrix WCS in degrees; PCi_j, CDELTi, CROTAi, alternate sets forbidden."""
    problems = _require(header, _WCS_STRINGS, where)
    problems += _require_numbers(header, _WCS_NUMBERS, where)
    problems += _expect(header, "CUNIT1", ("deg",), where)
    problems += _expect(header, "CUNIT2", ("deg",), where)
    for key in header:
        if re.fullmatch(_WCS_FORBIDDEN, key):
            problems.append(f"{where}: {key} is not allowed (one CD-matrix WCS only)")
    return problems


def _check_prov(hdul: fits.HDUList, where: str, origfile: bool = False) -> list[str]:
    """
    Provenance (§5.2): PROV1..PROVn in the primary, or PROVXTN = T and a
    PHASE3PROVENANCE table. Entries are unique and carry no ``[n]`` suffix. With
    ``origfile`` (source lists) every entry is a file name ending in ``.fits``;
    otherwise every entry is a raw ARCFILE without ``.fits``.
    """
    primary = hdul[0].header
    keys = _indexed_keys(primary, "PROV")
    tables = [hdu for hdu in hdul[1:] if hdu.name == "PHASE3PROVENANCE"]
    problems = []
    if len(tables) > 1:
        problems.append(f"{where}: {len(tables)} PHASE3PROVENANCE extensions, max one")
    if "PROVXTN" in primary:
        if primary["PROVXTN"] is not True:
            problems.append(f"{where}: PROVXTN = {primary['PROVXTN']!r}, expected T")
        if keys:
            problems.append(f"{where}: {', '.join(keys)} not allowed with PROVXTN")
        if not tables:
            problems.append(f"{where}: PROVXTN without a PHASE3PROVENANCE extension")
            return problems
        header = tables[0].header
        if (
            type(tables[0]) is not fits.BinTableHDU
            or header.get("TTYPE1") != "PROV"
            or header.get("TFORM1") != "68A"
        ):
            problems.append(
                f"{where}: PHASE3PROVENANCE must be a binary table with TTYPE1 'PROV' "
                "and TFORM1 '68A'"
            )
            return problems
    else:
        if tables:
            problems.append(f"{where}: PHASE3PROVENANCE extension without PROVXTN")
        if not keys:
            problems.append(f"{where}: missing provenance (PROV1 or PROVXTN)")
            return problems
        problems += _check_sequence(primary, "PROV", where)

    entries = _prov_entries(hdul)
    values = [value for _, value in entries]
    if len(set(values)) < len(values):
        problems.append(f"{where}: duplicate provenance entries")
    for label, value in entries:
        if not isinstance(value, str) or not value:
            problems.append(f"{where}: {label} = {value!r} is not a file reference")
            continue
        if value.endswith("]"):
            problems.append(f"{where}: {label} = {value!r} has an extension suffix")
        if origfile and not value.endswith(".fits"):
            problems.append(f"{where}: {label} = {value!r} must be a file name (.fits)")
        if not origfile and ".fits" in value:
            problems.append(
                f"{where}: {label} = {value!r}: a raw ARCFILE reference has no .fits"
            )
    return problems


def _check_table(hdu: fits.BinTableHDU, where: str) -> list[str]:
    """Binary table rules for catalogs and source lists (§5.18, §12.2, plan §3.3)."""
    header = hdu.header
    problems = []
    if list(header.keys())[7:8] != ["TFIELDS"]:
        problems.append(f"{where}: TFIELDS must be the 8th keyword")
    for key in header:
        if re.fullmatch(r"TZERO\d+|TSCAL\d+|EXTVER|EXTLEVEL", key):
            problems.append(f"{where}: {key} is not allowed")

    tfields = header.get("TFIELDS")
    if not isinstance(tfields, int):
        problems.append(f"{where}: missing TFIELDS")
        return problems
    data = hdu.data
    nrows = 0 if data is None else len(data)
    names = []
    ucds = {}
    for index in range(1, tfields + 1):
        ttype = header.get(f"TTYPE{index}")
        tform = header.get(f"TFORM{index}")
        tucd = header.get(f"TUCD{index}")
        tnull = header.get(f"TNULL{index}")
        label = f"column {index} ({ttype})"
        problems += _require(header, (f"TTYPE{index}", f"TFORM{index}"), where)
        problems += _require(header, (f"TUCD{index}", f"TCOMM{index}"), where)
        if f"TUNIT{index}" not in header:  # present for every column, may be empty
            problems.append(f"{where}: missing TUNIT{index}")

        if isinstance(ttype, str):
            if not re.fullmatch(r"[A-Za-z0-9_]+", ttype):
                problems.append(
                    f"{where}: TTYPE{index} = {ttype!r} has invalid characters"
                )
            if ttype.lower() in RESERVED_COLUMN_NAMES:
                problems.append(f"{where}: TTYPE{index} = {ttype!r} is a reserved word")
            if ttype.lower() in names:
                problems.append(f"{where}: TTYPE{index} = {ttype!r} is not unique")
            names.append(ttype.lower())
        if isinstance(tucd, str):
            ucds[index] = tucd
            if not check_ucd(tucd, check_controlled_vocabulary=True):
                problems.append(f"{where}: TUCD{index} = {tucd!r} is not a valid UCD")

        match = re.fullmatch(r"(\d*)([LABIJKED])", str(tform))
        if match is None:
            problems.append(f"{where}: TFORM{index} = {tform!r} is not supported")
            continue
        repeat, code = int(match.group(1) or 1), match.group(2)
        if not 1 <= repeat <= 1500:
            problems.append(
                f"{where}: TFORM{index} = {tform!r}, repeat must be 1..1500"
            )
        if tnull is not None and code not in "BIJK":
            problems.append(f"{where}: TNULL{index} only allowed on integer columns")

        # Null values (§12.2.6): NaN for floats, TNULL for integers, empty strings
        if nrows == 0 or code == "L":
            continue
        values = np.asarray(data.field(index - 1))
        if code in "ED":
            if np.isinf(values).any():
                problems.append(f"{where}: {label} contains Inf")
            null = np.isnan(values)
        elif code == "A":  # empty, or ASCII NUL as the first character
            null = np.array([value[:1] in ("", "\x00") for value in values])
        else:
            null = (
                values == tnull if tnull is not None else np.zeros(values.shape, bool)
            )
        if null.all():
            problems.append(f"{where}: {label} has no valid values")
        elif isinstance(tucd, str) and "meta.main" in tucd and null.any():
            problems.append(f"{where}: {label} is meta.main and has null values")

    ids = [index for index, ucd in ucds.items() if ucd == "meta.id;meta.main"]
    if len(ids) != 1:
        problems.append(
            f"{where}: {len(ids)} columns with TUCD 'meta.id;meta.main', expected one"
        )
    for index in ids:
        if f"TNULL{index}" in header:
            problems.append(
                f"{where}: TNULL{index} not allowed on the meta.id;meta.main column"
            )
    for main in ("pos.eq.ra;meta.main", "pos.eq.dec;meta.main"):
        columns = [index for index, ucd in ucds.items() if ucd == main]
        if len(columns) != 1:
            problems.append(f"{where}: {len(columns)} columns with TUCD {main!r}")
        for index in columns:
            if header.get(f"TUNIT{index}") != "deg":
                problems.append(f"{where}: TUNIT{index} of {main!r} must be 'deg'")
    return problems


# Helpers
def _require(header: fits.Header, keys: tuple[str, ...], where: str) -> list[str]:
    """String keywords that must be present and not empty (REFERENC may be empty)."""
    problems = []
    for key in keys:
        if key not in header:
            problems.append(f"{where}: missing {key}")
        elif not isinstance(header[key], str):
            problems.append(f"{where}: {key} = {header[key]!r} is not a string")
        elif header[key] == "" and key != "REFERENC":
            problems.append(f"{where}: {key} is empty")
    return problems


def _require_numbers(
    header: fits.Header, keys: tuple[str, ...], where: str, integer: bool = False
) -> list[str]:
    """Keywords that must be present with a numeric (or integer) value."""
    problems = []
    for key in keys:
        if key not in header:
            problems.append(f"{where}: missing {key}")
        elif integer and (type(header[key]) is not int):
            problems.append(f"{where}: {key} = {header[key]!r} is not an integer")
        elif not _is_number(header[key]):
            problems.append(f"{where}: {key} = {header[key]!r} is not a number")
    return problems


def _expect(header: fits.Header, key: str, allowed: tuple, where: str) -> list[str]:
    """A keyword that must be present with one of the allowed values."""
    if key not in header:
        return [f"{where}: missing {key}"]
    if header[key] not in allowed:
        expected = " or ".join(repr(value) for value in allowed)
        return [f"{where}: {key} = {header[key]!r}, expected {expected}"]
    return []


def _forbid(header: fits.Header, keys, where: str) -> list[str]:
    """Keywords that must not be present."""
    return [f"{where}: {key} is not allowed" for key in keys if key in header]


def _compare(
    header: fits.Header,
    reference: fits.Header,
    keys: tuple[str, ...],
    where: str,
    label: str,
) -> list[str]:
    """Keywords whose value in header differs from the reference header."""
    problems = []
    for key in keys:
        value, expected = header.get(key), reference.get(key)
        if value != expected:
            problems.append(f"{where}: {key} = {value!r}, {label} has {expected!r}")
    return problems


def _check_no_data(header: fits.Header, where: str) -> list[str]:
    """The primary HDU of every product carries no data."""
    if header.get("NAXIS", 0) != 0:
        return [
            f"{where}: primary HDU has data (NAXIS = {header['NAXIS']}), expected 0"
        ]
    return []


def _associated(
    path: str, header: fits.Header, prodcatg: str, where: str
) -> tuple[list[str], list[fits.Header] | None]:
    """
    Check that the primary names exactly one association (ASSON1) and that this file
    exists next to ``path`` with the given PRODCATG. Returns the findings and the
    headers of the associated file (None if it is unusable).
    """
    keys = _indexed_keys(header, "ASSON")
    if keys != ["ASSON1"]:
        found = ", ".join(keys) or "none"
        return [f"{where}: associations {found}, expected exactly ASSON1"], None
    asson = header["ASSON1"]
    headers = _read_headers(os.path.join(os.path.dirname(path), str(asson)))
    if headers is None:
        return [f"{where}: ASSON1 file {asson!r} not found next to this file"], None
    if headers[0].get("PRODCATG") != prodcatg:
        found = headers[0].get("PRODCATG")
        return [f"{where}: ASSON1 {asson!r} is {found!r}, expected {prodcatg!r}"], None
    return [], headers


def _read_headers(path: str) -> list[fits.Header] | None:
    """All headers of a FITS file, or None if it does not exist or cannot be read."""
    if not os.path.isfile(path):
        return None
    try:
        with fits.open(path, memmap=True) as hdul:
            return [hdu.header.copy() for hdu in hdul]
    except OSError:
        return None


def _indexed_keys(header: fits.Header, prefix: str) -> list[str]:
    """Indexed keywords PREFIXi (e.g. PROV1, PROV2, ...) in index order."""
    keys = [key for key in header if re.fullmatch(rf"{prefix}\d+", key)]
    return sorted(keys, key=lambda key: int(key[len(prefix) :]))


def _hduclass_keys(header: fits.Header) -> list[str]:
    """HDUCLASS and HDUCLASn keywords (MEF image mechanism, N/A for MEFIMAGE)."""
    return [key for key in header if key.startswith("HDUCLAS")]


def _prov_entries(hdul: fits.HDUList) -> list[tuple[str, object]]:
    """Provenance as (label, value) pairs: PROVi cards or PHASE3PROVENANCE rows."""
    primary = hdul[0].header
    if "PROVXTN" not in primary:
        return [(key, primary[key]) for key in _indexed_keys(primary, "PROV")]
    for hdu in hdul[1:]:
        if (
            hdu.name == "PHASE3PROVENANCE"
            and type(hdu) is fits.BinTableHDU
            and "PROV" in hdu.columns.names
        ):
            rows = enumerate(hdu.data["PROV"], start=1)
            return [(f"PHASE3PROVENANCE row {row}", str(value)) for row, value in rows]
    return []


def _is_number(value) -> bool:
    """True for int and float header values (not bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    raise SystemExit(main())
