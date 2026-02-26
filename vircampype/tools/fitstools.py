import os.path
import re
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Table
from astropy.time import Time

from vircampype.pipeline.log import PipelineLog
from vircampype.pipeline.misc import *
from vircampype.tools.miscellaneous import *
from vircampype.tools.systemtools import (
    make_path_system_tempfile,
    run_commands_shell_parallel,
    which,
)
from vircampype.tools.wcstools import header_reset_wcs

__all__ = [
    "check_card_value",
    "make_card",
    "make_cards",
    "copy_keywords",
    "add_key_primary_hdu",
    "make_mef_image",
    "merge_headers",
    "add_float_to_header",
    "add_str_to_header",
    "convert_bitpix_image",
    "fix_vircam_headers",
    "delete_keyword_from_header",
    "add_int_to_header",
    "replace_data",
    "mjd2dateobs",
    "compress_images",
    "make_gaia_refcat",
    "combine_mjd_images",
    "fits2ldac",
    "read_fits_headers",
]


def read_fits_headers(path: str) -> list[fits.Header]:
    """Read and clean all HDU headers from a single FITS file."""
    with fits.open(path) as hdulist:
        fileheaders = []
        for hdu in hdulist:
            hdr = hdu.header
            try:
                hdr.remove("HIERARCH ESO DET CHIP PXSPACE")
            except KeyError:
                pass
            fileheaders.append(hdr)
    return fileheaders


def check_card_value(value: object) -> str | int | float:
    """
    Validate and normalise a FITS header card value.

    Callable values are converted to their string representation via
    ``func2string``. All other non-numeric, non-string values are cast
    to ``str``.

    Parameters
    ----------
    value : object
        Raw value to validate.

    Returns
    -------
    str or int or float
        Value safe to write into a FITS header card.
    """
    val = func2string(value) if hasattr(value, "__call__") else value

    if not (
        isinstance(val, str)
        or isinstance(val, (np.floating, float))
        or isinstance(val, (np.integer, int))
    ):
        val = str(val)

    return val


def make_card(
    keyword: str,
    value: object,
    comment: str | None = None,
    upper: bool = True,
) -> fits.Card:
    """
    Create a FITS header card from a keyword, value, and optional comment.

    Parameters
    ----------
    keyword : str
        Header keyword.
    value : object
        Value for the card; passed through ``check_card_value`` before use.
    comment : str, optional
        Comment string for the card.
    upper : bool, optional
        Convert *keyword* to upper case. Default is ``True``.

    Returns
    -------
    fits.Card
        Assembled FITS header card.

    Raises
    ------
    ValueError
        If the total card length exceeds 80 characters.
    """
    kw = keyword.upper() if upper else keyword
    kw = re.sub(" +", " ", kw)
    val = check_card_value(value=value)

    lcom = len(comment) if comment is not None else 0
    ltot = len(kw) + len(str(val)) + lcom
    if ltot > 80:
        raise ValueError(f"Card too long ({ltot})")

    return fits.Card(keyword=kw, value=val, comment=comment)


def make_cards(
    keywords: list[str],
    values: list,
    comments: list[str] | None = None,
) -> list[fits.Card]:
    """
    Create a list of FITS header cards from keywords, values, and comments.

    Parameters
    ----------
    keywords : list[str]
        Header keywords.
    values : list
        Values corresponding to each keyword.
    comments : list[str], optional
        Comments corresponding to each keyword. If omitted, no comments
        are written.

    Returns
    -------
    list[fits.Card]
        List of assembled FITS header cards.

    Raises
    ------
    TypeError
        If *keywords* or *values* are not lists.
    ValueError
        If the lengths of *keywords*, *values*, or *comments* do not match.
    """
    if not isinstance(keywords, list) or not isinstance(values, list):
        raise TypeError("keywords and values must be lists")

    if len(keywords) != len(values):
        raise ValueError("Keywords and Values don't match")

    if comments is not None:
        if len(comments) != len(keywords):
            raise ValueError("Comments don't match input")
    else:
        comments = [None for _ in range(len(keywords))]

    return [
        make_card(keyword=kw, value=val, comment=cm)
        for kw, val, cm in zip(keywords, values, comments)
    ]


def copy_keywords(
    path_1: str,
    path_2: str,
    keywords: list[str],
    hdu_1: int = 0,
    hdu_2: int = 0,
) -> None:
    """
    Copy specific header keywords from one FITS file to another.

    Parameters
    ----------
    path_1 : str
        Path to the file that receives the keywords.
    path_2 : str
        Path to the file that the keywords are taken from.
    keywords : list[str]
        Keywords to copy.
    hdu_1 : int, optional
        Extension index to write to in *path_1*. Default is 0 (primary).
    hdu_2 : int, optional
        Extension index to read from in *path_2*. Default is 0 (primary).
    """
    with (
        fits.open(path_1, mode="update") as hdulist_1,
        fits.open(path_2, mode="readonly") as hdulist_2,
    ):
        for k in keywords:
            hdulist_1[hdu_1].header[k] = hdulist_2[hdu_2].header[k]
            hdulist_1.flush()


def add_key_primary_hdu(
    path: str,
    key: str,
    value: str | int | float,
    comment: str | None = None,
) -> None:
    """
    Add or update a keyword in the primary HDU of a FITS file.

    Parameters
    ----------
    path : str
        Path to the FITS file.
    key : str
        Header keyword to add or update.
    value : str or int or float
        Value for the keyword.
    comment : str, optional
        Comment string for the card.
    """
    with fits.open(path, "update") as file:
        if comment is not None:
            file[0].header[key] = (value, comment)
        else:
            file[0].header[key] = value


def make_mef_image(
    paths_input: list[str],
    path_output: str,
    primeheader: fits.Header | None = None,
    add_constant: int | float | str | list | None = None,
    write_extname: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Combine multiple single-extension FITS images into one MEF file.

    Parameters
    ----------
    paths_input : list[str]
        Paths to the input FITS images.
    path_output : str
        Path for the output MEF file.
    primeheader : fits.Header, optional
        Header to use for the primary HDU. An empty header is used if
        not provided.
    add_constant : int or float or str or list, optional
        Constant added to each image's pixel data. If a ``str``, the
        constant is read from that header keyword of each input file.
        If a ``list``, one value per input file. Default is no offset.
    write_extname : bool, optional
        Write a standard ``EXTNAME`` keyword to each extension.
        Default is ``True``.
    overwrite : bool, optional
        Overwrite an existing output file. Default is ``False``.

    Raises
    ------
    ValueError
        If *paths_input* is empty.
    """
    if len(paths_input) == 0:
        raise ValueError("No images to combine")

    if not hasattr(add_constant, "__len__"):
        add_constant = [add_constant] * len(paths_input)

    hdulist = fits.HDUList()

    if primeheader is None:
        primeheader = fits.Header()
    hdulist.append(fits.PrimaryHDU(header=primeheader))

    for pidx, ac in enumerate(add_constant):
        with fits.open(paths_input[pidx]) as file:
            if isinstance(ac, (int, float)):
                const = ac
            elif isinstance(ac, str):
                const = file[0].header[ac]
            else:
                const = 0

            hdr = file[0].header.copy()
            if write_extname:
                hdr.set("EXTNAME", value=f"HDU{pidx + 1:>02d}", after="BITPIX")
            hdulist.append(fits.ImageHDU(data=file[0].data + const, header=hdr))

    hdulist.writeto(path_output, overwrite=overwrite)


def merge_headers(path_1: str, path_2: str, primary_only: bool = False) -> None:
    """
    Merge header keywords from one FITS file into another.

    Every keyword present in *path_2* but absent in *path_1* is copied
    to *path_1*. The file is flushed to disk afterwards.

    Parameters
    ----------
    path_1 : str
        Path to the file that receives new keywords.
    path_2 : str
        Path to the file that keywords are taken from.
    primary_only : bool, optional
        Only merge the primary HDU headers. Default is ``False``.
    """
    skip_list = ["SIMPLE", "NAXIS", "NAXIS1", "NAXIS2"]

    with (
        fits.open(path_1, mode="update") as hdulist_1,
        fits.open(path_2, mode="readonly") as hdulist_2,
    ):
        for hdu1, hdu2 in zip(hdulist_1, hdulist_2):
            if primary_only and not isinstance(hdu1, fits.PrimaryHDU):
                continue

            keys1 = list(hdu1.header.keys())
            for key2, val2 in hdu2.header.items():
                if key2 in skip_list:
                    continue
                if key2 not in keys1:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=VerifyWarning)
                        hdu1.header[key2] = val2

        hdulist_1.flush()


def add_float_to_header(
    header: fits.Header,
    key: str,
    value: float,
    decimals: int = 3,
    comment: str | None = None,
    remove_before: bool = True,
) -> None:
    """
    Write a float value to a FITS header card with fixed decimal formatting.

    Parameters
    ----------
    header : fits.Header
        Header to modify.
    key : str
        Header keyword.
    value : float
        Value to write.
    decimals : int, optional
        Number of decimal places (1–6). Default is 3.
    comment : str, optional
        Comment string for the card.
    remove_before : bool, optional
        Remove all existing occurrences of *key* before writing.
        Default is ``True``.

    Raises
    ------
    ValueError
        If *decimals* is outside the supported range 1–6.
    """
    if remove_before:
        try:
            header.remove(key, remove_all=True)
        except KeyError:
            pass

    if decimals == 1:
        c = fits.Card.fromstring(f"{key:8}= {value:0.1f}")
    elif decimals == 2:
        c = fits.Card.fromstring(f"{key:8}= {value:0.2f}")
    elif decimals == 3:
        c = fits.Card.fromstring(f"{key:8}= {value:0.3f}")
    elif decimals == 4:
        c = fits.Card.fromstring(f"{key:8}= {value:0.4f}")
    elif decimals == 5:
        c = fits.Card.fromstring(f"{key:8}= {value:0.5f}")
    elif decimals == 6:
        c = fits.Card.fromstring(f"{key:8}= {value:0.6f}")
    else:
        raise ValueError("Add more options for decimals")
    c.comment = comment
    header.append(c)


def add_int_to_header(
    header: fits.Header,
    key: str,
    value: int,
    comment: str | None = None,
    remove_before: bool = True,
) -> None:
    """
    Write an integer value to a FITS header card with fixed integer formatting.

    Parameters
    ----------
    header : fits.Header
        Header to modify.
    key : str
        Header keyword.
    value : int
        Integer value to write.
    comment : str, optional
        Comment string for the card.
    remove_before : bool, optional
        Remove all existing occurrences of *key* before writing.
        Default is ``True``.
    """
    if remove_before:
        try:
            header.remove(key, remove_all=True)
        except KeyError:
            pass
    c = fits.Card.fromstring(f"{key:8}= {value:0d}")
    c.comment = comment
    header.append(c)


def add_str_to_header(
    header: fits.Header,
    key: str,
    value: str,
    comment: str | None = None,
    remove_before: bool = True,
) -> None:
    """
    Write a string value to a FITS header card.

    Parameters
    ----------
    header : fits.Header
        Header to modify.
    key : str
        Header keyword.
    value : str
        String value to write.
    comment : str, optional
        Comment string for the card.
    remove_before : bool, optional
        Remove all existing occurrences of *key* before writing.
        Default is ``True``.
    """
    if remove_before:
        try:
            header.remove(key, remove_all=True)
        except KeyError:
            pass
    if comment is None:
        comment = ""
    header[key] = (value, comment)


def convert_bitpix_image(
    path: str,
    new_type: type,
    offset: int | float | None = None,
) -> None:
    """
    Convert image pixel data to a new data type across all HDUs.

    Parameters
    ----------
    path : str
        Path to the FITS file (modified in place).
    new_type : type
        Target numpy data type (e.g. ``np.float32``, ``np.int16``).
    offset : int or float, optional
        Offset added to each pixel value before the type conversion.
    """
    with fits.open(path, mode="update") as hdul:
        for hdu in hdul:
            if hasattr(hdu.data, "__len__"):
                if offset is not None:
                    if "int" in str(new_type):
                        hdu.data = np.rint(hdu.data + offset).astype(new_type)
                    else:
                        hdu.data = (hdu.data + offset).astype(new_type)
                else:
                    if "int" in str(new_type):
                        hdu.data = np.rint(hdu.data).astype(new_type)
                    else:
                        hdu.data = hdu.data.astype(new_type)


def delete_keyword_from_header(header: fits.Header, keyword: str) -> fits.Header:
    """
    Remove a keyword from a FITS header, silently ignoring missing keys.

    Parameters
    ----------
    header : fits.Header
        Header to modify.
    keyword : str
        Keyword to remove.

    Returns
    -------
    fits.Header
        The modified header (same object, modified in place).
    """
    try:
        del header[keyword]
    except KeyError:
        pass
    return header


def replace_data(file_a: str, file_b: str) -> None:
    """
    Copy pixel data from *file_a* into *file_b*, leaving headers untouched.

    Both files must have the same number of HDUs and matching extension
    types.

    Parameters
    ----------
    file_a : str
        Path to the source file (data is read from here).
    file_b : str
        Path to the target file (data is written here, headers kept).

    Raises
    ------
    ValueError
        If the files have different numbers of HDUs or mismatched extension
        types.
    """
    with (
        fits.open(file_a) as hdul_a,
        fits.open(file_b, mode="update") as hdul_b,
    ):
        if len(hdul_a) != len(hdul_b):
            raise ValueError(
                f"Number of HDUs not equal. n(A)={len(hdul_a)}; n(B)={len(hdul_b)}"
            )

        for idx_hdu in range(len(hdul_a)):
            hdu_a, hdu_b = hdul_a[idx_hdu], hdul_b[idx_hdu]
            if hdu_a.__class__ != hdu_b.__class__:
                raise ValueError("Extension types do not match")
            hdu_b.data = hdu_a.data

        hdul_b.flush()


def mjd2dateobs(mjd: float) -> str:
    """
    Convert a Modified Julian Date to a FITS DATE-OBS string.

    Parameters
    ----------
    mjd : float
        Modified Julian Date.

    Returns
    -------
    str
        ISO 8601 date-time string in FITS DATE-OBS format.
    """
    return Time(mjd, format="mjd").fits


def fix_vircam_headers(
    prime_header: fits.Header, data_headers: list[fits.Header]
) -> None:
    """
    Reset WCS information and remove useless keywords from VIRCAM headers.

    Reads the field RA/DEC from the primary header, overwrites ``CRVAL1``/
    ``CRVAL2`` in each extension, fits a simple TAN WCS from the footprint,
    and strips a predefined set of dispensable keywords from both primary
    and extension headers.

    Parameters
    ----------
    prime_header : fits.Header
        Primary FITS header of the VIRCAM raw file.
    data_headers : list[fits.Header]
        Extension headers, one per detector. Modified in place.
    """
    log = PipelineLog()

    try:
        log.info("Attempting to rewrite WCS info in headers")

        tra = str(prime_header["HIERARCH ESO TEL TARG ALPHA"])
        tde = str(prime_header["HIERARCH ESO TEL TARG DELTA"])
        log.info(f"Found RA/DEC in headers: {tra} / {tde}")

        if tde.startswith("-"):
            decsign = -1
            tde = tde[1:]
        else:
            decsign = 1

        # Silly fix for short ALPHA/DELTA strings
        tra = "0" * (6 - len(tra.split(".")[0])) + tra
        tde = "0" * (6 - len(tde.split(".")[0])) + tde

        fra = 15 * (float(tra[:2]) + float(tra[2:4]) / 60 + float(tra[4:]) / 3600)
        fde = decsign * (float(tde[:2]) + float(tde[2:4]) / 60 + float(tde[4:]) / 3600)
        log.info(f"Computed RA/DEC: {fra} / {fde}")

    except KeyError:
        log.warning("Could not find RA/DEC in headers")
        fra, fde = None, None

    for idx_hdr in range(len(data_headers)):
        try:
            crval1 = fra if fra is not None else data_headers[idx_hdr]["CRVAL1"]
            log.info(f"Overwriting CRVAL1 with {crval1} in extension {idx_hdr + 1}")
            data_headers[idx_hdr]["CRVAL1"] = crval1
            crval2 = fde if fde is not None else data_headers[idx_hdr]["CRVAL2"]
            log.info(f"Overwriting CRVAL2 with {crval2} in extension {idx_hdr + 1}")
            data_headers[idx_hdr]["CRVAL2"] = crval2
        except KeyError:
            log.warning("Could not reset CRVAL1/CRVAL2 in headers")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            data_headers[idx_hdr] = header_reset_wcs(data_headers[idx_hdr])

        for kw in useless_extension_keywords:
            data_headers[idx_hdr].remove(kw, ignore_missing=True, remove_all=True)

    for kw in useless_primary_keywords:
        prime_header.remove(kw, ignore_missing=True, remove_all=True)


def compress_images(
    images: list[str],
    q: int | float = 4,
    exe: str = "fpack",
    n_jobs: int = 1,
) -> None:
    """
    Compress FITS images in parallel using ``fpack``.

    Already-compressed files (``*.fits.fz``) are skipped.

    Parameters
    ----------
    images : list[str]
        Paths to the FITS files to compress.
    q : int or float, optional
        ``fpack`` quantisation factor. Default is 4.
    exe : str, optional
        Name or path of the ``fpack`` binary. Default is ``"fpack"``.
    n_jobs : int, optional
        Number of parallel compression jobs. Default is 1.
    """
    fpack = which(exe)
    paths_out = [x.replace(".fits", ".fits.fz") for x in images]
    done = [os.path.isfile(x) for x in paths_out]
    cmds = [f"{fpack} -q {q} {x}" for x in images]
    cmds = [c for c, d in zip(cmds, done) if not d]
    run_commands_shell_parallel(cmds=cmds, n_jobs=n_jobs)


def make_gaia_refcat(
    table_in: Table,
    path_ldac_out: str,
    epoch_in: int | float = 2016.0,
    epoch_out: int | float | None = None,
    key_ra: str = "ra",
    key_ra_error: str = "ra_error",
    key_dec: str = "dec",
    key_dec_error: str = "dec_error",
    key_pmra: str = "pmra",
    key_pmra_error: str = "pmra_error",
    key_pmdec: str = "pmdec",
    key_pmdec_error: str = "pmdec_error",
    key_ruwe: str = "ruwe",
    key_gmag: str = "mag",
    key_gflux: str = "flux",
    key_gflux_error: str = "flux_error",
) -> Table:
    """
    Build an astrometric reference catalogue in LDAC format from Gaia data.

    Sources with non-finite positions, fluxes, or proper motions, or with
    RUWE >= 1.5, are removed. Coordinates are optionally propagated to a
    target epoch via proper motion before writing.

    Parameters
    ----------
    table_in : Table
        Input Gaia catalogue table.
    path_ldac_out : str
        Output path for the LDAC FITS catalogue.
    epoch_in : int or float, optional
        Decimal year epoch of the input catalogue. Default is 2016.0.
    epoch_out : int or float, optional
        Target decimal year epoch. If given, coordinates are propagated
        via proper motion. If ``None``, no propagation is applied.
    key_ra : str, optional
        Column name for right ascension. Default is ``"ra"``.
    key_ra_error : str, optional
        Column name for RA uncertainty. Default is ``"ra_error"``.
    key_dec : str, optional
        Column name for declination. Default is ``"dec"``.
    key_dec_error : str, optional
        Column name for Dec uncertainty. Default is ``"dec_error"``.
    key_pmra : str, optional
        Column name for proper motion in RA * cos(Dec). Default is ``"pmra"``.
    key_pmra_error : str, optional
        Column name for proper motion RA uncertainty. Default is
        ``"pmra_error"``.
    key_pmdec : str, optional
        Column name for proper motion in Dec. Default is ``"pmdec"``.
    key_pmdec_error : str, optional
        Column name for proper motion Dec uncertainty. Default is
        ``"pmdec_error"``.
    key_ruwe : str, optional
        Column name for RUWE. Default is ``"ruwe"``.
    key_gmag : str, optional
        Column name for G-band magnitude. Default is ``"mag"``.
    key_gflux : str, optional
        Column name for G-band flux. Default is ``"flux"``.
    key_gflux_error : str, optional
        Column name for G-band flux uncertainty. Default is
        ``"flux_error"``.

    Returns
    -------
    Table
        Output catalogue table (also written to *path_ldac_out*).
    """
    # Clean data
    keep = (
        np.isfinite(table_in[key_ra])
        & np.isfinite(table_in[key_dec])
        & np.isfinite(table_in[key_gflux])
        & np.isfinite(table_in[key_gflux_error])
        & np.isfinite(table_in[key_pmra])
        & np.isfinite(table_in[key_pmdec])
        & (table_in[key_ruwe] < 1.5)
    )
    table_in = table_in[keep]

    # Build sky coordinates (always needed for output table)
    sc = SkyCoord(
        ra=table_in[key_ra],
        dec=table_in[key_dec],
        pm_ra_cosdec=table_in[key_pmra],
        pm_dec=table_in[key_pmdec],
        obstime=Time(epoch_in, format="decimalyear"),
    )

    # Apply space motion if an output epoch is requested
    if epoch_out is not None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sc = sc.apply_space_motion(
                new_obstime=Time(epoch_out, format="decimalyear")
            )
    else:
        epoch_out = epoch_in

    # Create output table
    table_out = Table()

    table_out["ra"] = sc.ra.degree
    table_out["ra_error"] = table_in[key_ra_error].value / 3_600_000
    table_out["dec"] = sc.dec.degree
    table_out["dec_error"] = table_in[key_dec_error].value / 3_600_000
    table_out["pmra"] = table_in[key_pmra].value
    table_out["pmra_error"] = table_in[key_pmra_error].value
    table_out["pmdec"] = table_in[key_pmdec].value
    table_out["pmdec_error"] = table_in[key_pmdec_error].value
    table_out["mag"] = table_in[key_gmag].value
    table_out["mag_error"] = (
        1.0857 * table_in[key_gflux_error].value / table_in[key_gflux].value
    )
    table_out["obsdate"] = epoch_out

    # Sort by DEC
    sortindex = np.argsort(table_out["dec"])
    for nn in table_out.columns.keys():
        table_out[nn] = table_out[nn][sortindex]

    # Write temporary fits table, convert to LDAC, remove temp file
    path_temp = make_path_system_tempfile(suffix=".cat.fits")
    table_out.write(path_temp, overwrite=True)
    fits2ldac(path_in=path_temp, path_out=path_ldac_out)
    os.remove(path_temp)

    return table_out


def fits2ldac(path_in: str, path_out: str, extension: int = 1) -> None:
    """
    Convert a plain FITS binary table to LDAC format (required by SCAMP).

    The output file contains an ``LDAC_IMHEAD`` extension (storing the
    original header as a character array) and an ``LDAC_OBJECTS`` extension
    (storing the original table data).

    Parameters
    ----------
    path_in : str
        Path to the input FITS file.
    path_out : str
        Path for the output LDAC FITS file.
    extension : int, optional
        HDU index to read from *path_in*. Default is 1.
    """
    data, header = fits.getdata(path_in, extension, header=True)

    ext2_str = header.tostring(endcard=False, padding=False)
    ext2_data = np.array([ext2_str])
    formatstr = str(len(ext2_str)) + "A"
    col1 = fits.Column(name="Field Header Card", array=ext2_data, format=formatstr)
    cols = fits.ColDefs([col1])
    ext2 = fits.BinTableHDU.from_columns(cols)
    ext2.header["EXTNAME"] = "LDAC_IMHEAD"
    ext2.header["TDIM1"] = f"(80, {len(ext2_str) / 80})"
    ext3 = fits.BinTableHDU(data)
    ext3.header["EXTNAME"] = "LDAC_OBJECTS"
    prihdu = fits.PrimaryHDU(header=fits.Header())

    hdulist = fits.HDUList([prihdu, ext2, ext3])
    hdulist.writeto(path_out, overwrite=True)
    hdulist.close()


def combine_mjd_images(
    path_file_a: str, path_file_b: str, path_file_out: str, overwrite: bool = False
) -> None:
    """
    Sum the pixel data of two MJD FITS images, preserving headers from file A.

    Both files must have identical structure (same number of HDUs and
    matching extension types). Pixel arrays are cast to ``float64`` before
    summation.

    Parameters
    ----------
    path_file_a : str
        Path to the first FITS file; its headers are kept in the output.
    path_file_b : str
        Path to the second FITS file; only its pixel data is used.
    path_file_out : str
        Output path for the combined FITS file.
    overwrite : bool, optional
        Overwrite an existing output file. Default is ``False``.

    Raises
    ------
    ValueError
        If the two files have different numbers of HDUs.
    """
    with fits.open(path_file_a) as hdul_a, fits.open(path_file_b) as hdul_b:
        if len(hdul_a) != len(hdul_b):
            raise ValueError("Files incompatible")

        hdul_o = fits.HDUList([])

        for hdu_a, hdu_b in zip(hdul_a, hdul_b):
            da, db = hdu_a.data, hdu_b.data
            if (da is None) and (db is None):
                hdul_o.append(hdu_a)
            elif da.ndim == 2:
                hdul_o.append(
                    fits.ImageHDU(
                        data=da.astype(np.float64) + db.astype(np.float64),
                        header=hdu_a.header,
                    )
                )

        hdul_o.writeto(path_file_out, overwrite=overwrite)
