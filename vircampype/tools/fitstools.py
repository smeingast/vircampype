import logging
import os.path
import re
import time
import warnings
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Table
from astropy.time import Time

from vircampype.pipeline.misc import *
from vircampype.tools.messaging import print_header, print_message
from vircampype.tools.miscellaneous import *
from vircampype.tools.systemtools import (
    make_path_system_tempfile,
    move_file,
    remove_file,
    run_commands_shell_parallel,
    which,
)
from vircampype.tools.wcstools import header_reset_wcs

__all__ = [
    "build_qc_summary",
    "build_qc_summary_row",
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
    "tile_fits",
    "write_tile_pair",
    "WEIGHT_BUNIT",
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
    val = func2string(value) if callable(value) else value

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
    cache_dir: str | None = None,
    n_blank: int = 0,
    extension_cards: list[tuple[str, object, str]] = (),
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
    cache_dir : str, optional
        Directory for the temporary intermediate file. If ``None``,
        the system temp directory is used.
    n_blank : int, optional
        Number of blank cards appended to the header of every HDU, the
        reserved header block of Phase 3 products (36), so that later card
        edits and checksums never move the data. Default is 0.
    extension_cards : list[tuple[str, object, str]], optional
        ``(key, value, comment)`` cards set in every extension (e.g. BUNIT).

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
            for key, value, comment in extension_cards:
                hdr.set(key, value, comment)
            hdulist.append(fits.ImageHDU(data=file[0].data + const, header=hdr))

    for hdu in hdulist:
        for _ in range(n_blank):
            hdu.header.append(fits.Card(), end=True)

    # Write via local temp to avoid many small NAS writes
    path_temp = make_path_system_tempfile(suffix=".fits", base_dir=cache_dir)
    try:
        hdulist.writeto(path_temp, overwrite=True)
        move_file(path_temp, path_output)
    finally:
        remove_file(path_temp)


# Unit of SWarp's coadd weight: inverse variance of the image (adu)
WEIGHT_BUNIT = ("adu**(-2)", "Inverse variance of the image")

# SWarp's host and run stamps, its comment section titles, and cards that the
# observation block of the image carries (EXPTIME is the exposure-map median, MJD-OBS
# the first exposure)
_SWARP_DROP = ("ORIGIN", "DATE", "AUTHOR", "COMMENT", "HISTORY", "EXPTIME", "MJD-OBS")
_STRUCTURE_KEYS = ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND")


def write_tile_pair(
    path_image: str,
    path_weight: str,
    path_image_out: str,
    path_weight_out: str,
    primary: fits.Header,
    image_cards: list[tuple[str, object, str]] = (),
    extra_hdus: list = (),
) -> None:
    """
    Write the tile image and its weight from SWarp's coadd image and weight.

    The image goes to the primary HDU of *path_image_out*: the cards of *primary*
    (observation block, provenance) first, then SWarp's WCS and software cards,
    BUNIT and *image_cards*, followed by *extra_hdus* (e.g. a PHASE3PROVENANCE
    table). The weight goes to the primary of *path_weight_out* with SWarp's
    header, OBJECT and FILTER of the image and BUNIT (inverse variance). SWarp's
    ORIGIN, DATE, AUTHOR, COMMENT/HISTORY, EXPTIME and MJD-OBS are dropped (the
    weight also loses GAIN and SATURATE, which SWarp writes as zeros). Every HDU
    gets one reserved block of 36 blank cards so that later card edits never move
    the data. Both inputs stay memory-mapped and unscaled, so the arrays are copied
    bitwise.

    Parameters
    ----------
    path_image : str
        Path to SWarp's coadd image.
    path_weight : str
        Path to SWarp's coadd weight.
    path_image_out : str
        Output path of the image; must not exist.
    path_weight_out : str
        Output path of the weight; must not exist.
    primary : fits.Header
        Observation and provenance cards of the image.
    image_cards : list[tuple[str, object, str]], optional
        Additional ``(key, value, comment)`` cards for the image header.
    extra_hdus : list, optional
        HDUs appended after the image.
    """
    with (
        fits.open(path_image, memmap=True, do_not_scale_image_data=True) as image,
        fits.open(path_weight, memmap=True, do_not_scale_image_data=True) as weight,
    ):
        header = fits.Header()
        for card in primary.cards:
            header.set(card.keyword, card.value, card.comment)
        for card in image[0].header.cards:
            key = card.keyword
            if (
                card.is_blank
                or key in (*_SWARP_DROP, *_STRUCTURE_KEYS)
                or key in header
            ):
                continue
            header.append(card)
        header.set("BUNIT", "adu", "Physical unit of the data")
        for key, value, comment in image_cards:
            header.set(key, value, comment)
        hdul_image = fits.HDUList(
            [fits.PrimaryHDU(data=image[0].data, header=header), *extra_hdus]
        )

        header_weight = weight[0].header.copy()
        for key in (*_SWARP_DROP, "GAIN", "SATURATE"):
            header_weight.remove(key, ignore_missing=True, remove_all=True)
        for key in ("OBJECT", "FILTER"):
            header_weight.set(key, primary[key], primary.comments[key])
        header_weight.set("BUNIT", *WEIGHT_BUNIT)
        hdul_weight = fits.HDUList(
            [fits.PrimaryHDU(data=weight[0].data, header=header_weight)]
        )

        for hdul, path in (
            (hdul_image, path_image_out),
            (hdul_weight, path_weight_out),
        ):
            hdul.update_extend()  # EXTEND first, so that it does not take a blank card
            for hdu in hdul:
                for _ in range(36):
                    hdu.header.append(fits.Card(), end=True)
            hdul.writeto(path)


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


# Warn only once per run about the (benign) RA/DEC->CRVAL fallback; it recurs
# per file for datasets without ESO TEL TARG keys and would otherwise spam.
_crval_fallback_warned = False


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
    log = logging.getLogger(__name__)

    try:
        tra = str(prime_header["HIERARCH ESO TEL TARG ALPHA"])
        tde = str(prime_header["HIERARCH ESO TEL TARG DELTA"])

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

    except KeyError:
        global _crval_fallback_warned
        msg = (
            "Could not find RA/DEC in headers for CRVAL rewrite (using existing CRVAL)"
        )
        if _crval_fallback_warned:
            log.debug(msg)
        else:
            log.warning(f"{msg}; suppressing further occurrences this run")
            _crval_fallback_warned = True
        fra, fde = None, None

    for idx_hdr in range(len(data_headers)):
        try:
            crval1 = fra if fra is not None else data_headers[idx_hdr]["CRVAL1"]
            data_headers[idx_hdr]["CRVAL1"] = crval1
            crval2 = fde if fde is not None else data_headers[idx_hdr]["CRVAL2"]
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
    delete_originals: bool = True,
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
    delete_originals : bool, optional
        If True, pass ``-D -Y`` to ``fpack`` to delete originals after
        compression. Default is False.
    """
    fpack = which(exe)
    paths_out = [x.replace(".fits", ".fits.fz") for x in images]
    done = [os.path.isfile(x) for x in paths_out]
    flags = "-D -Y " if delete_originals else ""
    cmds = [f"{fpack} {flags}-q {q} {x}" for x in images]
    cmds = [c for c, d in zip(cmds, done) if not d]
    run_commands_shell_parallel(cmds=cmds, n_jobs=n_jobs, label="Compressing")


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
    ruwe_max: float = 1.5,
    key_gmag: str = "mag",
    key_gflux: str = "flux",
    key_gflux_error: str = "flux_error",
    cache_dir: str | None = None,
) -> Table:
    """
    Build an astrometric reference catalogue in LDAC format from Gaia data.

    Sources with non-finite positions, fluxes, or proper motions, or with
    RUWE >= ``ruwe_max``, are removed. Coordinates are optionally propagated
    to a target epoch via proper motion before writing.

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
    ruwe_max : float, optional
        Maximum RUWE; sources with ``RUWE >= ruwe_max`` are removed.
        Default is ``1.5``.
    key_gmag : str, optional
        Column name for G-band magnitude. Default is ``"mag"``.
    key_gflux : str, optional
        Column name for G-band flux. Default is ``"flux"``.
    key_gflux_error : str, optional
        Column name for G-band flux uncertainty. Default is
        ``"flux_error"``.
    cache_dir : str, optional
        Directory for the temporary intermediate file. If ``None``,
        the system temp directory is used.

    Returns
    -------
    Table
        Output catalogue table (also written to *path_ldac_out*).
    """
    # Clean data. PM errors must be finite too: they propagate into the warped
    # position errors below (a NaN pm error would turn ra/dec_error into NaN).
    keep = (
        np.isfinite(table_in[key_ra])
        & np.isfinite(table_in[key_dec])
        & np.isfinite(table_in[key_gflux])
        & np.isfinite(table_in[key_gflux_error])
        & np.isfinite(table_in[key_pmra])
        & np.isfinite(table_in[key_pmra_error])
        & np.isfinite(table_in[key_pmdec])
        & np.isfinite(table_in[key_pmdec_error])
        & (table_in[key_ruwe] < ruwe_max)
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

    # Default to the input epoch (no proper-motion warp) when none is requested.
    if epoch_out is None:
        epoch_out = epoch_in

    # Warp baseline in years (Gaia epoch -> output epoch); 0 when no warp.
    dt = epoch_out - epoch_in

    # Apply space motion when warping to a different epoch.
    if dt != 0:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sc = sc.apply_space_motion(
                new_obstime=Time(epoch_out, format="decimalyear")
            )

    # Create output table
    table_out = Table()

    # Propagate the proper-motion uncertainty over the warp baseline into the
    # position errors: warping shifts each position by mu*dt, which carries an
    # uncertainty sigma_mu*dt that adds in quadrature to the catalog-epoch
    # position error. Gaia ra_error and pmra share the alpha* = alpha*cos(dec)
    # basis (and apply_space_motion uses pm_ra_cosdec=pmra), so the per-axis
    # quadrature add is convention-consistent; dt=0 (no warp) leaves the errors
    # unchanged. The same-axis position<->PM covariance (ra_pmra_corr /
    # dec_pmdec_corr) is not in DR3's default Vizier view and is neglected.
    table_out["ra"] = sc.ra.degree
    table_out["ra_error"] = (
        np.hypot(table_in[key_ra_error].value, dt * table_in[key_pmra_error].value)
        / 3_600_000
    )
    table_out["dec"] = sc.dec.degree
    table_out["dec_error"] = (
        np.hypot(table_in[key_dec_error].value, dt * table_in[key_pmdec_error].value)
        / 3_600_000
    )
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
    path_temp = make_path_system_tempfile(suffix=".cat.fits", base_dir=cache_dir)
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
    path_file_a: str,
    path_file_b: str,
    path_file_out: str,
    *,
    offset: float,
    overwrite: bool = False,
) -> None:
    """
    Sum the pixel data of two MJD FITS images plus a constant offset, preserving
    headers from file A.

    Both files must have identical structure (same number of HDUs and
    matching extension types). Pixel arrays are cast to ``float64`` before
    summation. The offset is the reference MJD that was subtracted from the
    integer plane when the statistics images were built (``MJDREF``); it is
    written to the primary header of the output.
    Pixels that are 0 in both planes (no coverage) stay 0.

    Parameters
    ----------
    path_file_a : str
        Path to the first FITS file; its headers are kept in the output.
    path_file_b : str
        Path to the second FITS file; only its pixel data is used.
    path_file_out : str
        Output path for the combined FITS file.
    offset : float
        Constant added to every pixel (the reference MJD).
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
                mjd = da.astype(np.float64) + db.astype(np.float64) + offset
                # Pixels without coverage (both planes 0) stay 0, not the offset
                mjd[(da == 0) & (db == 0)] = 0.0
                hdul_o.append(fits.ImageHDU(data=mjd, header=hdu_a.header))

        hdul_o[0].header["MJDREF"] = (offset, "Reference MJD added to the planes")
        hdul_o.writeto(path_file_out, overwrite=overwrite)


def build_qc_summary_row(
    image_path: str,
    product_type: str,
    filter_keyword: str,
) -> dict:
    """Extract QC metrics for a single product (stack, tile or pawprint).

    All values are read from the image headers: the Phase 3 photometric cards
    of every data HDU (``PHOTZP``, ``PHOTZPER``, ``PSF_FWHM``, ``ELLIPTIC``,
    ``ABMAGLIM``, ``ABMAGSAT``) and the completeness cards of the first data
    HDU (``COMP90MD``, ``COMP50MD``).

    Parameters
    ----------
    image_path : str
        Path to the stack or tile FITS image.
    product_type : str
        ``"stack"``, ``"tile"`` or ``"pawprint"``.
    filter_keyword : str
        FITS keyword for the filter name (e.g. ``"HIERARCH ESO INS FILT1 NAME"``).

    Returns
    -------
    dict
        Row dictionary suitable for building an `astropy.table.Table`.

    """

    keys = ("PHOTZP", "PHOTZPER", "PSF_FWHM", "ELLIPTIC", "ABMAGLIM", "ABMAGSAT")
    values = {key: [] for key in keys}

    with fits.open(image_path) as hdul:
        phdr = hdul[0].header
        name = os.path.basename(image_path)
        filt = phdr.get(filter_keyword, "")
        ncombine = phdr.get("NCOMBINE", 0)
        astirms = phdr.get("ASTIRMS", np.nan)
        astrrms = phdr.get("ASTRRMS", np.nan)

        # Data HDUs: the primary of a tile image, else the image extensions
        if phdr.get("NAXIS") == 2:
            hdrs = [phdr]
        else:
            hdrs = [
                h.header
                for h in hdul[1:]
                if isinstance(h, fits.ImageHDU) and h.header["NAXIS"] == 2
            ]
        for hdr in hdrs:
            for key in keys:
                if key in hdr:
                    values[key].append(float(hdr[key]))
        comp90 = float(hdrs[0].get("COMP90MD", np.nan)) if hdrs else np.nan
        comp50 = float(hdrs[0].get("COMP50MD", np.nan)) if hdrs else np.nan

    def _median(key: str) -> float:
        return float(np.nanmedian(values[key])) if values[key] else np.nan

    photzp_scatter = (
        float(np.nanstd(values["PHOTZP"]))
        if product_type in ("stack", "pawprint") and values["PHOTZP"]
        else np.nan
    )

    def _round(value: float, digits: int) -> float:
        return round(value, digits) if not np.isnan(value) else np.nan

    return {
        "name": name,
        "type": product_type,
        "filter": filt,
        "n_combined": ncombine,
        "photzp": _round(_median("PHOTZP"), 4),
        "photzper": _round(_median("PHOTZPER"), 4),
        "photzp_scatter": _round(photzp_scatter, 4),
        "astirms": _round(astirms, 3),
        "astrrms": _round(astrrms, 3),
        "psf_fwhm": _round(_median("PSF_FWHM"), 3),
        "ellipticity": _round(_median("ELLIPTIC"), 3),
        "abmaglim": _round(_median("ABMAGLIM"), 3),
        "abmagsat": _round(_median("ABMAGSAT"), 3),
        "comp90": _round(comp90, 3),
        "comp50": _round(comp50, 3),
    }


def _compute_tile_edges(n: int, ntiles: int) -> list[tuple[int, int]]:
    """
    Split [0, n) into *ntiles* contiguous ranges whose union covers the interval.

    Returns list of (start, end) with end exclusive. First tiles may be 1 pixel
    larger when *n* is not evenly divisible.
    """
    base = n // ntiles
    rem = n % ntiles
    edges = []
    start = 0
    for i in range(ntiles):
        size = base + (1 if i < rem else 0)
        edges.append((start, start + size))
        start += size
    return edges


def _update_wcs_for_cutout(header: fits.Header, x0: int, y0: int) -> fits.Header:
    """Shift CRPIX so the WCS remains valid for a cutout starting at (x0, y0)."""
    hdr = header.copy()
    if "CRPIX1" in hdr:
        hdr["CRPIX1"] = hdr["CRPIX1"] - x0
    if "CRPIX2" in hdr:
        hdr["CRPIX2"] = hdr["CRPIX2"] - y0
    hdr["TIL_X0"] = (x0, "Tile origin X in parent image (0-based)")
    hdr["TIL_Y0"] = (y0, "Tile origin Y in parent image (0-based)")
    return hdr


def tile_fits(
    image_path: str,
    out_dir: str,
    tile_size_arcmin: float = 10.0,
    pixel_scale_arcsec: float = 1 / 3,
    overlap_pix: int = 0,
    weight_path: str | None = None,
    prefix: str = "tile",
    overwrite: bool = True,
    image_hdu: int | str = 0,
    weight_hdu: int | str = 0,
) -> tuple[list[dict], int]:
    """
    Tile a FITS image (and optional weight) into sub-images of a given angular size.

    Parameters
    ----------
    image_path : str
        Path to the input FITS image.
    out_dir : str
        Output directory for the tiles.
    tile_size_arcmin : float
        Target tile size in arcminutes.
    pixel_scale_arcsec : float
        Pixel scale in arcseconds.
    overlap_pix : int
        Overlap in pixels on each side of a tile.
    weight_path : str or None
        Path to the weight image. If given, weight tiles are written alongside.
    prefix : str
        Filename prefix for the output tiles.
    overwrite : bool
        Overwrite existing tile files.
    image_hdu : int or str
        Index or EXTNAME of the image HDU (default: the primary).
    weight_hdu : int or str
        Index or EXTNAME of the weight HDU (default: the primary).

    Returns
    -------
    tiles : list[dict]
        One dict per tile with keys ``"image"``, ``"weight"`` (or None), and
        ``"grid_index"`` (i, j).
    n_written : int
        Number of tiles that were newly written (0 when all existed already).

    """
    log = logging.getLogger(__name__)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_size_pix = int(round(tile_size_arcmin * 60.0 / pixel_scale_arcsec))
    log.info(
        f"Tiling {os.path.basename(image_path)} into ~{tile_size_arcmin}' "
        f"sub-tiles -> {out_dir}"
    )

    with fits.open(image_path, memmap=True) as hdul:
        data = hdul[image_hdu].data
        header = hdul[image_hdu].header

        if data is None or data.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image_path}")

        ny_img, nx_img = data.shape
        nx_tiles = max(1, int(round(nx_img / tile_size_pix)))
        ny_tiles = max(1, int(round(ny_img / tile_size_pix)))

        x_ranges = _compute_tile_edges(nx_img, nx_tiles)
        y_ranges = _compute_tile_edges(ny_img, ny_tiles)

        # Optionally open weight
        weight_data = None
        if weight_path is not None:
            whdul = fits.open(weight_path, memmap=True)
            weight_data = whdul[weight_hdu].data

        tiles = []
        n_written = 0
        # Per-sub-tile progress bar for minutes of memmap I/O. Main-thread
        # caller only; no-op off a TTY.
        from vircampype.pipeline.progress import track

        with track(
            f"Tiling {os.path.basename(image_path)}", nx_tiles * ny_tiles
        ) as advance:
            for j, (y_start, y_end) in enumerate(y_ranges):
                for i, (x_start, x_end) in enumerate(x_ranges):
                    # Expand by overlap, clip to image bounds
                    x0 = max(0, x_start - overlap_pix)
                    x1 = min(nx_img, x_end + overlap_pix)
                    y0 = max(0, y_start - overlap_pix)
                    y1 = min(ny_img, y_end + overlap_pix)

                    tile_header = _update_wcs_for_cutout(header, x0, y0)
                    tile_data = np.array(data[y0:y1, x0:x1], copy=False)

                    tile_name = f"{prefix}_x{i:03d}_y{j:03d}.fits"
                    tile_path = str(out_dir / tile_name)

                    write_tile = overwrite or not os.path.isfile(tile_path)
                    # Defined here: the weight check below needs it too.
                    expected_shape = (y1 - y0, x1 - x0)

                    # Verify existing tile matches expected geometry
                    if not write_tile:
                        with fits.open(tile_path) as existing:
                            eh = existing[0].header
                            existing_ok = (
                                eh.get("NAXIS1") == expected_shape[1]
                                and eh.get("NAXIS2") == expected_shape[0]
                                and eh.get("TIL_X0") == x0
                                and eh.get("TIL_Y0") == y0
                            )
                        if not existing_ok:
                            raise ValueError(
                                f"Existing tile {tile_path} does not match "
                                f"expected geometry (shape={expected_shape}, "
                                f"origin=({x0},{y0})). Remove it or use "
                                f"overwrite=True."
                            )

                    if write_tile:
                        fits.writeto(
                            tile_path,
                            tile_data,
                            header=tile_header,
                            overwrite=overwrite,
                            output_verify="fix",
                        )
                        n_written += 1

                    tile_info = {
                        "image": tile_path,
                        "weight": None,
                        "grid_index": (i, j),
                    }

                    # Write weight tile
                    if weight_data is not None:
                        weight_tile = np.array(weight_data[y0:y1, x0:x1], copy=False)
                        weight_name = f"{prefix}_x{i:03d}_y{j:03d}.weight.fits"
                        weight_tile_path = str(out_dir / weight_name)

                        write_weight = overwrite or not os.path.isfile(weight_tile_path)

                        if not write_weight:
                            with fits.open(weight_tile_path) as existing:
                                eh = existing[0].header
                                existing_ok = (
                                    eh.get("NAXIS1") == expected_shape[1]
                                    and eh.get("NAXIS2") == expected_shape[0]
                                    and eh.get("TIL_X0") == x0
                                    and eh.get("TIL_Y0") == y0
                                )
                            if not existing_ok:
                                raise ValueError(
                                    f"Existing weight tile {weight_tile_path} "
                                    f"does not match expected geometry."
                                )

                        if write_weight:
                            fits.writeto(
                                weight_tile_path,
                                weight_tile,
                                header=tile_header,
                                overwrite=overwrite,
                                output_verify="fix",
                            )
                        tile_info["weight"] = weight_tile_path

                    tiles.append(tile_info)
                    advance()

        if weight_data is not None:
            whdul.close()

    log.info(
        f"Tiled {os.path.basename(image_path)} into {len(tiles)} sub-tiles "
        f"({n_written} newly written)"
    )
    return tiles, n_written


def build_qc_summary(
    setup,
    stacks=None,
    tile=None,
    pawprints=None,
) -> str | None:
    """Build a QC summary table aggregating key metrics from stacks, tile and pawprints.

    The summary is rebuilt on every call and the file is overwritten; the pipeline
    status ``qc_summary`` decides whether this function is called.

    Parameters
    ----------
    setup
        Pipeline Setup instance.
    stacks
        Stack images (FitsImages), or None.
    tile
        Tile image (FitsImages), or None.
    pawprints
        Pawprint product images (FitsImages), or None.

    Returns
    -------
    str or None
        Path to the written summary table, or None if no rows were collected.

    """

    print_header(
        header="QC SUMMARY TABLE",
        silent=setup.silent,
        left=None,
        right=None,
    )
    log = logging.getLogger(__name__)
    tstart = time.time()

    path_out = os.path.join(setup.folders["qc"], "qc_summary.ecsv")
    kw = {"filter_keyword": setup.keywords.filter_name}
    rows = []

    # Collect rows from stacks
    if stacks is not None:
        for idx in range(len(stacks.paths_full)):
            try:
                rows.append(
                    build_qc_summary_row(
                        image_path=stacks.paths_full[idx],
                        product_type="stack",
                        **kw,
                    )
                )
            except Exception as e:
                log.warning(f"QC summary: skipping stack {idx}: {e}")

    # Collect row from tile
    if tile is not None:
        try:
            rows.append(
                build_qc_summary_row(
                    image_path=tile.paths_full[0],
                    product_type="tile",
                    **kw,
                )
            )
        except Exception as e:
            log.warning(f"QC summary: skipping tile: {e}")

    # Collect rows from pawprints
    if pawprints is not None:
        for idx in range(len(pawprints.paths_full)):
            try:
                rows.append(
                    build_qc_summary_row(
                        image_path=pawprints.paths_full[idx],
                        product_type="pawprint",
                        **kw,
                    )
                )
            except Exception as e:
                log.warning(f"QC summary: skipping pawprint {idx}: {e}")

    if not rows:
        print_message(
            message="No products available for QC summary",
            kind="warning",
            end=None,
        )
        return None

    # Build and write table
    qc_table = Table(rows=rows)
    qc_table.write(path_out, format="ascii.ecsv", overwrite=True)

    print_message(
        message=f"QC summary written to {os.path.basename(path_out)} "
        f"({len(rows)} products)",
        kind="okblue",
        end="\n",
        logger=log,
    )
    print_message(
        message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
        kind="okblue",
        end="\n",
    )

    return path_out
