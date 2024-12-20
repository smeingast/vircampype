import os.path
import re
import warnings
import numpy as np

from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from typing import Union, Optional
from vircampype.pipeline.misc import *
from astropy.coordinates import SkyCoord
from vircampype.tools.miscellaneous import *
from vircampype.pipeline.log import PipelineLog
from astropy.io.fits.verify import VerifyWarning
from vircampype.tools.wcstools import header_reset_wcs
from vircampype.tools.systemtools import which, run_commands_shell_parallel

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
]


def check_card_value(value):
    """
    Checks if the given value for a FITS header entry is valid and transforms it to a
    writeable parameter.

    Parameters
    ----------
    value
        The value to check

    Returns
    -------
    Checked value

    """

    # If the value is a callable:
    val = func2string(value) if hasattr(value, "__call__") else value

    # Convert to string if necessary
    if (
        not (isinstance(val, str))
        | (isinstance(val, (np.floating, float)))
        | (isinstance(val, (np.integer, int)))
    ):
        val = str(val)

    # Return
    return val


def make_card(keyword, value, comment=None, upper=True):
    """
    Create a FITS header card based on keyword, value, and comment.

    Parameters
    ----------
    keyword : str
        The keyword for the FITS card.
    value
        The value to write for the given keyword
    comment : optional, str
        Optionally, a comment to write.
    upper : optional, bool
        Whether to conert the keyword to upper case.

    Returns
    -------
    FITS Card

    """

    # Make upper case if set
    kw = keyword.upper() if upper else keyword

    # Remove double spaces
    kw = re.sub(" +", " ", kw)

    # Check value
    val = check_card_value(value=value)

    # Raise error if card too long
    lcom = len(comment) if comment is not None else 0
    ltot = len(kw) + len(str(val)) + lcom
    if ltot > 80:
        raise ValueError("Card too long ({0})".format(ltot))

    # Return card
    return fits.Card(keyword=kw, value=val, comment=comment)


def make_cards(keywords, values, comments=None):
    """
    Creates a list of FITS header cards from given keywords, values, and comments

    Parameters
    ----------
    keywords : list[str]
        List of keywords.
    values : list
        List of values.
    comments : list[str], optional
        List of comments.
    Returns
    -------
    iterable
        List containing FITS header cards.

    """

    # Length of input must match
    if not isinstance(keywords, list) | isinstance(values, list):
        raise TypeError("keywords and values must be lists")

    # Length must be the same for keywords and values
    if len(keywords) != len(values):
        raise ValueError("Keywords and Values don't match")

    # If comments are supplied, they must match
    if comments is not None:
        if len(comments) != len(keywords):
            raise ValueError("Comments don't match input")
    # If nothing is supplied we just have None
    else:
        comments = [None for _ in range(len(keywords))]

    # Create FITS header cards
    cards = []
    for kw, val, cm in zip(keywords, values, comments):
        cards.append(make_card(keyword=kw, value=val, comment=cm))

    # Return
    return cards


def copy_keywords(path_1, path_2, keywords, hdu_1=0, hdu_2=0):
    """
    Copies specific keywords from file 2 to file 1. Also both HDUs can be specified.
    Default are primary HDUs.

    Parameters
    ----------
    path_1 : str
        Path to file where the keywords should be copied to.
    path_2 : str
        Path to file where the keywords should be copied from.
    keywords : iterable
        List of keywords to copy.
    hdu_1 : int, optional
        Extension number where to copy to. Default is 0 (primary).
    hdu_2 : int, optional
        Extension number where to copy from. Default is 0 (primary).

    """

    # Get HDUlists for both files
    with fits.open(path_1, mode="update") as hdulist_1, fits.open(
        path_2, mode="readonly"
    ) as hdulist_2:
        # Loop over files and update header
        for k in keywords:
            hdulist_1[hdu_1].header[k] = hdulist_2[hdu_2].header[k]


def add_key_primary_hdu(path, key, value, comment=None):
    """
    Add key/value/comment to primary HDU.

    Parameters
    ----------
    path : str
        Path to file.
    key : str
        Key to be added/modified.
    value : str, int, float
        Value of card to be added.
    comment : str, optional
        If set, also write a comment

    """

    with fits.open(path, "update") as file:
        if comment is not None:
            file[0].header[key] = (value, comment)
        else:
            file[0].header[key] = value


def make_mef_image(
    paths_input,
    path_output,
    primeheader=None,
    add_constant=None,
    write_extname=True,
    overwrite=False,
):
    """
    Creates an MEF image file from multiple input image file.

    Parameters
    ----------
    paths_input : iterable
        List of input paths.
    path_output : str
        Path of output file.
    primeheader : fits.Header, optional
        If set, the primary header for the output file.
    add_constant : int, float, str, optional
        A constant value that is added to each input file upon combining the files.
        If given as a string, then
        the value of each added constant will be read from the header.
    write_extname : bool, optional
        If set, write standard EXTNAME keyword.
    overwrite : bool, optional
        Whether an existing file should be overwritten.

    """

    if len(paths_input) == 0:
        raise ValueError("No images to combine")

    # Make add_constant loopable if passed as None or string or constant
    if not hasattr(add_constant, "len"):
        add_constant = [add_constant] * len(paths_input)

    # Create empty HDUlist
    hdulist = fits.HDUList()

    # Make Primary header
    if primeheader is None:
        primeheader = fits.Header()

    # Put primary HDU
    hdulist.append(fits.PrimaryHDU(header=primeheader))

    # Construct image HDUs from input
    for pidx, ac in zip(range(len(paths_input)), add_constant):
        with fits.open(paths_input[pidx]) as file:
            # Determine constant to add
            if isinstance(ac, (int, float)):
                const = ac
            elif isinstance(ac, str):
                const = file[0].header[ac]
            else:
                const = 0

            # Grab header
            hdr = file[0].header.copy()

            # Write EXTNAME
            if write_extname:
                hdr.set("EXTNAME", value="HDU{0:>02d}".format(pidx + 1), after="BITPIX")

            # Append HDU
            hdulist.append(fits.ImageHDU(data=file[0].data + const, header=hdr))

    # Write final HDUlist to disk
    hdulist.writeto(path_output, overwrite=overwrite)


def merge_headers(path_1, path_2, primary_only=False):
    """
    Merges header entries of file 2 into file 1, in the sense that every new item in
    header 2 that is not present in header 1, is copied to file 1.
    Forces a new write of the fits file in the end (flush).

    Parameters
    ----------
    path_1 : str
        Path of file 1. Where keywords are copied to.
    path_2 : str
        Path of file 2. Where keywords are taken from.
    primary_only : bool, optional
        If only primary header should be merged.

    """

    skip_list = ["SIMPLE", "NAXIS", "NAXIS1", "NAXIS2"]

    # Get HDUlists for both files
    with fits.open(path_1, mode="update") as hdulist_1, fits.open(
        path_2, mode="readonly"
    ) as hdulist_2:
        # Iterate over HDUs
        for hdu1, hdu2 in zip(hdulist_1, hdulist_2):
            # Check for Primary HDU
            if primary_only:
                if not isinstance(hdu1, fits.PrimaryHDU):
                    continue

            keys1 = list(hdu1.header.keys())

            # Iterate over every item in 2
            for key2, val2 in hdu2.header.items():
                if key2 in skip_list:
                    continue

                # If not in header 1, put there, but ignore HIERARCH warnings
                if key2 not in keys1:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=VerifyWarning)
                        hdu1.header[key2] = val2

        # Flush changes to first file
        hdulist_1.flush()


def add_float_to_header(
    header, key, value, decimals=3, comment=None, remove_before=True
):
    """
    Adds float to header with fixed format.

    Parameters
    ----------
    header : fits.Header
        FITS header to be modified.
    key : str
        Key of header entry.
    value : float, ndarray
        Value of header entry.
    decimals : int, optional
        How many decimals to write
    comment : str, optional
        Comment of header entry.
    remove_before : bool, optional
        If set, removes all occurences of 'key' from header. Default is true

    """
    # If the key is already there, remove it
    if remove_before:
        try:
            header.remove(key, remove_all=True)
        except KeyError:
            pass

    if decimals == 1:
        c = fits.Card.fromstring("{0:8}= {1:0.1f}".format(key, value))
    elif decimals == 2:
        c = fits.Card.fromstring("{0:8}= {1:0.2f}".format(key, value))
    elif decimals == 3:
        c = fits.Card.fromstring("{0:8}= {1:0.3f}".format(key, value))
    elif decimals == 4:
        c = fits.Card.fromstring("{0:8}= {1:0.4f}".format(key, value))
    elif decimals == 5:
        c = fits.Card.fromstring("{0:8}= {1:0.5f}".format(key, value))
    elif decimals == 6:
        c = fits.Card.fromstring("{0:8}= {1:0.6f}".format(key, value))
    else:
        raise ValueError("Add more options for decimals")
    c.comment = comment
    header.append(c)


def add_int_to_header(header, key, value, comment=None, remove_before=True):
    if remove_before:
        try:
            header.remove(key, remove_all=True)
        except KeyError:
            pass
    c = fits.Card.fromstring("{0:8}= {1:0d}".format(key, value))
    c.comment = comment
    header.append(c)


def add_str_to_header(header, key, value, comment=None, remove_before=True):
    if remove_before:
        try:
            header.remove(key, remove_all=True)
        except KeyError:
            pass
    if comment is None:
        comment = ""
    header[key] = (value, comment)


def convert_bitpix_image(path, new_type, offset=None):
    """
    Converts image data to the requested data type across all HDUs.

    Parameters
    ----------
    path : str
        Path to FITS file.
    new_type
        New data type.
    offset : int, float, optional
        Optional offset to add to the data.

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


def delete_keyword_from_header(header, keyword):
    """
    Deletes given keyword from header.

    Parameters
    ----------
    header : fits.Header
        astropy fits header.
    keyword : str
        Which keyword to delete

    Returns
    -------
    fits.Header
        Cleaned fits header.

    """
    try:
        del header[keyword]
    except KeyError:
        pass
    return header


def replace_data(file_a: str, file_b: str):
    """
    Copies all data from file A to file B, but leaves the headers untouched.

    Parameters
    ----------
    file_a : str
        Path to file A.
    file_b : str
        Path to file B.

    """

    # Open both files
    hdul_a = fits.open(file_a)
    hdul_b = fits.open(file_b, mode="update")

    # Number of HDUs must be equal
    if len(hdul_a) != len(hdul_b):
        raise ValueError(
            "Number of HDUs not equal. n(A)={0}; n(B)={1}".format(
                len(hdul_a), len(hdul_b)
            )
        )

    # Loop over HDUs:
    for idx_hdu in range(len(hdul_a)):
        hdu_a, hdu_b = hdul_a[idx_hdu], hdul_b[idx_hdu]

        # Extension type must match
        if hdu_a.__class__ != hdu_b.__class__:
            raise ValueError("Extension types to not match")

        # Copy data
        hdu_b.data = hdu_a.data

    # Flush data to file B
    hdul_b.flush()

    # Close files
    hdul_a.close()
    hdul_b.close()


def mjd2dateobs(mjd):
    """
    Convert MJD to fits date-obs format.

    Parameters
    ----------
    mjd : float

    Returns
    -------
    str
        DATE-OBS string.

    """
    return Time(mjd, format="mjd").fits


def fix_vircam_headers(prime_header, data_headers):
    """Resets the WCS info and purges useless keywords from vircam headers."""
    log = PipelineLog()

    try:
        log.info("Attempting to rewrite WCS info in headers")

        tra = str(prime_header["HIERARCH ESO TEL TARG ALPHA"])
        tde = str(prime_header["HIERARCH ESO TEL TARG DELTA"])
        log.info(f"Found RA/DEC in headers: {tra} / {tde}")

        # Get declination sign and truncate string if necessary
        if tde.startswith("-"):
            decsign = -1
            tde = tde[1:]
        else:
            decsign = 1

        # Silly fix for short ALPHA/DELTA strings
        tra = "0" * (6 - len(tra.split(".")[0])) + tra
        tde = "0" * (6 - len(tde.split(".")[0])) + tde

        # Compute field RA/DEC
        fra = 15 * (float(tra[:2]) + float(tra[2:4]) / 60 + float(tra[4:]) / 3600)
        fde = decsign * (float(tde[:2]) + float(tde[2:4]) / 60 + float(tde[4:]) / 3600)
        log.info(f"Computed RA/DEC: {fra} / {fde}")

    except KeyError:
        log.warning("Could not find RA/DEC in headers")
        fra, fde = None, None

    for idx_hdr in range(len(data_headers)):
        # Overwrite with consistently working keyword
        try:
            crval1 = fra if fra is not None else data_headers[idx_hdr]["CRVAL1"]
            log.info(f"Overwriting CRVAL1 with {crval1} in extension {idx_hdr + 1}")
            data_headers[idx_hdr]["CRVAL1"] = crval1
            crval2 = fde if fde is not None else data_headers[idx_hdr]["CRVAL2"]
            log.info(f"Overwriting CRVAL2 with {crval2} in extension {idx_hdr + 1}")
            data_headers[idx_hdr]["CRVAL2"] = crval2
        except KeyError:
            log.warning("Could not reset CRVAL1/CRVAL2 in headers")
            pass

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            data_headers[idx_hdr] = header_reset_wcs(data_headers[idx_hdr])

        # Remove useless keywords if set
        for kw in useless_extension_keywords:
            # log.info(f"Removing '{kw}' from extension {idx_hdr + 1}")
            data_headers[idx_hdr].remove(kw, ignore_missing=True, remove_all=True)

    # Purge also primary header
    for kw in useless_primary_keywords:
        # log.info(f"Removing '{kw}' from primary header")
        prime_header.remove(kw, ignore_missing=True, remove_all=True)


def compress_images(images, q=4, exe="fpack", n_jobs=1):
    """
    Compress images in parallel with fpack.

    Parameters
    ----------
    images : list, iterable
        List of images.
    q : int, float, optional
        Compression ratio.
    exe : str, optional
        Binary name.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------

    """

    # Find executable
    fpack = which(exe)

    # Check if files are already there
    paths_out = [x.replace(".fits", ".fits.fz") for x in images]
    done = [os.path.isfile(x) for x in paths_out]

    # Build commands
    cmds = ["{0} -q {1} {2}".format(fpack, q, x) for x in images]

    # Clean commands
    cmds = [c for c, d in zip(cmds, done) if not d]

    # Run commands
    run_commands_shell_parallel(cmds=cmds, n_jobs=n_jobs)


def make_gaia_refcat(
    table_in: Table,
    path_ldac_out: str,
    epoch_in: Union[int, float] = 2016.0,
    epoch_out: Optional[Union[int, float]] = None,
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
    key_gflux_error: str = "flux_error"
) -> Table:
    """
    Generates an astrometric reference catalog based on downloaded Gaia data.


    Parameters
    ----------
    table_in : Table
        Input table.
    path_ldac_out : str
        Path to output table.
    epoch_in : float
        Input epoch of catalog.
    epoch_out : float, optional
        If set, transforms the coordinates to the given output epoch.
    key_ra : str, optional
        Key for RA.
    key_ra_error : str, optional
        Key for RA error.
    key_dec : str, optional
        Key for DEC.
    key_dec_error : str, optional
        Key for DEC error.
    key_pmra : str, optional
        Key for proper motion in RA.
    key_pmra_error : str, optional
        Key for proper motion error in RA.
    key_pmdec : str, optional
        Key for proper motion in DEC.
    key_pmdec_error : str, optional
        Key for proper motion error in DEC.
    key_ruwe : str, optional
        Key for RUWE.
    key_gmag : str, optional
        Key for G magnitude.
    key_gflux : str, optional
        Key for G flux.
    key_gflux_error : str, optional
        Key for G flux error.

    Returns
    -------
    Table
        Output table.

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

    # Transform positions
    if epoch_out is not None:
        sc = SkyCoord(
            ra=table_in[key_ra],
            dec=table_in[key_dec],
            pm_ra_cosdec=table_in[key_pmra],
            pm_dec=table_in[key_pmdec],
            obstime=Time(epoch_in, format="decimalyear"),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sc = sc.apply_space_motion(
                new_obstime=Time(epoch_out, format="decimalyear")
            )
    else:
        epoch_out = epoch_in

    # Create output table
    table_out = Table()

    # Add positions
    table_out["ra"] = sc.ra.degree
    table_out["ra_error"] = table_in[key_ra_error].value / 3_600_000
    table_out["dec"] = sc.dec.degree
    table_out["dec_error"] = table_in[key_dec_error].value / 3_600_000

    # Add proper motions
    table_out["pmra"] = table_in[key_pmra].value
    table_out["pmra_error"] = table_in[key_pmra_error].value
    table_out["pmdec"] = table_in[key_pmdec].value
    table_out["pmdec_error"] = table_in[key_pmdec_error].value

    # Add magnitudes
    table_out["mag"] = table_in[key_gmag].value
    table_out["mag_error"] = (
        1.0857
        * table_in[key_gflux_error].value
        / table_in[key_gflux].value
    )

    # Add date
    table_out["obsdate"] = epoch_out

    # Sort by DEC
    sortindex = np.argsort(table_out["dec"])
    for nn in table_out.columns.keys():
        table_out[nn] = table_out[nn][sortindex]

    # Write temporary fits_table
    path_temp = "/tmp/astr_ref_temp.cat.fits"
    table_out.write(path_temp, overwrite=True)

    # Convert to LDAC
    fits2ldac(path_in=path_temp, path_out=path_ldac_out)

    # Remove temporary file
    os.remove(path_temp)

    # Return warped table
    return table_out


def fits2ldac(path_in, path_out, extension=1):
    """
    Convert fits table on disk to LDAC-compatible format (for scamp).
    Copied from https://codingdict.com/sources/py/astropy.io.fits/18055.html
    """

    # Read data
    data, header = fits.getdata(path_in, extension, header=True)

    # Create HDUs
    ext2_str = header.tostring(endcard=False, padding=False)
    ext2_data = np.array([ext2_str])
    formatstr = str(len(ext2_str)) + "A"
    col1 = fits.Column(name="Field Header Card", array=ext2_data, format=formatstr)
    cols = fits.ColDefs([col1])
    ext2 = fits.BinTableHDU.from_columns(cols)
    ext2.header["EXTNAME"] = "LDAC_IMHEAD"
    ext2.header["TDIM1"] = "(80, {0})".format(len(ext2_str) / 80)
    ext3 = fits.BinTableHDU(data)
    ext3.header["EXTNAME"] = "LDAC_OBJECTS"
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)

    # Write HDUList to output LDAC fits table
    hdulist = fits.HDUList([prihdu, ext2, ext3])
    hdulist.writeto(path_out, overwrite=True)
    hdulist.close()


def combine_mjd_images(path_file_a, path_file_b, path_file_out, overwrite=False):
    with fits.open(path_file_a) as hdul_a, fits.open(path_file_b) as hdul_b:
        # Files must have same number of HDUs
        if len(hdul_a) != len(hdul_b):
            raise ValueError("Files incompatible")

        # Instantiate output HDU list
        hdul_o = fits.HDUList([])

        # Loop over HDUs and combine
        for hdu_a, hdu_b in zip(hdul_a, hdul_b):
            da, db = hdu_a.data, hdu_b.data
            if (da is None) & (db is None):
                hdul_o.append(hdu_a)
            elif da.ndim == 2:
                hdul_o.append(
                    fits.ImageHDU(
                        data=da.astype(np.float64) + db.astype(np.float64),
                        header=hdu_a.header,
                    )
                )

        # Write to disk
        hdul_o.writeto(path_file_out, overwrite=overwrite)


if __name__ == "__main__":
    path_base = "/Users/stefan/iCloud/Projects/VISIONS/Pipeline/other/download/"
    t_in_path = f"{path_base}merged.fits"
    t_out = f"{path_base}merged.ldac.fits"
    t_in = Table.read(t_in_path)
    e_in = 2016.0
    e_out = 2016.0
    make_gaia_refcat(table_in=t_in, path_ldac_out=t_out, epoch_in=e_in, epoch_out=e_out,
                     key_ra="ra", key_ra_error="ra_error", key_dec="dec",
                     key_dec_error="dec_error", key_pmra="pmra",
                     key_pmra_error="pmra_error", key_pmdec="pmdec",
                     key_pmdec_error="pmdec_error", key_ruwe="ruwe",
                     key_gmag="phot_g_mean_mag", key_gflux="phot_g_mean_flux",
                     key_gflux_error="phot_g_mean_flux_error")
