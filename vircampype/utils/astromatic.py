from itertools import groupby
from astropy.io.fits.header import Header
from vircampype.utils.miscellaneous import read_setup


def yml2config(path, skip=None, **kwargs):
    """
    Reads a YML file at a given path and converts the entries to a string that can be passed to astromatic tools.

    Parameters
    ----------
    path : str
        Path to YML file.
    skip : list, optional
        If set, ignore the given keywords in the list
    kwargs
        Any available setup parameter can be overwritten (e.g. catalog_name="catalog.fits")

    Returns
    -------
    str
        Full string constructed from YML setup.

    """

    setup = read_setup(path_yaml=path)

    # Loop over setup and construct command
    s = ""
    for key, val in setup.items():

        # Skip if set
        if skip is not None:
            if key.lower() in [s.lower() for s in skip]:
                continue

        # Convert key to lower case
        key = key.lower()

        # Strip any whitespace
        if isinstance(val, str):
            val = val.replace(" ", "")

        # Overwrite with kwargs
        if key in kwargs:
            s += "-{0} {1} ".format(key.upper(), kwargs[key])
        else:
            s += "-{0} {1} ".format(key.upper(), val)

    return s


def read_scamp_header(path):

    with open(path, "r") as file:
        # data = file.read()
        header = file.readlines()

    # Clean content for non ASCII characters
    header_clean = []
    for idx in range(len(header)):
        if header[idx].startswith("COMMENT") or header[idx].startswith("HISTORY"):
            continue
        else:
            header_clean.append(header[idx])

    # Group headers by chip
    headers_split = [list(group) for k, group in groupby(header_clean, lambda x: x.startswith("END")) if not k]

    # Convert to headers and return
    return [Header.fromstring("\n".join(hc), sep="\n") for hc in headers_split]


def replace_astrometry(headers, path_scamp_hdr):

    # Read scamp header
    headers_scamp = read_scamp_header(path_scamp_hdr)

    # Loop over input headers and put scamp data
    for hi, hs in zip(headers, headers_scamp):

        # replace keywords
        for card in hs.cards:
            hi[card.keyword] = (card.value, card.comment)

    return headers
