from typing import Any, Callable, List, Union

import numpy as np

__all__ = [
    "string2list",
    "string2func",
    "func2string",
    "prune_list",
    "flat_list",
    "convert_dtype",
    "fits2numpy",
    "numpy2fits",
    "skycoord2visionsid",
    "write_list",
]


def convert_dtype(dtype):
    try:
        cdict = dict(
            bool="i1", int16="i2", int32="i4", int64="i8", float32="f4", float64="f8"
        )
        return cdict[dtype]
    except KeyError:
        return dtype


# Copied from astropy
fits2numpy = {
    "L": "i1",
    "B": "u1",
    "I": "i2",
    "J": "i4",
    "K": "i8",
    "E": "f4",
    "D": "f8",
    "C": "c8",
    "M": "c16",
    "A": "a",
}
numpy2fits = {val: key for key, val in fits2numpy.items()}
for i in np.arange(1, 30):
    numpy2fits[f"U{i}"] = f"{i}A"


def string2func(s: str) -> Callable:
    """
    Converts a string to a corresponding statistical function.

    The allowed input functions are "median", "mean", "clipped_median" and
    "clipped_mean". Input is case-insensitive.

    Parameters
    ----------
    s : str
        Input string representing a function name.

    Returns
    -------
    Callable
        Corresponding statistical function.

    Raises
    ------
    ValueError
        If input string does not correspond to an allowed function.
    """

    # Import
    from vircampype.tools.mathtools import clipped_mean, clipped_median

    if s.lower() == "median":
        return np.nanmedian
    if s.lower() == "mean":
        return np.nanmean
    if s.lower() == "clipped_median":
        return clipped_median
    if s.lower() == "clipped_mean":
        return clipped_mean
    else:
        raise ValueError(f"Metric '{s}' not supported")


def func2string(func):
    """
    Simple helper function to return a string for fits header card construction.

    Parameters
    ----------
    func : callable
        Callable to check

    Returns
    -------
    str
        String describing the function.

    """

    if "median" in str(func):
        return "median"
    elif "mean" in str(func):
        return "mean"


def flat_list(inlist: List[Union[List[Any], np.ndarray]]) -> List[Any]:
    """
    Flattens a list with sublists.

    Parameters
    ----------
    inlist : List[List[Any]]
        The input list of lists to be flattened.

    Returns
    -------
    List[Any]
        A flattened list where each element of the sublists is now an
        element of a single list.
    """

    return [item for sublist in inlist for item in sublist]


def string2list(s, sep=",", dtype=float):
    """
    Separates a string into list elements

    Parameters
    ----------
    s : str
        String to separate.
    sep : str
        Separator in string.
    dtype
        Data dtype

    Returns
    -------
    iterable
        Split list.
    """

    return [dtype(x) for x in s.split(sep)]


def prune_list(ll, n_min):
    """
    Removes all list entries which contain fewer items than 'n_min'.

    Parameters
    ----------
    ll : List
        Input list to prune.
    n_min : int
        Minimum length in list entries.

    Returns
    -------
    List
        Cleaned list.

    """

    # Loop over entries and get good indices
    popidx = [idx for idx in range(len(ll)) if len(ll[idx]) < n_min]

    # Remove the bad ones
    for idx in sorted(popidx, reverse=True):
        ll.pop(idx)

    return ll


# TODO: This should only be used for the vizier catalog
def skycoord2visionsid(skycoord):
    """
    Constructs the VISIONS ID from astropy sky coordinates.

    Parameters
    ----------
    skycoord : SkyCoord
        Astropy SkyCoord instance.

    Returns
    -------
    iterable
        List with IDs for each entry in skycoord.

    """

    # Determine declination sign
    sign = ["-" if np.sign(dec) < 0.0 else "+" for dec in skycoord.dec.degree]

    # Construct id
    id1 = np.around(skycoord.ra.degree, decimals=6)
    id2 = np.around(skycoord.dec.degree, decimals=6) / np.sign(
        np.around(skycoord.dec.degree, decimals=6)
    )

    # Return string
    return [
        "{0:0>10.6f}{1}{2:0>9.6f}".format(ra, s, dec)
        for ra, s, dec in zip(id1, sign, id2)
    ]


def write_list(path_file: str, lst: List):
    """
    Write list to a given file.

    Parameters
    ----------
    path_file : str
        Output path.
    lst : List
        input list.

    """
    with open(path_file, "w") as outfile:
        outfile.write("\n".join(lst))
        outfile.write("\n")
