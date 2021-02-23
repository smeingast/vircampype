import numpy as np

__all__ = ["string2list", "string2func", "func2string", "prune_list", "flat_list", "convert_dtype", "fits2numpy",
           "numpy2fits", "skycoord2visionsid"]


convert_dtype = dict(int16="i2", int32="i4", int64="i8", float32="f4", float64="f8")
# Copied from astropy
fits2numpy = {'L': 'i1', 'B': 'u1', 'I': 'i2', 'J': 'i4', 'K': 'i8', 'E': 'f4',
              'D': 'f8', 'C': 'c8', 'M': 'c16', 'A': 'a'}
numpy2fits = {val: key for key, val in fits2numpy.items()}


def string2func(s):
    if s.lower() == "median":
        return np.nanmedian
    elif s.lower() == "mean":
        return np.nanmean
    else:
        raise ValueError("Metric '{0}' not suppoerted".format(s))


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


def flat_list(inlist):
    """ Flattens a list with sublists. """
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
    sign = ["-" if np.sign(dec) < 0. else "+" for dec in skycoord.dec.degree]

    # Construct id
    id1 = np.around(skycoord.ra.degree, decimals=6)
    id2 = np.around(skycoord.dec.degree, decimals=6) / np.sign(np.around(skycoord.dec.degree, decimals=6))

    # Return string
    return ["{0:0>10.6f}{1}{2:0>9.6f}".format(ra, s, dec) for ra, s, dec in zip(id1, sign, id2)]
