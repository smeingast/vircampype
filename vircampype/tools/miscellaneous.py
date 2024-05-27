import numpy as np
from typing import List, Callable, Any, Tuple, Union

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
    "convert_position_error",
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
    from vircampype.tools.mathtools import clipped_median, clipped_mean

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
    id2 = np.around(skycoord.dec.degree, decimals=6)/np.sign(
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


def convert_position_error(errmaj: Union[np.ndarray, list, tuple],
                           errmin: Union[np.ndarray, list, tuple],
                           errpa: Union[np.ndarray, list, tuple],
                           degrees: bool = True) -> (
        Tuple)[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Right Ascension (RA) and Declination (Dec) errors and
    their correlation coefficient from major and minor errors and position angle.

    Parameters
    ----------
    errmaj : np.ndarray, list, tuple
        Major axis errors.
    errmin : np.ndarray, list, tuple
        Minor axis errors.
    errpa : np.ndarray, list, tuple
        Position angle errors; in degrees if degrees=True, otherwise in radians.
    degrees : bool, optional
        Indicates whether the position angle is in degrees (default is True).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        RA error, Dec error, and RA-Dec correlation coefficient, each as an np.ndarray.

    Notes
    -----
    The function converts position angle errors from degrees to radians if necessary,
    calculates cosine and sine of these angles, and uses these to compute elements of
    the covariance matrix. Variances along the RA and Dec directions and the covariance
    between RA and Dec are also computed, which are then used to derive the standard
    deviations (errors) and correlation coefficient.

    """
    # Make sure the input is a numpy array
    errmaj, errmin = np.asarray(errmaj), np.asarray(errmin)

    # Convert position angles from degrees to radians
    if degrees:
        theta_rad = np.deg2rad(errpa)
    else:
        theta_rad = np.asarray(errpa)

    # Calculate the components of the rotation matrix for each set
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    # Preallocate the 3D matrix array (N sets of 2x2 matrices)
    cc = np.zeros((len(errmaj), 2, 2))

    # Define each element of the covariance matrices
    cc[:, 0, 0] = cos_theta ** 2*errmaj ** 2 + sin_theta ** 2*errmin ** 2  # var_RA
    cc[:, 0, 1] = (cos_theta*sin_theta)*(errmaj ** 2 - errmin ** 2)  # cov_RA_Dec
    cc[:, 1, 0] = cc[:, 0, 1]  # cov_Dec_RA (symmetric to cov_RA_Dec)
    cc[:, 1, 1] = sin_theta ** 2*errmaj ** 2 + cos_theta ** 2*errmin ** 2  # var_Dec

    # Compute the RA and Dec errors and correlation coefficients
    ra_error = np.sqrt(cc[:, 0, 0])
    dec_error = np.sqrt(cc[:, 1, 1])
    ra_dec_corr = cc[:, 0, 1]/np.sqrt(cc[:, 0, 0]*cc[:, 1, 1])

    # Return
    return ra_error, dec_error, ra_dec_corr
