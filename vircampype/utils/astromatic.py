from vircampype.utils.miscellaneous import read_setup


def yml2config(path, **kwargs):
    """
    Reads a YML file at a given path and converts the entries to a string that can be passed to astromatic tools.

    Parameters
    ----------
    path : str
        Path to YML file.
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

        # Convert key to lower case
        key = key.lower()

        # Overwrite with kwargs
        if key in kwargs:
            s += "-{0} {1} ".format(key.upper(), kwargs[key])
        else:
            s += "-{0} {1} ".format(key.upper(), val)

    return s
