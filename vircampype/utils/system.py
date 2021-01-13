# =========================================================================== #
import os
import sys
import yaml
import shutil
import importlib
import subprocess

from pkgutil import iter_modules
from itertools import zip_longest


# Define objects in this module
__all__ = ["run_cmds", "run_command_bash", "module_exists", "which", "read_setup", "remove_file", "copy_file",
           "make_folder", "yml2config", "get_resource_path", "notify"]


def run_cmds(cmds, n_processes=1, silent=True):
    """
    Runs a list of shell commands

    Parameters
    ----------
    cmds : list
        List of shell commands#
    n_processes : int, optional
        Number of parallel processes.
    silent : bool, optional
        Whether or not to print information about the process. Default is True.

    Returns
    -------

    """

    if silent:
        groups = [(subprocess.Popen(cmd, shell=True, executable="/bin/zsh", stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL) for cmd in cmds)] * n_processes
    else:
        groups = [(subprocess.Popen(cmd, shell=True, executable="/bin/zsh") for cmd in cmds)] * n_processes

    # Run processes
    for processes in zip_longest(*groups):  # run len(processes) == limit at a time
        for p in filter(None, processes):
            p.wait()


def run_command_bash(cmd, silent=False):
    if silent:
        subprocess.run(cmd, shell=True, executable="/bin/zsh", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(cmd, shell=True, executable="/bin/zsh")


def module_exists(module_name):
    """
    Check if module exists.

    Parameters
    ----------
    module_name : str
        Module name to check for.

    Returns
    -------
    bool
        True or False depending on whether module is installed,

    """
    return module_name in (name for loader, name, ispkg in iter_modules())


def which(program):
    """
    Returns the path for an arbitrary executable shell program defined in the PAHT environment variable.

    Parameters
    ----------
    program : str
        Shell binary name

    Returns
    -------

    """
    import os

    # Check if path contains file and is executable
    def is_exe(f_path):
        return os.path.isfile(f_path) and os.access(f_path, os.X_OK)

    # Get path and name
    fpath, fname = os.path.split(program)

    if fpath:
        # If a path is given, and the file is executable, we just return the path
        if is_exe(program):
            return program

    # If no path is given (as usual) we loop through $PATH
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')

            # Create executable names at current path
            exe_file = os.path.join(path, program)

            # Test is we have a match and return if so
            if is_exe(exe_file):
                return exe_file

    # If we don't find anything, we return None
    return None


def read_setup(path_yaml: str):

    # Read YAML
    with open(path_yaml, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def remove_file(path):
    try:
        os.remove(path)
    except OSError:
        pass


def copy_file(a, b):
    shutil.copy2(a, b)


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


def get_resource_path(package, resource):
    """
    Returns the path to an included resource.

    Parameters
    ----------
    package : str
        package name (e.g. vircampype.resources.sextractor).
    resource : str
        Name of the resource (e.g. default.conv)

    Returns
    -------
    str
        Path to resource.

    """

    # Import package
    importlib.import_module(name=package)

    # Return path to resource
    return os.path.join(os.path.dirname(sys.modules[package].__file__), resource)


def notify(title="", subtitle="", message=""):
    """ macOS notification wrapper """
    t = "-title {!r}".format(title)
    s = "-subtitle {!r}".format(subtitle)
    m = "-message {!r}".format(message)
    os.system("terminal-notifier {}".format(" ".join([m, t, s])))
