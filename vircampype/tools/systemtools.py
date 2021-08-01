import os
import sys
import yaml
import glob
import shutil
import subprocess
import importlib

from itertools import zip_longest

__all__ = ["make_folder", "which", "read_yml", "yml2config", "run_commands_shell_parallel", "run_command_shell",
           "get_resource_path", "copy_file", "remove_file", "remove_directory", "clean_directory", "notify"]


def make_folder(path: str):
    """ Creates folder at specified path. """
    if not os.path.exists(path):
        os.makedirs(path)


def which(program: str):
    """
    Returns the path for an arbitrary executable shell program defined in the PAHT environment variable.

    Parameters
    ----------
    program : str
        Shell binary name

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


def read_yml(path_yml: str):

    # Read YAML
    with open(path_yml, "r") as stream:
        try:
            return yaml.full_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def yml2config(path_yml: str, skip=None, **kwargs):
    """
    Reads a YML file at a given path and converts the entries to a string that can be passed to astromatic tools.

    Parameters
    ----------
    path_yml : str
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

    # Read file
    setup = read_yml(path_yml=path_yml)

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


def run_commands_shell_parallel(cmds, n_jobs: int = 1, shell: str = "zsh", silent: bool = True):
    """
    Runs a list of shell commands in parallel.

    Parameters
    ----------
    cmds : list
        List of shell commands
    n_jobs : int, optional
        Number of parallel jobs.
    shell : str
        Shell name. Default is 'zsh'.
    silent : bool, optional
        Whether or not to print information about the process. Default is True.

    """

    if silent:
        groups = [(subprocess.Popen(cmd, shell=True, executable=which(shell), stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL) for cmd in cmds)] * n_jobs
    else:
        groups = [(subprocess.Popen(cmd, shell=True, executable=which(shell)) for cmd in cmds)] * n_jobs

    # Run processes
    for processes in zip_longest(*groups):  # run len(processes) == limit at a time
        for p in filter(None, processes):
            p.wait()


def run_command_shell(cmd, shell: str = "zsh", silent: bool = False):
    """
    Runs a single shell command in the specified shell.

    Parameters
    ----------
    cmd : str
        Command to run.
    shell : str
        Shell executable name. Default is 'zsh'.
    silent : bool
        Whether to run silently.

    """
    if silent:
        subprocess.run(cmd, shell=True, executable=which(shell),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(cmd, shell=True, executable=which(shell))


def get_resource_path(package: str, resource: str):
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


def copy_file(a, b):
    shutil.copy2(a, b)


def remove_file(filepath: str):
    try:
        os.remove(filepath)
    except OSError:
        pass


def remove_directory(path_folder: str):
    try:
        shutil.rmtree(path_folder)
    except FileNotFoundError:
        pass


def clean_directory(directorypath: str, pattern: str = "*"):
    """ Function to remove files in a directory, following a name pattern. """
    if not directorypath.endswith("/"):
        directorypath = directorypath + "/"
    for f in glob.glob(directorypath + pattern):
        remove_file(f)


def notify(message, title=None, subtitle=None, sound="default", open_url=None, ignore_dnd=False):
    """ macOS notification wrapper built around terminal-notifier """
    me = "-message {!r}".format(message)
    ti = "-title {!r}".format(title) if title is not None else ""
    su = "-subtitle {!r}".format(subtitle) if subtitle is not None else ""
    so = "-sound {!r}".format(sound) if sound is not None else ""
    op = "-open {!r}".format(open_url) if open_url is not None else ""
    ig = "-ignoreDnD" if ignore_dnd else ""
    print("terminal-notifier {}".format(" ".join([me, ti, su, so, op, ig])))
    os.system("terminal-notifier {}".format(" ".join([me, ti, su, so, op, ig])))
