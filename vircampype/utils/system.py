# =========================================================================== #
import subprocess
from pkgutil import iter_modules
from itertools import zip_longest


# Define objects in this module
__all__ = ["run_cmds", "run_command_bash", "module_exists", "which"]


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