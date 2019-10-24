# =========================================================================== #
import os
import subprocess
from itertools import zip_longest


# Define objects in this module
__all__ = ["run_cmds", "run_command_bash"]


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
        groups = [(subprocess.Popen(cmd.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                   for cmd in cmds)] * n_processes
    else:
        groups = [(subprocess.Popen(cmd.split(" ")) for cmd in cmds)] * n_processes

    # Run processes
    for processes in zip_longest(*groups):  # run len(processes) == limit at a time
        for p in filter(None, processes):
            p.wait()


# TODO: Rename to run_command_shell?
def run_command_bash(cmd, silent=False):
    if silent:
        subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(cmd, shell=True, executable="/bin/bash")
