# =========================================================================== #
import subprocess
from itertools import zip_longest


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
        groups = [(subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                   for cmd in cmds)] * n_processes
    else:
        groups = [(subprocess.Popen(cmd.split(" ")) for cmd in cmds)] * n_processes

    # Run processes
    for processes in zip_longest(*groups):  # run len(processes) == limit at a time
        for p in filter(None, processes):
            p.wait()


def run_command_bash(cmd, silent=False):
    if silent:
        subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, shell=True, executable="/bin/bash")
