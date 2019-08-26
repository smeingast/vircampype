# =========================================================================== #
import subprocess


def run_cmds(cmds, silent=True):
    """
    Runs a list of shell commands

    Parameters
    ----------
    cmds : list
        List of shell commands
    silent : bool, optional
        Whether or not to print information about the process. Default is True.

    Returns
    -------

    """

    for cmd in cmds:

        if silent:

            # Run
            p = subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            # Check for error
            for line in p.stdout.readlines():
                if b"error" in line.lower():
                    print(line)
                    raise Exception("Error detected")

            p.stdout.close()

        else:

            # Run
            p = subprocess.Popen(cmd.split(" "))

        # Wait for completion
        p.wait()

    # TODO: Return something
