import glob
import importlib
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from itertools import zip_longest
from pathlib import Path
from typing import List

import yaml

__all__ = [
    "make_folder",
    "which",
    "read_yml",
    "yml2config",
    "run_commands_shell_parallel",
    "run_command_shell",
    "get_resource_path",
    "rsync_file",
    "copy_file",
    "remove_file",
    "remove_directory",
    "clean_directory",
    "notify",
    "make_symlinks",
    "make_executable",
    "cmd_prepend_libraries",
    "remove_ansi_codes",
    "wait_for_no_process",
    "make_path_system_tempfile",
    "make_system_tempdir",
]


def make_folder(path: str) -> None:
    """
    Creates a folder at the specified path.

    Parameters
    ----------
    path : str
        Path where the new folder should be created.

    Returns
    -------
    None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def which(program: str) -> str:
    """
    Returns the path for an arbitrary executable shell program defined in the PATH
    environment variable.

    Parameters
    ----------
    program : str
        Shell binary name.

    Returns
    -------
    str
        The full path to the executable shell program. If the program is not found,
        returns an informative message.
    """

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
    return f"No executable found for specified program {program}."


def read_yml(path_yml: str) -> dict:
    """
    Reads a YAML file and returns its content.

    Parameters
    ----------
    path_yml : str
        Path to the YAML file to read.

    Returns
    -------
    dict
        The data from the YAML file. Could be a list, dict, etc.
        Depending on the YAML contents. If an error occurs during reading,
        None is returned.
    """
    # Read YAML
    with open(path_yml, "r") as stream:
        try:
            return yaml.full_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise ValueError(f"Could not read YAML file at {path_yml}.")


def yml2config(path_yml: str, skip=None, **kwargs) -> str:
    """
    Reads a YML file at a given path and converts the entries to a string that can be
    passed to astromatic tools.

    Parameters
    ----------
    path_yml : str
        Path to YML file.
    skip : list, optional
        If set, ignore the given keywords in the list
    kwargs
        Any available setup parameter can be overwritten
        (e.g. catalog_name="catalog.fits")

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
            s += f"-{key.upper()} {kwargs[key]} "
        else:
            s += f"-{key.upper()} {val} "

    return s


def cmd_prepend_libraries(cmd: str) -> str:
    """
    Prepends libraries to the command based on the system environment.

    Parameters
    ----------
    cmd : str
        The command to which libraries are to be prepended.

    Returns
    -------
    str
        The modified command with libraries prepended.
        If system is MacOS (darwin) and LD_LIBRARY_PATH or DYLD_LIBRARY_PATH
        are in the system environment, these are prepended to the command.
        If DYLD_LIBRARY_PATH is not in the system environment, a fallback is used.

    Notes
    -----
    This function serves as a workaround for a known issue on macOS systems
    where dynamic libraries need to be manually appended.
    Ref: https://stackoverflow.com/questions/48657710/
    """

    # Get system environment
    sys_env = os.environ.copy()

    # If in a Mac, we need to append the dynamic libraries manually
    if sys.platform == "darwin":
        if "LD_LIBRARY_PATH" in sys_env:
            cmd = f"export LD_LIBRARY_PATH={sys_env['LD_LIBRARY_PATH']} && {cmd}"
        if "DYLD_LIBRARY_PATH" in sys_env:
            cmd = f"export DYLD_LIBRARY_PATH={sys_env['DYLD_LIBRARY_PATH']} && {cmd}"
        # This is a silly workaround because DYLD_LIBRARY_PATH is for some reason
        # not in sys_env even though it should be defined.
        else:
            dyld_library_path = (
                "/opt/intel/oneapi/compiler/latest/mac/compiler/"
                "lib:/opt/intel/oneapi/mkl/latest/lib:"
            )
            cmd = f"export DYLD_LIBRARY_PATH={dyld_library_path} && {cmd}"

    return cmd


def run_commands_shell_parallel(
    cmds, n_jobs: int = 1, shell: str = "zsh", silent: bool = True
):
    """
    Runs a list of shell commands in parallel.

    Parameters
    ----------
    cmds : sized
        List of shell commands
    n_jobs : int, optional
        Number of parallel jobs.
    shell : str
        Shell name. Default is 'zsh'.
    silent : bool, optional
        Whether or not to print information about the process. Default is True.

    """

    # Append dynamic libraries
    cmds = [cmd_prepend_libraries(cmd) for cmd in cmds]

    if silent:
        groups = [
            (
                subprocess.Popen(
                    cmd,
                    shell=True,
                    executable=which(shell),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                for cmd in cmds
            )
        ] * n_jobs
    else:
        groups = [
            (subprocess.Popen(cmd, shell=True, executable=which(shell)) for cmd in cmds)
        ] * n_jobs

    # Run processes
    for processes in zip_longest(*groups):  # run len(processes) == limit at a time
        for p in filter(None, processes):
            p.wait()


def run_command_shell(
    cmd: str, shell: str = "zsh", silent: bool = False
) -> tuple[str, str]:
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

    Returns
    ----------
    tuple
        The stdout and stderr of the command.
    """

    # Append dynamic libraries
    cmd = cmd_prepend_libraries(cmd)

    # Run
    result = subprocess.run(
        cmd,
        shell=True,
        executable=which(shell),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Decode the command output
    try:
        stdout = result.stdout.decode("utf-8").strip()
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
    except UnicodeDecodeError:  # if utf-8 fails, try latin-1
        stdout = result.stdout.decode("latin-1", errors="replace").strip()
        stderr = result.stderr.decode("latin-1", errors="replace").strip()

    # If not in silent mode, print the output to terminal
    if not silent:
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)

    # Return command outputs (stdout and stderr)
    return remove_ansi_codes(stdout), remove_ansi_codes(stderr)


def get_resource_path(package: str, resource: str) -> str:
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


def rsync_file(src: str, dst: str) -> None:
    """
    Copy ``src`` to ``dst`` via ``rsync``, skipping the transfer if contents match.

    Uses ``rsync -a --checksum`` to preserve metadata and compare files by checksum
    (slower than size/mtime checks, but more accurate). Creates the destination
    parent directory if needed.

    Parameters
    ----------
    src : str
        Source file path.
    dst : str
        Destination file path.

    Returns
    -------
    None

    Raises
    ------
    subprocess.CalledProcessError
        If ``rsync`` exits with a non-zero status.
    FileNotFoundError
        If ``rsync`` is not found on ``PATH``.
    """
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rsync",
        "-a",
        "--checksum",
        "--",
        str(src_p),
        str(dst_p),
    ]
    subprocess.run(cmd, check=True)


def copy_file(a: str, b: str) -> None:
    """
    Function that copies a file from a source path to a destination.

    Parameters
    ----------
    a : str
        Source file path.
    b : str
        Destination file path.

    Returns
    -------
    None
    """
    try:
        shutil.copy2(a, b)
    except OSError:
        shutil.copy(a, b)


def remove_file(filepath: str) -> None:
    """
    Function to remove a file. If the file is not found, the function does nothing.

    Parameters
    ----------
    filepath : str
        The path to the file to remove.

    Returns
    -------
    None
    """
    try:
        os.remove(filepath)
    except OSError:
        pass


def remove_directory(path_folder: str) -> None:
    """
    Function to remove a directory.
    If the directory is not found, the function does nothing.

    Parameters
    ----------
    path_folder : str
        The path to the folder to remove.

    Returns
    -------
    None
    """
    try:
        shutil.rmtree(path_folder)
    except FileNotFoundError:
        pass


def clean_directory(directorypath: str, pattern: str = "*") -> None:
    """
    Function to remove files in a directory, following a name pattern.

    Parameters
    ----------
    directorypath : str
        The directory that needs to be cleaned.
    pattern : str, optional
        The name pattern to match file names. Default is "*", matching all files.

    Returns
    -------
    None
    """
    if not directorypath.endswith("/"):
        directorypath = directorypath + "/"
    for f in glob.glob(directorypath + pattern):
        remove_file(f)


def notify(
    message: str,
    title: str = None,
    subtitle: str = None,
    sound: str = "default",
    open_url: str = None,
    ignore_dnd: bool = False,
) -> None:
    """
    macOS notification wrapper built around terminal-notifier.

    Parameters
    ----------
    message : str
        The message to display in the notification.
    title : str, optional
        The title of the notification. Default is None.
    subtitle : str, optional
        The subtitle of the notification. Default is None.
    sound : str, optional
        The sound to play with the notification. Default is "default".
    open_url : str, optional
        The URL to open when the notification is clicked. Default is None.
    ignore_dnd : bool, optional
        If True, the notification will bypass Do Not Disturb mode. Default is False.

    Returns
    -------
    None
    """
    me = "-message {!r}".format(message)
    ti = "-title {!r}".format(title) if title is not None else ""
    su = "-subtitle {!r}".format(subtitle) if subtitle is not None else ""
    so = "-sound {!r}".format(sound) if sound is not None else ""
    op = "-open {!r}".format(open_url) if open_url is not None else ""
    ig = "-ignoreDnD" if ignore_dnd else ""
    os.system("terminal-notifier {}".format(" ".join([me, ti, su, so, op, ig])))


def make_symlinks(paths_files: List[str], paths_links: List[str]) -> None:
    """
    Create symbolic links for all items in paths_files to corresponding paths in
    paths_links

    Parameters
    ----------
    paths_files : List[str]
        List of source paths.
    paths_links : List[str]
        List of destination paths for the symbolic links.

    Returns
    -------
    None
    """
    for pp, ll in zip(paths_files, paths_links):
        if not os.path.isfile(ll):
            os.symlink(pp, ll)


def make_executable(path: str) -> None:
    """
    Make a file executable.

    Parameters
    ----------
    path : str
        Path to the file to make executable

    Returns
    -------
    None
    """
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)


def remove_ansi_codes(s: str) -> str:
    """
    Remove ANSI escape codes from the string s.

    Parameters
    ----------
    s : str
        The string to remove escape codes from.

    Returns
    -------
    str
        The string s with all ANSI escape codes removed.

    """
    # ANSI escape codes start with the sequence ESC[,
    # followed by a semicolon-separated series of numbers, and end with an 'm'
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", s)


def _any_process_running(executable: str) -> bool:
    """
    Check whether a process with the given executable name is currently running.

    Parameters
    ----------
    executable : str
        Executable name to match exactly (as in `pgrep -x`, e.g. `"scamp"`).

    Returns
    -------
    bool
        `True` if at least one matching process exists, otherwise `False`.
    """
    cp = subprocess.run(
        ["pgrep", "-x", executable],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return cp.returncode == 0


def wait_for_no_process(
    executable: str,
    poll_s: float = 2.0,
    timeout_s: float | None = None,
):
    """
    Wait until no process with the given executable name is running.

    Parameters
    ----------
    executable : str
        Executable name to match exactly (as in `pgrep -x`, e.g. `"scamp"`).
    poll_s : float, optional
        Seconds between checks. Default is 2.0.
    timeout_s : float | None, optional
        Max seconds to wait; `None` waits indefinitely.

    Raises
    ------
    TimeoutError
        If the timeout is exceeded.
    """
    t0 = time.time()
    while _any_process_running(executable):
        if timeout_s is not None and (time.time() - t0) > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for other {executable} processes to finish."
            )
        time.sleep(poll_s)


def make_path_system_tempfile(prefix: str = "", suffix: str = ".tmp") -> str:
    """
    Return a unique temp file path in the system temp directory.

    Parameters
    ----------
    prefix : str, optional
        Prefix string prepended to the filename, by default "".
    suffix : str, optional
        File suffix (including leading dot), by default ".tmp".

    Returns
    -------
    str
        Temp file path (file is not created).
    """
    return os.path.join(tempfile.gettempdir(), f"{prefix}{uuid.uuid4().hex}{suffix}")


def make_system_tempdir() -> str:
    """
    Create a unique temporary directory in the system temp directory.

    Returns
    -------
    str
        Path to the created temporary directory.
    """
    path = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)
    os.makedirs(path)
    return path + os.sep
