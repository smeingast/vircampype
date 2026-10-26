import tomllib
from importlib.metadata import version
from pathlib import Path

# A source checkout (worktree on PYTHONPATH) reports its own pyproject version, not a
# possibly stale installed one; installed packages use the package metadata.
_pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
if _pyproject.is_file():
    __version__ = tomllib.loads(_pyproject.read_text())["project"]["version"]
else:
    __version__ = version("vircampype")
