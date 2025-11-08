

from ._version import __version__
from . import readligo  # re-export the submodule so tests can do: from ligotools import readligo

__all__ = ["readligo", "__version__"]
