"""Functional electrophysiology analysis helpers."""
import os as _os  # set MKL/OpenMP env vars before importing any deps that may load MKL
_os.environ.setdefault("KMP_WARNINGS", "0")
_os.environ.setdefault("MKL_VERBOSE", "0")
_os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")
_os.environ.setdefault("MPLCONFIGDIR", _os.path.join(_os.getcwd(), ".mpl-cache"))

from . import artifacts, data_io, metrics, modeling, plotting, preprocessing  # re-export namespaces
from .preprocessing.dataset import PrepConfig, Recording, RestingActivityDataset

__all__ = [
    "artifacts",
    "data_io",
    "metrics",
    "modeling",
    "plotting",
    "preprocessing",
    "PrepConfig",
    "Recording",
    "RestingActivityDataset",
]
