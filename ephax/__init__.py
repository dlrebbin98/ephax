"""Lightweight helpers for resting activity analysis.

Modules:
- compute: pure numerical routines (no plotting)
- viz: plotting functions that accept computed results
- models: small dataclasses for structured outputs
"""

from . import compute, models, helper_functions  # re-export namespaces (avoid viz to prevent cycles)
from .analyzers import IFRAnalyzer  # convenience import
from .analyzers.ifr import IFRConfig  # expose IFR config like others
from .prep import RestingActivityDataset, PrepConfig, Recording
from .analyzers.firing_distance import FiringDistanceAnalyzer
from .analyzers.layout_grid import LayoutGridPlotter
from .analyzers.cofiring_temporal import CofiringTemporalAnalyzer, CofiringTemporalConfig
from .analyzers.stability import StabilityAnalyzer, StabilityConfig
from .analyzers.dct import DCTAnalyzer
