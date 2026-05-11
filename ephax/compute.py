"""Compatibility facade for pure computation helpers.

New code should import from ``ephax.metrics`` and ``ephax.modeling`` directly.
"""

from __future__ import annotations

from .metrics.binning import bin_series
from .metrics.cofiring import aggregate_cofiring_heatmap, cofiring_proportions, cofiring_vs_distance_by_delay
from .metrics.ifr import ifr_peaks
from .modeling.gmm import fit_ifr_gmm

__all__ = [
    "aggregate_cofiring_heatmap",
    "bin_series",
    "cofiring_proportions",
    "cofiring_vs_distance_by_delay",
    "fit_ifr_gmm",
    "ifr_peaks",
]
