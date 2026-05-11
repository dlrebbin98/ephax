"""Compatibility wrappers for legacy helper imports.

New code should import from focused modules:
- ephax.data_io
- ephax.preprocessing
- ephax.metrics
- ephax.modeling
- ephax.plotting
"""

from __future__ import annotations

from .data_io import (
    load_raw,
    load_spikes,
    load_spikes_data,
    load_spikes_npz,
    load_spikes_raw,
)
from .metrics.ifr import calculate_ifr, prepare_ifr_timeseries_panel, prepare_ifr_timeseries_panels
from .modeling.likelihood import likelihood_ratio_test, log_likelihood
from .plotting.style import truncate_colormap
from .preprocessing.geometry import assign_r_distance, assign_r_distance_all, assign_r_theta_distance
from .preprocessing.selection import get_activity_sorted_electrodes

__all__ = [
    "assign_r_distance",
    "assign_r_distance_all",
    "assign_r_theta_distance",
    "calculate_ifr",
    "get_activity_sorted_electrodes",
    "likelihood_ratio_test",
    "load_raw",
    "load_spikes",
    "load_spikes_data",
    "load_spikes_npz",
    "load_spikes_raw",
    "log_likelihood",
    "prepare_ifr_timeseries_panel",
    "prepare_ifr_timeseries_panels",
    "truncate_colormap",
]
