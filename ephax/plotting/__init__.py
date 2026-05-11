"""Plotting helpers that are separate from metric computation."""

from .burst import (
    plot_aligned_electrode_heatmap,
    plot_gamma_population_windows,
    plot_high_activity_burst_windows,
    plot_macro_burst_detector_comparison_windows,
    plot_population_ifr_summary,
    save_average_hex_gif,
)
from .cofiring import plot_cofiring_heatmap
from .firing_distance import plot_binned_distance_series
from .ifr import (
    plot_ifr_histogram,
    plot_ifr_timeseries,
    plot_ifr_timeseries_panel,
    plot_ifr_timeseries_panels,
)
from .style import truncate_colormap

__all__ = [
    "plot_cofiring_heatmap",
    "plot_binned_distance_series",
    "plot_aligned_electrode_heatmap",
    "plot_high_activity_burst_windows",
    "plot_macro_burst_detector_comparison_windows",
    "plot_ifr_histogram",
    "plot_ifr_timeseries",
    "plot_ifr_timeseries_panel",
    "plot_ifr_timeseries_panels",
    "plot_gamma_population_windows",
    "plot_population_ifr_summary",
    "save_average_hex_gif",
    "truncate_colormap",
]
