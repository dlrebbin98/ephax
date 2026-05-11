from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="Intel MKL WARNING")
warnings.filterwarnings("ignore", message="RuntimeWarning: overflow encountered")

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from ..metrics.ifr import ifr_peaks, prepare_ifr_timeseries_panels
from ..modeling.gmm import fit_ifr_gmm
from ..models import IFRPeaks, GMMFit
from ..plotting.cofiring import plot_cofiring_heatmap
from ..plotting.ifr import (
    plot_ifr_histogram,
    plot_ifr_timeseries as _plot_ifr_timeseries,
    plot_ifr_timeseries_panels,
)
from ..prep import RestingActivityDataset, PrepConfig


@dataclass
class IFRConfig:
    """Configuration for IFR analyses and plots.

    - log_scale: use log10 domain for values and plots.
    - n_components: optional fixed number of GMM components; if None, BIC search is used.
    - hist_bins: bins for aggregate IFR histogram.
    - overlay_gmm: draw GMM component curves and their sum on the histogram.
    - show_kde/show_peaks: optional KDE and peak markers on the histogram.
    - ts_bins: bins for the per-recording IFR histogram (bottom panel).
    - time_grid_hz: target sampling rate for IFR heatmaps (decoupled from sf).
    - max_time_points: cap on columns in IFR heatmaps to bound memory.
    """
    log_scale: bool = True
    n_components: Optional[int] = None
    random_state: int = 0
    overlay_gmm: bool = True
    show_kde: bool = False
    show_peaks: bool = False
    ts_bins: int = 50                    # bins for per-recording hist in timeseries view
    time_grid_hz: float = 100.0          # resampling rate for IFR heatmap
    max_time_points: int = 5000          # cap to guard memory

"""IFR plotting and analysis utilities."""

class IFRAnalyzer:
    """High-level interface for instantaneous firing rate (IFR) analyses.

    Wraps compute functions, manages per-recording windows and selections, and
    provides plotting helpers (aggregate histogram with GMM overlay and
    per-recording IFR heatmaps).
    """

    def __init__(
        self,
        spikes_list: Iterable[dict],
        start_times: Iterable[float],
        end_times: Iterable[float],
        config: IFRConfig | None = None,
    ) -> None:
        self.spikes_list = list(spikes_list)
        self.start_times = list(start_times)
        self.end_times = list(end_times)
        self.cfg = config or IFRConfig()
        self._peaks: IFRPeaks | None = None
        # Optional dataset context for simpler APIs
        self._ds: RestingActivityDataset | None = None
        self._refs_per_recording = None

    @classmethod
    def from_dataset(
        cls,
        dataset: RestingActivityDataset,
        config: IFRConfig | None = None,
        selection_prep_config: PrepConfig | None = None,
    ) -> "IFRAnalyzer":
        """Construct from a dataset, optionally storing refs via PrepConfig."""
        spikes_list, _layouts, start_times, end_times = dataset.to_legacy()
        inst = cls(spikes_list, start_times, end_times, config=config)
        inst._ds = dataset
        if selection_prep_config is not None:
            inst._refs_per_recording = dataset.select_ref_electrodes(selection_prep_config)
        return inst

    # Compute
    def peaks(self) -> IFRPeaks:
        """Compute pooled IFR values across recordings and return peak metadata."""
        if self._peaks is None:
            self._peaks = ifr_peaks(
                self.spikes_list,
                self.start_times,
                self.end_times,
                log_scale=self.cfg.log_scale,
                selected_refs_per_recording=self._refs_per_recording,
            )
        return self._peaks

    def fit_gmm(self, values: Optional[np.ndarray] = None) -> GMMFit:
        """Fit a Gaussian mixture model to IFR values (log or linear domain)."""
        vals = values if values is not None else self.peaks().values
        return fit_ifr_gmm(vals, log_scale=self.cfg.log_scale, n_components=self.cfg.n_components)

    # Viz
    def plot_histogram(self, hist_bins: int = 100, show: bool = False, ax=None):
        """Plot pooled IFR histogram with optional KDE/peaks and GMM overlay."""
        peaks = self.peaks()
        fit = self.fit_gmm(peaks.values) if self.cfg.overlay_gmm else None
        return plot_ifr_histogram(peaks, self.cfg, fit=fit, hist_bins=hist_bins, show=show, ax=ax)

    # Time-series heatmap per recording (legacy plot_ifr integrated here)
    def plot_timeseries(
        self,
        selected_electrodes_per_recording=None,
        title: Optional[str] = None,
        recording_titles: Optional[Sequence[str] | Callable[[int], str]] = None,
    ):
        """Plot IFR heatmap + histogram for each recording separately.

        - selected_electrodes_per_recording: list of electrode id iterables; if a flat iterable
          is provided and there is a single recording, it is applied to that recording.
        - title: optional prefix added to each recording heatmap title.
        - recording_titles: optional per-recording labels (sequence or callable).
        Uses visualization parameters from IFRConfig.
        Returns a list of (fig, (ax_heatmap, ax_hist)) per recording plotted.
        """
        panels = self.timeseries_panels(selected_electrodes_per_recording)
        return plot_ifr_timeseries_panels(
            panels,
            self.cfg,
            title=title,
            recording_titles=recording_titles,
        )

    def timeseries_panels(self, selected_electrodes_per_recording=None):
        """Prepare plot-ready IFR time-series panels without creating figures."""
        selections = self._resolve_selected_electrodes(selected_electrodes_per_recording)
        return prepare_ifr_timeseries_panels(
            self.spikes_list,
            self.start_times,
            self.end_times,
            selections,
            log_scale=self.cfg.log_scale,
            time_grid_hz=self.cfg.time_grid_hz,
            max_time_points=self.cfg.max_time_points,
        )

    def _resolve_selected_electrodes(self, selected_electrodes_per_recording=None):
        if selected_electrodes_per_recording is None:
            if self._refs_per_recording is not None:
                selected_electrodes_per_recording = self._refs_per_recording
            elif self._ds is not None:
                default_sel = PrepConfig(mode="top", top_start=10, top_stop=110, top_use_recording_window=True, verbose=False)
                selected_electrodes_per_recording = self._ds.select_ref_electrodes(default_sel)
            else:
                selected_electrodes_per_recording = [
                    np.unique(np.asarray(sd.get("electrode", []), dtype=int)) for sd in self.spikes_list
                ]
        return _normalize_selected_electrodes(selected_electrodes_per_recording, len(self.spikes_list))


# Convenience module-level function to plot IFR time series per recording
def plot_ifr_timeseries(
    spikes_data_list,
    start_times,
    end_times,
    selected_electrodes_per_recording,
    config: IFRConfig | None = None,
    title: Optional[str] = None,
    recording_titles: Optional[Sequence[str] | Callable[[int], str]] = None,
):
    """Plot IFR time series per recording using analyzer API.

    - spikes_data_list: list of spikes dicts (or a single dict)
    - start_times, end_times: per-recording window lists (or scalars for single recording)
    - selected_electrodes_per_recording: list of lists of electrode ids per recording (or a list for single recording)
    - sf: sampling frequency (Hz) to set time grid resolution
    - log_scale: whether to plot log10(IFR)
    - bins: histogram bins
    Returns a list of (fig, (ax1, ax2)) tuples, one per recording.
    """
    # Normalize inputs to lists
    if not isinstance(spikes_data_list, (list, tuple)):
        spikes_data_list = [spikes_data_list]
    if np.isscalar(start_times):
        start_times = [float(start_times)] * len(spikes_data_list)
    if np.isscalar(end_times):
        end_times = [float(end_times)] * len(spikes_data_list)
    selected_electrodes_per_recording = _normalize_selected_electrodes(
        selected_electrodes_per_recording,
        len(spikes_data_list),
    )

    analyzer = IFRAnalyzer(spikes_data_list, start_times, end_times, config=config)
    return _plot_ifr_timeseries(
        analyzer.spikes_list,
        analyzer.start_times,
        analyzer.end_times,
        selected_electrodes_per_recording,
        analyzer.cfg,
        title=title,
        recording_titles=recording_titles,
    )


def _normalize_selected_electrodes(selected_electrodes_per_recording, n_recordings: int):
    selected = list(selected_electrodes_per_recording)
    if n_recordings == 1:
        if len(selected) == 0:
            return [[]]
        first = selected[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            return [list(first)]
        return [selected]

    if len(selected) != n_recordings:
        raise ValueError(
            "selected_electrodes_per_recording must contain one electrode list per recording "
            f"({n_recordings} expected, got {len(selected)})"
        )
    if selected and not isinstance(selected[0], (list, tuple, np.ndarray)):
        raise ValueError("multiple recordings require nested electrode selections")
    return [list(electrodes) for electrodes in selected]
