from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt

from ..compute import ifr_peaks, fit_ifr_gmm
from ..models import IFRPeaks, GMMFit, CofiringHeatmap
from ..prep import RestingActivityDataset, PrepConfig
from ..helper_functions import calculate_ifr


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
    # Visualization defaults
    hist_bins: int = 100
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
    def plot_histogram(self, show: bool = False, ax=None):
        """Plot pooled IFR histogram with optional KDE/peaks and GMM overlay."""
        peaks = self.peaks()
        vals = peaks.values
        x = peaks.kde_x
        y = peaks.kde_y
        fig, ax = (plt.gcf(), ax) if ax is not None else plt.subplots(figsize=(10, 6))
        ax.hist(vals, bins=self.cfg.hist_bins, density=True, alpha=0.3, color="0.7", label="Data Histogram")
        if self.cfg.show_kde:
            ax.plot(x, y, color="k", lw=2, label="KDE")
        if self.cfg.show_peaks and len(peaks.peaks_x) > 0:
            ax.scatter(peaks.peaks_x, peaks.peaks_y, color="r", zorder=3, label="Peaks")

        if self.cfg.overlay_gmm:
            fit = self.fit_gmm(vals)
            # Means for plotting in the same domain as vals
            if self.cfg.log_scale:
                means = np.log10(fit.means_hz)
            else:
                means = fit.means_hz
            std = fit.std
            weights = fit.weights
            from scipy.stats import norm
            sum_pdf = np.zeros_like(x)
            colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(means))))
            for k, (mu, s, w) in enumerate(zip(means, std, weights)):
                if not np.isfinite(s) or s <= 0:
                    continue
                comp_pdf = w * norm.pdf(x, loc=mu, scale=s)
                sum_pdf += comp_pdf
                label_hz = 10 ** mu if self.cfg.log_scale else mu
                ax.plot(x, comp_pdf, lw=2, color=colors[k % len(colors)], label=f"{label_hz:.2f} Hz")
                ymax = w * norm.pdf(mu, loc=mu, scale=s)
                ax.annotate(f"{label_hz:.2f} Hz", (mu, ymax), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
            ax.plot(x, sum_pdf, "r--", lw=2, label="Sum of Gaussians")

        ax.set_xlabel("Log(IFR)" if self.cfg.log_scale else "IFR (Hz)")
        ax.set_ylabel("Density")
        ax.set_title("Gaussian Mixture Model Fit to IFR Data")
        ax.legend()
        if show:
            plt.show()
        return fig, ax

    # Time-series heatmap per recording (legacy plot_ifr integrated here)
    def plot_timeseries(self, selected_electrodes_per_recording=None):
        """Plot IFR heatmap + histogram for each recording separately.

        - selected_electrodes_per_recording: list of electrode id iterables; if a flat iterable
          is provided and there is a single recording, it is applied to that recording.
        Uses visualization parameters from IFRConfig.
        Returns a list of (fig, (ax_heatmap, ax_hist)) per recording plotted.
        """
        # Resolve default selection if not provided
        if selected_electrodes_per_recording is None:
            if self._refs_per_recording is not None:
                selected_electrodes_per_recording = self._refs_per_recording
            elif self._ds is not None:
                # Default to 'top' selection if dataset available
                default_sel = PrepConfig(mode="top", top_start=10, top_stop=110, top_use_recording_window=True, verbose=False)
                selected_electrodes_per_recording = self._ds.select_ref_electrodes(default_sel)
            else:
                # Fallback: use all electrodes present per recording
                selected_electrodes_per_recording = [
                    np.unique(np.asarray(sd.get("electrode", []), dtype=int)) for sd in self.spikes_list
                ]
        # Normalize selection input
        if hasattr(selected_electrodes_per_recording, "__iter__") and len(self.spikes_list) == 1:
            # If a flat list was passed for a single recording, wrap it
            if not hasattr(selected_electrodes_per_recording[0], "__iter__") or isinstance(
                selected_electrodes_per_recording, (list, tuple)
            ) and selected_electrodes_per_recording and not hasattr(selected_electrodes_per_recording[0], "__len__"):
                selected_electrodes_per_recording = [selected_electrodes_per_recording]

        results = []
        for i, (spikes_data, st, et) in enumerate(zip(self.spikes_list, self.start_times, self.end_times)):
            sel = selected_electrodes_per_recording[i] if len(self.spikes_list) > 1 else selected_electrodes_per_recording[0]
            # Calculate IFR per electrode
            ifr_data, _, all_ifr_values = calculate_ifr(spikes_data, sel, st, et)
            # Time grid: decouple from sf to avoid huge arrays; cap by max_time_points
            duration = max(0.0, float(et) - float(st))
            target_hz = float(self.cfg.time_grid_hz) if self.cfg.time_grid_hz and self.cfg.time_grid_hz > 0 else 100.0
            n = int(max(1, min(duration * target_hz, float(self.cfg.max_time_points))))
            time_points = np.linspace(float(st), float(et), n, dtype=np.float32)
            valid_electrodes = []
            # Preallocate heatmap with compact dtype
            heatmap = np.empty((len(sel), n), dtype=np.float32)
            row_idx = 0
            for el in sel:
                if el in ifr_data:
                    times, vals = ifr_data[el]
                    if self.cfg.log_scale:
                        vals = np.where(vals == 0, 1e-3, vals)
                        vals = np.log10(vals)
                        vals = np.where(np.isinf(vals), -3, vals)
                    heatmap[row_idx, :] = np.interp(time_points, times.astype(np.float32), vals).astype(np.float32)
                    valid_electrodes.append(el)
                    row_idx += 1
            if row_idx == 0:
                continue
            H = heatmap[:row_idx, :]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            im = ax1.imshow(
                H,
                aspect="auto",
                origin="lower",
                extent=[float(st), float(et), -0.5, len(valid_electrodes) - 0.5],
                cmap="viridis",
                interpolation="nearest",
            )
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Channel by Firing Frequency Rank")
            if self.cfg.log_scale:
                ax1.set_title(f"Log Instantaneous Firing Rate Across Top {len(valid_electrodes)} electrodes")
                cbar_label = "Log Instantaneous Firing Rate (Hz)"
            else:
                ax1.set_title(f"Instantaneous Firing Rate Across Top {len(valid_electrodes)} electrodes")
                cbar_label = "Instantaneous Firing Rate (Hz)"
            ax1.set_yticks([0, len(valid_electrodes) - 1])
            ax1.set_yticklabels([1, len(valid_electrodes)])
            cbar = fig.colorbar(im, ax=ax1)
            cbar.set_label(cbar_label)

            # Histogram
            hist_vals = all_ifr_values.copy()
            if self.cfg.log_scale:
                hist_vals = hist_vals[hist_vals > 1e-3]
                hist_vals = np.log10(hist_vals)
            ax2.hist(hist_vals, bins=self.cfg.ts_bins, color="blue", edgecolor="black")
            ax2.set_xlabel("Instantaneous Firing Rate (Hz)" if not self.cfg.log_scale else "Log Instantaneous Firing Rate (Hz)")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Histogram of Instantaneous Firing Rates")
            plt.tight_layout()
            results.append((fig, (ax1, ax2)))
        return results


# Co-firing heatmap plot (moved from viz.py for consolidation)
def plot_cofiring_heatmap(
    heatmap: CofiringHeatmap,
    normalize: bool = False,
    cmap_name: str = "magma",
    show: bool = False,
    ax=None,
):
    """Plot co-firing heatmap Z(distance, delay) with optional t0 normalization."""
    Z = heatmap.Z.copy()
    delays = heatmap.delays
    if normalize:
        if np.any(np.isclose(delays, 0)):
            t0_idx = int(np.argmin(np.abs(delays[:-1] - 0)))
            base = Z[t0_idx, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                Z = np.divide(Z, base, out=np.zeros_like(Z), where=base != 0)

    fig, ax = (plt.gcf(), ax) if ax is not None else plt.subplots(figsize=(10, 6))
    extent = [float(heatmap.distance_bins.min()), float(heatmap.distance_bins.max()), float(delays.min()), float(delays.max())]
    cax = ax.imshow(Z, aspect="auto", cmap=plt.get_cmap(cmap_name), extent=extent, origin="lower")
    ax.set_xlabel("Distance from Electrode ($\\mu m$)")
    ax.set_ylabel("Delay (ms)")
    ax.set_title(f"{'Normalized ' if normalize else ''} p(Co-Firing) vs Distance and Time")
    ax.set_facecolor("black")
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Probability")
    if show:
        plt.show()
    return fig, ax


# Convenience module-level function to plot IFR time series per recording
def plot_ifr_timeseries(
    spikes_data_list,
    start_times,
    end_times,
    selected_electrodes_per_recording,
    config: IFRConfig | None = None,
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
    if selected_electrodes_per_recording and (
        not isinstance(selected_electrodes_per_recording[0], (list, tuple, np.ndarray))
    ):
        selected_electrodes_per_recording = [selected_electrodes_per_recording]

    analyzer = IFRAnalyzer(spikes_data_list, start_times, end_times, config=config)
    return analyzer.plot_timeseries(selected_electrodes_per_recording)
