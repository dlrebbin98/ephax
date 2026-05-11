from __future__ import annotations

from typing import Callable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from ..metrics.ifr import prepare_ifr_timeseries_panels
from ..models import GMMFit, IFRPeaks, IFRTimeSeriesPanel


def plot_ifr_histogram(
    peaks: IFRPeaks,
    config,
    fit: GMMFit | None = None,
    hist_bins: int = 100,
    show: bool = False,
    ax=None,
):
    """Plot pooled IFR histogram with optional KDE, peaks, and GMM overlay."""
    vals = peaks.values
    x = peaks.kde_x
    y = peaks.kde_y
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=hist_bins, density=True, alpha=0.3, color="0.7", label="Data Histogram")

    if getattr(config, "show_kde", False):
        ax.plot(x, y, color="k", lw=2, label="KDE")
    if getattr(config, "show_peaks", False) and len(peaks.peaks_x) > 0:
        ax.scatter(peaks.peaks_x, peaks.peaks_y, color="r", zorder=3, label="Peaks")

    if getattr(config, "overlay_gmm", False) and fit is not None:
        if getattr(config, "log_scale", True):
            means = np.log10(fit.means_hz)
        else:
            means = fit.means_hz
        std = fit.std
        weights = fit.weights
        sum_pdf = np.zeros_like(x)
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(means))))
        for k, (mu, s, w) in enumerate(zip(means, std, weights)):
            if not np.isfinite(s) or s <= 0:
                continue
            comp_pdf = w * norm.pdf(x, loc=mu, scale=s)
            sum_pdf += comp_pdf
            label_hz = 10 ** mu if getattr(config, "log_scale", True) else mu
            ax.plot(x, comp_pdf, lw=2, color=colors[k % len(colors)], label=f"{label_hz:.2f} Hz")
            ymax = w * norm.pdf(mu, loc=mu, scale=s)
            ax.annotate(
                f"{label_hz:.2f} Hz",
                (mu, ymax),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.plot(x, sum_pdf, "r--", lw=2, label="Sum of Gaussians")

    ax.set_xlabel("Log(IFR)" if getattr(config, "log_scale", True) else "IFR (Hz)")
    ax.set_ylabel("Density")
    ax.set_title("Gaussian Mixture Model Fit to IFR Data")
    ax.legend()
    if show:
        plt.show()
    return fig, ax


def plot_ifr_timeseries(
    spikes_data_list,
    start_times,
    end_times,
    selected_electrodes_per_recording,
    config,
    title: Optional[str] = None,
    recording_titles: Optional[Sequence[str] | Callable[[int], str]] = None,
):
    """Compatibility wrapper that prepares and plots IFR panels per recording."""
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

    panels = prepare_ifr_timeseries_panels(
        spikes_data_list,
        start_times,
        end_times,
        selected_electrodes_per_recording,
        log_scale=config.log_scale,
        time_grid_hz=config.time_grid_hz,
        max_time_points=config.max_time_points,
    )
    return plot_ifr_timeseries_panels(panels, config, title=title, recording_titles=recording_titles)


def plot_ifr_timeseries_panels(
    panels: Sequence[IFRTimeSeriesPanel],
    config,
    title: Optional[str] = None,
    recording_titles: Optional[Sequence[str] | Callable[[int], str]] = None,
):
    """Render prepared IFR time-series panels."""
    results = []
    for panel in panels:
        rec_label = _recording_label(panel.recording_index, recording_titles)
        results.append(plot_ifr_timeseries_panel(panel, config, title=title, recording_label=rec_label))
    return results


def plot_ifr_timeseries_panel(
    panel: IFRTimeSeriesPanel,
    config,
    title: Optional[str] = None,
    recording_label: Optional[str] = None,
):
    """Render one prepared IFR heatmap + histogram panel."""
    fig, (ax_heatmap, ax_hist) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    im = ax_heatmap.imshow(
        panel.heatmap,
        aspect="auto",
        origin="lower",
        extent=[panel.start_time, panel.end_time, -0.5, len(panel.electrodes) - 0.5],
        cmap="viridis",
        interpolation="nearest",
    )
    ax_heatmap.set_xlabel("Time (s)")
    ax_heatmap.set_ylabel("Channel by Firing Frequency Rank")
    ax_heatmap.set_title(_timeseries_title(panel, title=title, recording_label=recording_label))
    ax_heatmap.set_yticks([0, len(panel.electrodes) - 1])
    ax_heatmap.set_yticklabels([1, len(panel.electrodes)])
    cbar = fig.colorbar(im, ax=ax_heatmap)
    cbar.set_label(_ifr_colorbar_label(panel.log_scale))

    ax_hist.hist(panel.histogram_values, bins=config.ts_bins, color="blue", edgecolor="black")
    if panel.log_scale:
        ticks = ax_hist.get_xticks()
        ax_hist.set_xticks(ticks)
        ax_hist.set_xticklabels([_log_tick_label(x) for x in ticks])
    ax_hist.set_xlabel(_ifr_axis_label(panel.log_scale))
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Histogram of Instantaneous Firing Rates")
    return fig, (ax_heatmap, ax_hist)


def _recording_label(index: int, recording_titles: Optional[Sequence[str] | Callable[[int], str]]) -> Optional[str]:
    if recording_titles is None:
        return None
    if callable(recording_titles):
        return recording_titles(index)
    if index < len(recording_titles):
        return recording_titles[index]
    return None


def _timeseries_title(panel: IFRTimeSeriesPanel, title: Optional[str] = None, recording_label: Optional[str] = None) -> str:
    prefix = "Log " if panel.log_scale else ""
    base = f"{prefix}Instantaneous Firing Rate Across Top {len(panel.electrodes)} electrodes"
    if recording_label:
        base = f"{recording_label}: {base}"
    if title:
        base = f"{title} | {base}"
    return base


def _ifr_axis_label(log_scale: bool) -> str:
    return "Log Instantaneous Firing Rate (Hz)" if log_scale else "Instantaneous Firing Rate (Hz)"


def _ifr_colorbar_label(log_scale: bool) -> str:
    return _ifr_axis_label(log_scale)


def _log_tick_label(value: float) -> str:
    return f"$10^{{{int(value)}}}$" if float(value).is_integer() else f"$10^{{{value:.1f}}}$"


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
