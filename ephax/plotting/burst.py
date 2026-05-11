from __future__ import annotations

from io import BytesIO
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator

from ..models import AlignedBurstEvents, NetworkActivityState, PopulationIFR


def plot_population_ifr_summary(
    population: PopulationIFR,
    *,
    heatmap_title: str = "Electrode IFR matrix",
    mean_title: str = "Population mean IFR",
    mean_log_scale: bool = False,
    smooth_sigma_sec: float | None = None,
):
    """Plot selected-electrode IFR heatmap plus population mean trace."""
    positive = population.ifr_matrix[population.ifr_matrix > 0]
    if positive.size == 0:
        raise ValueError("IFR matrix contains no positive values for log-scale plotting.")
    vmin = max(1e-3, float(np.quantile(positive, 0.01)))
    vmax = max(float(positive.max()), vmin * (1.0 + 1e-6))

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.6], width_ratios=[40.0, 1.6])
    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_mean = fig.add_subplot(gs[1, 0], sharex=ax_heatmap)
    cax = fig.add_subplot(gs[0, 1])
    fig.add_subplot(gs[1, 1]).axis("off")

    im = ax_heatmap.imshow(
        np.clip(population.ifr_matrix, vmin, vmax),
        aspect="auto",
        origin="lower",
        extent=[population.time_grid[0], population.time_grid[-1], 0.5, population.ifr_matrix.shape[0] + 0.5],
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax_heatmap.set_ylabel("Selected electrode rank")
    ax_heatmap.set_yticks([1, population.ifr_matrix.shape[0]])
    ax_heatmap.set_title(heatmap_title)
    _format_log_colorbar(fig.colorbar(im, cax=cax), "Instantaneous firing rate (Hz, log scale)")

    smooth_label = "Smoothed mean IFR"
    if smooth_sigma_sec is not None:
        smooth_label = f"{smooth_label} (sigma={smooth_sigma_sec:.2f} s)"
    if mean_log_scale:
        floor = max(vmin, 1e-3)
        ax_mean.plot(population.time_grid, np.clip(population.mean_ifr, floor, None), color="0.70", lw=1.0, label="Population mean IFR")
        ax_mean.plot(population.time_grid, np.clip(population.mean_ifr_smooth, floor, None), color="black", lw=2.0, label=smooth_label)
        ax_mean.set_yscale("log")
        ax_mean.set_ylabel("Hz (log scale)")
    else:
        ax_mean.plot(population.time_grid, population.mean_ifr, color="0.70", lw=1.0, label="Population mean IFR")
        ax_mean.plot(population.time_grid, population.mean_ifr_smooth, color="black", lw=2.0, label=smooth_label)
        ax_mean.set_ylabel("Hz")
    ax_mean.set_xlabel("Time (s)")
    ax_mean.set_title(mean_title)
    ax_mean.legend(loc="upper right")
    ax_mean.set_xlim(float(population.time_grid[0]), float(population.time_grid[-1]))
    return fig, (ax_heatmap, ax_mean)


def plot_gamma_population_windows(aligned: AlignedBurstEvents, *, title: str | None = None):
    """Plot individual and mean population windows aligned to gamma anchors."""
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for row in aligned.population_windows:
        ax.plot(aligned.relative_time_ms, row, color="0.78", alpha=0.35)
    if aligned.population_windows.size:
        ax.plot(
            aligned.relative_time_ms,
            aligned.population_windows.mean(axis=0),
            color="black",
            lw=2.5,
            label="Mean gamma-centered population trace",
        )
    ax.axvline(0.0, color="crimson", ls="--", lw=1.2, label="Nested gamma anchor")
    ax.set_xlabel("Time relative to gamma anchor (ms)")
    ax.set_ylabel("Population spike-density rate (Hz)")
    ax.set_title(title or f"Gamma-centered population windows ({len(aligned.valid_anchors)} events)")
    ax.legend(loc="upper right")
    return fig, ax


def plot_aligned_electrode_heatmap(
    aligned: AlignedBurstEvents,
    ordered_rate: np.ndarray,
    *,
    y_label: str,
    title: str,
    bottom_traces: list[tuple[str, np.ndarray, str]] | None = None,
):
    """Plot an aligned per-electrode rate heatmap with optional summary traces."""
    positive = ordered_rate[ordered_rate > 0]
    if positive.size == 0:
        raise ValueError("Aligned rate matrix contains no positive values.")
    vmin = max(1e-2, float(np.quantile(positive, 0.05)))
    vmax = max(vmin * 10.0, float(np.quantile(positive, 0.995)))

    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.2], width_ratios=[40.0, 1.6])
    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_trace = fig.add_subplot(gs[1, 0], sharex=ax_heatmap)
    cax = fig.add_subplot(gs[0, 1])
    fig.add_subplot(gs[1, 1]).axis("off")

    im = ax_heatmap.imshow(
        np.clip(ordered_rate, vmin, vmax),
        aspect="auto",
        origin="lower",
        extent=[aligned.relative_time_ms[0], aligned.relative_time_ms[-1], 0.5, ordered_rate.shape[0] + 0.5],
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax_heatmap.axvline(0.0, color="crimson", ls="--", lw=1.0)
    ax_heatmap.set_ylabel(y_label)
    ax_heatmap.set_title(title)
    _format_log_colorbar(fig.colorbar(im, cax=cax), "Spike-density rate (Hz, log scale)")

    if bottom_traces is None:
        bottom_traces = [("All selected electrodes", aligned.population_windows.mean(axis=0), "0.45")]
    for label, values, color in bottom_traces:
        ax_trace.plot(aligned.relative_time_ms, values, color=color, lw=1.8, label=label)
    ax_trace.axvline(0.0, color="crimson", ls="--", lw=1.0)
    ax_trace.set_xlabel("Time relative to gamma anchor (ms)")
    ax_trace.set_ylabel("Population rate (Hz)")
    ax_trace.set_title("Mean gamma-centered population trace")
    ax_trace.legend(loc="upper right")
    return fig, (ax_heatmap, ax_trace)


def plot_macro_burst_detector_comparison_windows(
    *,
    time_grid,
    mean_ifr,
    mean_ifr_smooth,
    network_activity: NetworkActivityState,
    rate_peak_epochs: pd.DataFrame,
    network_epochs: pd.DataFrame,
    rate_peak_anchors: pd.DataFrame | None = None,
    network_anchors: pd.DataFrame | None = None,
    n_windows: int = 3,
    pad_s: float = 1.5,
    participation_threshold: float | None = None,
    window_source: str = "rate_peak",
):
    """Plot padded windows around macro bursts to compare two burst detectors."""
    time_grid = np.asarray(time_grid, dtype=float)
    mean_ifr = np.asarray(mean_ifr, dtype=float)
    mean_ifr_smooth = np.asarray(mean_ifr_smooth, dtype=float)
    if window_source == "rate_peak":
        source_epochs = rate_peak_epochs
        source_label = "rate-peak"
    elif window_source in {"network", "participation", "participation_gated"}:
        source_epochs = network_epochs
        source_label = "participation-gated"
    else:
        raise ValueError("window_source must be 'rate_peak' or 'network'.")
    if source_epochs.empty:
        raise ValueError(f"{source_label} epochs are empty; no comparison windows are available.")

    windows = source_epochs.sort_values("start_time_s").head(int(n_windows)).reset_index(drop=True)
    fig, axes = plt.subplots(len(windows), 1, figsize=(14, max(3.2, 3.0 * len(windows))), sharex=False, constrained_layout=True)
    axes = np.atleast_1d(axes)
    positive = mean_ifr[mean_ifr > 0]
    floor = max(1e-2, float(np.quantile(positive, 0.02))) if positive.size else 1e-2

    for panel_idx, (ax, window) in enumerate(zip(axes, windows.itertuples(index=False))):
        start_s = max(float(time_grid[0]), float(window.start_time_s) - float(pad_s))
        stop_s = min(float(time_grid[-1]), float(window.end_time_s) + float(pad_s))
        mask = (time_grid >= start_s) & (time_grid <= stop_s)
        ax.plot(time_grid[mask], np.clip(mean_ifr[mask], floor, None), color="black", lw=1.3, label="Population mean IFR" if panel_idx == 0 else None)
        ax.plot(time_grid[mask], np.clip(mean_ifr_smooth[mask], floor, None), color="0.5", lw=1.1, label="Smoothed population IFR" if panel_idx == 0 else None)
        ax.set_yscale("log")
        ax.set_ylabel("Hz")
        ax.set_xlim(start_s, stop_s)

        _shade_epochs(ax, rate_peak_epochs, start_s, stop_s, color="tab:orange", alpha=0.20, label="Rate-peak macro" if panel_idx == 0 else None)
        _shade_epochs(ax, network_epochs, start_s, stop_s, color="tab:blue", alpha=0.16, label="Participation macro" if panel_idx == 0 else None)
        _scatter_epoch_peaks(ax, rate_peak_epochs, start_s, stop_s, time_grid, mean_ifr, floor, color="tab:orange", marker="o", label="Rate-peak macro peak" if panel_idx == 0 else None)
        _scatter_epoch_peaks(ax, network_epochs, start_s, stop_s, time_grid, mean_ifr, floor, color="tab:blue", marker="v", label="Participation macro peak" if panel_idx == 0 else None)
        _scatter_anchor_times(ax, rate_peak_anchors, start_s, stop_s, time_grid, mean_ifr, floor, color="crimson", marker="o", label="Rate-peak gamma anchor" if panel_idx == 0 else None)
        _scatter_anchor_times(ax, network_anchors, start_s, stop_s, time_grid, mean_ifr, floor, color="navy", marker="x", label="Participation gamma anchor" if panel_idx == 0 else None)

        ax2 = ax.twinx()
        state_mask = (network_activity.time_centers_s >= start_s) & (network_activity.time_centers_s <= stop_s)
        ax2.plot(
            network_activity.time_centers_s[state_mask],
            network_activity.participation_fraction[state_mask],
            color="tab:blue",
            lw=1.0,
            alpha=0.9,
            label="Participation fraction" if panel_idx == 0 else None,
        )
        if participation_threshold is not None:
            ax2.axhline(
                float(participation_threshold),
                color="tab:blue",
                ls="--",
                lw=0.9,
                alpha=0.8,
                label="Participation threshold" if panel_idx == 0 else None,
            )
        ax2.set_ylim(0.0, max(0.25, min(1.0, float(network_activity.participation_fraction[state_mask].max()) * 1.20 if np.any(state_mask) else 0.25)))
        ax2.set_ylabel("Participation")
        ax.set_title(f"Window {panel_idx + 1}: {float(window.start_time_s):.2f}-{float(window.end_time_s):.2f} s")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if panel_idx == 0:
            ax.legend(lines + lines2, labels + labels2, loc="upper right", ncol=2, fontsize=9)
        if panel_idx == len(windows) - 1:
            ax.set_xlabel("Time (s)")
    return fig, axes


def plot_high_activity_burst_windows(
    *,
    time_grid,
    mean_ifr,
    mean_ifr_smooth,
    network_activity: NetworkActivityState,
    high_activity_epochs: pd.DataFrame,
    burst_epochs: pd.DataFrame,
    high_activity_threshold_hz: float | None = None,
    participation_threshold: float | None = None,
    n_windows: int = 3,
    pad_s: float = 1.5,
):
    """Plot high-activity periods, nested bursts, and max-participation anchors."""
    time_grid = np.asarray(time_grid, dtype=float)
    mean_ifr = np.asarray(mean_ifr, dtype=float)
    mean_ifr_smooth = np.asarray(mean_ifr_smooth, dtype=float)
    if high_activity_epochs.empty:
        raise ValueError("high_activity_epochs is empty; no comparison windows are available.")

    windows = high_activity_epochs.sort_values("start_time_s").head(int(n_windows)).reset_index(drop=True)
    fig, axes = plt.subplots(len(windows), 1, figsize=(14, max(3.2, 3.0 * len(windows))), sharex=False, constrained_layout=True)
    axes = np.atleast_1d(axes)
    positive = mean_ifr[mean_ifr > 0]
    floor = max(1e-2, float(np.quantile(positive, 0.02))) if positive.size else 1e-2

    for panel_idx, (ax, window) in enumerate(zip(axes, windows.itertuples(index=False))):
        start_s = max(float(time_grid[0]), float(window.start_time_s) - float(pad_s))
        stop_s = min(float(time_grid[-1]), float(window.end_time_s) + float(pad_s))
        mask = (time_grid >= start_s) & (time_grid <= stop_s)
        ax.plot(time_grid[mask], np.clip(mean_ifr[mask], floor, None), color="black", lw=1.3, label="Population mean IFR" if panel_idx == 0 else None)
        ax.plot(time_grid[mask], np.clip(mean_ifr_smooth[mask], floor, None), color="0.5", lw=1.1, label="Smoothed population IFR" if panel_idx == 0 else None)
        if high_activity_threshold_hz is not None:
            ax.axhline(
                max(float(high_activity_threshold_hz), floor),
                color="0.35",
                ls="--",
                lw=0.9,
                label="High-activity threshold" if panel_idx == 0 else None,
            )
        ax.set_yscale("log")
        ax.set_ylabel("Hz")
        ax.set_xlim(start_s, stop_s)

        _shade_epochs(ax, high_activity_epochs, start_s, stop_s, color="tab:green", alpha=0.13, label="High activity" if panel_idx == 0 else None)
        _shade_epochs(ax, burst_epochs, start_s, stop_s, color="tab:blue", alpha=0.20, label="Participation burst" if panel_idx == 0 else None)
        _scatter_burst_anchors(ax, burst_epochs, start_s, stop_s, time_grid, mean_ifr, floor, label="Max-participation burst anchor" if panel_idx == 0 else None)

        ax2 = ax.twinx()
        state_mask = (network_activity.time_centers_s >= start_s) & (network_activity.time_centers_s <= stop_s)
        ax2.plot(
            network_activity.time_centers_s[state_mask],
            network_activity.participation_fraction[state_mask],
            color="tab:blue",
            lw=1.0,
            alpha=0.9,
            label="Participation fraction" if panel_idx == 0 else None,
        )
        if participation_threshold is not None:
            ax2.axhline(
                float(participation_threshold),
                color="tab:blue",
                ls="--",
                lw=0.9,
                alpha=0.8,
                label="Participation threshold" if panel_idx == 0 else None,
            )
        ymax = 0.25
        if np.any(state_mask):
            ymax = max(ymax, min(1.0, float(network_activity.participation_fraction[state_mask].max()) * 1.20))
        ax2.set_ylim(0.0, ymax)
        ax2.set_ylabel("Participation")
        ax.set_title(f"High-activity window {panel_idx + 1}: {float(window.start_time_s):.2f}-{float(window.end_time_s):.2f} s")

        if panel_idx == 0:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right", ncol=2, fontsize=9)
        if panel_idx == len(windows) - 1:
            ax.set_xlabel("Time (s)")
    return fig, axes


def _scatter_epoch_peaks(ax, epochs, start_s, stop_s, time_grid, mean_ifr, floor, *, color, marker, label):
    if epochs is None or epochs.empty or "coarse_peak_time_s" not in epochs:
        return
    times = epochs["coarse_peak_time_s"].to_numpy(dtype=float)
    times = times[(times >= start_s) & (times <= stop_s)]
    if times.size == 0:
        return
    values = np.clip(np.interp(times, time_grid, mean_ifr), floor, None)
    ax.scatter(times, values, color=color, marker=marker, s=26, zorder=3, label=label)


def _scatter_burst_anchors(ax, burst_epochs, start_s, stop_s, time_grid, mean_ifr, floor, *, label):
    if burst_epochs is None or burst_epochs.empty:
        return
    time_column = "anchor_time_s" if "anchor_time_s" in burst_epochs else "coarse_peak_time_s"
    if time_column not in burst_epochs:
        return
    times = burst_epochs[time_column].to_numpy(dtype=float)
    times = times[(times >= start_s) & (times <= stop_s)]
    if times.size == 0:
        return
    values = np.clip(np.interp(times, time_grid, mean_ifr), floor, None)
    ax.scatter(times, values, color="navy", marker="D", s=32, zorder=4, label=label)


def save_average_hex_gif(
    aligned: AlignedBurstEvents,
    layout: dict | pd.DataFrame,
    output_path: str | Path,
    *,
    frame_step_ms: float = 2.0,
    bin_ms: float = 1.0,
    gridsize: int = 35,
    xlim: tuple[float, float] = (0.0, 3850.0),
    ylim: tuple[float, float] = (0.0, 2100.0),
    duration: float = 0.14,
    dpi: int = 140,
) -> Path:
    """Save a hex-binned GIF of average aligned electrode rate."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_rate = aligned.aligned_rate.mean(axis=0)
    any_spike = aligned.aligned_spikes.any(axis=0)
    layout_df = pd.DataFrame(layout).drop_duplicates("electrode")
    layout_df = layout_df[layout_df["electrode"].isin(aligned.electrodes)].copy()
    x = layout_df["x"].to_numpy(dtype=float)
    y = layout_df["y"].to_numpy(dtype=float)
    electrodes = layout_df["electrode"].to_numpy(dtype=int)
    idx = np.array([np.flatnonzero(aligned.electrodes == int(el))[0] for el in electrodes], dtype=int)

    positive = mean_rate[mean_rate > 0]
    vmin = max(1e-2, float(np.quantile(positive, 0.05)))
    vmax = max(vmin * 10.0, float(np.quantile(positive, 0.995)))
    extent = (*xlim, *ylim)
    box_aspect = (ylim[1] - ylim[0]) / max(xlim[1] - xlim[0], 1e-9)
    frame_step = max(1, int(round(float(frame_step_ms) / float(bin_ms))))
    frame_indices = np.arange(0, len(aligned.relative_time_ms), frame_step, dtype=int)

    with imageio.get_writer(output_path, mode="I", duration=duration) as writer:
        for time_index in frame_indices:
            fig, ax = plt.subplots(figsize=(7.4, 7.0))
            hb = ax.hexbin(
                x,
                y,
                C=np.clip(mean_rate[idx, int(time_index)], vmin, vmax),
                reduce_C_function=np.mean,
                gridsize=gridsize,
                cmap="magma",
                mincnt=1,
                linewidths=0.35,
                edgecolors="black",
                norm=LogNorm(vmin=vmin, vmax=vmax),
                extent=extent,
            )
            spike_mask = any_spike[idx, int(time_index)]
            if np.any(spike_mask):
                spike_hb = ax.hexbin(
                    x[spike_mask],
                    y[spike_mask],
                    gridsize=gridsize,
                    mincnt=1,
                    linewidths=0.35,
                    edgecolors="white",
                    facecolors="none",
                    extent=extent,
                )
                spike_hb.set_facecolor("none")
                spike_hb.set_edgecolor("white")
                spike_hb.set_linewidth(0.5)
            ax.set_facecolor("black")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel("x (um)")
            ax.set_ylabel("y (um)")
            ax.set_aspect("equal", adjustable="box")
            ax.set_box_aspect(box_aspect)
            ax.set_title(f"Hex-binned average electrode IFR | t = {aligned.relative_time_ms[time_index]:.0f} ms")
            _format_log_colorbar(fig.colorbar(hb, ax=ax), "Electrode IFR (Hz, log scale)")
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            buf.seek(0)
            writer.append_data(imageio.imread(buf))
            buf.close()
            plt.close(fig)
    return output_path


def _format_log_colorbar(cbar, label: str):
    cbar.set_label(label)
    cbar.locator = LogLocator(base=10)
    cbar.formatter = LogFormatterMathtext(base=10)
    cbar.update_ticks()
    return cbar


def _shade_epochs(ax, epochs: pd.DataFrame, start_s: float, stop_s: float, *, color: str, alpha: float, label: str | None):
    if epochs is None or epochs.empty:
        return
    overlap = epochs[(epochs["end_time_s"] >= start_s) & (epochs["start_time_s"] <= stop_s)]
    for idx, row in enumerate(overlap.itertuples(index=False)):
        ax.axvspan(max(start_s, float(row.start_time_s)), min(stop_s, float(row.end_time_s)), color=color, alpha=alpha, label=label if idx == 0 else None)


def _scatter_anchor_times(
    ax,
    anchors: pd.DataFrame | None,
    start_s: float,
    stop_s: float,
    time_grid,
    mean_ifr,
    floor: float,
    *,
    color: str,
    marker: str,
    label: str | None,
):
    if anchors is None or anchors.empty or "anchor_time_s" not in anchors:
        return
    anchor_times = anchors["anchor_time_s"].to_numpy(dtype=float)
    anchor_times = anchor_times[(anchor_times >= start_s) & (anchor_times <= stop_s)]
    if anchor_times.size == 0:
        return
    values = np.clip(np.interp(anchor_times, time_grid, mean_ifr), floor, None)
    ax.scatter(anchor_times, values, color=color, marker=marker, s=34, zorder=4, label=label)
