from __future__ import annotations

from io import BytesIO
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatterMathtext, LogLocator

from ..metrics.burst import AlignedBurstEvents, NetworkActivityState, PopulationIFR
from .style import COLORMAPS, FONT_SIZES, LINE_WIDTHS, PAPER_COLORS, UNIT_LABELS, figure_mode_defaults, resolve_figure_size


def plot_population_ifr_summary(
    population: PopulationIFR,
    *,
    heatmap_title: str = "Electrode IFR matrix",
    mean_title: str = "Population mean IFR",
    mean_log_scale: bool = False,
    smooth_sigma_sec: float | None = None,
    time_range: tuple[float, float] | None = None,
    heatmap_vmin: float | None = None,
    heatmap_vmax: float | None = None,
    heatmap_quantile: float = 0.01,
    mode="standalone",
    figsize: str | tuple[float, float] = "wide_single",
):
    """Plot selected-electrode IFR heatmap plus population mean trace."""
    defaults = figure_mode_defaults(mode)
    fig = plt.figure(figsize=resolve_figure_size(figsize), constrained_layout=True)
    axes = population_ifr_summary_axes_factory(fig, fig.add_gridspec(1, 1)[0, 0])
    draw_population_ifr_summary(
        population,
        axes,
        heatmap_title=heatmap_title,
        mean_title=mean_title,
        mean_log_scale=mean_log_scale,
        smooth_sigma_sec=smooth_sigma_sec,
        time_range=time_range,
        heatmap_vmin=heatmap_vmin,
        heatmap_vmax=heatmap_vmax,
        heatmap_quantile=heatmap_quantile,
        compact=defaults.compact,
        show_colorbar=defaults.show_colorbar,
        show_legend=defaults.show_legend,
    )
    return fig, (axes[0], axes[1])


def draw_population_ifr_summary(
    population: PopulationIFR,
    axes,
    *,
    heatmap_title: str = "Electrode IFR matrix",
    mean_title: str = "Population mean IFR",
    show_titles: bool | None = None,
    mean_log_scale: bool = False,
    smooth_sigma_sec: float | None = None,
    time_range: tuple[float, float] | None = None,
    heatmap_vmin: float | None = None,
    heatmap_vmax: float | None = None,
    heatmap_quantile: float = 0.01,
    compact: bool = False,
    show_colorbar: bool | None = True,
    show_legend: bool | None = False,
    colorbar_label: str | None = None,
):
    """Draw selected-electrode IFR heatmap plus population mean trace into axes."""
    ax_heatmap, ax_mean, cax = _unpack_population_axes(axes)
    if show_titles is None:
        show_titles = not compact
    population = _slice_population_ifr(population, time_range)
    positive = population.ifr_matrix[population.ifr_matrix > 0]
    if positive.size == 0:
        raise ValueError("IFR matrix contains no positive values for log-scale plotting.")
    quantile = float(np.clip(heatmap_quantile, 0.0, 1.0))
    vmin = max(1e-3, float(np.quantile(positive, quantile))) if heatmap_vmin is None else float(heatmap_vmin)
    vmax = float(positive.max()) if heatmap_vmax is None else float(heatmap_vmax)
    vmax = max(vmax, vmin * (1.0 + 1e-6))

    im = ax_heatmap.imshow(
        np.clip(population.ifr_matrix, vmin, vmax),
        aspect="auto",
        origin="lower",
        extent=[population.time_grid[0], population.time_grid[-1], 0.5, population.ifr_matrix.shape[0] + 0.5],
        cmap=COLORMAPS["heatmap"],
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax_heatmap.set_ylabel("Electrode index")
    ax_heatmap.set_yticks([1, population.ifr_matrix.shape[0]])
    if show_titles and heatmap_title:
        ax_heatmap.set_title(heatmap_title)
    cbar = None
    if show_colorbar is not False:
        cbar = ax_heatmap.figure.colorbar(im, cax=cax) if cax is not None else ax_heatmap.figure.colorbar(im, ax=ax_heatmap)
        resolved_cbar_label = colorbar_label if colorbar_label is not None else ("IFR (Hz)" if compact else "Instantaneous firing rate (Hz, log scale)")
        _format_log_colorbar(cbar, resolved_cbar_label)
    elif cax is not None:
        cax.axis("off")

    smooth_label = "Smoothed mean IFR"
    if smooth_sigma_sec is not None:
        smooth_label = f"{smooth_label} (sigma={smooth_sigma_sec:.2f} s)"
    if mean_log_scale:
        floor = max(vmin, 1e-3)
        ax_mean.plot(population.time_grid, np.clip(population.mean_ifr, floor, None), color="0.70", lw=LINE_WIDTHS["thin"], label="Population mean IFR")
        ax_mean.plot(population.time_grid, np.clip(population.mean_ifr_smooth, floor, None), color="black", lw=LINE_WIDTHS["emphasis"], label=smooth_label)
        ax_mean.set_yscale("log")
        ax_mean.set_ylabel("Mean IFR (Hz)")
    else:
        ax_mean.plot(population.time_grid, population.mean_ifr, color="0.70", lw=LINE_WIDTHS["thin"], label="Population mean IFR")
        ax_mean.plot(population.time_grid, population.mean_ifr_smooth, color="black", lw=LINE_WIDTHS["emphasis"], label=smooth_label)
        ax_mean.set_ylabel("Mean IFR (Hz)")
    ax_mean.set_xlabel(UNIT_LABELS["time_s"])
    if show_titles and mean_title:
        ax_mean.set_title(mean_title)
    if show_legend is not False:
        ax_mean.legend(loc="upper right")
    ax_mean.set_xlim(float(population.time_grid[0]), float(population.time_grid[-1]))
    return {
        "heatmap": im,
        "mappable": im,
        "colorbar": cbar,
        "colorbar_label": colorbar_label if colorbar_label is not None else ("IFR (Hz)" if compact else "Instantaneous firing rate (Hz, log scale)"),
        "mean": ax_mean,
    }


def _slice_population_ifr(population: PopulationIFR, time_range: tuple[float, float] | None) -> PopulationIFR:
    if time_range is None:
        return population
    start_s, stop_s = map(float, time_range)
    mask = (population.time_grid >= start_s) & (population.time_grid <= stop_s)
    if np.count_nonzero(mask) < 2:
        raise ValueError("time_range must include at least two population IFR samples.")
    return PopulationIFR(
        time_grid=population.time_grid[mask],
        electrodes=population.electrodes,
        ifr_matrix=population.ifr_matrix[:, mask],
        mean_ifr=population.mean_ifr[mask],
        mean_ifr_smooth=population.mean_ifr_smooth[mask],
        per_electrode_mean_hz=population.per_electrode_mean_hz,
    )


def population_ifr_summary_axes_factory(fig, subplotspec):
    """Create the nested axes bundle used by a population IFR summary panel."""
    gs = subplotspec.subgridspec(2, 2, height_ratios=[3.0, 1.6], width_ratios=[40.0, 1.6])
    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_mean = fig.add_subplot(gs[1, 0], sharex=ax_heatmap)
    cax = fig.add_subplot(gs[0, 1])
    spacer_ax = fig.add_subplot(gs[1, 1])
    spacer_ax.axis("off")
    spacer_ax.set_in_layout(False)
    return ax_heatmap, ax_mean, cax


def _unpack_population_axes(axes):
    if isinstance(axes, dict):
        return axes["heatmap"], axes["mean"], axes.get("colorbar")
    if isinstance(axes, (list, tuple)):
        if len(axes) == 2:
            return axes[0], axes[1], None
        if len(axes) == 3:
            return axes[0], axes[1], axes[2]
    raise ValueError("population IFR axes must be (heatmap, mean), (heatmap, mean, colorbar), or a matching dict.")


def plot_activity_state_ifr_kde_histograms(
    activity_kde_results: dict[str, dict[str, np.ndarray]],
    activity_values: dict[str, np.ndarray],
    *,
    states=("high_activity", "burst"),
    state_labels: dict[str, str] | None = None,
    state_colors: dict[str, str] | None = None,
    source_label: str | None = None,
    max_peak_labels: int = 6,
):
    """Plot activity-state IFR histograms with KDE-smoothed maxima."""
    states = list(states)
    fig, axes = plt.subplots(1, len(states), figsize=(7.5 * len(states), 4.8), squeeze=False, constrained_layout=True)
    axes = axes.reshape(-1)
    draw_activity_state_ifr_kde_histograms(
        activity_kde_results,
        axes,
        activity_values=activity_values,
        states=states,
        state_labels=state_labels,
        state_colors=state_colors,
        source_label=source_label,
        max_peak_labels=max_peak_labels,
        compact=False,
        show_legend=True,
    )
    if source_label:
        fig.suptitle(f"Binned-KDE maxima from {source_label}")
    return fig, axes


def draw_activity_state_ifr_kde_histograms(
    activity_kde_results: dict[str, dict[str, np.ndarray]],
    axes,
    *,
    activity_values: dict[str, np.ndarray] | None = None,
    states=("high_activity", "burst"),
    state_labels: dict[str, str] | None = None,
    state_colors: dict[str, str] | None = None,
    source_label: str | None = None,
    max_peak_labels: int = 6,
    compact: bool = False,
    show_legend: bool | None = True,
    show_colorbar: bool | None = None,
    hide_inner_xlabels: bool = True,
):
    """Draw activity-state IFR histograms with KDE-smoothed maxima into axes."""
    state_labels = state_labels or {
        "high_activity": "High activity, non-burst",
        "burst": "Burst",
    }
    state_colors = state_colors or {
        "low_activity": PAPER_COLORS["low_activity"],
        "high_activity": PAPER_COLORS["high_activity"],
        "burst": PAPER_COLORS["burst"],
    }
    states = list(states)
    axes = np.atleast_1d(axes).reshape(-1)
    if axes.size < len(states):
        raise ValueError(f"Need at least {len(states)} axes for states={states!r}.")
    activity_values = activity_values or {}

    artists = {}
    for idx, (ax, state) in enumerate(zip(axes[: len(states)], states)):
        hist = activity_kde_results[state]
        values = np.asarray(activity_values.get(state, []), dtype=float)
        widths = np.diff(hist["plot_edges_hz"])
        bars = ax.bar(
            hist["plot_centers_hz"],
            hist["counts"],
            width=widths,
            color=state_colors.get(state, "0.5"),
            alpha=0.42,
            edgecolor="white",
            linewidth=0.25,
            align="center",
            label=state_labels.get(state, state),
        )
        (line,) = ax.plot(
            hist["grid_hz"],
            hist["smoothed_counts"],
            color=state_colors.get(state, "0.5"),
            lw=LINE_WIDTHS["emphasis"],
        )
        peaks = ax.scatter(hist["peak_hz"], hist["peak_counts"], color="crimson", s=10 if compact else 20, zorder=3, label="Binned-KDE maxima")
        if max_peak_labels:
            for peak_hz, peak_counts in zip(hist["peak_hz"][: int(max_peak_labels)], hist["peak_counts"][: int(max_peak_labels)]):
                ax.text(
                    float(peak_hz),
                    float(peak_counts),
                    f"{peak_hz:.1f}",
                    fontsize=FONT_SIZES["small"],
                    ha="left",
                    va="bottom",
                )
        ax.set_xscale("log")
        positive_edges = np.asarray(hist["plot_edges_hz"], dtype=float)
        positive_edges = positive_edges[np.isfinite(positive_edges) & (positive_edges > 0)]
        if positive_edges.size >= 2:
            ax.set_xlim(float(positive_edges[0]), float(positive_edges[-1]))
        if compact and hide_inner_xlabels and idx < len(states) - 1:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", labelbottom=False)
        else:
            ax.set_xlabel("IFR (Hz)")
        ax.set_ylabel("Count")
        if compact:
            ax.set_title(state_labels.get(state, state), pad=2.0)
        else:
            ax.set_title(f"{state_labels.get(state, state)} IFR histogram (n={values.size:,})")
        ax.grid(False)
        if show_legend is not False:
            ax.legend(loc="upper left", fontsize=FONT_SIZES["small"])
        artists[state] = {"bars": bars, "line": line, "peaks": peaks}

    for ax in axes[len(states):]:
        ax.axis("off")
    return {"axes": axes[: len(states)], "artists": artists, "source_label": source_label}


def plot_electrode_peak_time_map(
    peak_map: pd.DataFrame,
    *,
    title: str | None = None,
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
    render_mode: str = "scatter",
    gridsize: int = 35,
    figsize: str | tuple[float, float] = "medium_single",
    mode="standalone",
):
    """Plot an HD-MEA map colored by electrode peak-time latency."""
    defaults = figure_mode_defaults(mode)
    fig, ax = plt.subplots(figsize=resolve_figure_size(figsize), constrained_layout=True)
    draw_electrode_peak_time_map(
        peak_map,
        ax,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        render_mode=render_mode,
        gridsize=gridsize,
        compact=defaults.compact,
        show_colorbar=defaults.show_colorbar,
    )
    return fig, ax


def draw_electrode_peak_time_map(
    peak_map: pd.DataFrame,
    ax,
    *,
    title: str | None = None,
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
    show_colorbar: bool | None = True,
    compact: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    marker_size: float | None = None,
    show_invalid: bool = True,
    render_mode: str = "scatter",
    gridsize: int = 35,
):
    """Draw an HD-MEA map colored by electrode peak-time latency."""
    required = {"x", "y", "peak_time_ms", "valid"}
    missing = required.difference(peak_map.columns)
    if missing:
        raise ValueError(f"peak_map is missing required columns: {sorted(missing)}")
    plot_df = peak_map.copy()
    valid = plot_df["valid"].astype(bool) & np.isfinite(plot_df["peak_time_ms"].to_numpy(dtype=float))
    finite_times = plot_df.loc[valid, "peak_time_ms"].to_numpy(dtype=float)
    if finite_times.size == 0:
        raise ValueError("peak_map contains no valid finite peak_time_ms values.")
    if vmin is None or vmax is None:
        limit = float(np.nanmax(np.abs(finite_times)))
        if not np.isfinite(limit) or limit <= 0:
            limit = 1.0
        vmin = -limit if vmin is None else float(vmin)
        vmax = limit if vmax is None else float(vmax)

    render_mode = str(render_mode).lower()
    if render_mode not in {"scatter", "hexbin"}:
        raise ValueError("render_mode must be 'scatter' or 'hexbin'.")
    size = float(marker_size if marker_size is not None else (6.0 if compact else 10.0))
    norm = Normalize(vmin=float(vmin), vmax=float(vmax))
    ax.set_facecolor("black")
    if render_mode == "scatter" and show_invalid and np.any(~valid):
        ax.scatter(
            plot_df.loc[~valid, "x"],
            plot_df.loc[~valid, "y"],
            s=size,
            c="0.20",
            edgecolors="none",
            alpha=0.45,
        )
    if render_mode == "hexbin":
        mappable = ax.hexbin(
            plot_df.loc[valid, "x"],
            plot_df.loc[valid, "y"],
            C=plot_df.loc[valid, "peak_time_ms"],
            reduce_C_function=np.mean,
            gridsize=int(gridsize),
            cmap=cmap,
            mincnt=1,
            linewidths=0.35,
            edgecolors="black",
            norm=norm,
            extent=(*xlim, *ylim) if xlim is not None and ylim is not None else None,
        )
    else:
        mappable = ax.scatter(
            plot_df.loc[valid, "x"],
            plot_df.loc[valid, "y"],
            c=plot_df.loc[valid, "peak_time_ms"],
            s=size,
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            alpha=0.95,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.set_xlabel("x ($\\mu m$)")
        ax.set_ylabel("y ($\\mu m$)")
    if title:
        ax.set_title(title)

    colorbar = None
    if show_colorbar is not False:
        colorbar = ax.figure.colorbar(mappable, ax=ax)
        colorbar.set_label("Peak time relative to event peak (ms)")
    return {
        "axes": ax,
        "artist": mappable,
        "scatter": mappable,
        "mappable": mappable,
        "colorbar": colorbar,
        "render_mode": render_mode,
        "colorbar_label": "Peak time relative to event peak (ms)",
    }


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
    window_order: str = "time",
):
    """Plot high-activity periods, nested bursts, and burst anchors."""
    windows = select_high_activity_windows(high_activity_epochs, n_windows=n_windows, window_order=window_order)
    fig, axes = plt.subplots(len(windows), 1, figsize=(14, max(3.2, 3.0 * len(windows))), sharex=False, constrained_layout=True)
    axes = np.atleast_1d(axes)
    draw_high_activity_burst_windows(
        {
            "time_grid": time_grid,
            "mean_ifr": mean_ifr,
            "mean_ifr_smooth": mean_ifr_smooth,
            "network_activity": network_activity,
            "high_activity_epochs": high_activity_epochs,
            "burst_epochs": burst_epochs,
            "high_activity_threshold_hz": high_activity_threshold_hz,
            "participation_threshold": participation_threshold,
        },
        axes,
        n_windows=n_windows,
        pad_s=pad_s,
        window_order=window_order,
        compact=False,
        show_legend=True,
    )
    return fig, axes


def select_high_activity_windows(
    high_activity_epochs: pd.DataFrame,
    *,
    n_windows: int = 3,
    window_order: str = "time",
) -> pd.DataFrame:
    """Select high-activity windows for plotting."""
    if high_activity_epochs.empty:
        raise ValueError("high_activity_epochs is empty; no comparison windows are available.")
    n_windows = max(1, int(n_windows))
    if window_order in {"time", "start_time", "chronological"}:
        return high_activity_epochs.sort_values("start_time_s").head(n_windows).reset_index(drop=True)
    if window_order in {"peak_ifr", "peak", "highest_peak"}:
        if "peak_mean_ifr_hz" not in high_activity_epochs:
            raise ValueError("window_order='peak_ifr' requires a peak_mean_ifr_hz column.")
        selected = high_activity_epochs.sort_values("peak_mean_ifr_hz", ascending=False).head(n_windows)
        return selected.sort_values("start_time_s").reset_index(drop=True)
    if window_order in {"duration", "longest"}:
        if "duration_ms" not in high_activity_epochs:
            raise ValueError("window_order='duration' requires a duration_ms column.")
        selected = high_activity_epochs.sort_values("duration_ms", ascending=False).head(n_windows)
        return selected.sort_values("start_time_s").reset_index(drop=True)
    raise ValueError("window_order must be 'time', 'peak_ifr', or 'duration'.")


def draw_high_activity_burst_windows(
    data: dict[str, object],
    axes,
    *,
    n_windows: int = 3,
    pad_s: float = 1.5,
    window_order: str = "time",
    title: str | list[str] | tuple[str, ...] | None = None,
    compact: bool = False,
    show_legend: bool | None = True,
    show_colorbar: bool | None = None,
):
    """Draw high-activity periods, nested bursts, and burst anchors into axes."""
    time_grid = data["time_grid"]
    mean_ifr = data["mean_ifr"]
    mean_ifr_smooth = data["mean_ifr_smooth"]
    network_activity = data["network_activity"]
    high_activity_epochs = data["high_activity_epochs"]
    burst_epochs = data["burst_epochs"]
    high_activity_threshold_hz = data.get("high_activity_threshold_hz")
    participation_threshold = data.get("participation_threshold")

    time_grid = np.asarray(time_grid, dtype=float)
    mean_ifr = np.asarray(mean_ifr, dtype=float)
    mean_ifr_smooth = np.asarray(mean_ifr_smooth, dtype=float)
    windows = select_high_activity_windows(high_activity_epochs, n_windows=n_windows, window_order=window_order)
    axes = np.atleast_1d(axes).reshape(-1)
    if axes.size < len(windows):
        raise ValueError(f"Need at least {len(windows)} axes for selected windows.")
    positive = mean_ifr[mean_ifr > 0]
    floor = max(1e-2, float(np.quantile(positive, 0.02))) if positive.size else 1e-2
    participation_color = "tab:blue"

    secondary_axes = []
    for panel_idx, (ax, window) in enumerate(zip(axes[: len(windows)], windows.itertuples(index=False))):
        start_s = max(float(time_grid[0]), float(window.start_time_s) - float(pad_s))
        stop_s = min(float(time_grid[-1]), float(window.end_time_s) + float(pad_s))
        mask = (time_grid >= start_s) & (time_grid <= stop_s)
        ax.plot(time_grid[mask], np.clip(mean_ifr[mask], floor, None), color="black", lw=LINE_WIDTHS["base"], label="Population mean IFR" if panel_idx == 0 else None)
        ax.plot(time_grid[mask], np.clip(mean_ifr_smooth[mask], floor, None), color="0.5", lw=LINE_WIDTHS["thin"], label="Smoothed population IFR" if panel_idx == 0 else None)
        ax.set_yscale("log")
        ax.set_ylabel("Mean IFR (Hz)")
        ax.set_xlim(start_s, stop_s)

        _shade_epochs(
            ax,
            high_activity_epochs,
            start_s,
            stop_s,
            color=PAPER_COLORS["high_activity"],
            alpha=0.13,
            label="High activity" if panel_idx == 0 else None,
        )
        _shade_epochs(
            ax,
            burst_epochs,
            start_s,
            stop_s,
            color=PAPER_COLORS["burst"],
            alpha=0.20,
            label="Participation burst" if panel_idx == 0 else None,
        )
        _scatter_burst_anchors(ax, burst_epochs, start_s, stop_s, time_grid, mean_ifr, floor, label="Burst activity anchor" if panel_idx == 0 else None)

        ax2 = ax.twinx()
        state_mask = (network_activity.time_centers_s >= start_s) & (network_activity.time_centers_s <= stop_s)
        ax2.plot(
            network_activity.time_centers_s[state_mask],
            network_activity.participation_fraction[state_mask],
            color=participation_color,
            lw=1.0,
            alpha=0.9,
            label="Participation fraction" if panel_idx == 0 else None,
        )
        ymax = 0.25
        if np.any(state_mask):
            ymax = max(ymax, min(1.0, float(network_activity.participation_fraction[state_mask].max()) * 1.20))
        ax2.set_ylim(0.0, ymax)
        ax2.set_ylabel("" if compact else "Participation proportion")
        ax2.yaxis.label.set_color(participation_color)
        ax2.tick_params(axis="y", colors=participation_color)
        ax2.spines["right"].set_color(participation_color)
        if compact:
            ax2.set_yticks([])
            ax2.tick_params(right=False, labelright=False)
            ax2.spines["right"].set_visible(False)
            ax2.set_in_layout(False)
        secondary_axes.append(ax2)
        panel_title = _resolve_panel_title(title, panel_idx)
        if panel_title is not None:
            ax.set_title(panel_title)
        elif compact:
            ax.set_title(f"{float(window.start_time_s):.2f}-{float(window.end_time_s):.2f} s")
        else:
            ax.set_title(f"High-activity window {panel_idx + 1}: {float(window.start_time_s):.2f}-{float(window.end_time_s):.2f} s")

        if panel_idx == 0 and show_legend is not False:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right", ncol=2, fontsize=FONT_SIZES["small"])
        if panel_idx == len(windows) - 1:
            ax.set_xlabel("Time (s)")
    for ax in axes[len(windows):]:
        ax.axis("off")
    return {"axes": axes[: len(windows)], "secondary_axes": secondary_axes, "windows": windows}


def _resolve_panel_title(title, idx: int) -> str | None:
    if title is None:
        return None
    if isinstance(title, str):
        return title if idx == 0 else None
    if idx < len(title):
        return title[idx]
    return None


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
    ax.scatter(times, values, color="navy", marker="D", s=10, zorder=4, label=label)


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


def save_event_hex_gif(
    aligned: AlignedBurstEvents,
    layout: dict | pd.DataFrame,
    window_idx: int,
    output_path: str | Path,
    *,
    title: str | None = None,
    frame_step_ms: float = 2.0,
    bin_ms: float | None = None,
    gridsize: int = 35,
    xlim: tuple[float, float] = (0.0, 3850.0),
    ylim: tuple[float, float] = (0.0, 2100.0),
    vmin: float | None = None,
    vmax: float | None = None,
    scale_values: np.ndarray | None = None,
    cmap: str = "magma",
    duration: float = 0.14,
    dpi: int = 140,
    show_spike_outline: bool = True,
) -> Path:
    """Save a hex-binned GIF for one aligned burst event."""
    window_idx = int(window_idx)
    if window_idx < 0 or window_idx >= aligned.aligned_rate.shape[0]:
        raise IndexError("window_idx is outside the aligned event tensor.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layout_df = pd.DataFrame(layout).drop_duplicates("electrode")
    layout_df = layout_df[layout_df["electrode"].isin(aligned.electrodes)].copy()
    x = layout_df["x"].to_numpy(dtype=float)
    y = layout_df["y"].to_numpy(dtype=float)
    electrodes = layout_df["electrode"].to_numpy(dtype=int)
    idx = np.array([np.flatnonzero(aligned.electrodes == int(el))[0] for el in electrodes], dtype=int)

    event_rate = aligned.aligned_rate[window_idx]
    event_spikes = aligned.aligned_spikes[window_idx]
    scale_source = event_rate if scale_values is None else np.asarray(scale_values, dtype=float)
    positive = scale_source[np.isfinite(scale_source) & (scale_source > 0)]
    if positive.size == 0:
        raise ValueError("No positive aligned rates were available for hex GIF scaling.")
    resolved_vmin = max(1e-2, float(np.quantile(positive, 0.05))) if vmin is None else float(vmin)
    resolved_vmax = max(resolved_vmin * 10.0, float(np.quantile(positive, 0.995))) if vmax is None else float(vmax)
    resolved_vmax = max(resolved_vmax, resolved_vmin * (1.0 + 1e-6))

    if bin_ms is None:
        diffs = np.diff(np.asarray(aligned.relative_time_ms, dtype=float))
        bin_ms = float(np.nanmedian(diffs)) if diffs.size else 1.0
    frame_step = max(1, int(round(float(frame_step_ms) / max(float(bin_ms), 1e-12))))
    frame_indices = np.arange(0, len(aligned.relative_time_ms), frame_step, dtype=int)

    extent = (*xlim, *ylim)
    box_aspect = (ylim[1] - ylim[0]) / max(xlim[1] - xlim[0], 1e-9)
    with imageio.get_writer(output_path, mode="I", duration=duration) as writer:
        for time_index in frame_indices:
            fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
            hb = ax.hexbin(
                x,
                y,
                C=np.clip(event_rate[idx, int(time_index)], resolved_vmin, resolved_vmax),
                reduce_C_function=np.mean,
                gridsize=gridsize,
                cmap=cmap,
                mincnt=1,
                linewidths=0.35,
                edgecolors="black",
                norm=LogNorm(vmin=resolved_vmin, vmax=resolved_vmax),
                extent=extent,
            )
            spike_mask = event_spikes[idx, int(time_index)]
            if show_spike_outline and np.any(spike_mask):
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
            resolved_title = title or f"Aligned event {window_idx}"
            ax.set_title(f"{resolved_title} | t = {aligned.relative_time_ms[time_index]:.0f} ms")
            _format_log_colorbar(fig.colorbar(hb, ax=ax), "Instantaneous firing rate (Hz, log scale)")
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
