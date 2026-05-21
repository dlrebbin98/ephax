from __future__ import annotations

import numpy as np

from ..models import WaveAnalysisResult
from .style import COLORMAPS, LINE_WIDTHS


def draw_wave_timing_panel(
    result: WaveAnalysisResult,
    ax,
    *,
    title: str | None = None,
    show_colorbar: bool | None = False,
    show_legend: bool | None = None,
    compact: bool = True,
    cmap: str | None = None,
):
    """Draw origin-aligned x-bin wave timing and linear speed fit."""
    heatmap = result.heatmap
    fit = result.fit_summary.iloc[0]
    bin_summary = result.bin_summary
    directions = result.event_direction

    time_ms = heatmap.index.to_numpy(dtype=float)
    x_um = heatmap.columns.to_numpy(dtype=float)
    values = heatmap.to_numpy(dtype=float)
    x_bin_um = float(fit.get("x_bin_um", np.nan))
    if not np.isfinite(x_bin_um):
        x_bin_um = float(np.nanmedian(np.diff(x_um))) if x_um.size > 1 else 1.0
    array_width_um = float(fit["array_width_um"])
    extent = [
        float(max(0.0, x_um.min() - 0.5 * x_bin_um)),
        float(min(array_width_um, x_um.max() + 0.5 * x_bin_um)),
        float(time_ms.min()),
        float(time_ms.max()),
    ]
    image = ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap or COLORMAPS["heatmap"],
        alpha=0.95,
    )
    ax.axhline(0.0, color="white", ls="--", lw=LINE_WIDTHS["thin"])
    errors = 1.96 * bin_summary["sem_peak_time_ms"].to_numpy(dtype=float)
    ax.errorbar(
        bin_summary["origin_x_um"],
        bin_summary["mean_peak_time_ms"],
        yerr=errors,
        fmt="o",
        color="black",
        capsize=2.5 if compact else 4.0,
        lw=LINE_WIDTHS["base"],
        ms=3.0 if compact else 4.5,
        label="Mean +/- 95% CI",
    )
    x_line = np.linspace(float(bin_summary["origin_x_um"].min()), float(bin_summary["origin_x_um"].max()), 200)
    y_line = float(fit["slope_ms_per_um"]) * x_line + float(fit["intercept_ms"])
    ax.plot(x_line, y_line, color="crimson", ls="--", lw=LINE_WIDTHS["emphasis"], label="Linear fit")
    ax.set_xlim(0.0, array_width_um)
    ax.set_xlabel("Distance from inferred origin side (um)")
    ax.set_ylabel("Peak time relative to event peak (ms)")
    counts = directions["event_direction"].value_counts().reindex(["left_to_right", "right_to_left"], fill_value=0)
    if title is None and not compact:
        title = (
            f"Wave-peak timing, speed ~ {float(fit['implied_speed_um_per_ms']):.0f} um/ms "
            f"(L->R {int(counts['left_to_right'])}, R->L {int(counts['right_to_left'])})"
        )
    if title:
        ax.set_title(title)
    if show_legend is not False:
        ax.legend(loc="upper left")
    colorbar = ax.figure.colorbar(image, ax=ax) if show_colorbar else None
    if colorbar is not None:
        colorbar.set_label("Mean spike-density rate (Hz)")
    return {"axes": ax, "mappable": image, "heatmap": image, "colorbar": colorbar}


def draw_wave_bootstrap_panel(
    result: WaveAnalysisResult,
    ax,
    *,
    title: str | None = None,
    compact: bool = True,
    bins: int = 30,
    show_legend: bool | None = None,
    show_colorbar: bool | None = None,
):
    """Draw bootstrap implied speed distribution for a wave result."""
    fit = result.fit_summary.iloc[0]
    speeds = np.asarray(result.bootstrap_speeds, dtype=float)
    speeds = speeds[np.isfinite(speeds)]
    artists = {}
    if speeds.size:
        hist = ax.hist(
            speeds,
            bins=int(bins),
            color="0.75",
            edgecolor="0.35",
            linewidth=LINE_WIDTHS["thin"],
        )
        artists["histogram"] = hist
        fit_speed = float(fit["implied_speed_um_per_ms"])
        boot_mean = float(fit["bootstrap_speed_mean_um_per_ms"])
        ci_low = float(fit["bootstrap_speed_ci_low_um_per_ms"])
        ci_high = float(fit["bootstrap_speed_ci_high_um_per_ms"])
        artists["fit_line"] = ax.axvline(
            fit_speed,
            color="crimson",
            lw=LINE_WIDTHS["emphasis"],
            label=f"Fit {fit_speed:.0f} um/ms",
        )
        artists["mean_line"] = ax.axvline(
            boot_mean,
            color="black",
            lw=LINE_WIDTHS["base"],
            ls="--",
            label=f"Mean {boot_mean:.0f} um/ms",
        )
        artists["ci_span"] = ax.axvspan(
            ci_low,
            ci_high,
            color="gold",
            alpha=0.25,
            label=f"95% CI [{ci_low:.0f}, {ci_high:.0f}]",
        )
        if show_legend is not False:
            ax.legend(loc="upper right")
    else:
        artists["message"] = ax.text(
            0.5,
            0.5,
            "Bootstrap speed distribution unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Implied propagation speed (um/ms)")
    ax.set_ylabel("Bootstrap count")
    if title:
        ax.set_title(title)
    return {"axes": ax, "artists": artists}
