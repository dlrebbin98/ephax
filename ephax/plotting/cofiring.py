from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..metrics.cofiring import CofiringHeatmap
from .style import UNIT_LABELS, figure_mode_defaults, resolve_figure_size


def plot_cofiring_heatmap(
    heatmap: CofiringHeatmap,
    normalize: bool = False,
    cmap_name: str = "magma",
    title: str | None = None,
    show: bool = False,
    ax=None,
    mode="standalone",
    figsize: str | tuple[float, float] = "medium_single",
):
    """Plot co-firing heatmap Z(distance, delay) with optional t0 normalization."""
    defaults = figure_mode_defaults(mode)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=resolve_figure_size(figsize))
    draw_cofiring_heatmap(
        heatmap,
        ax,
        normalize=normalize,
        cmap_name=cmap_name,
        title=title,
        compact=defaults.compact,
        show_colorbar=defaults.show_colorbar,
        show_legend=defaults.show_legend,
    )
    if show:
        plt.show()
    return fig, ax


def draw_cofiring_heatmap(
    heatmap: CofiringHeatmap,
    ax,
    *,
    normalize: bool = False,
    cmap_name: str = "magma",
    title: str | None = None,
    compact: bool = False,
    show_colorbar: bool | None = True,
    show_legend: bool | None = None,
):
    """Draw a co-firing heatmap into caller-provided axes."""
    Z = heatmap.Z.copy()
    delays = heatmap.delays
    if normalize and np.any(np.isclose(delays, 0)):
        t0_idx = int(np.argmin(np.abs(delays[:-1] - 0)))
        base = Z[t0_idx, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.divide(Z, base, out=np.zeros_like(Z), where=base != 0)

    extent = [
        float(heatmap.distance_bins.min()),
        float(heatmap.distance_bins.max()),
        float(delays.min()),
        float(delays.max()),
    ]
    image = ax.imshow(Z, aspect="auto", cmap=plt.get_cmap(cmap_name), extent=extent, origin="lower")
    ax.set_xlabel("Distance from Electrode ($\\mu m$)")
    ax.set_ylabel("Delay (ms)")
    resolved_title = title
    if resolved_title is None and not compact:
        resolved_title = f"{'Normalized ' if normalize else ''} p(Co-Firing) vs Distance and Time"
    if resolved_title:
        ax.set_title(resolved_title)
    ax.set_facecolor("black")
    cbar = None
    if show_colorbar is not False:
        cbar = ax.figure.colorbar(image, ax=ax)
        cbar.set_label(UNIT_LABELS["probability"])
    return {
        "heatmap": image,
        "mappable": image,
        "colorbar": cbar,
        "colorbar_label": UNIT_LABELS["probability"],
    }
