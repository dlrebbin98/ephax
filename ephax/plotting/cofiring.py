from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..models import CofiringHeatmap


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
    if normalize and np.any(np.isclose(delays, 0)):
        t0_idx = int(np.argmin(np.abs(delays[:-1] - 0)))
        base = Z[t0_idx, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.divide(Z, base, out=np.zeros_like(Z), where=base != 0)

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 6))
    extent = [
        float(heatmap.distance_bins.min()),
        float(heatmap.distance_bins.max()),
        float(delays.min()),
        float(delays.max()),
    ]
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
