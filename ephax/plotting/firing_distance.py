from __future__ import annotations

import matplotlib.pyplot as plt


def plot_binned_distance_series(
    result,
    *,
    ax=None,
    color: str = "blue",
    label: str = "Mean",
    xlabel: str = "Distance from Electrode ($\\mu m$)",
    ylabel: str = "Mean",
    title: str | None = None,
):
    """Plot a binned distance series with stderr shading."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 6))
    ax.plot(result.binned.centers, result.binned.mean, color=color, label=label)
    ax.fill_between(
        result.binned.centers,
        result.binned.mean - result.binned.stderr,
        result.binned.mean + result.binned.stderr,
        alpha=0.4,
        color=color,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax
