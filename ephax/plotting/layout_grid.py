from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def draw_grid_avghz(
    result,
    ax,
    *,
    title: str | None = None,
    cmap_name: str = "magma",
    compact: bool = False,
    show_colorbar: bool | None = True,
    colorbar_label: str = "Average firing rate (Hz)",
):
    """Draw a precomputed layout-grid average firing-rate result into an axes."""
    grid_plot = _sanitize_grid_for_plot(result.grid)
    finite_vals = grid_plot[np.isfinite(grid_plot)]
    if finite_vals.size:
        vmin = float(np.nanmin(finite_vals))
        vmax = float(np.nanmax(finite_vals))
    else:
        vmin, vmax = result.vmin, result.vmax

    norm = _build_grid_norm(vmin, vmax)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("black")
    image = ax.imshow(
        grid_plot.T,
        origin="lower",
        cmap=cmap,
        norm=norm,
        extent=[result.x_min, result.x_max, result.y_min, result.y_max],
    )
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.grid(False)
    if title:
        ax.set_title(title)
    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.set_xlabel("X-coordinate ($\\mu m$)")
        ax.set_ylabel("Y-coordinate ($\\mu m$)")

    cbar = None
    if show_colorbar is not False:
        cbar = ax.figure.colorbar(image, ax=ax)
        cbar.set_label(colorbar_label)
    return {
        "heatmap": image,
        "mappable": image,
        "colorbar": cbar,
        "colorbar_label": colorbar_label,
    }


def _sanitize_grid_for_plot(grid):
    arr = np.asarray(grid, dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    arr[arr <= 0] = np.nan
    return arr


def _build_grid_norm(vmin: float, vmax: float):
    from matplotlib.colors import LogNorm, Normalize

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= 0 or vmax <= vmin:
        safe_vmin = 0.0 if not np.isfinite(vmin) else float(min(vmin, vmax if np.isfinite(vmax) else vmin))
        safe_vmax = 1.0 if not np.isfinite(vmax) else float(max(vmax, safe_vmin + 1e-6))
        return Normalize(vmin=safe_vmin, vmax=safe_vmax)
    return LogNorm(vmin=float(vmin), vmax=float(vmax))
