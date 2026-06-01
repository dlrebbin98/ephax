from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def grid_avghz_panel_axes_factory(fig, subplotspec, *, n_items: int = 6, ncols: int = 3, include_colorbar: bool = True):
    """Create nested axes for a multi-recording average firing-rate grid panel."""
    n_items = max(1, int(n_items))
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_items / ncols))
    width_ratios = [1.0] * ncols + ([0.045] if include_colorbar else [])
    gs = subplotspec.subgridspec(nrows, ncols + int(include_colorbar), width_ratios=width_ratios)
    axes = np.empty((nrows, ncols), dtype=object)
    for row in range(nrows):
        for col in range(ncols):
            axes[row, col] = fig.add_subplot(gs[row, col])
    cax = fig.add_subplot(gs[:, -1]) if include_colorbar else None
    return axes, cax


def draw_grid_avghz_panel(
    results,
    axes,
    *,
    recording_titles=None,
    ncols: int | None = None,
    cmap_name: str = "magma",
    compact: bool = False,
    show_legend: bool | None = None,
    show_colorbar: bool | None = True,
    colorbar_label: str = "Mean IFR (Hz)",
):
    """Draw multiple precomputed layout-grid average firing-rate results into axes."""
    results = list(results)
    if not results:
        raise ValueError("At least one GridResult is required.")
    axes_array, cax = _unpack_grid_panel_axes(axes)
    flat_axes = axes_array.reshape(-1)
    if flat_axes.size < len(results):
        raise ValueError(f"Need at least {len(results)} axes for grid results.")

    global_vmin, global_vmax = _grid_global_limits(results)
    norm = _build_grid_norm(global_vmin, global_vmax)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("black")
    last_image = None
    rendered = {}
    for idx, ax in enumerate(flat_axes):
        if idx >= len(results):
            ax.axis("off")
            continue
        result = results[idx]
        grid_plot = _sanitize_grid_for_plot(result.grid)
        last_image = ax.imshow(
            grid_plot.T,
            origin="lower",
            cmap=cmap,
            norm=norm,
            extent=[result.x_min, result.x_max, result.y_min, result.y_max],
        )
        title = _resolve_recording_title(recording_titles, idx)
        if title:
            ax.set_title(title)
        ax.set_facecolor("black")
        ax.set_aspect("equal")
        ax.grid(False)
        if compact:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.set_xlabel("X-coordinate ($\\mu m$)")
            ax.set_ylabel("Y-coordinate ($\\mu m$)")
        rendered[idx] = {"heatmap": last_image, "mappable": last_image}

    colorbar = None
    if show_colorbar is not False and last_image is not None:
        if cax is not None:
            colorbar = flat_axes[0].figure.colorbar(last_image, cax=cax)
        else:
            plotted_axes = [ax for ax in flat_axes[: len(results)] if ax.get_visible()]
            colorbar = flat_axes[0].figure.colorbar(last_image, ax=plotted_axes)
        colorbar.set_label(colorbar_label)
    elif cax is not None:
        cax.axis("off")

    return {
        "axes": axes_array,
        "mappable": last_image,
        "heatmap": last_image,
        "colorbar": colorbar,
        "colorbar_label": colorbar_label,
        "panels": rendered,
    }


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


def _unpack_grid_panel_axes(axes):
    if isinstance(axes, dict):
        axes_array = np.asarray(axes["axes"], dtype=object)
        cax = axes.get("colorbar")
    elif isinstance(axes, (list, tuple)) and len(axes) == 2 and not hasattr(axes[0], "imshow"):
        axes_array = np.asarray(axes[0], dtype=object)
        cax = axes[1]
    else:
        axes_array = np.asarray(axes, dtype=object)
        cax = None
    return np.atleast_1d(axes_array), cax


def _grid_global_limits(results):
    chunks = []
    for result in results:
        grid_plot = _sanitize_grid_for_plot(result.grid)
        vals = grid_plot[np.isfinite(grid_plot)]
        if vals.size:
            chunks.append(vals)
    if not chunks:
        return 1e-6, 1e-6
    values = np.concatenate(chunks)
    return float(np.nanmin(values)), float(np.nanmax(values))


def _resolve_recording_title(recording_titles, idx: int) -> str | None:
    if recording_titles is None:
        return f"Recording {idx + 1}"
    if callable(recording_titles):
        return recording_titles(idx)
    if idx < len(recording_titles):
        return recording_titles[idx]
    return f"Recording {idx + 1}"


def _build_grid_norm(vmin: float, vmax: float):
    from matplotlib.colors import LogNorm, Normalize

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= 0 or vmax <= vmin:
        safe_vmin = 0.0 if not np.isfinite(vmin) else float(min(vmin, vmax if np.isfinite(vmax) else vmin))
        safe_vmax = 1.0 if not np.isfinite(vmax) else float(max(vmax, safe_vmin + 1e-6))
        return Normalize(vmin=safe_vmin, vmax=safe_vmax)
    return LogNorm(vmin=float(vmin), vmax=float(vmax))
