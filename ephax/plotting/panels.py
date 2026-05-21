from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

from .style import FONT_SIZES, LINE_WIDTHS


def add_panel_label(
    ax,
    label: str,
    *,
    x: float = -0.08,
    y: float = 1.04,
    fontsize: float | None = None,
    lowercase: bool = True,
    **kwargs,
):
    """Add a bold panel label in axes coordinates."""
    if not label:
        return None
    display_label = label.lower() if lowercase and len(label) == 1 and label.isalpha() else label
    defaults = {
        "transform": ax.transAxes,
        "ha": "left",
        "va": "bottom",
        "fontweight": "bold",
        "fontstyle": "normal",
        "fontsize": fontsize or FONT_SIZES["panel_label"],
    }
    defaults.update(kwargs)
    return ax.text(x, y, display_label, **defaults)


def add_panel_bundle_label(
    axes,
    label: str | None,
    *,
    extra_artists=None,
    x_pad: float = 0.012,
    y_pad: float = 0.0,
    fontsize: float | None = None,
    lowercase: bool = True,
    **kwargs,
):
    """Add a panel label relative to a full axes bundle in figure coordinates."""
    if not label:
        return None
    flat_axes = list(_flatten_axes(axes))
    if not flat_axes:
        return None
    fig = flat_axes[0].figure
    bbox = _axes_bundle_bbox(flat_axes, extra_artists=extra_artists)
    display_label = label.lower() if lowercase and len(label) == 1 and label.isalpha() else label
    x = max(bbox.x0 - float(x_pad), 0.012)
    defaults = {
        "ha": "left",
        "va": "bottom",
        "fontweight": "bold",
        "fontstyle": "normal",
        "fontsize": fontsize or FONT_SIZES["panel_label"],
    }
    defaults.update(kwargs)
    text = fig.text(x, bbox.y1 + float(y_pad), display_label, **defaults)
    text.set_in_layout(False)
    return text


def add_panel_suptitle(
    axes,
    title: str | None,
    *,
    y_pad: float = 0.018,
    child_title_pad: float = 0.095,
    fontsize: float | None = None,
    fontweight: str = "normal",
    **kwargs,
):
    """Add a title centered above a panel axes bundle in figure coordinates."""
    if not title:
        return None
    flat_axes = list(_flatten_axes(axes))
    if not flat_axes:
        return None
    fig = flat_axes[0].figure
    bbox = _axes_bundle_bbox(flat_axes)
    title_clearance = float(child_title_pad) if any(ax.get_title() for ax in flat_axes) else 0.0
    defaults = {
        "ha": "center",
        "va": "bottom",
        "fontsize": fontsize or FONT_SIZES["title"],
        "fontweight": fontweight,
    }
    defaults.update(kwargs)
    text = fig.text((bbox.x0 + bbox.x1) / 2.0, bbox.y1 + float(y_pad) + title_clearance, title, **defaults)
    text.set_in_layout(False)
    return text


def add_scale_bar(
    ax,
    size: float,
    label: str,
    *,
    location: str = "lower right",
    pad_fraction: float = 0.06,
    color: str = "black",
    linewidth: float = LINE_WIDTHS["base"],
    text_pad_fraction: float = 0.02,
):
    """Add a horizontal data-coordinate scale bar to an axes."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_span = x1 - x0
    y_span = y1 - y0
    if location not in {"lower right", "lower left", "upper right", "upper left"}:
        raise ValueError("location must be lower right, lower left, upper right, or upper left.")

    left_side = location.endswith("left")
    lower_side = location.startswith("lower")
    x_start = x0 + pad_fraction * x_span if left_side else x1 - pad_fraction * x_span - size
    x_end = x_start + size
    y = y0 + pad_fraction * y_span if lower_side else y1 - pad_fraction * y_span
    va = "top" if lower_side else "bottom"
    text_y = y - text_pad_fraction * y_span if lower_side else y + text_pad_fraction * y_span

    line = ax.plot([x_start, x_end], [y, y], color=color, lw=linewidth, solid_capstyle="butt")
    text = ax.text((x_start + x_end) / 2.0, text_y, label, ha="center", va=va, color=color, fontsize=FONT_SIZES["small"])
    return line[0], text


def set_legend_visible(ax, visible: bool | None, *, loc: str = "best", **kwargs):
    """Apply a consistent optional legend policy."""
    if visible is None:
        return None
    legend = ax.get_legend()
    if visible:
        if legend is None:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend = ax.legend(loc=loc, **kwargs)
        elif kwargs:
            legend = ax.legend(loc=loc, **kwargs)
    elif legend is not None:
        legend.remove()
        legend = None
    return legend


def despine(ax, *, top: bool = True, right: bool = True):
    """Hide selected spines."""
    ax.spines["top"].set_visible(not top)
    ax.spines["right"].set_visible(not right)
    return ax


def _flatten_axes(axes):
    if isinstance(axes, Axes):
        yield axes
    elif isinstance(axes, dict):
        for value in axes.values():
            yield from _flatten_axes(value)
    elif isinstance(axes, (list, tuple)):
        for value in axes:
            yield from _flatten_axes(value)
    elif hasattr(axes, "flat"):
        for value in axes.flat:
            yield from _flatten_axes(value)


def _axes_bundle_bbox(axes, *, extra_artists=None):
    """Return a figure-coordinate bbox for an axes bundle without forcing a draw.

    These helpers are called while composed figures are still being built. Calling
    ``fig.canvas.draw()`` here can invoke Matplotlib's constrained-layout solver
    before all artists are stable, producing transient collapsed-axes warnings.
    Axes positions are already in figure coordinates and are enough for placing
    external panel labels and suptitles.
    """
    bboxes = []
    for ax in axes:
        bboxes.append(ax.get_position())
    for artist in extra_artists or []:
        if artist is None:
            continue
        if hasattr(artist, "get_position"):
            x, y = artist.get_position()
            bboxes.append(Bbox.from_extents(float(x), float(y), float(x), float(y)))
    return Bbox.union(bboxes)
