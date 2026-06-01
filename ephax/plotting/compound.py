from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from matplotlib.axes import Axes

from .layout import FigureSpec, PanelSpec, make_figure_grid, panel_subplotspec, required_nrows
from .style import FONT_SIZES


@dataclass(frozen=True)
class PanelRenderSpec:
    panel: PanelSpec
    draw: Callable[..., Any]
    data: Any
    options: Mapping[str, Any] | None = None
    axes_factory: Callable[..., Any] | None = None


@dataclass(frozen=True)
class PanelGroupSpec:
    panel: PanelSpec
    children: Sequence[PanelRenderSpec]
    ncols: int | None = None
    shared_colorbar: bool = True
    shared_legend: bool = False


@dataclass
class ComposedFigure:
    fig: Any
    grid: Any
    axes: dict[str, Any] = field(default_factory=dict)
    rendered: dict[str, Any] = field(default_factory=dict)


def figure(
    width: float | str = "wide_single",
    height: float | None = None,
    *,
    height_mm: float | None = None,
    rows: Sequence[float] | None = None,
    row_gaps: float | Sequence[float] | None = None,
    hspace: float | None = None,
    wspace: float | None = None,
    ncols: int = 12,
    constrained_layout: bool = True,
    content_type: str = "original_research",
) -> FigureSpec:
    """Create a figure layout request for composition."""
    return FigureSpec(
        width=width,
        height=height,
        ncols=ncols,
        row_heights=rows,
        row_gaps=row_gaps,
        hspace=hspace,
        wspace=wspace,
        constrained_layout=constrained_layout,
        height_mm=height_mm,
        content_type=content_type,
    )


def panel(
    key: str,
    *,
    draw: Callable[..., Any] | None = None,
    data: Any = None,
    label: str | None = None,
    title: str | None = None,
    loc: tuple[int, int, int, int] = (0, 0, 1, 12),
    compact: bool = True,
    show_legend: bool | None = None,
    show_colorbar: bool | None = None,
    suptitle: str | None = None,
    options: Mapping[str, Any] | None = None,
    axes_factory: Callable[..., Any] | None = None,
) -> PanelSpec | PanelRenderSpec:
    """Create a simple panel request for composition."""
    panel_spec = _panel_spec_from_loc(
        key,
        label=label,
        loc=loc,
        compact=compact,
        show_legend=show_legend,
        show_colorbar=show_colorbar,
        suptitle=suptitle,
    )
    if draw is None:
        return panel_spec
    resolved_options = dict(options or {})
    if title is not None:
        resolved_options.setdefault("title", title)
    return PanelRenderSpec(
        panel=panel_spec,
        draw=draw,
        data=data,
        options=resolved_options,
        axes_factory=axes_factory,
    )


def group(
    key: str,
    *,
    children: Sequence[PanelRenderSpec],
    label: str | None = None,
    loc: tuple[int, int, int, int] = (0, 0, 1, 12),
    ncols: int | None = None,
    shared_colorbar: bool = True,
    shared_legend: bool = False,
    compact: bool = True,
    suptitle: str | None = None,
) -> PanelGroupSpec:
    """Create a grouped panel request with optional shared visual furniture."""
    return PanelGroupSpec(
        panel=_panel_spec_from_loc(key, label=label, loc=loc, compact=compact, suptitle=suptitle),
        children=children,
        ncols=ncols,
        shared_colorbar=shared_colorbar,
        shared_legend=shared_legend,
    )


def compose_figure(fig_spec: FigureSpec, panel_specs: Sequence[PanelSpec | PanelRenderSpec | PanelGroupSpec]) -> ComposedFigure:
    """Create a composed figure and optionally render panel content into it."""
    panels = [_panel_from_spec(spec) for spec in panel_specs]
    fig, grid = make_figure_grid(fig_spec, nrows=required_nrows(panels))
    composed = ComposedFigure(fig=fig, grid=grid)

    for spec in panel_specs:
        panel = _panel_from_spec(spec)
        content_spec, title_ax = _panel_content_spec(fig, grid, panel)
        if isinstance(spec, PanelGroupSpec):
            group_axes, group_rendered = _render_group(fig, content_spec, spec)
            composed.axes[panel.key] = group_axes
            composed.rendered[panel.key] = group_rendered
            if title_ax is not None:
                _draw_panel_header(title_ax, panel.suptitle)
        elif isinstance(spec, PanelRenderSpec):
            axes = _make_axes(fig, content_spec, spec)
            composed.axes[panel.key] = axes
            if title_ax is not None:
                _draw_panel_header(title_ax, panel.suptitle)
            options = dict(spec.options or {})
            options.setdefault("compact", panel.compact)
            options.setdefault("show_legend", panel.show_legend)
            options.setdefault("show_colorbar", panel.show_colorbar)
            composed.rendered[panel.key] = spec.draw(spec.data, axes, **options)
        else:
            axes = _make_axes(fig, content_spec, spec)
            composed.axes[panel.key] = axes
            if title_ax is not None:
                _draw_panel_header(title_ax, panel.suptitle)
        if panel.label:
            _draw_panel_label(fig, grid, panel)

    return composed


def _panel_spec_from_loc(
    key: str,
    *,
    label: str | None,
    loc: tuple[int, int, int, int],
    compact: bool,
    show_legend: bool | None = None,
    show_colorbar: bool | None = None,
    suptitle: str | None = None,
) -> PanelSpec:
    row, col, rowspan, colspan = loc
    return PanelSpec(
        key=key,
        label=label,
        row=int(row),
        col=int(col),
        rowspan=int(rowspan),
        colspan=int(colspan),
        compact=compact,
        show_legend=show_legend,
        show_colorbar=show_colorbar,
        suptitle=suptitle,
    )


def _panel_from_spec(spec: PanelSpec | PanelRenderSpec | PanelGroupSpec) -> PanelSpec:
    return spec.panel if isinstance(spec, (PanelRenderSpec, PanelGroupSpec)) else spec


def _panel_content_spec(fig, grid, panel: PanelSpec):
    subplotspec = panel_subplotspec(grid, panel)
    if not panel.suptitle:
        return subplotspec, None
    panel_height = subplotspec.get_position(fig).height
    title_height = min(0.04, 0.30 * panel_height)
    title_grid = subplotspec.subgridspec(
        2,
        1,
        height_ratios=[title_height, panel_height - title_height],
        hspace=0.0,
    )
    title_ax = fig.add_subplot(title_grid[0, 0])
    title_ax.set_axis_off()
    return title_grid[1, 0], title_ax


def _draw_panel_label(fig, grid, panel: PanelSpec):
    label_ax = fig.add_subplot(panel_subplotspec(grid, panel))
    label_ax.set_axis_off()
    label_ax.patch.set_visible(False)
    label_ax.set_zorder(10)
    label_ax.text(
        0.0,
        1.02,
        panel.label.lower() if len(panel.label) == 1 and panel.label.isalpha() else panel.label,
        ha="left",
        va="bottom",
        transform=label_ax.transAxes,
        fontweight="bold",
        fontstyle="normal",
        fontsize=FONT_SIZES["panel_label"],
    )


def _draw_panel_header(ax, title: str | None):
    if title:
        ax.text(
            0.5,
            0.5,
            title,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONT_SIZES["title"],
        )


def _make_axes(fig, subplotspec, spec: PanelSpec | PanelRenderSpec | PanelGroupSpec):
    if isinstance(spec, PanelGroupSpec):
        return fig.add_subplot(subplotspec)
    if isinstance(spec, PanelRenderSpec) and spec.axes_factory is not None:
        return spec.axes_factory(fig, subplotspec)
    return fig.add_subplot(subplotspec)


def _render_group(fig, parent_spec, spec: PanelGroupSpec):
    panel = spec.panel
    n_children = len(spec.children)
    if n_children < 1:
        raise ValueError(f"panel group {panel.key!r} must contain at least one child.")
    ncols = spec.ncols if spec.ncols is not None else n_children
    ncols = max(1, int(ncols))
    nrows = int((n_children + ncols - 1) // ncols)
    include_shared_cbar = bool(spec.shared_colorbar)
    width_ratios = [1.0] * ncols + ([0.035] if include_shared_cbar else [])
    child_grid = parent_spec.subgridspec(nrows, ncols + int(include_shared_cbar), width_ratios=width_ratios)

    axes: dict[str, Any] = {}
    rendered: dict[str, Any] = {}
    first_mappable = None
    first_cbar_label = None

    for idx, child in enumerate(spec.children):
        row = idx // ncols
        col = idx % ncols
        child_axes = _make_child_axes(fig, child_grid[row, col], child)
        axes[child.panel.key] = child_axes
        options = dict(child.options or {})
        options.setdefault("compact", True)
        options.setdefault("show_legend", False if spec.shared_legend else child.panel.show_legend)
        options.setdefault("show_colorbar", False if include_shared_cbar else child.panel.show_colorbar)
        result = child.draw(child.data, child_axes, **options)
        rendered[child.panel.key] = result
        if isinstance(result, Mapping):
            first_mappable = first_mappable or result.get("mappable") or result.get("heatmap")
            first_cbar_label = first_cbar_label or result.get("colorbar_label")

    if include_shared_cbar and first_mappable is not None:
        cax = fig.add_subplot(child_grid[:, -1])
        cbar = fig.colorbar(first_mappable, cax=cax)
        if first_cbar_label:
            cbar.set_label(first_cbar_label)
        axes["colorbar"] = cax
        rendered["colorbar"] = cbar

    return axes, rendered


def _make_child_axes(fig, subplotspec, child: PanelRenderSpec):
    if child.axes_factory is not None:
        return child.axes_factory(fig, subplotspec)
    return fig.add_subplot(subplotspec)


def _label_axes(axes):
    for ax in _iter_axes(axes):
        return ax
    raise ValueError("Could not find a Matplotlib Axes for panel label placement.")


def _iter_axes(axes):
    if isinstance(axes, Axes):
        yield axes
    elif isinstance(axes, Mapping):
        for value in axes.values():
            yield from _iter_axes(value)
    elif isinstance(axes, (list, tuple)):
        for value in axes:
            yield from _iter_axes(value)
    elif hasattr(axes, "flat"):
        for value in axes.flat:
            yield from _iter_axes(value)
