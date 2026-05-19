from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt

from .style import nature_figure_size


@dataclass(frozen=True)
class FigureSpec:
    width: float | str
    height: float | None = None
    ncols: int = 12
    row_heights: Sequence[float] | None = None
    row_gaps: float | Sequence[float] | None = None
    hspace: float | None = None
    wspace: float | None = None
    constrained_layout: bool = True
    height_mm: float | None = None
    content_type: str = "original_research"


@dataclass(frozen=True)
class PanelSpec:
    key: str
    label: str | None
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 12
    compact: bool = False
    show_legend: bool | None = None
    show_colorbar: bool | None = None
    suptitle: str | None = None


def make_figure_grid(spec: FigureSpec, nrows: int | None = None):
    """Create a figure and outer GridSpec for a 12-column composition."""
    resolved_nrows = _resolve_nrows(spec, nrows)
    height_ratios = _physical_row_heights(spec, resolved_nrows)
    fig = plt.figure(figsize=_figure_size_inches(spec), constrained_layout=spec.constrained_layout)
    grid = fig.add_gridspec(
        len(height_ratios),
        int(spec.ncols),
        height_ratios=height_ratios,
        hspace=spec.hspace,
        wspace=spec.wspace,
    )
    grid._ephax_logical_nrows = resolved_nrows
    grid._ephax_row_map = _logical_row_map(resolved_nrows, spec.row_gaps)
    return fig, grid


def add_panel_axes(fig, grid, panel: PanelSpec):
    """Add one axes occupying a validated panel span."""
    return fig.add_subplot(panel_subplotspec(grid, panel))


def panel_subplotspec(grid, panel: PanelSpec):
    """Return the GridSpec slice for a logical panel placement."""
    logical_nrows = getattr(grid, "_ephax_logical_nrows", grid.get_geometry()[0])
    _validate_panel(panel, logical_nrows, grid.get_geometry()[1])
    row_map = getattr(grid, "_ephax_row_map", list(range(logical_nrows)))
    start = row_map[panel.row]
    stop = row_map[panel.row + panel.rowspan - 1] + 1
    return grid[start:stop, panel.col : panel.col + panel.colspan]


def layout_preset(name: str) -> list[PanelSpec]:
    """Return a small reusable 12-column layout preset."""
    if name == "single":
        return [PanelSpec(key="a", label="a", row=0, col=0, colspan=12)]
    if name == "two_columns":
        return [
            PanelSpec(key="a", label="a", row=0, col=0, colspan=6),
            PanelSpec(key="b", label="b", row=0, col=6, colspan=6),
        ]
    if name == "wide_top_two_bottom":
        return [
            PanelSpec(key="a", label="a", row=0, col=0, colspan=12),
            PanelSpec(key="b", label="b", row=1, col=0, colspan=6),
            PanelSpec(key="c", label="c", row=1, col=6, colspan=6),
        ]
    raise ValueError("layout preset must be 'single', 'two_columns', or 'wide_top_two_bottom'.")


def required_nrows(panels: Sequence[PanelSpec]) -> int:
    if not panels:
        return 1
    return max(panel.row + panel.rowspan for panel in panels)


def _resolve_nrows(spec: FigureSpec, nrows: int | None) -> int:
    if nrows is not None:
        resolved = int(nrows)
    elif spec.row_heights is not None:
        resolved = len(spec.row_heights)
    else:
        resolved = 1
    if resolved < 1:
        raise ValueError("figure grid must have at least one row.")
    if spec.row_heights is not None and len(spec.row_heights) != resolved:
        raise ValueError("row_heights length must match the number of grid rows.")
    _resolve_row_gaps(spec.row_gaps, resolved)
    if int(spec.ncols) < 1:
        raise ValueError("FigureSpec.ncols must be at least 1.")
    return resolved


def _physical_row_heights(spec: FigureSpec, nrows: int) -> list[float]:
    content_heights = list(spec.row_heights) if spec.row_heights is not None else [1.0] * nrows
    gaps = _resolve_row_gaps(spec.row_gaps, nrows)
    if not gaps:
        return content_heights
    heights: list[float] = []
    for idx, height in enumerate(content_heights):
        heights.append(float(height))
        if idx < len(gaps):
            heights.append(float(gaps[idx]))
    return heights


def _logical_row_map(nrows: int, row_gaps: float | Sequence[float] | None) -> list[int]:
    gaps = _resolve_row_gaps(row_gaps, nrows)
    if not gaps:
        return list(range(nrows))
    return [idx * 2 for idx in range(nrows)]


def _resolve_row_gaps(row_gaps: float | Sequence[float] | None, nrows: int) -> list[float]:
    if row_gaps is None or nrows <= 1:
        return []
    if isinstance(row_gaps, (int, float)):
        gaps = [float(row_gaps)] * (nrows - 1)
    else:
        gaps = [float(gap) for gap in row_gaps]
    if len(gaps) != nrows - 1:
        raise ValueError("row_gaps must be a scalar or have one value between each figure row.")
    if any(gap < 0 for gap in gaps):
        raise ValueError("row_gaps values must be non-negative.")
    return gaps


def _figure_size_inches(spec: FigureSpec) -> tuple[float, float]:
    if isinstance(spec.width, str):
        height_mm = spec.height_mm if spec.height_mm is not None else spec.height
        if height_mm is None:
            raise ValueError("FigureSpec with a Nature width preset requires height_mm or height in mm.")
        return nature_figure_size(spec.width, height_mm=float(height_mm), content_type=spec.content_type)
    if spec.height is None:
        raise ValueError("FigureSpec with explicit inch width requires explicit inch height.")
    return float(spec.width), float(spec.height)


def _validate_panel(panel: PanelSpec, nrows: int, ncols: int) -> None:
    if panel.row < 0 or panel.col < 0:
        raise ValueError(f"panel {panel.key!r} row and col must be non-negative.")
    if panel.rowspan < 1 or panel.colspan < 1:
        raise ValueError(f"panel {panel.key!r} rowspan and colspan must be at least 1.")
    if panel.row + panel.rowspan > nrows:
        raise ValueError(f"panel {panel.key!r} extends beyond the figure rows.")
    if panel.col + panel.colspan > ncols:
        raise ValueError(f"panel {panel.key!r} extends beyond the figure columns.")
