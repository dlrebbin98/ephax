from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text


@dataclass(frozen=True)
class NatureFigureStyle:
    """Nature-branded research figure defaults in final-size points/mm."""

    font_family: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans")
    font_size: float = 5.5
    min_font_size: float = 5.0
    max_font_size: float = 7.0
    panel_label_size: float = 6.0
    axes_linewidth: float = 0.4
    tick_linewidth: float = 0.4
    major_tick_size: float = 1.2
    minor_tick_size: float = 0.8
    tick_pad: float = 0.5
    axes_labelpad: float = 1.0
    axes_titlepad: float = 2.0
    data_linewidth: float = 0.4
    emphasis_linewidth: float = 0.6
    inline_dpi: int = 160
    image_dpi: int = 450
    bitmap_min_dpi: int = 300
    one_column_mm: float = 88.0
    two_column_mm: float = 180.0


NATURE_STYLE = NatureFigureStyle()
FONT_FAMILY = list(NATURE_STYLE.font_family)
FONT_SIZES = {
    "small": 5.0,
    "base": NATURE_STYLE.font_size,
    "label": NATURE_STYLE.font_size,
    "title": 6,
    "panel_label": NATURE_STYLE.panel_label_size,
}
LINE_WIDTHS = {
    "thin": NATURE_STYLE.axes_linewidth,
    "base": NATURE_STYLE.data_linewidth,
    "emphasis": NATURE_STYLE.emphasis_linewidth,
}
STANDARD_COLORS = {
    "black": "black",
    "gray": "0.55",
    "light_gray": "0.75",
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
}
PAPER_COLORS = {
    "axonal_only": "#40bfff",
    "ephaptic_axonal": "#807fff",
    "ephaptic_term": "#c03fff",
    "control": "#40bfff",
    "treatment": "#807fff",
    "low_activity": "#8c8c8c",
    "high_activity": "#2ca02c",
    "burst": "#1f77b4",
    "reference": "#000000",
    "highlight": "#E69F00",
    "secondary": "#56B4E9",
    "neutral": "0.55",
}
PAPER_FONT_FAMILY = ["DejaVu Sans"]
UNIT_LABELS = {
    "time_s": "Time (s)",
    "time_ms": "Time (ms)",
    "distance_um": "Distance ($\\mu m$)",
    "ifr_hz": "Instantaneous firing rate (Hz)",
    "probability": "Probability",
}
COLORMAPS = {
    "heatmap": "viridis",
    "activity": "magma",
}
NATURE_FIGURE_WIDTHS_MM = {
    "original_research": {
        "one_column": 88.0,
        "two_column": 180.0,
    },
    "other_content": {
        "one_column": 58.0,
        "two_column": 121.0,
        "three_column": 185.0,
    },
    "protocols": {
        "one_column": 135.0,
        "full_page": 180.0,
    },
}
STANDALONE_FIGURE_SIZES = {
    "small_single": (100.0, 75.0),
    "medium_single": (140.0, 100.0),
    "wide_single": (180.0, 115.0),
}
FigureMode = Literal["standalone", "panel", "paper"]


@dataclass(frozen=True)
class FigureModeDefaults:
    mode: FigureMode
    compact: bool
    show_title: bool
    show_legend: bool | None
    show_colorbar: bool | None


FIGURE_MODE_DEFAULTS = {
    "standalone": FigureModeDefaults(
        mode="standalone",
        compact=False,
        show_title=True,
        show_legend=True,
        show_colorbar=True,
    ),
    "panel": FigureModeDefaults(
        mode="panel",
        compact=True,
        show_title=False,
        show_legend=False,
        show_colorbar=None,
    ),
    "paper": FigureModeDefaults(
        mode="paper",
        compact=True,
        show_title=False,
        show_legend=False,
        show_colorbar=None,
    ),
}


def mm_to_inches(value_mm: float) -> float:
    """Convert millimetres to inches."""
    return float(value_mm) / 25.4


def standalone_figure_size(kind: str = "medium_single") -> tuple[float, float]:
    """Return a publishable standalone figure size in inches."""
    try:
        width_mm, height_mm = STANDALONE_FIGURE_SIZES[kind]
    except KeyError as exc:
        valid = ", ".join(sorted(STANDALONE_FIGURE_SIZES))
        raise ValueError(f"unknown standalone figure size {kind!r}; valid sizes: {valid}") from exc
    return mm_to_inches(width_mm), mm_to_inches(height_mm)


def nature_figure_size(
    width: str | float = "two_column",
    *,
    height_mm: float,
    content_type: str = "original_research",
) -> tuple[float, float]:
    """Return a Nature figure size in inches from width presets and height in mm."""
    if isinstance(width, str):
        try:
            width_mm = NATURE_FIGURE_WIDTHS_MM[content_type][width]
        except KeyError as exc:
            valid_content = ", ".join(sorted(NATURE_FIGURE_WIDTHS_MM))
            valid_widths = ", ".join(sorted(NATURE_FIGURE_WIDTHS_MM.get(content_type, {})))
            raise ValueError(
                f"unknown Nature figure width {width!r} for content_type {content_type!r}; "
                f"valid content types: {valid_content}; valid widths for this content type: {valid_widths}"
            ) from exc
    else:
        width_mm = float(width)
    return mm_to_inches(width_mm), mm_to_inches(height_mm)


def resolve_figure_size(size: str | tuple[float, float] = "medium_single") -> tuple[float, float]:
    """Resolve standalone or explicit figure size to inches."""
    if isinstance(size, str):
        return standalone_figure_size(size)
    width, height = size
    return float(width), float(height)


def figure_mode_defaults(mode: FigureMode = "standalone") -> FigureModeDefaults:
    """Return default rendering options for a workflow stage."""
    try:
        return FIGURE_MODE_DEFAULTS[mode]
    except KeyError as exc:
        valid = ", ".join(sorted(FIGURE_MODE_DEFAULTS))
        raise ValueError(f"unknown figure mode {mode!r}; valid modes: {valid}") from exc


def seconds_to_ms(values):
    """Convert seconds to milliseconds."""
    return np.asarray(values, dtype=float) * 1000.0


def ms_to_seconds(values):
    """Convert milliseconds to seconds."""
    return np.asarray(values, dtype=float) / 1000.0


def apply_nature_style():
    """Apply Nature-compatible Matplotlib defaults and return the rcParams patch."""
    params = {
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZES["base"],
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["label"],
        "xtick.labelsize": FONT_SIZES["small"],
        "ytick.labelsize": FONT_SIZES["small"],
        "legend.fontsize": FONT_SIZES["small"],
        "axes.labelpad": NATURE_STYLE.axes_labelpad,
        "axes.titlepad": NATURE_STYLE.axes_titlepad,
        "axes.linewidth": LINE_WIDTHS["thin"],
        "xtick.major.width": NATURE_STYLE.tick_linewidth,
        "ytick.major.width": NATURE_STYLE.tick_linewidth,
        "xtick.minor.width": NATURE_STYLE.tick_linewidth,
        "ytick.minor.width": NATURE_STYLE.tick_linewidth,
        "xtick.major.size": NATURE_STYLE.major_tick_size,
        "ytick.major.size": NATURE_STYLE.major_tick_size,
        "xtick.minor.size": NATURE_STYLE.minor_tick_size,
        "ytick.minor.size": NATURE_STYLE.minor_tick_size,
        "xtick.major.pad": NATURE_STYLE.tick_pad,
        "ytick.major.pad": NATURE_STYLE.tick_pad,
        "xtick.minor.pad": NATURE_STYLE.tick_pad,
        "ytick.minor.pad": NATURE_STYLE.tick_pad,
        "lines.linewidth": LINE_WIDTHS["base"],
        "axes.grid": False,
        "figure.dpi": NATURE_STYLE.inline_dpi,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": NATURE_STYLE.image_dpi,
    }
    plt.rcParams.update(params)
    return params


def apply_paper_style():
    """Apply Nature defaults plus the paper-specific colour cycle."""
    params = apply_nature_style()
    palette = [
        PAPER_COLORS["axonal_only"],
        PAPER_COLORS["ephaptic_axonal"],
        PAPER_COLORS["high_activity"],
        PAPER_COLORS["burst"],
        PAPER_COLORS["low_activity"],
        PAPER_COLORS["highlight"],
        PAPER_COLORS["secondary"],
        PAPER_COLORS["reference"],
    ]
    cycle = plt.cycler(color=palette)
    plt.rcParams["font.family"] = PAPER_FONT_FAMILY
    plt.rcParams["axes.prop_cycle"] = cycle
    params["font.family"] = PAPER_FONT_FAMILY
    params["axes.prop_cycle"] = cycle
    return params


def apply_ephax_style():
    """Apply the standard Ephax Matplotlib style.

    The standard is now Nature-compatible; this name is kept for backward
    compatibility with existing notebooks.
    """
    return apply_nature_style()


def nature_figure_check(fig) -> list[str]:
    """Return non-mutating warnings for common Nature figure-style issues."""
    warnings: list[str] = []
    width_in, _height_in = fig.get_size_inches()
    width_mm = width_in * 25.4
    known_widths = [
        value
        for widths in NATURE_FIGURE_WIDTHS_MM.values()
        for value in widths.values()
    ]
    if known_widths and min(abs(width_mm - value) for value in known_widths) > 1.0:
        warnings.append(f"figure width {width_mm:.1f} mm does not match a Nature width preset.")

    for text in fig.findobj(match=Text):
        content = text.get_text().strip()
        if not content:
            continue
        size = float(text.get_fontsize())
        is_panel_label = (
            len(content) == 1
            and content.isalpha()
            and str(text.get_fontweight()).lower() in {"bold", "heavy", "700"}
        )
        if is_panel_label:
            if content != content.lower():
                warnings.append(f"panel label {content!r} should be lowercase.")
            if abs(size - NATURE_STYLE.panel_label_size) > 0.1:
                warnings.append(f"panel label {content!r} is {size:g} pt; Nature expects 8 pt.")
        elif size < NATURE_STYLE.min_font_size or size > NATURE_STYLE.max_font_size:
            warnings.append(
                f"text {content[:24]!r} is {size:g} pt; Nature figure text should be 5-7 pt."
            )
    return warnings


def truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 100):
    """Return a truncated copy of a colormap between [minval, maxval]."""
    return LinearSegmentedColormap.from_list(
        f"truncated({getattr(cmap, 'name', 'cmap')},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
