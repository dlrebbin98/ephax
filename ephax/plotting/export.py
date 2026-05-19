from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .style import NATURE_STYLE, apply_nature_style


def export_figure(
    fig,
    path_stem: str | Path,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    *,
    dpi: int = NATURE_STYLE.image_dpi,
    bbox_inches: str | None = "tight",
    apply_style: bool = True,
    **savefig_kwargs,
) -> list[Path]:
    """Export a figure to one or more formats from a path stem."""
    if apply_style:
        apply_nature_style()
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        clean_fmt = str(fmt).lstrip(".")
        out_path = stem.with_suffix(f".{clean_fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches, **savefig_kwargs)
        saved.append(out_path)
    return saved
