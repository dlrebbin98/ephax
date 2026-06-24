from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


DEFAULT_ARRAY_X_MIN_UM = 0.0
DEFAULT_ARRAY_Y_MIN_UM = 0.0
DEFAULT_ARRAY_X_MAX_UM = 3850.0
DEFAULT_ARRAY_Y_MAX_UM = 2100.0


@dataclass
class GridResult:
    grid: np.ndarray
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    vmin: float
    vmax: float


def compute_grid_avghz_pooled(
    recordings: Iterable,
    *,
    grid_size: float = 50.0,
    interpolate: bool = False,
) -> GridResult:
    """Compute a pooled average firing-rate grid across recordings."""
    recordings = list(recordings)
    if not recordings:
        raise ValueError("At least one recording is required.")
    layout_df = _combined_layout(recordings)
    pooled: dict[int, float] = {}
    counts: dict[int, int] = {}
    for rec in recordings:
        rates = _recording_rates(rec)
        for electrode, rate in rates.items():
            electrode = int(electrode)
            pooled[electrode] = pooled.get(electrode, 0.0) + float(rate)
            counts[electrode] = counts.get(electrode, 0) + 1
    if not pooled:
        raise ValueError("No firing data to pool.")
    avg_rates = {electrode: pooled[electrode] / counts[electrode] for electrode in pooled}
    return _grid_from_electrode_rates(layout_df, avg_rates, grid_size=grid_size, interpolate=interpolate)


def compute_grid_avghz_per_recording(
    recordings: Iterable,
    *,
    grid_size: float = 50.0,
    interpolate: bool = False,
) -> list[GridResult]:
    """Compute one average firing-rate grid per recording using shared extents."""
    recordings = list(recordings)
    if not recordings:
        return []
    layout_all = _combined_layout(recordings)
    x_n, y_n, x_min, x_max, y_min, y_max, x_bins, y_bins = _grid_extents(grid_size)
    results: list[GridResult] = []
    for rec in recordings:
        rates = _recording_rates(rec)
        result = _grid_from_electrode_rates(
            layout_all,
            rates,
            grid_size=grid_size,
            interpolate=interpolate,
            extents=(x_n, y_n, x_min, x_max, y_min, y_max, x_bins, y_bins),
        )
        results.append(result)
    return results


def format_recording_title(label: object, index: int) -> str:
    text = str(label).strip() if label is not None else ""
    if text:
        div_match = re.search(r"(?<![A-Za-z0-9])DIV[_\s-]*(\d+)", text, flags=re.IGNORECASE)
        well_match = re.search(r"(?<![A-Za-z0-9])well[_\s-]*(\d+)", text, flags=re.IGNORECASE)
        if div_match and well_match:
            return f"DIV {int(div_match.group(1))}: Well {int(well_match.group(1))}"
        return text
    return f"Recording {index + 1}"


def _combined_layout(recordings) -> pd.DataFrame:
    layouts = [pd.DataFrame(rec.layout) for rec in recordings]
    if not layouts:
        raise ValueError("At least one recording layout is required.")
    return pd.concat(layouts, ignore_index=True).drop_duplicates(subset=["electrode"])


def _recording_rates(rec) -> dict[int, float]:
    duration = float(rec.end_time - rec.start_time)
    if duration <= 0:
        return {}
    spikes = pd.DataFrame(rec.spikes)
    if spikes.empty or "electrode" not in spikes:
        return {}
    mask = (spikes["time"] >= float(rec.start_time)) & (spikes["time"] <= float(rec.end_time))
    counts = spikes.loc[mask, "electrode"].value_counts()
    return {int(electrode): float(count) / duration for electrode, count in counts.items()}


def _grid_extents(grid_size: float):
    x_min = DEFAULT_ARRAY_X_MIN_UM
    y_min = DEFAULT_ARRAY_Y_MIN_UM
    x_max = DEFAULT_ARRAY_X_MAX_UM
    y_max = DEFAULT_ARRAY_Y_MAX_UM
    x_bins = np.arange(x_min, x_max + float(grid_size), float(grid_size), dtype=float)
    y_bins = np.arange(y_min, y_max + float(grid_size), float(grid_size), dtype=float)
    x_n = max(1, len(x_bins) - 1)
    y_n = max(1, len(y_bins) - 1)
    return x_n, y_n, x_min, x_max, y_min, y_max, x_bins, y_bins


def _grid_from_electrode_rates(
    layout_df: pd.DataFrame,
    rates: dict[int, float],
    *,
    grid_size: float,
    interpolate: bool,
    extents=None,
) -> GridResult:
    x_n, y_n, x_min, x_max, y_min, y_max, x_bins, y_bins = extents or _grid_extents(grid_size)
    grid = np.zeros((x_n, y_n), dtype=float)
    counts = np.zeros((x_n, y_n), dtype=float)
    if rates:
        rate_df = layout_df[layout_df["electrode"].isin(rates.keys())].copy()
        for row in rate_df.itertuples(index=False):
            ix = int(np.digitize(float(row.x), x_bins) - 1)
            iy = int(np.digitize(float(row.y), y_bins) - 1)
            if 0 <= ix < x_n and 0 <= iy < y_n:
                grid[ix, iy] += float(rates[int(row.electrode)])
                counts[ix, iy] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        grid_avg = np.full_like(grid, np.nan, dtype=float)
        np.divide(grid, counts, out=grid_avg, where=counts != 0)
    if interpolate:
        grid_avg = _interpolate_grid(grid_avg, counts, x_min, y_min, grid_size)
    else:
        grid_avg[counts == 0] = np.nan
    valid = grid_avg[np.isfinite(grid_avg) & (grid_avg > 1e-6)]
    vmin, vmax = (1e-6, 1e-6) if valid.size == 0 else (float(valid.min()), float(valid.max()))
    return GridResult(grid=grid_avg, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, vmin=vmin, vmax=vmax)


def _interpolate_grid(grid_avg: np.ndarray, counts: np.ndarray, x_min: float, y_min: float, grid_size: float) -> np.ndarray:
    missing = counts == 0
    if not np.any(missing):
        return grid_avg
    x_centers = x_min + (np.arange(grid_avg.shape[0]) + 0.5) * float(grid_size)
    y_centers = y_min + (np.arange(grid_avg.shape[1]) + 0.5) * float(grid_size)
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    known = ~missing
    points = np.column_stack((grid_x[known], grid_y[known]))
    values = grid_avg[known]
    if points.shape[0] < 3:
        return grid_avg
    targets = np.column_stack((grid_x[missing], grid_y[missing]))
    interp_vals = griddata(points, values, targets, method="linear")
    if np.any(np.isnan(interp_vals)):
        nearest = griddata(points, values, targets, method="nearest")
        nan_mask = np.isnan(interp_vals)
        interp_vals[nan_mask] = nearest[nan_mask]
    filled = grid_avg.copy()
    filled[missing] = interp_vals
    return filled
