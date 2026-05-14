from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from ..models import AlignedBurstEvents, WaveAnalysisResult


def analyze_eventwise_waves(
    aligned: AlignedBurstEvents,
    layout: dict | pd.DataFrame,
    *,
    x_bin_um: float = 300.0,
    peak_search_start_ms: float = -15.0,
    peak_search_stop_ms: float = 20.0,
    trace_smooth_sigma_ms: float = 2.0,
    bin_ms: float = 1.0,
    min_electrodes_per_bin: int = 5,
    min_events_per_bin: int = 5,
    bootstrap_reps: int = 2000,
    random_seed: int = 0,
) -> WaveAnalysisResult:
    """Estimate eventwise x-axis propagation from gamma-aligned rate tensors."""
    if aligned.aligned_rate.size == 0:
        raise ValueError("No aligned events are available for wave analysis.")

    layout_df = pd.DataFrame(layout).drop_duplicates("electrode")
    layout_df = layout_df[layout_df["electrode"].isin(aligned.electrodes)].copy()
    layout_df = layout_df.set_index("electrode").loc[aligned.electrodes].reset_index()
    x_um = layout_df["x"].to_numpy(dtype=float)
    x_min = float(x_um.min())
    x_max = float(x_um.max())
    array_width_um = float(x_max - x_min)
    if array_width_um <= 0:
        raise ValueError("Wave analysis requires nonzero x-axis span.")

    peak_mask = (aligned.relative_time_ms >= float(peak_search_start_ms)) & (
        aligned.relative_time_ms <= float(peak_search_stop_ms)
    )
    if not np.any(peak_mask):
        raise ValueError("The configured wave peak-search window does not overlap the aligned time axis.")
    peak_indices = np.flatnonzero(peak_mask)
    sigma_bins = max(0.0, float(trace_smooth_sigma_ms) / float(bin_ms))

    abs_centers, abs_members = _bin_members_by_position(
        x_um,
        x_bin_um,
        min_electrodes_per_bin=min_electrodes_per_bin,
        centered_edges=True,
    )
    if len(abs_members) < 3:
        raise ValueError("Too few populated x-bins are available for wave-peak analysis.")

    abs_peak_df = _event_bin_peaks(
        aligned.aligned_rate,
        aligned.relative_time_ms,
        abs_centers,
        abs_members,
        peak_indices,
        sigma_bins,
        include_traces=False,
    )[0]
    if abs_peak_df.empty:
        raise ValueError("No eventwise absolute-x peaks could be extracted.")

    event_direction = _estimate_event_directions(abs_peak_df)
    if event_direction.empty:
        raise ValueError("No eventwise sweep directions could be estimated from x-bin peaks.")

    origin_edges = np.arange(0.0, array_width_um + float(x_bin_um), float(x_bin_um), dtype=float)
    if origin_edges.size < 2 or origin_edges[-1] < array_width_um:
        origin_edges = np.append(origin_edges, array_width_um)
    origin_centers = 0.5 * (origin_edges[:-1] + origin_edges[1:])
    members_by_direction = {
        "left_to_right": _members_for_edges(x_um - x_min, origin_edges, min_electrodes_per_bin),
        "right_to_left": _members_for_edges(x_max - x_um, origin_edges, min_electrodes_per_bin),
    }

    peak_rows = []
    trace_rows = []
    valid_anchors = pd.DataFrame(aligned.valid_anchors)
    for row in event_direction.itertuples(index=False):
        event_idx = int(row.window_idx)
        direction = str(row.event_direction)
        members = members_by_direction[direction]
        peaks, traces = _event_bin_peaks(
            aligned.aligned_rate[event_idx : event_idx + 1],
            aligned.relative_time_ms,
            origin_centers,
            members,
            peak_indices,
            sigma_bins,
            include_traces=True,
            event_index_offset=event_idx,
            event_direction=direction,
        )
        if not peaks.empty:
            if not valid_anchors.empty and event_idx < len(valid_anchors) and "coarse_event_idx" in valid_anchors.columns:
                peaks["coarse_event_idx"] = int(valid_anchors.iloc[event_idx]["coarse_event_idx"])
            peak_rows.append(peaks)
        if not traces.empty:
            trace_rows.append(traces)

    peak_df = pd.concat(peak_rows, ignore_index=True) if peak_rows else pd.DataFrame()
    trace_df = pd.concat(trace_rows, ignore_index=True) if trace_rows else pd.DataFrame()
    if peak_df.empty:
        raise ValueError("No eventwise origin-aligned x-bin peaks could be extracted.")

    bin_summary = summarize_wave_peaks(peak_df, min_events_per_bin=min_events_per_bin)
    if len(bin_summary) < 3:
        raise ValueError("Too few origin-aligned x-bins survived the minimum-event filter for wave-peak fitting.")

    fit_summary, boot_speeds = fit_wave_speed(
        peak_df,
        bin_summary,
        x_bin_um=x_bin_um,
        array_width_um=array_width_um,
        event_direction=event_direction,
        bootstrap_reps=bootstrap_reps,
        min_events_per_bin=min_events_per_bin,
        random_seed=random_seed,
    )
    heatmap = (
        trace_df.groupby(["time_ms", "origin_x_bin_idx", "origin_x_um"], as_index=False)["rate_hz"]
        .mean()
        .sort_values(["time_ms", "origin_x_um"])
        .pivot(index="time_ms", columns="origin_x_um", values="rate_hz")
        .sort_index()
        .sort_index(axis=1)
    )
    return WaveAnalysisResult(
        event_direction=event_direction,
        trace=trace_df,
        peaks=peak_df,
        bin_summary=bin_summary,
        fit_summary=fit_summary,
        heatmap=heatmap,
        bootstrap_speeds=boot_speeds,
    )


def summarize_wave_peaks(peak_df: pd.DataFrame, *, min_events_per_bin: int = 5) -> pd.DataFrame:
    summary = (
        peak_df.groupby(["origin_x_bin_idx", "origin_x_um"], as_index=False)
        .agg(
            mean_peak_time_ms=("peak_time_ms", "mean"),
            median_peak_time_ms=("peak_time_ms", "median"),
            std_peak_time_ms=("peak_time_ms", "std"),
            mean_peak_rate_hz=("peak_rate_hz", "mean"),
            n_events=("window_idx", "nunique"),
            n_observations=("window_idx", "size"),
        )
        .sort_values("origin_x_um")
        .reset_index(drop=True)
    )
    summary["std_peak_time_ms"] = summary["std_peak_time_ms"].fillna(0.0)
    summary["sem_peak_time_ms"] = summary["std_peak_time_ms"] / np.sqrt(
        np.maximum(1.0, summary["n_observations"].to_numpy(dtype=float))
    )
    return summary[summary["n_events"] >= int(min_events_per_bin)].reset_index(drop=True)


def fit_wave_speed(
    peak_df: pd.DataFrame,
    bin_summary: pd.DataFrame,
    *,
    x_bin_um: float,
    array_width_um: float,
    event_direction: pd.DataFrame,
    bootstrap_reps: int = 2000,
    min_events_per_bin: int = 5,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    x = bin_summary["origin_x_um"].to_numpy(dtype=float)
    y = bin_summary["mean_peak_time_ms"].to_numpy(dtype=float)
    weights = np.sqrt(np.maximum(1.0, bin_summary["n_observations"].to_numpy(dtype=float)))
    slope, intercept = np.polyfit(x, y, 1, w=weights)
    speed = 1.0 / abs(slope) if abs(slope) > 1e-9 else np.nan

    rng = np.random.default_rng(int(random_seed))
    unique_events = np.sort(peak_df["window_idx"].unique())
    boot_speeds = []
    for _ in range(int(bootstrap_reps)):
        sampled_events = rng.choice(unique_events, size=len(unique_events), replace=True)
        sampled_df = pd.concat([peak_df[peak_df["window_idx"] == int(event_id)] for event_id in sampled_events], ignore_index=True)
        sampled_summary = (
            sampled_df.groupby(["origin_x_bin_idx", "origin_x_um"], as_index=False)
            .agg(mean_peak_time_ms=("peak_time_ms", "mean"), n_rows=("window_idx", "size"), n_events=("window_idx", "nunique"))
            .sort_values("origin_x_um")
        )
        sampled_summary = sampled_summary[sampled_summary["n_events"] >= int(min_events_per_bin)]
        if len(sampled_summary) < 3 or sampled_summary["origin_x_um"].nunique() < 2:
            continue
        slope_boot, _ = np.polyfit(
            sampled_summary["origin_x_um"].to_numpy(dtype=float),
            sampled_summary["mean_peak_time_ms"].to_numpy(dtype=float),
            1,
            w=np.sqrt(np.maximum(1.0, sampled_summary["n_rows"].to_numpy(dtype=float))),
        )
        if np.isfinite(slope_boot) and abs(slope_boot) > 1e-9:
            boot_speeds.append(float(1.0 / abs(slope_boot)))

    boot_speeds = np.asarray(boot_speeds, dtype=float)
    if boot_speeds.size:
        ci_low, ci_high = np.percentile(boot_speeds, [2.5, 97.5])
        boot_median = float(np.median(boot_speeds))
        boot_mean = float(np.mean(boot_speeds))
    else:
        ci_low = ci_high = boot_median = boot_mean = np.nan

    counts = event_direction["event_direction"].value_counts().reindex(["left_to_right", "right_to_left"], fill_value=0)
    fit_summary = pd.DataFrame(
        [
            {
                "x_bin_um": float(x_bin_um),
                "array_width_um": float(array_width_um),
                "n_bins_retained": int(len(bin_summary)),
                "n_events_used": int(len(unique_events)),
                "n_events_left_to_right": int(counts["left_to_right"]),
                "n_events_right_to_left": int(counts["right_to_left"]),
                "slope_ms_per_um": float(slope),
                "intercept_ms": float(intercept),
                "implied_speed_um_per_ms": float(speed),
                "bootstrap_speed_mean_um_per_ms": float(boot_mean),
                "bootstrap_speed_median_um_per_ms": float(boot_median),
                "bootstrap_speed_ci_low_um_per_ms": float(ci_low),
                "bootstrap_speed_ci_high_um_per_ms": float(ci_high),
            }
        ]
    )
    return fit_summary, boot_speeds


def _bin_members_by_position(x_um, x_bin_um, *, min_electrodes_per_bin: int, centered_edges: bool):
    x_um = np.asarray(x_um, dtype=float)
    half = 0.5 * float(x_bin_um)
    if centered_edges:
        edges = np.arange(
            float(np.floor(x_um.min() / float(x_bin_um)) * float(x_bin_um) - half),
            float(np.ceil(x_um.max() / float(x_bin_um)) * float(x_bin_um) + 1.5 * half),
            float(x_bin_um),
        )
    else:
        edges = np.arange(float(x_um.min()), float(x_um.max()) + float(x_bin_um), float(x_bin_um))
    if edges.size < 2:
        edges = np.array([float(x_um.min() - half), float(x_um.max() + half)])
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, _members_for_edges(x_um, edges, min_electrodes_per_bin)


def _members_for_edges(values, edges, min_electrodes_per_bin):
    idx = np.digitize(values, edges, right=False) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    out = []
    for bin_idx in range(len(edges) - 1):
        members = np.flatnonzero(idx == bin_idx)
        if members.size >= int(min_electrodes_per_bin):
            out.append((int(bin_idx), members.astype(int)))
    return out


def _event_bin_peaks(
    aligned_rate,
    relative_time_ms,
    centers,
    members,
    peak_indices,
    sigma_bins,
    *,
    include_traces: bool,
    event_index_offset: int = 0,
    event_direction: str | None = None,
):
    peak_rows = []
    trace_rows = []
    for local_event_idx in range(aligned_rate.shape[0]):
        event_idx = int(event_index_offset + local_event_idx)
        for bin_idx, member_idx in members:
            trace = aligned_rate[local_event_idx, member_idx].mean(axis=0)
            if sigma_bins > 0:
                trace = gaussian_filter1d(trace, sigma=sigma_bins, mode="nearest")
            search_trace = trace[peak_indices]
            if not np.any(np.isfinite(search_trace)):
                continue
            peak_idx = int(peak_indices[int(np.nanargmax(search_trace))])
            row = {
                "window_idx": event_idx,
                "peak_time_ms": float(relative_time_ms[peak_idx]),
                "n_electrodes": int(member_idx.size),
            }
            if include_traces:
                row.update(
                    {
                        "event_direction": event_direction,
                        "origin_x_bin_idx": int(bin_idx),
                        "origin_x_um": float(centers[bin_idx]),
                        "peak_rate_hz": float(trace[peak_idx]),
                    }
                )
            else:
                row.update({"x_center_um": float(centers[bin_idx])})
            peak_rows.append(row)
            if include_traces:
                for time_ms, rate_hz in zip(relative_time_ms, trace):
                    trace_rows.append(
                        {
                            "window_idx": event_idx,
                            "event_direction": event_direction,
                            "origin_x_bin_idx": int(bin_idx),
                            "origin_x_um": float(centers[bin_idx]),
                            "time_ms": float(time_ms),
                            "rate_hz": float(rate_hz),
                        }
                    )
    return pd.DataFrame(peak_rows), pd.DataFrame(trace_rows)


def _estimate_event_directions(abs_peak_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_idx, event_df in abs_peak_df.groupby("window_idx"):
        event_df = event_df.sort_values("x_center_um")
        if len(event_df) < 3 or event_df["x_center_um"].nunique() < 2:
            continue
        x_vals = event_df["x_center_um"].to_numpy(dtype=float)
        y_vals = event_df["peak_time_ms"].to_numpy(dtype=float)
        weights = np.sqrt(np.maximum(1.0, event_df["n_electrodes"].to_numpy(dtype=float)))
        slope, intercept = np.polyfit(x_vals, y_vals, 1, w=weights)
        rows.append(
            {
                "window_idx": int(event_idx),
                "event_slope_ms_per_um": float(slope),
                "event_intercept_ms": float(intercept),
                "event_direction": "left_to_right" if slope > 0 else "right_to_left",
            }
        )
    return pd.DataFrame(rows)
