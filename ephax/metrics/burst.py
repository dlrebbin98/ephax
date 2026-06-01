from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import gaussian_kde

from ..models import AlignedBurstEvents, HighResTraces, NetworkActivityState, PopulationIFR
from .ifr import calculate_ifr


def build_population_ifr(
    recording,
    selected_electrodes,
    *,
    grid_hz: float = 50.0,
    smooth_sigma_sec: float = 0.15,
) -> PopulationIFR:
    """Build per-electrode and population IFR traces on a shared time grid."""
    ifr_data, per_electrode_mean_hz, _ = calculate_ifr(
        recording.spikes,
        selected_electrodes,
        recording.start_time,
        recording.end_time,
    )

    dt = 1.0 / float(grid_hz)
    time_grid = np.arange(recording.start_time, recording.end_time, dt, dtype=float)

    traces: list[np.ndarray] = []
    electrodes: list[int] = []
    avg_rates: list[float] = []
    for electrode in np.asarray(selected_electrodes, dtype=int):
        electrode = int(electrode)
        if electrode not in ifr_data:
            continue
        times, values = ifr_data[electrode]
        sample_idx = np.searchsorted(times, time_grid, side="right") - 1
        trace = np.zeros_like(time_grid, dtype=float)
        valid_mask = sample_idx >= 0
        if np.any(valid_mask):
            trace[valid_mask] = values[sample_idx[valid_mask]]
        traces.append(trace)
        electrodes.append(electrode)
        avg_rates.append(float(per_electrode_mean_hz[electrode]))

    if not traces:
        raise ValueError("No IFR traces were produced for the selected electrodes.")

    ifr_matrix = np.asarray(traces, dtype=float)
    mean_ifr = ifr_matrix.mean(axis=0)
    sigma_bins = max(float(smooth_sigma_sec) * float(grid_hz), 1e-6)
    mean_ifr_smooth = gaussian_filter1d(mean_ifr, sigma=sigma_bins, mode="nearest")

    return PopulationIFR(
        time_grid=time_grid,
        electrodes=np.asarray(electrodes, dtype=int),
        ifr_matrix=ifr_matrix,
        mean_ifr=mean_ifr,
        mean_ifr_smooth=mean_ifr_smooth,
        per_electrode_mean_hz=np.asarray(avg_rates, dtype=float),
    )


def interval_membership(times_s, intervals_df: pd.DataFrame) -> np.ndarray:
    """Return a mask indicating whether each time falls inside any interval."""
    times_s = np.asarray(times_s, dtype=float)
    mask = np.zeros(times_s.shape, dtype=bool)
    if intervals_df is None or len(intervals_df) == 0:
        return mask
    if not {"start_time_s", "end_time_s"}.issubset(intervals_df.columns):
        raise ValueError("intervals_df must contain start_time_s and end_time_s columns.")
    for row in intervals_df[["start_time_s", "end_time_s"]].itertuples(index=False):
        mask |= (times_s >= float(row.start_time_s)) & (times_s <= float(row.end_time_s))
    return mask


def extract_activity_state_ifr(
    recording,
    selected_refs,
    high_epochs_df: pd.DataFrame,
    burst_epochs_df: pd.DataFrame,
    *,
    max_hz: float | None = None,
) -> dict[str, np.ndarray]:
    """Extract raw ISI-derived IFR samples split by activity state.

    ``high_activity`` excludes samples that also fall inside a burst interval.
    """
    out = {"low_activity": [], "high_activity": [], "burst": []}
    spike_times = np.asarray(recording.spikes["time"], dtype=float)
    spike_electrodes = np.asarray(recording.spikes["electrode"], dtype=int)
    for electrode in np.asarray(selected_refs, dtype=int):
        electrode_times = np.sort(
            spike_times[
                (spike_electrodes == int(electrode))
                & (spike_times >= float(recording.start_time))
                & (spike_times <= float(recording.end_time))
            ]
        )
        if electrode_times.size < 2:
            continue
        isi_s = np.diff(electrode_times)
        valid = isi_s > 0
        if not np.any(valid):
            continue
        midpoints = 0.5 * (electrode_times[:-1][valid] + electrode_times[1:][valid])
        ifr_values = 1.0 / isi_s[valid]
        finite = np.isfinite(ifr_values) & (ifr_values > 0)
        if max_hz is not None:
            finite &= ifr_values <= float(max_hz)
        if not np.any(finite):
            continue
        midpoints = midpoints[finite]
        ifr_values = ifr_values[finite]
        in_burst = interval_membership(midpoints, burst_epochs_df)
        in_high = interval_membership(midpoints, high_epochs_df)
        out["burst"].append(ifr_values[in_burst])
        out["high_activity"].append(ifr_values[in_high & ~in_burst])
        out["low_activity"].append(ifr_values[~in_high])

    result = {}
    for state, chunks in out.items():
        nonempty_chunks = [chunk for chunk in chunks if chunk.size > 0]
        result[state] = np.concatenate(nonempty_chunks).astype(float) if nonempty_chunks else np.array([], dtype=float)
    return result


def _empty_binned_kde_summary(
    plot_edges_hz: np.ndarray | None = None,
    plot_centers_hz: np.ndarray | None = None,
    grid_hz: np.ndarray | None = None,
    smoothed_counts: np.ndarray | None = None,
):
    return {
        "plot_edges_hz": np.array([], dtype=float) if plot_edges_hz is None else np.asarray(plot_edges_hz, dtype=float),
        "plot_centers_hz": np.array([], dtype=float) if plot_centers_hz is None else np.asarray(plot_centers_hz, dtype=float),
        "counts": np.array([], dtype=float)
        if plot_centers_hz is None
        else np.zeros_like(np.asarray(plot_centers_hz, dtype=float)),
        "grid_hz": np.array([], dtype=float) if grid_hz is None else np.asarray(grid_hz, dtype=float),
        "smoothed_counts": np.array([], dtype=float)
        if smoothed_counts is None
        else np.asarray(smoothed_counts, dtype=float),
        "peak_hz": np.array([], dtype=float),
        "peak_counts": np.array([], dtype=float),
    }


def binned_kde_peak_summary(
    values_hz,
    *,
    log_bins: bool = True,
    n_bins: int = 260,
    grid_size: int = 8192,
    prominence_fraction: float = 0.012,
    distance_fraction: float = 0.006,
    bandwidth_scale: float = 0.22,
    max_hz: float | None = None,
) -> dict[str, np.ndarray]:
    """Summarize an IFR histogram with a KDE-smoothed count curve and peaks."""
    values_hz = np.asarray(values_hz, dtype=float)
    values_hz = values_hz[np.isfinite(values_hz) & (values_hz > 0)]
    if max_hz is not None:
        values_hz = values_hz[values_hz <= float(max_hz)]
    if values_hz.size < 2:
        return _empty_binned_kde_summary()

    if log_bins:
        domain_values = np.log10(values_hz)
        domain_edges = np.linspace(float(domain_values.min()), float(domain_values.max()), int(n_bins) + 1)
        domain_centers = 0.5 * (domain_edges[:-1] + domain_edges[1:])
        plot_edges_hz = np.power(10.0, domain_edges)
        plot_centers_hz = np.power(10.0, domain_centers)
        domain_grid = np.linspace(domain_edges[0], domain_edges[-1], int(grid_size))
        grid_hz = np.power(10.0, domain_grid)
    else:
        domain_values = values_hz
        domain_edges = np.linspace(float(values_hz.min()), float(values_hz.max()), int(n_bins) + 1)
        domain_centers = 0.5 * (domain_edges[:-1] + domain_edges[1:])
        plot_edges_hz = domain_edges
        plot_centers_hz = domain_centers
        domain_grid = np.linspace(domain_edges[0], domain_edges[-1], int(grid_size))
        grid_hz = domain_grid

    counts, _ = np.histogram(domain_values, bins=domain_edges)
    counts = counts.astype(float)
    valid = counts > 0
    if not np.any(valid):
        summary = _empty_binned_kde_summary(plot_edges_hz, plot_centers_hz, grid_hz, np.zeros_like(domain_grid))
        summary["counts"] = counts
        return summary
    if np.count_nonzero(valid) < 2:
        smoothed_counts = np.zeros_like(domain_grid, dtype=float)
        occupied_center = domain_centers[valid][0]
        peak_idx = int(np.argmin(np.abs(domain_grid - occupied_center)))
        smoothed_counts[peak_idx] = float(counts[valid][0])
        return {
            "plot_edges_hz": plot_edges_hz,
            "plot_centers_hz": plot_centers_hz,
            "counts": counts,
            "grid_hz": grid_hz,
            "smoothed_counts": smoothed_counts,
            "peak_hz": np.array([grid_hz[peak_idx]], dtype=float),
            "peak_counts": np.array([smoothed_counts[peak_idx]], dtype=float),
        }

    def scaled_scott(kde_obj):
        return kde_obj.scotts_factor() * float(bandwidth_scale)

    kde = gaussian_kde(domain_centers[valid], weights=counts[valid], bw_method=scaled_scott)
    density = kde(domain_grid)
    bin_width = float(np.mean(np.diff(domain_edges)))
    smoothed_counts = density * float(counts.sum()) * bin_width

    peak_idx, _ = find_peaks(
        smoothed_counts,
        prominence=float(np.nanmax(smoothed_counts)) * float(prominence_fraction),
        distance=max(1, int(len(domain_grid) * float(distance_fraction))),
    )
    if peak_idx.size == 0:
        peak_idx = np.array([int(np.argmax(smoothed_counts))], dtype=int)
    order = np.argsort(smoothed_counts[peak_idx])[::-1]
    peak_idx = peak_idx[order]

    return {
        "plot_edges_hz": plot_edges_hz,
        "plot_centers_hz": plot_centers_hz,
        "counts": counts,
        "grid_hz": grid_hz,
        "smoothed_counts": smoothed_counts,
        "peak_hz": grid_hz[peak_idx],
        "peak_counts": smoothed_counts[peak_idx],
    }


def activity_state_kde_peak_frequencies(
    activity_kde_results: dict[str, dict[str, np.ndarray]],
    states=("high_activity", "burst"),
    *,
    min_peak_hz: float = 30.0,
) -> np.ndarray:
    """Return sorted unique KDE peak frequencies for the requested states."""
    peaks = []
    for state in states:
        if state not in activity_kde_results:
            continue
        state_peaks = np.asarray(activity_kde_results[state].get("peak_hz", []), dtype=float)
        peaks.append(state_peaks[np.isfinite(state_peaks) & (state_peaks > float(min_peak_hz))])
    if not peaks:
        return np.array([], dtype=float)
    merged = np.concatenate(peaks)
    if merged.size == 0:
        return np.array([], dtype=float)
    return np.unique(np.sort(merged))


def detect_coarse_burst_epochs(
    time_axis,
    slow_signal,
    *,
    grid_hz: float,
    peak_distance_sec: float = 1.0,
    prominence_quantile: float = 0.90,
    prominence_scale: float = 0.20,
    rel_height: float = 0.20,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Detect merged coarse burst epochs from a slow population IFR trace."""
    time_axis = np.asarray(time_axis, dtype=float)
    slow_signal = np.asarray(slow_signal, dtype=float)
    peak_distance_bins = max(1, int(round(float(peak_distance_sec) * float(grid_hz))))
    prominence_floor = float(np.quantile(slow_signal, prominence_quantile) * prominence_scale)
    raw_peaks, peak_props = find_peaks(slow_signal, distance=peak_distance_bins, prominence=prominence_floor)

    slow_baseline = float(np.quantile(slow_signal, 0.25))
    intervals = []
    for peak_idx in raw_peaks:
        peak_height = float(slow_signal[peak_idx])
        cutoff = slow_baseline + rel_height * (peak_height - slow_baseline)

        left_idx = int(peak_idx)
        while left_idx > 0 and slow_signal[left_idx] > cutoff:
            left_idx -= 1

        right_idx = int(peak_idx)
        while right_idx < len(slow_signal) - 1 and slow_signal[right_idx] > cutoff:
            right_idx += 1

        intervals.append(
            {
                "left_idx": left_idx,
                "right_idx": right_idx,
                "peak_idx": int(peak_idx),
                "peak_height_hz": peak_height,
            }
        )

    intervals = sorted(intervals, key=lambda item: item["left_idx"])
    merged = []
    for interval in intervals:
        if not merged or interval["left_idx"] > merged[-1]["right_idx"]:
            merged.append(
                {
                    "left_idx": interval["left_idx"],
                    "right_idx": interval["right_idx"],
                    "peak_idx": interval["peak_idx"],
                    "peak_height_hz": interval["peak_height_hz"],
                    "subpeak_indices": [interval["peak_idx"]],
                }
            )
            continue

        current = merged[-1]
        current["right_idx"] = max(current["right_idx"], interval["right_idx"])
        current["subpeak_indices"].append(interval["peak_idx"])
        if interval["peak_height_hz"] > current["peak_height_hz"]:
            current["peak_idx"] = interval["peak_idx"]
            current["peak_height_hz"] = interval["peak_height_hz"]

    coarse_epochs = pd.DataFrame(
        [
            {
                "event_idx": event_idx,
                "start_idx": item["left_idx"],
                "end_idx": item["right_idx"],
                "start_time_s": float(time_axis[item["left_idx"]]),
                "end_time_s": float(time_axis[item["right_idx"]]),
                "coarse_peak_idx": item["peak_idx"],
                "coarse_peak_time_s": float(time_axis[item["peak_idx"]]),
                "coarse_peak_hz": float(item["peak_height_hz"]),
                "subpeak_count": len(item["subpeak_indices"]),
            }
            for event_idx, item in enumerate(merged)
        ]
    )
    return coarse_epochs, raw_peaks, peak_props


def build_network_activity_state(
    highres: HighResTraces,
    *,
    aggregation_ms: float = 10.0,
    active_rate_floor_hz: float = 1.0,
    threshold_baseline_quantile: float = 0.20,
    threshold_iqr_scale: float = 3.0,
) -> NetworkActivityState:
    """Aggregate per-electrode IFRs into a network activity state."""
    if highres.per_electrode_rate_hz.size == 0:
        raise ValueError("High-resolution traces are empty.")

    highres_dt_s = float(np.median(np.diff(highres.time_centers_s)))
    if not np.isfinite(highres_dt_s) or highres_dt_s <= 0:
        raise ValueError("High-resolution traces require a positive time step.")

    bins_per_aggregate = max(1, int(round((float(aggregation_ms) / 1000.0) / highres_dt_s)))
    n_time = highres.per_electrode_rate_hz.shape[1]
    n_bins = n_time // bins_per_aggregate
    if n_bins < 1:
        raise ValueError("Aggregation window is longer than the high-resolution trace.")

    trim_stop = n_bins * bins_per_aggregate
    rate_blocks = highres.per_electrode_rate_hz[:, :trim_stop].reshape(
        highres.per_electrode_rate_hz.shape[0],
        n_bins,
        bins_per_aggregate,
    )
    spike_blocks = highres.spike_presence[:, :trim_stop].reshape(
        highres.spike_presence.shape[0],
        n_bins,
        bins_per_aggregate,
    )

    per_electrode_rate_hz = rate_blocks.mean(axis=2)
    spike_counts = spike_blocks.sum(axis=2, dtype=np.int32)
    bin_edges_s = highres.bin_edges_s[: trim_stop + 1 : bins_per_aggregate]
    if len(bin_edges_s) == n_bins:
        bin_edges_s = np.append(bin_edges_s, highres.bin_edges_s[trim_stop])
    time_centers_s = 0.5 * (bin_edges_s[:-1] + bin_edges_s[1:])

    q_low = np.quantile(per_electrode_rate_hz, float(threshold_baseline_quantile), axis=1)
    q25 = np.quantile(per_electrode_rate_hz, 0.25, axis=1)
    q75 = np.quantile(per_electrode_rate_hz, 0.75, axis=1)
    thresholds = np.maximum(float(active_rate_floor_hz), q_low + float(threshold_iqr_scale) * (q75 - q25))
    active_mask = per_electrode_rate_hz > thresholds[:, None]

    active_electrode_counts = active_mask.sum(axis=0).astype(int)
    total_spike_counts = spike_counts.sum(axis=0).astype(int)
    participation_fraction = active_electrode_counts.astype(float) / max(1, len(highres.electrodes))
    population_activity_hz = per_electrode_rate_hz.mean(axis=0)
    network_score = population_activity_hz * participation_fraction

    return NetworkActivityState(
        bin_edges_s=bin_edges_s,
        time_centers_s=time_centers_s,
        electrodes=np.asarray(highres.electrodes, dtype=int),
        per_electrode_rate_hz=per_electrode_rate_hz,
        active_mask=active_mask,
        spike_counts=spike_counts,
        active_electrode_counts=active_electrode_counts,
        total_spike_counts=total_spike_counts,
        participation_fraction=participation_fraction,
        population_activity_hz=population_activity_hz,
        network_score=network_score,
        electrode_thresholds_hz=thresholds,
    )


def build_participation_activity_state(
    highres: HighResTraces,
    *,
    aggregation_ms: float = 10.0,
) -> NetworkActivityState:
    """Aggregate spike presence into a participation-rate activity state."""
    if highres.spike_presence.size == 0:
        raise ValueError("High-resolution traces are empty.")

    highres_dt_s = float(np.median(np.diff(highres.time_centers_s)))
    if not np.isfinite(highres_dt_s) or highres_dt_s <= 0:
        raise ValueError("High-resolution traces require a positive time step.")

    bins_per_aggregate = max(1, int(round((float(aggregation_ms) / 1000.0) / highres_dt_s)))
    n_time = highres.spike_presence.shape[1]
    n_bins = n_time // bins_per_aggregate
    if n_bins < 1:
        raise ValueError("Aggregation window is longer than the high-resolution trace.")

    trim_stop = n_bins * bins_per_aggregate
    rate_blocks = highres.per_electrode_rate_hz[:, :trim_stop].reshape(
        highres.per_electrode_rate_hz.shape[0],
        n_bins,
        bins_per_aggregate,
    )
    spike_blocks = highres.spike_presence[:, :trim_stop].reshape(
        highres.spike_presence.shape[0],
        n_bins,
        bins_per_aggregate,
    )

    per_electrode_rate_hz = rate_blocks.mean(axis=2)
    spike_counts = spike_blocks.sum(axis=2, dtype=np.int32)
    active_mask = spike_counts > 0
    bin_edges_s = highres.bin_edges_s[: trim_stop + 1 : bins_per_aggregate]
    if len(bin_edges_s) == n_bins:
        bin_edges_s = np.append(bin_edges_s, highres.bin_edges_s[trim_stop])
    time_centers_s = 0.5 * (bin_edges_s[:-1] + bin_edges_s[1:])

    active_electrode_counts = active_mask.sum(axis=0).astype(int)
    total_spike_counts = spike_counts.sum(axis=0).astype(int)
    participation_fraction = active_electrode_counts.astype(float) / max(1, len(highres.electrodes))
    population_activity_hz = per_electrode_rate_hz.mean(axis=0)
    network_score = population_activity_hz * participation_fraction

    return NetworkActivityState(
        bin_edges_s=bin_edges_s,
        time_centers_s=time_centers_s,
        electrodes=np.asarray(highres.electrodes, dtype=int),
        per_electrode_rate_hz=per_electrode_rate_hz,
        active_mask=active_mask,
        spike_counts=spike_counts,
        active_electrode_counts=active_electrode_counts,
        total_spike_counts=total_spike_counts,
        participation_fraction=participation_fraction,
        population_activity_hz=population_activity_hz,
        network_score=network_score,
        electrode_thresholds_hz=np.zeros(len(highres.electrodes), dtype=float),
    )


def detect_high_activity_epochs(
    time_axis,
    mean_ifr_smooth,
    *,
    mad_scale: float = 3.0,
    min_duration_ms: float = 30.0,
    max_gap_bins: int = 1,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Detect high-activity periods from a smoothed population IFR trace."""
    time_axis = np.asarray(time_axis, dtype=float)
    mean_ifr_smooth = np.asarray(mean_ifr_smooth, dtype=float)
    if time_axis.size != mean_ifr_smooth.size:
        raise ValueError("time_axis and mean_ifr_smooth must have the same length.")
    if time_axis.size == 0:
        return _empty_high_activity_epochs(), {"baseline_hz": np.nan, "mad_hz": np.nan, "threshold_hz": np.nan}

    finite = np.isfinite(mean_ifr_smooth)
    if not np.any(finite):
        return _empty_high_activity_epochs(), {"baseline_hz": np.nan, "mad_hz": np.nan, "threshold_hz": np.nan}

    signal = mean_ifr_smooth.copy()
    baseline = float(np.median(signal[finite]))
    mad = float(np.median(np.abs(signal[finite] - baseline)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 0:
        q25, q75 = np.quantile(signal[finite], [0.25, 0.75])
        robust_sigma = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    threshold = baseline + float(mad_scale) * robust_sigma
    candidate = finite & (signal > threshold)
    intervals = _merge_intervals(_boolean_intervals(candidate), max_gap_bins=max(0, int(max_gap_bins)))
    if not intervals:
        return _empty_high_activity_epochs(), {"baseline_hz": baseline, "mad_hz": mad, "threshold_hz": threshold}

    dt_s = float(np.median(np.diff(time_axis))) if time_axis.size > 1 else 0.0
    min_duration_s = float(min_duration_ms) / 1000.0
    rows = []
    for start_idx, stop_idx in intervals:
        end_idx = min(int(stop_idx) + 1, len(time_axis) - 1)
        start_time_s = float(time_axis[start_idx])
        end_time_s = float(time_axis[end_idx])
        if dt_s > 0 and end_idx == stop_idx:
            end_time_s += dt_s
        duration_s = max(0.0, end_time_s - start_time_s)
        if duration_s < min_duration_s:
            continue
        local = signal[start_idx : stop_idx + 1]
        peak_rel_idx = int(np.nanargmax(local)) if local.size else 0
        peak_idx = int(start_idx + peak_rel_idx)
        rows.append(
            {
                "event_idx": len(rows),
                "start_idx": int(start_idx),
                "end_idx": int(stop_idx),
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "peak_idx": peak_idx,
                "peak_time_s": float(time_axis[peak_idx]),
                "peak_mean_ifr_hz": float(signal[peak_idx]),
                "duration_ms": float(duration_s * 1000.0),
                "threshold_hz": threshold,
                "baseline_hz": baseline,
            }
        )
    if not rows:
        return _empty_high_activity_epochs(), {"baseline_hz": baseline, "mad_hz": mad, "threshold_hz": threshold}
    return pd.DataFrame(rows), {"baseline_hz": baseline, "mad_hz": mad, "threshold_hz": threshold}


def detect_participation_burst_epochs(
    activity: NetworkActivityState,
    high_activity_epochs: pd.DataFrame,
    *,
    min_participation_fraction: float = 0.20,
    min_duration_ms: float = 30.0,
    max_gap_bins: int = 1,
) -> pd.DataFrame:
    """Detect participation-defined bursts nested inside high-activity periods."""
    if activity.time_centers_s.size == 0 or high_activity_epochs.empty:
        return _empty_network_burst_epochs()

    high_activity_mask = np.zeros(activity.time_centers_s.size, dtype=bool)
    high_activity_event_idx = np.full(activity.time_centers_s.size, -1, dtype=int)
    for row in high_activity_epochs.itertuples(index=False):
        mask = (activity.time_centers_s >= float(row.start_time_s)) & (activity.time_centers_s <= float(row.end_time_s))
        high_activity_mask |= mask
        high_activity_event_idx[mask] = int(row.event_idx)

    candidate = high_activity_mask & (activity.participation_fraction >= float(min_participation_fraction))
    intervals = _merge_intervals(_boolean_intervals(candidate), max_gap_bins=max(0, int(max_gap_bins)))
    if not intervals:
        return _empty_network_burst_epochs()

    min_duration_s = float(min_duration_ms) / 1000.0
    min_active = max(1, int(np.ceil(float(min_participation_fraction) * len(activity.electrodes))))
    rows = []
    for start_idx, stop_idx in intervals:
        start_time_s = float(activity.bin_edges_s[start_idx])
        end_time_s = float(activity.bin_edges_s[stop_idx + 1])
        duration_s = end_time_s - start_time_s
        participating_electrodes = int(np.any(activity.active_mask[:, start_idx : stop_idx + 1], axis=1).sum())
        if duration_s < min_duration_s or participating_electrodes < min_active:
            continue

        local_participation = activity.participation_fraction[start_idx : stop_idx + 1]
        max_participation = float(np.nanmax(local_participation))
        candidate_rel = np.flatnonzero(local_participation == max_participation)
        if candidate_rel.size > 1:
            local_ifr = activity.population_activity_hz[start_idx : stop_idx + 1][candidate_rel]
            peak_rel_idx = int(candidate_rel[int(np.nanargmax(local_ifr))])
        else:
            peak_rel_idx = int(candidate_rel[0])
        peak_idx = int(start_idx + peak_rel_idx)

        rows.append(
            {
                "event_idx": len(rows),
                "high_activity_event_idx": int(high_activity_event_idx[peak_idx]),
                "start_idx": int(start_idx),
                "end_idx": int(stop_idx),
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "coarse_peak_idx": peak_idx,
                "coarse_peak_time_s": float(activity.time_centers_s[peak_idx]),
                "coarse_peak_hz": float(activity.population_activity_hz[peak_idx]),
                "anchor_time_s": float(activity.time_centers_s[peak_idx]),
                "anchor_participation_fraction": max_participation,
                "peak_participation_fraction": max_participation,
                "peak_active_electrodes": int(activity.active_electrode_counts[peak_idx]),
                "participating_electrodes": participating_electrodes,
                "total_spikes": int(activity.total_spike_counts[start_idx : stop_idx + 1].sum()),
                "duration_ms": float(duration_s * 1000.0),
                "min_active_electrodes": min_active,
                "subpeak_count": 1,
            }
        )

    if not rows:
        return _empty_network_burst_epochs()
    return pd.DataFrame(rows)


def refine_participation_burst_anchors(
    highres: HighResTraces,
    burst_epochs: pd.DataFrame,
    *,
    anchor_window_ms: float = 10.0,
) -> pd.DataFrame:
    """Refine burst anchors using rolling participation at high-resolution time steps.

    Burst periods remain defined by the coarser participation detector. The anchor
    is moved to the high-resolution time point with maximal rolling participation
    inside each burst, using population rate as a tie-breaker.
    """
    if burst_epochs.empty:
        return burst_epochs.copy()
    if highres.spike_presence.size == 0:
        raise ValueError("High-resolution traces are empty.")

    dt_ms = float(np.median(np.diff(highres.time_centers_s)) * 1000.0)
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("High-resolution traces require a positive time step.")

    window_bins = max(1, int(round(float(anchor_window_ms) / dt_ms)))
    kernel = np.ones(window_bins, dtype=int)
    rolling_active = np.asarray(
        [np.convolve(row.astype(int), kernel, mode="same") > 0 for row in highres.spike_presence],
        dtype=bool,
    )
    rolling_active_counts = rolling_active.sum(axis=0).astype(int)
    rolling_participation = rolling_active_counts.astype(float) / max(1, len(highres.electrodes))

    refined = burst_epochs.copy()
    if "anchor_time_s" in refined:
        refined["coarse_anchor_time_s"] = refined["anchor_time_s"].astype(float)
    else:
        refined["coarse_anchor_time_s"] = np.nan
    refined["anchor_window_ms"] = float(anchor_window_ms)
    refined["anchor_method"] = "rolling_participation"

    for idx, row in refined.iterrows():
        mask = (highres.time_centers_s >= float(row["start_time_s"])) & (highres.time_centers_s <= float(row["end_time_s"]))
        local_indices = np.flatnonzero(mask)
        if local_indices.size == 0:
            continue

        local_participation = rolling_participation[local_indices]
        max_participation = float(np.nanmax(local_participation))
        candidate_rel = np.flatnonzero(local_participation == max_participation)
        if candidate_rel.size > 1:
            local_rate = highres.population_rate_hz[local_indices[candidate_rel]]
            best_rel = int(candidate_rel[int(np.nanargmax(local_rate))])
        else:
            best_rel = int(candidate_rel[0])
        anchor_idx = int(local_indices[best_rel])

        refined.loc[idx, "anchor_time_s"] = float(highres.time_centers_s[anchor_idx])
        refined.loc[idx, "coarse_peak_time_s"] = float(highres.time_centers_s[anchor_idx])
        refined.loc[idx, "coarse_peak_idx"] = anchor_idx
        refined.loc[idx, "coarse_peak_hz"] = float(highres.population_rate_hz[anchor_idx])
        refined.loc[idx, "anchor_participation_fraction"] = max_participation
        refined.loc[idx, "peak_participation_fraction"] = max_participation
        refined.loc[idx, "peak_active_electrodes"] = int(rolling_active_counts[anchor_idx])
        refined.loc[idx, "anchor_population_rate_hz"] = float(highres.population_rate_hz[anchor_idx])
    return refined


def assign_max_population_ifr_burst_anchors(highres: HighResTraces, burst_epochs: pd.DataFrame) -> pd.DataFrame:
    """Anchor each burst epoch at its maximum high-resolution population IFR."""
    if burst_epochs.empty:
        return burst_epochs.copy()

    anchored = burst_epochs.copy()
    if "anchor_time_s" in anchored:
        anchored["participation_anchor_time_s"] = anchored["anchor_time_s"].astype(float)
    else:
        anchored["participation_anchor_time_s"] = np.nan
    anchored["anchor_method"] = "max_highres_population_ifr"

    for idx, row in anchored.iterrows():
        mask = (highres.time_centers_s >= float(row["start_time_s"])) & (
            highres.time_centers_s <= float(row["end_time_s"])
        )
        local_indices = np.flatnonzero(mask)
        if local_indices.size == 0:
            continue
        local_rate = highres.population_rate_hz[local_indices]
        anchor_idx = int(local_indices[int(np.nanargmax(local_rate))])
        anchored.loc[idx, "anchor_time_s"] = float(highres.time_centers_s[anchor_idx])
        anchored.loc[idx, "coarse_peak_time_s"] = float(highres.time_centers_s[anchor_idx])
        anchored.loc[idx, "coarse_peak_idx"] = anchor_idx
        anchored.loc[idx, "coarse_peak_hz"] = float(highres.population_rate_hz[anchor_idx])
        anchored.loc[idx, "anchor_population_rate_hz"] = float(highres.population_rate_hz[anchor_idx])
    return anchored


def detect_network_burst_epochs(
    activity: NetworkActivityState,
    *,
    min_participation_fraction: float = 0.20,
    min_active_electrodes: int = 10,
    merge_gap_ms: float = 50.0,
    min_duration_ms: float = 30.0,
    min_spikes: int = 20,
) -> pd.DataFrame:
    """Detect macro network bursts from participation-gated per-electrode IFRs."""
    if activity.time_centers_s.size == 0:
        return _empty_network_burst_epochs()

    bin_ms = float(np.median(np.diff(activity.bin_edges_s)) * 1000.0)
    min_active = max(1, int(min_active_electrodes))
    candidate = (
        (activity.participation_fraction >= float(min_participation_fraction))
        & (activity.active_electrode_counts >= min_active)
        & (activity.total_spike_counts > 0)
    )
    intervals = _boolean_intervals(candidate)
    if not intervals:
        return _empty_network_burst_epochs()

    merged_intervals = _merge_intervals(intervals, max_gap_bins=max(0, int(round(float(merge_gap_ms) / bin_ms))))
    min_duration_s = float(min_duration_ms) / 1000.0
    rows = []
    for start_idx, stop_idx in merged_intervals:
        start_time_s = float(activity.bin_edges_s[start_idx])
        end_time_s = float(activity.bin_edges_s[stop_idx + 1])
        duration_s = end_time_s - start_time_s
        total_spikes = int(activity.total_spike_counts[start_idx : stop_idx + 1].sum())
        participating_electrodes = int(np.any(activity.active_mask[:, start_idx : stop_idx + 1], axis=1).sum())
        if duration_s < min_duration_s or total_spikes < int(min_spikes) or participating_electrodes < min_active:
            continue
        local_score = activity.network_score[start_idx : stop_idx + 1]
        peak_rel_idx = int(np.nanargmax(local_score)) if local_score.size else 0
        peak_idx = int(start_idx + peak_rel_idx)
        rows.append(
            {
                "event_idx": len(rows),
                "start_idx": int(start_idx),
                "end_idx": int(stop_idx),
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "coarse_peak_idx": peak_idx,
                "coarse_peak_time_s": float(activity.time_centers_s[peak_idx]),
                "coarse_peak_hz": float(activity.population_activity_hz[peak_idx]),
                "peak_participation_fraction": float(activity.participation_fraction[peak_idx]),
                "peak_active_electrodes": int(activity.active_electrode_counts[peak_idx]),
                "participating_electrodes": participating_electrodes,
                "total_spikes": total_spikes,
                "duration_ms": float(duration_s * 1000.0),
                "subpeak_count": 1,
            }
        )

    if not rows:
        return _empty_network_burst_epochs()
    return pd.DataFrame(rows)


def build_highres_traces(
    recording,
    selected_electrodes,
    *,
    bin_ms: float = 1.0,
    smooth_sigma_ms: float = 3.0,
) -> HighResTraces:
    """Build high-resolution spike-density traces and spike-presence matrix."""
    dt = float(bin_ms) / 1000.0
    bin_edges = np.arange(recording.start_time, recording.end_time + dt, dt)
    time_centers = bin_edges[:-1] + 0.5 * dt

    spike_times = np.asarray(recording.spikes["time"], dtype=float)
    spike_electrodes = np.asarray(recording.spikes["electrode"], dtype=int)

    electrodes = np.asarray(selected_electrodes, dtype=int)
    per_electrode_rate_hz = np.empty((len(electrodes), len(time_centers)), dtype=np.float32)
    spike_presence = np.empty((len(electrodes), len(time_centers)), dtype=bool)
    spikes_by_electrode: dict[int, np.ndarray] = {}
    sigma_bins = float(smooth_sigma_ms) / float(bin_ms)
    for row_idx, electrode in enumerate(electrodes):
        electrode = int(electrode)
        electrode_times = np.sort(
            spike_times[
                (spike_electrodes == electrode)
                & (spike_times >= recording.start_time)
                & (spike_times <= recording.end_time)
            ]
        )
        spikes_by_electrode[electrode] = electrode_times

        counts, _ = np.histogram(electrode_times, bins=bin_edges)
        rate_hz = counts.astype(np.float32)
        rate_hz /= np.float32(dt)
        per_electrode_rate_hz[row_idx] = gaussian_filter1d(
            rate_hz,
            sigma=sigma_bins,
            mode="nearest",
            output=np.float32,
        )
        spike_presence[row_idx] = counts > 0

    population_rate_hz = per_electrode_rate_hz.mean(axis=0)

    return HighResTraces(
        bin_edges_s=bin_edges,
        time_centers_s=time_centers,
        electrodes=electrodes,
        per_electrode_rate_hz=per_electrode_rate_hz,
        population_rate_hz=population_rate_hz,
        spikes_by_electrode=spikes_by_electrode,
        spike_presence=spike_presence,
    )


def choose_nested_gamma_anchors(
    time_centers,
    population_rate_hz,
    *,
    coarse_start_s: float,
    coarse_end_s: float,
    coarse_event_idx: int,
    coarse_rel_height: float,
    bin_ms: float,
    search_ms: float = 120.0,
    search_to_epoch_end: bool = True,
    min_distance_ms: float = 40.0,
    prominence_frac: float = 0.08,
    prominence_abs_floor: float = 0.10,
    keep_height_frac: float = 0.50,
    baseline_floor: float = 0.0,
) -> list[dict[str, Any]]:
    """Select nested gamma-scale peak anchors inside one coarse burst epoch."""
    time_centers = np.asarray(time_centers, dtype=float)
    population_rate_hz = np.asarray(population_rate_hz, dtype=float)
    epoch_mask = (time_centers >= coarse_start_s) & (time_centers <= coarse_end_s)
    epoch_time = time_centers[epoch_mask]
    epoch_signal = population_rate_hz[epoch_mask]

    if epoch_signal.size == 0:
        nearest_idx = int(np.argmin(np.abs(time_centers - 0.5 * (float(coarse_start_s) + float(coarse_end_s)))))
        return [_fallback_anchor(time_centers, population_rate_hz, nearest_idx, coarse_event_idx, prominence_abs_floor)]

    local_baseline = max(float(np.quantile(epoch_signal, 0.10)), float(baseline_floor))
    local_peak = float(epoch_signal.max())
    onset_threshold = local_baseline + coarse_rel_height * (local_peak - local_baseline)
    onset_candidates = np.flatnonzero(epoch_signal >= onset_threshold)
    onset_idx = int(onset_candidates[0]) if onset_candidates.size else int(np.argmax(epoch_signal))
    onset_time_s = float(epoch_time[onset_idx])
    search_stop_s = float(coarse_end_s) if search_to_epoch_end else min(float(coarse_end_s), onset_time_s + search_ms / 1000.0)

    lead_mask = (epoch_time >= onset_time_s) & (epoch_time <= search_stop_s)
    lead_time = epoch_time[lead_mask]
    lead_signal = epoch_signal[lead_mask]
    if lead_signal.size == 0:
        fallback_idx = int(np.argmax(epoch_signal))
        return [
            _anchor_record(
                coarse_event_idx,
                0,
                onset_time_s,
                float(epoch_time[fallback_idx]),
                float(epoch_signal[fallback_idx]),
                0,
                0,
                1,
                onset_threshold,
                search_stop_s,
                prominence_abs_floor,
                float(epoch_signal[fallback_idx]),
            )
        ]

    min_distance_bins = max(1, int(round(float(min_distance_ms) / float(bin_ms))))
    prominence_floor = max(float(prominence_frac) * max(lead_signal.max() - local_baseline, 0.0), float(prominence_abs_floor))
    all_peak_idx, _ = find_peaks(lead_signal, distance=min_distance_bins)
    if all_peak_idx.size == 0:
        fallback_idx = int(np.argmax(lead_signal))
        all_peak_idx = np.array([fallback_idx], dtype=int)
        all_prominences = np.array([max(float(lead_signal[fallback_idx] - local_baseline), 0.0)], dtype=float)
    else:
        all_prominences = peak_prominences(lead_signal, all_peak_idx)[0]

    all_peak_heights = np.asarray(lead_signal[all_peak_idx], dtype=float)
    passes_prominence = all_prominences >= prominence_floor
    if np.any(passes_prominence):
        gamma_peak_idx = all_peak_idx[passes_prominence]
        gamma_prominences = all_prominences[passes_prominence]
        gamma_peak_heights = all_peak_heights[passes_prominence]
    else:
        fallback_pick = int(np.argmax(all_peak_heights))
        gamma_peak_idx = np.array([int(all_peak_idx[fallback_pick])], dtype=int)
        gamma_prominences = np.array([float(all_prominences[fallback_pick])], dtype=float)
        gamma_peak_heights = np.array([float(all_peak_heights[fallback_pick])], dtype=float)

    keep_height_floor = local_baseline + float(keep_height_frac) * max(local_peak - local_baseline, 0.0)
    keep_height_mask = gamma_peak_heights >= keep_height_floor
    keep_prom_mask = gamma_prominences >= 0.5 * float(gamma_prominences.max())
    keep_mask = keep_height_mask | keep_prom_mask
    kept_idx = gamma_peak_idx[keep_mask] if np.any(keep_mask) else gamma_peak_idx

    return [
        _anchor_record(
            coarse_event_idx,
            anchor_rank,
            onset_time_s,
            float(lead_time[int(local_idx)]),
            float(lead_signal[int(local_idx)]),
            int(all_peak_idx.size),
            int(passes_prominence.sum()),
            int(len(kept_idx)),
            onset_threshold,
            search_stop_s,
            prominence_floor,
            keep_height_floor,
        )
        for anchor_rank, local_idx in enumerate(np.sort(kept_idx))
    ]


def detect_nested_gamma_anchors(
    coarse_epochs: pd.DataFrame,
    highres: HighResTraces,
    *,
    coarse_rel_height: float = 0.20,
    bin_ms: float = 1.0,
    search_ms: float = 120.0,
    search_to_epoch_end: bool = True,
    min_distance_ms: float = 40.0,
    prominence_frac: float = 0.08,
    prominence_abs_floor: float = 0.10,
    keep_height_frac: float = 0.50,
) -> pd.DataFrame:
    """Detect nested gamma anchors for every coarse burst epoch."""
    baseline = float(np.quantile(highres.population_rate_hz, 0.25))
    records = []
    for row in coarse_epochs.itertuples(index=False):
        anchors = choose_nested_gamma_anchors(
            highres.time_centers_s,
            highres.population_rate_hz,
            coarse_start_s=row.start_time_s,
            coarse_end_s=row.end_time_s,
            coarse_event_idx=row.event_idx,
            coarse_rel_height=coarse_rel_height,
            bin_ms=bin_ms,
            search_ms=search_ms,
            search_to_epoch_end=search_to_epoch_end,
            min_distance_ms=min_distance_ms,
            prominence_frac=prominence_frac,
            prominence_abs_floor=prominence_abs_floor,
            keep_height_frac=keep_height_frac,
            baseline_floor=baseline,
        )
        for anchor in anchors:
            records.append(
                {
                    "coarse_event_idx": int(row.event_idx),
                    "start_time_s": float(row.start_time_s),
                    "end_time_s": float(row.end_time_s),
                    "coarse_peak_time_s": float(row.coarse_peak_time_s),
                    "coarse_peak_hz": float(row.coarse_peak_hz),
                    "subpeak_count": int(row.subpeak_count),
                    **anchor,
                }
            )
    return pd.DataFrame(records)


def align_highres_to_anchors(
    highres: HighResTraces,
    anchors: pd.DataFrame,
    *,
    pre_ms: float = 20.0,
    post_ms: float = 40.0,
    bin_ms: float = 1.0,
) -> AlignedBurstEvents:
    """Align high-resolution traces to gamma anchor times."""
    pre_bins = int(round(float(pre_ms) / float(bin_ms)))
    post_bins = int(round(float(post_ms) / float(bin_ms)))
    relative_time_ms = (np.arange(-pre_bins, post_bins + 1) * float(bin_ms)).astype(float)

    population_windows = []
    aligned_rate = []
    aligned_spikes = []
    valid_rows = []
    for row in anchors.itertuples(index=False):
        anchor_idx = int(np.argmin(np.abs(highres.time_centers_s - row.anchor_time_s)))
        window_start = anchor_idx - pre_bins
        window_stop = anchor_idx + post_bins + 1
        if window_start < 0 or window_stop > len(highres.time_centers_s):
            continue
        population_windows.append(highres.population_rate_hz[window_start:window_stop])
        aligned_rate.append(highres.per_electrode_rate_hz[:, window_start:window_stop])
        aligned_spikes.append(highres.spike_presence[:, window_start:window_stop])
        valid_rows.append(row._asdict())

    return AlignedBurstEvents(
        relative_time_ms=relative_time_ms,
        population_windows=np.asarray(population_windows, dtype=float),
        aligned_rate=np.asarray(aligned_rate, dtype=float),
        aligned_spikes=np.asarray(aligned_spikes, dtype=bool),
        valid_anchors=pd.DataFrame(valid_rows),
        electrodes=np.asarray(highres.electrodes, dtype=int),
    )


def summarize_aligned_electrode_rates(
    aligned: AlignedBurstEvents,
    *,
    peak_pre_ms: float = 5.0,
    peak_post_ms: float = 10.0,
    rebound_start_ms: float = 15.0,
    rebound_stop_ms: float = 25.0,
) -> pd.DataFrame:
    """Summarize average aligned rate per electrode and return peak-ordered table."""
    if aligned.aligned_rate.size == 0:
        return pd.DataFrame()
    mean_rate = aligned.aligned_rate.mean(axis=0)
    rel = aligned.relative_time_ms
    peak_mask = (rel >= peak_pre_ms * -1.0) & (rel <= peak_post_ms)
    rebound_mask = (rel >= rebound_start_ms) & (rel <= rebound_stop_ms)
    anchor_idx = int(np.argmin(np.abs(rel)))
    return (
        pd.DataFrame(
            {
                "electrode": aligned.electrodes.astype(int),
                "window_mean_hz": mean_rate.mean(axis=1),
                "peak_window_mean_hz": mean_rate[:, peak_mask].mean(axis=1),
                "rebound_window_mean_hz": mean_rate[:, rebound_mask].mean(axis=1),
                "anchor_rate_hz": mean_rate[:, anchor_idx],
                "max_rate_hz": mean_rate.max(axis=1),
            }
        )
        .sort_values(
            ["peak_window_mean_hz", "rebound_window_mean_hz", "max_rate_hz", "electrode"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )


def compute_electrode_peak_time_map(
    aligned: AlignedBurstEvents,
    layout: dict | pd.DataFrame,
    *,
    window_idx: int | None = None,
    peak_search_start_ms: float | None = None,
    peak_search_stop_ms: float | None = None,
    min_peak_rate_hz: float = 0.0,
) -> pd.DataFrame:
    """Map each aligned electrode to its peak-time latency and HD-MEA position."""
    rel = np.asarray(aligned.relative_time_ms, dtype=float)
    if rel.size == 0:
        raise ValueError("aligned.relative_time_ms is empty.")
    if aligned.aligned_rate.size == 0:
        return pd.DataFrame(columns=["electrode", "x", "y", "peak_time_ms", "peak_rate_hz", "valid"])

    start_ms = float(rel[0]) if peak_search_start_ms is None else float(peak_search_start_ms)
    stop_ms = float(rel[-1]) if peak_search_stop_ms is None else float(peak_search_stop_ms)
    if stop_ms < start_ms:
        raise ValueError("peak_search_stop_ms must be greater than or equal to peak_search_start_ms.")
    search_mask = (rel >= start_ms) & (rel <= stop_ms)
    if not np.any(search_mask):
        raise ValueError("The peak-search window does not overlap aligned.relative_time_ms.")
    search_times = rel[search_mask]

    if window_idx is None:
        rate = np.asarray(aligned.aligned_rate, dtype=float).mean(axis=0)
    else:
        window_idx = int(window_idx)
        if window_idx < 0 or window_idx >= aligned.aligned_rate.shape[0]:
            raise IndexError(f"window_idx {window_idx} is out of range for {aligned.aligned_rate.shape[0]} aligned events.")
        rate = np.asarray(aligned.aligned_rate[window_idx], dtype=float)

    if rate.ndim != 2 or rate.shape[0] != len(aligned.electrodes) or rate.shape[1] != rel.size:
        raise ValueError("aligned rate data must have shape electrodes x relative_time.")

    search_rate = rate[:, search_mask]
    finite_search = np.where(np.isfinite(search_rate), search_rate, -np.inf)
    peak_idx = np.argmax(finite_search, axis=1)
    peak_rate = finite_search[np.arange(finite_search.shape[0]), peak_idx]
    peak_time = search_times[peak_idx]
    valid = np.isfinite(peak_rate) & (peak_rate > float(min_peak_rate_hz))

    peak_df = pd.DataFrame(
        {
            "electrode": np.asarray(aligned.electrodes, dtype=int),
            "peak_time_ms": peak_time.astype(float),
            "peak_rate_hz": peak_rate.astype(float),
            "valid": valid.astype(bool),
        }
    )
    layout_df = pd.DataFrame(layout).drop_duplicates("electrode")
    layout_df = layout_df[["electrode", "x", "y"]].copy()
    layout_df["electrode"] = layout_df["electrode"].astype(int)
    out = layout_df.merge(peak_df, on="electrode", how="inner")
    out = out.sort_values(["valid", "peak_time_ms", "electrode"], ascending=[False, True, True]).reset_index(drop=True)

    if window_idx is not None:
        out["window_idx"] = window_idx
        anchors = pd.DataFrame(aligned.valid_anchors)
        if 0 <= window_idx < len(anchors):
            for col, value in anchors.iloc[window_idx].items():
                if col not in out.columns and np.isscalar(value):
                    out[col] = value
    return out


def order_aligned_rate_by_summary(aligned: AlignedBurstEvents, summary: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return electrodes and mean aligned rate ordered by summary table."""
    mean_rate = aligned.aligned_rate.mean(axis=0)
    ordered_electrodes = summary["electrode"].to_numpy(dtype=int)
    order_idx = np.array([np.flatnonzero(aligned.electrodes == int(el))[0] for el in ordered_electrodes], dtype=int)
    return ordered_electrodes, mean_rate[order_idx]


def order_aligned_rate_by_x(aligned: AlignedBurstEvents, layout: dict | pd.DataFrame, summary: pd.DataFrame | None = None):
    """Return x/y layout table and mean aligned rate ordered by array x-position."""
    layout_df = pd.DataFrame(layout).drop_duplicates("electrode")
    layout_df = layout_df[layout_df["electrode"].isin(aligned.electrodes)].copy()
    if summary is not None and not summary.empty:
        layout_df = layout_df.merge(summary, on="electrode", how="left")
        sort_cols = ["x", "y", "peak_window_mean_hz", "electrode"]
        ascending = [True, True, False, True]
    else:
        sort_cols = ["x", "y", "electrode"]
        ascending = [True, True, True]
    layout_df = layout_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    order_idx = np.array([np.flatnonzero(aligned.electrodes == int(el))[0] for el in layout_df["electrode"]], dtype=int)
    return layout_df, aligned.aligned_rate.mean(axis=0)[order_idx]


def _fallback_anchor(time_centers, population_rate_hz, idx: int, event_idx: int, prominence_floor: float) -> dict[str, Any]:
    fallback_time_s = float(time_centers[idx])
    fallback_height_hz = float(population_rate_hz[idx])
    return _anchor_record(
        event_idx,
        0,
        fallback_time_s,
        fallback_time_s,
        fallback_height_hz,
        0,
        0,
        1,
        fallback_height_hz,
        fallback_time_s,
        prominence_floor,
        fallback_height_hz,
    )


def _anchor_record(
    event_idx: int,
    rank: int,
    onset_time_s: float,
    anchor_time_s: float,
    anchor_height_hz: float,
    n_local_peaks: int,
    n_passing_prominence: int,
    n_kept_gamma_peaks: int,
    onset_threshold_hz: float,
    search_stop_s: float,
    prominence_floor_hz: float,
    keep_height_floor_hz: float,
) -> dict[str, Any]:
    return {
        "event_idx": int(event_idx),
        "gamma_peak_rank": int(rank),
        "onset_time_s": float(onset_time_s),
        "anchor_time_s": float(anchor_time_s),
        "anchor_height_hz": float(anchor_height_hz),
        "gamma_anchor_delay_ms": (float(anchor_time_s) - float(onset_time_s)) * 1000.0,
        "n_local_peaks": int(n_local_peaks),
        "n_passing_prominence": int(n_passing_prominence),
        "n_kept_gamma_peaks": int(n_kept_gamma_peaks),
        "onset_threshold_hz": float(onset_threshold_hz),
        "search_stop_s": float(search_stop_s),
        "prominence_floor_hz": float(prominence_floor_hz),
        "keep_height_floor_hz": float(keep_height_floor_hz),
    }


def _boolean_intervals(mask) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    intervals: list[tuple[int, int]] = []
    in_interval = False
    start_idx = 0
    for idx, value in enumerate(mask):
        if value and not in_interval:
            start_idx = idx
            in_interval = True
        elif not value and in_interval:
            intervals.append((start_idx, idx - 1))
            in_interval = False
    if in_interval:
        intervals.append((start_idx, len(mask) - 1))
    return intervals


def _merge_intervals(intervals: list[tuple[int, int]], *, max_gap_bins: int) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for start_idx, stop_idx in intervals[1:]:
        prev_start, prev_stop = merged[-1]
        if start_idx - prev_stop - 1 <= int(max_gap_bins):
            merged[-1] = (prev_start, max(prev_stop, stop_idx))
        else:
            merged.append((start_idx, stop_idx))
    return merged


def _empty_network_burst_epochs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_idx",
            "start_idx",
            "end_idx",
            "start_time_s",
            "end_time_s",
            "coarse_peak_idx",
            "coarse_peak_time_s",
            "coarse_peak_hz",
            "anchor_time_s",
            "anchor_participation_fraction",
            "peak_participation_fraction",
            "peak_active_electrodes",
            "participating_electrodes",
            "total_spikes",
            "duration_ms",
            "subpeak_count",
        ]
    )


def _empty_high_activity_epochs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_idx",
            "start_idx",
            "end_idx",
            "start_time_s",
            "end_time_s",
            "peak_idx",
            "peak_time_s",
            "peak_mean_ifr_hz",
            "duration_ms",
            "threshold_hz",
            "baseline_hz",
        ]
    )
