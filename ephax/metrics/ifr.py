from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy import stats
from scipy.stats import gaussian_kde

@dataclass
class IFRConfig:
    """Configuration for IFR plotting and peak-analysis helpers."""

    log_scale: bool = True
    n_components: Optional[int] = None
    random_state: int = 0
    overlay_gmm: bool = True
    show_kde: bool = False
    show_peaks: bool = False
    ts_bins: int = 50
    time_grid_hz: float = 100.0
    max_time_points: int = 5000


@dataclass
class IFRPeaks:
    values: np.ndarray
    kde_x: np.ndarray
    kde_y: np.ndarray
    peaks_x: np.ndarray
    peaks_y: np.ndarray
    peaks_hz: np.ndarray


@dataclass
class IFRTimeSeriesPanel:
    recording_index: int
    start_time: float
    end_time: float
    electrodes: np.ndarray
    time_points: np.ndarray
    heatmap: np.ndarray
    histogram_values: np.ndarray
    log_scale: bool


def calculate_ifr(spikes_data, selected_electrodes, start_time=None, end_time=None):
    """Compute instantaneous firing rate per electrode as a step function."""
    electrode_spikes = {int(el): [] for el in selected_electrodes}
    for t, el in zip(spikes_data["time"], spikes_data["electrode"]):
        if int(el) in electrode_spikes:
            electrode_spikes[int(el)].append(float(t))

    for el in electrode_spikes:
        electrode_spikes[el] = np.asarray(electrode_spikes[el], dtype=float)

    if start_time is None:
        start_time = float(np.min(spikes_data["time"])) if len(spikes_data["time"]) else 0.0
    if end_time is None:
        end_time = float(np.max(spikes_data["time"])) if len(spikes_data["time"]) else 0.0

    def _calc(times):
        if times.size < 2:
            return np.array([start_time, end_time], dtype=float), np.array([0.0, 0.0], dtype=float)
        ifr_times = [start_time]
        ifr_values = [0.0]
        first = times[0]
        ifr_times.append(first)
        ifr_values.append(0.0)
        for i in range(times.size - 1):
            a = times[i]
            b = times[i + 1]
            interval = max(1e-12, b - a)
            val = 1.0 / interval
            ifr_times.extend([a, b])
            ifr_values.extend([val, val])
        last = times[-1]
        ifr_times.append(last)
        ifr_values.append(0.0)
        ifr_times.append(end_time)
        ifr_values.append(0.0)
        return np.asarray(ifr_times, dtype=float), np.asarray(ifr_values, dtype=float)

    ifr_data = {}
    total_firing = {}
    all_ifr_values = []
    for el, times in electrode_spikes.items():
        sel = (times >= float(start_time)) & (times <= float(end_time))
        times_sel = times[sel]
        if times_sel.size > 0:
            t_arr, v_arr = _calc(times_sel)
            ifr_data[el] = (t_arr, v_arr)
            duration = max(1e-12, float(end_time) - float(start_time))
            total_firing[el] = float(times_sel.size) / duration
            all_ifr_values.extend(v_arr.tolist())
    return ifr_data, total_firing, np.asarray(all_ifr_values, dtype=float)


def prepare_ifr_timeseries_panel(
    spikes_data: dict,
    selected_electrodes: Iterable[int],
    start_time: float,
    end_time: float,
    *,
    recording_index: int = 0,
    log_scale: bool = True,
    time_grid_hz: float = 100.0,
    max_time_points: int = 5000,
) -> IFRTimeSeriesPanel | None:
    """Prepare one recording's IFR heatmap and histogram values for plotting."""
    selected_electrodes = list(selected_electrodes)
    if len(selected_electrodes) == 0:
        return None

    ifr_data, _, all_ifr_values = calculate_ifr(spikes_data, selected_electrodes, start_time, end_time)
    duration = max(0.0, float(end_time) - float(start_time))
    target_hz = float(time_grid_hz) if time_grid_hz and time_grid_hz > 0 else 100.0
    n_points = int(max(1, min(duration * target_hz, float(max_time_points))))
    time_points = np.linspace(float(start_time), float(end_time), n_points, dtype=np.float32)

    valid_electrodes = []
    rows = []
    for electrode in selected_electrodes:
        electrode = int(electrode)
        if electrode not in ifr_data:
            continue

        times, values = ifr_data[electrode]
        if log_scale:
            values = np.where(values == 0, 1e-3, values)
            values = np.log10(values)
            values = np.where(np.isinf(values), -3, values)
        rows.append(np.interp(time_points, times.astype(np.float32), values).astype(np.float32))
        valid_electrodes.append(electrode)

    if len(rows) == 0:
        return None

    histogram_values = all_ifr_values.copy()
    if log_scale:
        histogram_values = histogram_values[histogram_values > 1e-3]
        histogram_values = np.log10(histogram_values)

    return IFRTimeSeriesPanel(
        recording_index=int(recording_index),
        start_time=float(start_time),
        end_time=float(end_time),
        electrodes=np.asarray(valid_electrodes, dtype=int),
        time_points=time_points,
        heatmap=np.vstack(rows).astype(np.float32),
        histogram_values=np.asarray(histogram_values, dtype=float),
        log_scale=bool(log_scale),
    )


def prepare_ifr_timeseries_panels(
    spikes_data_list: Iterable[dict],
    start_times: Iterable[float],
    end_times: Iterable[float],
    selected_electrodes_per_recording: Iterable[Iterable[int]],
    *,
    log_scale: bool = True,
    time_grid_hz: float = 100.0,
    max_time_points: int = 5000,
) -> list[IFRTimeSeriesPanel]:
    """Prepare IFR time-series panels for all recordings with valid IFR data."""
    panels = []
    for idx, (spikes_data, start_time, end_time, selected_electrodes) in enumerate(
        zip(spikes_data_list, start_times, end_times, selected_electrodes_per_recording)
    ):
        panel = prepare_ifr_timeseries_panel(
            spikes_data,
            selected_electrodes,
            start_time,
            end_time,
            recording_index=idx,
            log_scale=log_scale,
            time_grid_hz=time_grid_hz,
            max_time_points=max_time_points,
        )
        if panel is not None:
            panels.append(panel)
    return panels


def ifr_peaks(
    spikes_data_list: Iterable[dict],
    start_times: Iterable[float],
    end_times: Iterable[float],
    log_scale: bool = True,
    selected_refs_per_recording: Optional[Iterable[Iterable[int]]] = None,
) -> IFRPeaks:
    """Compute IFR values across datasets and estimate peaks via KDE."""
    spikes_data_list = list(spikes_data_list)
    start_times = list(start_times)
    end_times = list(end_times)
    assert len(spikes_data_list) == len(start_times) == len(end_times), "Mismatch in lengths of input lists."

    values = []
    for idx, (spikes_data, start_time, end_time) in enumerate(zip(spikes_data_list, start_times, end_times)):
        if selected_refs_per_recording is None:
            raise ValueError("selected_refs_per_recording must be provided for ifr_peaks")
        try:
            selected_electrodes = list(selected_refs_per_recording[idx])
        except Exception:
            selected_electrodes = []
        _, _, ifr_vals = calculate_ifr(spikes_data, selected_electrodes, start_time, end_time)
        values.extend(ifr_vals)

    values = np.asarray(values)
    if log_scale:
        values = values[values > 1e-3]
        values = np.log10(values)

    kde = gaussian_kde(values)
    x_grid = np.linspace(values.min(), values.max(), 1000)
    kde_values = kde.evaluate(x_grid)
    peaks_indices, _ = find_peaks(kde_values)
    peaks_x = x_grid[peaks_indices]
    peaks_y = kde_values[peaks_indices]
    peaks_hz = 10 ** peaks_x if log_scale else peaks_x

    return IFRPeaks(values=values, kde_x=x_grid, kde_y=kde_values, peaks_x=peaks_x, peaks_y=peaks_y, peaks_hz=peaks_hz)


def build_activity_state_kde_for_div(
    div: int,
    *,
    recording_spec,
    load_recording,
    select_refs,
    config,
) -> dict[str, object]:
    """Aggregate activity-state IFR samples and KDE peaks for one DIV."""
    from .burst import (
        binned_kde_peak_summary,
        build_single_recording_burst_state,
    )

    chunks = {"low_activity": [], "high_activity": [], "burst": []}
    rows = []
    recording_duration_s = float(config.end_sec - config.start_sec)
    for well in config.wells:
        spec = recording_spec(well, div)
        ds = load_recording(spec)
        rec = ds.recordings[0]
        refs = select_refs(ds)
        state = build_single_recording_burst_state(rec, refs, config=config)
        state_values = _extract_activity_state_ifr_values(
            rec,
            refs,
            state["high_activity_epochs"],
            state["burst_epochs"],
            max_hz=config.ifr_max_hz,
        )
        for state_name, values in state_values.items():
            values = np.asarray(values, dtype=float)
            chunks[state_name].append(values[np.isfinite(values) & (values > 0) & (values <= config.ifr_max_hz)])
        rows.append(
            {
                "well": int(well),
                "div": int(div),
                "n_refs": int(len(refs)),
                "n_high_activity_periods": int(len(state["high_activity_epochs"])),
                "n_bursts": int(len(state["burst_epochs"])),
                "n_high_activity_periods_per_min": float(len(state["high_activity_epochs"])) / (recording_duration_s / 60.0),
                "n_bursts_per_min": float(len(state["burst_epochs"])) / (recording_duration_s / 60.0),
                "high_activity_threshold_hz": float(state["high_activity_info"]["threshold_hz"]),
            }
        )

    values_by_state = {}
    for state_name, state_chunks in chunks.items():
        nonempty = [chunk for chunk in state_chunks if chunk.size]
        values_by_state[state_name] = np.concatenate(nonempty).astype(float) if nonempty else np.array([], dtype=float)
    kde_inputs = {
        "high_activity": values_by_state["high_activity"],
        "burst": values_by_state["burst"],
    }
    kde_results = {
        state_name: binned_kde_peak_summary(
            values,
            log_bins=True,
            n_bins=260,
            grid_size=8192,
            prominence_fraction=0.012,
            distance_fraction=0.006,
            bandwidth_scale=0.22,
            max_hz=config.ifr_max_hz,
        )
        for state_name, values in kde_inputs.items()
    }
    return {
        "values": kde_inputs,
        "kde_results": kde_results,
        "summary": pd.DataFrame(rows),
        "source_label": f"stimRemovalNull wells 0-5, DIV {int(div)}",
    }


def _extract_activity_state_ifr_values(recording, selected_refs, high_epochs, burst_epochs, *, max_hz):
    from .burst import extract_activity_state_ifr

    return extract_activity_state_ifr(recording, selected_refs, high_epochs, burst_epochs, max_hz=max_hz)


def bootstrap_mean_ci(values, *, reps=2000, seed=0):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(np.mean(arr))
    if arr.size == 1 or int(reps) <= 0:
        return mean, mean, mean, int(arr.size)
    rng = np.random.default_rng(int(seed))
    boot = rng.choice(arr, size=(int(reps), arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi), int(arr.size)


def paired_neighbor_pvalues(df, metric, *, divs):
    rows = []
    if metric not in df:
        return pd.DataFrame(rows)
    for left_div, right_div in zip(divs[:-1], divs[1:]):
        left = df.loc[df["div"] == int(left_div), ["well", metric]].rename(columns={metric: "left_value"})
        right = df.loc[df["div"] == int(right_div), ["well", metric]].rename(columns={metric: "right_value"})
        pair = left.merge(right, on="well", how="inner")
        pair = pair[np.isfinite(pair["left_value"]) & np.isfinite(pair["right_value"])]
        if len(pair) < 3:
            p_value = np.nan
            test = "paired_ttest_insufficient_n"
        else:
            diff = pair["right_value"].to_numpy(dtype=float) - pair["left_value"].to_numpy(dtype=float)
            if np.allclose(diff, 0):
                p_value = 1.0
                test = "paired_ttest_all_zero_diff"
            else:
                p_value = float(stats.ttest_rel(pair["right_value"], pair["left_value"], nan_policy="omit").pvalue)
                test = "paired_ttest"
        rows.append(
            {
                "metric": metric,
                "left_div": int(left_div),
                "right_div": int(right_div),
                "n_pairs": int(len(pair)),
                "p_value": p_value,
                "test": test,
            }
        )
    return pd.DataFrame(rows)


def p_to_stars(p_value):
    if not np.isfinite(p_value) or p_value >= 0.05:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def activity_wave_supplementary_tables(
    data: dict[str, object],
    *,
    supp_divs,
    bootstrap_reps: int = 2000,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge activity-state and wave summaries into supplementary statistics tables."""
    from .waves import add_wave_direction_balance_columns, wave_fit_diagnostics

    activity = pd.concat(
        [data["activity_stats"][div]["per_recording"] for div in supp_divs],
        ignore_index=True,
    )
    wave_rows = []
    aggregate_rows = []
    for div in supp_divs:
        wave = data.get("wave", {}).get(div)
        if not wave:
            continue
        per_rec = add_wave_direction_balance_columns(wave["summary"])
        wave_rows.append(per_rec)
        if wave.get("result") is not None:
            aggregate = wave_fit_diagnostics(wave["result"])
            aggregate["div"] = int(div)
            aggregate["wave_peak_search_start_ms"] = wave.get("aggregate_wave_peak_search_start_ms", np.nan)
            aggregate["wave_peak_search_stop_ms"] = wave.get("aggregate_wave_peak_search_stop_ms", np.nan)
            aggregate_rows.append(aggregate)
    wave_recording = pd.concat(wave_rows, ignore_index=True) if wave_rows else pd.DataFrame()
    wave_aggregate = pd.DataFrame(aggregate_rows)
    merge_cols = [
        "div",
        "well",
        "n_events_used",
        "n_events_left_to_right",
        "n_events_right_to_left",
        "direction_total_events",
        "direction_balance_signed",
        "direction_imbalance_abs",
        "implied_speed_um_per_ms",
        "bootstrap_speed_ci_low_um_per_ms",
        "bootstrap_speed_ci_high_um_per_ms",
        "wave_peak_search_start_ms",
        "wave_peak_search_stop_ms",
        "n_bins_retained",
    ]
    if not wave_recording.empty:
        merged = activity.merge(wave_recording[[c for c in merge_cols if c in wave_recording]], on=["div", "well"], how="left")
    else:
        merged = activity.copy()

    summary_rows = []
    significance_rows = []
    metrics = [
        "n_high_activity_periods_per_min",
        "n_bursts_per_min",
        "high_activity_duration_median_ms",
        "high_activity_time_fraction",
        "burst_duration_median_ms",
        "burst_time_fraction",
        "burst_ibi_median_s",
        "burst_peak_participation_median",
        "burst_peak_active_electrodes_median",
        "direction_imbalance_abs",
        "implied_speed_um_per_ms",
        "n_bins_retained",
        "n_events_used",
    ]
    for metric in metrics:
        significance_rows.append(paired_neighbor_pvalues(merged, metric, divs=list(supp_divs)))
        for div, group in merged.groupby("div"):
            if metric not in group:
                continue
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            mean, ci_low, ci_high, n_values = bootstrap_mean_ci(
                values,
                reps=bootstrap_reps,
                seed=int(random_seed) + int(div) * 1009 + sum(ord(ch) for ch in metric),
            )
            summary_rows.append(
                {
                    "div": int(div),
                    "metric": metric,
                    "n_recordings": int(n_values),
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "median": float(np.median(values)),
                    "iqr_low": float(np.percentile(values, 25)),
                    "iqr_high": float(np.percentile(values, 75)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    significance = pd.concat([df for df in significance_rows if not df.empty], ignore_index=True) if significance_rows else pd.DataFrame()
    if not significance.empty:
        significance["stars"] = significance["p_value"].map(p_to_stars)
    return merged, pd.DataFrame(summary_rows), wave_aggregate, significance
