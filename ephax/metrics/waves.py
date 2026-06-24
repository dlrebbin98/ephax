from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .burst import (
    AlignedBurstEvents,
    align_highres_to_anchors,
    assign_max_population_ifr_burst_anchors,
    build_highres_traces,
    build_participation_activity_state,
    build_population_ifr,
    detect_high_activity_epochs,
    detect_participation_burst_epochs,
)


@dataclass
class WaveAnalysisResult:
    event_direction: object
    trace: object
    peaks: object
    bin_summary: object
    fit_summary: object
    heatmap: object
    bootstrap_speeds: np.ndarray


def wave_cache_key(
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
) -> str:
    """Return a stable cache key for eventwise wave-analysis settings."""
    return (
        f"xbin{float(x_bin_um):g}_"
        f"peak{float(peak_search_start_ms):g}to{float(peak_search_stop_ms):g}_"
        f"smooth{float(trace_smooth_sigma_ms):g}_"
        f"bin{float(bin_ms):g}_"
        f"minelec{int(min_electrodes_per_bin)}_"
        f"minevents{int(min_events_per_bin)}_"
        f"boot{int(bootstrap_reps)}_seed{int(random_seed)}"
    ).replace(".", "p").replace("-", "m")


def wave_cache_dir(root: str | Path, recording_id: str, **settings) -> Path:
    """Return the cache directory for one recording and wave-analysis settings."""
    return Path(root) / str(recording_id) / wave_cache_key(**settings)


def save_wave_result_cache(result: WaveAnalysisResult, cache_dir: str | Path) -> Path:
    """Write a WaveAnalysisResult to a directory of CSV cache files."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result.event_direction.to_csv(cache_dir / "event_direction.csv", index=False)
    result.trace.to_csv(cache_dir / "trace.csv", index=False)
    result.peaks.to_csv(cache_dir / "peaks.csv", index=False)
    result.bin_summary.to_csv(cache_dir / "bin_summary.csv", index=False)
    result.fit_summary.to_csv(cache_dir / "fit_summary.csv", index=False)
    pd.DataFrame(result.heatmap).to_csv(cache_dir / "heatmap.csv", index=True, index_label="time_ms")
    pd.DataFrame({"speed_um_per_ms": result.bootstrap_speeds}).to_csv(cache_dir / "bootstrap_speeds.csv", index=False)
    return cache_dir


def load_wave_result_cache(cache_dir: str | Path) -> WaveAnalysisResult | None:
    """Load a WaveAnalysisResult from a CSV cache directory if all files exist."""
    cache_dir = Path(cache_dir)
    required = [
        "event_direction.csv",
        "trace.csv",
        "peaks.csv",
        "bin_summary.csv",
        "fit_summary.csv",
        "heatmap.csv",
        "bootstrap_speeds.csv",
    ]
    if not all((cache_dir / name).exists() for name in required):
        return None
    heatmap = pd.read_csv(cache_dir / "heatmap.csv", index_col=0)
    heatmap.index = heatmap.index.astype(float)
    heatmap.columns = heatmap.columns.astype(float)
    boot_df = pd.read_csv(cache_dir / "bootstrap_speeds.csv")
    bootstrap_speeds = boot_df["speed_um_per_ms"].to_numpy(dtype=float)
    fit_summary = pd.read_csv(cache_dir / "fit_summary.csv")
    if "bootstrap_speed_mean_um_per_ms" not in fit_summary.columns:
        fit_summary["bootstrap_speed_mean_um_per_ms"] = float(np.mean(bootstrap_speeds)) if bootstrap_speeds.size else np.nan
    return WaveAnalysisResult(
        event_direction=pd.read_csv(cache_dir / "event_direction.csv"),
        trace=pd.read_csv(cache_dir / "trace.csv"),
        peaks=pd.read_csv(cache_dir / "peaks.csv"),
        bin_summary=pd.read_csv(cache_dir / "bin_summary.csv"),
        fit_summary=fit_summary,
        heatmap=heatmap,
        bootstrap_speeds=bootstrap_speeds,
    )


def add_wave_direction_balance_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Add signed and absolute direction-balance summaries to wave rows."""
    df = summary_df.copy()
    if df.empty:
        return df
    left = df.get("n_events_left_to_right", pd.Series(0, index=df.index)).astype(float)
    right = df.get("n_events_right_to_left", pd.Series(0, index=df.index)).astype(float)
    total = left + right
    df["direction_total_events"] = total.astype(int)
    df["direction_balance_signed"] = np.where(total > 0, (left - right) / total, np.nan)
    df["direction_imbalance_abs"] = np.abs(df["direction_balance_signed"])
    return df


def wave_fit_diagnostics(result: WaveAnalysisResult) -> dict[str, float]:
    """Return wave fit summary plus correlation and weighted R-squared diagnostics."""
    fit = result.fit_summary.iloc[0].to_dict()
    bins = result.bin_summary.copy()
    if bins.empty:
        fit.update({"fit_r_value": np.nan, "fit_r_squared": np.nan, "fit_weighted_r_squared": np.nan})
        return fit
    x = bins["origin_x_um"].to_numpy(dtype=float)
    y = bins["mean_peak_time_ms"].to_numpy(dtype=float)
    slope = float(fit.get("slope_ms_per_um", np.nan))
    intercept = float(fit.get("intercept_ms", np.nan))
    pred = slope * x + intercept
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(pred)
    if valid.sum() < 2:
        fit.update({"fit_r_value": np.nan, "fit_r_squared": np.nan, "fit_weighted_r_squared": np.nan})
        return fit

    fit["fit_r_value"] = float(np.corrcoef(x[valid], y[valid])[0, 1])
    fit["fit_r_squared"] = float(fit["fit_r_value"] ** 2)
    if "n_observations" in bins:
        weights = np.sqrt(np.maximum(1.0, bins.loc[valid, "n_observations"].to_numpy(dtype=float)))
    else:
        weights = np.ones(valid.sum(), dtype=float)
    y_valid = y[valid]
    pred_valid = pred[valid]
    y_bar = float(np.average(y_valid, weights=weights))
    ss_res = float(np.sum(weights * (y_valid - pred_valid) ** 2))
    ss_tot = float(np.sum(weights * (y_valid - y_bar) ** 2))
    fit["fit_weighted_r_squared"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return fit


def summarize_wave_recording_result(
    recording_id: str,
    well: int,
    div: int,
    result: WaveAnalysisResult,
    settings: dict[str, object],
    *,
    n_refs: int | float | None = None,
    n_bursts: int | float | None = None,
) -> dict[str, object]:
    """Create one compact per-recording summary row from a wave result."""
    fit = result.fit_summary.iloc[0]

    def _float_or_nan(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    row = {
        "recording_id": str(recording_id),
        "well": int(well),
        "div": int(div),
        "wave_peak_search_start_ms": _float_or_nan(settings.get("peak_search_start_ms", np.nan)),
        "wave_peak_search_stop_ms": _float_or_nan(settings.get("peak_search_stop_ms", np.nan)),
        "n_events_used": int(fit.get("n_events_used", 0)),
        "n_events_left_to_right": int(fit.get("n_events_left_to_right", 0)),
        "n_events_right_to_left": int(fit.get("n_events_right_to_left", 0)),
        "implied_speed_um_per_ms": _float_or_nan(fit.get("implied_speed_um_per_ms", np.nan)),
        "bootstrap_speed_mean_um_per_ms": _float_or_nan(fit.get("bootstrap_speed_mean_um_per_ms", np.nan)),
        "bootstrap_speed_median_um_per_ms": _float_or_nan(fit.get("bootstrap_speed_median_um_per_ms", np.nan)),
        "bootstrap_speed_ci_low_um_per_ms": _float_or_nan(fit.get("bootstrap_speed_ci_low_um_per_ms", np.nan)),
        "bootstrap_speed_ci_high_um_per_ms": _float_or_nan(fit.get("bootstrap_speed_ci_high_um_per_ms", np.nan)),
        "n_bins_retained": _float_or_nan(fit.get("n_bins_retained", np.nan)),
    }
    if n_refs is not None:
        row["n_refs"] = int(n_refs) if np.isfinite(float(n_refs)) else np.nan
    if n_bursts is not None:
        row["n_bursts_or_cached_events"] = int(n_bursts) if np.isfinite(float(n_bursts)) else np.nan
    return row


def load_cached_wave_for_recording(
    cache_root: str | Path,
    recording_id: str,
    setting_candidates: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> tuple[WaveAnalysisResult | None, dict[str, object] | None]:
    """Load the first compatible cached wave result for one recording."""
    for settings in setting_candidates:
        cached = load_wave_result_cache(wave_cache_dir(cache_root, recording_id, **settings))
        if cached is not None:
            return cached, dict(settings)
    return None, None


def load_cached_aggregate_wave(
    cache_root: str | Path,
    aggregate_id: str,
    setting_candidates: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> tuple[WaveAnalysisResult | None, dict[str, object] | None]:
    """Load the first compatible cached aggregate wave result."""
    return load_cached_wave_for_recording(cache_root, aggregate_id, setting_candidates)


def wave_settings_from_config(
    *,
    x_bin_um: float,
    peak_search_start_ms: float,
    peak_search_stop_ms: float,
    trace_smooth_sigma_ms: float,
    bin_ms: float,
    min_electrodes_per_bin: int,
    min_events_per_bin: int,
    bootstrap_reps: int,
    random_seed: int,
) -> dict[str, object]:
    """Build a wave-analysis settings dictionary compatible with existing caches."""
    return {
        "x_bin_um": x_bin_um,
        "peak_search_start_ms": peak_search_start_ms,
        "peak_search_stop_ms": peak_search_stop_ms,
        "trace_smooth_sigma_ms": trace_smooth_sigma_ms,
        "bin_ms": bin_ms,
        "min_electrodes_per_bin": min_electrodes_per_bin,
        "min_events_per_bin": min_events_per_bin,
        "bootstrap_reps": bootstrap_reps,
        "random_seed": random_seed,
    }


def supplementary_wave_setting_candidates(settings: dict[str, object]) -> list[dict[str, object]]:
    """Return current wave settings followed by the historical peak-window fallback."""
    current = dict(settings)
    fallback = dict(current)
    fallback["peak_search_start_ms"] = -15.0
    fallback["peak_search_stop_ms"] = 20.0
    return [current, fallback]


def load_cached_wave_for_div(
    div: int,
    *,
    recording_spec,
    wells,
    wave_cache_root: str | Path,
    settings: dict[str, object],
) -> dict[str, object]:
    """Load cached per-recording and aggregate wave summaries for one DIV."""
    candidates = supplementary_wave_setting_candidates(settings)
    aggregate_id = f"aggregate_stim_removal_null_wells{'-'.join(map(str, wells))}_DIVs{int(div)}"
    aggregate_result, aggregate_settings = load_cached_aggregate_wave(wave_cache_root, aggregate_id, candidates)

    rows = []
    per_recording = []
    for well in wells:
        spec = recording_spec(well, div)
        cached, used_settings = load_cached_wave_for_recording(wave_cache_root, spec["recording_id"], candidates)
        if cached is None or used_settings is None:
            continue
        rows.append(summarize_wave_recording_result(spec["recording_id"], int(well), int(div), cached, used_settings))
        per_recording.append(cached.fit_summary.assign(recording_id=spec["recording_id"]))

    return {
        "result": aggregate_result,
        "summary": pd.DataFrame(rows),
        "per_recording_fit": pd.concat(per_recording, ignore_index=True) if per_recording else pd.DataFrame(),
        "aggregate_wave_peak_search_start_ms": np.nan
        if aggregate_settings is None
        else float(aggregate_settings["peak_search_start_ms"]),
        "aggregate_wave_peak_search_stop_ms": np.nan
        if aggregate_settings is None
        else float(aggregate_settings["peak_search_stop_ms"]),
    }


def build_aggregate_wave_for_div(
    div: int,
    *,
    recording_spec,
    load_recording,
    select_refs,
    wells,
    wave_cache_root: str | Path,
    settings: dict[str, object],
    burst_config,
    force_wave: bool = False,
    force_aggregate: bool = False,
) -> dict[str, object]:
    """Compute or load per-recording and aggregate wave results for one DIV."""
    results_by_recording = []
    rows = []
    for well in wells:
        spec = recording_spec(well, div)
        recording_id = spec["recording_id"]
        cached = None if force_wave else load_wave_result_cache(wave_cache_dir(wave_cache_root, recording_id, **settings))
        if cached is None:
            ds = load_recording(spec)
            rec = ds.recordings[0]
            refs = select_refs(ds)
            aligned, intermediates = build_burst_peak_aligned_events(
                rec,
                refs,
                ifr_grid_hz=burst_config.ifr_grid_hz,
                smooth_sigma_sec=burst_config.smooth_sigma_sec,
                highres_bin_ms=burst_config.highres_bin_ms,
                highres_smooth_sigma_ms=burst_config.highres_smooth_sigma_ms,
                network_bin_ms=burst_config.network_bin_ms,
                high_activity_mad_scale=burst_config.high_activity_mad_scale,
                high_activity_min_duration_ms=burst_config.high_activity_min_duration_ms,
                high_activity_max_gap_bins=burst_config.high_activity_max_gap_bins,
                network_min_participation_fraction=burst_config.network_min_participation_fraction,
                network_min_duration_ms=burst_config.network_min_duration_ms,
            )
            if len(pd.DataFrame(aligned.valid_anchors)) == 0:
                print(f"Skipping {recording_id}: no valid aligned burst anchors")
                continue
            result = compute_or_load_wave_result(
                recording_id,
                aligned,
                rec.layout,
                cache_root=wave_cache_root,
                force=force_wave,
                **settings,
            )
            n_refs = int(len(refs))
            n_bursts = int(len(intermediates["burst_epochs"]))
        else:
            result = cached
            n_refs = np.nan
            n_bursts = int(result.fit_summary.iloc[0]["n_events_used"])

        results_by_recording.append((recording_id, result))
        rows.append(
            summarize_wave_recording_result(
                recording_id,
                int(well),
                int(div),
                result,
                settings,
                n_refs=n_refs,
                n_bursts=n_bursts,
            )
        )

    if not results_by_recording:
        raise ValueError(f"No aggregate wave results were available for DIV {div}.")

    aggregate_id = f"aggregate_stim_removal_null_wells{'-'.join(map(str, wells))}_DIVs{int(div)}"
    aggregate_cache_dir = wave_cache_dir(wave_cache_root, aggregate_id, **settings)
    aggregate_result = None if force_aggregate else load_wave_result_cache(aggregate_cache_dir)
    if aggregate_result is None:
        aggregate_result, per_recording_fit = aggregate_wave_results(
            results_by_recording,
            x_bin_um=float(settings["x_bin_um"]),
            min_events_per_bin=int(settings["min_events_per_bin"]),
            bootstrap_reps=int(settings["bootstrap_reps"]),
            random_seed=int(settings["random_seed"]),
        )
        save_wave_result_cache(aggregate_result, aggregate_cache_dir)
    else:
        per_recording_fit = pd.concat(
            [result.fit_summary.assign(recording_id=recording_id) for recording_id, result in results_by_recording],
            ignore_index=True,
        )

    return {
        "result": aggregate_result,
        "summary": pd.DataFrame(rows),
        "per_recording_fit": per_recording_fit,
    }


def compute_or_load_wave_result(
    recording_id: str,
    aligned: AlignedBurstEvents,
    layout: dict | pd.DataFrame,
    *,
    cache_root: str | Path | None = None,
    force: bool = False,
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
    """Compute or load one eventwise wave-analysis result."""
    settings = {
        "x_bin_um": x_bin_um,
        "peak_search_start_ms": peak_search_start_ms,
        "peak_search_stop_ms": peak_search_stop_ms,
        "trace_smooth_sigma_ms": trace_smooth_sigma_ms,
        "bin_ms": bin_ms,
        "min_electrodes_per_bin": min_electrodes_per_bin,
        "min_events_per_bin": min_events_per_bin,
        "bootstrap_reps": bootstrap_reps,
        "random_seed": random_seed,
    }
    cache_dir = wave_cache_dir(cache_root, recording_id, **settings) if cache_root is not None else None
    if cache_dir is not None and not force:
        cached = load_wave_result_cache(cache_dir)
        if cached is not None:
            return cached

    result = analyze_eventwise_waves(aligned, layout, **settings)
    if cache_dir is not None:
        save_wave_result_cache(result, cache_dir)
    return result


def build_burst_peak_aligned_events(
    recording,
    selected_refs,
    *,
    ifr_grid_hz: float = 50.0,
    smooth_sigma_sec: float = 0.15,
    highres_bin_ms: float = 1.0,
    highres_smooth_sigma_ms: float = 3.0,
    network_bin_ms: float = 10.0,
    high_activity_mad_scale: float = 3.0,
    high_activity_min_duration_ms: float = 30.0,
    high_activity_max_gap_bins: int = 0,
    network_min_participation_fraction: float = 0.05,
    network_min_duration_ms: float = 20.0,
    align_pre_ms: float = 20.0,
    align_post_ms: float = 40.0,
) -> tuple[AlignedBurstEvents, dict[str, object]]:
    """Build burst-peak-centered aligned events for wave analysis."""
    population = build_population_ifr(recording, selected_refs, grid_hz=ifr_grid_hz, smooth_sigma_sec=smooth_sigma_sec)
    highres = build_highres_traces(
        recording,
        selected_refs,
        bin_ms=highres_bin_ms,
        smooth_sigma_ms=highres_smooth_sigma_ms,
    )
    high_epochs, high_info = detect_high_activity_epochs(
        population.time_grid,
        population.mean_ifr_smooth,
        mad_scale=high_activity_mad_scale,
        min_duration_ms=high_activity_min_duration_ms,
        max_gap_bins=high_activity_max_gap_bins,
    )
    participation = build_participation_activity_state(highres, aggregation_ms=network_bin_ms)
    burst_epochs = detect_participation_burst_epochs(
        participation,
        high_epochs,
        min_participation_fraction=network_min_participation_fraction,
        min_duration_ms=network_min_duration_ms,
    )
    burst_epochs = assign_max_population_ifr_burst_anchors(highres, burst_epochs)
    anchors = burst_epochs.copy()
    if not anchors.empty:
        anchors["coarse_event_idx"] = anchors["event_idx"].astype(int)
        anchors["gamma_peak_rank"] = 0
        anchors["onset_time_s"] = anchors["start_time_s"].astype(float)
        anchors["anchor_height_hz"] = anchors["anchor_population_rate_hz"].astype(float)
        anchors["gamma_anchor_delay_ms"] = (anchors["anchor_time_s"] - anchors["onset_time_s"]) * 1000.0
        anchors["anchor_type"] = "max_highres_population_ifr"
    aligned = align_highres_to_anchors(
        highres,
        anchors,
        pre_ms=align_pre_ms,
        post_ms=align_post_ms,
        bin_ms=highres_bin_ms,
    )
    return aligned, {
        "population": population,
        "highres": highres,
        "high_activity_epochs": high_epochs,
        "high_activity_info": high_info,
        "participation_activity": participation,
        "burst_epochs": burst_epochs,
    }


def aggregate_wave_results(
    results_by_recording: list[tuple[str, WaveAnalysisResult]],
    *,
    x_bin_um: float = 300.0,
    min_events_per_bin: int = 5,
    bootstrap_reps: int = 2000,
    random_seed: int = 0,
) -> tuple[WaveAnalysisResult, pd.DataFrame]:
    """Pool eventwise wave results across recordings and refit aggregate speed."""
    if not results_by_recording:
        raise ValueError("No wave results were provided for aggregation.")

    peak_frames = []
    trace_frames = []
    direction_frames = []
    fit_frames = []
    event_offset = 0
    for recording_id, result in results_by_recording:
        peaks = result.peaks.copy()
        trace = result.trace.copy()
        directions = result.event_direction.copy()
        fit = result.fit_summary.copy()

        for df in (peaks, trace, directions):
            df["recording_id"] = str(recording_id)
            if "window_idx" in df:
                df["local_window_idx"] = df["window_idx"].astype(int)
                df["window_idx"] = df["local_window_idx"] + int(event_offset)
        fit["recording_id"] = str(recording_id)

        peak_frames.append(peaks)
        trace_frames.append(trace)
        direction_frames.append(directions)
        fit_frames.append(fit)
        event_offset += int(result.peaks["window_idx"].nunique())

    combined_peaks = pd.concat(peak_frames, ignore_index=True)
    combined_trace = pd.concat(trace_frames, ignore_index=True)
    combined_directions = pd.concat(direction_frames, ignore_index=True)
    per_recording_fit = pd.concat(fit_frames, ignore_index=True)

    combined_bin_summary = summarize_wave_peaks(combined_peaks, min_events_per_bin=min_events_per_bin)
    if len(combined_bin_summary) < 3:
        raise ValueError("Too few aggregate x-bins survived the minimum-event filter for wave fitting.")
    aggregate_array_width_um = float(per_recording_fit["array_width_um"].median())
    combined_fit_summary, combined_boot_speeds = fit_wave_speed(
        combined_peaks,
        combined_bin_summary,
        x_bin_um=x_bin_um,
        array_width_um=aggregate_array_width_um,
        event_direction=combined_directions,
        bootstrap_reps=bootstrap_reps,
        min_events_per_bin=min_events_per_bin,
        random_seed=random_seed,
    )
    combined_heatmap = (
        combined_trace.groupby(["time_ms", "origin_x_um"], as_index=False)["rate_hz"]
        .mean()
        .sort_values(["time_ms", "origin_x_um"])
        .pivot(index="time_ms", columns="origin_x_um", values="rate_hz")
        .sort_index()
        .sort_index(axis=1)
    )
    return (
        WaveAnalysisResult(
            event_direction=combined_directions,
            trace=combined_trace,
            peaks=combined_peaks,
            bin_summary=combined_bin_summary,
            fit_summary=combined_fit_summary,
            heatmap=combined_heatmap,
            bootstrap_speeds=combined_boot_speeds,
        ),
        per_recording_fit,
    )


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
