from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import pandas as pd

import phase_waves
from ephax.lfp_wavefront_cache import WavefrontCacheConfig, require_wavefront_caches
from ephax.metrics.burst import (
    assign_max_population_ifr_burst_anchors,
    build_highres_traces,
    build_participation_activity_state,
    build_population_ifr,
    detect_high_activity_epochs,
    detect_participation_burst_epochs,
)
from ephax.prep import Recording
from phase_waves import (
    analyze_excitable_phase_front,
    append_wavefront_rows,
    bandpass_downsample,
    combine_excitable_phase_calibrations,
    estimate_excitable_phase,
    extract_phasors,
    initialize_wavefront_cache,
    inspect_file,
    load_chunk,
    make_local_phase_neighborhoods,
    read_wavefront_cache,
)


@dataclass(frozen=True)
class EventFrontBuildConfig:
    raw_file: Path
    cache_dir: Path
    cache: WavefrontCacheConfig
    wells: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    profile_calibration_events: int | None = None
    profile_scoring_events: int = 200
    min_amp: float = 0.0
    top_start: int = 0
    top_stop: int = 1000
    ifr_grid_hz: float = 50.0
    smooth_sigma_sec: float = 0.15
    high_activity_mad_scale: float = 3.0
    high_activity_min_duration_ms: float = 30.0
    high_activity_max_gap_bins: int = 0
    highres_bin_ms: float = 1.0
    highres_smooth_sigma_ms: float = 3.0
    network_bin_ms: float = 10.0
    network_min_participation_fraction: float = 0.05
    network_min_duration_ms: float = 20.0
    phase_pad_s: float = 0.25
    search_pre_ms: float = 20.0
    search_post_ms: float = 40.0
    phase_bins: int = 36
    pseudocount: float = 1.0
    theta_override_rad: float | None = None
    min_electrodes: int = 50
    match_max_ms: float = 8.0
    radius_mm: float = 0.30
    min_neighbors: int = 8
    max_radius_mm: float = 0.45
    max_neighbors: int = 24
    min_geometry_ratio: float = 0.10
    max_residual_ms: float = 4.0
    min_arrival_span_ms: float = 1.0
    min_distance_span_mm: float = 0.20
    max_speed_mm_per_s: float = 500.0
    min_arrival_amplitude: float = 0.0
    min_arrival_amplitude_percentile: float = 0.0
    max_channels_per_block: int = 256
    radial_center_grid_n: int = 15


def dataset_for_well(well: int) -> str:
    return f"data{int(well):04d}"


def load_data_store_spikes(file_path: str | Path, dataset: str, *, min_amp: float = 0.0):
    raw_path, rec_group_path = phase_waves._resolve_dataset_paths(dataset, file_path)
    with h5py.File(file_path, "r") as h5:
        rec_group = h5[rec_group_path]
        settings = rec_group["settings"]
        mapping_obj = settings["mapping"]
        if getattr(mapping_obj, "dtype", None) is not None and mapping_obj.dtype.fields is not None:
            mapping_arr = mapping_obj[:]
            layout = {name: np.asarray(mapping_arr[name]) for name in ("channel", "electrode", "x", "y")}
        else:
            layout = {name: np.asarray(mapping_obj[name])[:] for name in ("channel", "electrode", "x", "y")}
        sf = float(settings["sampling"][0])
        spikes_arr = rec_group["spikes"][:]
        frame_nos = h5[raw_path].parent.get("frame_nos")
        first_frame = (
            int(frame_nos[0])
            if frame_nos is not None and frame_nos.shape[0]
            else int(np.nanmin(spikes_arr["frameno"]))
            if spikes_arr.size
            else 0
        )
        spikes = {
            "time": (np.asarray(spikes_arr["frameno"], dtype=float) - first_frame) / sf,
            "channel": np.asarray(spikes_arr["channel"]),
            "amplitude": np.asarray(spikes_arr["amplitude"]),
        }
        channel_to_electrode = {int(ch): int(el) for ch, el in zip(layout["channel"], layout["electrode"])}
        spikes["electrode"] = np.asarray([channel_to_electrode.get(int(ch), -1) for ch in spikes["channel"]])
        keep = (
            (spikes["electrode"] >= 0)
            & (spikes["time"] >= 0)
            & (np.abs(spikes["amplitude"]) >= float(min_amp))
        )
        spikes = {key: np.asarray(value)[keep] for key, value in spikes.items()}
    return spikes, layout, sf


def ensure_wavefront_caches(config: EventFrontBuildConfig, *, build_missing: bool = True) -> dict[int, Path]:
    try:
        return require_wavefront_caches(config.cache, config.cache_dir, config.wells, require_all=True)
    except FileNotFoundError:
        if not build_missing:
            raise
    status = build_wavefront_caches(config)
    missing = status.loc[status["cache_state"].isin(["missing", "incompatible", "partial"])]
    if not missing.empty:
        print("Processed cache status:")
        print(status.to_string(index=False))
    return require_wavefront_caches(config.cache, config.cache_dir, config.wells, require_all=True)


def load_completed_wavefront_cache(config: EventFrontBuildConfig, well: int):
    status, cache = wavefront_cache_state(config, well)
    if status != "complete":
        return None
    return cache


def build_wavefront_caches(config: EventFrontBuildConfig) -> pd.DataFrame:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    contexts = {}
    rows = []
    for well in config.wells:
        context = _detect_context(config, int(well))
        contexts[int(well)] = context
        if context["bursts"].empty:
            rows.append(_status_row(config, int(well), context, "no_bursts"))
            continue
        state, _ = wavefront_cache_state(config, int(well), context)
        rows.append(_status_row(config, int(well), context, state))
    status = pd.DataFrame(rows)
    print(status.round(3).to_string(index=False))
    for well in config.wells:
        context = contexts[int(well)]
        if context["bursts"].empty:
            continue
        _process_well(config, int(well), context)
    return status


def wavefront_cache_state(config: EventFrontBuildConfig, well: int, context: dict | None = None):
    path = config.cache.wavefront_path(config.cache_dir, well)
    if not path.exists():
        return "missing", None
    cache = read_wavefront_cache(path)
    stored = json.loads(cache.get("wavefront_calibration", {}).get("config", "{}"))
    if not _summary_config_matches(config, stored):
        return "incompatible", None
    if context is not None and stored != _front_config(config, well, context):
        return "incompatible", None
    with h5py.File(path, "r") as h5:
        complete = bool(h5.attrs.get("complete", False))
    return ("complete" if complete else "partial"), cache


def _status_row(config: EventFrontBuildConfig, well: int, context: dict, state: str) -> dict:
    if context["bursts"].empty:
        return {
            "well": well,
            "cache_state": state,
            "detected_bursts": 0,
            "calibration_events": 0,
            "scored_events": 0,
            "retained_lfp_s": 0.0,
            "burst_detection_s": context["detection_elapsed_s"],
        }
    feature_path = config.cache.phasor_feature_path(config.cache_dir, well)
    cached_segments = 0
    if feature_path.exists():
        metadata = phase_waves.read_phasor_feature_metadata(feature_path)
        if metadata["config"] == _feature_config(config, well, context):
            cached_segments = len(metadata["segments"])
    return {
        "well": well,
        "cache_state": state,
        "detected_bursts": len(context["bursts"]),
        "calibration_events": len(context["calibration_events"]),
        "scored_events": len(context["scoring_events"]),
        "overlap_events": context["overlap_count"],
        "cached_segments": cached_segments,
        "pending_segments": len(context["merged_feature_intervals"]) - cached_segments,
        "retained_lfp_s": context["retained_lfp_s"],
        "burst_detection_s": context["detection_elapsed_s"],
    }


def _detect_context(config: EventFrontBuildConfig, well: int) -> dict:
    started = perf_counter()
    dataset = dataset_for_well(well)
    info = inspect_file(config.raw_file, dataset)
    spikes, layout, sf = load_data_store_spikes(config.raw_file, dataset, min_amp=config.min_amp)
    stop_s = min(
        info["raw_shape"][1] / info["fs_raw"],
        float(np.nanmax(spikes["time"])) if len(spikes["time"]) else np.inf,
    )
    recording = Recording(spikes=spikes, layout=layout, start_time=0.0, end_time=stop_s, sf=float(sf))
    refs = recording.refs_top(start=config.top_start, stop=config.top_stop)
    population = build_population_ifr(
        recording, refs, grid_hz=config.ifr_grid_hz, smooth_sigma_sec=config.smooth_sigma_sec
    )
    high_epochs, _ = detect_high_activity_epochs(
        population.time_grid,
        population.mean_ifr_smooth,
        mad_scale=config.high_activity_mad_scale,
        min_duration_ms=config.high_activity_min_duration_ms,
        max_gap_bins=config.high_activity_max_gap_bins,
    )
    highres = build_highres_traces(
        recording, refs, bin_ms=config.highres_bin_ms, smooth_sigma_ms=config.highres_smooth_sigma_ms
    )
    network = build_participation_activity_state(highres, aggregation_ms=config.network_bin_ms)
    bursts = detect_participation_burst_epochs(
        network,
        high_epochs,
        min_participation_fraction=config.network_min_participation_fraction,
        min_duration_ms=config.network_min_duration_ms,
    )
    bursts = assign_max_population_ifr_burst_anchors(highres, bursts).sort_values("anchor_time_s").reset_index(drop=True)
    if bursts.empty:
        return {"well": well, "bursts": bursts, "spikes": spikes, "stop_s": stop_s, "detection_elapsed_s": perf_counter() - started}
    selected = phase_waves.select_representative_event_indices(
        len(bursts),
        n_calibration=config.profile_calibration_events,
        n_scoring=config.profile_scoring_events,
    )
    calibration_events = bursts.iloc[selected["calibration_indices"]].copy()
    scoring_events = bursts.iloc[selected["scoring_indices"]].copy()
    calibration_intervals = [
        (
            max(0.0, float(row.start_time_s) - config.phase_pad_s),
            min(stop_s, float(row.end_time_s) + config.phase_pad_s),
        )
        for row in calibration_events.itertuples(index=False)
    ]
    scoring_intervals = [
        (
            max(0.0, float(row.anchor_time_s) - config.search_pre_ms / 1000.0 - config.phase_pad_s),
            min(stop_s, float(row.anchor_time_s) + config.search_post_ms / 1000.0 + config.phase_pad_s),
        )
        for row in scoring_events.itertuples(index=False)
    ]
    merged = phase_waves.merge_time_intervals(calibration_intervals + scoring_intervals)
    return {
        "well": well,
        "bursts": bursts,
        "spikes": spikes,
        "stop_s": stop_s,
        "calibration_events": calibration_events,
        "scoring_events": scoring_events,
        "calibration_intervals": calibration_intervals,
        "scoring_intervals": scoring_intervals,
        "merged_feature_intervals": merged,
        "overlap_count": int(selected["overlap_count"]),
        "retained_lfp_s": float(sum(stop - start for start, stop in merged)),
        "detection_elapsed_s": perf_counter() - started,
    }


def _front_config(config: EventFrontBuildConfig, well: int, context: dict) -> dict:
    result = _summary_config(config)
    result.update(
        {
            "file": str(config.raw_file),
            "dataset": dataset_for_well(well),
            "well": int(well),
            "div": int(config.cache.div),
            "calibration_event_ids": _event_ids(context["calibration_events"]),
            "scoring_event_ids": _event_ids(context["scoring_events"]),
            "calibration_scoring_overlap": int(context["overlap_count"]),
        }
    )
    return result


def _summary_config(config: EventFrontBuildConfig) -> dict:
    result = {
        "profile": config.cache.profile,
        "band_low": float(config.cache.band_low_hz),
        "band_high": float(config.cache.band_high_hz),
        "fs_ds": float(config.cache.fs_hz),
        "filter_pad_s": float(config.phase_pad_s),
        "theta_override_rad": None if config.theta_override_rad is None else float(config.theta_override_rad),
        "phase_bins": int(config.phase_bins),
        "pseudocount": float(config.pseudocount),
        "min_electrodes": int(config.min_electrodes),
        "match_max_ms": float(config.match_max_ms),
        "radius_mm": float(config.radius_mm),
        "min_neighbors": int(config.min_neighbors),
        "max_radius_mm": float(config.max_radius_mm),
        "max_neighbors": int(config.max_neighbors),
        "min_geometry_ratio": float(config.min_geometry_ratio),
        "max_residual_ms": float(config.max_residual_ms),
        "min_arrival_span_ms": float(config.min_arrival_span_ms),
        "min_distance_span_mm": float(config.min_distance_span_mm),
        "max_speed_mm_per_s": float(config.max_speed_mm_per_s),
    }
    if float(config.min_arrival_amplitude) > 0.0:
        result["min_arrival_amplitude"] = float(config.min_arrival_amplitude)
    if float(config.min_arrival_amplitude_percentile) > 0.0:
        result["min_arrival_amplitude_percentile"] = float(config.min_arrival_amplitude_percentile)
    return result


def _summary_config_matches(config: EventFrontBuildConfig, stored: dict) -> bool:
    expected = _summary_config(config)
    return all(stored.get(key) == value for key, value in expected.items())


def _feature_config(config: EventFrontBuildConfig, well: int, context: dict) -> dict:
    return {
        "file": str(config.raw_file),
        "dataset": dataset_for_well(well),
        "well": int(well),
        "div": int(config.cache.div),
        "profile": config.cache.profile,
        "calibration_event_ids": _event_ids(context["calibration_events"]),
        "scoring_event_ids": _event_ids(context["scoring_events"]),
        "band_low": float(config.cache.band_low_hz),
        "band_high": float(config.cache.band_high_hz),
        "fs_ds": float(config.cache.fs_hz),
        "filter_pad_s": float(config.phase_pad_s),
        "intervals": [[float(start), float(stop)] for start, stop in context["merged_feature_intervals"]],
    }


def _event_ids(frame: pd.DataFrame) -> list[int]:
    return [int(value) for value in frame["event_idx"].to_numpy(int)]


def _load_lfp_interval(config: EventFrontBuildConfig, well: int, interval):
    dataset = dataset_for_well(well)
    info = inspect_file(config.raw_file, dataset)
    start_s, stop_s = map(float, interval)
    start_frame = int(round(start_s * info["fs_raw"]))
    n_frames = max(1, int(round((stop_s - start_s) * info["fs_raw"])))
    raw, coords, fs_raw, mapping = load_chunk(config.raw_file, dataset, start_frame, n_frames)
    band = bandpass_downsample(
        raw,
        fs_raw,
        config.cache.band_low_hz,
        config.cache.band_high_hz,
        config.cache.fs_hz,
        max_channels_per_block=config.max_channels_per_block,
    )
    time = start_s + np.arange(band.shape[0]) / config.cache.fs_hz
    return band, time, coords, mapping


def _build_feature_cache(config: EventFrontBuildConfig, well: int, context: dict) -> Path:
    path = config.cache.phasor_feature_path(config.cache_dir, well)
    feature_config = _feature_config(config, well, context)
    merged = context["merged_feature_intervals"]
    if phase_waves.phasor_feature_cache_matches(path, feature_config):
        print(f"Well {well}: using phasor feature cache {path.name}")
        return path
    completed_segments = 0
    initialized = False
    if path.exists():
        metadata = phase_waves.read_phasor_feature_metadata(path)
        if metadata["config"] == feature_config:
            completed_segments = len(metadata["segments"])
            initialized = True
            print(f"Well {well}: resuming phasor features after {completed_segments}/{len(merged)} segments")
    print(f"Well {well}: retained LFP context {context['retained_lfp_s']:.2f}s across {len(merged)} merged segments")
    for segment_idx, interval in enumerate(merged[completed_segments:], start=completed_segments + 1):
        print(f"Well {well}: feature segment {segment_idx}/{len(merged)}")
        band, time, coords, mapping = _load_lfp_interval(config, well, interval)
        phasors, amplitude = extract_phasors(band)
        electrodes = np.asarray(mapping["electrode"], dtype=int)
        if not initialized:
            phase_waves.initialize_phasor_feature_cache(path, coords, electrodes, config=feature_config)
            initialized = True
        phase_waves.append_phasor_feature_segment(path, interval[0], time, phasors, amplitude)
    if not initialized:
        raise ValueError(f"Well {well}: no feature intervals were available")
    phase_waves.finalize_phasor_feature_cache(path)
    return path


def _process_well(config: EventFrontBuildConfig, well: int, context: dict):
    state, cache = wavefront_cache_state(config, well, context)
    if state == "complete":
        print(f"Well {well}: completed profile cache already exists; skipped")
        return cache
    path = config.cache.wavefront_path(config.cache_dir, well)
    feature_path = _build_feature_cache(config, well, context)
    feature_meta = phase_waves.read_phasor_feature_metadata(feature_path)
    coords_static = np.asarray(feature_meta["coords_mm"], dtype=float)
    neighborhoods = make_local_phase_neighborhoods(
        coords_static,
        radius_mm=config.radius_mm,
        min_neighbors=config.min_neighbors,
        max_radius_mm=config.max_radius_mm,
        max_neighbors=config.max_neighbors,
        min_geometry_ratio=config.min_geometry_ratio,
    )
    if state != "partial":
        _initialize_wavefront_result(config, well, context, path)
        cache = read_wavefront_cache(path)
    calibration = cache["wavefront_calibration"]
    existing_ids = set(map(int, cache.get("wavefront_events", {}).get("event_idx", [])))
    pending = [
        (int(row.event_idx), row)
        for row in context["scoring_events"].itertuples(index=False)
        if int(row.event_idx) not in existing_ids
    ]
    for pending_idx, (event_idx, row) in enumerate(pending, start=1):
        print(f"Well {well}: scoring event {pending_idx}/{len(pending)}")
        _append_event_front(config, well, context, feature_path, path, neighborhoods, calibration, event_idx, row)
    with h5py.File(path, "a") as h5:
        h5.attrs["complete"] = True
        h5.attrs["completed_event_count"] = int(len(context["scoring_events"]))
    print(f"Well {well}: completed {path.name}")
    return read_wavefront_cache(path)


def _initialize_wavefront_result(config: EventFrontBuildConfig, well: int, context: dict, path: Path) -> None:
    calibration_parts = []
    feature_path = config.cache.phasor_feature_path(config.cache_dir, well)
    for idx, interval in enumerate(context["calibration_intervals"], start=1):
        print(f"Well {well}: calibration slice {idx}/{len(context['calibration_intervals'])}")
        phasors, time, _, electrodes = phase_waves.read_phasor_feature_interval(feature_path, *interval)
        calibration_parts.append(
            estimate_excitable_phase(
                phasors,
                time,
                electrodes,
                np.asarray(context["spikes"]["time"], dtype=float),
                np.asarray(context["spikes"]["electrode"], dtype=int),
                n_bins=config.phase_bins,
                pseudocount=config.pseudocount,
            )
        )
    calibration = combine_excitable_phase_calibrations(calibration_parts, pseudocount=config.pseudocount)
    thetas = np.asarray([part["theta_excitable_rad"] for part in calibration_parts], dtype=float)
    resultant = float(abs(np.mean(np.exp(1j * thetas))))
    calibration["calibration_theta_resultant"] = resultant
    calibration["calibration_theta_circular_std_rad"] = float(
        np.sqrt(max(0.0, -2.0 * np.log(max(resultant, np.finfo(float).tiny))))
    )
    calibration["calibration_interval_count"] = int(len(calibration_parts))
    calibration["calibration_stable"] = int(resultant >= 0.50)
    if config.theta_override_rad is not None:
        calibration["theta_excitable_rad"] = float(config.theta_override_rad)
    initialize_wavefront_cache(path, calibration, config=_front_config(config, well, context))
    with h5py.File(path, "a") as h5:
        h5.attrs["complete"] = False


def _append_event_front(
    config: EventFrontBuildConfig,
    well: int,
    context: dict,
    feature_path: Path,
    cache_path: Path,
    neighborhoods,
    calibration,
    event_idx: int,
    row,
) -> None:
    anchor_s = float(row.anchor_time_s)
    read_interval = (
        max(0.0, anchor_s - config.search_pre_ms / 1000.0 - config.phase_pad_s),
        min(context["stop_s"], anchor_s + config.search_post_ms / 1000.0 + config.phase_pad_s),
    )
    phasors, time, coords, electrodes, amplitude = phase_waves.read_phasor_feature_interval(
        feature_path, *read_interval, include_amplitude=True
    )
    analysis = analyze_excitable_phase_front(
        phasors,
        time,
        coords,
        neighborhoods,
        amplitude=amplitude,
        min_arrival_amplitude=config.min_arrival_amplitude,
        min_arrival_amplitude_percentile=config.min_arrival_amplitude_percentile,
        theta_excitable_rad=float(calibration["theta_excitable_rad"]),
        anchor_time_s=anchor_s,
        frequency_hz=0.5 * (config.cache.band_low_hz + config.cache.band_high_hz),
        match_max_ms=config.match_max_ms,
        min_electrodes=config.min_electrodes,
        search_pre_ms=config.search_pre_ms,
        search_post_ms=config.search_post_ms,
        min_neighbors=config.min_neighbors,
        max_residual_ms=config.max_residual_ms,
        min_arrival_span_ms=config.min_arrival_span_ms,
        min_distance_span_mm=config.min_distance_span_mm,
        max_speed_mm_per_s=config.max_speed_mm_per_s,
        radial_center_grid_n=config.radial_center_grid_n,
    )
    front = analysis["front"]
    local = analysis["local"]
    append_wavefront_rows(
        cache_path,
        "wavefront_events",
        [{
            "event_idx": int(event_idx),
            "anchor_time_s": anchor_s,
            "front_time_s": float(front["front_time_s"]),
            "front_cycle_offset": int(front["cycle_offset"]),
            "front_n_electrodes": int(front["n_electrodes"]),
            "front_valid": int(front["valid"]),
            **analysis["planar"],
            **analysis["radial"],
        }],
    )
    electrodes = np.asarray(electrodes, dtype=int)
    arrival = np.asarray(front["arrival_time_s"], dtype=float)
    append_wavefront_rows(
        cache_path,
        "wavefront_local",
        {
            "event_idx": np.full(electrodes.size, event_idx),
            "anchor_time_s": np.full(electrodes.size, anchor_s),
            "electrode": electrodes,
            "x_mm": coords[:, 0],
            "y_mm": coords[:, 1],
            "arrival_time_s": arrival,
            "arrival_rel_anchor_ms": (arrival - anchor_s) * 1000.0,
            "arrival_amplitude": local["arrival_amplitude"],
            "arrival_amplitude_threshold": local["arrival_amplitude_threshold"],
            "velocity_x_mm_per_s": local["velocity_x_mm_per_s"],
            "velocity_y_mm_per_s": local["velocity_y_mm_per_s"],
            "speed_mm_per_s": local["speed_mm_per_s"],
            "residual_rms_ms": local["residual_rms_ms"],
            "arrival_span_ms": local["arrival_span_ms"],
            "distance_span_mm": local["distance_span_mm"],
            "arrival_gradient_norm_s_per_mm": local["arrival_gradient_norm_s_per_mm"],
            "speed_censored": local["speed_censored"].astype(np.uint8),
            "n_good": local["n_good"],
            "valid": local["valid"].astype(np.uint8),
        },
    )
