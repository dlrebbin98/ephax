from __future__ import annotations

from typing import Iterable

import numpy as np

from ..models import BinnedSeries, CofiringDistanceResult, FRDistanceResult


def avg_rate_vs_distance(
    recordings: Iterable,
    refs_per_recording: Iterable[Iterable[int]],
    *,
    log: bool = False,
    min_distance: float = 50,
    max_distance: float = 3500,
) -> FRDistanceResult:
    """Compute firing rate for electrode/ref pairs and bin by distance."""
    all_rates: list[np.ndarray] = []
    all_dists: list[np.ndarray] = []

    for rec, refs in zip(recordings, refs_per_recording):
        prepared = _prepare_recording_distance_inputs(rec, refs)
        if prepared is None:
            continue
        duration = float(rec.end_time - rec.start_time)
        if duration <= 0:
            continue

        electrodes, coords, refs_arr, ref_coords, spike_electrodes, _spike_times = prepared
        active_electrodes, counts = np.unique(spike_electrodes, return_counts=True)
        active_idx = _indices_for_values(electrodes, active_electrodes)
        valid_active = active_idx >= 0
        if not np.any(valid_active):
            continue
        active_electrodes = active_electrodes[valid_active]
        active_coords = coords[active_idx[valid_active]]
        active_rates = counts[valid_active].astype(float) / duration

        dists = _pairwise_distances(ref_coords, active_coords)
        keep = active_electrodes[None, :] != refs_arr[:, None]
        if not np.any(keep):
            continue
        all_dists.append(dists[keep].astype(float, copy=False))
        all_rates.append(np.broadcast_to(active_rates, dists.shape)[keep].astype(float, copy=False))

    distances = np.concatenate(all_dists) if all_dists else np.array([], dtype=float)
    rates = np.concatenate(all_rates) if all_rates else np.array([], dtype=float)
    return _fr_result(distances, rates, log=log, min_distance=min_distance, max_distance=max_distance)


def cofiring_avg_vs_distance(
    recordings: Iterable,
    refs_per_recording: Iterable[Iterable[int]],
    *,
    plusminus_ms: float = 2.0,
    log: bool = False,
    min_distance: float = 50,
    max_distance: float = 3500,
) -> CofiringDistanceResult:
    """Compute co-firing probability for electrode/ref pairs and bin by distance."""
    all_props: list[np.ndarray] = []
    all_dists: list[np.ndarray] = []
    pm_sec = float(plusminus_ms) / 1000.0

    for rec, refs in zip(recordings, refs_per_recording):
        prepared = _prepare_recording_distance_inputs(rec, refs)
        if prepared is None:
            continue

        electrodes, coords, refs_arr, ref_coords, spike_electrodes, spike_times = prepared
        active_electrodes = np.unique(spike_electrodes)
        active_idx = _indices_for_values(electrodes, active_electrodes)
        valid_active = active_idx >= 0
        if not np.any(valid_active):
            continue
        active_electrodes = active_electrodes[valid_active]
        active_coords = coords[active_idx[valid_active]]
        times_by_electrode = _sorted_spike_times_by_electrode(spike_electrodes, spike_times)

        for ref, ref_coord in zip(refs_arr, ref_coords):
            ref_times = times_by_electrode.get(int(ref), np.array([], dtype=float))
            if ref_times.size == 0:
                target_electrodes = active_electrodes[active_electrodes != int(ref)]
                if target_electrodes.size == 0:
                    continue
                target_idx = _indices_for_values(electrodes, target_electrodes)
                valid_targets = target_idx >= 0
                if not np.any(valid_targets):
                    continue
                target_coords = coords[target_idx[valid_targets]]
                all_dists.append(_pairwise_distances(ref_coord[None, :], target_coords).reshape(-1))
                all_props.append(np.zeros(target_coords.shape[0], dtype=float))
                continue

            starts = ref_times - pm_sec
            ends = ref_times + pm_sec
            props: list[float] = []
            target_coords_list: list[np.ndarray] = []
            for electrode, coord in zip(active_electrodes, active_coords):
                if int(electrode) == int(ref):
                    continue
                target_times = times_by_electrode.get(int(electrode), np.array([], dtype=float))
                count = _count_events_in_windows(target_times, starts, ends)
                props.append(float(count) / float(ref_times.size))
                target_coords_list.append(coord)
            if not props:
                continue
            target_coords = np.asarray(target_coords_list, dtype=float)
            all_dists.append(_pairwise_distances(ref_coord[None, :], target_coords).reshape(-1))
            all_props.append(np.asarray(props, dtype=float))

    distances = np.concatenate(all_dists) if all_dists else np.array([], dtype=float)
    props = np.concatenate(all_props) if all_props else np.array([], dtype=float)
    return _cofiring_result(distances, props, log=log, min_distance=min_distance, max_distance=max_distance)


def _fr_result(
    distances: list[float] | np.ndarray,
    rates: list[float] | np.ndarray,
    *,
    log: bool,
    min_distance: float,
    max_distance: float,
) -> FRDistanceResult:
    if len(distances) == 0:
        empty = np.array([])
        return FRDistanceResult(distances=empty, rates=empty, bins=empty, binned=BinnedSeries(empty, empty, empty))

    distances_arr = np.asarray(distances, dtype=float)
    rates_arr = np.asarray(rates, dtype=float)
    bins, binned = _bin_distance_series(distances_arr, rates_arr, log=log, min_distance=min_distance, max_distance=max_distance)
    return FRDistanceResult(distances=distances_arr, rates=rates_arr, bins=bins, binned=binned)


def _cofiring_result(
    distances: list[float] | np.ndarray,
    proportions: list[float] | np.ndarray,
    *,
    log: bool,
    min_distance: float,
    max_distance: float,
) -> CofiringDistanceResult:
    if len(distances) == 0:
        empty = np.array([])
        return CofiringDistanceResult(distances=empty, proportions=empty, bins=empty, binned=BinnedSeries(empty, empty, empty))

    distances_arr = np.asarray(distances, dtype=float)
    props_arr = np.asarray(proportions, dtype=float)
    bins, binned = _bin_distance_series(distances_arr, props_arr, log=log, min_distance=min_distance, max_distance=max_distance)
    return CofiringDistanceResult(distances=distances_arr, proportions=props_arr, bins=bins, binned=binned)


def _bin_distance_series(
    distances: np.ndarray,
    values: np.ndarray,
    *,
    log: bool,
    min_distance: float,
    max_distance: float,
) -> tuple[np.ndarray, BinnedSeries]:
    if log:
        bins = np.logspace(np.log10(max(min_distance, float(distances.min()))), np.log10(max_distance), num=74)
    else:
        bins = np.linspace(max(min_distance, float(distances.min())), max_distance, num=74)

    bin_idx = np.searchsorted(bins, distances, side="right") - 1
    bin_idx[distances == bins[-1]] = bins.size - 2
    in_range = (bin_idx >= 0) & (bin_idx < bins.size - 1) & np.isfinite(values)
    bin_idx = bin_idx[in_range]
    values = values[in_range]
    n_bins = bins.size - 1
    counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    sums = np.bincount(bin_idx, weights=values, minlength=n_bins)
    sums_sq = np.bincount(bin_idx, weights=values * values, minlength=n_bins)
    means = np.divide(sums, counts, out=np.full(n_bins, np.nan, dtype=float), where=counts > 0)
    variances = np.divide(sums_sq, counts, out=np.zeros(n_bins, dtype=float), where=counts > 0) - means * means
    variances = np.maximum(variances, 0.0)
    stderr = np.divide(np.sqrt(variances), np.sqrt(counts), out=np.full(n_bins, np.nan, dtype=float), where=counts > 0)
    centers = (bins[:-1] + bins[1:]) / 2
    valid = counts > 0
    return bins, BinnedSeries(centers=centers[valid], mean=means[valid], stderr=stderr[valid])


def _prepare_recording_distance_inputs(rec, refs):
    refs = np.asarray([] if refs is None else list(refs), dtype=int)
    if refs.size == 0:
        return None

    layout_electrodes = np.asarray(rec.layout["electrode"], dtype=int)
    layout_channels = np.asarray(rec.layout.get("channel", layout_electrodes), dtype=int)
    coords = np.column_stack([np.asarray(rec.layout["x"], dtype=float), np.asarray(rec.layout["y"], dtype=float)])

    valid_layout = np.isfinite(coords).all(axis=1)
    layout_electrodes = layout_electrodes[valid_layout]
    layout_channels = layout_channels[valid_layout]
    coords = coords[valid_layout]
    if layout_electrodes.size == 0:
        return None

    _, first_idx = np.unique(layout_electrodes, return_index=True)
    if first_idx.size != layout_electrodes.size:
        first_idx = np.sort(first_idx)
        layout_electrodes = layout_electrodes[first_idx]
        layout_channels = layout_channels[first_idx]
        coords = coords[first_idx]

    ref_idx = _indices_for_values(layout_electrodes, refs)
    valid_refs = ref_idx >= 0
    if not np.any(valid_refs):
        return None
    refs = refs[valid_refs]
    ref_coords = coords[ref_idx[valid_refs]]

    spike_times = np.asarray(rec.spikes["time"], dtype=float)
    if "channel" in rec.spikes:
        spike_channels = np.asarray(rec.spikes["channel"], dtype=int)
        channel_order = np.argsort(layout_channels)
        sorted_channels = layout_channels[channel_order]
        channel_pos = np.searchsorted(sorted_channels, spike_channels)
        valid_pos = channel_pos < sorted_channels.size
        valid_channel = np.zeros(spike_channels.shape, dtype=bool)
        valid_channel[valid_pos] = sorted_channels[channel_pos[valid_pos]] == spike_channels[valid_pos]
        spike_electrodes = np.empty(spike_channels.shape, dtype=int)
        spike_electrodes[valid_channel] = layout_electrodes[channel_order[channel_pos[valid_channel]]]
    else:
        spike_electrodes = np.asarray(rec.spikes["electrode"], dtype=int)
        valid_channel = np.ones(spike_electrodes.shape, dtype=bool)

    in_window = (
        valid_channel
        & np.isfinite(spike_times)
        & (spike_times >= float(rec.start_time))
        & (spike_times <= float(rec.end_time))
    )
    spike_times = spike_times[in_window]
    spike_electrodes = spike_electrodes[in_window]
    valid_spike_electrode = _indices_for_values(layout_electrodes, spike_electrodes) >= 0
    spike_times = spike_times[valid_spike_electrode]
    spike_electrodes = spike_electrodes[valid_spike_electrode]
    if spike_times.size == 0:
        return layout_electrodes, coords, refs, ref_coords, spike_electrodes, spike_times

    order = np.argsort(spike_times, kind="mergesort")
    return layout_electrodes, coords, refs, ref_coords, spike_electrodes[order], spike_times[order]


def _indices_for_values(sorted_or_unsorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    base = np.asarray(sorted_or_unsorted_values)
    values = np.asarray(values)
    out = np.full(values.shape, -1, dtype=int)
    if base.size == 0 or values.size == 0:
        return out
    order = np.argsort(base)
    sorted_base = base[order]
    pos = np.searchsorted(sorted_base, values)
    valid_pos = pos < sorted_base.size
    valid = np.zeros(values.shape, dtype=bool)
    valid[valid_pos] = sorted_base[pos[valid_pos]] == values[valid_pos]
    out[valid] = order[pos[valid]]
    return out


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.asarray(a, dtype=float)[:, None, :] - np.asarray(b, dtype=float)[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _sorted_spike_times_by_electrode(spike_electrodes: np.ndarray, spike_times: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    for electrode in np.unique(spike_electrodes):
        out[int(electrode)] = spike_times[spike_electrodes == electrode]
    return out


def _count_events_in_windows(times: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> int:
    if times.size == 0 or starts.size == 0:
        return 0
    idx = np.searchsorted(starts, times, side="right") - 1
    valid = idx >= 0
    if not np.any(valid):
        return 0
    idx_valid = idx[valid]
    return int(np.count_nonzero(times[valid] <= ends[idx_valid]))
