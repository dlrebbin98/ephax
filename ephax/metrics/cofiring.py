from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binned_statistic

from ..preprocessing.geometry import assign_r_distance


@dataclass
class CofiringHeatmap:
    Z: np.ndarray
    distance_bins: np.ndarray
    delays: np.ndarray


def cofiring_proportions(
    spikes_df: pd.DataFrame,
    stim_times: pd.Series | np.ndarray,
    window_size: float = 0.001,
    delay: float = 0.0,
    ref_electrode: int | None = None,
) -> Dict[int, float]:
    """Vectorized proportion of co-firing events per electrode within windows."""
    stim_times_arr = np.asarray(stim_times, dtype=float)
    if stim_times_arr.size == 0:
        return {int(e): 0.0 for e in set(spikes_df["electrode"].unique()) - {ref_electrode}}

    starts = stim_times_arr + float(delay)
    ends = starts + float(window_size)
    order = np.argsort(starts)
    starts_sorted = starts[order]
    ends_sorted = ends[order]

    t = spikes_df["time"].to_numpy()
    idx = np.searchsorted(starts_sorted, t, side="right") - 1
    valid_idx = idx >= 0
    covered = np.zeros_like(valid_idx, dtype=bool)
    if np.any(valid_idx):
        idx_clip = np.clip(idx[valid_idx], 0, len(ends_sorted) - 1)
        covered_subset = t[valid_idx] <= ends_sorted[idx_clip]
        covered[valid_idx] = covered_subset
    coinciding = spikes_df[covered]
    counts = coinciding[coinciding["electrode"] != ref_electrode].groupby("electrode").size().to_dict()
    total = len(stim_times)
    if total == 0:
        return {int(e): 0.0 for e in set(spikes_df["electrode"].unique()) - {ref_electrode}}
    return {int(e): counts.get(e, 0) / total for e in set(spikes_df["electrode"].unique()) - {ref_electrode}}


def cofiring_vs_distance_by_delay(
    spikes_data: dict,
    layout: dict,
    ref_electrode: int,
    start_time: float,
    end_time: float,
    window_size: float,
    delays: np.ndarray,
) -> Tuple[Dict[float, Dict[int, float]], Dict[int, float]]:
    """Compute co-firing proportions per electrode for several delays, and distances."""
    spikes_df = pd.DataFrame(spikes_data)
    layout_df = pd.DataFrame(layout)
    if "electrode" not in layout_df.columns or ref_electrode not in set(layout_df["electrode"].tolist()):
        empty = {float(d): {} for d in delays}
        return empty, {}
    spikes_df, layout_df = assign_r_distance(spikes_df, layout_df, ref_electrode)
    mask = (spikes_df["time"] >= start_time) & (spikes_df["time"] <= end_time)
    spikes_df_during = spikes_df[mask]

    firing_times = spikes_df_during["time"][spikes_df_during["electrode"] == ref_electrode]

    props_by_delay: Dict[float, Dict[int, float]] = {float(d): {} for d in delays}
    electrode_distances: Dict[int, float] = {}

    for delay in delays:
        delay_sec = delay / 1000.0
        props = cofiring_proportions(
            spikes_df_during,
            firing_times,
            window_size=window_size / 10000.0,
            delay=delay_sec,
            ref_electrode=ref_electrode,
        )
        for electrode, proportion in props.items():
            if electrode == ref_electrode:
                continue
            props_by_delay[float(delay)][electrode] = proportion
            if electrode not in electrode_distances:
                d = layout_df.loc[layout_df["electrode"] == electrode, "distance"].values[0]
                electrode_distances[int(electrode)] = float(d)

    return props_by_delay, electrode_distances


def aggregate_cofiring_heatmap(
    spikes_data_list,
    layout_list,
    ref_electrodes,
    start_times,
    end_times,
    window_size: float = 20,
    delays: np.ndarray = np.linspace(-20, 20, 21),
) -> CofiringHeatmap:
    """Aggregate co-firing heatmap across reference electrodes."""
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_per_ref_heatmap)(
            spikes_data_list,
            layout_list,
            ref_electrode,
            start_times,
            end_times,
            window_size,
            delays,
        )
        for ref_electrode in ref_electrodes
    )
    Z_stack = np.stack([Z for (Z, _) in results], axis=0)
    avg_Z = np.nanmean(Z_stack, axis=0)
    distance_bins = results[0][1]
    return CofiringHeatmap(Z=avg_Z, distance_bins=distance_bins, delays=np.asarray(delays))


def _per_ref_heatmap(spikes_data_list, layout_list, ref_electrode, start_times, end_times, window_size, delays):
    props_by_delay_all = {float(d): {} for d in delays}
    electrode_distances = {}
    for spikes_data, layout, start_time, end_time in zip(spikes_data_list, layout_list, start_times, end_times):
        props_by_delay, distances = cofiring_vs_distance_by_delay(
            spikes_data, layout, ref_electrode, start_time, end_time, window_size, delays
        )
        for d, mapping in props_by_delay.items():
            props_by_delay_all[d].update(mapping)
        electrode_distances.update(distances)

    dists = np.array(list(electrode_distances.values()), dtype=float)
    distance_bins = np.linspace(float(dists.min()), float(dists.max()), num=31)
    dist_by_e = electrode_distances

    Z = np.zeros((len(delays) - 1, len(distance_bins) - 1))
    for i in range(len(delays) - 1):
        d = float(delays[i])
        if not props_by_delay_all[d]:
            continue
        elecs = list(props_by_delay_all[d].keys())
        vals = np.array([props_by_delay_all[d][e] for e in elecs], dtype=float)
        elec_dists = np.array([dist_by_e[e] for e in elecs], dtype=float)
        bin_means, _, _ = binned_statistic(elec_dists, vals, statistic="mean", bins=distance_bins)
        counts, _, _ = binned_statistic(elec_dists, elec_dists, statistic="count", bins=distance_bins)
        valid = counts > 0
        Z[i, valid] = bin_means[valid]

    return Z, distance_bins
