from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic

from ..models import BinnedSeries, CofiringDistanceResult, FRDistanceResult
from ..preprocessing.geometry import assign_r_distance, assign_r_distance_all
from .cofiring import cofiring_proportions


def avg_rate_vs_distance(
    recordings: Iterable,
    refs_per_recording: Iterable[Iterable[int]],
    *,
    log: bool = False,
    min_distance: float = 50,
    max_distance: float = 3500,
) -> FRDistanceResult:
    """Compute firing rate for electrode/ref pairs and bin by distance."""
    all_rates: list[float] = []
    all_dists: list[float] = []

    for rec, refs in zip(recordings, refs_per_recording):
        refs = [] if refs is None else list(refs)
        if len(refs) == 0:
            continue
        spikes_df = pd.DataFrame(rec.spikes)
        layout_df = pd.DataFrame(rec.layout)
        spikes_df, distances_df = assign_r_distance_all(spikes_df, layout_df, refs)

        mask_t = (spikes_df["time"] >= rec.start_time) & (spikes_df["time"] <= rec.end_time)
        spikes_df_during = spikes_df[mask_t]
        duration = float(rec.end_time - rec.start_time)
        if duration <= 0:
            continue

        counts = spikes_df_during["electrode"].value_counts().reset_index()
        counts.columns = ["electrode", "counts"]
        counts["firing_rate"] = counts["counts"] / duration
        merged = pd.merge(counts, distances_df, on="electrode", how="inner")
        merged = merged[merged["electrode"] != merged["ref_electrode"]]
        all_rates.extend(merged["firing_rate"].astype(float).tolist())
        all_dists.extend(merged["distance"].astype(float).tolist())

    return _fr_result(all_dists, all_rates, log=log, min_distance=min_distance, max_distance=max_distance)


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
    all_props: list[float] = []
    all_dists: list[float] = []
    pm_sec = float(plusminus_ms) / 1000.0
    window_size_sec = 2.0 * pm_sec
    delay_sec = -pm_sec

    for rec, refs in zip(recordings, refs_per_recording):
        refs = [] if refs is None else list(refs)
        if len(refs) == 0:
            continue
        spikes_df_full = pd.DataFrame(rec.spikes)
        layout_df_full = pd.DataFrame(rec.layout)
        mask_t = (spikes_df_full["time"] >= rec.start_time) & (spikes_df_full["time"] <= rec.end_time)
        spikes_df_window = spikes_df_full[mask_t].copy()

        for ref in refs:
            spikes_df, layout_df = assign_r_distance(spikes_df_window.copy(), layout_df_full.copy(), int(ref))
            firing_times = spikes_df["time"][spikes_df["electrode"] == int(ref)]
            props = cofiring_proportions(
                spikes_df,
                firing_times,
                window_size=window_size_sec,
                delay=delay_sec,
                ref_electrode=int(ref),
            )
            for electrode, proportion in props.items():
                if electrode == int(ref):
                    continue
                all_props.append(float(proportion))
                distance = layout_df.loc[layout_df["electrode"] == electrode, "distance"].values[0]
                all_dists.append(float(distance))

    return _cofiring_result(all_dists, all_props, log=log, min_distance=min_distance, max_distance=max_distance)


def _fr_result(
    distances: list[float],
    rates: list[float],
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
    distances: list[float],
    proportions: list[float],
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

    bin_means, bin_edges, _ = binned_statistic(distances, values, statistic=np.nanmean, bins=bins)
    bin_stderr, _, _ = binned_statistic(
        distances,
        values,
        statistic=lambda x: np.std(x) / np.sqrt(len(x)),
        bins=bins,
    )
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid = ~np.isnan(bin_means)
    return bins, BinnedSeries(centers=centers[valid], mean=bin_means[valid], stderr=bin_stderr[valid])
