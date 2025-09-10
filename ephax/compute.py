from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, Optional, List

from joblib import Parallel, delayed
from scipy.stats import gaussian_kde, binned_statistic
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture

from .models import BinnedSeries, IFRPeaks, GMMFit, CofiringHeatmap

# Local imports from the existing repo
from ephax.helper_functions import assign_r_distance, get_activity_sorted_electrodes
from ephax.spikes import calculate_ifr


def bin_series(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> BinnedSeries:
    """Bin y by x into fixed bins and compute mean and stderr per bin.

    Returns BinnedSeries with centers, mean, stderr (nan-safe).
    """
    bin_means, bin_edges, _ = binned_statistic(x, y, statistic=np.nanmean, bins=bins)
    bin_std_err, _, _ = binned_statistic(
        x, y, statistic=lambda v: np.nanstd(v) / np.sqrt(np.sum(np.isfinite(v))), bins=bins
    )
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid = ~np.isnan(bin_means)
    return BinnedSeries(centers=centers[valid], mean=bin_means[valid], stderr=bin_std_err[valid])


def ifr_peaks(
    spikes_data_list: Iterable[dict],
    start_times: Iterable[float],
    end_times: Iterable[float],
    log_scale: bool = True,
    selected_refs_per_recording: Optional[Iterable[Iterable[int]]] = None,
) -> IFRPeaks:
    """Compute IFR values across datasets and estimate peaks via KDE.

    Mirrors previous get_ifr_peaks but returns a structured IFRPeaks.
    """
    assert len(spikes_data_list) == len(start_times) == len(end_times), (
        "Mismatch in lengths of input lists."
    )

    values = []
    for idx, (spikes_data, start_time, end_time) in enumerate(zip(spikes_data_list, start_times, end_times)):
        # Use explicitly provided refs per recording; this is required for analyzer flow
        s_time = float(start_time)
        e_time = float(end_time)
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


def fit_ifr_gmm(
    values: np.ndarray,
    log_scale: bool = True,
    n_components: int | None = None,
) -> GMMFit:
    """Fit Gaussian mixture to IFR values (in same domain as provided values).

    If n_components is None, choose the optimal number via BIC over 1..8.
    Returns means converted to Hz if log_scale=True.
    """
    X = values.reshape(-1, 1)
    if n_components is None:
        bics = []
        rng = range(1, 9)
        for n in rng:
            gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=0)
            gmm.fit(X)
            bics.append(gmm.bic(X))
        n_components = list(rng)[int(np.argmin(bics))]

    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=0)
    gmm.fit(X)

    means = gmm.means_.flatten()
    std = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    means_hz = 10 ** means if log_scale else means

    # Likelihood ratio vs 1-component (coarse; same as previous behavior)
    single = GaussianMixture(n_components=1, covariance_type="full", random_state=0)
    single.fit(X)
    ll_gmm = gmm.score(X) * len(values)
    ll_single = single.score(X) * len(values)
    # Degrees of freedom difference approximated as n_components-1
    from scipy.stats import chi2
    stat = 2 * (ll_gmm - ll_single)
    df = max(1, n_components - 1)
    p_value = float(1 - chi2.cdf(stat, df))

    return GMMFit(means_hz=np.asarray(means_hz), std=np.asarray(std), weights=np.asarray(weights), p_value=p_value)


def cofiring_proportions(
    spikes_df: pd.DataFrame,
    stim_times: pd.Series | np.ndarray,
    window_size: float = 0.001,
    delay: float = 0.0,
    ref_electrode: int | None = None,
) -> Dict[int, float]:
    """Vectorized proportion of co-firing events per electrode within windows.

    Parameters are in seconds. Excludes ref_electrode from the result.
    """
    # Avoid building an enormous N x M boolean mask. Instead, sort windows
    # and use searchsorted to test membership for each spike time.
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
    """Compute co-firing proportions per electrode for several delays, and distances.

    Returns (proportions_by_delay, electrode_distances_by_electrode).
    """
    spikes_df = pd.DataFrame(spikes_data)
    layout_df = pd.DataFrame(layout)
    # Guard: skip datasets that don't include the requested reference electrode
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
            spikes_df_during, firing_times, window_size=window_size / 10000.0, delay=delay_sec, ref_electrode=ref_electrode
        )
        for electrode, proportion in props.items():
            if electrode == ref_electrode:
                continue
            props_by_delay[float(delay)][electrode] = proportion
            if electrode not in electrode_distances:
                d = layout_df.loc[layout_df["electrode"] == electrode, "distance"].values[0]
                electrode_distances[int(electrode)] = float(d)

    return props_by_delay, electrode_distances


def _aggregate_single(
    spikes_data_list, layout_list, ref_electrode, start_times, end_times, window_size, delays
):
    props_by_delay, electrode_distances = {}, []
    for spikes_data, layout, start_time, end_time in zip(spikes_data_list, layout_list, start_times, end_times):
        p_by_d, distances = cofiring_vs_distance_by_delay(
            spikes_data, layout, ref_electrode, start_time, end_time, window_size, delays
        )
        for d in delays:
            props_by_delay.setdefault(float(d), {}).update(p_by_d[float(d)])
        electrode_distances.extend(distances.values())

    all_distances = np.asarray(electrode_distances)
    distance_bins = np.linspace(float(all_distances.min()), float(all_distances.max()), num=31)
    bin_counts, _, _ = binned_statistic(all_distances, all_distances, statistic="count", bins=distance_bins)
    valid_bins = bin_counts > 0

    Z = np.zeros((len(delays) - 1, len(distance_bins) - 1))
    for i in range(len(delays) - 1):
        d = float(delays[i])
        filtered_props = []
        filtered_dists = []
        for elec, prop in props_by_delay.get(d, {}).items():
            filtered_props.append(float(prop))
            # key is electrode, need distance
            # Store distance via provided props dict keys are electrodes; re-fetch by electrode
        # We cannot recover distances here directly, so rebuild from items
        for elec, prop in props_by_delay.get(d, {}).items():
            # find distance from the aggregate electrode_distances list? Not available by id; so recompute mapping
            # safer approach: build a mapping during above update loop, but we already have per-ref electrode call
            pass
    # NOTE: We do not complete here; aggregation is handled in aggregate_cofiring_heatmap below.
    return None


def aggregate_cofiring_heatmap(
    spikes_data_list,
    layout_list,
    ref_electrodes,
    start_times,
    end_times,
    window_size: float = 20,
    delays: np.ndarray = np.linspace(-20, 20, 21),
) -> CofiringHeatmap:
    """Aggregate co-firing heatmap across reference electrodes.

    Returns CofiringHeatmap(Z, distance_bins, delays).
    """
    # Use threads to avoid duplicating large inputs across processes.
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_per_ref_heatmap)(
            spikes_data_list, layout_list, ref_electrode, start_times, end_times, window_size, delays
        )
        for ref_electrode in ref_electrodes
    )
    Z_stack = np.stack([Z for (Z, _) in results], axis=0)
    avg_Z = np.nanmean(Z_stack, axis=0)
    distance_bins = results[0][1]
    return CofiringHeatmap(Z=avg_Z, distance_bins=distance_bins, delays=np.asarray(delays))


def _per_ref_heatmap(spikes_data_list, layout_list, ref_electrode, start_times, end_times, window_size, delays):
    # Collect per-ref proportions and distances
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

    # Precompute distance per electrode for quick lookup
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
        # Only keep bins with any electrodes
        # Determine valid bins by counts
        counts, _, _ = binned_statistic(elec_dists, elec_dists, statistic="count", bins=distance_bins)
        valid = counts > 0
        Z[i, valid] = bin_means[valid]

    return Z, distance_bins
