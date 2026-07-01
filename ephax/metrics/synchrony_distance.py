from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from .burst import (
    build_highres_traces,
    build_participation_activity_state,
    build_population_ifr,
    detect_high_activity_epochs,
    detect_participation_burst_epochs,
)


@dataclass(frozen=True)
class ActivityIntervalConfig:
    """Parameters for high-activity and burst interval detection."""

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
    activity_scope: str = "burst"


@dataclass(frozen=True)
class SynchronyDistanceConfig:
    """Parameters for distance-binned conditional spike synchrony."""

    activity_scope: str = "burst"
    lag_windows_ms: tuple[tuple[float, float], ...] = ((-2.0, 2.0),)
    primary_lag_window_ms: tuple[float, float] = (-2.0, 2.0)
    min_distance_um: float = 50.0
    max_distance_um: float = 3500.0
    distance_bin_um: float = 100.0
    matrix_bin_ms: float = 1.0
    null_method: str = "interval_jitter"
    jitter_ms: float = 25.0
    bootstrap_reps: int = 5
    n_surrogates: int = 5
    null_std_floor: float = 1e-4
    random_seed: int = 0
    estimator_version: str = "interval_summary_v1"

    def settings_dict(self) -> dict[str, object]:
        return {
            "activity_scope": self.activity_scope,
            "lag_windows_ms": [tuple(map(float, w)) for w in self.lag_windows_ms],
            "primary_lag_window_ms": tuple(map(float, self.primary_lag_window_ms)),
            "min_distance_um": float(self.min_distance_um),
            "max_distance_um": float(self.max_distance_um),
            "distance_bin_um": float(self.distance_bin_um),
            "matrix_bin_ms": float(self.matrix_bin_ms),
            "null_method": self.null_method,
            "jitter_ms": float(self.jitter_ms),
            "bootstrap_reps": int(self.bootstrap_reps),
            "n_surrogates": int(self.n_surrogates),
            "null_std_floor": float(self.null_std_floor),
            "random_seed": int(self.random_seed),
            "estimator_version": self.estimator_version,
            "compute_method": "interval_summary_matrix",
        }


@dataclass
class RecordingSynchronyInput:
    """One recording plus metadata needed for cache-compatible output rows."""

    recording: object
    electrodes: np.ndarray
    dataset: str
    well: int
    div: int
    recording_id: str
    selected_intervals: pd.DataFrame | None = None
    source_file: str | None = None


def _empty_intervals() -> pd.DataFrame:
    return pd.DataFrame(columns=["event_idx", "start_time_s", "end_time_s", "duration_s"])


def normalize_intervals(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a clean interval table with start/end/duration columns."""

    if df is None or len(df) == 0:
        return _empty_intervals()
    out = df.copy()
    if not {"start_time_s", "end_time_s"}.issubset(out.columns):
        raise ValueError("interval table must contain start_time_s and end_time_s")
    out["start_time_s"] = out["start_time_s"].astype(float)
    out["end_time_s"] = out["end_time_s"].astype(float)
    out = out[np.isfinite(out["start_time_s"]) & np.isfinite(out["end_time_s"])]
    out = out[out["end_time_s"] > out["start_time_s"]]
    out = out.sort_values("start_time_s", ignore_index=True)
    out["duration_s"] = out["end_time_s"] - out["start_time_s"]
    if "event_idx" not in out.columns:
        out["event_idx"] = np.arange(len(out), dtype=int)
    return out


def subtract_intervals(base_df: pd.DataFrame | None, remove_df: pd.DataFrame | None) -> pd.DataFrame:
    """Subtract remove intervals from base intervals."""

    base = normalize_intervals(base_df)
    remove = normalize_intervals(remove_df)
    if base.empty or remove.empty:
        return base.copy()

    rows = []
    for _, base_row in base.iterrows():
        pieces = [(float(base_row.start_time_s), float(base_row.end_time_s))]
        overlaps = remove[
            (remove["end_time_s"] > float(base_row.start_time_s))
            & (remove["start_time_s"] < float(base_row.end_time_s))
        ]
        for _, remove_row in overlaps.iterrows():
            remove_start = float(remove_row.start_time_s)
            remove_end = float(remove_row.end_time_s)
            next_pieces = []
            for piece_start, piece_end in pieces:
                if remove_end <= piece_start or remove_start >= piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if remove_start > piece_start:
                    next_pieces.append((piece_start, min(remove_start, piece_end)))
                if remove_end < piece_end:
                    next_pieces.append((max(remove_end, piece_start), piece_end))
            pieces = [(piece_start, piece_end) for piece_start, piece_end in next_pieces if piece_end > piece_start]
        for piece_start, piece_end in pieces:
            rows.append({"start_time_s": piece_start, "end_time_s": piece_end})

    if not rows:
        return _empty_intervals()
    out = pd.DataFrame(rows)
    out["event_idx"] = np.arange(len(out), dtype=int)
    out["duration_s"] = out["end_time_s"] - out["start_time_s"]
    return out


def choose_activity_intervals(
    high_epochs: pd.DataFrame | None,
    burst_epochs: pd.DataFrame | None,
    scope: str,
) -> pd.DataFrame:
    """Select high-activity, burst, or high-activity non-burst intervals."""

    high_epochs = normalize_intervals(high_epochs)
    burst_epochs = normalize_intervals(burst_epochs)
    if scope in {"high_activity", "high_activity_all"}:
        return high_epochs
    if scope == "burst":
        return burst_epochs
    if scope == "high_activity_non_burst":
        return subtract_intervals(high_epochs, burst_epochs)
    raise ValueError("activity scope must be 'high_activity_all', 'high_activity_non_burst', or 'burst'")


def detect_activity_intervals(
    recording,
    selected_electrodes: Iterable[int],
    config: ActivityIntervalConfig = ActivityIntervalConfig(),
) -> dict[str, object]:
    """Detect high-activity and participation-burst intervals for one recording."""

    refs = np.asarray(list(selected_electrodes), dtype=int)
    population = build_population_ifr(
        recording,
        refs,
        grid_hz=config.ifr_grid_hz,
        smooth_sigma_sec=config.smooth_sigma_sec,
    )
    high_epochs, high_info = detect_high_activity_epochs(
        population.time_grid,
        population.mean_ifr_smooth,
        mad_scale=config.high_activity_mad_scale,
        min_duration_ms=config.high_activity_min_duration_ms,
        max_gap_bins=config.high_activity_max_gap_bins,
    )
    try:
        highres = build_highres_traces(
            recording,
            refs,
            bin_ms=config.highres_bin_ms,
            smooth_sigma_ms=config.highres_smooth_sigma_ms,
        )
        participation = build_participation_activity_state(highres, aggregation_ms=config.network_bin_ms)
        burst_epochs = detect_participation_burst_epochs(
            participation,
            high_epochs,
            min_participation_fraction=config.network_min_participation_fraction,
            min_duration_ms=config.network_min_duration_ms,
        )
    except ValueError:
        highres = None
        participation = None
        burst_epochs = _empty_intervals()

    selected = choose_activity_intervals(high_epochs, burst_epochs, config.activity_scope)
    return {
        "population": population,
        "highres": highres,
        "participation": participation,
        "high_activity_epochs": normalize_intervals(high_epochs),
        "burst_epochs": normalize_intervals(burst_epochs),
        "selected_intervals": normalize_intervals(selected),
        "high_activity_info": high_info,
    }


def electrode_coords_um(layout) -> pd.DataFrame:
    df = pd.DataFrame(layout)
    df = df.dropna(subset=["electrode", "x", "y"])
    return df.drop_duplicates("electrode").set_index("electrode")[["x", "y"]].astype(float)


def distance_edges_and_centers(config: SynchronyDistanceConfig) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(
        float(config.min_distance_um),
        float(config.max_distance_um) + float(config.distance_bin_um),
        float(config.distance_bin_um),
        dtype=float,
    )
    if edges[-1] < float(config.max_distance_um):
        edges = np.append(edges, float(config.max_distance_um))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def build_pair_arrays(
    electrodes: Iterable[int],
    coords_df: pd.DataFrame,
    config: SynchronyDistanceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    electrodes = np.asarray([int(e) for e in electrodes if int(e) in coords_df.index], dtype=int)
    if electrodes.size < 2:
        empty_i = np.array([], dtype=int)
        empty_f = np.array([], dtype=float)
        return electrodes, empty_i, empty_i, empty_f, empty_i, empty_f

    xy = coords_df.loc[electrodes, ["x", "y"]].to_numpy(dtype=float)
    diff = xy[:, None, :] - xy[None, :, :]
    distance_um = np.sqrt(np.sum(diff * diff, axis=2))
    ref_idx, target_idx = np.where(~np.eye(electrodes.size, dtype=bool))
    pair_distance = distance_um[ref_idx, target_idx]
    edges, centers = distance_edges_and_centers(config)
    bin_idx = np.searchsorted(edges, pair_distance, side="right") - 1
    keep = (bin_idx >= 0) & (bin_idx < centers.size)
    return electrodes, ref_idx[keep], target_idx[keep], pair_distance[keep], bin_idx[keep], centers


def spike_presence_matrix(
    spike_times_s: np.ndarray,
    spike_electrodes: np.ndarray,
    elec_to_idx: dict[int, int],
    start_s: float,
    end_s: float,
    bin_ms: float,
) -> np.ndarray:
    """1 ms presence matrix (time bins x electrodes) for one interval.

    The general array-based primitive shared by the distance-binned and the
    direction-resolved synchrony paths. ``elec_to_idx`` maps electrode id to
    column; spikes on electrodes outside the mapping are dropped.
    """
    bin_s = float(bin_ms) / 1000.0
    n_bins = int(np.ceil((float(end_s) - float(start_s)) / bin_s))
    n_elec = len(elec_to_idx)
    if n_bins <= 0:
        return np.zeros((0, n_elec), dtype=np.uint8)

    times = np.asarray(spike_times_s, dtype=float)
    spike_electrodes = np.asarray(spike_electrodes, dtype=int)
    mask = (times >= float(start_s)) & (times < float(end_s))
    times = times[mask]
    spike_electrodes = spike_electrodes[mask]
    if times.size == 0:
        return np.zeros((n_bins, n_elec), dtype=np.uint8)

    cols = np.fromiter((elec_to_idx.get(int(e), -1) for e in spike_electrodes), dtype=int, count=spike_electrodes.size)
    valid = cols >= 0
    rows = np.floor((times[valid] - float(start_s)) / bin_s).astype(int)
    cols = cols[valid]
    valid_rows = (rows >= 0) & (rows < n_bins)
    mat = np.zeros((n_bins, n_elec), dtype=np.uint8)
    mat[rows[valid_rows], cols[valid_rows]] = 1
    return mat


def interval_spike_matrix(recording, electrodes: np.ndarray, start_s: float, end_s: float, bin_ms: float) -> np.ndarray:
    """Presence matrix for the selected ``electrodes`` over one interval of a recording."""
    elec_to_idx = {int(e): i for i, e in enumerate(np.asarray(electrodes, dtype=int))}
    return spike_presence_matrix(
        recording.spikes["time"], recording.spikes["electrode"], elec_to_idx, start_s, end_s, bin_ms
    )


def window_spike_matrix(spike_matrix: np.ndarray, lag_start_ms: float, lag_stop_ms: float, bin_ms: float) -> np.ndarray:
    lag_start_bins = int(np.floor(float(lag_start_ms) / float(bin_ms)))
    lag_stop_bins = int(np.ceil(float(lag_stop_ms) / float(bin_ms)))
    out = np.zeros_like(spike_matrix, dtype=np.uint8)
    n_time = spike_matrix.shape[0]
    for lag in range(lag_start_bins, lag_stop_bins + 1):
        if lag >= 0:
            if lag < n_time:
                out[: n_time - lag] |= spike_matrix[lag:]
        else:
            shift = -lag
            if shift < n_time:
                out[shift:] |= spike_matrix[: n_time - shift]
    return out


def circular_shift_columns(mat: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat)
    shifts = np.asarray(shifts, dtype=int)
    if mat.size == 0:
        return mat.copy()
    rows = (np.arange(mat.shape[0])[:, None] - shifts[None, :]) % mat.shape[0]
    return mat[rows, np.arange(mat.shape[1])[None, :]]


def _bin_pair_metrics_from_matrices(
    metric_matrices: dict[str, np.ndarray],
    trigger_counts: np.ndarray,
    ref_idx: np.ndarray,
    target_idx: np.ndarray,
    bin_idx: np.ndarray,
    centers: np.ndarray,
) -> pd.DataFrame:
    trigger_counts = np.asarray(trigger_counts, dtype=float)
    valid_pair = (trigger_counts[ref_idx] > 0) & np.isfinite(metric_matrices["p_obs"][ref_idx, target_idx])
    if not np.any(valid_pair):
        return pd.DataFrame()

    combined = None
    for metric_name, matrix in metric_matrices.items():
        pair_vals = np.asarray(matrix, dtype=float)[ref_idx, target_idx]
        base = pd.DataFrame({"bin_idx": bin_idx[valid_pair], metric_name: pair_vals[valid_pair]})
        if base.empty:
            continue
        if metric_name == "p_obs":
            grouped = base.groupby("bin_idx")[metric_name].agg(["mean", "median", "size"]).reset_index()
            combined = grouped.rename(columns={"mean": metric_name, "median": f"{metric_name}_median", "size": "n_pairs"})
            combined["distance_bin_center_um"] = centers[combined["bin_idx"].to_numpy(dtype=int)]
        else:
            grouped = base.groupby("bin_idx")[metric_name].agg(["mean", "median"]).reset_index()
            grouped = grouped.rename(columns={"mean": metric_name, "median": f"{metric_name}_median"})
            combined = combined.merge(grouped, on="bin_idx", how="left")

    if combined is None or combined.empty:
        return pd.DataFrame()
    trig_df = pd.DataFrame({"bin_idx": bin_idx[valid_pair], "n_trigger_spikes": trigger_counts[ref_idx][valid_pair]})
    trig_summary = trig_df.groupby("bin_idx")["n_trigger_spikes"].sum().reset_index()
    return combined.merge(trig_summary, on="bin_idx", how="left")


def compute_interval_synchrony_summaries(
    recording,
    electrodes: Iterable[int],
    intervals: pd.DataFrame,
    config: SynchronyDistanceConfig = SynchronyDistanceConfig(),
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute observed and null distance-bin synchrony once per interval."""

    if rng is None:
        rng = np.random.default_rng(config.random_seed)
    if config.null_method not in {"rate_expectation", "interval_jitter"}:
        raise ValueError("null_method must be 'rate_expectation' or 'interval_jitter'")

    coords = electrode_coords_um(recording.layout)
    electrodes, ref_idx, target_idx, _, bin_idx, centers = build_pair_arrays(electrodes, coords, config)
    if ref_idx.size == 0:
        return pd.DataFrame(), electrodes

    rows = []
    lag_windows = [tuple(map(float, window)) for window in config.lag_windows_ms]
    bin_ms = float(config.matrix_bin_ms)
    jitter_bins = max(1, int(round(float(config.jitter_ms) / bin_ms)))
    n_surrogates = int(config.n_surrogates)

    for interval_idx, interval in normalize_intervals(intervals).reset_index(drop=True).iterrows():
        mat = interval_spike_matrix(recording, electrodes, interval.start_time_s, interval.end_time_s, bin_ms)
        if mat.shape[0] < 1:
            continue
        if config.null_method == "interval_jitter" and mat.shape[0] < max(1, jitter_bins):
            continue

        interval_triggers = mat.sum(axis=0).astype(np.float64)
        mat_f = mat.astype(np.float32, copy=False)
        denom = interval_triggers[:, None]

        obs_hits = []
        null_hits_sum = []
        null_hits_sq_sum = []
        null_hits_var = []
        for lag_start_ms, lag_stop_ms in lag_windows:
            target_window = window_spike_matrix(mat, lag_start_ms, lag_stop_ms, bin_ms).astype(np.float32, copy=False)
            obs_hits.append(mat_f.T @ target_window)
            null_hits_sum.append(np.zeros((len(electrodes), len(electrodes)), dtype=np.float64))
            null_hits_sq_sum.append(np.zeros((len(electrodes), len(electrodes)), dtype=np.float64))
            null_hits_var.append(np.zeros((len(electrodes), len(electrodes)), dtype=np.float64))

            if config.null_method == "rate_expectation":
                p_target = target_window.mean(axis=0).astype(np.float64)
                null_hits_sum[-1] = interval_triggers[:, None] * p_target[None, :]
                null_hits_var[-1] = interval_triggers[:, None] * p_target[None, :] * (1.0 - p_target[None, :])

        if config.null_method == "interval_jitter":
            if n_surrogates <= 0:
                raise ValueError("n_surrogates must be > 0 for interval_jitter null")
            for _ in range(n_surrogates):
                shifts = rng.integers(-jitter_bins, jitter_bins + 1, size=mat.shape[1])
                shifted = circular_shift_columns(mat, shifts)
                for window_idx, (lag_start_ms, lag_stop_ms) in enumerate(lag_windows):
                    null_window = window_spike_matrix(shifted, lag_start_ms, lag_stop_ms, bin_ms).astype(np.float32, copy=False)
                    null_hit = mat_f.T @ null_window
                    null_hits_sum[window_idx] += null_hit
                    null_hits_sq_sum[window_idx] += null_hit * null_hit

        for window_idx, (lag_start_ms, lag_stop_ms) in enumerate(lag_windows):
            p_obs = np.divide(obs_hits[window_idx], denom, out=np.full_like(obs_hits[window_idx], np.nan, dtype=float), where=denom > 0)

            if config.null_method == "rate_expectation":
                null_mean_hits = null_hits_sum[window_idx]
                null_var_hits = np.maximum(null_hits_var[window_idx], 0.0)
            else:
                null_mean_hits = null_hits_sum[window_idx] / n_surrogates
                null_var_hits = null_hits_sq_sum[window_idx] / n_surrogates - null_mean_hits * null_mean_hits
                null_var_hits = np.maximum(null_var_hits, 0.0)

            p_null_mean = np.divide(null_mean_hits, denom, out=np.full_like(null_mean_hits, np.nan, dtype=float), where=denom > 0)
            p_null_std = np.divide(np.sqrt(null_var_hits), denom, out=np.full_like(null_var_hits, np.nan, dtype=float), where=denom > 0)
            excess = p_obs - p_null_mean
            z_sync = np.divide(excess, p_null_std, out=np.full_like(excess, np.nan), where=p_null_std >= float(config.null_std_floor))
            prob_eps = 0.5 / (denom + 1.0)
            log_ratio = np.log((p_obs + prob_eps) / (p_null_mean + prob_eps))

            combined = _bin_pair_metrics_from_matrices(
                {
                    "p_obs": p_obs,
                    "p_null_mean": p_null_mean,
                    "p_null_std": p_null_std,
                    "excess_sync": excess,
                    "z_sync": z_sync,
                    "log_ratio": log_ratio,
                },
                interval_triggers,
                ref_idx,
                target_idx,
                bin_idx,
                centers,
            )
            if combined.empty:
                continue
            combined["interval_idx"] = int(interval_idx)
            combined["interval_start_time_s"] = float(interval.start_time_s)
            combined["interval_end_time_s"] = float(interval.end_time_s)
            combined["lag_start_ms"] = float(lag_start_ms)
            combined["lag_stop_ms"] = float(lag_stop_ms)
            rows.append(combined)

    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), electrodes


def weighted_event_bootstrap(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    ci: tuple[float, float] = (2.5, 97.5),
    return_boot: bool = False,
) -> dict:
    """Pool per-event cell values by a weighted mean and bootstrap over events.

    ``values`` and ``weights`` are ``(n_events, n_cells)`` arrays: row ``e`` holds
    the per-cell value of one event (a distance bin for the scalar path, an
    orientation x distance cell for the directional path) and its weight (pair
    count). The pooled estimate is the event-weighted mean per cell,
    ``sum(values * weights) / sum(weights)``; the bootstrap resamples whole events
    with replacement, so events (not pairs or wells) are the unit of replication.
    This is the within-culture pooling shared by both synchrony paths.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape or values.ndim != 2:
        raise ValueError("values and weights must be matching (n_events, n_cells) arrays")
    n_events, n_cells = values.shape
    w = np.where(np.isfinite(weights) & np.isfinite(values), weights, 0.0)
    vw = np.where(w > 0, values, 0.0) * w

    def _pool(rows):
        num = vw[rows].sum(axis=0)
        den = w[rows].sum(axis=0)
        return np.divide(num, den, out=np.full(n_cells, np.nan), where=den > 0), den

    point, point_weight = _pool(np.arange(n_events))
    out = {
        "point": point,
        "point_weight": point_weight,
        "n_events": int(n_events),
        "ci_lo": np.full(n_cells, np.nan),
        "ci_hi": np.full(n_cells, np.nan),
        "se": np.full(n_cells, np.nan),
    }
    if n_events == 0:
        if return_boot:
            out["boot"] = np.zeros((0, n_cells))
        return out

    rng = np.random.default_rng(int(seed))
    boot = np.empty((int(n_boot), n_cells), dtype=float)
    for b in range(int(n_boot)):
        boot[b], _ = _pool(rng.integers(0, n_events, size=n_events))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["ci_lo"] = np.nanpercentile(boot, ci[0], axis=0)
        out["ci_hi"] = np.nanpercentile(boot, ci[1], axis=0)
        out["se"] = np.nanstd(boot, axis=0, ddof=1) if n_boot > 1 else np.full(n_cells, np.nan)
    if return_boot:
        out["boot"] = boot
    return out


def pool_distance_synchrony_across_recordings(
    interval_summary: pd.DataFrame,
    *,
    metric: str = "excess_sync",
    weight_col: str = "n_pairs",
    n_boot: int = 1000,
    seed: int = 0,
    ci: tuple[float, float] = (2.5, 97.5),
    return_boot: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Within-culture pooled distance-bin synchrony with an event-level bootstrap.

    ``interval_summary`` is the per-interval distance-bin table from
    :func:`compute_interval_synchrony_summaries`, concatenated across the wells of
    one culture with a ``recording_id`` column added. Each ``(recording_id,
    interval_idx)`` is one event; events are pooled across wells and resampled,
    so the well's between-replicate noise is traded for event-level power. With
    ``return_boot`` the per-bootstrap pooled curves are also returned (long form:
    ``bootstrap_idx, distance_bin_center_um, <metric>``).
    """
    empty = pd.DataFrame()
    if interval_summary.empty:
        return (empty, empty) if return_boot else empty
    df = interval_summary.copy()
    if "recording_id" not in df.columns:
        df["recording_id"] = "0"
    df["event_key"] = df["recording_id"].astype(str) + "|" + df["interval_idx"].astype(int).astype(str)
    events = np.asarray(sorted(df["event_key"].unique()))
    event_to_row = {e: i for i, e in enumerate(events)}
    bins = np.asarray(sorted(df["distance_bin_center_um"].unique()), dtype=float)
    bin_to_col = {b: j for j, b in enumerate(bins)}

    values = np.full((events.size, bins.size), np.nan)
    weights = np.zeros((events.size, bins.size))
    for row in df.itertuples(index=False):
        i = event_to_row[getattr(row, "event_key")]
        j = bin_to_col[float(getattr(row, "distance_bin_center_um"))]
        values[i, j] = float(getattr(row, metric))
        weights[i, j] = float(getattr(row, weight_col))

    res = weighted_event_bootstrap(values, weights, n_boot=n_boot, seed=seed, ci=ci, return_boot=return_boot)
    summary = pd.DataFrame(
        {
            "distance_bin_center_um": bins,
            metric: res["point"],
            f"{metric}_ci_lo": res["ci_lo"],
            f"{metric}_ci_hi": res["ci_hi"],
            f"{metric}_se": res["se"],
            "n_pairs": res["point_weight"],
            "n_events": int(res["n_events"]),
        }
    )
    if not return_boot:
        return summary
    boot = res.get("boot", np.zeros((0, bins.size)))
    boot_df = pd.DataFrame(boot, columns=bins)
    boot_df.insert(0, "bootstrap_idx", np.arange(boot.shape[0], dtype=int))
    boot_long = boot_df.melt(id_vars="bootstrap_idx", var_name="distance_bin_center_um", value_name=metric)
    boot_long["distance_bin_center_um"] = boot_long["distance_bin_center_um"].astype(float)
    return summary, boot_long


def bootstrap_interval_synchrony_summaries(
    interval_summary: pd.DataFrame,
    config: SynchronyDistanceConfig = SynchronyDistanceConfig(),
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Bootstrap distance-bin synchrony by resampling intervals."""

    if interval_summary.empty:
        return pd.DataFrame()
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    rows = []
    intervals = np.asarray(sorted(interval_summary["interval_idx"].unique()), dtype=int)
    metric_cols = ["p_obs", "p_null_mean", "p_null_std", "excess_sync", "z_sync", "log_ratio"]
    median_cols = [f"{metric}_median" for metric in metric_cols if f"{metric}_median" in interval_summary.columns]

    for bootstrap_idx in range(int(config.bootstrap_reps)):
        sampled_intervals = rng.choice(intervals, size=intervals.size, replace=True)
        sampled = pd.concat(
            [interval_summary[interval_summary["interval_idx"] == int(idx)] for idx in sampled_intervals],
            ignore_index=True,
        )
        if sampled.empty:
            continue
        for (lag_start_ms, lag_stop_ms, bin_idx), group in sampled.groupby(["lag_start_ms", "lag_stop_ms", "bin_idx"], sort=True):
            weights = group["n_pairs"].to_numpy(float)
            row = {
                "dataset": None,
                "well": None,
                "div": None,
                "recording_id": None,
                "activity_scope": config.activity_scope,
                "bootstrap_idx": int(bootstrap_idx),
                "compute_method": "interval_summary_matrix",
                "null_method": config.null_method,
                "lag_start_ms": float(lag_start_ms),
                "lag_stop_ms": float(lag_stop_ms),
                "distance_bin_idx": int(bin_idx),
                "distance_bin_center_um": float(group["distance_bin_center_um"].iloc[0]),
                "n_pairs": int(np.nansum(group["n_pairs"])),
                "n_trigger_spikes": int(np.nansum(group["n_trigger_spikes"])),
            }
            for metric in metric_cols:
                if metric not in group:
                    continue
                values = group[metric].to_numpy(float)
                finite = np.isfinite(values) & np.isfinite(weights)
                row[metric] = float(np.average(values[finite], weights=weights[finite])) if np.any(finite) else np.nan
            for metric_col in median_cols:
                values = group[metric_col].to_numpy(float)
                row[metric_col] = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _tag_recording_rows(df: pd.DataFrame, item: RecordingSynchronyInput, config: SynchronyDistanceConfig) -> pd.DataFrame:
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    df = df.copy()
    df["dataset"] = item.dataset
    df["well"] = int(item.well)
    df["div"] = int(item.div)
    df["recording_id"] = str(item.recording_id)
    df["activity_scope"] = config.activity_scope
    return df


def compute_recording_synchrony_distance(
    item: RecordingSynchronyInput,
    config: SynchronyDistanceConfig = SynchronyDistanceConfig(),
    rng: np.random.Generator | None = None,
    *,
    return_interval_summary: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Return cache-compatible bootstrapped synchrony rows for one recording.

    With ``return_interval_summary`` also return the tagged per-interval distance-bin
    summaries (the input to the event-level within-culture pooling) from the same
    single matrix/surrogate computation — no extra cost.
    """

    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    def _out(result, interval_summary):
        return (result, interval_summary) if return_interval_summary else result

    intervals = normalize_intervals(item.selected_intervals)
    if intervals.empty:
        return _out(pd.DataFrame(), pd.DataFrame())

    coords = electrode_coords_um(item.recording.layout)
    available = np.asarray([int(e) for e in np.asarray(item.electrodes, dtype=int) if int(e) in coords.index], dtype=int)
    if available.size < 2:
        return _out(pd.DataFrame(), pd.DataFrame())

    interval_summary, _ = compute_interval_synchrony_summaries(item.recording, available, intervals, config, rng)
    result = bootstrap_interval_synchrony_summaries(interval_summary, config, rng)
    result = _tag_recording_rows(result, item, config)
    interval_summary = _tag_recording_rows(interval_summary, item, config)
    return _out(result, interval_summary)


def compute_synchrony_distance_payload(
    recordings: Iterable[RecordingSynchronyInput],
    config: SynchronyDistanceConfig = SynchronyDistanceConfig(),
) -> dict[str, object]:
    """Compute a combined cache payload for multiple recordings."""

    rng = np.random.default_rng(config.random_seed)
    chunks = []
    for item in recordings:
        chunk = compute_recording_synchrony_distance(item, config, rng)
        if not chunk.empty:
            chunks.append(chunk)
    pair_sync_samples = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return {
        "pair_sync_samples": pair_sync_samples,
        "settings": config.settings_dict(),
    }


def write_synchrony_distance_cache(payload: dict[str, object], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def read_synchrony_distance_cache(path: str | Path) -> dict[str, object]:
    with Path(path).open("rb") as f:
        return pickle.load(f)
