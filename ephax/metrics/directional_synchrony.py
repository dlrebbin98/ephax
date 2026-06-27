"""Direction-resolved conditional synchrony over distance.

The activity-distance synchrony pipeline collapses every electrode pair into a
scalar distance bin. This module keeps the *orientation* of each directed pair so
synchrony can be resolved by direction, which is what makes spatial anisotropy of
coordinated firing measurable.

Design notes
------------
* Orientation is treated as **axial**: a pair i->j and its reverse describe the
  same line, so angles are folded into ``[0, pi)``. The natural anisotropy term is
  therefore the second angular harmonic ``cos(2*theta)``.
* The per-pair *excess synchrony* (observed minus null conditional firing
  probability) is a contrast, so it is robust to how many pairs of each
  orientation the electrode selection happens to sample. The anisotropy magnitude
  is a property of the pairs, not of the selection geometry.
* The null is computed at the spike-matrix level (before any binning), so both the
  fast closed-form ``rate_expectation`` null and the conservative
  ``interval_jitter`` surrogate null work unchanged with the orientation grouping.

The numerics mirror the matrix path of ``activity_distance_spike.ipynb``
(1 ms presence matrices, an OR'd lag window, ``mat.T @ window`` coincidence
counts) so results are comparable to the scalar synchrony curves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DirectionalSyncConfig:
    matrix_bin_ms: float = 1.0
    lag_window_ms: tuple[float, float] = (-5.0, 5.0)
    null_method: str = "interval_jitter"          # "interval_jitter" | "rate_expectation"
    jitter_ms: float = 25.0
    n_surrogates: int = 100
    min_trigger_spikes: int = 5
    min_distance_um: float = 50.0
    max_distance_um: float = 3500.0
    n_orientation_bins: int = 12
    # Angular index is fit over a coverage-complete distance range (full orientation
    # coverage requires distances within the array's short dimension).
    index_distance_um: tuple[float, float] = (150.0, 2000.0)
    index_ring_um: float = 200.0
    min_pairs_per_cell: int = 20
    random_seed: int = 0


@dataclass
class PairSynchrony:
    """Per-directed-pair point estimate pooled over intervals."""

    excess: np.ndarray          # p_obs - p_null
    p_obs: np.ndarray           # observed conditional firing probability
    distance_um: np.ndarray
    angle_rad: np.ndarray       # axial, folded to [0, pi)
    n_trigger: np.ndarray       # trigger spikes on the reference electrode


# --------------------------------------------------------------------------- #
# spike-matrix helpers (matrix path, mirrors the notebook)
# --------------------------------------------------------------------------- #
def interval_spike_matrix(times, electrodes, elec_to_idx, start_s, end_s, bin_ms):
    bin_s = float(bin_ms) / 1000.0
    n_bins = int(np.ceil((float(end_s) - float(start_s)) / bin_s))
    n_elec = len(elec_to_idx)
    if n_bins <= 0:
        return np.zeros((0, n_elec), dtype=np.uint8)
    times = np.asarray(times, dtype=float)
    electrodes = np.asarray(electrodes, dtype=int)
    mask = (times >= float(start_s)) & (times < float(end_s))
    times = times[mask]
    electrodes = electrodes[mask]
    if times.size == 0:
        return np.zeros((n_bins, n_elec), dtype=np.uint8)
    cols = np.fromiter((elec_to_idx.get(int(e), -1) for e in electrodes), dtype=int, count=electrodes.size)
    keep = cols >= 0
    rows = np.floor((times[keep] - float(start_s)) / bin_s).astype(int)
    cols = cols[keep]
    in_range = (rows >= 0) & (rows < n_bins)
    mat = np.zeros((n_bins, n_elec), dtype=np.uint8)
    mat[rows[in_range], cols[in_range]] = 1
    return mat


def window_matrix(mat, lag_start_ms, lag_stop_ms, bin_ms):
    """OR each column over the lag window so an entry marks 'a spike within window'."""
    lag_start = int(np.floor(float(lag_start_ms) / float(bin_ms)))
    lag_stop = int(np.ceil(float(lag_stop_ms) / float(bin_ms)))
    out = np.zeros_like(mat)
    n_time = mat.shape[0]
    for lag in range(lag_start, lag_stop + 1):
        if lag >= 0:
            if lag < n_time:
                out[: n_time - lag] |= mat[lag:]
        else:
            shift = -lag
            if shift < n_time:
                out[shift:] |= mat[: n_time - shift]
    return out


def circular_shift_columns(mat, shifts):
    mat = np.asarray(mat)
    shifts = np.asarray(shifts, dtype=int)
    if mat.size == 0:
        return mat.copy()
    rows = (np.arange(mat.shape[0])[:, None] - shifts[None, :]) % mat.shape[0]
    return mat[rows, np.arange(mat.shape[1])[None, :]]


# --------------------------------------------------------------------------- #
# orientation helpers
# --------------------------------------------------------------------------- #
def axial_angles(coords_xy: np.ndarray, ref_idx: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
    """Axial orientation (folded to ``[0, pi)``) of each directed pair ref->target."""
    d = np.asarray(coords_xy, dtype=float)[target_idx] - np.asarray(coords_xy, dtype=float)[ref_idx]
    return np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi)


def orientation_bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, np.pi, int(n_bins) + 1)


# --------------------------------------------------------------------------- #
# core: per-pair synchrony with orientation retained
# --------------------------------------------------------------------------- #
def compute_pair_synchrony(
    spike_times_s: np.ndarray,
    spike_electrodes: np.ndarray,
    coords_by_electrode: Mapping[int, Sequence[float]],
    electrodes: Iterable[int],
    intervals: Iterable[tuple[float, float]],
    config: DirectionalSyncConfig,
    rng: np.random.Generator | None = None,
) -> PairSynchrony:
    """Pooled per-directed-pair excess synchrony and orientation over intervals.

    ``electrodes`` is the already-selected electrode set (e.g. the top-activity
    references). ``intervals`` are ``(start_s, end_s)`` analysis windows. The null
    is accumulated at the matrix level so both null methods slot in unchanged.
    """
    if config.null_method not in {"interval_jitter", "rate_expectation"}:
        raise ValueError("null_method must be 'interval_jitter' or 'rate_expectation'")
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    electrodes = np.asarray([int(e) for e in electrodes if int(e) in coords_by_electrode], dtype=int)
    if electrodes.size < 2:
        empty = np.array([], dtype=float)
        return PairSynchrony(empty, empty, empty, empty, np.array([], dtype=float))
    electrodes = np.unique(electrodes)
    elec_to_idx = {int(e): i for i, e in enumerate(electrodes)}
    coords = np.array([coords_by_electrode[int(e)] for e in electrodes], dtype=float)
    n = electrodes.size

    spike_times_s = np.asarray(spike_times_s, dtype=float)
    spike_electrodes = np.asarray(spike_electrodes, dtype=int)

    bin_ms = float(config.matrix_bin_ms)
    lag = config.lag_window_ms
    jitter_bins = max(1, int(round(float(config.jitter_ms) / bin_ms)))
    n_surr = int(config.n_surrogates)

    obs_hits = np.zeros((n, n), dtype=np.float64)
    null_hits = np.zeros((n, n), dtype=np.float64)
    trig = np.zeros(n, dtype=np.float64)

    for start_s, end_s in intervals:
        mat = interval_spike_matrix(spike_times_s, spike_electrodes, elec_to_idx, start_s, end_s, bin_ms)
        if mat.shape[0] < 3:
            continue
        target = window_matrix(mat, lag[0], lag[1], bin_ms).astype(np.float32, copy=False)
        mat_f = mat.astype(np.float32, copy=False)
        ntrig = mat.sum(axis=0).astype(np.float64)
        obs_hits += mat_f.T @ target
        trig += ntrig
        if config.null_method == "rate_expectation":
            p_target = target.mean(axis=0).astype(np.float64)
            null_hits += ntrig[:, None] * p_target[None, :]
        else:
            acc = np.zeros((n, n), dtype=np.float64)
            for _ in range(n_surr):
                shifts = rng.integers(-jitter_bins, jitter_bins + 1, size=mat.shape[1])
                shifted = circular_shift_columns(mat, shifts)
                null_target = window_matrix(shifted, lag[0], lag[1], bin_ms).astype(np.float32, copy=False)
                acc += mat_f.T @ null_target
            null_hits += acc / max(1, n_surr)

    denom = trig[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        p_obs = np.where(denom > 0, obs_hits / denom, np.nan)
        p_null = np.where(denom > 0, null_hits / denom, np.nan)
    excess = p_obs - p_null

    valid_ref = trig >= float(config.min_trigger_spikes)
    ref_idx, target_idx = np.where(~np.eye(n, dtype=bool))
    keep = valid_ref[ref_idx]
    ref_idx, target_idx = ref_idx[keep], target_idx[keep]

    diff = coords[target_idx] - coords[ref_idx]
    distance = np.sqrt(np.sum(diff * diff, axis=1))
    angle = np.mod(np.arctan2(diff[:, 1], diff[:, 0]), np.pi)
    ex = excess[ref_idx, target_idx]
    po = p_obs[ref_idx, target_idx]

    in_range = (
        np.isfinite(ex)
        & (distance >= float(config.min_distance_um))
        & (distance <= float(config.max_distance_um))
    )
    return PairSynchrony(
        excess=ex[in_range],
        p_obs=po[in_range],
        distance_um=distance[in_range],
        angle_rad=angle[in_range],
        n_trigger=trig[ref_idx][in_range],
    )


# --------------------------------------------------------------------------- #
# anisotropy index (option A: de-confounded cos(2 theta) amplitude)
# --------------------------------------------------------------------------- #
def cos2_anisotropy_index(
    excess: np.ndarray,
    distance_um: np.ndarray,
    angle_rad: np.ndarray,
    *,
    distance_range: tuple[float, float] = (150.0, 2000.0),
    ring_um: float = 200.0,
    min_pairs: int = 200,
) -> dict:
    """Anisotropy magnitude as the cos(2*theta) amplitude of ring-demeaned excess.

    Per-distance-ring demeaning removes the isotropic radial part ``a0(r)`` (the
    decay/oscillation) so the residual carries only orientation structure; the
    second-harmonic amplitude of that residual is the de-confounded anisotropy
    magnitude and its phase is the dominant synchrony axis. This isolates the
    angular signal from the within-band distance-orientation coupling.
    """
    excess = np.asarray(excess, dtype=float)
    distance_um = np.asarray(distance_um, dtype=float)
    angle_rad = np.asarray(angle_rad, dtype=float)
    lo, hi = float(distance_range[0]), float(distance_range[1])
    m = np.isfinite(excess) & (distance_um >= lo) & (distance_um < hi)
    if int(np.count_nonzero(m)) < int(min_pairs):
        return {"amplitude": np.nan, "axis_rad": np.nan, "axis_deg": np.nan, "n_pairs": int(np.count_nonzero(m))}
    th, vv, dd = angle_rad[m], excess[m], distance_um[m]
    edges = np.arange(lo, hi + ring_um, ring_um)
    ring = np.clip(np.searchsorted(edges, dd, side="right") - 1, 0, edges.size - 2)
    resid = vv.copy()
    for r in range(edges.size - 1):
        sel = ring == r
        if np.any(sel):
            resid[sel] = vv[sel] - np.nanmean(vv[sel])
    design = np.column_stack([np.ones_like(th), np.cos(2 * th), np.sin(2 * th)])
    coef, *_ = np.linalg.lstsq(design, resid, rcond=None)
    _, c2, s2 = coef
    amp = float(np.hypot(c2, s2))
    axis = float(np.mod(0.5 * np.arctan2(s2, c2), np.pi))
    return {"amplitude": amp, "axis_rad": axis, "axis_deg": float(np.degrees(axis)), "n_pairs": int(th.size)}


def orientation_shuffle_pvalue(
    excess, distance_um, angle_rad, *, distance_range=(150.0, 2000.0),
    ring_um=200.0, n_shuffle=1000, seed=0,
) -> float:
    """One-sided p-value: permute orientation labels, refit, compare amplitudes."""
    obs = cos2_anisotropy_index(excess, distance_um, angle_rad,
                                distance_range=distance_range, ring_um=ring_um)["amplitude"]
    if not np.isfinite(obs):
        return np.nan
    rng = np.random.default_rng(int(seed))
    angle_rad = np.asarray(angle_rad, dtype=float)
    ge = 0
    for _ in range(int(n_shuffle)):
        amp = cos2_anisotropy_index(excess, distance_um, rng.permutation(angle_rad),
                                    distance_range=distance_range, ring_um=ring_um)["amplitude"]
        if np.isfinite(amp) and amp >= obs:
            ge += 1
    return (ge + 1) / (int(n_shuffle) + 1)


# --------------------------------------------------------------------------- #
# orientation x distance profile (for the radial-by-orientation plot)
# --------------------------------------------------------------------------- #
def orientation_distance_profile(
    values: np.ndarray,
    distance_um: np.ndarray,
    angle_rad: np.ndarray,
    *,
    n_orientation_bins: int = 6,
    distance_edges: np.ndarray | None = None,
    min_distance_um: float = 50.0,
    max_distance_um: float = 3500.0,
    distance_step_um: float = 200.0,
    min_pairs_per_cell: int = 20,
) -> dict:
    """Mean ``values`` per (orientation, distance) cell; under-powered cells -> NaN."""
    values = np.asarray(values, dtype=float)
    distance_um = np.asarray(distance_um, dtype=float)
    angle_rad = np.asarray(angle_rad, dtype=float)
    ang_edges = orientation_bin_edges(n_orientation_bins)
    if distance_edges is None:
        distance_edges = np.arange(min_distance_um, max_distance_um + distance_step_um, distance_step_um)
    nd = distance_edges.size - 1
    ai = np.clip(np.searchsorted(ang_edges, angle_rad, side="right") - 1, 0, n_orientation_bins - 1)
    di = np.searchsorted(distance_edges, distance_um, side="right") - 1
    grid = np.full((n_orientation_bins, nd), np.nan)
    count = np.zeros((n_orientation_bins, nd), dtype=int)
    for a in range(n_orientation_bins):
        for d in range(nd):
            sel = (ai == a) & (di == d) & np.isfinite(values)
            c = int(np.count_nonzero(sel))
            count[a, d] = c
            if c >= int(min_pairs_per_cell):
                grid[a, d] = float(np.mean(values[sel]))
    return {
        "grid": grid,
        "count": count,
        "orientation_centers_rad": 0.5 * (ang_edges[:-1] + ang_edges[1:]),
        "distance_centers_um": 0.5 * (distance_edges[:-1] + distance_edges[1:]),
    }


def align_to_axis(angle_rad: np.ndarray, axis_rad: float) -> np.ndarray:
    """Orientation relative to a recording's own axis, folded to ``[0, pi)``."""
    return np.mod(np.asarray(angle_rad, dtype=float) - float(axis_rad), np.pi)


def aggregate_amplitude(amplitudes: Iterable[float]) -> dict:
    """Mean / SEM of per-recording anisotropy amplitudes (axis-invariant)."""
    a = np.asarray([x for x in amplitudes], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": np.nan, "sem": np.nan, "n": 0}
    sem = float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else np.nan
    return {"mean": float(np.mean(a)), "sem": sem, "n": int(a.size)}
