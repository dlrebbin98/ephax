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

from .synchrony_distance import (
    circular_shift_columns,
    spike_presence_matrix,
    weighted_event_bootstrap,
    window_spike_matrix,
)


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


# The 1 ms presence matrix, lag-window OR, and circular-shift surrogate helpers
# live in ``synchrony_distance`` (the distance-binned path) and are imported above
# so both paths share one implementation. This module only adds the orientation
# bookkeeping on top of that shared matrix core.


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
# per-interval coincidence counts (shared by the pooled and event-resolved paths)
# --------------------------------------------------------------------------- #
def _interval_hits(mat, lag_window_ms, bin_ms, null_method, jitter_bins, n_surr, rng):
    """Observed/null coincidence counts and trigger counts for one interval.

    Returns ``(obs_hits (n, n), null_hits (n, n), trig (n,))``. ``null_hits`` is the
    mean null coincidence count: the closed-form rate expectation, or the surrogate
    average for the interval-jitter null. Both the pooled per-pair estimate and the
    per-event grids accumulate the same quantity, so the two paths stay aligned.
    """
    target = window_spike_matrix(mat, lag_window_ms[0], lag_window_ms[1], bin_ms).astype(np.float32, copy=False)
    mat_f = mat.astype(np.float32, copy=False)
    trig = mat.sum(axis=0).astype(np.float64)
    obs_hits = (mat_f.T @ target).astype(np.float64)
    if null_method == "rate_expectation":
        p_target = target.mean(axis=0).astype(np.float64)
        null_hits = trig[:, None] * p_target[None, :]
    else:
        acc = np.zeros(obs_hits.shape, dtype=np.float64)
        for _ in range(n_surr):
            shifts = rng.integers(-jitter_bins, jitter_bins + 1, size=mat.shape[1])
            shifted = circular_shift_columns(mat, shifts)
            null_target = window_spike_matrix(shifted, lag_window_ms[0], lag_window_ms[1], bin_ms).astype(np.float32, copy=False)
            acc += mat_f.T @ null_target
        null_hits = acc / max(1, n_surr)
    return obs_hits, null_hits, trig


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
        mat = spike_presence_matrix(spike_times_s, spike_electrodes, elec_to_idx, start_s, end_s, bin_ms)
        if mat.shape[0] < 3:
            continue
        o, nu, tr = _interval_hits(mat, lag, bin_ms, config.null_method, jitter_bins, n_surr, rng)
        obs_hits += o
        null_hits += nu
        trig += tr

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


# --------------------------------------------------------------------------- #
# event-resolved path: per-event orientation x distance grids for within-culture
# pooling (events, not wells, are the unit of replication)
# --------------------------------------------------------------------------- #
@dataclass
class DirectionalEventGrids:
    """Per-event excess-synchrony grids, axis-aligned to one recording's own axis.

    ``sum_excess`` and ``count`` are ``(n_events, n_orientation, n_distance)``; each
    event is one analysis interval. Orientation is *relative to this recording's
    anisotropy axis* (so grids from different wells can be summed without their axes
    cancelling). Pooling across wells = stacking events; the bootstrap resamples
    events. The pooled cell value is ``sum_excess.sum / count.sum``.
    """

    sum_excess: np.ndarray
    count: np.ndarray
    axis_rad: float
    orientation_centers_rad: np.ndarray
    distance_centers_um: np.ndarray


def _amplitude_distance_edges(config: DirectionalSyncConfig) -> np.ndarray:
    lo, hi = float(config.index_distance_um[0]), float(config.index_distance_um[1])
    return np.arange(lo, hi + float(config.index_ring_um), float(config.index_ring_um))


def compute_directional_synchrony(
    spike_times_s: np.ndarray,
    spike_electrodes: np.ndarray,
    coords_by_electrode: Mapping[int, Sequence[float]],
    electrodes: Iterable[int],
    intervals: Iterable[tuple[float, float]],
    config: DirectionalSyncConfig,
    rng: np.random.Generator | None = None,
    *,
    axis_rad: float | None = None,
    orientation_edges: np.ndarray | None = None,
    distance_edges: np.ndarray | None = None,
    absolute_grids: bool = False,
) -> tuple[PairSynchrony, DirectionalEventGrids]:
    """Pooled per-pair synchrony **and** per-event axis-aligned grids in one pass.

    Replaces a separate :func:`compute_pair_synchrony` call: the returned
    ``PairSynchrony`` is the interval-pooled per-pair estimate (full distance range,
    for the radial/orientation curves) and the ``DirectionalEventGrids`` carry the
    per-event orientation x distance excess needed for within-culture event pooling.

    The recording's anisotropy axis is estimated once with the cheap closed-form
    null (axis is robust to the null choice) unless ``axis_rad`` is supplied, then
    the per-event grids are accumulated under ``config.null_method``.

    If ``absolute_grids`` is True the same single pass also accumulates a second set
    of grids binned by **absolute** orientation (``axis_rad = 0``, i.e. physical angle
    on the array, 0 rad = +x), and the return becomes a 3-tuple
    ``(pair_sync, aligned_grids, absolute_grids)``. Absolute grids can be pooled
    across wells to test for a *common physical* axis (the aligned grids cannot).
    """
    if config.null_method not in {"interval_jitter", "rate_expectation"}:
        raise ValueError("null_method must be 'interval_jitter' or 'rate_expectation'")
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    electrodes = np.asarray([int(e) for e in electrodes if int(e) in coords_by_electrode], dtype=int)
    bin_ms = float(config.matrix_bin_ms)
    lag = config.lag_window_ms
    jitter_bins = max(1, int(round(float(config.jitter_ms) / bin_ms)))
    n_surr = int(config.n_surrogates)

    if orientation_edges is None:
        orientation_edges = orientation_bin_edges(config.n_orientation_bins)
    if distance_edges is None:
        distance_edges = _amplitude_distance_edges(config)
    orientation_edges = np.asarray(orientation_edges, dtype=float)
    distance_edges = np.asarray(distance_edges, dtype=float)
    n_orient = orientation_edges.size - 1
    n_dist = distance_edges.size - 1
    orient_centers = 0.5 * (orientation_edges[:-1] + orientation_edges[1:])
    dist_centers = 0.5 * (distance_edges[:-1] + distance_edges[1:])

    empty_grids = DirectionalEventGrids(
        np.zeros((0, n_orient, n_dist)), np.zeros((0, n_orient, n_dist), dtype=int),
        float("nan"), orient_centers, dist_centers,
    )
    if electrodes.size < 2:
        empty = np.array([], dtype=float)
        empty_ps = PairSynchrony(empty, empty, empty, empty, empty)
        if absolute_grids:
            return empty_ps, empty_grids, empty_grids
        return empty_ps, empty_grids

    electrodes = np.unique(electrodes)
    elec_to_idx = {int(e): i for i, e in enumerate(electrodes)}
    coords = np.array([coords_by_electrode[int(e)] for e in electrodes], dtype=float)
    n = electrodes.size
    spike_times_s = np.asarray(spike_times_s, dtype=float)
    spike_electrodes = np.asarray(spike_electrodes, dtype=int)
    intervals = list(intervals)

    ref_idx, target_idx = np.where(~np.eye(n, dtype=bool))
    diff = coords[target_idx] - coords[ref_idx]
    distance = np.sqrt(np.sum(diff * diff, axis=1))
    angle = np.mod(np.arctan2(diff[:, 1], diff[:, 0]), np.pi)

    # axis: cheap closed-form pass unless supplied
    if axis_rad is None:
        ps_axis = compute_pair_synchrony(
            spike_times_s, spike_electrodes, coords_by_electrode, electrodes, intervals,
            DirectionalSyncConfig(**{**config.__dict__, "null_method": "rate_expectation"}),
            np.random.default_rng(config.random_seed),
        )
        axis_rad = cos2_anisotropy_index(
            ps_axis.excess, ps_axis.distance_um, ps_axis.angle_rad,
            distance_range=config.index_distance_um, ring_um=config.index_ring_um,
        )["axis_rad"]
    axis_rad = float(axis_rad) if np.isfinite(axis_rad) else 0.0

    # per-pair cell assignment in orientation relative to this recording's axis
    rel = align_to_axis(angle, axis_rad)
    ai = np.clip(np.searchsorted(orientation_edges, rel, side="right") - 1, 0, n_orient - 1)
    di = np.searchsorted(distance_edges, distance, side="right") - 1
    in_grid = (di >= 0) & (di < n_dist)
    g_ref = ref_idx[in_grid]
    g_tgt = target_idx[in_grid]
    g_cell = (ai[in_grid] * n_dist + di[in_grid]).astype(int)
    min_trig = float(config.min_trigger_spikes)

    if absolute_grids:
        ai_abs = np.clip(np.searchsorted(orientation_edges, angle, side="right") - 1, 0, n_orient - 1)
        g_cell_abs = (ai_abs[in_grid] * n_dist + di[in_grid]).astype(int)
    sum_excess_abs: list[np.ndarray] = []
    count_abs: list[np.ndarray] = []

    obs_hits = np.zeros((n, n), dtype=np.float64)
    null_hits = np.zeros((n, n), dtype=np.float64)
    trig = np.zeros(n, dtype=np.float64)
    sum_excess_events: list[np.ndarray] = []
    count_events: list[np.ndarray] = []

    for start_s, end_s in intervals:
        mat = spike_presence_matrix(spike_times_s, spike_electrodes, elec_to_idx, start_s, end_s, bin_ms)
        if mat.shape[0] < 3:
            continue
        o, nu, tr = _interval_hits(mat, lag, bin_ms, config.null_method, jitter_bins, n_surr, rng)
        obs_hits += o
        null_hits += nu
        trig += tr

        trig_ref = tr[g_ref]
        with np.errstate(invalid="ignore", divide="ignore"):
            ex = (o[g_ref, g_tgt] - nu[g_ref, g_tgt]) / trig_ref
        ok = (trig_ref >= min_trig) & np.isfinite(ex)
        se = np.zeros(n_orient * n_dist, dtype=np.float64)
        cn = np.zeros(n_orient * n_dist, dtype=np.float64)
        np.add.at(se, g_cell[ok], ex[ok])
        np.add.at(cn, g_cell[ok], 1.0)
        sum_excess_events.append(se.reshape(n_orient, n_dist))
        count_events.append(cn.reshape(n_orient, n_dist))
        if absolute_grids:
            se_a = np.zeros(n_orient * n_dist, dtype=np.float64)
            cn_a = np.zeros(n_orient * n_dist, dtype=np.float64)
            np.add.at(se_a, g_cell_abs[ok], ex[ok])
            np.add.at(cn_a, g_cell_abs[ok], 1.0)
            sum_excess_abs.append(se_a.reshape(n_orient, n_dist))
            count_abs.append(cn_a.reshape(n_orient, n_dist))

    denom = trig[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        p_obs_mat = np.where(denom > 0, obs_hits / denom, np.nan)
        p_null_mat = np.where(denom > 0, null_hits / denom, np.nan)
    excess_mat = p_obs_mat - p_null_mat

    valid_ref = trig >= min_trig
    keep = valid_ref[ref_idx]
    kref, ktgt = ref_idx[keep], target_idx[keep]
    kdist, kangle = distance[keep], angle[keep]
    kex = excess_mat[kref, ktgt]
    kpo = p_obs_mat[kref, ktgt]
    in_range = (
        np.isfinite(kex)
        & (kdist >= float(config.min_distance_um))
        & (kdist <= float(config.max_distance_um))
    )
    pair_sync = PairSynchrony(
        excess=kex[in_range],
        p_obs=kpo[in_range],
        distance_um=kdist[in_range],
        angle_rad=kangle[in_range],
        n_trigger=trig[kref][in_range],
    )
    grids = DirectionalEventGrids(
        sum_excess=np.array(sum_excess_events) if sum_excess_events else np.zeros((0, n_orient, n_dist)),
        count=(np.array(count_events) if count_events else np.zeros((0, n_orient, n_dist))).astype(int),
        axis_rad=axis_rad,
        orientation_centers_rad=orient_centers,
        distance_centers_um=dist_centers,
    )
    if absolute_grids:
        grids_absolute = DirectionalEventGrids(
            sum_excess=np.array(sum_excess_abs) if sum_excess_abs else np.zeros((0, n_orient, n_dist)),
            count=(np.array(count_abs) if count_abs else np.zeros((0, n_orient, n_dist))).astype(int),
            axis_rad=0.0,
            orientation_centers_rad=orient_centers,
            distance_centers_um=dist_centers,
        )
        return pair_sync, grids, grids_absolute
    return pair_sync, grids


def cos2_amplitude_from_grid(
    mean_grid: np.ndarray,
    orientation_centers_rad: np.ndarray,
    distance_centers_um: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    min_cells: int = 6,
    distance_range: tuple[float, float] | None = None,
) -> dict:
    """cos(2*theta) amplitude/axis of a ring-demeaned orientation x distance grid.

    The grid analogue of :func:`cos2_anisotropy_index`: demean each distance column
    over orientation (removes the isotropic radial part) and fit the second angular
    harmonic of the residual, optionally weighting cells by their pair counts.
    ``distance_range`` restricts the fit to a coverage-complete distance band (the
    grid may span a wider range for the radial/polar views).
    """
    g = np.array(mean_grid, dtype=float)
    resid = g.copy()
    for d in range(g.shape[1]):
        col = g[:, d]
        m = np.isfinite(col)
        if m.any():
            resid[m, d] = col[m] - col[m].mean()
    th = np.repeat(np.asarray(orientation_centers_rad, dtype=float)[:, None], g.shape[1], axis=1)
    dd = np.repeat(np.asarray(distance_centers_um, dtype=float)[None, :], g.shape[0], axis=0)
    flat_resid = resid.ravel()
    flat_th = th.ravel()
    w = np.ones_like(flat_resid) if weights is None else np.asarray(weights, dtype=float).ravel()
    valid = np.isfinite(flat_resid) & np.isfinite(w) & (w > 0)
    if distance_range is not None:
        valid &= (dd.ravel() >= float(distance_range[0])) & (dd.ravel() < float(distance_range[1]))
    if int(np.count_nonzero(valid)) < int(min_cells):
        return {"amplitude": np.nan, "axis_rad": np.nan, "axis_deg": np.nan, "n_cells": int(np.count_nonzero(valid))}
    design = np.column_stack([np.ones(int(valid.sum())), np.cos(2 * flat_th[valid]), np.sin(2 * flat_th[valid])])
    sw = np.sqrt(w[valid])
    coef, *_ = np.linalg.lstsq(design * sw[:, None], flat_resid[valid] * sw, rcond=None)
    _, c2, s2 = coef
    amp = float(np.hypot(c2, s2))
    axis = float(np.mod(0.5 * np.arctan2(s2, c2), np.pi))
    return {"amplitude": amp, "axis_rad": axis, "axis_deg": float(np.degrees(axis)), "n_cells": int(valid.sum())}


def pooled_anisotropy_across_wells(
    event_grids: Iterable[DirectionalEventGrids],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    ci: tuple[float, float] = (2.5, 97.5),
    min_cells: int = 6,
    distance_range: tuple[float, float] | None = None,
) -> dict:
    """Within-culture pooled anisotropy amplitude with an event-level bootstrap.

    ``event_grids`` are the per-well :class:`DirectionalEventGrids` of one culture
    (one DIV); each well's grids are already aligned to its own axis. Events are
    stacked across wells and resampled with replacement, trading the n=wells mean
    CI for event-level power. Returns the pooled amplitude, its bootstrap CI/SE, and
    the per-well amplitudes (same grid estimator) for an overlay.
    """
    grids = [g for g in event_grids if g is not None and g.sum_excess.shape[0] > 0]
    if not grids:
        return {"amplitude": np.nan, "axis_deg": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "se": np.nan, "n_events": 0, "n_wells": 0, "per_well_amplitude": [], "boot_amplitude": np.array([])}
    oc = grids[0].orientation_centers_rad
    dc = grids[0].distance_centers_um
    n_orient, n_dist = oc.size, dc.size

    sum_excess = np.concatenate([g.sum_excess.reshape(g.sum_excess.shape[0], -1) for g in grids], axis=0)
    count = np.concatenate([g.count.reshape(g.count.shape[0], -1) for g in grids], axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(count > 0, sum_excess / np.where(count > 0, count, 1.0), np.nan)

    res = weighted_event_bootstrap(values, count, n_boot=n_boot, seed=seed, ci=ci, return_boot=True)
    fit_weights = res["point_weight"].reshape(n_orient, n_dist)

    def _amp(point_flat):
        return cos2_amplitude_from_grid(point_flat.reshape(n_orient, n_dist), oc, dc,
                                        weights=fit_weights, min_cells=min_cells,
                                        distance_range=distance_range)["amplitude"]

    point = cos2_amplitude_from_grid(res["point"].reshape(n_orient, n_dist), oc, dc,
                                     weights=fit_weights, min_cells=min_cells,
                                     distance_range=distance_range)
    boot = res.get("boot", np.zeros((0, n_orient * n_dist)))
    boot_amp = np.array([_amp(boot[b]) for b in range(boot.shape[0])], dtype=float)
    boot_amp = boot_amp[np.isfinite(boot_amp)]

    per_well = []
    for g in grids:
        s = g.sum_excess.sum(axis=0)
        c = g.count.astype(float).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mg = np.where(c > 0, s / np.where(c > 0, c, 1.0), np.nan)
        per_well.append(cos2_amplitude_from_grid(mg, oc, dc, weights=c, min_cells=min_cells,
                                                 distance_range=distance_range)["amplitude"])

    return {
        "amplitude": point["amplitude"],
        "axis_deg": point["axis_deg"],
        "ci_lo": float(np.nanpercentile(boot_amp, ci[0])) if boot_amp.size else np.nan,
        "ci_hi": float(np.nanpercentile(boot_amp, ci[1])) if boot_amp.size else np.nan,
        "se": float(np.nanstd(boot_amp, ddof=1)) if boot_amp.size > 1 else np.nan,
        "n_events": int(sum_excess.shape[0]),
        "n_wells": len(grids),
        "per_well_amplitude": per_well,
        "boot_amplitude": boot_amp,
    }


def pooled_orientation_distance_grid(
    event_grids: Iterable[DirectionalEventGrids],
    *,
    min_count: int = 0,
) -> dict | None:
    """Event-pooled mean excess on the orientation x distance grid.

    Stacks all events of all wells and returns the count-weighted mean per cell —
    the 2-D map behind the radial/polar view. Works for either the axis-aligned or
    the absolute-orientation grids (whichever were passed in). Cells with fewer than
    ``min_count`` pooled pairs are masked to NaN to suppress sparse, noisy cells
    (e.g. long-distance bins where only a few orientations have pairs).
    """
    grids = [g for g in event_grids if g is not None and g.sum_excess.shape[0] > 0]
    if not grids:
        return None
    s = sum(g.sum_excess.sum(axis=0) for g in grids)
    c = sum(g.count.astype(float).sum(axis=0) for g in grids)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > float(min_count), s / np.where(c > 0, c, 1.0), np.nan)
    return {
        "grid": mean,
        "count": c,
        "orientation_centers_rad": grids[0].orientation_centers_rad,
        "distance_centers_um": grids[0].distance_centers_um,
    }


def pooled_distance_curve(
    event_grids: Iterable[DirectionalEventGrids],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    ci: tuple[float, float] = (2.5, 97.5),
    orientation_indices: Sequence[int] | None = None,
    return_boot: bool = False,
) -> dict:
    """Excess synchrony vs distance (orientation summed) with an event bootstrap.

    Sums over orientation and pools events across wells, so the CI reflects the
    event-level sample size — much tighter than a per-well-mean interval. Pass
    ``orientation_indices`` to restrict to an orientation **sector** (e.g. the
    absolute "vertical" vs "horizontal" bins) for a direction-resolved distance
    curve; omit it for the fully isotropic curve. With ``return_boot`` the per-
    bootstrap curves are returned too (for deriving a CI on a derived quantity such
    as the interference wavelength).
    """
    grids = [g for g in event_grids if g is not None and g.sum_excess.shape[0] > 0]
    if not grids:
        out = {"distance_centers_um": np.array([]), "excess": np.array([]),
               "ci_lo": np.array([]), "ci_hi": np.array([]), "se": np.array([]),
               "n_pairs": np.array([]), "n_events": 0}
        if return_boot:
            out["boot"] = np.zeros((0, 0))
        return out
    dc = grids[0].distance_centers_um
    if orientation_indices is None:
        sel = slice(None)
    else:
        sel = np.asarray(list(orientation_indices), dtype=int)
    sum_excess = np.concatenate([g.sum_excess[:, sel, :].sum(axis=1) for g in grids], axis=0)   # (E, n_dist)
    count = np.concatenate([g.count.astype(float)[:, sel, :].sum(axis=1) for g in grids], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(count > 0, sum_excess / np.where(count > 0, count, 1.0), np.nan)
    res = weighted_event_bootstrap(values, count, n_boot=n_boot, seed=seed, ci=ci, return_boot=return_boot)
    out = {
        "distance_centers_um": dc,
        "excess": res["point"],
        "ci_lo": res["ci_lo"],
        "ci_hi": res["ci_hi"],
        "se": res["se"],
        "n_pairs": res["point_weight"],
        "n_events": int(res["n_events"]),
    }
    if return_boot:
        out["boot"] = res.get("boot", np.zeros((0, dc.size)))
    return out


# --------------------------------------------------------------------------- #
# spatial periodicity: 1-D spectrum of the synchrony-vs-distance curve and
# 2-D autocorrelation of a spatial field (interference-pattern evidence)
# --------------------------------------------------------------------------- #
def _detrend_radial(distance_um: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaN and remove the monotonic radial trend (decay) before spectral analysis."""
    r = np.asarray(distance_um, dtype=float)
    v = np.asarray(values, dtype=float)
    m = np.isfinite(r) & np.isfinite(v)
    r, v = r[m], v[m]
    if r.size < 4:
        return r, (v - v.mean() if v.size else v)
    try:
        from scipy.optimize import curve_fit

        l0 = max((r.max() - r.min()) / 2.0, 1.0)
        popt, _ = curve_fit(
            lambda x, a, l, c: a * np.exp(-x / l) + c, r, v,
            p0=[float(v.max() - v.min()), l0, float(v.min())], maxfev=10000,
        )
        resid = v - (popt[0] * np.exp(-r / popt[1]) + popt[2])
    except Exception:
        resid = v - np.polyval(np.polyfit(r, v, 2), r)
    return r, resid


def radial_power_spectrum(
    distance_um: np.ndarray,
    values: np.ndarray,
    *,
    n_wavelength: int = 256,
    min_wavelength_um: float | None = None,
    max_wavelength_um: float | None = None,
    detrend: bool = True,
) -> dict:
    """Spatial power spectrum of an excess-synchrony-vs-distance curve.

    The interference hypothesis predicts a peak at the model wavelength. The radial
    decay is removed first (so a monotonic curve has no spectral peak), then the
    periodogram is evaluated directly at a dense grid of candidate wavelengths
    (robust to the few, uniformly-spaced distance bins).

    ``max_wavelength_um`` is capped to **half the measured distance span** by default:
    wavelengths longer than that span less than ~2 cycles over the array and are not
    resolvable, so the search is restricted to the resolvable band. Raise it (toward
    the full span) only to *look at* the under-resolved long-wavelength end knowingly.
    """
    if detrend:
        r, resid = _detrend_radial(distance_um, values)
    else:
        r = np.asarray(distance_um, dtype=float)
        v = np.asarray(values, dtype=float)
        m = np.isfinite(r) & np.isfinite(v)
        r, resid = r[m], v[m] - (v[m].mean() if m.any() else 0.0)
    if r.size < 4:
        return {"wavelength_um": np.array([]), "power": np.array([]),
                "peak_wavelength_um": np.nan, "peak_power": np.nan,
                "distance_um": r, "residual": resid}
    span = float(r.max() - r.min())
    dx = float(np.median(np.diff(np.sort(r))))
    lo = float(min_wavelength_um) if min_wavelength_um else 2.0 * dx
    hi = float(max_wavelength_um) if max_wavelength_um else 0.5 * span
    wl = np.linspace(lo, hi, int(n_wavelength))
    freqs = 1.0 / wl
    resid = resid - resid.mean()
    power = np.array([np.abs(np.sum(resid * np.exp(-2j * np.pi * f * r))) ** 2 for f in freqs]) / r.size
    k = int(np.argmax(power))
    return {"wavelength_um": wl, "power": power, "peak_wavelength_um": float(wl[k]),
            "peak_power": float(power[k]), "distance_um": r, "residual": resid}


def interference_periodicity_test(
    distance_um: np.ndarray,
    values: np.ndarray,
    *,
    n_perm: int = 1000,
    seed: int = 0,
    **spectrum_kwargs,
) -> dict:
    """Permutation test that the radial spectrum has a real periodic peak.

    The null shuffles the detrended residual across distance bins (destroying spatial
    order); ``p_value`` is the fraction of shuffles whose peak power matches or exceeds
    the observed peak. Returns the full spectrum plus the peak wavelength and p-value.
    """
    spec = radial_power_spectrum(distance_um, values, **spectrum_kwargs)
    if spec["wavelength_um"].size == 0:
        return {**spec, "p_value": np.nan}
    r = spec["distance_um"]
    resid = spec["residual"]
    freqs = 1.0 / spec["wavelength_um"]
    obs = spec["peak_power"]
    rng = np.random.default_rng(int(seed))
    ge = 0
    for _ in range(int(n_perm)):
        vp = rng.permutation(resid)
        power = np.array([np.abs(np.sum(vp * np.exp(-2j * np.pi * f * r))) ** 2 for f in freqs]) / r.size
        if power.max() >= obs:
            ge += 1
    return {**spec, "p_value": (ge + 1) / (int(n_perm) + 1)}


# --------------------------------------------------------------------------- #
# 2-D autocorrelation of a spatial field (banded interference-pattern evidence)
# --------------------------------------------------------------------------- #
def masked_autocorr_2d(grid: np.ndarray) -> dict:
    """Normalized 2-D spatial autocorrelation of a field with NaN (unsampled) cells.

    Overlap-normalized so missing cells do not bias the estimate; the result is scaled
    to 1 at zero lag. Periodic side-lobes reveal a spatial wavelength, and their angle
    its orientation — direct evidence of banded structure.
    """
    from scipy.signal import fftconvolve

    g = np.asarray(grid, dtype=float)
    mask = np.isfinite(g)
    a = np.where(mask, g - np.nanmean(g), 0.0)
    num = fftconvolve(a, a[::-1, ::-1], mode="full")
    cnt = fftconvolve(mask.astype(float), mask[::-1, ::-1].astype(float), mode="full")
    with np.errstate(invalid="ignore", divide="ignore"):
        ac = np.where(cnt > 0.5, num / np.where(cnt > 0.5, cnt, 1.0), np.nan)
    ny, nx = g.shape
    c0 = ac[ny - 1, nx - 1]
    if np.isfinite(c0) and c0 != 0:
        ac = ac / c0
    return {"autocorr": ac, "lag_x": np.arange(-(nx - 1), nx), "lag_y": np.arange(-(ny - 1), ny)}


def radial_profile_2d(
    values_2d: np.ndarray,
    lag_x: np.ndarray,
    lag_y: np.ndarray,
    *,
    pixel_um: float,
    bin_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Radial (angle-averaged) profile of a 2-D map indexed by integer lags."""
    yy, xx = np.meshgrid(np.asarray(lag_y) * pixel_um, np.asarray(lag_x) * pixel_um, indexing="ij")
    rr = np.sqrt(xx ** 2 + yy ** 2).ravel()
    v = np.asarray(values_2d, dtype=float).ravel()
    edges = np.arange(0.0, np.nanmax(rr) + bin_um, bin_um)
    idx = np.clip(np.searchsorted(edges, rr, side="right") - 1, 0, edges.size - 2)
    prof = np.full(edges.size - 1, np.nan)
    for b in range(edges.size - 1):
        sel = (idx == b) & np.isfinite(v)
        if sel.any():
            prof[b] = float(v[sel].mean())
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, prof
