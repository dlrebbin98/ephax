import numpy as np
import pytest

from ephax.metrics.directional_synchrony import (
    DirectionalSyncConfig,
    align_to_axis,
    axial_angles,
    compute_pair_synchrony,
    cos2_anisotropy_index,
    orientation_distance_profile,
    orientation_shuffle_pvalue,
)


def test_axial_angles_fold_reverse_pairs_together():
    coords = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    ref = np.array([0, 1, 0, 2])
    tgt = np.array([1, 0, 2, 0])
    ang = axial_angles(coords, ref, tgt)
    # i->j and j->i (horizontal) collapse to the same axial angle (0 rad)
    assert ang[0] == pytest.approx(ang[1])
    # horizontal pair ~ 0, vertical pair ~ pi/2
    assert ang[0] == pytest.approx(0.0, abs=1e-9)
    assert ang[2] == pytest.approx(np.pi / 2, abs=1e-9)


def test_align_to_axis_is_relative_and_folded():
    ang = np.array([0.0, np.pi / 2, 0.9 * np.pi])
    rel = align_to_axis(ang, axis_rad=np.pi / 2)
    assert np.all((rel >= 0) & (rel < np.pi))
    assert rel[1] == pytest.approx(0.0, abs=1e-9)


def _synthetic_pairs(amplitude, axis_rad, n=6000, seed=1, decay=True):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, np.pi, size=n)
    dist = rng.uniform(200.0, 1800.0, size=n)
    base = (0.1 * np.exp(-dist / 1200.0)) if decay else np.full(n, 0.1)
    excess = base + amplitude * np.cos(2 * (ang - axis_rad)) + rng.normal(0, 1e-4, size=n)
    return excess, dist, ang


def test_index_recovers_injected_amplitude_and_axis():
    amp_true, axis_true = 0.02, np.deg2rad(140.0)
    excess, dist, ang = _synthetic_pairs(amp_true, axis_true)
    out = cos2_anisotropy_index(excess, dist, ang, distance_range=(200.0, 1800.0), ring_um=200.0)
    assert out["amplitude"] == pytest.approx(amp_true, rel=0.15)
    # axis is axial: compare modulo pi
    diff = abs(out["axis_rad"] - axis_true) % np.pi
    diff = min(diff, np.pi - diff)
    assert diff < np.deg2rad(8.0)


def test_ring_demeaning_removes_radial_trend_isotropic_is_null():
    # strong radial decay, zero true anisotropy -> de-confounded amplitude ~ 0
    excess, dist, ang = _synthetic_pairs(0.0, 0.0, decay=True)
    out = cos2_anisotropy_index(excess, dist, ang, distance_range=(200.0, 1800.0), ring_um=200.0)
    assert out["amplitude"] < 0.002


def test_shuffle_pvalue_significant_for_real_anisotropy():
    excess, dist, ang = _synthetic_pairs(0.02, np.deg2rad(60.0))
    p = orientation_shuffle_pvalue(excess, dist, ang, distance_range=(200.0, 1800.0),
                                   ring_um=200.0, n_shuffle=200, seed=0)
    assert p < 0.05


def test_orientation_distance_profile_shapes_and_masking():
    excess, dist, ang = _synthetic_pairs(0.02, 0.0)
    prof = orientation_distance_profile(
        excess, dist, ang, n_orientation_bins=6,
        min_distance_um=200.0, max_distance_um=1800.0, distance_step_um=200.0,
        min_pairs_per_cell=10,
    )
    assert prof["grid"].shape == (6, prof["distance_centers_um"].size)
    assert prof["orientation_centers_rad"].size == 6
    assert np.isfinite(prof["grid"]).any()


def _two_train_recording(lag_ms, sf_hz=1000.0, dur_s=60.0, rate_hz=20.0, seed=0):
    """Two electrodes: electrode 1 fires a copy of electrode 0 shifted by lag."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * rate_hz)
    t0 = np.sort(rng.uniform(0, dur_s, size=n))
    t1 = t0 + lag_ms / 1000.0
    times = np.concatenate([t0, t1])
    elec = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    order = np.argsort(times)
    coords = {0: (0.0, 0.0), 1: (100.0, 0.0)}
    return times[order], elec[order], coords


def test_compute_pair_synchrony_detects_coincidence():
    times, elec, coords = _two_train_recording(lag_ms=1.0)
    cfg = DirectionalSyncConfig(null_method="rate_expectation", min_trigger_spikes=5,
                                min_distance_um=10.0, lag_window_ms=(-3.0, 3.0))
    ps = compute_pair_synchrony(times, elec, coords, electrodes=[0, 1],
                                intervals=[(0.0, 60.0)], config=cfg)
    assert ps.excess.size >= 1
    # tightly coupled trains -> strongly positive excess synchrony
    assert np.nanmean(ps.excess) > 0.5
    # all pairs are horizontal -> axial angle 0
    assert np.allclose(ps.angle_rad, 0.0, atol=1e-9)


def test_compute_pair_synchrony_jitter_null_runs():
    times, elec, coords = _two_train_recording(lag_ms=1.0)
    cfg = DirectionalSyncConfig(null_method="interval_jitter", n_surrogates=10,
                                min_trigger_spikes=5, min_distance_um=10.0,
                                lag_window_ms=(-3.0, 3.0), jitter_ms=25.0)
    ps = compute_pair_synchrony(times, elec, coords, electrodes=[0, 1],
                                intervals=[(0.0, 60.0)], config=cfg)
    assert ps.excess.size >= 1
    assert np.isfinite(ps.excess).all()
