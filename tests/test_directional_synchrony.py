import numpy as np
import pytest

from ephax.metrics.directional_synchrony import (
    DirectionalSyncConfig,
    align_to_axis,
    axial_angles,
    compute_directional_synchrony,
    compute_pair_synchrony,
    cos2_amplitude_from_grid,
    cos2_anisotropy_index,
    interference_periodicity_test,
    masked_autocorr_2d,
    orientation_bin_edges,
    orientation_distance_profile,
    orientation_shuffle_pvalue,
    pooled_anisotropy_across_wells,
    radial_power_spectrum,
    radial_profile_2d,
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


def test_cos2_amplitude_from_grid_recovers_injected_amplitude_and_axis():
    edges = orientation_bin_edges(12)
    oc = 0.5 * (edges[:-1] + edges[1:])
    dc = np.array([200.0, 400.0, 600.0, 800.0])
    amp_true, axis_true = 0.05, np.deg2rad(40.0)
    grid = np.empty((oc.size, dc.size))
    for j, d in enumerate(dc):
        grid[:, j] = 0.1 * np.exp(-d / 500.0) + amp_true * np.cos(2 * (oc - axis_true))  # radial part demeaned away
    out = cos2_amplitude_from_grid(grid, oc, dc)
    assert out["amplitude"] == pytest.approx(amp_true, rel=0.1)
    diff = abs(out["axis_rad"] - axis_true) % np.pi
    assert min(diff, np.pi - diff) < np.deg2rad(8.0)


def _grid_recording(seed, n_el=36, dur_s=120.0, rate_hz=12.0, n_events=300):
    """6x6 array (200 um pitch) with coincident network events for synchrony tests."""
    rng = np.random.default_rng(seed)
    coords = {e: (200.0 * (e % 6), 200.0 * (e // 6)) for e in range(n_el)}
    times, elec = [], []
    for e in range(n_el):
        t = np.sort(rng.uniform(0, dur_s, int(rate_hz * dur_s)))
        times.append(t); elec.append(np.full(t.size, e))
    for a in np.sort(rng.uniform(1, dur_s - 1, n_events)):
        for e in range(n_el):
            if rng.random() < 0.6:
                times.append(np.array([a + rng.normal(0, 0.001)])); elec.append(np.array([e]))
    times = np.concatenate(times); elec = np.concatenate(elec).astype(int)
    order = np.argsort(times)
    return times[order], elec[order], coords


def test_compute_directional_synchrony_pooled_matches_pair_synchrony():
    times, elec, coords = _grid_recording(0)
    cfg = DirectionalSyncConfig(null_method="rate_expectation", min_trigger_spikes=5,
                                min_distance_um=50.0, max_distance_um=1200.0,
                                n_orientation_bins=12, index_distance_um=(150.0, 1000.0),
                                index_ring_um=200.0, lag_window_ms=(-5.0, 5.0))
    sel = np.arange(36)
    intervals = [(0.0, 60.0), (60.0, 120.0)]
    ps, grids = compute_directional_synchrony(times, elec, coords, sel, intervals, cfg,
                                              np.random.default_rng(0))
    ps_ref = compute_pair_synchrony(times, elec, coords, sel, intervals, cfg,
                                    np.random.default_rng(0))
    # the interval-pooled per-pair estimate is identical to the standalone path
    assert np.allclose(np.sort(ps.excess), np.sort(ps_ref.excess), equal_nan=True)
    # one grid per interval, shaped (n_events, n_orientation, n_distance)
    assert grids.sum_excess.shape[0] == 2
    assert grids.sum_excess.shape[1] == 12


def test_pooled_anisotropy_across_wells_runs_and_brackets_point():
    cfg = DirectionalSyncConfig(null_method="rate_expectation", min_trigger_spikes=5,
                                min_distance_um=50.0, max_distance_um=1200.0,
                                n_orientation_bins=12, index_distance_um=(150.0, 1000.0),
                                index_ring_um=200.0, lag_window_ms=(-5.0, 5.0))
    sel = np.arange(36)
    intervals = [(0.0, 40.0), (40.0, 80.0), (80.0, 120.0)]
    grids = []
    for w in range(4):
        times, elec, coords = _grid_recording(w)
        _, g = compute_directional_synchrony(times, elec, coords, sel, intervals, cfg,
                                             np.random.default_rng(0))
        grids.append(g)
    pooled = pooled_anisotropy_across_wells(grids, n_boot=200, seed=1)
    assert pooled["n_wells"] == 4
    assert pooled["n_events"] == 12          # 4 wells x 3 intervals, events are the unit
    assert len(pooled["per_well_amplitude"]) == 4
    assert np.isfinite(pooled["amplitude"])
    assert pooled["ci_lo"] <= pooled["amplitude"] <= pooled["ci_hi"]


def test_radial_power_spectrum_recovers_resolvable_wavelength():
    # a well-sampled wavelength (several cycles over the span) is recovered accurately;
    # the ~2540 um model band is ~1 cycle and inherently under-resolved (capped out).
    r = np.arange(150.0, 3500.0, 150.0)
    lam = 900.0
    excess = 0.12 * np.exp(-r / 1500.0) + 0.03 * np.cos(2 * np.pi * r / lam)
    spec = radial_power_spectrum(r, excess, min_wavelength_um=400.0, max_wavelength_um=1700.0)
    assert spec["peak_wavelength_um"] == pytest.approx(lam, rel=0.1)


def test_radial_power_spectrum_caps_max_wavelength_to_half_span():
    r = np.arange(150.0, 3500.0, 150.0)
    spec = radial_power_spectrum(r, np.exp(-r / 1500.0))   # default cap
    assert spec["wavelength_um"][-1] <= 0.5 * (r.max() - r.min()) + 1e-6


def test_interference_periodicity_test_significant_only_for_oscillation():
    r = np.arange(150.0, 3500.0, 150.0)
    osc = 0.12 * np.exp(-r / 1500.0) + 0.03 * np.cos(2 * np.pi * r / 900.0)
    flat = 0.12 * np.exp(-r / 1500.0) + np.random.default_rng(0).normal(0, 0.002, r.size)
    res_osc = interference_periodicity_test(r, osc, n_perm=400, seed=0,
                                            min_wavelength_um=400.0, max_wavelength_um=1700.0)
    res_flat = interference_periodicity_test(r, flat, n_perm=400, seed=0,
                                             min_wavelength_um=400.0, max_wavelength_um=1700.0)
    assert res_osc["p_value"] < 0.05
    assert res_flat["p_value"] > 0.05
    assert res_flat["peak_power"] < 0.2 * res_osc["peak_power"]


def test_masked_autocorr_2d_finds_grating_period():
    period_px = 5
    x = np.arange(40)
    grid = np.tile(np.cos(2 * np.pi * x / period_px), (20, 1))   # vertical grating, period along x
    grid[::7, ::11] = np.nan                                      # scattered missing cells
    out = masked_autocorr_2d(grid)
    ac = out["autocorr"]
    cy, cx = (ac.shape[0] - 1) // 2, (ac.shape[1] - 1) // 2
    assert ac[cy, cx] == pytest.approx(1.0, abs=1e-6)            # unit at zero lag
    # one period along +x is a positive side-lobe (anticorrelation at half period)
    assert ac[cy, cx + period_px] > 0.5
    assert ac[cy, cx + period_px // 2] < 0.0


def test_radial_profile_2d_shapes():
    out = masked_autocorr_2d(np.random.default_rng(0).normal(size=(12, 16)))
    centers, prof = radial_profile_2d(out["autocorr"], out["lag_x"], out["lag_y"],
                                      pixel_um=250.0, bin_um=250.0)
    assert centers.shape == prof.shape
    assert prof[0] == pytest.approx(1.0, abs=1e-6)               # zero-lag bin is the autocorr peak
