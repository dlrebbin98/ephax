from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ephax.metrics.lfp import (
    LFPWavefrontCacheConfig,
    analyze_excitable_phase_front,
    append_phasor_feature_segment,
    append_wavefront_rows,
    bandpass_downsample,
    combine_excitable_phase_calibrations,
    demodulated_window_phasor,
    detect_phase_crossings,
    estimate_excitable_phase,
    estimate_local_phase_velocity,
    estimate_omega,
    extract_phasors,
    finalize_phasor_feature_cache,
    fit_arrival_time_plane,
    fit_arrival_time_radial,
    fit_k_phasor_plane,
    fit_local_arrival_velocity,
    initialize_phasor_feature_cache,
    initialize_wavefront_cache,
    inspect_file,
    lfp_wavefront_cache_status,
    load_chunk,
    make_k_grid,
    make_local_phase_neighborhoods,
    make_neighbor_edges,
    merge_time_intervals,
    phase_gradient_alignment,
    phasor_feature_cache_matches,
    read_lfp_wavefront_cache,
    read_phasor_feature_interval,
    read_phasor_feature_metadata,
    require_lfp_wavefront_caches,
    select_coherent_phase_front,
    select_representative_event_indices,
)
from ephax.plotting.lfp import save_lfp_phase_gif

def test_phasor_feature_cache_round_trip_and_interval_merge(tmp_path):
    path = tmp_path / "phasor_features.h5"
    coords = np.array([[0.0, 0.0], [0.1, 0.2]])
    electrodes = np.array([10, 11])
    time_s = 1.0 + np.arange(11) / 500.0
    u = np.exp(1j * 2.0 * np.pi * 45.0 * time_s[:, None]) * np.array([[1.0, 1j]])
    config = {"band_low": 30.0, "band_high": 50.0, "fs_ds": 500.0}

    merged = merge_time_intervals([(1.0, 1.01), (1.005, 1.02), (2.0, 2.1)])
    assert merged == [(1.0, 1.02), (2.0, 2.1)]

    initialize_phasor_feature_cache(path, coords, electrodes, config=config)
    amplitude = np.linspace(1.0, 2.0, time_s.size)[:, None] * np.array([[1.0, 2.0]])
    append_phasor_feature_segment(path, 1.0, time_s, u, amplitude)
    assert not phasor_feature_cache_matches(path, config)
    finalize_phasor_feature_cache(path)
    assert phasor_feature_cache_matches(path, config)
    assert not phasor_feature_cache_matches(path, {**config, "fs_ds": 250.0})
    metadata = read_phasor_feature_metadata(path)
    assert metadata["complete"]
    assert metadata["segments"][0]["n_samples"] == time_s.size
    assert metadata["segments"][0]["has_amplitude"]

    cached_u, cached_t, cached_coords, cached_electrodes = read_phasor_feature_interval(path, 1.004, 1.012)
    cached_u_amp, cached_t_amp, _, _, cached_amp = read_phasor_feature_interval(
        path, 1.004, 1.012, include_amplitude=True
    )

    np.testing.assert_allclose(cached_t, time_s[2:6])
    np.testing.assert_allclose(cached_u, u[2:6], atol=1e-6)
    np.testing.assert_allclose(cached_u_amp, u[2:6], atol=1e-6)
    np.testing.assert_allclose(cached_t_amp, time_s[2:6])
    np.testing.assert_allclose(cached_amp, amplitude[2:6], rtol=1e-6)
    np.testing.assert_allclose(cached_coords, coords)
    np.testing.assert_array_equal(cached_electrodes, electrodes)

def test_phasor_feature_cache_partial_resume_does_not_duplicate_segments(tmp_path):
    path = tmp_path / "phasor_features_partial.h5"
    coords = np.array([[0.0, 0.0], [0.1, 0.2]])
    electrodes = np.array([10, 11])
    config = {
        "profile": "permissive",
        "calibration_event_ids": [1, 3],
        "scoring_event_ids": [2],
        "band_low": 30.0,
        "band_high": 50.0,
        "fs_ds": 500.0,
        "filter_pad_s": 0.25,
    }
    first_time = 1.0 + np.arange(6) / 500.0
    second_time = 2.0 + np.arange(6) / 500.0
    u = np.ones((6, 2), dtype=np.complex64)

    initialize_phasor_feature_cache(path, coords, electrodes, config=config)
    append_phasor_feature_segment(path, 1.0, first_time, u)
    partial = read_phasor_feature_metadata(path)
    assert not partial["complete"]
    assert len(partial["segments"]) == 1

    append_phasor_feature_segment(path, 2.0, second_time, u)
    finalize_phasor_feature_cache(path)
    restored = read_phasor_feature_metadata(path)
    assert restored["complete"]
    assert [segment["start_s"] for segment in restored["segments"]] == [1.0, 2.0]
    assert not phasor_feature_cache_matches(path, {**config, "profile": "confirm"})
    assert not phasor_feature_cache_matches(path, {**config, "scoring_event_ids": [4]})
    assert not phasor_feature_cache_matches(path, {**config, "filter_pad_s": 0.5})

def test_select_representative_event_indices_prefers_disjoint_sets():
    selected = select_representative_event_indices(80, n_calibration=25, n_scoring=15)

    assert selected["calibration_indices"].size == 25
    assert selected["scoring_indices"].size == 15
    assert selected["overlap_count"] == 0
    assert np.intersect1d(selected["calibration_indices"], selected["scoring_indices"]).size == 0
    np.testing.assert_array_equal(
        selected["scoring_indices"],
        select_representative_event_indices(80, n_calibration=25, n_scoring=15)["scoring_indices"],
    )

def test_select_representative_event_indices_reports_low_count_overlap():
    selected = select_representative_event_indices(20, n_calibration=15, n_scoring=10)

    assert selected["calibration_indices"].size == 15
    assert selected["scoring_indices"].size == 10
    assert selected["overlap_count"] == 5

def test_select_representative_event_indices_confirm_uses_all_for_calibration():
    selected = select_representative_event_indices(20, n_calibration=None, n_scoring=8)

    np.testing.assert_array_equal(selected["calibration_indices"], np.arange(20))
    assert selected["scoring_indices"].size == 8
    assert selected["overlap_count"] == 8

def test_fit_k_phasor_plane_recovers_synthetic_wave_vector():
    x, y = np.meshgrid(np.linspace(0.0, 2.0, 8), np.linspace(0.0, 1.5, 7))
    coords = np.column_stack((x.ravel(), y.ravel()))
    true_k = np.array([2.4, -1.6])
    ubar = np.exp(1j * (coords @ true_k))
    weights = np.ones(coords.shape[0])
    k_grid = make_k_grid(lambda_min_mm=1.2, lambda_max_mm=10.0, n_grid=41)

    fit = fit_k_phasor_plane(ubar, weights, coords, k_grid, refine=True)

    assert fit["R_fit"] > 0.99
    np.testing.assert_allclose([fit["kx"], fit["ky"]], true_k, atol=0.08)

def test_extract_phasors_handles_zero_amplitude_channels():
    data = np.column_stack(
        [
            np.sin(np.linspace(0.0, 4.0 * np.pi, 200)),
            np.zeros(200),
        ]
    )

    u, amp = extract_phasors(data)

    assert u.shape == data.shape
    assert amp.shape == data.shape
    assert np.all(np.isfinite(u.real))
    assert np.allclose(u[:, 1], 0.0)
    assert np.nanmean(np.abs(u[:, 0])) > 0.99

def test_demodulated_window_phasor_prevents_temporal_cancellation():
    fs = 500.0
    t = np.arange(100) / fs
    coords = np.column_stack((np.linspace(0.0, 2.0, 12), np.zeros(12)))
    true_k = np.array([2.0, 0.0])
    omega = 2.0 * np.pi * 40.0
    u = np.exp(1j * (omega * t[:, None] + coords @ true_k))
    amp = np.ones_like(u.real)

    naive = np.mean(u, axis=0)
    demod = demodulated_window_phasor(u, amp, fs, omega)
    est_omega, _ = estimate_omega(u, None, fs, method="channel_phase_derivative")

    assert np.mean(np.abs(naive)) < 0.1
    assert np.mean(np.abs(demod)) > 0.99
    assert abs(est_omega - omega) < 1e-6

def test_bandpass_downsample_rejects_band_above_downsampled_nyquist():
    data = np.random.default_rng(0).normal(size=(2000, 4))

    with pytest.raises(ValueError, match="downsampled Nyquist"):
        bandpass_downsample(data, fs_raw=1000.0, band_low=80.0, band_high=180.0, fs_ds=300.0)

def test_phase_gradient_alignment_low_for_random_spatial_phase():
    rng = np.random.default_rng(1)
    x, y = np.meshgrid(np.linspace(0.0, 2.5, 12), np.linspace(0.0, 1.5, 8))
    coords = np.column_stack((x.ravel(), y.ravel()))
    theta = rng.uniform(-np.pi, np.pi, coords.shape[0])
    edges = make_neighbor_edges(coords, n_neighbors=4)

    pgd = phase_gradient_alignment(theta, coords, edges)

    assert np.isfinite(pgd)
    assert pgd < 0.5

def test_estimate_local_phase_velocity_recovers_synthetic_plane_wave():
    fs = 500.0
    t = np.arange(40) / fs
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8))
    coords = np.column_stack((x.ravel(), y.ravel()))
    true_k = np.array([4.0, -2.0])
    true_omega = 2.0 * np.pi * 45.0
    u = np.exp(1j * (true_omega * t[:, None] + coords @ true_k))
    amp = np.ones_like(u.real)
    neighborhoods = make_local_phase_neighborhoods(
        coords,
        radius_mm=0.35,
        min_neighbors=8,
        max_radius_mm=0.50,
        max_neighbors=16,
    )

    local = estimate_local_phase_velocity(
        u,
        amp,
        coords,
        neighborhoods,
        fs_ds=fs,
        omega_smooth_ms=8.0,
        min_r_local=0.95,
        min_cycles_local=0.0,
    )

    valid = local["valid"]
    assert np.count_nonzero(valid) > 0
    np.testing.assert_allclose(np.nanmedian(local["kx"][valid]), true_k[0], atol=0.08)
    np.testing.assert_allclose(np.nanmedian(local["ky"][valid]), true_k[1], atol=0.08)
    np.testing.assert_allclose(
        np.nanmedian(local["speed_mm_per_s"][valid]),
        abs(true_omega) / np.linalg.norm(true_k),
        rtol=0.02,
    )

def test_make_local_phase_neighborhoods_rejects_collinear_geometry():
    coords = np.column_stack((np.linspace(0.0, 1.0, 20), np.zeros(20)))

    neighborhoods = make_local_phase_neighborhoods(
        coords,
        radius_mm=0.5,
        min_neighbors=4,
        max_radius_mm=0.8,
        max_neighbors=10,
        min_geometry_ratio=0.1,
    )

    assert not np.any(neighborhoods["valid"])

def test_estimate_excitable_phase_corrects_phase_occupancy():
    phase = np.concatenate([np.full(700, -1.0), np.full(300, 1.0)])
    u = np.exp(1j * phase)[:, None]
    time_s = np.arange(phase.size, dtype=float) / 1000.0
    electrodes = np.array([11])
    spike_idx = np.concatenate([np.arange(0, 700, 35), np.arange(700, 1000, 5)])

    calibration = estimate_excitable_phase(
        u,
        time_s,
        electrodes,
        time_s[spike_idx],
        np.full(spike_idx.size, 11),
        n_bins=36,
    )

    assert abs(np.angle(np.exp(1j * (calibration["theta_excitable_rad"] - 1.0)))) < 0.2
    combined = combine_excitable_phase_calibrations([calibration, calibration])
    np.testing.assert_allclose(combined["theta_excitable_rad"], calibration["theta_excitable_rad"])

def test_detect_phase_crossings_interpolates_subsample_times():
    time_s = np.array([0.0, 0.01, 0.02])
    theta = np.array([-0.5, 0.5, 1.5])
    u = np.exp(1j * theta)[:, None]

    crossings = detect_phase_crossings(u, time_s, 0.0)

    np.testing.assert_allclose(crossings[0], [0.005], atol=1e-12)

def test_select_coherent_phase_front_does_not_mix_neighboring_cycles():
    anchor = 1.0
    period = 1.0 / 45.0
    crossing = anchor + np.linspace(-0.003, 0.003, 80)
    crossings = [np.array([value - period, value, value + period]) for value in crossing]

    front = select_coherent_phase_front(crossings, anchor, min_electrodes=50, match_max_ms=8.0)

    assert front["valid"] == 1
    assert front["n_electrodes"] == 80
    assert abs(front["front_time_s"] - anchor) < 1e-6
    np.testing.assert_allclose(front["arrival_time_s"], crossing)

def test_local_arrival_velocity_recovers_planar_front():
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10))
    coords = np.column_stack((x.ravel(), y.ravel()))
    gradient = np.array([0.004, -0.002])
    arrival = 1.0 + coords @ gradient
    neighborhoods = make_local_phase_neighborhoods(
        coords,
        radius_mm=0.35,
        min_neighbors=8,
        max_radius_mm=0.5,
        max_neighbors=20,
    )

    local = fit_local_arrival_velocity(arrival, coords, neighborhoods, min_neighbors=8)
    valid = local["valid"]
    expected_velocity = gradient / np.sum(gradient * gradient)

    assert np.count_nonzero(valid) > 0
    np.testing.assert_allclose(np.nanmedian(local["velocity_x_mm_per_s"][valid]), expected_velocity[0], rtol=0.02)
    np.testing.assert_allclose(np.nanmedian(local["velocity_y_mm_per_s"][valid]), expected_velocity[1], rtol=0.02)
    global_fit = fit_arrival_time_plane(arrival, coords)
    np.testing.assert_allclose(global_fit["planar_speed_mm_per_s"], 1.0 / np.linalg.norm(gradient), rtol=0.01)

def test_arrival_time_radial_fit_recovers_source_and_speed():
    x, y = np.meshgrid(np.linspace(0.0, 3.0, 14), np.linspace(0.0, 2.0, 10))
    coords = np.column_stack((x.ravel(), y.ravel()))
    center = np.array([1.4, 0.9])
    speed = 120.0
    arrival = 1.0 + np.linalg.norm(coords - center, axis=1) / speed

    fit = fit_arrival_time_radial(arrival, coords, center_grid_n=31)

    np.testing.assert_allclose([fit["radial_x0_mm"], fit["radial_y0_mm"]], center, atol=0.12)
    np.testing.assert_allclose(fit["radial_speed_mm_per_s"], speed, rtol=0.05)
    assert fit["radial_sign"] == 1

def test_noisy_local_arrivals_are_retained_but_fail_qc():
    rng = np.random.default_rng(5)
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8))
    coords = np.column_stack((x.ravel(), y.ravel()))
    arrival = 1.0 + coords @ np.array([0.004, 0.002]) + rng.normal(0.0, 0.010, coords.shape[0])
    neighborhoods = make_local_phase_neighborhoods(coords, radius_mm=0.4, min_neighbors=8, max_radius_mm=0.5)

    local = fit_local_arrival_velocity(arrival, coords, neighborhoods, min_neighbors=8, max_residual_ms=1.0)

    assert np.isfinite(local["speed_mm_per_s"]).any()
    assert not np.any(local["valid"])

def test_fast_local_arrivals_are_retained_but_marked_censored():
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10))
    coords = np.column_stack((x.ravel(), y.ravel()))
    arrival = 1.0 + coords[:, 0] / 1000.0
    neighborhoods = make_local_phase_neighborhoods(
        coords,
        radius_mm=0.35,
        min_neighbors=8,
        max_radius_mm=0.5,
        max_neighbors=20,
    )

    local = fit_local_arrival_velocity(
        arrival,
        coords,
        neighborhoods,
        min_neighbors=8,
        min_arrival_span_ms=0.0,
        max_speed_mm_per_s=500.0,
    )

    finite = np.isfinite(local["speed_mm_per_s"])
    assert np.any(finite)
    assert np.all(local["speed_censored"][finite])
    assert not np.any(local["valid"])
    np.testing.assert_allclose(np.nanmedian(local["arrival_gradient_norm_s_per_mm"]), 1.0 / 1000.0, rtol=0.02)

def test_local_arrival_velocity_requires_spatial_support():
    x, y = np.meshgrid(np.linspace(0.0, 0.10, 10), np.linspace(0.0, 0.10, 10))
    coords = np.column_stack((x.ravel(), y.ravel()))
    arrival = 1.0 + coords[:, 0] / 100.0
    neighborhoods = make_local_phase_neighborhoods(
        coords,
        radius_mm=0.20,
        min_neighbors=8,
        max_radius_mm=0.25,
        max_neighbors=20,
    )

    local = fit_local_arrival_velocity(
        arrival,
        coords,
        neighborhoods,
        min_neighbors=8,
        min_arrival_span_ms=0.0,
        min_distance_span_mm=0.20,
    )

    assert np.isfinite(local["speed_mm_per_s"]).any()
    assert not np.any(local["valid"])

def test_local_arrival_velocity_applies_minimum_arrival_amplitude_qc():
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8))
    coords = np.column_stack((x.ravel(), y.ravel()))
    arrival = 1.0 + coords[:, 0] / 100.0
    neighborhoods = make_local_phase_neighborhoods(
        coords,
        radius_mm=0.35,
        min_neighbors=8,
        max_radius_mm=0.5,
        max_neighbors=20,
    )

    low_amp = fit_local_arrival_velocity(
        arrival,
        coords,
        neighborhoods,
        arrival_amplitude=np.full(arrival.shape, 0.5),
        min_arrival_amplitude=1.0,
        min_arrival_span_ms=0.0,
        min_distance_span_mm=0.0,
    )
    high_amp = fit_local_arrival_velocity(
        arrival,
        coords,
        neighborhoods,
        arrival_amplitude=np.full(arrival.shape, 2.0),
        min_arrival_amplitude=1.0,
        min_arrival_span_ms=0.0,
        min_distance_span_mm=0.0,
    )

    assert not np.isfinite(low_amp["speed_mm_per_s"]).any()
    assert not np.any(low_amp["valid"])
    assert np.any(high_amp["valid"])
    np.testing.assert_allclose(low_amp["arrival_amplitude"], 0.5)
    np.testing.assert_allclose(low_amp["arrival_amplitude_threshold"], 1.0)

def test_local_arrival_velocity_accepts_per_channel_arrival_amplitude_threshold():
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8))
    coords = np.column_stack((x.ravel(), y.ravel()))
    arrival = 1.0 + coords[:, 0] / 100.0
    neighborhoods = make_local_phase_neighborhoods(
        coords,
        radius_mm=0.35,
        min_neighbors=8,
        max_radius_mm=0.5,
        max_neighbors=20,
    )
    threshold = np.ones(arrival.shape, dtype=float)
    threshold[: arrival.size // 2] = 3.0

    local = fit_local_arrival_velocity(
        arrival,
        coords,
        neighborhoods,
        arrival_amplitude=np.full(arrival.shape, 2.0),
        min_arrival_amplitude=threshold,
        min_arrival_span_ms=0.0,
        min_distance_span_mm=0.0,
    )

    assert not np.any(local["valid"][: arrival.size // 2])
    assert np.any(local["valid"][arrival.size // 2 :])
    np.testing.assert_allclose(local["arrival_amplitude_threshold"], threshold)

def test_analyze_excitable_phase_front_recovers_selected_planar_crossing():
    fs = 1000.0
    time_s = np.arange(0.94, 1.06, 1.0 / fs)
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8))
    coords = np.column_stack((x.ravel(), y.ravel()))
    arrival = 1.0 + coords @ np.array([0.003, -0.001])
    omega = 2.0 * np.pi * 45.0
    u = np.exp(1j * omega * (time_s[:, None] - arrival[None, :]))
    neighborhoods = make_local_phase_neighborhoods(
        coords,
        radius_mm=0.35,
        min_neighbors=8,
        max_radius_mm=0.5,
        max_neighbors=20,
    )

    result = analyze_excitable_phase_front(
        u,
        time_s,
        coords,
        neighborhoods,
        theta_excitable_rad=0.0,
        anchor_time_s=1.0,
        min_electrodes=50,
        min_neighbors=8,
    )

    assert result["front"]["valid"] == 1
    np.testing.assert_allclose(result["front"]["arrival_time_s"], arrival, atol=1e-6)
    np.testing.assert_allclose(result["planar"]["planar_speed_mm_per_s"], 1.0 / np.hypot(0.003, -0.001), rtol=0.01)

def test_analyze_excitable_phase_front_applies_arrival_amplitude_percentile():
    fs = 1000.0
    time_s = np.arange(0.94, 1.06, 1.0 / fs)
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8))
    coords = np.column_stack((x.ravel(), y.ravel()))
    arrival = 1.0 + coords @ np.array([0.003, -0.001])
    omega = 2.0 * np.pi * 45.0
    u = np.exp(1j * omega * (time_s[:, None] - arrival[None, :]))
    amplitude = np.full(u.shape, 2.0, dtype=float)
    neighborhoods = make_local_phase_neighborhoods(
        coords,
        radius_mm=0.35,
        min_neighbors=8,
        max_radius_mm=0.5,
        max_neighbors=20,
    )

    result = analyze_excitable_phase_front(
        u,
        time_s,
        coords,
        neighborhoods,
        amplitude=amplitude,
        min_arrival_amplitude_percentile=50.0,
        theta_excitable_rad=0.0,
        anchor_time_s=1.0,
        min_electrodes=50,
        min_neighbors=8,
        min_arrival_span_ms=0.0,
        min_distance_span_mm=0.0,
    )

    assert result["front"]["valid"] == 1
    assert np.any(result["local"]["valid"])
    np.testing.assert_allclose(result["arrival_amplitude_threshold"], 2.0)

def test_wavefront_cache_round_trip(tmp_path):
    path = tmp_path / "wavefront.h5"
    calibration = {
        "phase_bin_edges_rad": np.linspace(-np.pi, np.pi, 5),
        "phase_bin_centers_rad": np.linspace(-2.0, 2.0, 4),
        "occupancy_counts": np.arange(4),
        "spike_phase_counts": np.arange(4) + 1,
        "relative_spike_probability": np.ones(4),
        "theta_excitable_rad": 0.5,
        "n_spikes_sampled": 10,
    }
    initialize_wavefront_cache(path, calibration, config={"band_low": 42.0, "band_high": 48.0})
    append_wavefront_rows(path, "wavefront_events", [{"event_idx": 1, "front_time_s": 2.0}])
    append_wavefront_rows(
        path,
        "wavefront_local",
        {"event_idx": np.array([1, 1]), "electrode": np.array([10, 11]), "speed_mm_per_s": np.array([90.0, 110.0])},
    )

    payload = read_lfp_wavefront_cache(path)

    assert set(payload) == {"wavefront_calibration", "wavefront_events", "wavefront_local"}
    np.testing.assert_allclose(payload["wavefront_calibration"]["theta_excitable_rad"], 0.5)
    np.testing.assert_allclose(payload["wavefront_events"]["front_time_s"], [2.0])
    np.testing.assert_allclose(payload["wavefront_local"]["speed_mm_per_s"], [90.0, 110.0])
    with pytest.raises(ValueError, match="match the existing cache schema"):
        append_wavefront_rows(path, "wavefront_events", [{"event_idx": 2}])

def _write_synthetic_maxtwo(path):
    fs = 1000.0
    n_channels = 6
    n_frames = 1000
    lsb = 1.0
    t = np.arange(n_frames) / fs
    x_um = np.linspace(0.0, 2500.0, n_channels)
    y_um = np.zeros(n_channels)
    kx = 2.0
    signal = np.sin(2.0 * np.pi * 40.0 * t[:, None] + kx * (x_um[None, :] / 1000.0))
    raw = np.round(signal.T * 1000.0).astype(np.int16).astype(np.uint16)

    mapping_dtype = np.dtype([("channel", "<i4"), ("electrode", "<i4"), ("x", "<f8"), ("y", "<f8")])
    mapping = np.zeros(n_channels, dtype=mapping_dtype)
    mapping["channel"] = np.arange(n_channels)
    mapping["electrode"] = 100 + np.arange(n_channels)
    mapping["x"] = x_um
    mapping["y"] = y_um

    with h5py.File(path, "w") as h5:
        rec = h5.create_group("data_store/data0000")
        group = rec.create_group("groups/all_channels")
        padded = np.zeros((1024, n_frames), dtype=np.uint16)
        padded[:n_channels] = raw
        group.create_dataset("raw", data=padded)
        group.create_dataset("channels", data=np.arange(1024, dtype=np.uint16))
        group.create_dataset("frame_nos", data=np.arange(n_frames, dtype=np.uint64))
        settings = rec.create_group("settings")
        settings.create_dataset("sampling", data=np.array([fs]))
        settings.create_dataset("lsb", data=np.array([lsb / 1000.0]))
        settings.create_dataset("mapping", data=mapping)

def _write_synthetic_routed_maxtwo(path):
    fs = 1000.0
    n_frames = 200
    lsb = 1.0
    routed_channels = np.array([10, 12], dtype=np.uint16)
    raw = np.vstack(
        [
            np.arange(n_frames, dtype=np.uint16),
            np.arange(n_frames, dtype=np.uint16) + 100,
        ]
    )

    mapping_dtype = np.dtype([("channel", "<i4"), ("electrode", "<i4"), ("x", "<f8"), ("y", "<f8")])
    mapping = np.zeros(2, dtype=mapping_dtype)
    mapping["channel"] = routed_channels
    mapping["electrode"] = [100, 101]
    mapping["x"] = [0.0, 1000.0]
    mapping["y"] = [0.0, 0.0]

    with h5py.File(path, "w") as h5:
        rec = h5.create_group("data_store/data0000")
        group = rec.create_group("groups/routed")
        group.create_dataset("raw", data=raw)
        group.create_dataset("channels", data=routed_channels)
        group.create_dataset("frame_nos", data=np.arange(n_frames, dtype=np.uint64))
        settings = rec.create_group("settings")
        settings.create_dataset("sampling", data=np.array([fs]))
        settings.create_dataset("lsb", data=np.array([lsb]))
        settings.create_dataset("mapping", data=mapping)

def test_load_chunk_inspects_and_converts_coordinates(tmp_path):
    path = tmp_path / "synthetic.raw.h5"
    _write_synthetic_maxtwo(path)

    info = inspect_file(path, "data0000")
    data, coords, fs, mapping = load_chunk(path, "data0000", 10, 100)

    assert info["raw_shape"] == (1024, 1000)
    assert data.shape == (100, 6)
    assert coords.shape == (6, 2)
    assert coords[:, 0].max() == 2.5
    assert fs == 1000.0
    assert mapping["channel"].tolist() == list(range(6))

def test_load_chunk_resolves_routed_data_store_and_well_aliases(tmp_path):
    path = tmp_path / "synthetic_routed.raw.h5"
    _write_synthetic_routed_maxtwo(path)

    info = inspect_file(path, "well000")
    data, coords, fs, mapping = load_chunk(path, "0", 5, 3)

    assert info["raw_path"] == "/data_store/data0000/groups/routed/raw"
    assert data.shape == (3, 2)
    np.testing.assert_allclose(data[:, 0], [5.0, 6.0, 7.0])
    np.testing.assert_allclose(data[:, 1], [105.0, 106.0, 107.0])
    np.testing.assert_allclose(coords[:, 0], [0.0, 1.0])
    assert fs == 1000.0
    assert mapping["channel"].tolist() == [10, 12]

def test_lfp_wavefront_cache_config_canonical_names():
    config = LFPWavefrontCacheConfig(
        source="stim_removal_null_lfp",
        profile="confirm",
        div=23,
        band_low_hz=30.0,
        band_high_hz=50.0,
        fs_hz=500.0,
        lambda_min_mm=0.5,
        lambda_max_mm=10.0,
        radial=True,
    )

    assert (
        config.wavefront_filename(4)
        == "wavefront_stim_removal_null_lfp_confirm_well4_DIV23_data0004_burst_30-50Hz_fs500_lambda0.5-10_radial1.h5"
    )
    assert (
        config.phasor_feature_filename(4)
        == "phasor_features_stim_removal_null_lfp_confirm_well4_DIV23_data0004_30-50Hz_fs500.h5"
    )

def test_require_lfp_wavefront_caches_rejects_noncanonical_or_incomplete(tmp_path):
    config = LFPWavefrontCacheConfig(
        source="stim_removal_null_lfp",
        profile="confirm",
        div=23,
        band_low_hz=30.0,
        band_high_hz=50.0,
        fs_hz=500.0,
        lambda_min_mm=0.5,
        lambda_max_mm=10.0,
        radial=True,
    )
    canonical = config.wavefront_path(tmp_path, 0)

    assert lfp_wavefront_cache_status(canonical) == (False, "missing")
    with pytest.raises(FileNotFoundError):
        require_lfp_wavefront_caches(config, tmp_path, [0], require_all=True)

    with h5py.File(canonical, "w") as h5:
        h5.create_group("wavefront_calibration")
        h5.create_group("wavefront_events")
        h5.create_group("wavefront_local")
        h5.attrs["complete"] = True

    assert lfp_wavefront_cache_status(canonical) == (True, "ok")
    assert require_lfp_wavefront_caches(config, tmp_path, [0], require_all=True) == {0: canonical}

def test_save_lfp_phase_gif_smoke(tmp_path):
    t = np.arange(4, dtype=float) / 500.0
    coords = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    phase = 2.0 * np.pi * 40.0 * t[:, None] + np.array([[0.0, 0.5, 1.0]])
    path = save_lfp_phase_gif(tmp_path / "lfp_phase.gif", phase, t, coords, frame_step=2, dpi=40)

    assert path.exists()
    assert path.stat().st_size > 0
