from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase_waves import (
    SPATIAL_CLASS,
    WAVE_MODEL_CLASS,
    bandpass_downsample,
    classify_wave_model,
    classify_spatial_aperture,
    compute_phase_velocity,
    demodulated_window_phasor,
    estimate_omega,
    extract_phasors,
    fit_k_radial_phasor,
    fit_k_phasor_plane,
    fit_pgd_plane,
    inspect_file,
    load_chunk,
    make_k_grid,
    make_neighbor_edges,
    make_radial_grid,
    phase_gradient_alignment,
    prepare_k_grid_basis,
    process_file,
    process_pgd_file,
    wavelength_edge_flags,
    _direction_stability_mask,
)


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


def test_fit_k_radial_phasor_recovers_synthetic_source_and_wavenumber():
    x, y = np.meshgrid(np.linspace(0.0, 3.0, 12), np.linspace(0.0, 2.0, 9))
    coords = np.column_stack((x.ravel(), y.ravel()))
    true_center = np.array([1.0, 0.8])
    true_k = 2.0 * np.pi / 1.4
    distances = np.linalg.norm(coords - true_center[None, :], axis=1)
    ubar = np.exp(1j * true_k * distances)
    weights = np.ones(coords.shape[0])
    radial_grid = make_radial_grid(
        coords,
        center_grid_n=13,
        center_margin_mm=0.0,
        lambda_min_mm=1.0,
        lambda_max_mm=2.0,
        k_grid_n=31,
    )

    fit = fit_k_radial_phasor(ubar, weights, coords, radial_grid, refine=True)

    assert fit["R_radial"] > 0.98
    np.testing.assert_allclose([fit["radial_x0_mm"], fit["radial_y0_mm"]], true_center, atol=0.12)
    np.testing.assert_allclose(fit["radial_k"], true_k, atol=0.15)
    assert fit["radial_sign"] == 1


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


def test_fit_pgd_plane_recovers_synthetic_plane_wave():
    x, y = np.meshgrid(np.linspace(0.0, 2.5, 12), np.linspace(0.0, 1.5, 8))
    coords = np.column_stack((x.ravel(), y.ravel()))
    true_k = np.array([2.0, -1.0])
    phasor = np.exp(1j * (coords @ true_k))
    weights = np.ones(coords.shape[0])
    k_grid = make_k_grid(lambda_min_mm=1.0, lambda_max_mm=10.0, n_grid=41)
    k_grid = prepare_k_grid_basis(k_grid, coords)
    edges = make_neighbor_edges(coords, n_neighbors=4)

    fit = fit_pgd_plane(phasor, weights, coords, k_grid, gradient_edges=edges, refine=True)

    assert fit["pgd"] > 0.95
    assert fit["pgd_gradient_alignment"] > 0.90
    np.testing.assert_allclose([fit["bx"], fit["by"]], true_k, atol=0.08)


def test_phase_gradient_alignment_low_for_random_spatial_phase():
    rng = np.random.default_rng(1)
    x, y = np.meshgrid(np.linspace(0.0, 2.5, 12), np.linspace(0.0, 1.5, 8))
    coords = np.column_stack((x.ravel(), y.ravel()))
    theta = rng.uniform(-np.pi, np.pi, coords.shape[0])
    edges = make_neighbor_edges(coords, n_neighbors=4)

    pgd = phase_gradient_alignment(theta, coords, edges)

    assert np.isfinite(pgd)
    assert pgd < 0.5


def test_fit_pgd_plane_low_for_random_spatial_phase():
    rng = np.random.default_rng(2)
    x, y = np.meshgrid(np.linspace(0.0, 3.0, 16), np.linspace(0.0, 2.0, 12))
    coords = np.column_stack((x.ravel(), y.ravel()))
    phasor = np.exp(1j * rng.uniform(-np.pi, np.pi, coords.shape[0]))
    weights = np.ones(coords.shape[0])
    k_grid = prepare_k_grid_basis(make_k_grid(lambda_min_mm=1.0, lambda_max_mm=10.0, n_grid=21), coords)
    edges = make_neighbor_edges(coords, n_neighbors=4)

    fit = fit_pgd_plane(phasor, weights, coords, k_grid, gradient_edges=edges, refine=False)

    assert np.isfinite(fit["pgd"])
    assert fit["pgd"] < 0.5


def test_pgd_direction_stability_rejects_fast_jumps():
    direction_rad = np.deg2rad([0.0, 1.0, 2.0, 50.0, 51.0, 52.0])

    stable = _direction_stability_mask(direction_rad, fs_ds=1000.0, max_deg_per_ms=3.0)

    assert stable.tolist() == [True, True, False, False, True, True]


def test_classify_spatial_aperture_uses_projected_array_extent():
    coords = np.column_stack((np.linspace(0.0, 2.5, 6), np.zeros(6)))

    weak = classify_spatial_aperture(2.0 * np.pi / 10.0, 0.0, coords)
    assert weak["spatial_class"] == SPATIAL_CLASS["near_sync"]
    np.testing.assert_allclose(weak["aperture_along_k_mm"], 2.5)
    np.testing.assert_allclose(weak["cycles_across_array"], 0.25)

    resolvable = classify_spatial_aperture(2.0 * np.pi / 2.5, 0.0, coords)
    assert resolvable["spatial_class"] == SPATIAL_CLASS["resolvable_wave"]
    np.testing.assert_allclose(resolvable["cycles_across_array"], 1.0)

    invalid = classify_spatial_aperture(0.0, 0.0, coords)
    assert invalid["spatial_class"] == SPATIAL_CLASS["invalid"]
    assert np.isnan(invalid["aperture_along_k_mm"])


def test_compute_phase_velocity_uses_constant_phase_direction():
    velocity = compute_phase_velocity(2.0, 0.0, 10.0, f_peak=10.0 / (2.0 * np.pi))

    np.testing.assert_allclose(velocity["v_phi_mm_per_s"], 5.0)
    np.testing.assert_allclose(velocity["phase_speed_mm_per_s"], 5.0)
    np.testing.assert_allclose(velocity["phase_speed_peak_mm_per_s"], 5.0)
    np.testing.assert_allclose(velocity["velocity_x_mm_per_s"], -5.0)
    np.testing.assert_allclose(velocity["velocity_y_mm_per_s"], 0.0)
    np.testing.assert_allclose(abs(velocity["velocity_direction_rad"]), np.pi)

    reversed_velocity = compute_phase_velocity(2.0, 0.0, -10.0)
    np.testing.assert_allclose(reversed_velocity["v_phi_mm_per_s"], -5.0)
    np.testing.assert_allclose(reversed_velocity["phase_speed_mm_per_s"], 5.0)
    np.testing.assert_allclose(reversed_velocity["velocity_x_mm_per_s"], 5.0)


def test_wavelength_edge_flags_identifies_censored_speeds():
    assert wavelength_edge_flags(4.95, 0.5, 5.0, 0.98)["at_lambda_max"] == 1
    assert wavelength_edge_flags(4.95, 0.5, 5.0, 0.98)["speed_censored"] == 1
    assert wavelength_edge_flags(0.50, 0.5, 5.0, 0.98)["at_lambda_min"] == 1
    assert wavelength_edge_flags(2.0, 0.5, 5.0, 0.98) == {
        "at_lambda_min": 0,
        "at_lambda_max": 0,
        "speed_censored": 0,
    }


def test_classify_wave_model_compares_planar_and_radial_support():
    assert (
        classify_wave_model(
            planar_valid=True,
            radial_valid=True,
            spatial_class=SPATIAL_CLASS["resolvable_wave"],
            radial_cycles_across_array=1.0,
            delta_r=0.08,
            delta_r_min=0.05,
        )
        == WAVE_MODEL_CLASS["radial_like"]
    )
    assert (
        classify_wave_model(
            planar_valid=True,
            radial_valid=True,
            spatial_class=SPATIAL_CLASS["resolvable_wave"],
            radial_cycles_across_array=1.0,
            delta_r=-0.08,
            delta_r_min=0.05,
        )
        == WAVE_MODEL_CLASS["planar_like"]
    )
    assert (
        classify_wave_model(
            planar_valid=False,
            radial_valid=False,
            spatial_class=SPATIAL_CLASS["near_sync"],
            radial_cycles_across_array=0.1,
            delta_r=np.nan,
        )
        == WAVE_MODEL_CLASS["near_sync"]
    )


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


def test_process_file_streams_hdf5_windows(tmp_path):
    path = tmp_path / "synthetic.raw.h5"
    out = tmp_path / "phase_waves.h5"
    _write_synthetic_maxtwo(path)

    process_file(
        path,
        "data0000",
        out,
        fs_ds=250.0,
        band_low=30.0,
        band_high=80.0,
        chunk_s=0.5,
        overlap_s=0.1,
        win_ms=100.0,
        hop_ms=50.0,
        lambda_min=1.0,
        lambda_max=10.0,
        k_grid_n=17,
        refine=False,
        start_s=0.0,
        stop_s=1.0,
        max_channels_per_block=3,
        fit_radial=True,
        radial_center_grid_n=5,
        radial_center_margin_mm=0.0,
        speed_min_cycles=1.0,
        lambda_edge_fraction=0.98,
        speed_min_r=0.18,
    )

    with h5py.File(out, "r") as h5:
        group = h5["phase_waves"]
        n = group["t_center_s"].shape[0]
        assert n > 2
        for name in ("kx", "ky", "k_norm", "lambda_mm", "direction_x", "direction_y", "direction_rad", "aperture_along_k_mm", "phase_span_rad", "cycles_across_array", "spatial_class", "propagation_valid", "omega", "v_phi_mm_per_s", "phase_speed_mm_per_s", "phase_speed_peak_mm_per_s", "velocity_x_mm_per_s", "velocity_y_mm_per_s", "velocity_direction_rad", "planar_at_lambda_min", "planar_at_lambda_max", "planar_speed_censored", "planar_velocity_valid", "radial_x0_mm", "radial_y0_mm", "radial_k", "radial_lambda_mm", "radial_sign", "radial_aperture_mm", "radial_phase_span_rad", "radial_cycles_across_array", "R_radial", "delta_R_radial_minus_planar", "radial_phase_speed_mm_per_s", "radial_at_lambda_min", "radial_at_lambda_max", "radial_speed_censored", "radial_velocity_valid", "wave_model_class", "radial_valid", "R_fit", "valid", "interval_idx"):
            assert group[name].shape == (n,)
        assert np.isfinite(group["R_fit"][:]).any()
        assert np.isfinite(group["phase_speed_mm_per_s"][:]).any()
        assert set(group["interval_idx"][:].tolist()) == {0}
        assert set(group["spatial_class"][:].tolist()).issubset(set(SPATIAL_CLASS.values()))
        assert set(group["propagation_valid"][:].tolist()).issubset({0, 1})
        assert set(group["wave_model_class"][:].tolist()).issubset(set(WAVE_MODEL_CLASS.values()))
        assert set(group["radial_valid"][:].tolist()).issubset({0, 1})
        assert set(group["radial_speed_censored"][:].tolist()).issubset({0, 1})
        assert set(group["radial_velocity_valid"][:].tolist()).issubset({0, 1})
        assert set(group["planar_velocity_valid"][:].tolist()).issubset({0, 1})
        assert "spatial_class_mapping" in group.attrs
        assert "wave_model_class_mapping" in group.attrs
        assert "radial_velocity_sign_convention" in group.attrs
        assert "velocity_sign_convention" in group.attrs
        assert "config" in group.attrs


def test_process_file_accepts_multiple_intervals(tmp_path):
    path = tmp_path / "synthetic.raw.h5"
    out = tmp_path / "phase_waves_intervals.h5"
    _write_synthetic_maxtwo(path)

    process_file(
        path,
        "data0000",
        out,
        fs_ds=250.0,
        band_low=30.0,
        band_high=80.0,
        chunk_s=0.3,
        overlap_s=0.05,
        win_ms=80.0,
        hop_ms=80.0,
        lambda_min=1.0,
        lambda_max=10.0,
        k_grid_n=9,
        refine=False,
        intervals=[(0.0, 0.35), (0.70, 1.0)],
        max_channels_per_block=3,
    )

    with h5py.File(out, "r") as h5:
        g = h5["phase_waves"]
        t = g["t_center_s"][:]
        assert t.size > 0
        assert set(g["interval_idx"][:].tolist()).issubset({0, 1})
        assert np.all(((t >= 0.0) & (t < 0.35)) | ((t >= 0.70) & (t < 1.0)))


def test_process_pgd_file_writes_streaming_outputs(tmp_path):
    path = tmp_path / "synthetic.raw.h5"
    out = tmp_path / "pgd_waves.h5"
    _write_synthetic_maxtwo(path)

    process_pgd_file(
        path,
        "data0000",
        out,
        fs_ds=250.0,
        band_low=30.0,
        band_high=80.0,
        chunk_s=0.3,
        overlap_s=0.05,
        analysis_pad_s=0.02,
        lambda_min=1.0,
        lambda_max=10.0,
        k_grid_n=13,
        refine=False,
        intervals=[(0.0, 0.35), (0.70, 1.0)],
        pgd_threshold=0.3,
        pgd_min_duration_ms=5.0,
        pgd_max_direction_change_deg_per_ms=20.0,
        max_channels_per_block=3,
    )

    with h5py.File(out, "r") as h5:
        assert "pgd_waves" in h5
        g = h5["pgd_waves"]
        for name in (
            "pgd",
            "pgd_r2_adj",
            "pgd_gradient_alignment",
            "bx",
            "by",
            "k_norm_pgd",
            "speed_pgd_mm_per_s",
            "pgd_valid",
            "interval_idx",
        ):
            assert name in g
        t = g["t_center_s"][:]
        assert t.size > 0
        assert np.isfinite(g["pgd"][:]).any()
        assert np.isfinite(g["speed_pgd_mm_per_s"][:]).any()
        assert set(g["interval_idx"][:].tolist()).issubset({0, 1})
        assert set(g["pgd_valid"][:].tolist()).issubset({0, 1})
        assert np.all(((t >= 0.0) & (t < 0.35)) | ((t >= 0.70) & (t < 1.0)))
        assert "pgd_convention" in g.attrs
        assert "validity_criteria" in g.attrs
