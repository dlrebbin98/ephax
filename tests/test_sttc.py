from __future__ import annotations

import numpy as np

from ephax.metrics.sttc import (
    compute_sttc_adjacency,
    concatenate_spike_times_over_intervals,
    read_sttc_network_cache,
    spike_time_tiling_coefficient,
    thresholded_edge_table,
    write_sttc_network_cache,
)


def test_identical_spike_trains_have_high_sttc():
    times = np.array([0.1, 0.5, 0.9, 1.3])
    value = spike_time_tiling_coefficient(times, times, duration_s=2.0, dt_s=0.01)

    assert value > 0.95


def test_shifted_sparse_spike_trains_are_near_zero_or_negative():
    a = np.array([0.1, 0.5, 0.9, 1.3])
    b = np.array([0.3, 0.7, 1.1, 1.5])
    value = spike_time_tiling_coefficient(a, b, duration_s=2.0, dt_s=0.01)

    assert value < 0.05


def test_interval_concatenation_removes_gaps():
    spikes = np.array([0.1, 0.9, 10.1, 10.9])
    intervals = np.array([[0.0, 1.0], [10.0, 11.0]])

    concatenated, duration = concatenate_spike_times_over_intervals(spikes, intervals)

    np.testing.assert_allclose(concatenated, [0.1, 0.9, 1.1, 1.9])
    assert duration == 2.0


def test_sttc_adjacency_is_symmetric_with_zero_diagonal():
    spikes = {
        1: np.array([0.1, 0.5, 0.9]),
        2: np.array([0.1, 0.5, 0.9]),
        3: np.array([0.2, 0.6, 1.0]),
    }

    result = compute_sttc_adjacency(spikes, duration_s=1.2, dt_ms=10.0, min_rate_hz=0.01)
    adjacency = result["adjacency"]

    assert adjacency.shape == (3, 3)
    np.testing.assert_allclose(adjacency, adjacency.T)
    np.testing.assert_allclose(np.diag(adjacency), 0.0)
    assert adjacency[0, 1] > 0.9


def test_low_rate_electrodes_are_excluded_from_adjacency():
    spikes = {
        1: np.array([0.1, 0.2, 0.3]),
        2: np.array([0.15]),
    }

    result = compute_sttc_adjacency(spikes, duration_s=100.0, dt_ms=10.0, min_rate_hz=0.02)

    assert result["electrodes"].tolist() == [1]
    assert result["adjacency"].shape == (1, 1)


def test_thresholded_edge_table_and_cache_roundtrip(tmp_path):
    adjacency = np.array([[0.0, 0.8, 0.1], [0.8, 0.0, 0.4], [0.1, 0.4, 0.0]], dtype=float)
    electrodes = np.array([10, 20, 30])
    coords = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    rates = np.array([1.0, 2.0, 3.0])
    intervals = np.array([[0.0, 1.0]])

    edges = thresholded_edge_table(adjacency, electrodes, coords, top_fraction=0.4)

    assert edges.iloc[0]["source_electrode"] == 10
    assert edges.iloc[0]["target_electrode"] == 20
    assert edges.iloc[0]["weight"] == 0.8
    assert edges.iloc[0]["distance_um"] == 5.0

    path = write_sttc_network_cache(
        tmp_path / "network.h5",
        adjacency=adjacency,
        electrodes=electrodes,
        coords_um=coords,
        rates_hz=rates,
        intervals=intervals,
        config={"dt_ms": 10.0, "label": "test"},
    )
    loaded = read_sttc_network_cache(path)

    np.testing.assert_allclose(loaded["adjacency"], adjacency)
    np.testing.assert_array_equal(loaded["electrodes"], electrodes)
    np.testing.assert_allclose(loaded["coords_um"], coords)
    np.testing.assert_allclose(loaded["rates_hz"], rates)
    np.testing.assert_allclose(loaded["intervals_s"], intervals)
    assert loaded["attrs"]["dt_ms"] == 10.0
