from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ephax.metrics.synchrony_distance import (
    SynchronyDistanceConfig,
    compute_recording_synchrony_distance,
    normalize_intervals,
    pool_distance_synchrony_across_recordings,
    subtract_intervals,
    weighted_event_bootstrap,
    RecordingSynchronyInput,
)


def _sync_item():
    config = SynchronyDistanceConfig(
        min_distance_um=50.0, max_distance_um=250.0, distance_bin_um=100.0,
        null_method="rate_expectation", bootstrap_reps=3, n_surrogates=1, random_seed=1,
    )
    item = RecordingSynchronyInput(
        recording=_recording(), electrodes=np.array([0, 1, 2, 3]), dataset="synthetic",
        well=2, div=14, recording_id="synthetic_well2_DIV14",
        selected_intervals=normalize_intervals(pd.DataFrame({"start_time_s": [0.0, 0.25], "end_time_s": [0.25, 0.5]})),
    )
    return item, config


def _recording():
    spikes = {
        "time": np.array([0.100, 0.101, 0.200, 0.201, 0.300, 0.350]),
        "electrode": np.array([0, 1, 0, 1, 2, 3]),
    }
    layout = {
        "electrode": np.array([0, 1, 2, 3]),
        "x": np.array([0.0, 100.0, 0.0, 100.0]),
        "y": np.array([0.0, 0.0, 100.0, 100.0]),
    }
    return SimpleNamespace(spikes=spikes, layout=layout, start_time=0.0, end_time=0.5)


def test_subtract_intervals_splits_overlap():
    base = pd.DataFrame({"start_time_s": [0.0], "end_time_s": [10.0]})
    remove = pd.DataFrame({"start_time_s": [3.0], "end_time_s": [5.0]})

    out = subtract_intervals(base, remove)

    assert np.allclose(out["start_time_s"], [0.0, 5.0])
    assert np.allclose(out["end_time_s"], [3.0, 10.0])
    assert np.allclose(out["duration_s"], [3.0, 5.0])


def test_compute_recording_synchrony_distance_outputs_cache_schema():
    config = SynchronyDistanceConfig(
        lag_windows_ms=((-2.0, 2.0),),
        primary_lag_window_ms=(-2.0, 2.0),
        min_distance_um=50.0,
        max_distance_um=250.0,
        distance_bin_um=100.0,
        null_method="rate_expectation",
        bootstrap_reps=2,
        n_surrogates=1,
        random_seed=1,
    )
    item = RecordingSynchronyInput(
        recording=_recording(),
        electrodes=np.array([0, 1, 2, 3]),
        dataset="synthetic",
        well=0,
        div=1,
        recording_id="synthetic_well0_DIV1",
        selected_intervals=normalize_intervals(pd.DataFrame({"start_time_s": [0.0], "end_time_s": [0.5]})),
    )

    out = compute_recording_synchrony_distance(item, config, np.random.default_rng(1))

    required = {
        "dataset",
        "well",
        "div",
        "recording_id",
        "activity_scope",
        "bootstrap_idx",
        "compute_method",
        "null_method",
        "lag_start_ms",
        "lag_stop_ms",
        "distance_bin_center_um",
        "p_obs",
        "p_null_mean",
        "excess_sync",
        "n_pairs",
        "n_trigger_spikes",
    }
    assert required.issubset(out.columns)
    assert not out.empty
    assert set(out["compute_method"]) == {"interval_summary_matrix"}
    assert out["p_obs"].between(0, 1).all()
    assert out["p_null_mean"].between(0, 1).all()


def test_weighted_event_bootstrap_recovers_weighted_mean():
    values = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    weights = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    res = weighted_event_bootstrap(values, weights, n_boot=300, seed=0)
    # weighted mean per cell: (1+3+2*5)/4 = 3.5 ; (10+30+2*50)/4 = 35
    assert np.allclose(res["point"], [3.5, 35.0])
    assert res["n_events"] == 3
    assert res["ci_lo"][0] <= res["point"][0] <= res["ci_hi"][0]


def test_compute_recording_synchrony_distance_returns_tagged_interval_summary():
    item, config = _sync_item()
    result, interval_summary = compute_recording_synchrony_distance(
        item, config, np.random.default_rng(1), return_interval_summary=True)
    assert not interval_summary.empty
    # the per-interval summary carries interval + distance bin + the tags for pooling
    for col in ("interval_idx", "distance_bin_center_um", "excess_sync", "n_pairs",
                "well", "div", "recording_id", "activity_scope"):
        assert col in interval_summary.columns, col
    assert set(interval_summary["recording_id"]) == {"synthetic_well2_DIV14"}
    # default call still returns just the bootstrapped frame
    plain = compute_recording_synchrony_distance(item, config, np.random.default_rng(1))
    assert isinstance(plain, pd.DataFrame)


def test_pool_distance_synchrony_across_recordings_return_boot():
    rows = []
    for rid in ("w0", "w1"):
        for interval_idx in range(3):
            for center, excess, npairs in [(100.0, 0.2, 50), (200.0, 0.1, 40)]:
                rows.append({"recording_id": rid, "interval_idx": interval_idx,
                             "distance_bin_center_um": center, "excess_sync": excess, "n_pairs": npairs})
    summary, boot = pool_distance_synchrony_across_recordings(
        pd.DataFrame(rows), n_boot=50, seed=0, return_boot=True)
    assert {"bootstrap_idx", "distance_bin_center_um", "excess_sync"}.issubset(boot.columns)
    assert boot["bootstrap_idx"].nunique() == 50
    assert set(boot["distance_bin_center_um"]) == {100.0, 200.0}


def test_pool_distance_synchrony_across_recordings_pools_events():
    rows = []
    for rid in ("well0", "well1"):
        for interval_idx in range(3):
            for center, excess, npairs in [(100.0, 0.2, 50), (200.0, 0.1, 40)]:
                rows.append({
                    "recording_id": rid,
                    "interval_idx": interval_idx,
                    "distance_bin_center_um": center,
                    "excess_sync": excess,
                    "n_pairs": npairs,
                })
    interval_summary = pd.DataFrame(rows)
    out = pool_distance_synchrony_across_recordings(interval_summary, n_boot=200, seed=0)
    assert set(out["distance_bin_center_um"]) == {100.0, 200.0}
    # 2 wells x 3 intervals = 6 pooled events
    assert (out["n_events"] == 6).all()
    near = out[out["distance_bin_center_um"] == 100.0].iloc[0]
    assert near["excess_sync"] == pytest.approx(0.2)
    assert near["excess_sync_ci_lo"] <= near["excess_sync"] <= near["excess_sync_ci_hi"]
