from types import SimpleNamespace

import numpy as np
import pandas as pd

from ephax.metrics.synchrony_distance import (
    SynchronyDistanceConfig,
    compute_recording_synchrony_distance,
    normalize_intervals,
    subtract_intervals,
    RecordingSynchronyInput,
)


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
