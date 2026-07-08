import numpy as np
import pandas as pd

from ephax.metrics.event_selection import (
    FrontNullConfig,
    compute_front_null_support,
    front_significant_events,
    peri_anchor_spike_counts,
)


def test_peri_anchor_spike_counts_window():
    spike_times = np.array([0.10, 0.11, 0.12, 0.50, 0.98, 0.99, 1.00, 1.01])
    counts = peri_anchor_spike_counts(spike_times, [0.11, 0.99], pre_s=0.025, post_s=0.040)
    assert list(counts) == [3, 4]


def test_peri_anchor_spike_counts_boundaries_inclusive():
    spike_times = np.array([0.0, 0.025, 0.040, 0.041])
    # anchor 0.0, window [-0.025, 0.040]: includes 0.0, 0.025, 0.040 (inclusive) but not 0.041
    assert list(peri_anchor_spike_counts(spike_times, [0.0], pre_s=0.025, post_s=0.040)) == [3]


def test_front_significant_events_threshold():
    event_speed_df = pd.DataFrame({
        "well": [0, 0, 1, 1],
        "event_idx": [1, 2, 1, 2],
        "median_speed_mm_per_s": [80.0, 90.0, 100.0, 110.0],
        "n_valid_local_fits": [12, 3, 20, 25],
    })
    # per-well threshold = max over null types of the 0.95 quantile of null n_valid
    front_null_df = pd.DataFrame({
        "well": [0] * 4 + [1] * 4,
        "null_type": ["coordinate_shuffle", "coordinate_shuffle", "arrival_time_shuffle", "arrival_time_shuffle"] * 2,
        "n_valid": [4, 5, 3, 4, 20, 22, 18, 21],
    })
    out = front_significant_events(event_speed_df, front_null_df, support_quantile=0.95)
    # well 0 threshold = max(quantile([4,5],.95)=4.95, quantile([3,4],.95)=3.95) = 4.95
    # well 1 threshold = max(quantile([20,22],.95)=21.9, quantile([18,21],.95)=20.85) = 21.9
    assert np.allclose(out.loc[out.well == 0, "n_valid_significance_threshold"].unique(), [4.95])
    assert np.allclose(out.loc[out.well == 1, "n_valid_significance_threshold"].unique(), [21.9])
    assert list(out["front_significant"]) == [True, False, False, True]


def test_front_significant_events_preserves_rows_and_speed():
    event_speed_df = pd.DataFrame({
        "well": [0, 0], "event_idx": [1, 2],
        "median_speed_mm_per_s": [80.0, 90.0], "n_valid_local_fits": [10, 10],
    })
    front_null_df = pd.DataFrame({
        "well": [0, 0], "null_type": ["coordinate_shuffle", "arrival_time_shuffle"], "n_valid": [1, 1],
    })
    out = front_significant_events(event_speed_df, front_null_df)
    assert len(out) == 2
    assert "median_speed_mm_per_s" in out.columns
    assert out["front_significant"].all()  # observed 10 >> null 1


def _make_local_df():
    # two events on a small isotropic grid so local arrival-plane fits are resolvable
    rng = np.random.default_rng(0)
    rows = []
    xs, ys = np.meshgrid(np.linspace(0, 0.6, 8), np.linspace(0, 0.6, 8))
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    for well, event_idx, vx in [(0, 1, 0.2), (0, 2, 0.15)]:
        arrival = (coords[:, 0] * vx + coords[:, 1] * 0.05) + rng.normal(0, 1e-4, coords.shape[0])
        for (x, y), t in zip(coords, arrival):
            rows.append({"well": well, "event_idx": event_idx, "x_mm": x, "y_mm": y,
                         "arrival_time_s": float(t), "arrival_amplitude": 1.0,
                         "arrival_amplitude_threshold": 0.0})
    return pd.DataFrame(rows)


def test_compute_front_null_support_smoke():
    local_df = _make_local_df()
    event_speed_df = pd.DataFrame({
        "well": [0, 0], "event_idx": [1, 2],
        "median_speed_mm_per_s": [5.0, 6.7], "n_valid_local_fits": [40, 40],
    })
    out = compute_front_null_support(
        local_df, event_speed_df,
        null_types=("coordinate_shuffle", "arrival_time_shuffle"),
        reps=2, seed=1, max_events_per_well=25, cfg=FrontNullConfig(),
    )
    assert set(["well", "event_idx", "null_type", "null_rep", "n_valid"]).issubset(out.columns)
    # 2 events x 2 null types x 2 reps = 8 rows
    assert len(out) == 8
    assert out["n_valid"].ge(0).all()


def test_compute_front_null_support_cache_roundtrip(tmp_path):
    local_df = _make_local_df()
    event_speed_df = pd.DataFrame({
        "well": [0, 0], "event_idx": [1, 2],
        "median_speed_mm_per_s": [5.0, 6.7], "n_valid_local_fits": [40, 40],
    })
    cache = tmp_path / "front_null.csv"
    a = compute_front_null_support(local_df, event_speed_df, reps=1, seed=1, cache_path=cache)
    assert cache.exists()
    b = compute_front_null_support(local_df, event_speed_df, reps=1, seed=1, cache_path=cache)  # from cache
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
