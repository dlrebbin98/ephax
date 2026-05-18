import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ephax import PrepConfig, Recording, RestingActivityDataset, workflows
from ephax.metrics.burst import (
    activity_state_kde_peak_frequencies,
    align_highres_to_anchors,
    binned_kde_peak_summary,
    build_highres_traces,
    build_network_activity_state,
    build_participation_activity_state,
    build_population_ifr,
    detect_coarse_burst_epochs,
    detect_high_activity_epochs,
    detect_nested_gamma_anchors,
    detect_network_burst_epochs,
    detect_participation_burst_epochs,
    extract_activity_state_ifr,
    order_aligned_rate_by_summary,
    order_aligned_rate_by_x,
    refine_participation_burst_anchors,
    summarize_aligned_electrode_rates,
)
from ephax.plotting.burst import (
    plot_aligned_electrode_heatmap,
    plot_activity_state_ifr_kde_histograms,
    plot_gamma_population_windows,
    plot_high_activity_burst_windows,
    plot_macro_burst_detector_comparison_windows,
    plot_population_ifr_summary,
)


def fixture_burst_dataset():
    times = np.array(
        [
            0.95,
            1.00,
            1.05,
            1.10,
            1.15,
            1.20,
            2.95,
            3.00,
            3.05,
            3.10,
            3.15,
            3.20,
        ]
    )
    electrodes = np.array([101, 101, 102, 102, 103, 103, 101, 101, 102, 102, 103, 103])
    spikes = {
        "time": times,
        "channel": np.array([1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3]),
        "amplitude": np.ones(times.size),
        "electrode": electrodes,
    }
    layout = {
        "channel": np.array([1, 2, 3]),
        "electrode": np.array([101, 102, 103]),
        "x": np.array([0.0, 100.0, 200.0]),
        "y": np.array([0.0, 0.0, 0.0]),
    }
    rec = Recording(spikes=spikes, layout=layout, start_time=0.0, end_time=4.0, sf=1000.0)
    return RestingActivityDataset([rec], sf=1000.0)


def test_burst_metrics_extract_notebook_core_path():
    ds = fixture_burst_dataset()
    rec = ds.recordings[0]
    refs = np.array([101, 102, 103])

    population = build_population_ifr(rec, refs, grid_hz=20.0, smooth_sigma_sec=0.05)
    coarse_epochs, raw_peaks, _props = detect_coarse_burst_epochs(
        population.time_grid,
        population.mean_ifr_smooth,
        grid_hz=20.0,
        peak_distance_sec=0.5,
        prominence_quantile=0.50,
        prominence_scale=0.10,
        rel_height=0.20,
    )
    highres = build_highres_traces(rec, refs, bin_ms=10.0, smooth_sigma_ms=10.0)
    anchors = detect_nested_gamma_anchors(
        coarse_epochs,
        highres,
        bin_ms=10.0,
        min_distance_ms=20.0,
        prominence_abs_floor=0.0,
    )
    aligned = align_highres_to_anchors(highres, anchors, pre_ms=20.0, post_ms=40.0, bin_ms=10.0)
    summary = summarize_aligned_electrode_rates(aligned)
    ordered_electrodes, ordered_rate = order_aligned_rate_by_summary(aligned, summary)
    x_layout, x_rate = order_aligned_rate_by_x(aligned, rec.layout, summary)

    assert population.ifr_matrix.shape[0] == 3
    assert len(coarse_epochs) >= 1
    assert raw_peaks.size >= 1
    assert len(anchors) >= 1
    assert aligned.aligned_rate.ndim == 3
    assert summary["electrode"].tolist()
    assert ordered_electrodes.size == ordered_rate.shape[0]
    assert len(x_layout) == x_rate.shape[0]


def test_extract_activity_state_ifr_splits_raw_isi_samples():
    spikes = {
        "time": np.array([0.00, 0.10, 0.20, 0.30, 1.00, 1.05]),
        "channel": np.ones(6, dtype=int),
        "amplitude": np.ones(6),
        "electrode": np.full(6, 101, dtype=int),
    }
    layout = {
        "channel": np.array([1]),
        "electrode": np.array([101]),
        "x": np.array([0.0]),
        "y": np.array([0.0]),
    }
    rec = Recording(spikes=spikes, layout=layout, start_time=0.0, end_time=1.1, sf=1000.0)
    high_epochs = pd.DataFrame([{"start_time_s": 0.10, "end_time_s": 0.32}])
    burst_epochs = pd.DataFrame([{"start_time_s": 0.24, "end_time_s": 0.28}])

    states = extract_activity_state_ifr(rec, np.array([101]), high_epochs, burst_epochs)

    assert np.allclose(states["high_activity"], [10.0])
    assert np.allclose(states["burst"], [10.0])
    assert states["low_activity"].size == 3


def test_binned_kde_peak_summary_and_frequency_merge():
    rng = np.random.default_rng(0)
    high_values = np.concatenate([rng.normal(18.0, 0.8, 200), rng.normal(45.0, 1.0, 300)])
    burst_values = np.concatenate([rng.normal(28.0, 0.9, 200), rng.normal(90.0, 2.0, 300)])
    high_hist = binned_kde_peak_summary(high_values, log_bins=False, n_bins=80, grid_size=1024)
    burst_hist = binned_kde_peak_summary(burst_values, log_bins=False, n_bins=80, grid_size=1024)

    assert high_hist["peak_hz"].size >= 1
    assert burst_hist["peak_hz"].size >= 1
    assert np.all(np.diff(high_hist["peak_counts"]) <= 0)

    freqs = activity_state_kde_peak_frequencies(
        {"high_activity": high_hist, "burst": burst_hist},
        min_peak_hz=30.0,
    )

    assert freqs.size >= 2
    assert np.all(freqs > 30.0)
    assert np.all(np.diff(freqs) > 0)

    fig, _axes = plot_activity_state_ifr_kde_histograms(
        {"high_activity": high_hist, "burst": burst_hist},
        {"high_activity": high_values, "burst": burst_values},
    )
    plt.close(fig)


def test_network_activity_detector_requires_participating_electrodes():
    rec = fixture_burst_dataset().recordings[0]
    refs = np.array([101, 102, 103])

    highres = build_highres_traces(rec, refs, bin_ms=10.0, smooth_sigma_ms=1.0)
    activity = build_network_activity_state(
        highres,
        aggregation_ms=50.0,
        active_rate_floor_hz=1.0,
        threshold_iqr_scale=0.0,
    )
    epochs = detect_network_burst_epochs(
        activity,
        min_participation_fraction=0.50,
        min_active_electrodes=2,
        merge_gap_ms=100.0,
        min_duration_ms=40.0,
        min_spikes=2,
    )

    assert len(epochs) == 2
    assert epochs["participating_electrodes"].min() >= 2
    assert epochs["total_spikes"].min() >= 2


def test_network_activity_detector_rejects_single_electrode_burst():
    spikes = {
        "time": np.array([1.00, 1.03, 1.06, 1.09, 1.12, 1.15]),
        "channel": np.ones(6, dtype=int),
        "amplitude": np.ones(6),
        "electrode": np.full(6, 101, dtype=int),
    }
    layout = {
        "channel": np.array([1, 2, 3]),
        "electrode": np.array([101, 102, 103]),
        "x": np.array([0.0, 100.0, 200.0]),
        "y": np.array([0.0, 0.0, 0.0]),
    }
    rec = Recording(spikes=spikes, layout=layout, start_time=0.0, end_time=2.0, sf=1000.0)
    highres = build_highres_traces(rec, np.array([101, 102, 103]), bin_ms=10.0, smooth_sigma_ms=1.0)
    activity = build_network_activity_state(
        highres,
        aggregation_ms=50.0,
        active_rate_floor_hz=1.0,
        threshold_iqr_scale=0.0,
    )
    epochs = detect_network_burst_epochs(
        activity,
        min_participation_fraction=0.50,
        min_active_electrodes=2,
        merge_gap_ms=100.0,
        min_duration_ms=50.0,
        min_spikes=4,
    )

    assert epochs.empty


def test_high_activity_participation_detector_nests_bursts_and_anchors_at_max_participation():
    ds = fixture_burst_dataset()
    rec = ds.recordings[0]
    refs = np.array([101, 102, 103])

    population = build_population_ifr(rec, refs, grid_hz=20.0, smooth_sigma_sec=0.05)
    high_epochs, info = detect_high_activity_epochs(
        population.time_grid,
        population.mean_ifr_smooth,
        mad_scale=1.0,
        min_duration_ms=40.0,
    )
    highres = build_highres_traces(rec, refs, bin_ms=10.0, smooth_sigma_ms=1.0)
    participation = build_participation_activity_state(highres, aggregation_ms=50.0)
    bursts = detect_participation_burst_epochs(
        participation,
        high_epochs,
        min_participation_fraction=0.50,
        min_duration_ms=40.0,
    )
    refined = refine_participation_burst_anchors(highres, bursts, anchor_window_ms=50.0)

    assert info["threshold_hz"] > info["baseline_hz"]
    assert len(high_epochs) >= 1
    assert len(bursts) >= 1
    assert "anchor_time_s" in bursts
    assert "coarse_anchor_time_s" in refined
    assert "anchor_population_rate_hz" in refined
    for row in bursts.itertuples(index=False):
        mask = (participation.time_centers_s >= row.start_time_s) & (participation.time_centers_s <= row.end_time_s)
        assert row.anchor_participation_fraction == participation.participation_fraction[mask].max()
    for row in refined.itertuples(index=False):
        assert row.start_time_s <= row.anchor_time_s <= row.end_time_s


def test_burst_plotting_consumes_metric_outputs():
    ds = fixture_burst_dataset()
    rec = ds.recordings[0]
    refs = np.array([101, 102, 103])

    population = build_population_ifr(rec, refs, grid_hz=20.0, smooth_sigma_sec=0.05)
    highres = build_highres_traces(rec, refs, bin_ms=10.0, smooth_sigma_ms=10.0)
    anchors = pd.DataFrame(
        [
            {
                "coarse_event_idx": 0,
                "gamma_peak_rank": 0,
                "anchor_time_s": 1.05,
            }
        ]
    )
    aligned = align_highres_to_anchors(highres, anchors, pre_ms=20.0, post_ms=40.0, bin_ms=10.0)
    summary = summarize_aligned_electrode_rates(aligned)
    _ordered_electrodes, ordered_rate = order_aligned_rate_by_summary(aligned, summary)

    fig1, _ = plot_population_ifr_summary(population)
    fig2, _ = plot_gamma_population_windows(aligned)
    fig3, _ = plot_aligned_electrode_heatmap(
        aligned,
        ordered_rate,
        y_label="Electrodes",
        title="Aligned",
    )
    activity = build_network_activity_state(highres, aggregation_ms=50.0, active_rate_floor_hz=1.0)
    rate_peak_epochs = pd.DataFrame(
        [
            {
                "event_idx": 0,
                "start_time_s": 0.95,
                "end_time_s": 1.20,
                "coarse_peak_time_s": 1.05,
            }
        ]
    )
    network_epochs = pd.DataFrame(
        [
            {
                "event_idx": 0,
                "start_time_s": 0.98,
                "end_time_s": 1.18,
                "coarse_peak_time_s": 1.05,
            }
        ]
    )
    fig4, axes4 = plot_macro_burst_detector_comparison_windows(
        time_grid=population.time_grid,
        mean_ifr=population.mean_ifr,
        mean_ifr_smooth=population.mean_ifr_smooth,
        network_activity=activity,
        rate_peak_epochs=rate_peak_epochs,
        network_epochs=network_epochs,
        rate_peak_anchors=anchors,
        network_anchors=anchors,
        n_windows=1,
        pad_s=0.5,
        window_source="network",
    )
    high_epochs, info = detect_high_activity_epochs(
        population.time_grid,
        population.mean_ifr_smooth,
        mad_scale=1.0,
        min_duration_ms=40.0,
    )
    participation = build_participation_activity_state(highres, aggregation_ms=50.0)
    bursts = detect_participation_burst_epochs(
        participation,
        high_epochs,
        min_participation_fraction=0.50,
        min_duration_ms=40.0,
    )
    fig5, axes5 = plot_high_activity_burst_windows(
        time_grid=population.time_grid,
        mean_ifr=population.mean_ifr,
        mean_ifr_smooth=population.mean_ifr_smooth,
        network_activity=participation,
        high_activity_epochs=high_epochs,
        burst_epochs=bursts,
        high_activity_threshold_hz=info["threshold_hz"],
        participation_threshold=0.50,
        n_windows=1,
        pad_s=0.5,
    )

    assert fig1 is not None
    assert fig2 is not None
    assert fig3 is not None
    assert fig4 is not None
    assert fig5 is not None
    assert len(axes4) == 1
    assert len(axes5) == 1
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)


def test_burst_workflow_writes_lightweight_checkpoints(monkeypatch, tmp_path):
    monkeypatch.setattr(workflows, "build_dataset", lambda config: fixture_burst_dataset())
    config = {
        "run": {"id": "burst_run", "output_dir": str(tmp_path)},
        "dataset": {
            "source": "npz",
            "file_info": [
                {
                    "path": "unused.npz",
                    "recording_id": "rec0",
                    "culture_id": "culture0",
                    "div": 12,
                    "start_time": 0.0,
                    "end_time": 4.0,
                    "well": 1,
                }
            ],
        },
        "selection": {"mode": "top", "top_start": 0, "top_stop": 3, "verbose": False},
        "analyses": {
            "burst": {
                "enabled": True,
                "ifr_grid_hz": 20.0,
                "smooth_sigma_sec": 0.05,
                "burst_min_distance_sec": 0.5,
                "burst_prominence_quantile": 0.50,
                "burst_prominence_scale": 0.10,
                "highres_bin_ms": 10.0,
                "highres_smooth_sigma_ms": 10.0,
                "gamma_min_distance_ms": 20.0,
                "gamma_prominence_abs_floor": 0.0,
                "align_pre_ms": 20.0,
                "align_post_ms": 40.0,
            },
        },
    }

    result = workflows.run_workflow(config)

    assert result["burst_checkpoints"]["coarse_epochs"].exists()
    assert result["burst_checkpoints"]["nested_gamma_anchors"].exists()
    assert result["burst_checkpoints"]["aligned_electrode_summary"].exists()
    assert not result["burst_electrode_summary"].empty


def test_burst_workflow_can_use_network_participation_detector(monkeypatch, tmp_path):
    monkeypatch.setattr(workflows, "build_dataset", lambda config: fixture_burst_dataset())
    config = {
        "run": {"id": "network_burst_run", "output_dir": str(tmp_path)},
        "dataset": {"source": "npz", "file_info": [{"path": "unused.npz", "start_time": 0.0, "end_time": 4.0, "well": 1}]},
        "selection": {"mode": "top", "top_start": 0, "top_stop": 3, "verbose": False},
        "analyses": {
            "burst": {
                "enabled": True,
                "detector": "network_participation",
                "highres_bin_ms": 10.0,
                "highres_smooth_sigma_ms": 1.0,
                "network_bin_ms": 50.0,
                "network_active_rate_floor_hz": 1.0,
                "network_threshold_iqr_scale": 0.0,
                "network_min_participation_fraction": 0.50,
                "network_min_active_electrodes": 2,
                "network_merge_gap_ms": 100.0,
                "network_min_duration_ms": 40.0,
                "network_min_spikes": 2,
                "gamma_min_distance_ms": 20.0,
                "gamma_prominence_abs_floor": 0.0,
                "align_pre_ms": 20.0,
                "align_post_ms": 40.0,
            },
        },
    }

    result = workflows.run_workflow(config)

    assert result["burst_detector"] == "network_participation"
    assert result["burst_network_activity"] is not None
    assert len(result["burst_coarse_epochs"]) == 2
