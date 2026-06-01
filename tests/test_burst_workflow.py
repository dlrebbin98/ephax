import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from ephax import PrepConfig, Recording, RestingActivityDataset, workflows
from ephax.models import WaveAnalysisResult
from ephax.models import AlignedBurstEvents
from ephax.metrics.burst import (
    activity_state_kde_peak_frequencies,
    align_highres_to_anchors,
    assign_max_population_ifr_burst_anchors,
    binned_kde_peak_summary,
    build_highres_traces,
    build_network_activity_state,
    build_participation_activity_state,
    build_population_ifr,
    compute_electrode_peak_time_map,
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
from ephax.metrics.waves import aggregate_wave_results, load_wave_result_cache, save_wave_result_cache
from ephax.plotting.burst import (
    draw_activity_state_ifr_kde_histograms,
    draw_electrode_peak_time_map,
    draw_high_activity_burst_windows,
    plot_aligned_electrode_heatmap,
    plot_activity_state_ifr_kde_histograms,
    plot_electrode_peak_time_map,
    plot_gamma_population_windows,
    plot_high_activity_burst_windows,
    plot_macro_burst_detector_comparison_windows,
    plot_population_ifr_summary,
    select_high_activity_windows,
)
from ephax.plotting.waves import draw_wave_bootstrap_panel, draw_wave_timing_panel


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


def fixture_aligned_peak_map_events():
    rel = np.array([-10.0, 0.0, 10.0, 20.0])
    aligned_rate = np.array(
        [
            [
                [1.0, 5.0, 2.0, 0.0],
                [0.0, 1.0, 6.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 2.0, 9.0, 0.0],
                [0.0, 8.0, 2.0, 1.0],
                [0.0, 0.0, 7.0, 1.0],
            ],
        ],
        dtype=float,
    )
    aligned = AlignedBurstEvents(
        relative_time_ms=rel,
        population_windows=np.zeros((2, rel.size)),
        aligned_rate=aligned_rate,
        aligned_spikes=aligned_rate > 0,
        valid_anchors=pd.DataFrame({"coarse_event_idx": [10, 11], "gamma_peak_rank": [0, 0]}),
        electrodes=np.array([101, 102, 103]),
    )
    layout = {
        "electrode": np.array([101, 102, 103, 999]),
        "x": np.array([0.0, 100.0, 200.0, 300.0]),
        "y": np.array([0.0, 20.0, 40.0, 60.0]),
    }
    return aligned, layout


def test_compute_electrode_peak_time_map_for_single_event_and_aggregate():
    aligned, layout = fixture_aligned_peak_map_events()

    event_map = compute_electrode_peak_time_map(aligned, layout, window_idx=0, min_peak_rate_hz=0.5)
    event_by_electrode = event_map.set_index("electrode")

    assert event_by_electrode.loc[101, "peak_time_ms"] == 0.0
    assert event_by_electrode.loc[102, "peak_time_ms"] == 10.0
    assert bool(event_by_electrode.loc[103, "valid"]) is False
    assert event_by_electrode.loc[101, "coarse_event_idx"] == 10

    aggregate_map = compute_electrode_peak_time_map(aligned, layout)
    aggregate_by_electrode = aggregate_map.set_index("electrode")

    assert aggregate_by_electrode.loc[101, "peak_time_ms"] == 10.0
    assert aggregate_by_electrode.loc[102, "peak_time_ms"] == 0.0
    assert aggregate_by_electrode.loc[103, "peak_time_ms"] == 10.0
    assert 999 not in aggregate_by_electrode.index


def test_compute_electrode_peak_time_map_respects_search_window():
    aligned, layout = fixture_aligned_peak_map_events()

    peak_map = compute_electrode_peak_time_map(
        aligned,
        layout,
        window_idx=1,
        peak_search_start_ms=-10.0,
        peak_search_stop_ms=0.0,
    )
    by_electrode = peak_map.set_index("electrode")

    assert by_electrode.loc[101, "peak_time_ms"] == 0.0
    assert by_electrode.loc[102, "peak_time_ms"] == 0.0
    assert by_electrode.loc[103, "peak_time_ms"] == -10.0


def test_draw_electrode_peak_time_map_renders_supplied_axis():
    aligned, layout = fixture_aligned_peak_map_events()
    peak_map = compute_electrode_peak_time_map(aligned, layout, window_idx=1)
    fig, ax = plt.subplots(figsize=(4, 2))

    rendered = draw_electrode_peak_time_map(peak_map, ax, title="event order", show_colorbar=True)

    assert rendered["mappable"] is not None
    assert rendered["colorbar"] is not None
    assert ax.get_title() == "event order"
    assert ax.get_facecolor() == (0.0, 0.0, 0.0, 1.0)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(4, 2))
    rendered2 = draw_electrode_peak_time_map(peak_map, ax2, show_colorbar=False)
    assert rendered2["colorbar"] is None
    plt.close(fig2)


def test_draw_electrode_peak_time_map_supports_hexbin_mode():
    aligned, layout = fixture_aligned_peak_map_events()
    peak_map = compute_electrode_peak_time_map(aligned, layout, window_idx=1)
    fig, ax = plt.subplots(figsize=(4, 2))

    rendered = draw_electrode_peak_time_map(
        peak_map,
        ax,
        render_mode="hexbin",
        gridsize=8,
        xlim=(0.0, 250.0),
        ylim=(0.0, 80.0),
        show_colorbar=False,
    )

    assert rendered["render_mode"] == "hexbin"
    assert rendered["mappable"].get_array().size > 0
    assert rendered["colorbar"] is None
    plt.close(fig)


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
    assert highres.per_electrode_rate_hz.dtype == np.float32
    assert highres.spike_presence.dtype == bool
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


def test_draw_activity_state_ifr_kde_histograms_uses_provided_axes():
    rng = np.random.default_rng(1)
    high_values = rng.lognormal(mean=3.0, sigma=0.2, size=100)
    burst_values = rng.lognormal(mean=4.0, sigma=0.2, size=100)
    high_hist = binned_kde_peak_summary(high_values, log_bins=True, n_bins=40, grid_size=512)
    burst_hist = binned_kde_peak_summary(burst_values, log_bins=True, n_bins=40, grid_size=512)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    rendered = draw_activity_state_ifr_kde_histograms(
        {"high_activity": high_hist, "burst": burst_hist},
        axes,
        activity_values={"high_activity": high_values, "burst": burst_values},
        compact=True,
        show_legend=False,
    )

    assert set(rendered["artists"]) == {"high_activity", "burst"}
    assert axes[0].get_xscale() == "log"
    assert axes[1].get_title() == "Burst"
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
    assert participation.spike_counts.dtype == np.int32
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


def test_assign_max_population_ifr_burst_anchors_uses_peak_rate():
    ds = fixture_burst_dataset()
    rec = ds.recordings[0]
    refs = np.array([101, 102, 103])
    highres = build_highres_traces(rec, refs, bin_ms=10.0, smooth_sigma_ms=1.0)
    burst_epochs = pd.DataFrame(
        [
            {"event_idx": 0, "start_time_s": 0.94, "end_time_s": 1.21},
            {"event_idx": 1, "start_time_s": 2.94, "end_time_s": 3.21},
        ]
    )

    anchored = assign_max_population_ifr_burst_anchors(highres, burst_epochs)

    assert "participation_anchor_time_s" in anchored
    assert "anchor_population_rate_hz" in anchored
    for row in anchored.itertuples(index=False):
        mask = (highres.time_centers_s >= row.start_time_s) & (highres.time_centers_s <= row.end_time_s)
        assert row.anchor_population_rate_hz == highres.population_rate_hz[mask].max()
        assert row.start_time_s <= row.anchor_time_s <= row.end_time_s


def fixture_wave_result(offset: float = 0.0) -> WaveAnalysisResult:
    peak_rows = []
    trace_rows = []
    for window_idx in [0, 1]:
        for bin_idx, x_um in enumerate([50.0, 150.0, 250.0]):
            peak_time = offset + 0.01 * x_um + 0.1 * window_idx
            peak_rows.append(
                {
                    "window_idx": window_idx,
                    "event_direction": "left_to_right",
                    "origin_x_bin_idx": bin_idx,
                    "origin_x_um": x_um,
                    "peak_time_ms": peak_time,
                    "peak_rate_hz": 10.0 + bin_idx,
                    "n_electrodes": 5,
                }
            )
            for time_ms in [-5.0, 0.0, 5.0]:
                trace_rows.append(
                    {
                        "window_idx": window_idx,
                        "event_direction": "left_to_right",
                        "origin_x_bin_idx": bin_idx,
                        "origin_x_um": x_um,
                        "time_ms": time_ms,
                        "rate_hz": 10.0 + bin_idx + time_ms * 0.1,
                    }
                )
    peaks = pd.DataFrame(peak_rows)
    trace = pd.DataFrame(trace_rows)
    bin_summary = (
        peaks.groupby(["origin_x_bin_idx", "origin_x_um"], as_index=False)
        .agg(
            mean_peak_time_ms=("peak_time_ms", "mean"),
            median_peak_time_ms=("peak_time_ms", "median"),
            std_peak_time_ms=("peak_time_ms", "std"),
            mean_peak_rate_hz=("peak_rate_hz", "mean"),
            n_events=("window_idx", "nunique"),
            n_observations=("window_idx", "size"),
        )
        .sort_values("origin_x_um")
    )
    bin_summary["sem_peak_time_ms"] = bin_summary["std_peak_time_ms"] / np.sqrt(bin_summary["n_observations"])
    event_direction = pd.DataFrame(
        [
            {"window_idx": 0, "event_slope_ms_per_um": 0.01, "event_intercept_ms": offset, "event_direction": "left_to_right"},
            {"window_idx": 1, "event_slope_ms_per_um": 0.01, "event_intercept_ms": offset + 0.1, "event_direction": "left_to_right"},
        ]
    )
    fit_summary = pd.DataFrame(
        [
            {
                "x_bin_um": 100.0,
                "array_width_um": 300.0,
                "n_bins_retained": 3,
                "n_events_used": 2,
                "n_events_left_to_right": 2,
                "n_events_right_to_left": 0,
                "slope_ms_per_um": 0.01,
                "intercept_ms": offset,
                "implied_speed_um_per_ms": 100.0,
                "bootstrap_speed_mean_um_per_ms": 100.0,
                "bootstrap_speed_median_um_per_ms": 100.0,
                "bootstrap_speed_ci_low_um_per_ms": 90.0,
                "bootstrap_speed_ci_high_um_per_ms": 110.0,
            }
        ]
    )
    heatmap = trace.groupby(["time_ms", "origin_x_um"], as_index=False)["rate_hz"].mean().pivot(
        index="time_ms",
        columns="origin_x_um",
        values="rate_hz",
    )
    return WaveAnalysisResult(
        event_direction=event_direction,
        trace=trace,
        peaks=peaks,
        bin_summary=bin_summary,
        fit_summary=fit_summary,
        heatmap=heatmap,
        bootstrap_speeds=np.array([90.0, 100.0, 110.0]),
    )


def test_aggregate_wave_results_offsets_window_indices():
    aggregate, per_recording = aggregate_wave_results(
        [("rec0", fixture_wave_result()), ("rec1", fixture_wave_result(offset=1.0))],
        x_bin_um=100.0,
        min_events_per_bin=1,
        bootstrap_reps=25,
        random_seed=0,
    )

    assert aggregate.peaks["window_idx"].nunique() == 4
    assert "local_window_idx" in aggregate.peaks
    assert set(aggregate.peaks["recording_id"]) == {"rec0", "rec1"}
    assert set(aggregate.peaks.loc[aggregate.peaks["recording_id"] == "rec1", "local_window_idx"]) == {0, 1}
    assert aggregate.heatmap.shape[1] == 3
    assert len(per_recording) == 2


def test_wave_result_cache_round_trips(tmp_path):
    result = fixture_wave_result()
    cache_dir = save_wave_result_cache(result, tmp_path / "wave_cache")

    loaded = load_wave_result_cache(cache_dir)

    assert loaded is not None
    assert loaded.peaks.shape == result.peaks.shape
    assert loaded.heatmap.shape == result.heatmap.shape
    assert np.allclose(loaded.bootstrap_speeds, result.bootstrap_speeds)


def test_draw_wave_panels_render_prepared_result():
    result = fixture_wave_result()
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    timing = draw_wave_timing_panel(result, axes[0], compact=True)
    boot = draw_wave_bootstrap_panel(result, axes[1], compact=True)

    assert timing["mappable"] is not None
    assert "histogram" in boot["artists"]
    assert axes[0].get_xlabel() == "Distance from inferred origin side (um)"
    assert axes[1].get_xlabel() == "Implied propagation speed (um/ms)"
    plt.close(fig)


def test_draw_wave_bootstrap_panel_handles_empty_bootstrap():
    result = fixture_wave_result()
    result.bootstrap_speeds = np.array([], dtype=float)
    fig, ax = plt.subplots(figsize=(4, 3))

    rendered = draw_wave_bootstrap_panel(result, ax)

    assert "message" in rendered["artists"]
    plt.close(fig)


def test_select_high_activity_windows_can_use_peak_ifr():
    epochs = pd.DataFrame(
        [
            {"event_idx": 0, "start_time_s": 1.0, "end_time_s": 1.2, "peak_mean_ifr_hz": 10.0, "duration_ms": 200.0},
            {"event_idx": 1, "start_time_s": 2.0, "end_time_s": 2.2, "peak_mean_ifr_hz": 50.0, "duration_ms": 200.0},
            {"event_idx": 2, "start_time_s": 3.0, "end_time_s": 3.2, "peak_mean_ifr_hz": 20.0, "duration_ms": 200.0},
        ]
    )

    selected = select_high_activity_windows(epochs, n_windows=1, window_order="peak_ifr")

    assert selected["event_idx"].tolist() == [1]


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
    fig6, ax6 = plt.subplots(figsize=(5, 3))
    rendered6 = draw_high_activity_burst_windows(
        {
            "time_grid": population.time_grid,
            "mean_ifr": population.mean_ifr,
            "mean_ifr_smooth": population.mean_ifr_smooth,
            "network_activity": participation,
            "high_activity_epochs": high_epochs,
            "burst_epochs": bursts,
            "high_activity_threshold_hz": info["threshold_hz"],
            "participation_threshold": 0.50,
        },
        [ax6],
        n_windows=1,
        pad_s=0.5,
        window_order="peak_ifr",
        compact=True,
        show_legend=False,
    )

    assert fig1 is not None
    assert fig2 is not None
    assert fig3 is not None
    assert fig4 is not None
    assert fig5 is not None
    assert rendered6["windows"].shape[0] == 1
    assert rendered6["secondary_axes"][0].yaxis.label.get_color() == "tab:blue"
    assert rendered6["secondary_axes"][0].spines["right"].get_edgecolor() == to_rgba("tab:blue")
    assert len(axes4) == 1
    assert len(axes5) == 1
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)
    plt.close(fig6)


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
