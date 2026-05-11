import numpy as np
import pandas as pd

from ephax.metrics.transfer_entropy import (
    blocks_to_bins,
    build_observation_summary,
    build_signed_target_map,
    build_trigger_summary,
    counts_from_te_states,
    extract_first_stable_ridge,
    run_discrete_te,
    transfer_entropy_bits_from_counts,
)
from ephax.metrics.waves import analyze_eventwise_waves
from ephax.models import AlignedBurstEvents, HighResTraces


def aligned_wave_fixture():
    relative = np.arange(-20.0, 50.0, 10.0)
    electrodes = np.array([101, 102, 103, 104, 105, 106])
    x = np.array([0.0, 20.0, 100.0, 120.0, 200.0, 220.0])
    aligned = np.zeros((4, len(electrodes), len(relative)), dtype=float)
    for event_idx in range(aligned.shape[0]):
        for electrode_idx, xpos in enumerate(x):
            peak_time = -10.0 + (xpos / 100.0) * 10.0
            aligned[event_idx, electrode_idx] = 5.0 + 50.0 * np.exp(-0.5 * ((relative - peak_time) / 6.0) ** 2)
    return AlignedBurstEvents(
        relative_time_ms=relative,
        population_windows=aligned.mean(axis=1),
        aligned_rate=aligned,
        aligned_spikes=aligned > 40.0,
        valid_anchors=pd.DataFrame({"coarse_event_idx": np.arange(aligned.shape[0]), "anchor_time_s": np.arange(aligned.shape[0])}),
        electrodes=electrodes,
    ), {
        "electrode": electrodes,
        "channel": np.arange(len(electrodes)),
        "x": x,
        "y": np.zeros_like(x),
    }


def highres_te_fixture():
    electrodes = np.array([101, 102, 103])
    n_bins = 80
    spike_presence = np.zeros((3, n_bins), dtype=bool)
    source_times = np.array([12, 22, 32, 42, 52, 62])
    spike_presence[0, source_times] = True
    spike_presence[1, source_times + 2] = True
    spike_presence[2, source_times + 4] = True
    highres = HighResTraces(
        bin_edges_s=np.arange(n_bins + 1) / 1000.0,
        time_centers_s=(np.arange(n_bins) + 0.5) / 1000.0,
        electrodes=electrodes,
        per_electrode_rate_hz=spike_presence.astype(float) * 1000.0,
        population_rate_hz=spike_presence.mean(axis=0) * 1000.0,
        spikes_by_electrode={int(el): np.flatnonzero(spike_presence[i]) / 1000.0 for i, el in enumerate(electrodes)},
        spike_presence=spike_presence,
    )
    anchors = pd.DataFrame({"coarse_event_idx": np.arange(len(source_times)), "anchor_time_s": highres.time_centers_s[source_times]})
    aligned = AlignedBurstEvents(
        relative_time_ms=np.arange(-5.0, 11.0, 1.0),
        population_windows=np.zeros((len(source_times), 16)),
        aligned_rate=np.zeros((len(source_times), 3, 16)),
        aligned_spikes=np.stack([spike_presence[:, t - 5 : t + 11] for t in source_times]),
        valid_anchors=anchors,
        electrodes=electrodes,
    )
    layout = {
        "electrode": electrodes,
        "channel": np.arange(3),
        "x": np.array([0.0, 100.0, 200.0]),
        "y": np.zeros(3),
    }
    return highres, aligned, layout


def test_wave_analysis_extracts_direction_and_speed_tables():
    aligned, layout = aligned_wave_fixture()

    result = analyze_eventwise_waves(
        aligned,
        layout,
        x_bin_um=100.0,
        min_electrodes_per_bin=1,
        min_events_per_bin=1,
        bootstrap_reps=20,
        random_seed=0,
    )

    assert not result.event_direction.empty
    assert not result.peaks.empty
    assert not result.bin_summary.empty
    assert result.fit_summary.loc[0, "n_bins_retained"] >= 3
    assert np.isfinite(result.fit_summary.loc[0, "implied_speed_um_per_ms"])


def test_transfer_entropy_primitives_and_single_config_runner():
    highres, aligned, layout = highres_te_fixture()

    assert blocks_to_bins([(1.0, 2.0)], bin_ms=1.0) == [(1, 2)]
    counts = counts_from_te_states(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 1, 1]),
        2,
        2,
    )
    assert transfer_entropy_bits_from_counts(counts) >= 0.0

    triggers = build_trigger_summary(scope="gamma", highres=highres, selected_electrodes=highres.electrodes, aligned=aligned)
    target_bins = blocks_to_bins([(1.0, 2.0)], bin_ms=1.0)
    source_bins = blocks_to_bins([(1.0, 2.0)], bin_ms=1.0)
    obs = build_observation_summary(
        triggers,
        highres.spike_presence,
        np.array([0, 1, 2]),
        target_bins,
        source_bins,
        control_exclusion_bins=1,
        rng=np.random.default_rng(0),
    )
    signed_centers, target_map = build_signed_target_map(layout, highres.electrodes, signed_dx_bin_um=100.0)

    assert not triggers.empty
    assert not obs.empty
    assert signed_centers.tolist() == [-200.0, -100.0, 0.0, 100.0, 200.0]
    assert target_map[0]

    result = run_discrete_te(
        highres=highres,
        layout=layout,
        selected_electrodes=highres.electrodes,
        trigger_summary=triggers,
        target_history_blocks_ms=[(1.0, 2.0)],
        source_history_blocks_ms=[(1.0, 2.0)],
        temporal_bin_ms=1.0,
        signed_dx_bin_um=100.0,
        delay_start_ms=0.0,
        delay_stop_ms=4.0,
        n_surrogates=0,
        min_observations=1,
        min_effect_bits=0.0,
        alpha=1.0,
        bootstrap_reps=10,
        random_seed=0,
    )

    assert result.raw_te_bits.shape == (5, 5)
    assert result.effective_observations.max() > 0


def test_transfer_entropy_ridge_and_speed_fit_are_stable():
    delay = np.array([0.0, 1.0, 2.0])
    dx = np.array([-200.0, -100.0, 0.0, 100.0, 200.0])
    te = np.zeros((3, 5))
    te[1, [0, 1, 3, 4]] = 0.1
    p = np.ones((3, 5)) * 0.01
    obs = np.ones((3, 5)) * 10
    cond = np.ones((3, 5)) * 0.5

    ridge = extract_first_stable_ridge(
        delay,
        dx,
        te,
        p,
        obs,
        cond,
        min_obs=1,
        min_effect=0.01,
        alpha=0.05,
        smooth_sigma_bins=0.0,
        tolerance_bins=1,
    )

    assert set(ridge["direction"]) == {"leftward", "rightward"}
