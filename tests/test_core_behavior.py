import numpy as np
import pandas as pd
import pytest

from ephax import PrepConfig, Recording, RestingActivityDataset
from ephax.data_io import validate_layout, validate_spikes_data
from ephax.metrics.cofiring import cofiring_proportions
from ephax.metrics.ifr import calculate_ifr, prepare_ifr_timeseries_panels
from ephax.preprocessing.geometry import assign_r_distance


def fixture_spikes():
    return {
        "time": np.array([0.0, 0.1, 0.2, 0.3, 0.9, 1.0]),
        "channel": np.array([1, 1, 2, 2, 3, 3]),
        "amplitude": np.array([10, 11, 12, 13, 14, 15]),
        "electrode": np.array([101, 101, 102, 102, 103, 103]),
    }


def fixture_layout():
    return {
        "channel": np.array([1, 2, 3]),
        "electrode": np.array([101, 102, 103]),
        "x": np.array([0.0, 3.0, 0.0]),
        "y": np.array([0.0, 4.0, 5.0]),
    }


def fixture_dataset():
    rec = Recording(spikes=fixture_spikes(), layout=fixture_layout(), start_time=0.0, end_time=1.0, sf=1000.0)
    return RestingActivityDataset([rec], sf=1000.0)


def test_public_import_and_ifr_panel_prep_smoke():
    ds = fixture_dataset()
    cfg = PrepConfig(mode="top", top_start=0, top_stop=2, verbose=False)
    refs = ds.select_ref_electrodes(cfg)
    spikes_list = [rec.spikes for rec in ds.recordings]
    start_times = [rec.start_time for rec in ds.recordings]
    end_times = [rec.end_time for rec in ds.recordings]
    panels = prepare_ifr_timeseries_panels(
        spikes_list,
        start_times,
        end_times,
        refs,
        log_scale=False,
        time_grid_hz=10.0,
        max_time_points=20,
    )
    assert [list(ref) for ref in refs] == [[102, 101]]
    assert len(panels) == 1
    assert panels[0].electrodes.tolist() == [102, 101]


def test_validate_spikes_and_layout_reject_bad_shapes():
    assert set(validate_spikes_data(fixture_spikes())) == {"time", "channel", "amplitude", "electrode"}
    assert set(validate_layout(fixture_layout())) == {"channel", "electrode", "x", "y"}
    bad = fixture_spikes()
    bad["time"] = bad["time"][:-1]
    with pytest.raises(ValueError):
        validate_spikes_data(bad)


def test_reference_selection_and_filter_active_verbose_false():
    ds = fixture_dataset()
    refs = ds.select_ref_electrodes(PrepConfig(mode="top", top_start=0, top_stop=2, verbose=False))
    assert refs[0].tolist() == [102, 101]
    filtered = ds.filter_active(PrepConfig(mode="top", top_start=0, top_stop=2, verbose=False))
    assert set(filtered.recordings[0].spikes["electrode"].tolist()) == {101, 102}


def test_calculate_ifr_simple_spike_train():
    ifr_data, total_firing, all_values = calculate_ifr(fixture_spikes(), [101], start_time=0.0, end_time=1.0)
    times, values = ifr_data[101]
    assert total_firing[101] == 2.0
    assert 10.0 in values
    assert times[0] == 0.0
    assert times[-1] == 1.0
    assert all_values.size > 0


def test_cofiring_proportions_window_behavior():
    spikes = pd.DataFrame(
        {
            "time": [0.0, 0.001, 0.003, 0.010],
            "electrode": [1, 2, 2, 3],
        }
    )
    props = cofiring_proportions(spikes, np.array([0.0]), window_size=0.002, delay=0.0, ref_electrode=1)
    assert props[2] == 1.0
    assert props[3] == 0.0


def test_distance_assignment_preserves_mapping():
    spikes_df, layout_df = assign_r_distance(pd.DataFrame(fixture_spikes()), pd.DataFrame(fixture_layout()), 101)
    dist_by_electrode = layout_df.set_index("electrode")["distance"].to_dict()
    assert dist_by_electrode[101] == 0.0
    assert dist_by_electrode[102] == 5.0
    assert set(spikes_df["electrode"]) == {101, 102, 103}


def test_randomize_electrode_mapping_is_deterministic():
    rec = fixture_dataset().recordings[0]
    a = rec.randomize_electrode_mapping(seed=7)
    b = rec.randomize_electrode_mapping(seed=7)
    np.testing.assert_array_equal(a.layout["electrode"], b.layout["electrode"])
    np.testing.assert_array_equal(a.spikes["electrode"], b.spikes["electrode"])
