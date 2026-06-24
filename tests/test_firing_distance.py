import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ephax import Recording, RestingActivityDataset
from ephax.metrics.firing_distance import (
    CofiringDistanceResult,
    FRDistanceResult,
    avg_rate_vs_distance,
    cofiring_avg_vs_distance,
    correlation_curve_from_frequencies,
    frequency_peak_weights,
)
from ephax.plotting.firing_distance import plot_binned_distance_series


def fixture_dataset():
    spikes = {
        "time": np.array([0.0, 0.1, 0.2, 0.3, 0.9, 1.0]),
        "channel": np.array([1, 1, 2, 2, 3, 3]),
        "amplitude": np.array([10, 11, 12, 13, 14, 15]),
        "electrode": np.array([101, 101, 102, 102, 103, 103]),
    }
    layout = {
        "channel": np.array([1, 2, 3]),
        "electrode": np.array([101, 102, 103]),
        "x": np.array([0.0, 3.0, 0.0]),
        "y": np.array([0.0, 4.0, 5.0]),
    }
    rec = Recording(spikes=spikes, layout=layout, start_time=0.0, end_time=1.0, sf=1000.0)
    return RestingActivityDataset([rec], sf=1000.0)


def test_firing_distance_metrics_are_pure_compute_entry_points():
    ds = fixture_dataset()
    refs = [np.array([101])]

    rate = avg_rate_vs_distance(ds.recordings, refs, min_distance=0, max_distance=10)
    cofiring = cofiring_avg_vs_distance(ds.recordings, refs, plusminus_ms=150.0, min_distance=0, max_distance=10)

    assert isinstance(rate, FRDistanceResult)
    assert isinstance(cofiring, CofiringDistanceResult)
    assert rate.distances.tolist() == [5.0, 5.0]
    assert rate.rates.tolist() == [2.0, 2.0]
    assert cofiring.distances.tolist() == [5.0, 5.0]
    assert cofiring.proportions.tolist() == [0.5, 0.0]


def test_frequency_peak_weights_select_explicit_values():
    ok, gamma_hz, weights = frequency_peak_weights(
        np.array([12.0, 40.0, 120.0]),
        peak_min_hz=30.0,
        peak_max_hz=100.0,
    )

    assert ok
    assert gamma_hz.tolist() == [40.0]
    assert weights.tolist() == [1.0]


def test_correlation_curve_from_frequencies_uses_explicit_values():
    curve = correlation_curve_from_frequencies(
        np.array([40.0]),
        peak_min_hz=30.0,
        peak_max_hz=100.0,
    )

    assert curve is not None
    r_um, values = curve
    assert r_um.size == values.size


def test_firing_distance_plotting_helper_consumes_result_object():
    ds = fixture_dataset()
    result = avg_rate_vs_distance(ds.recordings, [np.array([101])], min_distance=0, max_distance=10)

    fig, ax = plot_binned_distance_series(result, ylabel="Rate", title="Rate by distance")

    assert ax.get_ylabel() == "Rate"
    assert ax.get_title() == "Rate by distance"
    plt.close(fig)
