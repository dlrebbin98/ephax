import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ephax import FiringDistanceAnalyzer, PrepConfig, Recording, RestingActivityDataset
from ephax.metrics.firing_distance import avg_rate_vs_distance, cofiring_avg_vs_distance
from ephax.models import CofiringDistanceResult, FRDistanceResult
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


def test_firing_distance_analyzer_delegates_compute_to_metrics():
    ds = fixture_dataset()
    analyzer = FiringDistanceAnalyzer(
        ds,
        refs_per_recording=[np.array([101])],
        selection_prep_config=PrepConfig(mode="top", top_start=0, top_stop=2, verbose=False),
    )

    rate = analyzer.avg_rate_vs_distance(min_distance=0, max_distance=10)
    cofiring = analyzer.cofiring_avg_vs_distance(plusminus_ms=150.0, min_distance=0, max_distance=10)

    assert rate.distances.tolist() == [5.0, 5.0]
    assert cofiring.distances.tolist() == [5.0, 5.0]
    assert cofiring.proportions.tolist() == [0.5, 0.0]


def test_firing_distance_analyzer_uses_fixed_frequency_values():
    ds = fixture_dataset()
    analyzer = FiringDistanceAnalyzer(
        ds,
        refs_per_recording=[np.array([101])],
        frequency_values_hz=np.array([12.0, 40.0, 120.0]),
    )

    ok, gamma_hz, weights = analyzer._compute_ifr_peaks_weights(peak_min_hz=30.0, peak_max_hz=100.0)

    assert ok
    assert gamma_hz.tolist() == [40.0]
    assert weights.tolist() == [1.0]


def test_fixed_frequency_values_skip_ifr_gmm(monkeypatch):
    ds = fixture_dataset()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("GMM fitting should not run when fixed frequencies are supplied")

    monkeypatch.setattr("ephax.analyzers.ifr.IFRAnalyzer.fit_gmm", fail_if_called)
    analyzer = FiringDistanceAnalyzer(
        ds,
        refs_per_recording=[np.array([101])],
        frequency_values_hz=np.array([40.0]),
    )

    curve = analyzer.correlation_curve(peak_min_hz=30.0, peak_max_hz=100.0)

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
