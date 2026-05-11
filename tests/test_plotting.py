import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ephax import IFRAnalyzer, IFRConfig, PrepConfig
from ephax.metrics.ifr import prepare_ifr_timeseries_panel
from ephax.models import CofiringHeatmap, GMMFit, IFRPeaks
from ephax.plotting.cofiring import plot_cofiring_heatmap
from ephax.plotting.ifr import plot_ifr_histogram, plot_ifr_timeseries, plot_ifr_timeseries_panel


def fixture_spikes():
    return {
        "time": np.array([0.0, 0.1, 0.2, 0.3, 0.9, 1.0]),
        "channel": np.array([1, 1, 2, 2, 3, 3]),
        "amplitude": np.array([10, 11, 12, 13, 14, 15]),
        "electrode": np.array([101, 101, 102, 102, 103, 103]),
    }


def fixture_dataset():
    from ephax import Recording, RestingActivityDataset

    layout = {
        "channel": np.array([1, 2, 3]),
        "electrode": np.array([101, 102, 103]),
        "x": np.array([0.0, 3.0, 0.0]),
        "y": np.array([0.0, 4.0, 5.0]),
    }
    rec = Recording(spikes=fixture_spikes(), layout=layout, start_time=0.0, end_time=1.0, sf=1000.0)
    return RestingActivityDataset([rec], sf=1000.0)


def test_plot_ifr_histogram_returns_figure_axes():
    x = np.linspace(0.0, 2.0, 50)
    peaks = IFRPeaks(
        values=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        kde_x=x,
        kde_y=np.exp(-((x - 0.3) ** 2)),
        peaks_x=np.array([0.3]),
        peaks_y=np.array([1.0]),
        peaks_hz=np.array([2.0]),
    )
    cfg = IFRConfig(log_scale=False, overlay_gmm=True, show_kde=True, show_peaks=True)
    fit = GMMFit(means_hz=np.array([0.3]), std=np.array([0.1]), weights=np.array([1.0]))

    fig, ax = plot_ifr_histogram(peaks, cfg, fit=fit, hist_bins=5)

    assert fig is not None
    assert ax.get_xlabel() == "IFR (Hz)"
    plt.close(fig)


def test_plot_ifr_timeseries_returns_recording_figures():
    cfg = IFRConfig(log_scale=False, overlay_gmm=False, time_grid_hz=10.0, max_time_points=20, ts_bins=5)

    results = plot_ifr_timeseries(
        [fixture_spikes()],
        [0.0],
        [1.0],
        [[101, 102]],
        cfg,
        title="test",
        recording_titles=["rec0"],
    )

    assert len(results) == 1
    fig, (ax_heatmap, ax_hist) = results[0]
    assert "rec0" in ax_heatmap.get_title()
    assert ax_hist.get_ylabel() == "Frequency"
    plt.close(fig)


def test_plot_ifr_timeseries_requires_nested_selection_for_multiple_recordings():
    cfg = IFRConfig(log_scale=False, overlay_gmm=False, time_grid_hz=10.0, max_time_points=20, ts_bins=5)

    with pytest.raises(ValueError, match="multiple recordings require nested"):
        plot_ifr_timeseries(
            [fixture_spikes(), fixture_spikes()],
            [0.0, 0.0],
            [1.0, 1.0],
            [101, 102],
            cfg,
        )


def test_prepare_ifr_timeseries_panel_keeps_computation_out_of_plotting():
    panel = prepare_ifr_timeseries_panel(
        fixture_spikes(),
        [101, 102],
        0.0,
        1.0,
        log_scale=False,
        time_grid_hz=10.0,
        max_time_points=20,
    )

    assert panel is not None
    assert panel.heatmap.shape == (2, 10)
    assert panel.electrodes.tolist() == [101, 102]
    assert panel.histogram_values.size > 0

    cfg = IFRConfig(log_scale=False, overlay_gmm=False, ts_bins=5)
    fig, (ax_heatmap, ax_hist) = plot_ifr_timeseries_panel(panel, cfg, recording_label="rec0")
    assert "rec0" in ax_heatmap.get_title()
    assert ax_hist.get_ylabel() == "Frequency"
    plt.close(fig)


def test_plot_cofiring_heatmap_returns_figure_axes():
    heatmap = CofiringHeatmap(
        Z=np.array([[0.1, 0.2], [0.3, 0.4]]),
        distance_bins=np.array([0.0, 100.0, 200.0]),
        delays=np.array([-1.0, 0.0, 1.0]),
    )

    fig, ax = plot_cofiring_heatmap(heatmap, normalize=False)

    assert fig is not None
    assert ax.get_ylabel() == "Delay (ms)"
    plt.close(fig)


def test_ifr_analyzer_plot_methods_delegate_to_plotting():
    ds = fixture_dataset()
    prep = PrepConfig(mode="top", top_start=0, top_stop=2, verbose=False)
    cfg = IFRConfig(log_scale=False, overlay_gmm=False, time_grid_hz=10.0, max_time_points=20, ts_bins=5)
    analyzer = IFRAnalyzer.from_dataset(ds, config=cfg, selection_prep_config=prep)

    hist_fig, hist_ax = analyzer.plot_histogram(hist_bins=5)
    ts_results = analyzer.plot_timeseries(recording_titles=["rec0"])
    panels = analyzer.timeseries_panels()

    assert hist_ax.get_ylabel() == "Density"
    assert len(ts_results) == 1
    assert len(panels) == 1
    plt.close(hist_fig)
    plt.close(ts_results[0][0])
