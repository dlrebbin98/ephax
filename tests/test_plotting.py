import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ephax import PrepConfig
from ephax.metrics.ifr import IFRConfig, ifr_peaks, prepare_ifr_timeseries_panel, prepare_ifr_timeseries_panels
from ephax.metrics.cofiring import CofiringHeatmap
from ephax.metrics.burst import PopulationIFR
from ephax.metrics.ifr import IFRPeaks
from ephax.metrics.layout_grid import GridResult
from ephax.modeling.gmm import GMMFit, fit_ifr_gmm
from ephax.plotting.burst import (
    draw_population_ifr_summary,
    plot_population_ifr_summary,
    population_ifr_summary_axes_factory,
)
from ephax.plotting.cofiring import draw_cofiring_heatmap, plot_cofiring_heatmap
from ephax.plotting.compound import PanelGroupSpec, PanelRenderSpec, compose_figure, figure, group, panel
from ephax.plotting.export import export_figure
from ephax.plotting.ifr import (
    draw_ifr_timeseries_panel,
    ifr_timeseries_axes_factory,
    plot_ifr_histogram,
    plot_ifr_timeseries,
    plot_ifr_timeseries_panel,
    plot_ifr_timeseries_panels,
)
from ephax.plotting.layout_grid import draw_grid_avghz, draw_grid_avghz_panel, grid_avghz_panel_axes_factory
from ephax.plotting.layout import FigureSpec, PanelSpec, add_panel_axes, make_figure_grid
from ephax.plotting.panels import add_panel_bundle_label, add_panel_label, add_panel_suptitle
from ephax.plotting.style import (
    NATURE_STYLE,
    PAPER_COLORS,
    PAPER_FONT_FAMILY,
    apply_nature_style,
    apply_paper_style,
    figure_mode_defaults,
    mm_to_inches,
    nature_figure_check,
    nature_figure_size,
    standalone_figure_size,
)


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


def fixture_population_ifr():
    time_grid = np.linspace(0.0, 1.0, 10)
    ifr_matrix = np.vstack([
        np.linspace(1.0, 5.0, 10),
        np.linspace(2.0, 8.0, 10),
        np.linspace(3.0, 10.0, 10),
    ])
    return PopulationIFR(
        time_grid=time_grid,
        electrodes=np.array([101, 102, 103]),
        ifr_matrix=ifr_matrix,
        mean_ifr=ifr_matrix.mean(axis=0),
        mean_ifr_smooth=ifr_matrix.mean(axis=0),
        per_electrode_mean_hz=ifr_matrix.mean(axis=1),
    )


def fixture_cofiring_heatmap():
    return CofiringHeatmap(
        Z=np.array([[0.1, 0.2], [0.3, 0.4]]),
        distance_bins=np.array([0.0, 100.0, 200.0]),
        delays=np.array([-1.0, 0.0, 1.0]),
    )


def fixture_grid_result():
    return GridResult(
        grid=np.array([[0.1, 0.2], [0.3, np.nan]]),
        x_min=0.0,
        x_max=100.0,
        y_min=0.0,
        y_max=100.0,
        vmin=0.1,
        vmax=0.3,
    )


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


def test_draw_ifr_timeseries_panel_uses_provided_axes():
    panel = prepare_ifr_timeseries_panel(
        fixture_spikes(),
        [101, 102],
        0.0,
        1.0,
        log_scale=False,
        time_grid_hz=10.0,
        max_time_points=20,
    )
    cfg = IFRConfig(log_scale=False, overlay_gmm=False, ts_bins=5)
    fig = plt.figure(figsize=(6, 4), constrained_layout=True)
    axes = ifr_timeseries_axes_factory(fig, fig.add_gridspec(1, 1)[0, 0])

    rendered = draw_ifr_timeseries_panel(panel, axes, cfg, recording_label="rec0", compact=True)

    assert rendered["heatmap"] is not None
    assert axes[1].get_ylabel() == "Frequency"
    plt.close(fig)


def test_plot_cofiring_heatmap_returns_figure_axes():
    heatmap = fixture_cofiring_heatmap()

    fig, ax = plot_cofiring_heatmap(heatmap, normalize=False)

    assert fig is not None
    assert ax.get_ylabel() == "Delay (ms)"
    plt.close(fig)


def test_draw_cofiring_heatmap_uses_provided_axes():
    fig, ax = plt.subplots(figsize=(4, 3))

    rendered = draw_cofiring_heatmap(fixture_cofiring_heatmap(), ax, show_colorbar=False, compact=True)

    assert rendered["heatmap"] is not None
    assert rendered["colorbar"] is None
    assert ax.get_xlabel() == "Distance from Electrode ($\\mu m$)"
    plt.close(fig)


def test_draw_cofiring_heatmap_accepts_title_override_in_compact_mode():
    fig, ax = plt.subplots(figsize=(4, 3))

    draw_cofiring_heatmap(fixture_cofiring_heatmap(), ax, title="custom cofiring", show_colorbar=False, compact=True)

    assert ax.get_title() == "custom cofiring"
    plt.close(fig)


def test_draw_grid_avghz_uses_provided_axes():
    fig, ax = plt.subplots(figsize=(3, 2))

    rendered = draw_grid_avghz(fixture_grid_result(), ax, title="empirical", compact=True, show_colorbar=False)

    assert rendered["mappable"] is not None
    assert rendered["colorbar"] is None
    assert ax.get_title() == "empirical"
    assert ax.get_facecolor() == (0.0, 0.0, 0.0, 1.0)
    plt.close(fig)


def test_draw_grid_avghz_panel_uses_provided_axes():
    fig = plt.figure(figsize=(6, 3), constrained_layout=True)
    axes = grid_avghz_panel_axes_factory(fig, fig.add_gridspec(1, 1)[0, 0], n_items=2, ncols=2)

    rendered = draw_grid_avghz_panel(
        [fixture_grid_result(), fixture_grid_result()],
        axes,
        recording_titles=["well 0", "well 1"],
        compact=True,
    )

    assert rendered["mappable"] is not None
    assert rendered["colorbar"] is not None
    assert axes[0].reshape(-1)[0].get_title() == "well 0"
    assert axes[0].reshape(-1)[1].get_facecolor() == (0.0, 0.0, 0.0, 1.0)
    plt.close(fig)


def test_plot_population_ifr_summary_returns_figure_axes():
    fig, (ax_heatmap, ax_mean) = plot_population_ifr_summary(fixture_population_ifr())

    assert ax_heatmap.get_ylabel() == "Electrode index"
    assert ax_mean.get_xlabel() == "Time (s)"
    plt.close(fig)


def test_draw_population_ifr_summary_uses_provided_axes():
    fig = plt.figure(figsize=(6, 4), constrained_layout=True)
    axes = population_ifr_summary_axes_factory(fig, fig.add_gridspec(1, 1)[0, 0])

    rendered = draw_population_ifr_summary(fixture_population_ifr(), axes, compact=True)

    assert rendered["heatmap"] is not None
    assert axes[1].get_ylabel() == "Mean IFR (Hz)"
    assert axes[0].get_title() == ""
    plt.close(fig)


def test_draw_population_ifr_summary_can_show_titles_in_compact_mode():
    fig = plt.figure(figsize=(6, 4), constrained_layout=True)
    axes = population_ifr_summary_axes_factory(fig, fig.add_gridspec(1, 1)[0, 0])

    draw_population_ifr_summary(
        fixture_population_ifr(),
        axes,
        compact=True,
        show_titles=True,
        heatmap_title="heatmap override",
        mean_title="mean override",
    )

    assert axes[0].get_title() == "heatmap override"
    assert axes[1].get_title() == "mean override"
    plt.close(fig)


def test_make_figure_grid_and_panel_validation():
    fig, grid = make_figure_grid(FigureSpec(width=8, height=4, row_heights=[1.0, 1.0]), nrows=2)
    ax = add_panel_axes(fig, grid, PanelSpec(key="a", label="A", row=0, col=0, colspan=6))

    assert ax.figure is fig
    with pytest.raises(ValueError, match="extends beyond the figure columns"):
        add_panel_axes(fig, grid, PanelSpec(key="bad", label=None, row=0, col=10, colspan=4))
    plt.close(fig)


def test_make_figure_grid_supports_explicit_row_gaps():
    fig, grid = make_figure_grid(
        FigureSpec(width=8, height=4, row_heights=[1.0, 1.0, 1.0], row_gaps=[0.2, 0.4]),
        nrows=3,
    )

    assert grid.get_geometry()[0] == 5
    assert grid._ephax_row_map == [0, 2, 4]

    top = add_panel_axes(fig, grid, PanelSpec(key="top", label=None, row=0, col=0))
    bottom = add_panel_axes(fig, grid, PanelSpec(key="bottom", label=None, row=2, col=0))

    assert top.figure is fig
    assert bottom.figure is fig
    with pytest.raises(ValueError, match="one value between each figure row"):
        make_figure_grid(FigureSpec(width=8, height=4, row_heights=[1.0, 1.0, 1.0], row_gaps=[0.2]), nrows=3)
    plt.close(fig)


def test_nature_size_helpers_and_style_defaults():
    assert mm_to_inches(25.4) == pytest.approx(1.0)
    assert nature_figure_size("one_column", height_mm=50.0) == pytest.approx((88.0 / 25.4, 50.0 / 25.4))
    assert nature_figure_size("two_column", height_mm=80.0) == pytest.approx((180.0 / 25.4, 80.0 / 25.4))

    params = apply_nature_style()

    assert params["font.size"] == NATURE_STYLE.font_size
    assert plt.rcParams["axes.grid"] is False
    assert plt.rcParams["figure.dpi"] == NATURE_STYLE.inline_dpi
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["ps.fonttype"] == 42


def test_paper_style_uses_paper_palette_and_dejavu_font():
    params = apply_paper_style()

    assert PAPER_COLORS["axonal_only"] == "#40bfff"
    assert PAPER_COLORS["ephaptic_axonal"] == "#807fff"
    assert PAPER_COLORS["low_activity"] == "#8c8c8c"
    assert PAPER_COLORS["high_activity"] == "#2ca02c"
    assert PAPER_COLORS["burst"] == "#1f77b4"
    assert params["font.family"] == PAPER_FONT_FAMILY
    assert plt.rcParams["font.family"] == PAPER_FONT_FAMILY


def test_standalone_size_and_mode_defaults():
    assert standalone_figure_size("small_single") == pytest.approx((100.0 / 25.4, 75.0 / 25.4))
    assert standalone_figure_size("medium_single") == pytest.approx((140.0 / 25.4, 100.0 / 25.4))
    assert standalone_figure_size("wide_single") == pytest.approx((180.0 / 25.4, 115.0 / 25.4))

    standalone = figure_mode_defaults("standalone")
    panel = figure_mode_defaults("panel")
    paper = figure_mode_defaults("paper")

    assert standalone.compact is False
    assert standalone.show_legend is True
    assert standalone.show_colorbar is True
    assert panel.compact is True
    assert panel.show_legend is False
    assert paper.compact is True


def test_standalone_plot_wrappers_use_content_sized_defaults():
    fig, _ax = plot_cofiring_heatmap(fixture_cofiring_heatmap())

    width_in, height_in = fig.get_size_inches()
    assert width_in == pytest.approx(140.0 / 25.4)
    assert height_in == pytest.approx(100.0 / 25.4)
    plt.close(fig)


def test_ifr_and_population_wrappers_use_wide_single_defaults():
    ifr_panel = prepare_ifr_timeseries_panel(
        fixture_spikes(),
        [101, 102],
        0.0,
        1.0,
        log_scale=False,
        time_grid_hz=10.0,
        max_time_points=20,
    )
    cfg = IFRConfig(log_scale=False, overlay_gmm=False, ts_bins=5)

    ifr_fig, _axes = plot_ifr_timeseries_panel(ifr_panel, cfg)
    pop_fig, _axes = plot_population_ifr_summary(fixture_population_ifr())

    assert ifr_fig.get_size_inches()[0] == pytest.approx(180.0 / 25.4)
    assert pop_fig.get_size_inches()[0] == pytest.approx(180.0 / 25.4)
    plt.close(ifr_fig)
    plt.close(pop_fig)


def test_figure_spec_accepts_nature_width_preset():
    fig, _grid = make_figure_grid(FigureSpec(width="two_column", height_mm=90.0))

    width_in, height_in = fig.get_size_inches()
    assert width_in == pytest.approx(180.0 / 25.4)
    assert height_in == pytest.approx(90.0 / 25.4)
    plt.close(fig)


def test_panel_label_defaults_to_nature_lowercase_style():
    fig, ax = plt.subplots(figsize=(3, 2))

    text = add_panel_label(ax, "A")

    assert text.get_text() == "a"
    assert text.get_fontsize() == NATURE_STYLE.panel_label_size
    assert text.get_fontweight() == "bold"
    plt.close(fig)


def test_panel_label_can_keep_custom_label_case():
    fig, ax = plt.subplots(figsize=(3, 2))

    text = add_panel_label(ax, "A", lowercase=False)

    assert text.get_text() == "A"
    plt.close(fig)


def test_nature_figure_check_reports_common_issues():
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.text(0.5, 0.5, "too large", fontsize=12)
    add_panel_label(ax, "A", lowercase=False)

    warnings = nature_figure_check(fig)

    assert any("does not match a Nature width preset" in warning for warning in warnings)
    assert any("should be lowercase" in warning for warning in warnings)
    assert any("should be 5-7 pt" in warning for warning in warnings)
    plt.close(fig)


def test_compose_figure_mixes_reusable_panels():
    ifr_panel = prepare_ifr_timeseries_panel(
        fixture_spikes(),
        [101, 102],
        0.0,
        1.0,
        log_scale=False,
        time_grid_hz=10.0,
        max_time_points=20,
    )
    cfg = IFRConfig(log_scale=False, overlay_gmm=False, ts_bins=5)
    composed = compose_figure(
        FigureSpec(width=10, height=7, row_heights=[1.0, 1.0]),
        [
            PanelRenderSpec(
                panel=PanelSpec(key="ifr", label="A", row=0, col=0, colspan=6, compact=True),
                draw=lambda data, axes, **opts: draw_ifr_timeseries_panel(data, axes, cfg, **opts),
                data=ifr_panel,
                axes_factory=ifr_timeseries_axes_factory,
            ),
            PanelRenderSpec(
                panel=PanelSpec(key="burst", label="B", row=0, col=6, colspan=6, compact=True),
                draw=draw_population_ifr_summary,
                data=fixture_population_ifr(),
                axes_factory=population_ifr_summary_axes_factory,
            ),
            PanelRenderSpec(
                panel=PanelSpec(key="cofiring", label="C", row=1, col=0, colspan=12, compact=True),
                draw=draw_cofiring_heatmap,
                data=fixture_cofiring_heatmap(),
            ),
        ],
    )

    assert set(composed.axes) == {"ifr", "burst", "cofiring"}
    assert set(composed.rendered) == {"ifr", "burst", "cofiring"}
    plt.close(composed.fig)


def test_simple_figure_panel_group_api_matches_composition_behavior():
    ifr_panel = prepare_ifr_timeseries_panel(
        fixture_spikes(),
        [101, 102],
        0.0,
        1.0,
        log_scale=False,
        time_grid_hz=10.0,
        max_time_points=20,
    )
    cfg = IFRConfig(log_scale=False, overlay_gmm=False, ts_bins=5)

    composed = compose_figure(
        figure(width=8, height=5, rows=[1.0, 1.0]),
        [
            panel(
                "ifr",
                label="a",
                loc=(0, 0, 1, 6),
                draw=lambda data, axes, **opts: draw_ifr_timeseries_panel(data, axes, cfg, **opts),
                data=ifr_panel,
                axes_factory=ifr_timeseries_axes_factory,
            ),
            group(
                "days",
                label="b",
                loc=(1, 0, 1, 12),
                children=[
                    panel("day1", draw=draw_cofiring_heatmap, data=fixture_cofiring_heatmap()),
                    panel("day2", draw=draw_cofiring_heatmap, data=fixture_cofiring_heatmap()),
                ],
                shared_colorbar=True,
            ),
        ],
    )

    assert set(composed.axes) == {"ifr", "days"}
    assert set(composed.axes["days"]) == {"day1", "day2", "colorbar"}
    assert "colorbar" in composed.rendered["days"]
    plt.close(composed.fig)


def test_panel_title_is_forwarded_as_default_draw_option():
    def draw_line(_data, ax, **opts):
        ax.plot([0, 1], [0, 1])
        if opts.get("title"):
            ax.set_title(opts["title"])
        return {"axes": ax}

    composed = compose_figure(
        figure(width=4, height=3),
        [
            panel(
                "line",
                title="forwarded title",
                draw=draw_line,
            )
        ],
    )

    assert composed.axes["line"].get_title() == "forwarded title"
    plt.close(composed.fig)


def test_compose_figure_adds_panel_suptitles():
    def draw_line(_data, ax, **_opts):
        ax.plot([0, 1], [0, 1])
        return {"axes": ax}

    composed = compose_figure(
        figure(width=6, height=3),
        [
            panel(
                "left",
                label="a",
                suptitle="Nested panel title",
                loc=(0, 0, 1, 6),
                draw=draw_line,
            ),
            group(
                "right",
                label="b",
                suptitle="Grouped panel title",
                loc=(0, 6, 1, 6),
                children=[
                    panel("child1", draw=draw_cofiring_heatmap, data=fixture_cofiring_heatmap()),
                    panel("child2", draw=draw_cofiring_heatmap, data=fixture_cofiring_heatmap()),
                ],
                shared_colorbar=True,
            ),
        ],
    )

    axes_text = [text.get_text() for ax in composed.fig.axes for text in ax.texts]
    assert "Nested panel title" in axes_text
    assert "Grouped panel title" in axes_text
    assert "a" in axes_text
    assert "b" in axes_text
    plt.close(composed.fig)


def test_compose_figure_labels_numpy_axes_bundle_without_suptitle():
    def axes_factory(fig, subplotspec):
        gs = subplotspec.subgridspec(2, 1)
        return np.asarray([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])], dtype=object)

    def draw_lines(_data, axes, **_opts):
        for ax in axes:
            ax.plot([0, 1], [0, 1])
        return {"axes": axes}

    composed = compose_figure(
        figure(width=4, height=3),
        [
            panel(
                "array_bundle",
                label="d",
                draw=draw_lines,
                axes_factory=axes_factory,
            )
        ],
    )

    axes_text = [text.get_text() for ax in composed.fig.axes for text in ax.texts]
    assert "d" in axes_text
    plt.close(composed.fig)


def test_compose_figure_panel_labels_align_with_and_without_suptitle():
    def draw_line(_data, ax, **_opts):
        ax.plot([0, 1], [0, 1])
        return {"axes": ax}

    composed = compose_figure(
        figure(width=6, height=3, constrained_layout=False),
        [
            panel("plain", label="a", loc=(0, 0, 1, 6), draw=draw_line),
            panel("titled", label="b", suptitle="Panel title", loc=(0, 6, 1, 6), draw=draw_line),
        ],
    )
    composed.fig.subplots_adjust(left=0.02, right=0.98)
    composed.fig.canvas.draw()

    labels = {
        text.get_text(): text
        for ax in composed.fig.axes
        for text in ax.texts
        if text.get_text() in {"a", "b"}
    }
    label_positions = {
        label: text.get_transform().transform(text.get_position())
        for label, text in labels.items()
    }
    assert label_positions["a"][1] == pytest.approx(label_positions["b"][1])
    plt.close(composed.fig)


def test_panel_suptitle_clears_child_axes_titles():
    fig, axes = plt.subplots(1, 2, figsize=(5, 2), constrained_layout=True)
    for idx, ax in enumerate(axes):
        ax.plot([0, 1], [idx, idx + 1])
        ax.set_title(f"child {idx + 1}")

    text = add_panel_suptitle(axes, "parent title", y_pad=0.01)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    parent_bbox = text.get_window_extent(renderer)
    child_top = max(ax.title.get_window_extent(renderer).y1 for ax in axes)

    assert parent_bbox.y0 > child_top
    plt.close(fig)


def test_panel_bundle_label_can_include_suptitle():
    fig, axes = plt.subplots(1, 2, figsize=(5, 2), constrained_layout=True)
    for idx, ax in enumerate(axes):
        ax.plot([0, 1], [idx, idx + 1])
        ax.set_title(f"child {idx + 1}")

    suptitle = add_panel_suptitle(axes, "parent title", y_pad=0.01)
    label = add_panel_bundle_label(axes, "A", extra_artists=[suptitle])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_bbox = label.get_window_extent(renderer)

    assert label.get_text() == "a"
    assert label_bbox.x0 >= 0
    assert label_bbox.y0 >= suptitle.get_window_extent(renderer).y0
    plt.close(fig)


def test_panel_group_renders_one_label_and_shared_colorbar():
    group = PanelGroupSpec(
        panel=PanelSpec(key="group", label="A", row=0, col=0, colspan=12, compact=True),
        shared_colorbar=True,
        children=[
            PanelRenderSpec(
                panel=PanelSpec(key="day1", label=None, row=0, col=0, compact=True),
                draw=draw_cofiring_heatmap,
                data=fixture_cofiring_heatmap(),
            ),
            PanelRenderSpec(
                panel=PanelSpec(key="day2", label=None, row=0, col=0, compact=True),
                draw=draw_cofiring_heatmap,
                data=fixture_cofiring_heatmap(),
            ),
        ],
    )

    composed = compose_figure(FigureSpec(width=8, height=3), [group])

    assert set(composed.axes["group"]) == {"day1", "day2", "colorbar"}
    assert "colorbar" in composed.rendered["group"]
    labels = [text.get_text() for text in composed.fig.texts]
    labels.extend(text.get_text() for ax in composed.fig.axes for text in ax.texts)
    assert labels.count("a") == 1
    plt.close(composed.fig)


def test_export_figure_writes_multiple_formats(tmp_path):
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0, 1])

    saved = export_figure(fig, tmp_path / "figure_panel", dpi=72)

    assert [path.suffix for path in saved] == [".pdf", ".svg", ".png"]
    assert all(path.exists() for path in saved)
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["svg.fonttype"] == "none"
    plt.close(fig)


def test_ifr_functional_plot_path_builds_histogram_and_timeseries():
    ds = fixture_dataset()
    prep = PrepConfig(mode="top", top_start=0, top_stop=2, verbose=False)
    cfg = IFRConfig(log_scale=False, overlay_gmm=False, time_grid_hz=10.0, max_time_points=20, ts_bins=5)
    refs = ds.select_ref_electrodes(prep)
    spikes_list = [rec.spikes for rec in ds.recordings]
    start_times = [rec.start_time for rec in ds.recordings]
    end_times = [rec.end_time for rec in ds.recordings]
    peaks = ifr_peaks(
        spikes_list,
        start_times,
        end_times,
        log_scale=cfg.log_scale,
        selected_refs_per_recording=refs,
    )
    fit = fit_ifr_gmm(peaks.values, log_scale=cfg.log_scale, n_components=cfg.n_components) if cfg.overlay_gmm else None
    panels = prepare_ifr_timeseries_panels(
        spikes_list,
        start_times,
        end_times,
        refs,
        log_scale=cfg.log_scale,
        time_grid_hz=cfg.time_grid_hz,
        max_time_points=cfg.max_time_points,
    )

    hist_fig, hist_ax = plot_ifr_histogram(peaks, cfg, fit=fit, hist_bins=5)
    ts_results = plot_ifr_timeseries_panels(panels, cfg, recording_titles=["rec0"])

    assert hist_ax.get_ylabel() == "Density"
    assert len(ts_results) == 1
    assert len(panels) == 1
    plt.close(hist_fig)
    plt.close(ts_results[0][0])
