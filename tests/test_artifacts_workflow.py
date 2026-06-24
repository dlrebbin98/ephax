from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ephax.artifacts import (
    RunPaths,
    ensure_run_dirs,
    load_ifr_timeseries_panels,
    save_csv,
    save_figure,
    save_ifr_timeseries_checkpoints,
    write_manifest,
)
from ephax.cli.build_atomic_figures import build_ifr_timeseries_figures
from ephax.metrics.ifr import IFRConfig, prepare_ifr_timeseries_panels
from ephax.preprocessing.dataset import Recording, RestingActivityDataset
from ephax import workflows


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


def workflow_config(tmp_path: Path):
    return {
        "run": {
            "id": "test_run",
            "output_dir": str(tmp_path),
            "figure_formats": ["png"],
        },
        "dataset": {
            "source": "h5",
            "data_root": "unused",
            "min_amp": 0,
            "file_info": [
                {
                    "folder": "unused",
                    "filename": "recording.raw.h5",
                    "recording_id": "culture_a_div12_well1",
                    "culture_id": "culture_a",
                    "div": 12,
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "well": 1,
                }
            ],
        },
        "selection": {
            "mode": "top",
            "top_start": 0,
            "top_stop": 2,
            "verbose": False,
        },
        "analyses": {},
    }


def test_run_paths_from_config(tmp_path):
    config = {"run": {"id": "abc", "output_dir": str(tmp_path)}}
    paths = RunPaths.from_config(config)
    assert paths.run_id == "abc"
    assert paths.run_dir == tmp_path / "abc"
    assert paths.figure_data == tmp_path / "abc" / "figure_data"
    assert paths.manifest == tmp_path / "abc" / "manifest.yaml"


def test_manifest_csv_and_figure_helpers(tmp_path):
    paths = ensure_run_dirs(RunPaths.from_config({"run": {"id": "abc", "output_dir": str(tmp_path)}}))
    manifest = write_manifest(paths, {"run": {"id": "abc"}}, metadata={"answer": 42})
    with manifest.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    assert payload["run_id"] == "abc"
    assert payload["metadata"]["answer"] == 42

    csv_path = save_csv(pd.DataFrame({"x": [1, 2]}), paths.figure_data / "tiny")
    reloaded = pd.read_csv(csv_path)
    assert reloaded["x"].tolist() == [1, 2]

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    saved = save_figure(fig, paths.atomic_figures / "line", formats=("png",))
    plt.close(fig)
    assert saved == [paths.atomic_figures / "line.png"]
    assert saved[0].exists()


def test_run_workflow_writes_checkpoint_csvs(monkeypatch, tmp_path):
    monkeypatch.setattr(workflows, "build_dataset", lambda config: fixture_dataset())

    result = workflows.run_workflow(workflow_config(tmp_path))

    assert result["manifest"].exists()
    assert result["recordings_csv"].exists()
    assert result["selected_refs_csv"].exists()
    assert result["output_dir"] == tmp_path / "test_run"

    recordings = pd.read_csv(result["recordings_csv"])
    assert recordings.loc[0, "recording_id"] == "culture_a_div12_well1"
    assert recordings.loc[0, "culture_id"] == "culture_a"
    assert recordings.loc[0, "div"] == 12
    assert recordings.loc[0, "n_spikes"] == 6
    assert recordings.loc[0, "n_electrodes"] == 3

    refs = pd.read_csv(result["selected_refs_csv"])
    assert refs["electrode"].tolist() == [102, 101]
    assert refs["recording_id"].tolist() == ["culture_a_div12_well1", "culture_a_div12_well1"]


def test_ifr_timeseries_checkpoints_round_trip_and_atomic_figure(tmp_path):
    paths = ensure_run_dirs(RunPaths.from_config({"run": {"id": "abc", "output_dir": str(tmp_path)}}))
    ds = fixture_dataset()
    refs = ds.select_ref_electrodes(workflows.build_prep_config({"selection": {"mode": "top", "top_start": 0, "top_stop": 2}}))
    cfg = IFRConfig(log_scale=False, overlay_gmm=False, time_grid_hz=10.0, max_time_points=20, ts_bins=5)
    spikes_list = [rec.spikes for rec in ds.recordings]
    start_times = [rec.start_time for rec in ds.recordings]
    end_times = [rec.end_time for rec in ds.recordings]
    panels = prepare_ifr_timeseries_panels(
        spikes_list,
        start_times,
        end_times,
        refs,
        log_scale=cfg.log_scale,
        time_grid_hz=cfg.time_grid_hz,
        max_time_points=cfg.max_time_points,
    )

    saved = save_ifr_timeseries_checkpoints(panels, paths)
    assert saved["heatmap"].exists()
    assert saved["histogram"].exists()
    assert saved["electrodes"].exists()

    reloaded = load_ifr_timeseries_panels(paths.figure_data)
    assert len(reloaded) == 1
    np.testing.assert_array_equal(reloaded[0].electrodes, panels[0].electrodes)
    np.testing.assert_allclose(reloaded[0].heatmap, panels[0].heatmap)

    figures = build_ifr_timeseries_figures(paths.run_dir, ts_bins=5)
    assert figures == [paths.atomic_figures / "ifr_timeseries_recording_0.png"]
    assert figures[0].exists()


def test_run_workflow_writes_ifr_checkpoints_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(workflows, "build_dataset", lambda config: fixture_dataset())
    config = workflow_config(tmp_path)
    config["analyses"] = {
        "ifr": {
            "enabled": True,
            "log_scale": False,
            "overlay_gmm": False,
            "time_grid_hz": 10.0,
            "max_time_points": 20,
            "ts_bins": 5,
            "write_timeseries_checkpoints": True,
        }
    }

    result = workflows.run_workflow(config)

    assert result["ifr_timeseries_checkpoints"]["heatmap"].exists()
    with result["manifest"].open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    assert "ifr_timeseries_heatmap" in manifest["metadata"]["checkpoints"]
