from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from .models import DiscreteTEResult, IFRTimeSeriesPanel, WaveAnalysisResult


@dataclass(frozen=True)
class RunPaths:
    """Standard output paths for one reproducible analysis run."""

    run_id: str
    run_dir: Path
    figure_data: Path
    atomic_figures: Path
    composed_figures: Path
    logs: Path
    manifest: Path

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RunPaths":
        run_cfg = config.get("run", {})
        output_dir = Path(run_cfg.get("output_dir", "outputs")).expanduser()
        run_id = run_cfg.get("id")
        if run_id:
            run_dir = output_dir / str(run_id)
        else:
            run_dir = output_dir
            run_id = run_dir.name
        return cls(
            run_id=str(run_id),
            run_dir=run_dir,
            figure_data=run_dir / "figure_data",
            atomic_figures=run_dir / "atomic_figures",
            composed_figures=run_dir / "composed_figures",
            logs=run_dir / "logs",
            manifest=run_dir / "manifest.yaml",
        )


def ensure_run_dirs(paths: RunPaths) -> RunPaths:
    for directory in (
        paths.run_dir,
        paths.figure_data,
        paths.atomic_figures,
        paths.composed_figures,
        paths.logs,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def write_manifest(paths: RunPaths, config: dict[str, Any], metadata: dict[str, Any] | None = None) -> Path:
    ensure_run_dirs(paths)
    payload = {
        "run_id": paths.run_id,
        "run_dir": str(paths.run_dir),
        "config": config,
        "metadata": metadata or {},
    }
    with paths.manifest.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return paths.manifest


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    csv_path = Path(path)
    if csv_path.suffix != ".csv":
        csv_path = csv_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=index)
    return csv_path


def save_figure(fig, path_stem: str | Path, formats: Iterable[str] = ("png",)) -> list[Path]:
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        clean_fmt = str(fmt).lstrip(".")
        out_path = stem.with_suffix(f".{clean_fmt}")
        fig.savefig(out_path, bbox_inches="tight")
        saved.append(out_path)
    return saved


def ifr_timeseries_heatmap_dataframe(panels: Iterable[IFRTimeSeriesPanel]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel in panels:
        for row_idx, electrode in enumerate(panel.electrodes):
            for time_idx, (time_value, ifr_value) in enumerate(zip(panel.time_points, panel.heatmap[row_idx])):
                rows.append(
                    {
                        "recording_index": panel.recording_index,
                        "electrode": int(electrode),
                        "electrode_rank": row_idx,
                        "time_index": time_idx,
                        "time": float(time_value),
                        "ifr_value": float(ifr_value),
                        "start_time": panel.start_time,
                        "end_time": panel.end_time,
                        "log_scale": bool(panel.log_scale),
                    }
                )
    return pd.DataFrame(rows)


def ifr_timeseries_histogram_dataframe(panels: Iterable[IFRTimeSeriesPanel]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel in panels:
        for value_idx, value in enumerate(panel.histogram_values):
            rows.append(
                {
                    "recording_index": panel.recording_index,
                    "value_index": value_idx,
                    "ifr_value": float(value),
                    "log_scale": bool(panel.log_scale),
                }
            )
    return pd.DataFrame(rows)


def ifr_selected_electrodes_dataframe(panels: Iterable[IFRTimeSeriesPanel]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel in panels:
        for rank, electrode in enumerate(panel.electrodes):
            rows.append(
                {
                    "recording_index": panel.recording_index,
                    "electrode_rank": rank,
                    "electrode": int(electrode),
                    "start_time": panel.start_time,
                    "end_time": panel.end_time,
                    "log_scale": bool(panel.log_scale),
                }
            )
    return pd.DataFrame(rows)


def save_ifr_timeseries_checkpoints(
    panels: Iterable[IFRTimeSeriesPanel],
    output_dir: str | Path | RunPaths,
    prefix: str = "ifr_timeseries",
) -> dict[str, Path]:
    """Write IFR time-series panel checkpoints as CSV tables."""
    panels = list(panels)
    figure_data = output_dir.figure_data if isinstance(output_dir, RunPaths) else Path(output_dir)
    return {
        "heatmap": save_csv(ifr_timeseries_heatmap_dataframe(panels), figure_data / f"{prefix}_heatmap.csv"),
        "histogram": save_csv(ifr_timeseries_histogram_dataframe(panels), figure_data / f"{prefix}_histogram.csv"),
        "electrodes": save_csv(ifr_selected_electrodes_dataframe(panels), figure_data / f"{prefix}_selected_electrodes.csv"),
    }


def load_ifr_timeseries_panels(
    figure_data_dir: str | Path,
    prefix: str = "ifr_timeseries",
) -> list[IFRTimeSeriesPanel]:
    """Reconstruct IFR time-series panels from CSV checkpoints."""
    figure_data = Path(figure_data_dir)
    heatmap_df = pd.read_csv(figure_data / f"{prefix}_heatmap.csv")
    hist_df = pd.read_csv(figure_data / f"{prefix}_histogram.csv")
    electrode_df = pd.read_csv(figure_data / f"{prefix}_selected_electrodes.csv")

    panels: list[IFRTimeSeriesPanel] = []
    for recording_index, heatmap_group in heatmap_df.groupby("recording_index", sort=True):
        heatmap_group = heatmap_group.sort_values(["electrode_rank", "time_index"])
        electrode_group = electrode_df[electrode_df["recording_index"] == recording_index].sort_values("electrode_rank")
        hist_group = hist_df[hist_df["recording_index"] == recording_index].sort_values("value_index")

        time_points = (
            heatmap_group[["time_index", "time"]]
            .drop_duplicates()
            .sort_values("time_index")["time"]
            .to_numpy(dtype=np.float32)
        )
        n_rows = int(heatmap_group["electrode_rank"].max()) + 1
        n_cols = len(time_points)
        heatmap = np.empty((n_rows, n_cols), dtype=np.float32)
        for rank, rank_group in heatmap_group.groupby("electrode_rank", sort=True):
            heatmap[int(rank), :] = rank_group.sort_values("time_index")["ifr_value"].to_numpy(dtype=np.float32)

        panels.append(
            IFRTimeSeriesPanel(
                recording_index=int(recording_index),
                start_time=float(heatmap_group["start_time"].iloc[0]),
                end_time=float(heatmap_group["end_time"].iloc[0]),
                electrodes=electrode_group["electrode"].to_numpy(dtype=int),
                time_points=time_points,
                heatmap=heatmap,
                histogram_values=hist_group["ifr_value"].to_numpy(dtype=float),
                log_scale=bool(heatmap_group["log_scale"].iloc[0]),
            )
        )
    return panels


def save_burst_checkpoints(
    *,
    paths: RunPaths,
    coarse_epochs: pd.DataFrame,
    nested_anchors: pd.DataFrame,
    valid_anchors: pd.DataFrame,
    electrode_summary: pd.DataFrame,
    prefix: str = "burst",
) -> dict[str, Path]:
    """Write lightweight burst-analysis checkpoint tables."""
    return {
        "coarse_epochs": save_csv(coarse_epochs, paths.figure_data / f"{prefix}_coarse_epochs.csv"),
        "nested_gamma_anchors": save_csv(nested_anchors, paths.figure_data / f"{prefix}_nested_gamma_anchors.csv"),
        "valid_gamma_anchors": save_csv(valid_anchors, paths.figure_data / f"{prefix}_valid_gamma_anchors.csv"),
        "aligned_electrode_summary": save_csv(electrode_summary, paths.figure_data / f"{prefix}_aligned_electrode_summary.csv"),
    }


def save_wave_checkpoints(result: WaveAnalysisResult, paths: RunPaths, prefix: str = "wave") -> dict[str, Path]:
    """Write wave-analysis result tables as CSV checkpoints."""
    return {
        "event_direction": save_csv(result.event_direction, paths.figure_data / f"{prefix}_event_direction.csv"),
        "peaks": save_csv(result.peaks, paths.figure_data / f"{prefix}_peaks.csv"),
        "trace": save_csv(result.trace, paths.figure_data / f"{prefix}_trace.csv"),
        "bin_summary": save_csv(result.bin_summary, paths.figure_data / f"{prefix}_bin_summary.csv"),
        "fit_summary": save_csv(result.fit_summary, paths.figure_data / f"{prefix}_fit_summary.csv"),
        "heatmap": save_csv(pd.DataFrame(result.heatmap), paths.figure_data / f"{prefix}_heatmap.csv", index=True),
        "bootstrap_speeds": save_csv(
            pd.DataFrame({"speed_um_per_ms": result.bootstrap_speeds}),
            paths.figure_data / f"{prefix}_bootstrap_speeds.csv",
        ),
    }


def save_te_checkpoints(result: DiscreteTEResult, paths: RunPaths, prefix: str = "te") -> dict[str, Path]:
    """Write discrete transfer-entropy result tables as CSV checkpoints."""
    return {
        "trigger_summary": save_csv(result.trigger_summary, paths.figure_data / f"{prefix}_trigger_summary.csv"),
        "observation_summary": save_csv(result.observation_summary, paths.figure_data / f"{prefix}_observation_summary.csv"),
        "ridge_summary": save_csv(result.ridge_summary, paths.figure_data / f"{prefix}_ridge_summary.csv"),
        "fit_summary": save_csv(result.fit_summary, paths.figure_data / f"{prefix}_fit_summary.csv"),
        "surfaces": save_csv(_te_surfaces_dataframe(result), paths.figure_data / f"{prefix}_surfaces.csv"),
    }


def _te_surfaces_dataframe(result: DiscreteTEResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for delay_idx, delay_ms in enumerate(result.delay_centers_ms):
        for dx_idx, signed_dx_um in enumerate(result.signed_dx_centers_um):
            rows.append(
                {
                    "delay_ms": float(delay_ms),
                    "signed_dx_um": float(signed_dx_um),
                    "conditional_probability": float(result.conditional_probability[delay_idx, dx_idx])
                    if np.isfinite(result.conditional_probability[delay_idx, dx_idx])
                    else np.nan,
                    "raw_te_bits": float(result.raw_te_bits[delay_idx, dx_idx])
                    if np.isfinite(result.raw_te_bits[delay_idx, dx_idx])
                    else np.nan,
                    "bias_corrected_te_bits": float(result.bias_corrected_te_bits[delay_idx, dx_idx])
                    if np.isfinite(result.bias_corrected_te_bits[delay_idx, dx_idx])
                    else np.nan,
                    "te_pvalue": float(result.te_pvalue[delay_idx, dx_idx])
                    if np.isfinite(result.te_pvalue[delay_idx, dx_idx])
                    else np.nan,
                    "effective_observations": float(result.effective_observations[delay_idx, dx_idx]),
                }
            )
    return pd.DataFrame(rows)
