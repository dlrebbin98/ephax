from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .artifacts import (
    RunPaths,
    ensure_run_dirs,
    save_burst_checkpoints,
    save_csv,
    save_ifr_timeseries_checkpoints,
    save_te_checkpoints,
    save_wave_checkpoints,
    write_manifest,
)
from .analyzers.ifr import IFRAnalyzer, IFRConfig
from .metrics.burst import (
    align_highres_to_anchors,
    build_highres_traces,
    build_network_activity_state,
    build_population_ifr,
    detect_coarse_burst_epochs,
    detect_nested_gamma_anchors,
    detect_network_burst_epochs,
    summarize_aligned_electrode_rates,
)
from .metrics.transfer_entropy import build_trigger_summary, run_discrete_te
from .metrics.waves import analyze_eventwise_waves
from .prep import PrepConfig, RestingActivityDataset


@dataclass(frozen=True)
class RunOptions:
    seed: int = 0
    n_jobs: int = 1
    output_dir: Path = Path("outputs")
    data_root: Path | None = None


def load_workflow_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Workflow config must be a mapping.")
    return data


def _file_info_entries(dataset_cfg: dict[str, Any]) -> list[tuple]:
    entries = dataset_cfg.get("file_info", [])
    if not isinstance(entries, list):
        raise ValueError("dataset.file_info must be a list.")
    normalized = []
    source = dataset_cfg.get("source", "h5")
    for entry in entries:
        if not isinstance(entry, dict):
            normalized.append(tuple(entry))
            continue
        if source == "h5":
            normalized.append(
                (
                    entry.get("folder", ""),
                    entry["filename"],
                    entry["start_time"],
                    entry["end_time"],
                    entry["well"],
                )
            )
        elif source == "npz":
            normalized.append(
                (
                    entry.get("path", entry.get("folder_or_div")),
                    entry["start_time"],
                    entry["end_time"],
                    entry.get("well", 0),
                )
            )
        else:
            raise ValueError("dataset.source must be 'h5' or 'npz'.")
    return normalized


def _source_path_from_entry(entry: Any, source: str) -> str:
    if isinstance(entry, dict):
        if source == "h5":
            folder = entry.get("folder", "")
            filename = entry.get("filename", "")
            return str(Path(str(folder)) / str(filename)) if folder else str(filename)
        return str(entry.get("path", entry.get("folder_or_div", "")))
    if source == "h5" and len(entry) >= 2:
        return str(Path(str(entry[0])) / str(entry[1]))
    return str(entry[0]) if entry else ""


def _recording_metadata_entries(dataset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    entries = dataset_cfg.get("file_info", [])
    source = dataset_cfg.get("source", "h5")
    metadata: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if isinstance(entry, dict):
            start_time = entry.get("start_time")
            end_time = entry.get("end_time")
            well = entry.get("well", 0)
            recording_id = entry.get("recording_id", f"recording_{idx}")
            culture_id = entry.get("culture_id", recording_id)
            div = entry.get("div")
        else:
            if source == "h5":
                _, _, start_time, end_time, well = entry
            else:
                _, start_time, end_time, well = entry
            recording_id = f"recording_{idx}"
            culture_id = recording_id
            div = None
        metadata.append(
            {
                "recording_id": recording_id,
                "culture_id": culture_id,
                "div": div,
                "well": well,
                "source_path": _source_path_from_entry(entry, source),
                "start_time": start_time,
                "end_time": end_time,
            }
        )
    return metadata


def build_dataset(config: dict[str, Any]) -> RestingActivityDataset:
    dataset_cfg = config.get("dataset", {})
    source = dataset_cfg.get("source", "h5")
    data_root = dataset_cfg.get("data_root")
    return RestingActivityDataset.from_file_info(
        _file_info_entries(dataset_cfg),
        source=source,
        min_amp=dataset_cfg.get("min_amp", 0),
        base_dir=data_root,
    )


def build_prep_config(config: dict[str, Any]) -> PrepConfig:
    selection = dict(config.get("selection", {}))
    allowed = PrepConfig.__dataclass_fields__.keys()
    return PrepConfig(**{key: value for key, value in selection.items() if key in allowed})


def recordings_table(dataset: RestingActivityDataset, config: dict[str, Any]) -> pd.DataFrame:
    metadata_entries = _recording_metadata_entries(config.get("dataset", {}))
    rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(dataset.recordings):
        meta = metadata_entries[idx] if idx < len(metadata_entries) else {}
        electrodes = np.asarray(rec.spikes.get("electrode", []))
        rows.append(
            {
                "recording_id": meta.get("recording_id", f"recording_{idx}"),
                "culture_id": meta.get("culture_id", meta.get("recording_id", f"recording_{idx}")),
                "div": meta.get("div"),
                "well": meta.get("well"),
                "source_path": meta.get("source_path", ""),
                "start_time": float(rec.start_time),
                "end_time": float(rec.end_time),
                "sf": float(rec.sf),
                "n_spikes": int(len(rec.spikes.get("time", []))),
                "n_electrodes": int(np.unique(electrodes).size) if electrodes.size else 0,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "recording_id",
            "culture_id",
            "div",
            "well",
            "source_path",
            "start_time",
            "end_time",
            "sf",
            "n_spikes",
            "n_electrodes",
        ],
    )


def selected_refs_table(recordings: pd.DataFrame, refs_per_recording: list[np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, refs in enumerate(refs_per_recording):
        if idx >= len(recordings):
            continue
        rec_row = recordings.iloc[idx]
        for electrode in refs:
            rows.append(
                {
                    "recording_id": rec_row["recording_id"],
                    "culture_id": rec_row["culture_id"],
                    "div": rec_row["div"],
                    "well": rec_row["well"],
                    "electrode": int(electrode),
                }
            )
    return pd.DataFrame(rows, columns=["recording_id", "culture_id", "div", "well", "electrode"])


def run_workflow(config: dict[str, Any]) -> dict[str, Any]:
    paths = ensure_run_dirs(RunPaths.from_config(config))

    dataset = build_dataset(config)
    prep_config = build_prep_config(config)
    refs = dataset.select_ref_electrodes(prep_config)
    recordings_df = recordings_table(dataset, config)
    selected_refs_df = selected_refs_table(recordings_df, refs)

    recordings_csv = save_csv(recordings_df, paths.figure_data / "recordings.csv")
    selected_refs_csv = save_csv(selected_refs_df, paths.figure_data / "selected_refs.csv")
    checkpoints = {
        "recordings": str(recordings_csv),
        "selected_refs": str(selected_refs_csv),
    }

    result: dict[str, Any] = {
        "dataset": dataset,
        "refs_per_recording": refs,
        "output_dir": paths.run_dir,
        "paths": paths,
        "recordings_csv": recordings_csv,
        "selected_refs_csv": selected_refs_csv,
        "recordings": recordings_df,
        "selected_refs": selected_refs_df,
    }

    ifr_cfg = config.get("analyses", {}).get("ifr", {})
    if ifr_cfg.get("enabled", False):
        kwargs = {key: value for key, value in ifr_cfg.items() if key in IFRConfig.__dataclass_fields__}
        analyzer = IFRAnalyzer.from_dataset(dataset, config=IFRConfig(**kwargs), selection_prep_config=prep_config)
        result["ifr_analyzer"] = analyzer
        if ifr_cfg.get("write_timeseries_checkpoints", False):
            ifr_checkpoints = save_ifr_timeseries_checkpoints(analyzer.timeseries_panels(), paths)
            checkpoints.update({f"ifr_timeseries_{key}": str(value) for key, value in ifr_checkpoints.items()})
            result["ifr_timeseries_checkpoints"] = ifr_checkpoints

    burst_cfg = config.get("analyses", {}).get("burst", {})
    if burst_cfg.get("enabled", False):
        if len(dataset.recordings) != 1:
            raise ValueError("The current burst workflow supports exactly one recording.")
        rec = dataset.recordings[0]
        selected = refs[0]
        population = build_population_ifr(
            rec,
            selected,
            grid_hz=burst_cfg.get("ifr_grid_hz", 50.0),
            smooth_sigma_sec=burst_cfg.get("smooth_sigma_sec", 0.15),
        )
        highres = build_highres_traces(
            rec,
            selected,
            bin_ms=burst_cfg.get("highres_bin_ms", 1.0),
            smooth_sigma_ms=burst_cfg.get("highres_smooth_sigma_ms", 3.0),
        )
        network_activity = None
        raw_peak_idx = np.array([], dtype=int)
        burst_detector = burst_cfg.get("detector", "rate_peak")
        if burst_detector == "network_participation":
            network_activity = build_network_activity_state(
                highres,
                aggregation_ms=burst_cfg.get("network_bin_ms", 10.0),
                active_rate_floor_hz=burst_cfg.get("network_active_rate_floor_hz", 1.0),
                threshold_baseline_quantile=burst_cfg.get("network_threshold_baseline_quantile", 0.20),
                threshold_iqr_scale=burst_cfg.get("network_threshold_iqr_scale", 3.0),
            )
            coarse_epochs = detect_network_burst_epochs(
                network_activity,
                min_participation_fraction=burst_cfg.get("network_min_participation_fraction", 0.20),
                min_active_electrodes=burst_cfg.get("network_min_active_electrodes", 10),
                merge_gap_ms=burst_cfg.get("network_merge_gap_ms", 50.0),
                min_duration_ms=burst_cfg.get("network_min_duration_ms", 30.0),
                min_spikes=burst_cfg.get("network_min_spikes", 20),
            )
        elif burst_detector == "rate_peak":
            coarse_epochs, raw_peak_idx, _raw_peak_props = detect_coarse_burst_epochs(
                population.time_grid,
                population.mean_ifr_smooth,
                grid_hz=burst_cfg.get("ifr_grid_hz", 50.0),
                peak_distance_sec=burst_cfg.get("burst_min_distance_sec", 1.0),
                prominence_quantile=burst_cfg.get("burst_prominence_quantile", 0.90),
                prominence_scale=burst_cfg.get("burst_prominence_scale", 0.20),
                rel_height=burst_cfg.get("coarse_epoch_rel_height", 0.20),
            )
        else:
            raise ValueError("analyses.burst.detector must be 'rate_peak' or 'network_participation'.")
        nested_anchors = detect_nested_gamma_anchors(
            coarse_epochs,
            highres,
            coarse_rel_height=burst_cfg.get("coarse_epoch_rel_height", 0.20),
            bin_ms=burst_cfg.get("highres_bin_ms", 1.0),
            search_ms=burst_cfg.get("gamma_search_ms", 120.0),
            search_to_epoch_end=burst_cfg.get("gamma_search_to_epoch_end", True),
            min_distance_ms=burst_cfg.get("gamma_min_distance_ms", 40.0),
            prominence_frac=burst_cfg.get("gamma_prominence_frac", 0.08),
            prominence_abs_floor=burst_cfg.get("gamma_prominence_abs_floor", 0.10),
            keep_height_frac=burst_cfg.get("gamma_keep_height_frac", 0.50),
        )
        aligned = align_highres_to_anchors(
            highres,
            nested_anchors,
            pre_ms=burst_cfg.get("align_pre_ms", 20.0),
            post_ms=burst_cfg.get("align_post_ms", 40.0),
            bin_ms=burst_cfg.get("highres_bin_ms", 1.0),
        )
        electrode_summary = summarize_aligned_electrode_rates(
            aligned,
            peak_pre_ms=burst_cfg.get("sort_peak_pre_ms", 5.0),
            peak_post_ms=burst_cfg.get("sort_peak_post_ms", 10.0),
        )
        burst_checkpoints = save_burst_checkpoints(
            paths=paths,
            coarse_epochs=coarse_epochs,
            nested_anchors=nested_anchors,
            valid_anchors=aligned.valid_anchors,
            electrode_summary=electrode_summary,
        )
        checkpoints.update({f"burst_{key}": str(value) for key, value in burst_checkpoints.items()})
        result.update(
            {
                "burst_population": population,
                "burst_highres": highres,
                "burst_network_activity": network_activity,
                "burst_detector": burst_detector,
                "burst_coarse_epochs": coarse_epochs,
                "burst_raw_peak_idx": raw_peak_idx,
                "burst_nested_anchors": nested_anchors,
                "burst_aligned": aligned,
                "burst_electrode_summary": electrode_summary,
                "burst_checkpoints": burst_checkpoints,
            }
        )

        wave_cfg = config.get("analyses", {}).get("wave", {})
        if wave_cfg.get("enabled", False):
            wave_result = analyze_eventwise_waves(
                aligned,
                rec.layout,
                x_bin_um=wave_cfg.get("x_bin_um", 300.0),
                peak_search_start_ms=wave_cfg.get("peak_search_start_ms", -15.0),
                peak_search_stop_ms=wave_cfg.get("peak_search_stop_ms", 20.0),
                trace_smooth_sigma_ms=wave_cfg.get("trace_smooth_sigma_ms", 2.0),
                bin_ms=burst_cfg.get("highres_bin_ms", 1.0),
                min_electrodes_per_bin=wave_cfg.get("min_electrodes_per_bin", 5),
                min_events_per_bin=wave_cfg.get("min_events_per_bin", 5),
                bootstrap_reps=wave_cfg.get("bootstrap_reps", 2000),
                random_seed=wave_cfg.get("random_seed", config.get("run", {}).get("seed", 0)),
            )
            wave_checkpoints = save_wave_checkpoints(wave_result, paths)
            checkpoints.update({f"wave_{key}": str(value) for key, value in wave_checkpoints.items()})
            result["wave_result"] = wave_result
            result["wave_checkpoints"] = wave_checkpoints

        te_cfg = config.get("analyses", {}).get("transfer_entropy", {})
        if te_cfg.get("enabled", False):
            te_scope = te_cfg.get("scope", "gamma")
            trigger_summary = build_trigger_summary(
                scope=te_scope,
                highres=highres,
                selected_electrodes=selected,
                aligned=aligned,
                coarse_epochs=coarse_epochs,
            )
            te_result = run_discrete_te(
                highres=highres,
                layout=rec.layout,
                selected_electrodes=selected,
                trigger_summary=trigger_summary,
                target_history_blocks_ms=te_cfg.get("target_history_blocks_ms", [(1.0, 2.0), (3.0, 5.0), (6.0, 10.0)]),
                source_history_blocks_ms=te_cfg.get("source_history_blocks_ms", [(1.0, 5.0)]),
                temporal_bin_ms=te_cfg.get("temporal_bin_ms", burst_cfg.get("highres_bin_ms", 1.0)),
                signed_dx_bin_um=te_cfg.get("signed_dx_bin_um", 200.0),
                delay_start_ms=te_cfg.get("delay_start_ms", 0.0),
                delay_stop_ms=te_cfg.get("delay_stop_ms", 40.0),
                control_exclusion_ms=te_cfg.get("control_exclusion_ms", 3.0),
                controls_per_trigger=te_cfg.get("controls_per_trigger", 1),
                n_surrogates=te_cfg.get("n_surrogates", 100),
                min_observations=te_cfg.get("min_observations", 800),
                min_effect_bits=te_cfg.get("min_effect_bits", 0.001),
                alpha=te_cfg.get("alpha", 0.05),
                delay_smooth_sigma_bins=te_cfg.get("delay_smooth_sigma_bins", 1.0),
                local_delay_tolerance_bins=te_cfg.get("local_delay_tolerance_bins", 2),
                bootstrap_reps=te_cfg.get("bootstrap_reps", 1000),
                random_seed=te_cfg.get("random_seed", config.get("run", {}).get("seed", 0)),
                max_triggers=te_cfg.get("max_triggers"),
            )
            te_checkpoints = save_te_checkpoints(te_result, paths)
            checkpoints.update({f"transfer_entropy_{key}": str(value) for key, value in te_checkpoints.items()})
            result["transfer_entropy_result"] = te_result
            result["transfer_entropy_checkpoints"] = te_checkpoints

    manifest = write_manifest(
        paths,
        config,
        metadata={
            "n_recordings": len(dataset.recordings),
            "checkpoints": checkpoints,
        },
    )
    result["manifest"] = manifest

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an ephax YAML workflow.")
    parser.add_argument("config", help="Path to workflow YAML config.")
    args = parser.parse_args(argv)
    result = run_workflow(load_workflow_config(args.config))
    print(f"Loaded {len(result['dataset'].recordings)} recording(s).")
    print(f"Output directory: {result['output_dir']}")
    print(f"Manifest: {result['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
