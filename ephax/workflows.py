from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .analyzers.ifr import IFRAnalyzer, IFRConfig
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


def run_workflow(config: dict[str, Any]) -> dict[str, Any]:
    run_cfg = config.get("run", {})
    output_dir = Path(run_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(config)
    prep_config = build_prep_config(config)
    refs = dataset.select_ref_electrodes(prep_config)

    result: dict[str, Any] = {
        "dataset": dataset,
        "refs_per_recording": refs,
        "output_dir": output_dir,
    }

    ifr_cfg = config.get("analyses", {}).get("ifr", {})
    if ifr_cfg.get("enabled", False):
        kwargs = {key: value for key, value in ifr_cfg.items() if key in IFRConfig.__dataclass_fields__}
        analyzer = IFRAnalyzer.from_dataset(dataset, config=IFRConfig(**kwargs), selection_prep_config=prep_config)
        result["ifr_analyzer"] = analyzer

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an ephax YAML workflow.")
    parser.add_argument("config", help="Path to workflow YAML config.")
    args = parser.parse_args(argv)
    result = run_workflow(load_workflow_config(args.config))
    print(f"Loaded {len(result['dataset'].recordings)} recording(s).")
    print(f"Output directory: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
