from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Union

import h5py
import numpy as np


SPIKES_KEYS = ("time", "channel", "amplitude", "electrode")
LAYOUT_KEYS = ("channel", "electrode", "x", "y")


def validate_spikes_data(spikes_data: dict) -> dict:
    """Validate and normalize a legacy spikes dict."""
    missing = [key for key in SPIKES_KEYS if key not in spikes_data]
    if missing:
        raise KeyError(f"spikes_data missing required keys: {missing}")
    normalized = {key: np.asarray(spikes_data[key]) for key in SPIKES_KEYS}
    lengths = {key: len(value) for key, value in normalized.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"spikes_data arrays must have the same length: {lengths}")
    return normalized


def validate_layout(layout: dict) -> dict:
    """Validate and normalize a legacy layout dict."""
    missing = [key for key in LAYOUT_KEYS if key not in layout]
    if missing:
        raise KeyError(f"layout missing required keys: {missing}")
    normalized = {key: np.asarray(layout[key]) for key in LAYOUT_KEYS}
    lengths = {key: len(value) for key, value in normalized.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"layout arrays must have the same length: {lengths}")
    return normalized


def data_roots(base_dir: Optional[Union[str, Path]] = None) -> list[Path]:
    """Return explicit and environment data roots, in precedence order."""
    roots: list[Path] = []
    if base_dir is not None:
        roots.append(Path(base_dir).expanduser())
    env_root = os.environ.get("EPHAX_DATA_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    return roots


def resolve_path(path_like: str | Path, base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve a file or folder path without machine-specific fallbacks."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent
    for root in [*data_roots(base_dir), Path.cwd(), project_root, package_dir]:
        candidate = root / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def load_spikes(filename: str | Path, well_no: int, min_amp: float = 10):
    """Load compressed per-well spike data from the repo's processed H5 format."""
    with h5py.File(filename, "r") as f:
        well_key = f"well{well_no:0>3}"
        if well_key not in f["wells"]:
            raise ValueError(f"Well {well_no} not found in the file.")

        well_group = f["wells"][well_key]

        spikes_data = {key: well_group["spikes"][key][:] for key in well_group["spikes"].keys()}
        amplitudes = np.abs(spikes_data["amplitude"])
        mask = amplitudes >= min_amp
        for field in list(spikes_data.keys()):
            spikes_data[field] = spikes_data[field][mask]

        event_data = {key: well_group["events"][key][:] for key in well_group["events"].keys()}
        layout = {key: well_group["layout"][key][:] for key in well_group["layout"].keys()}
        sf = well_group["sf"][()]
        stimulus_electrode = f["stimulus_electrode"][()]

    return validate_spikes_data(spikes_data), event_data, validate_layout(layout), sf, stimulus_electrode


def load_spikes_data(file_info: Iterable[tuple], min_amp: float = 0, base_dir: Optional[Union[str, Path]] = None):
    """Load processed H5 recordings from tuples of (folder, filename, start, end, well)."""
    spikes_data_list = []
    layout_list = []
    start_times = []
    end_times = []
    sf = None

    for folder, filename, start_time, end_time, well in file_info:
        folder_path = resolve_path(folder, base_dir=base_dir)
        h5_path = folder_path / filename if folder_path.is_dir() else folder_path
        if h5_path.is_dir():
            h5_path = h5_path / filename
        if not h5_path.exists():
            raise FileNotFoundError(f"Couldn't locate recording file: {h5_path}")
        spikes_data, _, layout, sf, _ = load_spikes(h5_path, well, min_amp=min_amp)
        spikes_data_list.append(spikes_data)
        layout_list.append(layout)
        start_times.append(start_time)
        end_times.append(end_time)

    if sf is None:
        raise ValueError("No recordings loaded; check file_info entries.")

    return sf, spikes_data_list, layout_list, start_times, end_times


def load_spikes_npz(file_info: Iterable[tuple], min_amp: float = 0, base_dir: Optional[Union[str, Path]] = None):
    """Load NPZ recordings from tuples of (path_or_div, start, end, well)."""
    spikes_data_list = []
    layout_list = []
    start_times = []
    end_times = []
    sf = None

    for folder_or_div, start_time, end_time, well in file_info:
        entry = Path(str(folder_or_div))
        if entry.suffix == ".npz" or entry.exists():
            npz_path = resolve_path(entry, base_dir=base_dir)
        else:
            filename = f"DIV{folder_or_div}_stim_removal_well_{well}_exp_data.npz"
            npz_path = resolve_path(filename, base_dir=base_dir)
        if not npz_path.exists():
            raise FileNotFoundError(f"Couldn't locate npz file: {npz_path}")

        data = np.load(npz_path)
        sf = data["samp_rate"]
        layout = {
            "channel": data["channelmap"][:, 0],
            "electrode": data["channelmap"][:, 2],
            "x": data["channelmap"][:, 3],
            "y": data["channelmap"][:, 4],
        }
        layout = validate_layout(layout)

        channel_to_electrode = {ch: el for ch, el in zip(layout["channel"], layout["electrode"])}
        times = data["spike_data"]["frameno"] / sf
        channels = data["spike_data"]["channel"]
        amplitudes = np.abs(data["spike_data"]["amplitude"])
        electrodes = [channel_to_electrode.get(ch, None) for ch in channels]

        amp_mask = amplitudes >= min_amp
        valid_elec_mask = np.array([e is not None for e in electrodes])
        mask = amp_mask & valid_elec_mask

        spikes_data = validate_spikes_data(
            {
                "time": np.asarray(times)[mask],
                "channel": np.asarray(channels)[mask],
                "amplitude": np.asarray(amplitudes)[mask],
                "electrode": np.asarray([int(e) for e in np.asarray(electrodes)[mask]]),
            }
        )

        spikes_data_list.append(spikes_data)
        layout_list.append(layout)
        start_times.append(start_time)
        end_times.append(end_time)

    if sf is None:
        raise ValueError("No npz recordings loaded; check file_info entries.")

    return sf, spikes_data_list, layout_list, start_times, end_times
