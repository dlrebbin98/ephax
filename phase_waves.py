from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.io import savemat
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt, welch
from scipy.spatial import cKDTree


RESULT_COLUMNS = (
    "t_center_s",
    "kx",
    "ky",
    "k_norm",
    "lambda_mm",
    "direction_x",
    "direction_y",
    "direction_rad",
    "aperture_along_k_mm",
    "phase_span_rad",
    "cycles_across_array",
    "spatial_class",
    "propagation_valid",
    "omega",
    "f_peak",
    "v_phi_mm_per_s",
    "phase_speed_mm_per_s",
    "phase_speed_peak_mm_per_s",
    "velocity_x_mm_per_s",
    "velocity_y_mm_per_s",
    "velocity_direction_rad",
    "planar_at_lambda_min",
    "planar_at_lambda_max",
    "planar_speed_censored",
    "planar_velocity_valid",
    "radial_x0_mm",
    "radial_y0_mm",
    "radial_k",
    "radial_lambda_mm",
    "radial_sign",
    "radial_aperture_mm",
    "radial_phase_span_rad",
    "radial_cycles_across_array",
    "R_radial",
    "delta_R_radial_minus_planar",
    "radial_phase_speed_mm_per_s",
    "radial_at_lambda_min",
    "radial_at_lambda_max",
    "radial_speed_censored",
    "radial_velocity_valid",
    "wave_model_class",
    "radial_valid",
    "R_fit",
    "mean_weight",
    "n_good",
    "valid",
    "interval_idx",
)

PGD_COLUMNS = (
    "t_center_s",
    "pgd",
    "pgd_r2_adj",
    "pgd_gradient_alignment",
    "bx",
    "by",
    "k_norm_pgd",
    "direction_pgd_rad",
    "speed_pgd_mm_per_s",
    "omega_pgd",
    "frequency_hz",
    "pgd_valid",
    "pgd_sustained",
    "pgd_direction_stable",
    "pgd_cycles_across_array",
    "pgd_aperture_along_k_mm",
    "interval_idx",
    "n_good",
)

SPATIAL_CLASS = {
    "invalid": 0,
    "near_sync": 1,
    "weak_gradient": 2,
    "resolvable_wave": 3,
}

WAVE_MODEL_CLASS = {
    "invalid": 0,
    "near_sync": 1,
    "planar_like": 2,
    "radial_like": 3,
    "ambiguous": 4,
}


def _as_posix_h5_path(path: str) -> str:
    path = str(path)
    return "/" + path.strip("/")


def _dataset_alias_from_well(well: int | str) -> str:
    text = str(well).strip().lower()
    if text.startswith("well"):
        text = text[4:]
    return f"data{int(text):04d}"


def _resolve_dataset_paths(dataset: str, file_path: str | Path | None = None) -> tuple[str, str]:
    """Return ``(raw_path, recording_group_path)`` for a MaxTwo dataset spec."""
    dataset = str(dataset).strip().strip("/")
    if dataset.isdigit() or dataset.lower().startswith("well"):
        dataset = _dataset_alias_from_well(dataset)

    if dataset.startswith("data") and "/" not in dataset:
        rec_group = f"/data_store/{dataset}"
        if file_path is not None:
            with h5py.File(file_path, "r") as h5:
                if rec_group not in h5:
                    raise KeyError(f"Recording group {rec_group!r} not found in {file_path}")
                groups = h5[rec_group].get("groups")
                if groups is None:
                    raise KeyError(f"Recording group {rec_group!r} has no 'groups' subgroup")
                for group_name in ("all_channels", "routed"):
                    candidate = f"{rec_group}/groups/{group_name}/raw"
                    if candidate in h5:
                        return candidate, rec_group
                for group_name, group in groups.items():
                    if isinstance(group, h5py.Group) and "raw" in group:
                        return f"{rec_group}/groups/{group_name}/raw", rec_group
                raise KeyError(f"No raw dataset found under {rec_group}/groups")
        return f"{rec_group}/groups/all_channels/raw", rec_group

    raw_path = _as_posix_h5_path(dataset)
    marker = "/groups/"
    if marker not in raw_path:
        raise ValueError(
            "--dataset must be a data_store alias like 'data0000' or a raw dataset path "
            "like '/data_store/data0000/groups/all_channels/raw'."
        )
    rec_group = raw_path.split(marker, 1)[0]
    return raw_path, rec_group


def _read_mapping(settings_group: h5py.Group) -> dict[str, np.ndarray]:
    mapping = settings_group["mapping"]
    if isinstance(mapping, h5py.Dataset) and mapping.dtype.fields is not None:
        arr = mapping[:]
        return {name: np.asarray(arr[name]) for name in ("channel", "electrode", "x", "y")}
    return {name: np.asarray(mapping[name])[:] for name in ("channel", "electrode", "x", "y")}


def inspect_file(file_path: str | Path, dataset: str = "data0000") -> dict[str, object]:
    """Inspect a MaxTwo raw dataset without loading voltage data."""
    raw_path, rec_group_path = _resolve_dataset_paths(dataset, file_path)
    with h5py.File(file_path, "r") as h5:
        raw = h5[raw_path]
        rec_group = h5[rec_group_path]
        settings = rec_group["settings"]
        mapping = _read_mapping(settings)
        fs_raw = float(settings["sampling"][0]) if "sampling" in settings else np.nan
        lsb = float(settings["lsb"][0]) if "lsb" in settings else np.nan
        return {
            "raw_path": raw_path,
            "recording_group": rec_group_path,
            "raw_shape": tuple(int(v) for v in raw.shape),
            "n_mapped_channels": int(len(mapping["channel"])),
            "fs_raw": fs_raw,
            "lsb": lsb,
            "x_um_min": float(np.nanmin(mapping["x"])),
            "x_um_max": float(np.nanmax(mapping["x"])),
            "y_um_min": float(np.nanmin(mapping["y"])),
            "y_um_max": float(np.nanmax(mapping["y"])),
        }


def load_chunk(
    file_path: str | Path,
    dataset: str,
    start_frame: int,
    n_frames: int,
    *,
    channels: Iterable[int] | None = None,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, np.ndarray]]:
    """Read a bounded raw-voltage chunk as ``(time, channels)`` in volts.

    Coordinates are returned in millimeters. MaxTwo mapping coordinates are stored
    in micrometers, so this function converts them before fitting wave vectors.
    """
    raw_path, rec_group_path = _resolve_dataset_paths(dataset, file_path)
    start_frame = int(max(0, start_frame))
    n_frames = int(max(0, n_frames))

    with h5py.File(file_path, "r") as h5:
        raw = h5[raw_path]
        rec_group = h5[rec_group_path]
        settings = rec_group["settings"]
        fs_raw = float(settings["sampling"][0])
        lsb = float(settings["lsb"][0])
        mapping = _read_mapping(settings)

        total_frames = int(raw.shape[1])
        stop_frame = min(total_frames, start_frame + n_frames)
        if stop_frame <= start_frame:
            empty = np.empty((0, 0), dtype=dtype)
            return empty, np.empty((0, 2), dtype=np.float64), fs_raw, mapping

        mapped_channels_all = np.asarray(mapping["channel"], dtype=np.int64)
        raw_group = raw.parent
        if "channels" in raw_group and int(raw_group["channels"].shape[0]) == int(raw.shape[0]):
            raw_channels = np.asarray(raw_group["channels"][:], dtype=np.int64)
            channel_to_row = {int(channel): row for row, channel in enumerate(raw_channels)}
            row_indices_all = np.asarray([channel_to_row.get(int(channel), -1) for channel in mapped_channels_all], dtype=np.int64)
            available = row_indices_all >= 0
        else:
            row_indices_all = mapped_channels_all.copy()
            available = (row_indices_all >= 0) & (row_indices_all < int(raw.shape[0]))

        mapped_channels = mapped_channels_all
        if channels is not None:
            requested = np.asarray(list(channels), dtype=np.int64)
            keep = np.isin(mapped_channels, requested)
        else:
            keep = np.ones(mapped_channels.shape, dtype=bool)

        keep &= available & np.isfinite(mapping["x"]) & np.isfinite(mapping["y"])
        mapped_channels = mapped_channels[keep]
        row_indices = row_indices_all[keep]
        order = np.argsort(mapped_channels)
        read_rows = row_indices[order]

        h5_order = np.argsort(read_rows)
        inverse_h5_order = np.argsort(h5_order)
        raw_block = raw[read_rows[h5_order], start_frame:stop_frame][inverse_h5_order]
        data = np.asarray(raw_block, dtype=dtype).T
        data *= dtype(lsb)

        coords_um = np.column_stack((np.asarray(mapping["x"])[keep], np.asarray(mapping["y"])[keep]))
        coords_mm = np.asarray(coords_um[order], dtype=np.float64) / 1000.0
        sorted_mapping = {name: np.asarray(values)[keep][order] for name, values in mapping.items()}

    return data, coords_mm, fs_raw, sorted_mapping


def lowpass_filter_voltage(
    data: np.ndarray,
    fs_raw: float,
    lowpass_hz: float,
    *,
    order: int = 4,
    max_channels_per_block: int = 128,
) -> np.ndarray:
    """Apply a zero-phase low-pass filter to voltage data shaped as time x channels."""
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must have shape (time, channels)")
    if data.shape[0] == 0 or data.shape[1] == 0:
        return np.asarray(data, dtype=np.float32)
    if not (0.0 < float(lowpass_hz) < 0.5 * float(fs_raw)):
        raise ValueError("lowpass_hz must satisfy 0 < lowpass_hz < fs_raw/2")

    sos = butter(int(order), float(lowpass_hz), btype="lowpass", fs=float(fs_raw), output="sos")
    blocks: list[np.ndarray] = []
    for start in range(0, data.shape[1], int(max_channels_per_block)):
        stop = min(data.shape[1], start + int(max_channels_per_block))
        block = np.asarray(data[:, start:stop], dtype=np.float32)
        block = np.nan_to_num(block, copy=False)
        blocks.append(sosfiltfilt(sos, block, axis=0).astype(np.float32, copy=False))
    return np.concatenate(blocks, axis=1)


def export_first_high_activity_raw_mat(
    file_path: str | Path,
    dataset: str,
    high_activity_epochs,
    output_path: str | Path,
    *,
    lowpass_hz: float = 500.0,
    source_label: str = "",
    well: int | None = None,
    div: int | None = None,
    overwrite: bool = True,
    filter_order: int = 4,
    max_channels_per_block: int = 128,
) -> dict[str, object]:
    """Export the first high-activity raw LFP segment as a MATLAB-readable MAT file.

    The saved voltage matrix is shaped as samples x channels and contains the
    zero-phase low-pass filtered raw voltage.
    """
    output_path = Path(output_path)
    if output_path.exists() and not bool(overwrite):
        return {"path": str(output_path), "written": False, "reason": "exists"}
    if high_activity_epochs is None or len(high_activity_epochs) == 0:
        raise ValueError("No high-activity epochs detected; cannot export first high-activity raw LFP segment.")

    first_high_activity = high_activity_epochs.sort_values("start_time_s").iloc[0]
    export_start_s = float(first_high_activity["start_time_s"])
    export_stop_s = float(first_high_activity["end_time_s"])
    info = inspect_file(file_path, dataset)
    fs_raw = float(info["fs_raw"])
    export_start_frame = int(round(export_start_s * fs_raw))
    export_n_frames = max(1, int(round((export_stop_s - export_start_s) * fs_raw)))
    raw_export, coords_export_mm, fs_export, mapping_export = load_chunk(
        file_path,
        dataset,
        export_start_frame,
        export_n_frames,
    )
    filtered_export = lowpass_filter_voltage(
        raw_export,
        fs_export,
        lowpass_hz,
        order=filter_order,
        max_channels_per_block=max_channels_per_block,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mat_payload = {
        "voltage_lowpass": filtered_export,
        "t_s": export_start_s + np.arange(filtered_export.shape[0], dtype=np.float64) / float(fs_export),
        "coords_mm": coords_export_mm,
        "fs_raw": float(fs_export),
        "lowpass_hz": float(lowpass_hz),
        "filter_order": int(filter_order),
        "start_time_s": export_start_s,
        "stop_time_s": export_stop_s,
        "start_frame": int(export_start_frame),
        "dataset": str(dataset),
        "source_file": str(file_path),
        "source_label": str(source_label),
        "channel": np.asarray(mapping_export["channel"]),
        "electrode": np.asarray(mapping_export["electrode"]),
        "x_um": np.asarray(mapping_export["x"]),
        "y_um": np.asarray(mapping_export["y"]),
    }
    if well is not None:
        mat_payload["well"] = int(well)
    if div is not None:
        mat_payload["DIV"] = int(div)
    savemat(output_path, mat_payload, do_compression=True)
    return {
        "path": str(output_path),
        "written": True,
        "shape": tuple(int(v) for v in filtered_export.shape),
        "fs_raw": float(fs_export),
        "lowpass_hz": float(lowpass_hz),
        "start_time_s": export_start_s,
        "stop_time_s": export_stop_s,
    }


def _resample_ratio(fs_raw: float, fs_ds: float) -> tuple[int, int]:
    ratio = Fraction(float(fs_ds) / float(fs_raw)).limit_denominator(1000)
    return int(ratio.numerator), int(ratio.denominator)


def bandpass_downsample(
    data: np.ndarray,
    fs_raw: float,
    band_low: float = 30.0,
    band_high: float = 80.0,
    fs_ds: float = 500.0,
    *,
    order: int = 4,
    max_channels_per_block: int = 128,
) -> np.ndarray:
    """Demean, SOS-bandpass, and downsample a chunk without channel loops."""
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must have shape (time, channels)")
    if data.shape[0] == 0 or data.shape[1] == 0:
        return np.empty((0, data.shape[1] if data.ndim == 2 else 0), dtype=np.float32)
    if not (0 < fs_ds <= fs_raw):
        raise ValueError("fs_ds must be positive and <= fs_raw")
    if not (0 < band_low < band_high < fs_raw / 2):
        raise ValueError("band limits must satisfy 0 < band_low < band_high < fs_raw/2")
    if not (band_high < fs_ds / 2):
        raise ValueError(
            "band_high must be below the downsampled Nyquist frequency; "
            f"got band_high={band_high:g} Hz with fs_ds={fs_ds:g} Hz"
        )

    sos = butter(order, [band_low, band_high], btype="bandpass", fs=fs_raw, output="sos")
    up, down = _resample_ratio(fs_raw, fs_ds)
    blocks: list[np.ndarray] = []
    for start in range(0, data.shape[1], int(max_channels_per_block)):
        stop = min(data.shape[1], start + int(max_channels_per_block))
        block = np.asarray(data[:, start:stop], dtype=np.float32)
        block = block - np.nanmean(block, axis=0, keepdims=True)
        block = np.nan_to_num(block, copy=False)
        filtered = sosfiltfilt(sos, block, axis=0).astype(np.float32, copy=False)
        ds = resample_poly(filtered, up, down, axis=0).astype(np.float32, copy=False)
        blocks.append(ds)
    return np.concatenate(blocks, axis=1)


def extract_phasors(data_band_ds: np.ndarray, *, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Return unit phasors ``u`` and amplitudes ``A`` from bandpassed data."""
    z = hilbert(np.asarray(data_band_ds), axis=0)
    amp = np.abs(z).astype(np.float32, copy=False)
    denom = np.where(amp > eps, amp, 1.0)
    u = (z / denom).astype(np.complex64, copy=False)
    u[amp <= eps] = 0.0
    return u, amp


def _dmd_rank(singular_values: np.ndarray, energy_fraction: float, max_rank: int) -> tuple[int, np.ndarray]:
    """Choose a truncated SVD rank from cumulative snapshot energy."""
    singular_values = np.asarray(singular_values, dtype=np.float64)
    if singular_values.ndim != 1 or singular_values.size == 0:
        raise ValueError("singular_values must be a non-empty vector")
    if not (0.0 < float(energy_fraction) <= 1.0):
        raise ValueError("energy_fraction must satisfy 0 < energy_fraction <= 1")
    if int(max_rank) < 1:
        raise ValueError("max_rank must be positive")
    energy = singular_values**2
    total = float(np.sum(energy))
    cumulative = np.cumsum(energy) / total if total > 0 else np.zeros_like(energy)
    numerical_rank = int(np.count_nonzero(singular_values > np.finfo(float).eps * singular_values[0] * singular_values.size))
    if numerical_rank == 0:
        return 0, cumulative
    requested = int(np.searchsorted(cumulative, float(energy_fraction), side="left") + 1)
    return min(requested, numerical_rank, int(max_rank)), cumulative


def compute_exact_dmd(
    samples: np.ndarray,
    dt_s: float,
    *,
    energy_fraction: float = 0.99,
    max_rank: int = 30,
) -> dict[str, np.ndarray | float | int | bool]:
    """Compute truncated exact DMD for samples shaped ``(time, channels)``.

    ``reconstruction_contribution`` is a ranking proxy based on the fitted
    initial amplitude, spatial-mode norm, and finite excerpt duration. It is
    not a physical energy estimate.
    """
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (time, channels)")
    if samples.shape[0] < 3 or samples.shape[1] < 1:
        raise ValueError("DMD requires at least three samples and one channel")
    if not np.isfinite(float(dt_s)) or float(dt_s) <= 0:
        raise ValueError("dt_s must be positive")
    clean = np.nan_to_num(samples)
    x = clean[:-1].T
    y = clean[1:].T
    u, singular_values, vh = np.linalg.svd(x, full_matrices=False)
    rank, cumulative_energy = _dmd_rank(singular_values, energy_fraction, max_rank)
    if rank == 0:
        return {
            "eigenvalues": np.empty(0, dtype=np.complex128),
            "frequency_hz": np.empty(0, dtype=np.float64),
            "growth_rate_per_s": np.empty(0, dtype=np.float64),
            "amplitudes": np.empty(0, dtype=np.complex128),
            "reconstruction_contribution": np.empty(0, dtype=np.float64),
            "contribution_fraction": np.empty(0, dtype=np.float64),
            "modes": np.empty((samples.shape[1], 0), dtype=np.complex128),
            "singular_values": singular_values,
            "cumulative_energy": cumulative_energy,
            "retained_rank": 0,
            "rank_cap_bound": False,
        }
    ur = u[:, :rank]
    sr = singular_values[:rank]
    vr = vh[:rank].conj().T
    a_tilde = (ur.conj().T @ y @ vr) / sr[None, :]
    eigenvalues, eigenvectors = np.linalg.eig(a_tilde)
    modes = ((y @ vr) / sr[None, :]) @ eigenvectors
    amplitudes = np.linalg.lstsq(modes, x[:, 0], rcond=None)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        continuous = np.log(eigenvalues) / float(dt_s)
    frequency_hz = np.imag(continuous) / (2.0 * np.pi)
    growth_rate = np.real(continuous)
    powers = np.arange(x.shape[1], dtype=np.float64)
    log_abs_eigenvalue = np.log(np.maximum(np.abs(eigenvalues), np.finfo(float).tiny))
    temporal_exponents = 2.0 * log_abs_eigenvalue[:, None] * powers[None, :]
    max_temporal_exponent = np.max(temporal_exponents, axis=1)
    log_temporal_norm = 0.5 * (
        max_temporal_exponent + np.log(np.sum(np.exp(temporal_exponents - max_temporal_exponent[:, None]), axis=1))
    )
    log_contribution = (
        np.log(np.maximum(np.abs(amplitudes), np.finfo(float).tiny))
        + np.log(np.maximum(np.linalg.norm(modes, axis=0), np.finfo(float).tiny))
        + log_temporal_norm
    )
    finite_log_contribution = np.isfinite(log_contribution)
    contribution_fraction = np.zeros_like(log_contribution)
    contribution = np.zeros_like(log_contribution)
    if np.any(finite_log_contribution):
        offset = float(np.max(log_contribution[finite_log_contribution]))
        scaled = np.exp(np.clip(log_contribution[finite_log_contribution] - offset, -700.0, 0.0))
        contribution_fraction[finite_log_contribution] = scaled / np.sum(scaled)
        contribution[finite_log_contribution] = np.exp(np.clip(log_contribution[finite_log_contribution], -700.0, 700.0))
    return {
        "eigenvalues": eigenvalues,
        "frequency_hz": frequency_hz,
        "growth_rate_per_s": growth_rate,
        "amplitudes": amplitudes,
        "reconstruction_contribution": contribution,
        "contribution_fraction": contribution_fraction,
        "modes": modes,
        "singular_values": singular_values,
        "cumulative_energy": cumulative_energy,
        "retained_rank": int(rank),
        "rank_cap_bound": bool(rank == int(max_rank) and rank < singular_values.size and cumulative_energy[rank - 1] < float(energy_fraction)),
    }


def compute_hankel_dmd(
    samples: np.ndarray,
    dt_s: float,
    *,
    n_delays: int = 20,
    energy_fraction: float = 0.99,
    max_rank: int = 30,
) -> dict[str, np.ndarray | float | int | bool]:
    """Compute delay-embedded DMD and expose the first delay block as a spatial mode."""
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (time, channels)")
    if int(n_delays) < 2 or int(n_delays) > samples.shape[0] - 2:
        raise ValueError("n_delays must satisfy 2 <= n_delays <= n_time - 2")
    n_delays = int(n_delays)
    lifted = np.concatenate([samples[offset : samples.shape[0] - n_delays + offset + 1] for offset in range(n_delays)], axis=1)
    result = compute_exact_dmd(lifted, dt_s, energy_fraction=energy_fraction, max_rank=max_rank)
    result["modes"] = np.asarray(result["modes"])[: samples.shape[1]]
    result["n_delays"] = n_delays
    result["lifted_state_dimension"] = int(samples.shape[1] * n_delays)
    return result


def centered_excerpt_bounds(
    anchor_time_s: float,
    context_s: float,
    *,
    recording_start_s: float = 0.0,
    recording_stop_s: float,
) -> tuple[float, float]:
    """Return a fixed-duration excerpt around an anchor, shifted inside recording bounds."""
    duration = float(context_s)
    start_limit = float(recording_start_s)
    stop_limit = float(recording_stop_s)
    if not (duration > 0 and stop_limit > start_limit):
        raise ValueError("Require positive context_s and recording_stop_s > recording_start_s")
    if duration > stop_limit - start_limit:
        raise ValueError("context_s exceeds the available recording duration")
    start = float(anchor_time_s) - 0.5 * duration
    start = min(max(start, start_limit), stop_limit - duration)
    return start, start + duration


def select_evenly_spaced_events(events: Iterable[dict[str, object]], max_events: int = 25) -> list[dict[str, object]]:
    """Select deterministic evenly spaced event records without changing their order."""
    events = list(events)
    if int(max_events) < 1:
        raise ValueError("max_events must be positive")
    if len(events) <= int(max_events):
        return events
    indices = np.linspace(0, len(events) - 1, int(max_events), dtype=int)
    return [events[int(index)] for index in indices]


def run_event_dmd_screen(
    events: Iterable[dict[str, object]],
    load_event_samples,
    coords_mm: np.ndarray,
    *,
    electrode_ids: np.ndarray | None = None,
    dt_s: float = 1.0 / 500.0,
    energy_fraction: float = 0.99,
    max_rank: int = 30,
    hankel_delays: int = 20,
    store_top_modes: int = 8,
) -> dict[str, object]:
    """Run exact and Hankel DMD sequentially for event-wise signal dictionaries."""
    coords_mm = np.asarray(coords_mm, dtype=np.float64)
    if coords_mm.ndim != 2 or coords_mm.shape[1] != 2:
        raise ValueError("coords_mm must have shape (n_channels, 2)")
    electrode_ids = np.arange(coords_mm.shape[0], dtype=np.int64) if electrode_ids is None else np.asarray(electrode_ids)
    if electrode_ids.shape != (coords_mm.shape[0],):
        raise ValueError("electrode_ids must match coords_mm rows")
    event_rows: list[dict[str, object]] = []
    mode_rows: list[dict[str, object]] = []
    singular_rows: list[dict[str, object]] = []
    stored_modes: list[np.ndarray] = []
    for event_idx, event in enumerate(events):
        event = dict(event)
        signals = load_event_samples(event)
        event_rows.append({"event_idx": event_idx, **event})
        for signal_path, samples in signals.items():
            samples = np.asarray(samples)
            if samples.shape[1] != coords_mm.shape[0]:
                raise ValueError("Loaded event channel count does not match coords_mm")
            for variant, result in (
                ("exact", compute_exact_dmd(samples, dt_s, energy_fraction=energy_fraction, max_rank=max_rank)),
                (
                    "hankel",
                    compute_hankel_dmd(
                        samples,
                        dt_s,
                        n_delays=hankel_delays,
                        energy_fraction=energy_fraction,
                        max_rank=max_rank,
                    ),
                ),
            ):
                singular_rows.append(
                    {
                        "event_idx": event_idx,
                        "signal_path": signal_path,
                        "variant": variant,
                        "retained_rank": int(result["retained_rank"]),
                        "rank_cap_bound": bool(result["rank_cap_bound"]),
                        "singular_values": np.asarray(result["singular_values"], dtype=np.float64),
                        "cumulative_energy": np.asarray(result["cumulative_energy"], dtype=np.float64),
                    }
                )
                contributions = np.asarray(result["contribution_fraction"], dtype=np.float64)
                top = set(np.argsort(contributions)[::-1][: int(store_top_modes)].tolist())
                modes = np.asarray(result["modes"])
                for mode_idx in range(modes.shape[1]):
                    stored_mode_idx = -1
                    if mode_idx in top:
                        stored_mode_idx = len(stored_modes)
                        stored_modes.append(np.asarray(modes[:, mode_idx], dtype=np.complex64))
                    eigenvalue = np.asarray(result["eigenvalues"])[mode_idx]
                    amplitude = np.asarray(result["amplitudes"])[mode_idx]
                    mode_rows.append(
                        {
                            "event_idx": event_idx,
                            "signal_path": signal_path,
                            "variant": variant,
                            "mode_idx": mode_idx,
                            "eigenvalue_real": float(np.real(eigenvalue)),
                            "eigenvalue_imag": float(np.imag(eigenvalue)),
                            "frequency_hz": float(np.asarray(result["frequency_hz"])[mode_idx]),
                            "growth_rate_per_s": float(np.asarray(result["growth_rate_per_s"])[mode_idx]),
                            "amplitude_abs": float(np.abs(amplitude)),
                            "reconstruction_contribution": float(np.asarray(result["reconstruction_contribution"])[mode_idx]),
                            "contribution_fraction": float(contributions[mode_idx]),
                            "retained_rank": int(result["retained_rank"]),
                            "rank_cap_bound": bool(result["rank_cap_bound"]),
                            "stored_mode_idx": stored_mode_idx,
                        }
                    )
    return {
        "coords_mm": coords_mm,
        "electrode_ids": electrode_ids,
        "events": event_rows,
        "mode_metrics": mode_rows,
        "singular_summaries": singular_rows,
        "spatial_modes": np.asarray(stored_modes, dtype=np.complex64).reshape((-1, coords_mm.shape[0])),
    }


def _write_dmd_table(group: h5py.Group, rows: list[dict[str, object]], *, skip: set[str] | None = None) -> None:
    skip = set() if skip is None else set(skip)
    if not rows:
        return
    for key in rows[0]:
        if key in skip:
            continue
        values = [row[key] for row in rows]
        if isinstance(values[0], str):
            group.create_dataset(key, data=np.asarray(values, dtype=h5py.string_dtype("utf-8")))
        else:
            group.create_dataset(key, data=np.asarray(values))


def write_dmd_cache(path: str | Path, screen: dict[str, object], *, config: dict[str, object]) -> None:
    """Write a compact standalone DMD inspection cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["config"] = json.dumps(config, sort_keys=True)
        h5.create_dataset("coords_mm", data=np.asarray(screen["coords_mm"], dtype=np.float64))
        h5.create_dataset("electrode_ids", data=np.asarray(screen["electrode_ids"]))
        modes = np.asarray(screen["spatial_modes"], dtype=np.complex64)
        h5.create_dataset("spatial_modes_real", data=modes.real.astype(np.float32, copy=False))
        h5.create_dataset("spatial_modes_imag", data=modes.imag.astype(np.float32, copy=False))
        _write_dmd_table(h5.create_group("events"), list(screen["events"]))
        _write_dmd_table(h5.create_group("mode_metrics"), list(screen["mode_metrics"]))
        singular_group = h5.create_group("singular_summaries")
        singular_rows = list(screen["singular_summaries"])
        _write_dmd_table(singular_group, singular_rows, skip={"singular_values", "cumulative_energy"})
        if singular_rows:
            width = max(len(row["singular_values"]) for row in singular_rows)
            for key in ("singular_values", "cumulative_energy"):
                values = np.full((len(singular_rows), width), np.nan, dtype=np.float64)
                for row_idx, row in enumerate(singular_rows):
                    row_values = np.asarray(row[key], dtype=np.float64)
                    values[row_idx, : row_values.size] = row_values
                singular_group.create_dataset(key, data=values)


def read_dmd_cache(path: str | Path) -> dict[str, object]:
    """Read a standalone DMD inspection cache."""
    with h5py.File(path, "r") as h5:
        result: dict[str, object] = {
            "config": json.loads(h5.attrs["config"]),
            "coords_mm": h5["coords_mm"][:],
            "electrode_ids": h5["electrode_ids"][:],
            "spatial_modes": h5["spatial_modes_real"][:] + 1j * h5["spatial_modes_imag"][:],
        }
        for group_name in ("events", "mode_metrics", "singular_summaries"):
            group = h5[group_name]
            payload = {}
            for key, dataset in group.items():
                values = dataset[:]
                if values.dtype.kind in {"S", "O"}:
                    values = values.astype(str)
                payload[key] = values
            result[group_name] = payload
    return result


def dmd_cache_matches(path: str | Path, config: dict[str, object]) -> tuple[bool, str]:
    """Return whether a DMD cache exists and exactly matches its processing configuration."""
    path = Path(path)
    if not path.exists():
        return False, "missing"
    try:
        with h5py.File(path, "r") as h5:
            cached = json.loads(h5.attrs["config"])
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        return False, f"unreadable: {exc}"
    return (True, "compatible") if cached == config else (False, "configuration differs")


def make_k_grid(lambda_min_mm: float, lambda_max_mm: float, n_grid: int = 41) -> dict[str, np.ndarray | float]:
    """Build a square ``kx, ky`` search grid with radial wavelength bounds."""
    if not (0 < lambda_min_mm < lambda_max_mm):
        raise ValueError("Require 0 < lambda_min_mm < lambda_max_mm")
    k_min = 2.0 * np.pi / float(lambda_max_mm)
    k_max = 2.0 * np.pi / float(lambda_min_mm)
    axis = np.linspace(-k_max, k_max, int(n_grid), dtype=np.float64)
    return {"kx": axis, "ky": axis, "k_min": k_min, "k_max": k_max}


def prepare_k_grid_basis(
    k_grid: dict[str, np.ndarray | float],
    coords_mm: np.ndarray,
    *,
    max_basis_bytes: int = 256_000_000,
) -> dict[str, np.ndarray | float]:
    """Add a precomputed planar phasor basis for faster repeated window fits."""
    coords_mm = np.asarray(coords_mm, dtype=np.float64)
    if coords_mm.ndim != 2 or coords_mm.shape[1] != 2:
        raise ValueError("coords_mm must have shape (n_channels, 2)")
    result: dict[str, np.ndarray | float] = dict(k_grid)
    kx_values = np.asarray(k_grid["kx"], dtype=np.float64)
    ky_values = np.asarray(k_grid["ky"], dtype=np.float64)
    k_min = float(k_grid.get("k_min", 0.0))
    k_max = float(k_grid.get("k_max", np.inf))
    kx_mesh, ky_mesh = np.meshgrid(kx_values, ky_values, indexing="ij")
    kx_flat = kx_mesh.ravel()
    ky_flat = ky_mesh.ravel()
    k_norm = np.hypot(kx_flat, ky_flat)
    valid_k = (k_norm >= k_min) & (k_norm <= k_max)
    kx_flat = kx_flat[valid_k]
    ky_flat = ky_flat[valid_k]
    result["kx_flat"] = kx_flat
    result["ky_flat"] = ky_flat
    basis_bytes = int(kx_flat.size) * int(coords_mm.shape[0]) * np.dtype(np.complex64).itemsize
    if kx_flat.size and basis_bytes <= int(max_basis_bytes):
        safe_coords = np.nan_to_num(coords_mm, nan=0.0)
        phase = kx_flat[:, None] * safe_coords[None, :, 0] + ky_flat[:, None] * safe_coords[None, :, 1]
        basis = np.exp(-1j * phase).astype(np.complex64, copy=False)
        coord_finite = np.isfinite(coords_mm).all(axis=1)
        basis[:, ~coord_finite] = 0.0
        result["basis"] = basis
    return result


def make_radial_grid(
    coords_mm: np.ndarray,
    *,
    center_grid_n: int = 15,
    center_margin_mm: float = 0.0,
    lambda_min_mm: float = 0.5,
    lambda_max_mm: float = 5.0,
    k_grid_n: int = 41,
    max_basis_bytes: int = 512_000_000,
) -> dict[str, np.ndarray | float]:
    """Build candidate centers and positive radial wavenumbers for radial waves.

    By default, candidate centers are constrained to the mapped HD-MEA footprint
    bounding box. A positive ``center_margin_mm`` explicitly allows off-array
    origins.
    """
    coords_mm = np.asarray(coords_mm, dtype=np.float64)
    if coords_mm.ndim != 2 or coords_mm.shape[1] != 2:
        raise ValueError("coords_mm must have shape (n_channels, 2)")
    if not (0 < lambda_min_mm < lambda_max_mm):
        raise ValueError("Require 0 < lambda_min_mm < lambda_max_mm")
    finite = np.isfinite(coords_mm).all(axis=1)
    if np.count_nonzero(finite) < 3:
        centers = np.empty((0, 2), dtype=np.float64)
    else:
        valid_coords = coords_mm[finite]
        margin = max(0.0, float(center_margin_mm))
        x_axis = np.linspace(float(np.nanmin(valid_coords[:, 0]) - margin), float(np.nanmax(valid_coords[:, 0]) + margin), int(center_grid_n))
        y_axis = np.linspace(float(np.nanmin(valid_coords[:, 1]) - margin), float(np.nanmax(valid_coords[:, 1]) + margin), int(center_grid_n))
        xx, yy = np.meshgrid(x_axis, y_axis)
        centers = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64, copy=False)
    k_min = 2.0 * np.pi / float(lambda_max_mm)
    k_max = 2.0 * np.pi / float(lambda_min_mm)
    k_values = np.linspace(k_min, k_max, int(k_grid_n), dtype=np.float64)
    result: dict[str, np.ndarray | float] = {
        "centers": centers,
        "k": k_values,
        "k_min": k_min,
        "k_max": k_max,
        "center_margin_mm": float(center_margin_mm),
    }
    if centers.size and k_values.size:
        coord_finite = np.isfinite(coords_mm).all(axis=1)
        safe_coords = np.nan_to_num(coords_mm, nan=0.0)
        distances = np.linalg.norm(safe_coords[None, :, :] - centers[:, None, :], axis=2).astype(np.float32, copy=False)
        distances[:, ~coord_finite] = 0.0
        result["distances"] = distances
        result["coord_finite"] = coord_finite
        basis_bytes = int(centers.shape[0]) * int(k_values.size) * int(coords_mm.shape[0]) * np.dtype(np.complex64).itemsize
        if basis_bytes <= int(max_basis_bytes):
            phase = distances[:, None, :] * k_values[None, :, None]
            result["basis"] = np.exp(-1j * phase).astype(np.complex64, copy=False)
    return result


def _objective_score(k: np.ndarray, ubar: np.ndarray, w: np.ndarray, coords: np.ndarray) -> float:
    phase = coords @ np.asarray(k, dtype=np.float64)
    return float(np.abs(np.sum(w * ubar * np.exp(-1j * phase))))


def fit_k_phasor_plane(
    ubar: np.ndarray,
    w: np.ndarray,
    coords: np.ndarray,
    k_grid: dict[str, np.ndarray | float] | tuple[np.ndarray, np.ndarray],
    *,
    refine: bool = True,
    min_weight: float = 1e-6,
) -> dict[str, float]:
    """Fit a planar phase wave by maximizing weighted circular alignment."""
    ubar_full = np.asarray(ubar, dtype=np.complex128)
    w_full = np.asarray(w, dtype=np.float64)
    coords_full = np.asarray(coords, dtype=np.float64)
    if coords_full.shape != (ubar_full.size, 2):
        raise ValueError("coords must have shape (n_channels, 2)")

    finite = np.isfinite(w_full) & (w_full > min_weight) & np.isfinite(ubar_full.real) & np.isfinite(ubar_full.imag)
    finite &= np.isfinite(coords_full).all(axis=1)
    if np.count_nonzero(finite) < 3:
        return {"kx": np.nan, "ky": np.nan, "R_fit": np.nan, "n_good": int(np.count_nonzero(finite))}

    ubar = ubar_full[finite]
    w = w_full[finite]
    coords = coords_full[finite]
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        return {"kx": np.nan, "ky": np.nan, "R_fit": np.nan, "n_good": int(w.size)}

    if isinstance(k_grid, dict):
        kx_values = np.asarray(k_grid["kx"], dtype=np.float64)
        ky_values = np.asarray(k_grid["ky"], dtype=np.float64)
        k_min = float(k_grid.get("k_min", 0.0))
        k_max = float(k_grid.get("k_max", np.inf))
    else:
        kx_values, ky_values = (np.asarray(k_grid[0], dtype=np.float64), np.asarray(k_grid[1], dtype=np.float64))
        k_min = 0.0
        k_max = np.inf

    best_score = -np.inf
    best_k = np.array([np.nan, np.nan], dtype=np.float64)
    weighted_ubar = w * ubar
    basis = k_grid.get("basis") if isinstance(k_grid, dict) else None
    if isinstance(k_grid, dict) and isinstance(basis, np.ndarray) and basis.shape[1] == ubar_full.size:
        kx_flat = np.asarray(k_grid["kx_flat"], dtype=np.float64)
        ky_flat = np.asarray(k_grid["ky_flat"], dtype=np.float64)
        weighted_full = np.zeros(ubar_full.size, dtype=np.complex128)
        weighted_full[finite] = w_full[finite] * ubar_full[finite]
        scores = np.abs(basis @ weighted_full)
        best_idx = int(np.nanargmax(scores))
        best_score = float(scores[best_idx])
        best_k[:] = (float(kx_flat[best_idx]), float(ky_flat[best_idx]))
    else:
        for kx in kx_values:
            phase_x = kx * coords[:, 0]
            for ky in ky_values:
                k_norm = float(np.hypot(kx, ky))
                if k_norm < k_min or k_norm > k_max:
                    continue
                score = float(np.abs(np.sum(weighted_ubar * np.exp(-1j * (phase_x + ky * coords[:, 1])))))
                if score > best_score:
                    best_score = score
                    best_k[:] = (kx, ky)

    if refine and np.all(np.isfinite(best_k)):
        def neg_score(k: np.ndarray) -> float:
            k_norm = float(np.hypot(k[0], k[1]))
            if k_norm < k_min or k_norm > k_max:
                return 1e9 + (min(abs(k_norm - k_min), abs(k_norm - k_max)) * 1e6)
            return -_objective_score(k, ubar, w, coords)

        result = minimize(neg_score, best_k, method="Nelder-Mead", options={"maxiter": 100, "xatol": 1e-5, "fatol": 1e-5})
        if result.success or np.isfinite(result.fun):
            candidate = np.asarray(result.x, dtype=np.float64)
            candidate_score = _objective_score(candidate, ubar, w, coords)
            if candidate_score >= best_score:
                best_k = candidate
                best_score = candidate_score

    return {
        "kx": float(best_k[0]),
        "ky": float(best_k[1]),
        "R_fit": float(best_score / w_sum),
        "n_good": int(w.size),
    }


def _circular_mean(theta: np.ndarray) -> float:
    z = np.nanmean(np.exp(1j * np.asarray(theta, dtype=np.float64)))
    return float(np.angle(z)) if np.isfinite(z.real) and np.isfinite(z.imag) and abs(z) > 0 else np.nan


def circular_linear_r2_adj(theta: np.ndarray, theta_hat: np.ndarray, *, n_params: int = 3) -> float:
    """Adjusted circular-linear rho^2 used as model-fit PGD in traveling-wave work."""
    theta = np.asarray(theta, dtype=np.float64)
    theta_hat = np.asarray(theta_hat, dtype=np.float64)
    good = np.isfinite(theta) & np.isfinite(theta_hat)
    n = int(np.count_nonzero(good))
    if n <= int(n_params) + 1:
        return np.nan
    theta = theta[good]
    theta_hat = theta_hat[good]
    theta_bar = _circular_mean(theta)
    theta_hat_bar = _circular_mean(theta_hat)
    if not np.isfinite(theta_bar) or not np.isfinite(theta_hat_bar):
        return np.nan
    s1 = np.sin(theta - theta_bar)
    s2 = np.sin(theta_hat - theta_hat_bar)
    denom = float(np.sqrt(np.sum(s1 * s1) * np.sum(s2 * s2)))
    if denom <= 0:
        return np.nan
    rho = float(np.sum(s1 * s2) / denom)
    rho2 = rho * rho
    adj = 1.0 - ((1.0 - rho2) * (n - 1.0) / (n - float(n_params) - 1.0))
    return float(np.clip(adj, 0.0, 1.0))


def make_neighbor_edges(coords_mm: np.ndarray, *, n_neighbors: int = 6, max_distance_mm: float | None = None) -> np.ndarray:
    """Return unique nearest-neighbor edges for local phase-gradient alignment."""
    coords_mm = np.asarray(coords_mm, dtype=np.float64)
    finite = np.isfinite(coords_mm).all(axis=1)
    if np.count_nonzero(finite) < 3:
        return np.empty((0, 2), dtype=np.int64)
    finite_idx = np.flatnonzero(finite)
    tree = cKDTree(coords_mm[finite])
    k = min(int(n_neighbors) + 1, finite_idx.size)
    distances, neighbors = tree.query(coords_mm[finite], k=k)
    edges: set[tuple[int, int]] = set()
    for local_i in range(finite_idx.size):
        for dist, local_j in zip(np.atleast_1d(distances[local_i])[1:], np.atleast_1d(neighbors[local_i])[1:]):
            if max_distance_mm is not None and float(dist) > float(max_distance_mm):
                continue
            i = int(finite_idx[local_i])
            j = int(finite_idx[int(local_j)])
            if i != j:
                edges.add((min(i, j), max(i, j)))
    return np.asarray(sorted(edges), dtype=np.int64)


def phase_gradient_alignment(theta: np.ndarray, coords_mm: np.ndarray, edges: np.ndarray) -> float:
    """Rubino-style PGD: norm(mean local gradients) / mean(norm local gradients).

    Local gradients are estimated by a small least-squares phase plane around each
    electrode using its neighbor edges. Pairwise edge projections alone bias
    oblique waves downward because each edge sees only the component along that
    edge.
    """
    theta = np.asarray(theta, dtype=np.float64)
    coords_mm = np.asarray(coords_mm, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.shape[0] == 0:
        return np.nan
    neighbors: list[list[int]] = [[] for _ in range(theta.size)]
    for i, j in edges:
        if 0 <= i < theta.size and 0 <= j < theta.size:
            neighbors[int(i)].append(int(j))
            neighbors[int(j)].append(int(i))

    gradients = []
    for i, neigh in enumerate(neighbors):
        if len(neigh) < 2 or not np.isfinite(theta[i]) or not np.isfinite(coords_mm[i]).all():
            continue
        neigh_arr = np.asarray(neigh, dtype=np.int64)
        good = np.isfinite(theta[neigh_arr]) & np.isfinite(coords_mm[neigh_arr]).all(axis=1)
        neigh_arr = neigh_arr[good]
        if neigh_arr.size < 2:
            continue
        delta_r = coords_mm[neigh_arr] - coords_mm[i]
        delta_phi = np.angle(np.exp(1j * (theta[neigh_arr] - theta[i])))
        finite = np.isfinite(delta_phi) & np.isfinite(delta_r).all(axis=1) & (np.sum(delta_r * delta_r, axis=1) > 0)
        if np.count_nonzero(finite) < 2:
            continue
        try:
            grad, *_ = np.linalg.lstsq(delta_r[finite], delta_phi[finite], rcond=None)
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(grad).all():
            gradients.append(grad)
    if len(gradients) < 3:
        return np.nan
    gradients = np.asarray(gradients, dtype=np.float64)
    norms = np.linalg.norm(gradients, axis=1)
    good_norm = np.isfinite(norms) & (norms > 0)
    if np.count_nonzero(good_norm) < 3:
        return np.nan
    mean_gradient = np.nanmean(gradients[good_norm], axis=0)
    mean_norm = float(np.nanmean(norms[good_norm]))
    if mean_norm <= 0:
        return np.nan
    return float(np.clip(np.linalg.norm(mean_gradient) / mean_norm, 0.0, 1.0))


def make_local_phase_neighborhoods(
    coords_mm: np.ndarray,
    *,
    radius_mm: float = 0.30,
    min_neighbors: int = 12,
    max_radius_mm: float = 0.45,
    max_neighbors: int = 24,
    min_geometry_ratio: float = 0.10,
) -> dict[str, object]:
    """Precompute hybrid-radius neighborhoods for local phase-gradient fits.

    The center electrode is included in each returned index array but is not
    counted toward ``min_neighbors`` or ``max_neighbors``. Sparse neighborhoods
    expand from ``radius_mm`` up to ``max_radius_mm`` using nearest neighbors.
    """
    coords_mm = np.asarray(coords_mm, dtype=np.float64)
    if coords_mm.ndim != 2 or coords_mm.shape[1] != 2:
        raise ValueError("coords_mm must have shape (n_channels, 2)")
    if not (0 < radius_mm <= max_radius_mm):
        raise ValueError("Require 0 < radius_mm <= max_radius_mm")
    if not (2 <= min_neighbors <= max_neighbors):
        raise ValueError("Require 2 <= min_neighbors <= max_neighbors")
    if not (0 <= min_geometry_ratio <= 1):
        raise ValueError("min_geometry_ratio must lie in [0, 1]")

    finite = np.isfinite(coords_mm).all(axis=1)
    finite_idx = np.flatnonzero(finite)
    neighborhoods: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(coords_mm.shape[0])]
    geometry_ratio = np.full(coords_mm.shape[0], np.nan, dtype=np.float64)
    valid = np.zeros(coords_mm.shape[0], dtype=bool)
    if finite_idx.size < min_neighbors + 1:
        return {
            "indices": neighborhoods,
            "geometry_ratio": geometry_ratio,
            "valid": valid,
            "radius_mm": float(radius_mm),
            "max_radius_mm": float(max_radius_mm),
            "min_neighbors": int(min_neighbors),
            "max_neighbors": int(max_neighbors),
        }

    finite_coords = coords_mm[finite]
    tree = cKDTree(finite_coords)
    query_k = min(int(max_neighbors) + 1, finite_idx.size)
    distances, nearest = tree.query(finite_coords, k=query_k)
    for local_i, center_idx in enumerate(finite_idx):
        dist_i = np.atleast_1d(distances[local_i])
        nearest_i = np.atleast_1d(nearest[local_i])
        is_other = nearest_i != local_i
        within_default = is_other & (dist_i <= float(radius_mm))
        selected = nearest_i[within_default]
        if selected.size < min_neighbors:
            within_extended = is_other & (dist_i <= float(max_radius_mm))
            selected = nearest_i[within_extended]
        selected = selected[: int(max_neighbors)]
        if selected.size < min_neighbors:
            continue

        indices = np.concatenate(([int(center_idx)], finite_idx[selected].astype(np.int64, copy=False)))
        local_coords = coords_mm[indices]
        centered = local_coords - np.mean(local_coords, axis=0, keepdims=True)
        covariance = centered.T @ centered / max(1, local_coords.shape[0] - 1)
        eigvals = np.linalg.eigvalsh(covariance)
        ratio = float(eigvals[0] / eigvals[-1]) if eigvals[-1] > 0 else 0.0
        geometry_ratio[int(center_idx)] = ratio
        if ratio < float(min_geometry_ratio):
            continue
        neighborhoods[int(center_idx)] = indices
        valid[int(center_idx)] = True

    return {
        "indices": neighborhoods,
        "geometry_ratio": geometry_ratio,
        "valid": valid,
        "radius_mm": float(radius_mm),
        "max_radius_mm": float(max_radius_mm),
        "min_neighbors": int(min_neighbors),
        "max_neighbors": int(max_neighbors),
    }


def estimate_local_phase_velocity(
    u: np.ndarray,
    amp: np.ndarray,
    coords_mm: np.ndarray,
    neighborhoods: dict[str, object],
    *,
    fs_ds: float,
    omega_smooth_ms: float = 8.0,
    min_r_local: float = 0.50,
    min_cycles_local: float = 0.50,
    max_speed_mm_per_s: float | None = None,
    min_amplitude_percentile: float = 20.0,
    amplitude_weighted: bool = True,
) -> dict[str, np.ndarray]:
    """Estimate local phase gradients and normal phase-advance velocities.

    The local wrapped phase differences are regressed against coordinate
    differences. Temporal phase derivatives are unwrapped only along time and
    averaged inside each spatial neighborhood. The returned velocity is the
    normal phase velocity ``-omega * k / |k|^2``.
    """
    u = np.asarray(u)
    amp = np.asarray(amp, dtype=np.float64)
    coords_mm = np.asarray(coords_mm, dtype=np.float64)
    if u.ndim != 2 or amp.shape != u.shape:
        raise ValueError("u and amp must have matching shape (n_samples, n_channels)")
    if coords_mm.shape != (u.shape[1], 2):
        raise ValueError("coords_mm must have shape (n_channels, 2)")
    if fs_ds <= 0:
        raise ValueError("fs_ds must be positive")

    n_samples, n_channels = u.shape
    shape = (n_samples, n_channels)
    kx = np.full(shape, np.nan, dtype=np.float64)
    ky = np.full(shape, np.nan, dtype=np.float64)
    omega = np.full(shape, np.nan, dtype=np.float64)
    r_local = np.full(shape, np.nan, dtype=np.float64)
    cycles = np.full(shape, np.nan, dtype=np.float64)
    velocity_x = np.full(shape, np.nan, dtype=np.float64)
    velocity_y = np.full(shape, np.nan, dtype=np.float64)
    speed = np.full(shape, np.nan, dtype=np.float64)

    theta = np.unwrap(np.angle(u), axis=0)
    channel_omega = np.gradient(theta, 1.0 / float(fs_ds), axis=0)
    smooth_samples = max(1, int(round(float(omega_smooth_ms) * float(fs_ds) / 1000.0)))
    if smooth_samples > 1:
        channel_omega = uniform_filter1d(channel_omega, size=smooth_samples, axis=0, mode="nearest")

    neighborhood_indices = neighborhoods.get("indices")
    neighborhood_valid = np.asarray(neighborhoods.get("valid"), dtype=bool)
    if not isinstance(neighborhood_indices, list) or neighborhood_valid.shape != (n_channels,):
        raise ValueError("neighborhoods must come from make_local_phase_neighborhoods")

    for center_idx in np.flatnonzero(neighborhood_valid):
        indices = np.asarray(neighborhood_indices[int(center_idx)], dtype=np.int64)
        if indices.size < 3:
            continue
        delta_r = coords_mm[indices] - coords_mm[int(center_idx)]
        dx = delta_r[:, 0]
        dy = delta_r[:, 1]
        phase_delta = np.angle(u[:, indices] * np.conj(u[:, int(center_idx), None]))
        usable_amp = np.isfinite(amp[:, indices]) & (amp[:, indices] > 0)
        if amplitude_weighted:
            weights = np.where(usable_amp, amp[:, indices], 0.0)
        else:
            weights = usable_amp.astype(np.float64)

        a11 = np.sum(weights * dx[None, :] * dx[None, :], axis=1)
        a12 = np.sum(weights * dx[None, :] * dy[None, :], axis=1)
        a22 = np.sum(weights * dy[None, :] * dy[None, :], axis=1)
        b1 = np.sum(weights * dx[None, :] * phase_delta, axis=1)
        b2 = np.sum(weights * dy[None, :] * phase_delta, axis=1)
        determinant = a11 * a22 - a12 * a12
        solvable = np.isfinite(determinant) & (determinant > 1e-15)
        grad_x = np.divide(a22 * b1 - a12 * b2, determinant, out=np.full(n_samples, np.nan), where=solvable)
        grad_y = np.divide(a11 * b2 - a12 * b1, determinant, out=np.full(n_samples, np.nan), where=solvable)
        kx[:, int(center_idx)] = grad_x
        ky[:, int(center_idx)] = grad_y

        predicted_delta = grad_x[:, None] * dx[None, :] + grad_y[:, None] * dy[None, :]
        residual_phasor = np.exp(1j * (phase_delta - predicted_delta))
        weight_sum = np.sum(weights, axis=1)
        r_local[:, int(center_idx)] = np.divide(
            np.abs(np.sum(weights * residual_phasor, axis=1)),
            weight_sum,
            out=np.full(n_samples, np.nan),
            where=weight_sum > 0,
        )
        omega[:, int(center_idx)] = np.divide(
            np.sum(weights * channel_omega[:, indices], axis=1),
            weight_sum,
            out=np.full(n_samples, np.nan),
            where=weight_sum > 0,
        )

        k_norm = np.hypot(grad_x, grad_y)
        unit_x = np.divide(grad_x, k_norm, out=np.full(n_samples, np.nan), where=k_norm > 0)
        unit_y = np.divide(grad_y, k_norm, out=np.full(n_samples, np.nan), where=k_norm > 0)
        projection = delta_r[:, 0][None, :] * unit_x[:, None] + delta_r[:, 1][None, :] * unit_y[:, None]
        aperture = np.nanmax(projection, axis=1) - np.nanmin(projection, axis=1)
        cycles[:, int(center_idx)] = k_norm * aperture / (2.0 * np.pi)
        speed[:, int(center_idx)] = np.divide(
            np.abs(omega[:, int(center_idx)]),
            k_norm,
            out=np.full(n_samples, np.nan),
            where=k_norm > 0,
        )
        velocity_x[:, int(center_idx)] = np.divide(
            -omega[:, int(center_idx)] * grad_x,
            k_norm * k_norm,
            out=np.full(n_samples, np.nan),
            where=k_norm > 0,
        )
        velocity_y[:, int(center_idx)] = np.divide(
            -omega[:, int(center_idx)] * grad_y,
            k_norm * k_norm,
            out=np.full(n_samples, np.nan),
            where=k_norm > 0,
        )

    finite_amp = amp[np.isfinite(amp)]
    amp_floor = float(np.percentile(finite_amp, min_amplitude_percentile)) if finite_amp.size else np.nan
    valid = (
        np.isfinite(speed)
        & np.isfinite(r_local)
        & (r_local >= float(min_r_local))
        & np.isfinite(cycles)
        & (cycles >= float(min_cycles_local))
        & np.isfinite(amp)
        & (amp >= amp_floor)
    )
    if max_speed_mm_per_s is not None:
        valid &= speed <= float(max_speed_mm_per_s)

    return {
        "kx": kx,
        "ky": ky,
        "omega": omega,
        "R_local": r_local,
        "cycles_across_neighborhood": cycles,
        "speed_mm_per_s": speed,
        "velocity_x_mm_per_s": velocity_x,
        "velocity_y_mm_per_s": velocity_y,
        "valid": valid,
        "amplitude_floor": np.asarray(amp_floor),
    }


def _phase_bin_edges(n_bins: int) -> np.ndarray:
    if int(n_bins) < 4:
        raise ValueError("n_bins must be at least 4")
    return np.linspace(-np.pi, np.pi, int(n_bins) + 1, dtype=np.float64)


def estimate_excitable_phase(
    u: np.ndarray,
    time_s: np.ndarray,
    electrodes: np.ndarray,
    spike_times_s: np.ndarray,
    spike_electrodes: np.ndarray,
    *,
    n_bins: int = 36,
    pseudocount: float = 1.0,
) -> dict[str, np.ndarray | float | int]:
    """Estimate spike-associated Hilbert phase after correcting phase occupancy."""
    u = np.asarray(u)
    time_s = np.asarray(time_s, dtype=np.float64)
    electrodes = np.asarray(electrodes)
    spike_times_s = np.asarray(spike_times_s, dtype=np.float64)
    spike_electrodes = np.asarray(spike_electrodes)
    if u.ndim != 2 or time_s.shape != (u.shape[0],):
        raise ValueError("u and time_s must have shapes (n_samples, n_channels) and (n_samples,)")
    if electrodes.shape != (u.shape[1],):
        raise ValueError("electrodes must have shape (n_channels,)")
    if spike_times_s.shape != spike_electrodes.shape:
        raise ValueError("spike_times_s and spike_electrodes must have matching shapes")
    if time_s.size < 2 or np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must contain at least two strictly increasing values")
    if pseudocount < 0:
        raise ValueError("pseudocount must be non-negative")

    edges = _phase_bin_edges(n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    occupancy_counts = np.histogram(np.angle(u).ravel(), bins=edges)[0].astype(np.float64)
    spike_phase_counts = np.zeros(int(n_bins), dtype=np.float64)
    electrode_to_idx = {int(electrode): idx for idx, electrode in enumerate(electrodes)}
    sampled_phases = []
    in_time = (spike_times_s >= time_s[0]) & (spike_times_s <= time_s[-1])
    for electrode in np.unique(spike_electrodes[in_time]):
        channel_idx = electrode_to_idx.get(int(electrode))
        if channel_idx is None:
            continue
        times = spike_times_s[in_time & (spike_electrodes == electrode)]
        real = np.interp(times, time_s, np.real(u[:, channel_idx]))
        imag = np.interp(times, time_s, np.imag(u[:, channel_idx]))
        sampled_phases.append(np.angle(real + 1j * imag))
    spike_phases = np.concatenate(sampled_phases) if sampled_phases else np.empty(0, dtype=np.float64)
    if spike_phases.size:
        spike_phase_counts += np.histogram(spike_phases, bins=edges)[0]

    occupancy_probability = (occupancy_counts + float(pseudocount)) / (
        np.sum(occupancy_counts) + float(pseudocount) * int(n_bins)
    )
    spike_probability = (spike_phase_counts + float(pseudocount)) / (
        np.sum(spike_phase_counts) + float(pseudocount) * int(n_bins)
    )
    relative_spike_probability = np.divide(
        spike_probability,
        occupancy_probability,
        out=np.zeros_like(spike_probability),
        where=occupancy_counts > 0,
    )
    mean_relative = float(np.mean(relative_spike_probability))
    if mean_relative > 0:
        relative_spike_probability /= mean_relative
    best_idx = int(np.nanargmax(relative_spike_probability))
    return {
        "phase_bin_edges_rad": edges,
        "phase_bin_centers_rad": centers,
        "occupancy_counts": occupancy_counts,
        "spike_phase_counts": spike_phase_counts,
        "relative_spike_probability": relative_spike_probability,
        "theta_excitable_rad": float(centers[best_idx]),
        "n_spikes_sampled": int(spike_phases.size),
    }


def combine_excitable_phase_calibrations(
    calibrations: Iterable[dict[str, np.ndarray | float | int]],
    *,
    pseudocount: float = 1.0,
) -> dict[str, np.ndarray | float | int]:
    """Combine chunk-level phase calibration counts without retaining phasors."""
    calibrations = list(calibrations)
    if not calibrations:
        raise ValueError("At least one calibration is required")
    edges = np.asarray(calibrations[0]["phase_bin_edges_rad"], dtype=np.float64)
    centers = np.asarray(calibrations[0]["phase_bin_centers_rad"], dtype=np.float64)
    occupancy = np.zeros(centers.size, dtype=np.float64)
    spike_counts = np.zeros(centers.size, dtype=np.float64)
    n_spikes = 0
    for calibration in calibrations:
        if not np.allclose(calibration["phase_bin_edges_rad"], edges):
            raise ValueError("All calibrations must use the same phase bins")
        occupancy += np.asarray(calibration["occupancy_counts"], dtype=np.float64)
        spike_counts += np.asarray(calibration["spike_phase_counts"], dtype=np.float64)
        n_spikes += int(calibration["n_spikes_sampled"])
    occupancy_probability = (occupancy + float(pseudocount)) / (np.sum(occupancy) + float(pseudocount) * centers.size)
    spike_probability = (spike_counts + float(pseudocount)) / (np.sum(spike_counts) + float(pseudocount) * centers.size)
    relative = np.divide(spike_probability, occupancy_probability, out=np.zeros_like(spike_probability), where=occupancy > 0)
    mean_relative = float(np.mean(relative))
    if mean_relative > 0:
        relative /= mean_relative
    return {
        "phase_bin_edges_rad": edges,
        "phase_bin_centers_rad": centers,
        "occupancy_counts": occupancy,
        "spike_phase_counts": spike_counts,
        "relative_spike_probability": relative,
        "theta_excitable_rad": float(centers[int(np.nanargmax(relative))]),
        "n_spikes_sampled": int(n_spikes),
    }


def detect_phase_crossings(
    u: np.ndarray,
    time_s: np.ndarray,
    theta_target_rad: float,
    *,
    start_s: float | None = None,
    stop_s: float | None = None,
    direction: int = 1,
) -> list[np.ndarray]:
    """Return interpolated preferred-phase crossing times for each channel."""
    u = np.asarray(u)
    time_s = np.asarray(time_s, dtype=np.float64)
    if u.ndim != 2 or time_s.shape != (u.shape[0],):
        raise ValueError("u and time_s must have shapes (n_samples, n_channels) and (n_samples,)")
    if time_s.size < 2 or np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must contain at least two strictly increasing values")
    if direction not in {-1, 0, 1}:
        raise ValueError("direction must be -1, 0, or 1")
    keep_start = float(time_s[0] if start_s is None else start_s)
    keep_stop = float(time_s[-1] if stop_s is None else stop_s)
    theta = np.unwrap(np.angle(u), axis=0)
    crossings: list[np.ndarray] = []
    for channel_idx in range(theta.shape[1]):
        values = theta[:, channel_idx]
        channel_crossings = []
        for sample_idx in range(values.size - 1):
            a = float(values[sample_idx])
            b = float(values[sample_idx + 1])
            if not np.isfinite(a) or not np.isfinite(b) or np.isclose(a, b):
                continue
            delta = b - a
            if direction > 0 and delta <= 0:
                continue
            if direction < 0 and delta >= 0:
                continue
            low, high = sorted((a, b))
            n_min = int(np.ceil((low - float(theta_target_rad)) / (2.0 * np.pi)))
            n_max = int(np.floor((high - float(theta_target_rad)) / (2.0 * np.pi)))
            for cycle_idx in range(n_min, n_max + 1):
                level = float(theta_target_rad) + 2.0 * np.pi * cycle_idx
                fraction = (level - a) / delta
                if 0.0 <= fraction <= 1.0:
                    crossing = float(time_s[sample_idx] + fraction * (time_s[sample_idx + 1] - time_s[sample_idx]))
                    if keep_start <= crossing <= keep_stop:
                        channel_crossings.append(crossing)
        crossings.append(np.asarray(channel_crossings, dtype=np.float64))
    return crossings


def select_coherent_phase_front(
    crossings_by_channel: list[np.ndarray],
    anchor_time_s: float,
    *,
    frequency_hz: float = 45.0,
    match_max_ms: float = 8.0,
    min_electrodes: int = 50,
    search_pre_ms: float = 20.0,
    search_post_ms: float = 40.0,
) -> dict[str, np.ndarray | float | int]:
    """Choose the recurrent crossing front nearest an event anchor."""
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    tolerance_s = float(match_max_ms) / 1000.0
    if tolerance_s <= 0:
        raise ValueError("match_max_ms must be positive")
    period_s = 1.0 / float(frequency_hz)
    search_start = float(anchor_time_s) - float(search_pre_ms) / 1000.0
    search_stop = float(anchor_time_s) + float(search_post_ms) / 1000.0
    first_offset = int(np.floor((search_start - float(anchor_time_s)) / period_s))
    last_offset = int(np.ceil((search_stop - float(anchor_time_s)) / period_s))
    candidates = []
    for cycle_offset in range(first_offset, last_offset + 1):
        center = float(anchor_time_s) + cycle_offset * period_s
        arrivals = np.full(len(crossings_by_channel), np.nan, dtype=np.float64)
        for channel_idx, channel_crossings in enumerate(crossings_by_channel):
            values = np.asarray(channel_crossings, dtype=np.float64)
            if values.size == 0:
                continue
            nearest_idx = int(np.argmin(np.abs(values - center)))
            if abs(float(values[nearest_idx]) - center) <= tolerance_s:
                arrivals[channel_idx] = float(values[nearest_idx])
        for _ in range(2):
            finite = np.isfinite(arrivals)
            if not np.any(finite):
                break
            center = float(np.nanmedian(arrivals))
            for channel_idx, channel_crossings in enumerate(crossings_by_channel):
                values = np.asarray(channel_crossings, dtype=np.float64)
                if values.size == 0:
                    arrivals[channel_idx] = np.nan
                    continue
                nearest_idx = int(np.argmin(np.abs(values - center)))
                nearest = float(values[nearest_idx])
                arrivals[channel_idx] = nearest if abs(nearest - center) <= tolerance_s else np.nan
        n_electrodes = int(np.count_nonzero(np.isfinite(arrivals)))
        if n_electrodes:
            candidates.append((abs(center - float(anchor_time_s)), -n_electrodes, center, cycle_offset, arrivals))
    if not candidates:
        return {
            "arrival_time_s": np.full(len(crossings_by_channel), np.nan, dtype=np.float64),
            "front_time_s": np.nan,
            "cycle_offset": 0,
            "n_electrodes": 0,
            "valid": 0,
        }
    _, neg_n, center, cycle_offset, arrivals = min(candidates, key=lambda item: (item[0], item[1]))
    n_electrodes = int(-neg_n)
    return {
        "arrival_time_s": arrivals,
        "front_time_s": float(center),
        "cycle_offset": int(cycle_offset),
        "n_electrodes": n_electrodes,
        "valid": int(n_electrodes >= int(min_electrodes)),
    }


def sample_at_event_times(values: np.ndarray, time_s: np.ndarray, event_time_s: np.ndarray) -> np.ndarray:
    """Interpolate per-channel time series values at per-channel event times."""
    values = np.asarray(values, dtype=np.float64)
    time_s = np.asarray(time_s, dtype=np.float64)
    event_time_s = np.asarray(event_time_s, dtype=np.float64)
    if values.ndim != 2 or time_s.shape != (values.shape[0],):
        raise ValueError("values and time_s must have shapes (n_samples, n_channels) and (n_samples,)")
    if event_time_s.shape != (values.shape[1],):
        raise ValueError("event_time_s must have shape (n_channels,)")
    out = np.full(event_time_s.shape, np.nan, dtype=np.float64)
    for channel_idx, event_time in enumerate(event_time_s):
        if np.isfinite(event_time):
            out[channel_idx] = np.interp(event_time, time_s, values[:, channel_idx], left=np.nan, right=np.nan)
    return out


def fit_local_arrival_velocity(
    arrival_time_s: np.ndarray,
    coords_mm: np.ndarray,
    neighborhoods: dict[str, object],
    *,
    arrival_amplitude: np.ndarray | None = None,
    min_arrival_amplitude: float | np.ndarray = 0.0,
    match_max_ms: float = 8.0,
    min_neighbors: int = 8,
    max_residual_ms: float = 4.0,
    min_arrival_span_ms: float = 1.0,
    min_distance_span_mm: float = 0.20,
    max_speed_mm_per_s: float | None = 500.0,
) -> dict[str, np.ndarray]:
    """Fit local arrival-time planes and derive resolvable normal front velocities.

    Raw fits are retained even when the inferred speed is faster than the
    configured resolution ceiling. ``speed_censored`` marks these weak-gradient
    fits so they can be counted without treating the reciprocal gradient as a
    precise velocity estimate.
    """
    arrival_time_s = np.asarray(arrival_time_s, dtype=np.float64)
    coords_mm = np.asarray(coords_mm, dtype=np.float64)
    if coords_mm.shape != (arrival_time_s.size, 2):
        raise ValueError("coords_mm must have shape (n_channels, 2)")
    if arrival_amplitude is None:
        arrival_amplitude = np.full(arrival_time_s.size, np.nan, dtype=np.float64)
    arrival_amplitude = np.asarray(arrival_amplitude, dtype=np.float64)
    if arrival_amplitude.shape != arrival_time_s.shape:
        raise ValueError("arrival_amplitude must have shape (n_channels,)")
    min_arrival_amplitude = np.asarray(min_arrival_amplitude, dtype=np.float64)
    if min_arrival_amplitude.ndim == 0:
        arrival_amplitude_threshold = np.full(
            arrival_time_s.size, float(min_arrival_amplitude), dtype=np.float64
        )
    else:
        arrival_amplitude_threshold = min_arrival_amplitude
        if arrival_amplitude_threshold.shape != arrival_time_s.shape:
            raise ValueError("min_arrival_amplitude must be scalar or have shape (n_channels,)")
    neighborhood_indices = neighborhoods.get("indices")
    neighborhood_valid = np.asarray(neighborhoods.get("valid"), dtype=bool)
    if not isinstance(neighborhood_indices, list) or neighborhood_valid.shape != arrival_time_s.shape:
        raise ValueError("neighborhoods must come from make_local_phase_neighborhoods")

    n_channels = arrival_time_s.size
    ax = np.full(n_channels, np.nan, dtype=np.float64)
    ay = np.full(n_channels, np.nan, dtype=np.float64)
    residual_ms = np.full(n_channels, np.nan, dtype=np.float64)
    arrival_span_ms = np.full(n_channels, np.nan, dtype=np.float64)
    distance_span_mm = np.full(n_channels, np.nan, dtype=np.float64)
    speed = np.full(n_channels, np.nan, dtype=np.float64)
    velocity_x = np.full(n_channels, np.nan, dtype=np.float64)
    velocity_y = np.full(n_channels, np.nan, dtype=np.float64)
    n_good = np.zeros(n_channels, dtype=np.int64)
    tolerance_s = float(match_max_ms) / 1000.0
    amplitude_valid = np.ones(n_channels, dtype=bool)
    threshold_enabled = np.isfinite(arrival_amplitude_threshold) & (arrival_amplitude_threshold > 0.0)
    if np.any(threshold_enabled):
        amplitude_valid = np.where(
            threshold_enabled,
            np.isfinite(arrival_amplitude) & (arrival_amplitude >= arrival_amplitude_threshold),
            True,
        )
    candidate_centers = neighborhood_valid & np.isfinite(arrival_time_s) & amplitude_valid
    for center_idx in np.flatnonzero(candidate_centers):
        indices = np.asarray(neighborhood_indices[int(center_idx)], dtype=np.int64)
        local_times = arrival_time_s[indices]
        keep = (
            np.isfinite(local_times)
            & (np.abs(local_times - arrival_time_s[int(center_idx)]) <= tolerance_s)
            & amplitude_valid[indices]
        )
        indices = indices[keep]
        if indices.size < int(min_neighbors) + 1:
            continue
        delta_r = coords_mm[indices] - coords_mm[int(center_idx)]
        design = np.column_stack((delta_r, np.ones(indices.size, dtype=np.float64)))
        try:
            params, *_ = np.linalg.lstsq(design, arrival_time_s[indices], rcond=None)
        except np.linalg.LinAlgError:
            continue
        gradient = np.asarray(params[:2], dtype=np.float64)
        gradient_norm = float(np.linalg.norm(gradient))
        if not np.isfinite(gradient_norm) or gradient_norm <= 0:
            continue
        fitted = design @ params
        residual = (arrival_time_s[indices] - fitted) * 1000.0
        ax[int(center_idx)] = gradient[0]
        ay[int(center_idx)] = gradient[1]
        residual_ms[int(center_idx)] = float(np.sqrt(np.mean(residual * residual)))
        arrival_span_ms[int(center_idx)] = float((np.max(arrival_time_s[indices]) - np.min(arrival_time_s[indices])) * 1000.0)
        pairwise_distance = np.linalg.norm(delta_r[:, None, :] - delta_r[None, :, :], axis=2)
        distance_span_mm[int(center_idx)] = float(np.nanmax(pairwise_distance))
        speed[int(center_idx)] = 1.0 / gradient_norm
        velocity_x[int(center_idx)] = gradient[0] / (gradient_norm * gradient_norm)
        velocity_y[int(center_idx)] = gradient[1] / (gradient_norm * gradient_norm)
        n_good[int(center_idx)] = int(indices.size)
    gradient_norm = np.hypot(ax, ay)
    speed_censored = np.zeros(n_channels, dtype=bool)
    if max_speed_mm_per_s is not None:
        speed_censored = np.isfinite(speed) & (speed > float(max_speed_mm_per_s))
    valid = (
        np.isfinite(speed)
        & np.isfinite(residual_ms)
        & (residual_ms <= float(max_residual_ms))
        & np.isfinite(arrival_span_ms)
        & (arrival_span_ms >= float(min_arrival_span_ms))
        & np.isfinite(distance_span_mm)
        & (distance_span_mm >= float(min_distance_span_mm))
        & (n_good >= int(min_neighbors) + 1)
        & amplitude_valid
        & ~speed_censored
    )
    return {
        "arrival_gradient_x_s_per_mm": ax,
        "arrival_gradient_y_s_per_mm": ay,
        "velocity_x_mm_per_s": velocity_x,
        "velocity_y_mm_per_s": velocity_y,
        "speed_mm_per_s": speed,
        "residual_rms_ms": residual_ms,
        "arrival_span_ms": arrival_span_ms,
        "distance_span_mm": distance_span_mm,
        "arrival_amplitude": arrival_amplitude,
        "arrival_amplitude_threshold": arrival_amplitude_threshold,
        "arrival_gradient_norm_s_per_mm": gradient_norm,
        "speed_censored": speed_censored,
        "n_good": n_good,
        "valid": valid,
    }


def fit_arrival_time_plane(arrival_time_s: np.ndarray, coords_mm: np.ndarray) -> dict[str, float | int]:
    """Fit one event-level planar arrival-time surface."""
    arrival_time_s = np.asarray(arrival_time_s, dtype=np.float64)
    coords_mm = np.asarray(coords_mm, dtype=np.float64)
    finite = np.isfinite(arrival_time_s) & np.isfinite(coords_mm).all(axis=1)
    if np.count_nonzero(finite) < 3:
        return {
            "planar_speed_mm_per_s": np.nan,
            "planar_velocity_x_mm_per_s": np.nan,
            "planar_velocity_y_mm_per_s": np.nan,
            "planar_gradient_x_s_per_mm": np.nan,
            "planar_gradient_y_s_per_mm": np.nan,
            "planar_intercept_s": np.nan,
            "planar_residual_rms_ms": np.nan,
            "planar_n_good": int(np.count_nonzero(finite)),
        }
    design = np.column_stack((coords_mm[finite], np.ones(np.count_nonzero(finite))))
    params, *_ = np.linalg.lstsq(design, arrival_time_s[finite], rcond=None)
    gradient = params[:2]
    norm = float(np.linalg.norm(gradient))
    fitted = design @ params
    residual = (arrival_time_s[finite] - fitted) * 1000.0
    return {
        "planar_speed_mm_per_s": float(1.0 / norm) if norm > 0 else np.nan,
        "planar_velocity_x_mm_per_s": float(gradient[0] / (norm * norm)) if norm > 0 else np.nan,
        "planar_velocity_y_mm_per_s": float(gradient[1] / (norm * norm)) if norm > 0 else np.nan,
        "planar_gradient_x_s_per_mm": float(gradient[0]),
        "planar_gradient_y_s_per_mm": float(gradient[1]),
        "planar_intercept_s": float(params[2]),
        "planar_residual_rms_ms": float(np.sqrt(np.mean(residual * residual))),
        "planar_n_good": int(np.count_nonzero(finite)),
    }


def fit_arrival_time_radial(
    arrival_time_s: np.ndarray,
    coords_mm: np.ndarray,
    *,
    center_grid_n: int = 15,
) -> dict[str, float | int]:
    """Fit one event-level radial arrival-time surface over on-array centers."""
    arrival_time_s = np.asarray(arrival_time_s, dtype=np.float64)
    coords_mm = np.asarray(coords_mm, dtype=np.float64)
    finite = np.isfinite(arrival_time_s) & np.isfinite(coords_mm).all(axis=1)
    if np.count_nonzero(finite) < 3:
        return {
            "radial_x0_mm": np.nan,
            "radial_y0_mm": np.nan,
            "radial_speed_mm_per_s": np.nan,
            "radial_sign": 0,
            "radial_slope_s_per_mm": np.nan,
            "radial_intercept_s": np.nan,
            "radial_residual_rms_ms": np.nan,
            "radial_n_good": int(np.count_nonzero(finite)),
        }
    coords = coords_mm[finite]
    times = arrival_time_s[finite]
    x_grid = np.linspace(np.min(coords[:, 0]), np.max(coords[:, 0]), int(center_grid_n))
    y_grid = np.linspace(np.min(coords[:, 1]), np.max(coords[:, 1]), int(center_grid_n))
    best = None
    for x0 in x_grid:
        for y0 in y_grid:
            distance = np.linalg.norm(coords - np.array([x0, y0]), axis=1)
            design = np.column_stack((distance, np.ones(distance.size)))
            params, *_ = np.linalg.lstsq(design, times, rcond=None)
            residual = times - design @ params
            rss = float(np.sum(residual * residual))
            if best is None or rss < best[0]:
                best = (rss, float(x0), float(y0), float(params[0]), residual)
    _, x0, y0, slope, residual = best
    return {
        "radial_x0_mm": x0,
        "radial_y0_mm": y0,
        "radial_speed_mm_per_s": float(1.0 / abs(slope)) if slope != 0 else np.nan,
        "radial_sign": int(np.sign(slope)),
        "radial_slope_s_per_mm": float(slope),
        "radial_intercept_s": float(times.mean() - slope * np.linalg.norm(coords - np.array([x0, y0]), axis=1).mean()),
        "radial_residual_rms_ms": float(np.sqrt(np.mean((residual * 1000.0) ** 2))),
        "radial_n_good": int(np.count_nonzero(finite)),
    }


def analyze_excitable_phase_front(
    u: np.ndarray,
    time_s: np.ndarray,
    coords_mm: np.ndarray,
    neighborhoods: dict[str, object],
    *,
    amplitude: np.ndarray | None = None,
    min_arrival_amplitude: float = 0.0,
    min_arrival_amplitude_percentile: float = 0.0,
    theta_excitable_rad: float,
    anchor_time_s: float,
    frequency_hz: float = 45.0,
    match_max_ms: float = 8.0,
    min_electrodes: int = 50,
    search_pre_ms: float = 20.0,
    search_post_ms: float = 40.0,
    min_neighbors: int = 8,
    max_residual_ms: float = 4.0,
    min_arrival_span_ms: float = 1.0,
    min_distance_span_mm: float = 0.20,
    max_speed_mm_per_s: float | None = 500.0,
    radial_center_grid_n: int = 15,
) -> dict[str, object]:
    """Analyze one event's preferred-phase crossing front."""
    crossings = detect_phase_crossings(
        u,
        time_s,
        theta_excitable_rad,
        start_s=float(anchor_time_s) - float(search_pre_ms) / 1000.0,
        stop_s=float(anchor_time_s) + float(search_post_ms) / 1000.0,
    )
    front = select_coherent_phase_front(
        crossings,
        anchor_time_s,
        frequency_hz=frequency_hz,
        match_max_ms=match_max_ms,
        min_electrodes=min_electrodes,
        search_pre_ms=search_pre_ms,
        search_post_ms=search_post_ms,
    )
    arrival_time_s = np.asarray(front["arrival_time_s"], dtype=np.float64)
    arrival_amplitude = (
        sample_at_event_times(amplitude, time_s, arrival_time_s)
        if amplitude is not None
        else np.full(arrival_time_s.shape, np.nan, dtype=np.float64)
    )
    min_arrival_amplitude_percentile = float(min_arrival_amplitude_percentile)
    if not 0.0 <= min_arrival_amplitude_percentile <= 100.0:
        raise ValueError("min_arrival_amplitude_percentile must be in [0, 100]")
    arrival_amplitude_threshold = np.full(
        arrival_time_s.shape, float(min_arrival_amplitude), dtype=np.float64
    )
    if min_arrival_amplitude_percentile > 0.0:
        if amplitude is None:
            arrival_amplitude_threshold = np.full(arrival_time_s.shape, np.nan, dtype=np.float64)
        else:
            percentile_threshold = np.nanpercentile(
                np.asarray(amplitude, dtype=np.float64),
                min_arrival_amplitude_percentile,
                axis=0,
            )
            arrival_amplitude_threshold = np.maximum(arrival_amplitude_threshold, percentile_threshold)
    local = fit_local_arrival_velocity(
        arrival_time_s,
        coords_mm,
        neighborhoods,
        arrival_amplitude=arrival_amplitude,
        min_arrival_amplitude=arrival_amplitude_threshold,
        match_max_ms=match_max_ms,
        min_neighbors=min_neighbors,
        max_residual_ms=max_residual_ms,
        min_arrival_span_ms=min_arrival_span_ms,
        min_distance_span_mm=min_distance_span_mm,
        max_speed_mm_per_s=max_speed_mm_per_s,
    )
    return {
        "crossings_by_channel": crossings,
        "front": front,
        "arrival_amplitude": arrival_amplitude,
        "arrival_amplitude_threshold": arrival_amplitude_threshold,
        "local": local,
        "planar": fit_arrival_time_plane(arrival_time_s, coords_mm),
        "radial": fit_arrival_time_radial(arrival_time_s, coords_mm, center_grid_n=radial_center_grid_n),
    }


def initialize_wavefront_cache(
    path: str | Path,
    calibration: dict[str, np.ndarray | float | int],
    *,
    config: dict[str, object] | None = None,
) -> None:
    """Create a wavefront HDF5 cache with calibration and appendable result groups."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        calibration_group = h5.create_group("wavefront_calibration")
        for key, value in calibration.items():
            if np.isscalar(value):
                calibration_group.attrs[key] = value
            else:
                calibration_group.create_dataset(key, data=np.asarray(value))
        if config is not None:
            calibration_group.attrs["config"] = json.dumps(config, sort_keys=True)
        h5.create_group("wavefront_events")
        h5.create_group("wavefront_local")


def append_wavefront_rows(path: str | Path, group_name: str, rows: dict[str, np.ndarray] | list[dict[str, object]]) -> None:
    """Append column-oriented or record-oriented rows to one wavefront cache group."""
    if isinstance(rows, list):
        if not rows:
            return
        rows = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    else:
        rows = {key: np.asarray(value) for key, value in rows.items()}
    if not rows:
        return
    lengths = {value.shape[0] if value.ndim else 1 for value in rows.values()}
    if len(lengths) != 1:
        raise ValueError("All appended wavefront columns must have matching lengths")
    n_rows = int(next(iter(lengths)))
    with h5py.File(path, "a") as h5:
        group = h5.require_group(str(group_name).strip("/"))
        if group.keys() and set(group.keys()) != set(rows):
            raise ValueError("Appended wavefront columns must match the existing cache schema")
        existing = {dataset.shape[0] for dataset in group.values()}
        old_size = int(next(iter(existing))) if existing else 0
        if existing and len(existing) != 1:
            raise ValueError("Existing wavefront cache columns have inconsistent lengths")
        for key, values in rows.items():
            values = np.atleast_1d(values)
            if key not in group:
                dataset = group.create_dataset(key, shape=(old_size,), maxshape=(None,), dtype=values.dtype, chunks=True)
            else:
                dataset = group[key]
            dataset.resize((old_size + n_rows,))
            dataset[old_size:] = values


def read_wavefront_cache(path: str | Path) -> dict[str, dict[str, object]]:
    """Read wavefront calibration and result groups from HDF5."""
    result: dict[str, dict[str, object]] = {}
    with h5py.File(path, "r") as h5:
        for group_name in ("wavefront_calibration", "wavefront_events", "wavefront_local"):
            if group_name not in h5:
                continue
            group = h5[group_name]
            payload: dict[str, object] = {key: dataset[:] for key, dataset in group.items()}
            payload.update({key: value for key, value in group.attrs.items()})
            result[group_name] = payload
    return result


def merge_time_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping time intervals while preserving sorted coverage."""
    ordered = sorted((float(start), float(stop)) for start, stop in intervals if float(stop) > float(start))
    merged: list[list[float]] = []
    for start, stop in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return [(start, stop) for start, stop in merged]


def select_representative_event_indices(
    n_events: int,
    *,
    n_calibration: int | None,
    n_scoring: int,
) -> dict[str, np.ndarray | int]:
    """Select deterministic, approximately even calibration and scoring events.

    Scoring events are selected first. Calibration events are selected from the
    remaining events where possible. When there are too few events for disjoint
    sets, the calibration set is filled deterministically from the scoring set.
    ``n_calibration=None`` selects every event for confirmatory calibration.
    """
    n_events = int(n_events)
    n_scoring = int(n_scoring)
    if n_events < 1:
        raise ValueError("n_events must be positive")
    if n_scoring < 1:
        raise ValueError("n_scoring must be positive")
    if n_calibration is not None and int(n_calibration) < 1:
        raise ValueError("n_calibration must be positive or None")

    def evenly_spaced(values: np.ndarray, n_keep: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.int64)
        n_keep = min(int(n_keep), values.size)
        if n_keep == values.size:
            return values.copy()
        positions = np.round(np.linspace(0, values.size - 1, n_keep)).astype(np.int64)
        return values[positions]

    all_indices = np.arange(n_events, dtype=np.int64)
    scoring = evenly_spaced(all_indices, n_scoring)
    if n_calibration is None:
        calibration = all_indices
    else:
        n_calibration = min(int(n_calibration), n_events)
        available = np.setdiff1d(all_indices, scoring, assume_unique=True)
        calibration = evenly_spaced(available, min(n_calibration, available.size))
        if calibration.size < n_calibration:
            fill = evenly_spaced(scoring, n_calibration - calibration.size)
            calibration = np.sort(np.concatenate((calibration, fill)))
    overlap = int(np.intersect1d(calibration, scoring, assume_unique=True).size)
    return {
        "calibration_indices": calibration,
        "scoring_indices": scoring,
        "overlap_count": overlap,
    }


def initialize_phasor_feature_cache(
    path: str | Path,
    coords_mm: np.ndarray,
    electrodes: np.ndarray,
    *,
    config: dict[str, object],
) -> None:
    """Create a compact cache for sequentially processed narrowband phasors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coords_mm = np.asarray(coords_mm, dtype=np.float64)
    electrodes = np.asarray(electrodes)
    if coords_mm.shape != (electrodes.size, 2):
        raise ValueError("coords_mm must have shape (n_electrodes, 2)")
    with h5py.File(path, "w") as h5:
        h5.attrs["config"] = json.dumps(config, sort_keys=True)
        h5.attrs["complete"] = False
        h5.create_dataset("coords_mm", data=coords_mm)
        h5.create_dataset("electrodes", data=electrodes)
        h5.create_group("segments")


def append_phasor_feature_segment(
    path: str | Path,
    start_s: float,
    time_s: np.ndarray,
    u: np.ndarray,
    amplitude: np.ndarray | None = None,
) -> None:
    """Append one independently filtered phasor segment to a feature cache."""
    time_s = np.asarray(time_s, dtype=np.float64)
    u = np.asarray(u, dtype=np.complex64)
    if u.ndim != 2 or time_s.shape != (u.shape[0],):
        raise ValueError("u and time_s must have shapes (n_samples, n_channels) and (n_samples,)")
    if amplitude is not None:
        amplitude = np.asarray(amplitude, dtype=np.float32)
        if amplitude.shape != u.shape:
            raise ValueError("amplitude must have the same shape as u")
    if time_s.size < 1 or np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must contain strictly increasing values")
    with h5py.File(path, "a") as h5:
        if u.shape[1] != h5["electrodes"].shape[0]:
            raise ValueError("Segment channel count does not match feature-cache electrodes")
        segments = h5.require_group("segments")
        name = f"segment{len(segments):04d}"
        group = segments.create_group(name)
        group.attrs["start_s"] = float(start_s)
        group.attrs["stop_s"] = float(time_s[-1])
        group.create_dataset("time_s", data=time_s)
        group.create_dataset("u_real", data=np.real(u).astype(np.float32, copy=False), compression="lzf", shuffle=True)
        group.create_dataset("u_imag", data=np.imag(u).astype(np.float32, copy=False), compression="lzf", shuffle=True)
        if amplitude is not None:
            group.create_dataset("amplitude", data=amplitude.astype(np.float32, copy=False), compression="lzf", shuffle=True)


def finalize_phasor_feature_cache(path: str | Path) -> None:
    """Mark a phasor cache complete after all requested segments were written."""
    with h5py.File(path, "a") as h5:
        h5.attrs["complete"] = True


def phasor_feature_cache_matches(path: str | Path, config: dict[str, object]) -> bool:
    """Return whether a complete phasor cache matches its preprocessing config."""
    path = Path(path)
    if not path.exists():
        return False
    with h5py.File(path, "r") as h5:
        return bool(h5.attrs.get("complete", False)) and json.loads(h5.attrs.get("config", "{}")) == config


def read_phasor_feature_metadata(path: str | Path) -> dict[str, object]:
    """Read compact metadata without decompressing cached phasor arrays."""
    with h5py.File(path, "r") as h5:
        segments = [
            {
                "name": name,
                "start_s": float(group.attrs["start_s"]),
                "stop_s": float(group.attrs["stop_s"]),
                "n_samples": int(group["time_s"].shape[0]),
                "has_amplitude": "amplitude" in group,
            }
            for name, group in h5["segments"].items()
        ]
        return {
            "config": json.loads(h5.attrs.get("config", "{}")),
            "complete": bool(h5.attrs.get("complete", False)),
            "coords_mm": h5["coords_mm"][:],
            "electrodes": h5["electrodes"][:],
            "segments": segments,
        }


def read_phasor_feature_interval(
    path: str | Path,
    start_s: float,
    stop_s: float,
    *,
    include_amplitude: bool = False,
) -> tuple[np.ndarray, ...]:
    """Read a requested interval from one cached phasor segment.

    Returns ``u, time_s, coords_mm, electrodes`` or ``u, time_s, coords_mm,
    electrodes, amplitude`` when ``include_amplitude=True``. Requested
    intervals must be fully covered by one merged feature segment.
    """
    start_s = float(start_s)
    stop_s = float(stop_s)
    if stop_s <= start_s:
        raise ValueError("Require stop_s > start_s")
    with h5py.File(path, "r") as h5:
        coords_mm = h5["coords_mm"][:]
        electrodes = h5["electrodes"][:]
        for group in h5["segments"].values():
            time_s = group["time_s"][:]
            sample_step = float(np.median(np.diff(time_s))) if time_s.size > 1 else 0.0
            tolerance = 0.51 * sample_step
            if start_s < float(time_s[0]) - tolerance or stop_s > float(time_s[-1]) + 1.01 * sample_step:
                continue
            keep = (time_s >= start_s - tolerance) & (time_s < stop_s - tolerance)
            u = group["u_real"][keep].astype(np.float32) + 1j * group["u_imag"][keep].astype(np.float32)
            if include_amplitude:
                amplitude = group["amplitude"][keep] if "amplitude" in group else np.full(u.shape, np.nan, dtype=np.float32)
                return u.astype(np.complex64, copy=False), time_s[keep], coords_mm, electrodes, amplitude
            return u.astype(np.complex64, copy=False), time_s[keep], coords_mm, electrodes
    raise ValueError(f"No cached phasor segment covers interval [{start_s:g}, {stop_s:g}] s")


def write_wavefront_sensitivity_cache(
    path: str | Path,
    rows: list[dict[str, object]],
    *,
    config: dict[str, object],
) -> None:
    """Write a compact table of event-wise wavefront sensitivity summaries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["config"] = json.dumps(config, sort_keys=True)
        group = h5.create_group("wavefront_sensitivity")
        if not rows:
            return
        for key in rows[0]:
            values = [row[key] for row in rows]
            if isinstance(values[0], str):
                group.create_dataset(key, data=np.asarray(values, dtype=h5py.string_dtype("utf-8")))
            else:
                group.create_dataset(key, data=np.asarray(values))


def read_wavefront_sensitivity_cache(path: str | Path) -> dict[str, object]:
    """Read a wavefront sensitivity table written by ``write_wavefront_sensitivity_cache``."""
    with h5py.File(path, "r") as h5:
        group = h5["wavefront_sensitivity"]
        rows = {}
        for key, dataset in group.items():
            values = dataset[:]
            if values.dtype.kind in {"S", "O"}:
                values = values.astype(str)
            rows[key] = values
        return {"config": json.loads(h5.attrs["config"]), "rows": rows}


def wavefront_sensitivity_cache_matches(path: str | Path, config: dict[str, object]) -> tuple[bool, str]:
    """Return whether a wavefront sensitivity cache exactly matches its configuration."""
    path = Path(path)
    if not path.exists():
        return False, "missing"
    try:
        with h5py.File(path, "r") as h5:
            cached = json.loads(h5.attrs["config"])
            if "wavefront_sensitivity" not in h5:
                return False, "missing /wavefront_sensitivity"
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        return False, f"unreadable: {exc}"
    return (True, "compatible") if cached == config else (False, "configuration differs")


def fit_pgd_plane(
    phasor: np.ndarray,
    weights: np.ndarray,
    coords_mm: np.ndarray,
    k_grid: dict[str, np.ndarray | float],
    *,
    gradient_edges: np.ndarray | None = None,
    refine: bool = False,
    min_weight: float = 1e-6,
) -> dict[str, float]:
    """Fit one instantaneous planar phase map and return PGD-compatible metrics."""
    fit = fit_k_phasor_plane(phasor, weights, coords_mm, k_grid, refine=refine, min_weight=min_weight)
    bx = float(fit["kx"])
    by = float(fit["ky"])
    k_norm = float(np.hypot(bx, by)) if np.isfinite(bx) and np.isfinite(by) else np.nan
    finite = np.isfinite(weights) & (weights > min_weight) & np.isfinite(phasor.real) & np.isfinite(phasor.imag)
    finite &= np.isfinite(coords_mm).all(axis=1)
    theta = np.angle(np.asarray(phasor, dtype=np.complex128))
    theta_hat = coords_mm @ np.array([bx, by], dtype=np.float64) if np.isfinite(k_norm) else np.full(theta.shape, np.nan)
    pgd_r2 = circular_linear_r2_adj(theta[finite], theta_hat[finite], n_params=3)
    pgd_align = phase_gradient_alignment(theta, coords_mm, gradient_edges) if gradient_edges is not None else np.nan
    return {
        "bx": bx,
        "by": by,
        "k_norm_pgd": k_norm,
        "pgd_r2_adj": float(pgd_r2),
        "pgd_gradient_alignment": float(pgd_align),
        "pgd": float(pgd_r2),
        "n_good": int(np.count_nonzero(finite)),
    }


def _radial_objective_score(center: np.ndarray, k_abs: float, sign: float, ubar: np.ndarray, w: np.ndarray, coords: np.ndarray) -> float:
    distances = np.linalg.norm(coords - np.asarray(center, dtype=np.float64)[None, :], axis=1)
    phase = float(sign) * float(k_abs) * distances
    return float(np.abs(np.sum(w * ubar * np.exp(-1j * phase))))


def fit_k_radial_phasor(
    ubar: np.ndarray,
    w: np.ndarray,
    coords: np.ndarray,
    radial_grid: dict[str, np.ndarray | float],
    *,
    refine: bool = True,
    min_weight: float = 1e-6,
) -> dict[str, float]:
    """Fit a source-centered radial phase wave by weighted circular alignment."""
    ubar_full = np.asarray(ubar, dtype=np.complex128)
    w_full = np.asarray(w, dtype=np.float64)
    coords_full = np.asarray(coords, dtype=np.float64)
    if coords_full.shape != (ubar_full.size, 2):
        raise ValueError("coords must have shape (n_channels, 2)")

    finite = np.isfinite(w_full) & (w_full > min_weight) & np.isfinite(ubar_full.real) & np.isfinite(ubar_full.imag)
    finite &= np.isfinite(coords_full).all(axis=1)
    if np.count_nonzero(finite) < 3:
        return {
            "radial_x0_mm": np.nan,
            "radial_y0_mm": np.nan,
            "radial_k": np.nan,
            "radial_sign": 0,
            "R_radial": np.nan,
            "radial_n_good": int(np.count_nonzero(finite)),
        }

    ubar = ubar_full[finite]
    w = w_full[finite]
    coords = coords_full[finite]
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        return {
            "radial_x0_mm": np.nan,
            "radial_y0_mm": np.nan,
            "radial_k": np.nan,
            "radial_sign": 0,
            "R_radial": np.nan,
            "radial_n_good": int(w.size),
        }

    centers = np.asarray(radial_grid["centers"], dtype=np.float64)
    k_values = np.asarray(radial_grid["k"], dtype=np.float64)
    k_min = float(radial_grid.get("k_min", np.nanmin(k_values) if k_values.size else 0.0))
    k_max = float(radial_grid.get("k_max", np.nanmax(k_values) if k_values.size else np.inf))
    if centers.size == 0 or k_values.size == 0:
        return {
            "radial_x0_mm": np.nan,
            "radial_y0_mm": np.nan,
            "radial_k": np.nan,
            "radial_sign": 0,
            "R_radial": np.nan,
            "radial_n_good": int(w.size),
        }

    weighted_ubar = w * ubar
    weighted_full = np.zeros(ubar_full.size, dtype=np.complex128)
    weighted_full[finite] = w_full[finite] * ubar_full[finite]
    best_score = -np.inf
    best_center = np.array([np.nan, np.nan], dtype=np.float64)
    best_k = np.nan
    best_sign = 0

    basis = radial_grid.get("basis")
    if isinstance(basis, np.ndarray) and basis.shape == (centers.shape[0], k_values.size, ubar_full.size):
        basis_flat = basis.reshape(centers.shape[0] * k_values.size, ubar_full.size)
        scores_pos = np.abs(basis_flat @ weighted_full)
        scores_neg = np.abs(np.conj(basis_flat) @ weighted_full)
        pos_idx = int(np.nanargmax(scores_pos))
        neg_idx = int(np.nanargmax(scores_neg))
        if float(scores_pos[pos_idx]) >= float(scores_neg[neg_idx]):
            best_score = float(scores_pos[pos_idx])
            flat_idx = pos_idx
            best_sign = 1
        else:
            best_score = float(scores_neg[neg_idx])
            flat_idx = neg_idx
            best_sign = -1
        center_idx, k_idx = np.unravel_index(flat_idx, (centers.shape[0], k_values.size))
        best_center = centers[int(center_idx)].astype(np.float64, copy=True)
        best_k = float(k_values[int(k_idx)])
    else:
        distances_grid = radial_grid.get("distances")
        if isinstance(distances_grid, np.ndarray) and distances_grid.shape == (centers.shape[0], ubar_full.size):
            distances_iter = distances_grid[:, finite].astype(np.float64, copy=False)
        else:
            distances_iter = None
        for center_idx, center in enumerate(centers):
            if distances_iter is None:
                distances = np.linalg.norm(coords - center[None, :], axis=1)
            else:
                distances = distances_iter[int(center_idx)]
            phase = k_values[:, None] * distances[None, :]
            center_basis = np.exp(-1j * phase)
            scores_pos = np.abs(center_basis @ weighted_ubar)
            scores_neg = np.abs(np.conj(center_basis) @ weighted_ubar)
            k_pos = int(np.nanargmax(scores_pos))
            k_neg = int(np.nanargmax(scores_neg))
            if float(scores_pos[k_pos]) > best_score:
                best_score = float(scores_pos[k_pos])
                best_center = center.astype(np.float64, copy=True)
                best_k = float(k_values[k_pos])
                best_sign = 1
            if float(scores_neg[k_neg]) > best_score:
                best_score = float(scores_neg[k_neg])
                best_center = center.astype(np.float64, copy=True)
                best_k = float(k_values[k_neg])
                best_sign = -1

    if refine and np.all(np.isfinite(best_center)) and np.isfinite(best_k):
        center_min = np.nanmin(centers, axis=0)
        center_max = np.nanmax(centers, axis=0)

        def neg_score(params: np.ndarray) -> float:
            center = np.asarray(params[:2], dtype=np.float64)
            k_abs = float(params[2])
            if k_abs < k_min or k_abs > k_max:
                return 1e9 + min(abs(k_abs - k_min), abs(k_abs - k_max)) * 1e6
            center_penalty = np.sum(np.maximum(center_min - center, 0.0) ** 2 + np.maximum(center - center_max, 0.0) ** 2)
            return -_radial_objective_score(center, k_abs, best_sign, ubar, w, coords) + float(center_penalty) * 1e6

        initial = np.array([best_center[0], best_center[1], best_k], dtype=np.float64)
        result = minimize(neg_score, initial, method="Nelder-Mead", options={"maxiter": 150, "xatol": 1e-5, "fatol": 1e-5})
        if result.success or np.isfinite(result.fun):
            candidate = np.asarray(result.x, dtype=np.float64)
            candidate_center = candidate[:2]
            candidate_k = float(candidate[2])
            if k_min <= candidate_k <= k_max:
                candidate_score = _radial_objective_score(candidate_center, candidate_k, best_sign, ubar, w, coords)
                if candidate_score >= best_score:
                    best_center = candidate_center
                    best_k = candidate_k
                    best_score = candidate_score

    return {
        "radial_x0_mm": float(best_center[0]),
        "radial_y0_mm": float(best_center[1]),
        "radial_k": float(best_k),
        "radial_sign": int(best_sign),
        "R_radial": float(best_score / w_sum),
        "radial_n_good": int(w.size),
    }


def classify_spatial_aperture(
    kx: float,
    ky: float,
    coords_mm: np.ndarray,
    *,
    near_sync_cycles: float = 0.25,
    resolvable_cycles: float = 0.5,
) -> dict[str, float | int]:
    """Classify whether a fitted wave spans enough phase across the array."""
    coords_mm = np.asarray(coords_mm, dtype=np.float64)
    if coords_mm.ndim != 2 or coords_mm.shape[1] != 2:
        raise ValueError("coords_mm must have shape (n_channels, 2)")
    k_norm = float(np.hypot(kx, ky)) if np.isfinite(kx) and np.isfinite(ky) else np.nan
    if not np.isfinite(k_norm) or k_norm <= 0 or coords_mm.shape[0] < 2:
        return {
            "aperture_along_k_mm": np.nan,
            "phase_span_rad": np.nan,
            "cycles_across_array": np.nan,
            "spatial_class": SPATIAL_CLASS["invalid"],
        }

    direction = np.array([float(kx) / k_norm, float(ky) / k_norm], dtype=np.float64)
    finite = np.isfinite(coords_mm).all(axis=1)
    if np.count_nonzero(finite) < 2:
        return {
            "aperture_along_k_mm": np.nan,
            "phase_span_rad": np.nan,
            "cycles_across_array": np.nan,
            "spatial_class": SPATIAL_CLASS["invalid"],
        }

    projected = coords_mm[finite] @ direction
    aperture = float(np.nanmax(projected) - np.nanmin(projected))
    phase_span = float(k_norm * aperture)
    cycles = float(phase_span / (2.0 * np.pi))
    if not np.isfinite(cycles):
        spatial_class = SPATIAL_CLASS["invalid"]
    elif cycles <= float(near_sync_cycles):
        spatial_class = SPATIAL_CLASS["near_sync"]
    elif cycles < float(resolvable_cycles):
        spatial_class = SPATIAL_CLASS["weak_gradient"]
    else:
        spatial_class = SPATIAL_CLASS["resolvable_wave"]

    return {
        "aperture_along_k_mm": aperture,
        "phase_span_rad": phase_span,
        "cycles_across_array": cycles,
        "spatial_class": spatial_class,
    }


def classify_radial_aperture(
    radial_x0_mm: float,
    radial_y0_mm: float,
    radial_k: float,
    coords_mm: np.ndarray,
) -> dict[str, float]:
    """Compute finite-aperture support for a radial wave fit."""
    coords_mm = np.asarray(coords_mm, dtype=np.float64)
    if (
        coords_mm.ndim != 2
        or coords_mm.shape[1] != 2
        or coords_mm.shape[0] < 2
        or not np.isfinite(radial_x0_mm)
        or not np.isfinite(radial_y0_mm)
        or not np.isfinite(radial_k)
        or radial_k == 0
    ):
        return {"radial_aperture_mm": np.nan, "radial_phase_span_rad": np.nan, "radial_cycles_across_array": np.nan}
    finite = np.isfinite(coords_mm).all(axis=1)
    if np.count_nonzero(finite) < 2:
        return {"radial_aperture_mm": np.nan, "radial_phase_span_rad": np.nan, "radial_cycles_across_array": np.nan}
    center = np.array([float(radial_x0_mm), float(radial_y0_mm)], dtype=np.float64)
    distances = np.linalg.norm(coords_mm[finite] - center[None, :], axis=1)
    aperture = float(np.nanmax(distances) - np.nanmin(distances))
    phase_span = float(abs(radial_k) * aperture)
    cycles = float(phase_span / (2.0 * np.pi))
    return {"radial_aperture_mm": aperture, "radial_phase_span_rad": phase_span, "radial_cycles_across_array": cycles}


def classify_wave_model(
    *,
    planar_valid: bool,
    radial_valid: bool,
    spatial_class: int,
    radial_cycles_across_array: float,
    delta_r: float,
    delta_r_min: float = 0.05,
) -> int:
    """Classify the best-supported wave model for a fitted window."""
    if not planar_valid and not radial_valid:
        if int(spatial_class) in {SPATIAL_CLASS["near_sync"], SPATIAL_CLASS["weak_gradient"]} or (
            np.isfinite(radial_cycles_across_array) and radial_cycles_across_array < 0.5
        ):
            return WAVE_MODEL_CLASS["near_sync"]
        return WAVE_MODEL_CLASS["invalid"]
    if radial_valid and np.isfinite(delta_r) and delta_r >= float(delta_r_min):
        return WAVE_MODEL_CLASS["radial_like"]
    if planar_valid and ((np.isfinite(delta_r) and delta_r <= -float(delta_r_min)) or not radial_valid):
        return WAVE_MODEL_CLASS["planar_like"]
    if radial_valid and not planar_valid:
        return WAVE_MODEL_CLASS["radial_like"]
    return WAVE_MODEL_CLASS["ambiguous"]


def compute_phase_velocity(kx: float, ky: float, omega: float, f_peak: float = np.nan) -> dict[str, float]:
    """Compute phase speed and propagation vector from the fitted phase plane.

    The fitted/demodulated convention is theta(r, t) ~= k dot r + omega t + phi.
    Constant-phase contours therefore move with vector velocity -omega * k / |k|^2.
    """
    k_norm = float(np.hypot(kx, ky)) if np.isfinite(kx) and np.isfinite(ky) else np.nan
    if not np.isfinite(k_norm) or k_norm <= 0 or not np.isfinite(omega):
        return {
            "v_phi_mm_per_s": np.nan,
            "phase_speed_mm_per_s": np.nan,
            "phase_speed_peak_mm_per_s": np.nan,
            "velocity_x_mm_per_s": np.nan,
            "velocity_y_mm_per_s": np.nan,
            "velocity_direction_rad": np.nan,
        }

    signed_v_phi = float(omega / k_norm)
    phase_speed = float(abs(omega) / k_norm)
    velocity_x = float(-omega * kx / (k_norm * k_norm))
    velocity_y = float(-omega * ky / (k_norm * k_norm))
    velocity_direction = float(np.arctan2(velocity_y, velocity_x))
    phase_speed_peak = np.nan
    if np.isfinite(f_peak) and f_peak >= 0:
        phase_speed_peak = float(2.0 * np.pi * f_peak / k_norm)

    return {
        "v_phi_mm_per_s": signed_v_phi,
        "phase_speed_mm_per_s": phase_speed,
        "phase_speed_peak_mm_per_s": phase_speed_peak,
        "velocity_x_mm_per_s": velocity_x,
        "velocity_y_mm_per_s": velocity_y,
        "velocity_direction_rad": velocity_direction,
    }


def compute_radial_phase_speed(radial_k: float, omega: float) -> float:
    """Return scalar radial phase speed abs(omega)/abs(k)."""
    if not np.isfinite(radial_k) or radial_k == 0 or not np.isfinite(omega):
        return np.nan
    return float(abs(omega) / abs(radial_k))


def wavelength_edge_flags(lambda_mm: float, lambda_min_mm: float, lambda_max_mm: float, edge_fraction: float = 0.98) -> dict[str, int]:
    """Flag wavelengths close to search bounds."""
    if not np.isfinite(lambda_mm) or not np.isfinite(lambda_min_mm) or not np.isfinite(lambda_max_mm):
        return {"at_lambda_min": 0, "at_lambda_max": 0, "speed_censored": 0}
    edge_fraction = float(edge_fraction)
    at_min = int(lambda_mm <= (float(lambda_min_mm) / edge_fraction))
    at_max = int(lambda_mm >= (edge_fraction * float(lambda_max_mm)))
    return {"at_lambda_min": at_min, "at_lambda_max": at_max, "speed_censored": at_max}


def estimate_omega(
    u_window: np.ndarray,
    data_window: np.ndarray | None,
    fs_ds: float,
    *,
    method: str = "channel_phase_derivative",
    band_low: float | None = None,
    band_high: float | None = None,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Estimate angular frequency and peak frequency for one time window."""
    u_window = np.asarray(u_window)
    if u_window.shape[0] < 2:
        return np.nan, np.nan

    if weights is None:
        umean = np.nanmean(u_window, axis=1)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        denom = float(np.nansum(weights))
        umean = np.sum(u_window * weights[None, :], axis=1) / denom if denom > 0 else np.nanmean(u_window, axis=1)

    omega_phase = np.nan
    if method in {"channel_phase_derivative", "both"}:
        theta = np.unwrap(np.angle(u_window), axis=0)
        dtheta = np.diff(theta, axis=0) * float(fs_ds)
        channel_omega = np.nanmean(dtheta, axis=0)
        channel_weight = np.nanmean(np.abs(u_window[:-1]), axis=0)
        if weights is not None:
            channel_weight = channel_weight * np.asarray(weights, dtype=np.float64)
        good = np.isfinite(channel_omega) & np.isfinite(channel_weight) & (channel_weight > 0)
        if np.count_nonzero(good) >= 1:
            omega_phase = float(np.average(channel_omega[good], weights=channel_weight[good]))

    if method in {"phase_derivative", "mean_phase_derivative", "both"} and not np.isfinite(omega_phase):
        good = np.abs(umean) > 1e-12
        if np.count_nonzero(good) >= 2:
            theta = np.unwrap(np.angle(umean[good]))
            dt = 1.0 / float(fs_ds)
            omega_phase = float(np.nanmean(np.diff(theta) / dt))

    f_peak = np.nan
    if data_window is not None:
        spatial_mean = np.nanmean(np.asarray(data_window), axis=1)
        if spatial_mean.size >= 4 and np.any(np.isfinite(spatial_mean)):
            nperseg = min(spatial_mean.size, max(8, int(round(fs_ds * 0.25))))
            freqs, psd = welch(np.nan_to_num(spatial_mean), fs=fs_ds, nperseg=nperseg)
            mask = np.ones_like(freqs, dtype=bool)
            if band_low is not None:
                mask &= freqs >= float(band_low)
            if band_high is not None:
                mask &= freqs <= float(band_high)
            if np.any(mask):
                masked_freqs = freqs[mask]
                masked_psd = psd[mask]
                if masked_psd.size:
                    f_peak = float(masked_freqs[int(np.nanargmax(masked_psd))])

    if method == "psd_peak" and np.isfinite(f_peak):
        return float(2.0 * np.pi * f_peak), f_peak
    if np.isfinite(omega_phase):
        return omega_phase, f_peak
    if np.isfinite(f_peak):
        return float(2.0 * np.pi * f_peak), f_peak
    return np.nan, f_peak


def demodulated_window_phasor(
    u_window: np.ndarray,
    amp_window: np.ndarray,
    fs_ds: float,
    omega: float,
) -> np.ndarray:
    """Average phasors after removing within-window temporal phase rotation."""
    u_window = np.asarray(u_window)
    amp_window = np.asarray(amp_window)
    if np.isfinite(omega):
        t = (np.arange(u_window.shape[0], dtype=np.float64) - 0.5 * (u_window.shape[0] - 1)) / float(fs_ds)
        u_window = u_window * np.exp(-1j * omega * t)[:, None]
    amp_sum = np.nansum(amp_window, axis=0)
    return np.divide(
        np.nansum(amp_window * u_window, axis=0),
        amp_sum,
        out=np.zeros(u_window.shape[1], dtype=np.complex64),
        where=amp_sum > 1e-12,
    )


def _init_output(path: str | Path, config: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        group = h5.create_group("phase_waves")
        integer_columns = {
            "n_good",
            "valid",
            "interval_idx",
            "spatial_class",
            "propagation_valid",
            "planar_at_lambda_min",
            "planar_at_lambda_max",
            "planar_speed_censored",
            "planar_velocity_valid",
            "radial_sign",
            "wave_model_class",
            "radial_valid",
            "radial_at_lambda_min",
            "radial_at_lambda_max",
            "radial_speed_censored",
            "radial_velocity_valid",
        }
        for name in RESULT_COLUMNS:
            dtype = "i8" if name in integer_columns else "f8"
            group.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(1024,), dtype=dtype)
        group.attrs["columns"] = json.dumps(RESULT_COLUMNS)
        group.attrs["config"] = json.dumps(config, default=str)
        group.attrs["spatial_class_mapping"] = json.dumps(SPATIAL_CLASS)
        group.attrs["wave_model_class_mapping"] = json.dumps(WAVE_MODEL_CLASS)
        group.attrs["spatial_class_thresholds"] = json.dumps(
            {"near_sync_cycles_lte": 0.25, "resolvable_wave_cycles_gte": 0.5}
        )
        group.attrs["velocity_sign_convention"] = (
            "The fitted phase convention is theta(r,t) ~= k dot r + omega t + phi; "
            "v_phi_mm_per_s is signed omega/|k|, phase_speed_mm_per_s is abs(omega)/|k|, "
            "and velocity vector is -omega*k/|k|^2."
        )
        group.attrs["radial_velocity_sign_convention"] = (
            "The radial phase convention is theta(r,t) ~= radial_sign*radial_k*|r-r0| + omega*t + phi; "
            "radial_phase_speed_mm_per_s is abs(omega)/abs(radial_k), and local velocity is "
            "-omega*radial_sign*(r-r0)/(|r-r0|*abs(radial_k))."
        )
        group.attrs["units"] = json.dumps(
            {
                "t_center_s": "s",
                "kx": "rad/mm",
                "ky": "rad/mm",
                "k_norm": "rad/mm",
                "lambda_mm": "mm",
                "direction_x": "unitless",
                "direction_y": "unitless",
                "direction_rad": "rad",
                "aperture_along_k_mm": "mm",
                "phase_span_rad": "rad",
                "cycles_across_array": "cycles",
                "omega": "rad/s",
                "f_peak": "Hz",
                "v_phi_mm_per_s": "mm/s",
                "phase_speed_mm_per_s": "mm/s",
                "phase_speed_peak_mm_per_s": "mm/s",
                "velocity_x_mm_per_s": "mm/s",
                "velocity_y_mm_per_s": "mm/s",
                "velocity_direction_rad": "rad",
                "planar_at_lambda_min": "bool",
                "planar_at_lambda_max": "bool",
                "planar_speed_censored": "bool",
                "planar_velocity_valid": "bool",
                "radial_x0_mm": "mm",
                "radial_y0_mm": "mm",
                "radial_k": "rad/mm",
                "radial_lambda_mm": "mm",
                "radial_sign": "unitless",
                "radial_aperture_mm": "mm",
                "radial_phase_span_rad": "rad",
                "radial_cycles_across_array": "cycles",
                "R_radial": "unitless",
                "delta_R_radial_minus_planar": "unitless",
                "radial_phase_speed_mm_per_s": "mm/s",
                "radial_at_lambda_min": "bool",
                "radial_at_lambda_max": "bool",
                "radial_speed_censored": "bool",
                "radial_velocity_valid": "bool",
            }
        )


def _append_rows(path: str | Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with h5py.File(path, "a") as h5:
        group = h5["phase_waves"]
        start = int(group[RESULT_COLUMNS[0]].shape[0])
        stop = start + len(rows)
        for name in RESULT_COLUMNS:
            ds = group[name]
            ds.resize((stop,))
            ds[start:stop] = np.asarray([row.get(name, np.nan) for row in rows], dtype=ds.dtype)


def _init_pgd_output(path: str | Path, config: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as h5:
        if "pgd_waves" in h5:
            del h5["pgd_waves"]
        group = h5.create_group("pgd_waves")
        integer_columns = {"pgd_valid", "pgd_sustained", "pgd_direction_stable", "interval_idx", "n_good"}
        for name in PGD_COLUMNS:
            dtype = "i8" if name in integer_columns else "f8"
            group.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(4096,), dtype=dtype)
        group.attrs["columns"] = json.dumps(PGD_COLUMNS)
        group.attrs["config"] = json.dumps(config, default=str)
        group.attrs["pgd_convention"] = (
            "pgd is pgd_r2_adj: adjusted circular-linear rho^2 between instantaneous phase and fitted planar phase. "
            "pgd_gradient_alignment is the neighbor-edge gradient-alignment PGD."
        )
        group.attrs["validity_criteria"] = json.dumps(
            {
                "pgd_threshold": config.get("pgd_threshold"),
                "min_duration_ms": config.get("pgd_min_duration_ms"),
                "max_direction_change_deg_per_ms": config.get("pgd_max_direction_change_deg_per_ms"),
            }
        )
        group.attrs["direction_convention"] = (
            "bx/by are phase-gradient slopes in rad/mm. direction_pgd_rad is propagation direction, "
            "defined as atan2(-by, -bx), i.e. opposite the phase-gradient vector."
        )
        group.attrs["units"] = json.dumps(
            {
                "t_center_s": "s",
                "pgd": "unitless",
                "pgd_r2_adj": "unitless",
                "pgd_gradient_alignment": "unitless",
                "bx": "rad/mm",
                "by": "rad/mm",
                "k_norm_pgd": "rad/mm",
                "direction_pgd_rad": "rad",
                "speed_pgd_mm_per_s": "mm/s",
                "omega_pgd": "rad/s",
                "frequency_hz": "Hz",
                "pgd_cycles_across_array": "cycles",
                "pgd_aperture_along_k_mm": "mm",
            }
        )


def _append_pgd_rows(path: str | Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with h5py.File(path, "a") as h5:
        group = h5["pgd_waves"]
        start = int(group[PGD_COLUMNS[0]].shape[0])
        stop = start + len(rows)
        for name in PGD_COLUMNS:
            ds = group[name]
            ds.resize((stop,))
            ds[start:stop] = np.asarray([row.get(name, np.nan) for row in rows], dtype=ds.dtype)


def _sustained_boolean_mask(candidate: np.ndarray, min_samples: int) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=bool)
    if candidate.size == 0:
        return candidate
    min_samples = max(1, int(min_samples))
    out = np.zeros(candidate.size, dtype=bool)
    starts = np.flatnonzero(candidate & np.r_[True, ~candidate[:-1]])
    stops = np.flatnonzero(candidate & np.r_[~candidate[1:], True]) + 1
    for start, stop in zip(starts, stops):
        if stop - start >= min_samples:
            out[start:stop] = True
    return out


def _direction_stability_mask(direction_rad: np.ndarray, fs_ds: float, max_deg_per_ms: float) -> np.ndarray:
    direction_rad = np.asarray(direction_rad, dtype=np.float64)
    if direction_rad.size == 0:
        return np.zeros(0, dtype=bool)
    if direction_rad.size == 1:
        return np.isfinite(direction_rad)
    delta = np.abs(np.angle(np.exp(1j * np.diff(direction_rad))))
    deg_per_ms = np.rad2deg(delta) * float(fs_ds) / 1000.0
    prev_ok = np.ones(direction_rad.size, dtype=bool)
    next_ok = np.ones(direction_rad.size, dtype=bool)
    prev_ok[1:] = deg_per_ms <= float(max_deg_per_ms)
    next_ok[:-1] = deg_per_ms <= float(max_deg_per_ms)
    return np.isfinite(direction_rad) & prev_ok & next_ok


def _window_rows(
    data_ds: np.ndarray,
    u: np.ndarray,
    amp: np.ndarray,
    coords_mm: np.ndarray,
    *,
    fs_ds: float,
    chunk_start_s: float,
    keep_start_s: float,
    keep_stop_s: float,
    win_ms: float,
    hop_ms: float,
    k_grid: dict[str, np.ndarray | float],
    radial_grid: dict[str, np.ndarray | float] | None,
    refine: bool,
    fit_radial: bool,
    band_low: float,
    band_high: float,
    r_min: float,
    coherence_min: float,
    radial_delta_r_min: float,
    speed_min_cycles: float,
    lambda_edge_fraction: float,
    speed_min_r: float,
    lambda_min: float,
    lambda_max: float,
) -> list[dict[str, float]]:
    win_n = max(2, int(round(float(win_ms) * fs_ds / 1000.0)))
    hop_n = max(1, int(round(float(hop_ms) * fs_ds / 1000.0)))
    rows: list[dict[str, float]] = []
    for start in range(0, max(0, u.shape[0] - win_n + 1), hop_n):
        stop = start + win_n
        center_s = chunk_start_s + ((start + stop) * 0.5 / fs_ds)
        if center_s < keep_start_s or center_s >= keep_stop_s:
            continue

        u_win = u[start:stop]
        amp_win = amp[start:stop]
        amp_weights = np.nanmean(amp_win, axis=0)
        omega, f_peak = estimate_omega(
            u_win,
            data_ds[start:stop],
            fs_ds,
            method="channel_phase_derivative",
            band_low=band_low,
            band_high=band_high,
            weights=amp_weights,
        )
        ubar = demodulated_window_phasor(u_win, amp_win, fs_ds, omega)
        weights = np.abs(ubar).astype(np.float64)
        mean_weight = float(np.nanmean(weights)) if weights.size else np.nan
        fit = fit_k_phasor_plane(ubar, weights, coords_mm, k_grid, refine=refine)
        kx = fit["kx"]
        ky = fit["ky"]
        k_norm = float(np.hypot(kx, ky)) if np.isfinite(kx) and np.isfinite(ky) else np.nan
        lambda_mm = float(2.0 * np.pi / k_norm) if np.isfinite(k_norm) and k_norm > 0 else np.nan
        direction_x = float(kx / k_norm) if np.isfinite(k_norm) and k_norm > 0 else np.nan
        direction_y = float(ky / k_norm) if np.isfinite(k_norm) and k_norm > 0 else np.nan
        direction_rad = float(np.arctan2(ky, kx)) if np.isfinite(kx) and np.isfinite(ky) else np.nan
        velocity = compute_phase_velocity(kx, ky, omega, f_peak)
        valid = int(np.isfinite(fit["R_fit"]) and fit["R_fit"] >= r_min and np.isfinite(mean_weight) and mean_weight >= coherence_min)
        aperture = classify_spatial_aperture(kx, ky, coords_mm)
        propagation_valid = int(valid and int(aperture["spatial_class"]) == SPATIAL_CLASS["resolvable_wave"])
        planar_edges = wavelength_edge_flags(lambda_mm, lambda_min, lambda_max, lambda_edge_fraction)
        planar_velocity_valid = int(
            bool(propagation_valid)
            and not planar_edges["at_lambda_min"]
            and not planar_edges["at_lambda_max"]
            and np.isfinite(aperture["cycles_across_array"])
            and aperture["cycles_across_array"] >= speed_min_cycles
            and np.isfinite(fit["R_fit"])
            and fit["R_fit"] >= speed_min_r
        )
        if fit_radial and radial_grid is not None:
            radial_fit = fit_k_radial_phasor(ubar, weights, coords_mm, radial_grid, refine=refine)
        else:
            radial_fit = {
                "radial_x0_mm": np.nan,
                "radial_y0_mm": np.nan,
                "radial_k": np.nan,
                "radial_sign": 0,
                "R_radial": np.nan,
            }
        radial_aperture = classify_radial_aperture(
            radial_fit["radial_x0_mm"],
            radial_fit["radial_y0_mm"],
            radial_fit["radial_k"],
            coords_mm,
        )
        radial_lambda_mm = (
            float(2.0 * np.pi / abs(radial_fit["radial_k"]))
            if np.isfinite(radial_fit["radial_k"]) and radial_fit["radial_k"] != 0
            else np.nan
        )
        delta_r = (
            float(radial_fit["R_radial"] - fit["R_fit"])
            if np.isfinite(radial_fit["R_radial"]) and np.isfinite(fit["R_fit"])
            else np.nan
        )
        radial_valid = int(
            np.isfinite(radial_fit["R_radial"])
            and radial_fit["R_radial"] >= r_min
            and np.isfinite(mean_weight)
            and mean_weight >= coherence_min
            and np.isfinite(radial_aperture["radial_cycles_across_array"])
            and radial_aperture["radial_cycles_across_array"] >= 0.5
        )
        radial_edges = wavelength_edge_flags(radial_lambda_mm, lambda_min, lambda_max, lambda_edge_fraction)
        radial_velocity_valid = int(
            bool(radial_valid)
            and not radial_edges["at_lambda_min"]
            and not radial_edges["at_lambda_max"]
            and np.isfinite(radial_aperture["radial_cycles_across_array"])
            and radial_aperture["radial_cycles_across_array"] >= speed_min_cycles
            and np.isfinite(radial_fit["R_radial"])
            and radial_fit["R_radial"] >= speed_min_r
            and np.isfinite(delta_r)
            and delta_r >= radial_delta_r_min
        )
        wave_model_class = classify_wave_model(
            planar_valid=bool(propagation_valid),
            radial_valid=bool(radial_valid),
            spatial_class=int(aperture["spatial_class"]),
            radial_cycles_across_array=float(radial_aperture["radial_cycles_across_array"]),
            delta_r=delta_r,
            delta_r_min=radial_delta_r_min,
        )
        rows.append(
            {
                "t_center_s": float(center_s),
                "kx": float(kx),
                "ky": float(ky),
                "k_norm": k_norm,
                "lambda_mm": lambda_mm,
                "direction_x": direction_x,
                "direction_y": direction_y,
                "direction_rad": direction_rad,
                "aperture_along_k_mm": float(aperture["aperture_along_k_mm"]),
                "phase_span_rad": float(aperture["phase_span_rad"]),
                "cycles_across_array": float(aperture["cycles_across_array"]),
                "spatial_class": int(aperture["spatial_class"]),
                "propagation_valid": propagation_valid,
                "omega": float(omega),
                "f_peak": float(f_peak),
                **velocity,
                "planar_at_lambda_min": int(planar_edges["at_lambda_min"]),
                "planar_at_lambda_max": int(planar_edges["at_lambda_max"]),
                "planar_speed_censored": int(planar_edges["speed_censored"]),
                "planar_velocity_valid": planar_velocity_valid,
                "radial_x0_mm": float(radial_fit["radial_x0_mm"]),
                "radial_y0_mm": float(radial_fit["radial_y0_mm"]),
                "radial_k": float(radial_fit["radial_k"]),
                "radial_lambda_mm": radial_lambda_mm,
                "radial_sign": int(radial_fit["radial_sign"]),
                "radial_aperture_mm": float(radial_aperture["radial_aperture_mm"]),
                "radial_phase_span_rad": float(radial_aperture["radial_phase_span_rad"]),
                "radial_cycles_across_array": float(radial_aperture["radial_cycles_across_array"]),
                "R_radial": float(radial_fit["R_radial"]),
                "delta_R_radial_minus_planar": delta_r,
                "radial_phase_speed_mm_per_s": compute_radial_phase_speed(radial_fit["radial_k"], omega),
                "radial_at_lambda_min": int(radial_edges["at_lambda_min"]),
                "radial_at_lambda_max": int(radial_edges["at_lambda_max"]),
                "radial_speed_censored": int(radial_edges["speed_censored"]),
                "radial_velocity_valid": radial_velocity_valid,
                "wave_model_class": int(wave_model_class),
                "radial_valid": radial_valid,
                "R_fit": float(fit["R_fit"]),
                "mean_weight": mean_weight,
                "n_good": int(fit["n_good"]),
                "valid": valid,
            }
        )
    return rows


def _instantaneous_omega(u: np.ndarray, amp: np.ndarray, fs_ds: float) -> np.ndarray:
    """Weighted instantaneous angular velocity per sample from channel phase derivatives."""
    u = np.asarray(u)
    amp = np.asarray(amp, dtype=np.float64)
    if u.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if u.shape[0] == 1:
        return np.full(1, np.nan, dtype=np.float64)
    theta = np.unwrap(np.angle(u), axis=0)
    omega_channel = np.gradient(theta, 1.0 / float(fs_ds), axis=0)
    good = np.isfinite(omega_channel) & np.isfinite(amp) & (amp > 0)
    numerator = np.nansum(np.where(good, omega_channel * amp, 0.0), axis=1)
    denominator = np.nansum(np.where(good, amp, 0.0), axis=1)
    return np.divide(numerator, denominator, out=np.full(u.shape[0], np.nan, dtype=np.float64), where=denominator > 0)


def _pgd_rows(
    u: np.ndarray,
    amp: np.ndarray,
    coords_mm: np.ndarray,
    *,
    fs_ds: float,
    chunk_start_s: float,
    keep_start_s: float,
    keep_stop_s: float,
    frequency_hz: float,
    k_grid: dict[str, np.ndarray | float],
    gradient_edges: np.ndarray,
    pgd_threshold: float,
    pgd_min_duration_ms: float,
    pgd_max_direction_change_deg_per_ms: float,
    refine: bool,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if u.shape[0] == 0:
        return rows
    omega = _instantaneous_omega(u, amp, fs_ds)
    sample_rows: list[dict[str, float]] = []
    for sample_idx in range(u.shape[0]):
        center_s = float(chunk_start_s + sample_idx / float(fs_ds))
        if center_s < keep_start_s or center_s >= keep_stop_s:
            continue
        weights = amp[sample_idx].astype(np.float64, copy=False)
        fit = fit_pgd_plane(
            u[sample_idx],
            weights,
            coords_mm,
            k_grid,
            gradient_edges=gradient_edges,
            refine=refine,
        )
        bx = fit["bx"]
        by = fit["by"]
        k_norm = fit["k_norm_pgd"]
        aperture = classify_spatial_aperture(bx, by, coords_mm)
        omega_i = float(omega[sample_idx]) if sample_idx < omega.size else np.nan
        speed = float(abs(omega_i) / k_norm) if np.isfinite(omega_i) and np.isfinite(k_norm) and k_norm > 0 else np.nan
        sample_rows.append(
            {
                "t_center_s": center_s,
                "pgd": float(fit["pgd"]),
                "pgd_r2_adj": float(fit["pgd_r2_adj"]),
                "pgd_gradient_alignment": float(fit["pgd_gradient_alignment"]),
                "bx": float(bx),
                "by": float(by),
                "k_norm_pgd": float(k_norm),
                "direction_pgd_rad": float(np.arctan2(-by, -bx)) if np.isfinite(bx) and np.isfinite(by) else np.nan,
                "speed_pgd_mm_per_s": speed,
                "omega_pgd": omega_i,
                "frequency_hz": float(frequency_hz),
                "pgd_valid": 0,
                "pgd_sustained": 0,
                "pgd_direction_stable": 0,
                "pgd_cycles_across_array": float(aperture["cycles_across_array"]),
                "pgd_aperture_along_k_mm": float(aperture["aperture_along_k_mm"]),
                "n_good": int(fit["n_good"]),
            }
        )
    if not sample_rows:
        return rows

    pgd_values = np.asarray([row["pgd"] for row in sample_rows], dtype=np.float64)
    k_norm_values = np.asarray([row["k_norm_pgd"] for row in sample_rows], dtype=np.float64)
    directions = np.asarray([row["direction_pgd_rad"] for row in sample_rows], dtype=np.float64)
    candidate = np.isfinite(pgd_values) & (pgd_values >= float(pgd_threshold)) & np.isfinite(k_norm_values) & (k_norm_values > 0)
    stable = _direction_stability_mask(directions, fs_ds, pgd_max_direction_change_deg_per_ms)
    sustained = _sustained_boolean_mask(candidate & stable, int(np.ceil(float(pgd_min_duration_ms) * float(fs_ds) / 1000.0)))
    for idx, row in enumerate(sample_rows):
        row["pgd_direction_stable"] = int(stable[idx])
        row["pgd_sustained"] = int(sustained[idx])
        row["pgd_valid"] = int(candidate[idx] and stable[idx] and sustained[idx])
        rows.append(row)
    return rows


def process_pgd_file(
    file_path: str | Path,
    dataset: str,
    out_path: str | Path,
    *,
    fs_raw: float | None = None,
    band_low: float = 30.0,
    band_high: float = 80.0,
    fs_ds: float = 500.0,
    chunk_s: float = 30.0,
    overlap_s: float = 1.0,
    start_s: float = 0.0,
    stop_s: float | None = None,
    intervals: Iterable[tuple[float, float]] | None = None,
    analysis_pad_s: float = 0.05,
    k_grid_n: int = 41,
    lambda_min: float = 0.5,
    lambda_max: float = 10.0,
    pgd_threshold: float = 0.5,
    pgd_min_duration_ms: float = 5.0,
    pgd_max_direction_change_deg_per_ms: float = 3.0,
    gradient_n_neighbors: int = 6,
    gradient_max_distance_mm: float | None = None,
    refine: bool = False,
    max_channels_per_block: int = 128,
) -> Path:
    """Stream instantaneous planar PGD metrics into ``/pgd_waves`` of an HDF5 file."""
    info = inspect_file(file_path, dataset)
    inferred_fs = float(info["fs_raw"])
    fs = inferred_fs if fs_raw is None or float(fs_raw) <= 0 else float(fs_raw)
    raw_path, _ = _resolve_dataset_paths(dataset, file_path)
    total_frames = int(info["raw_shape"][1])
    recording_stop_s = total_frames / fs
    if intervals is None:
        interval_list = [(float(start_s), recording_stop_s if stop_s is None else float(stop_s))]
    else:
        interval_list = [(float(a), float(b)) for a, b in intervals if np.isfinite(a) and np.isfinite(b) and float(b) > float(a)]
        if stop_s is not None:
            interval_list = [(max(float(start_s), a), min(float(stop_s), b)) for a, b in interval_list]
        else:
            interval_list = [(max(float(start_s), a), b) for a, b in interval_list]
        interval_list = [(max(0.0, a), min(recording_stop_s, b)) for a, b in interval_list if b > a]
    chunk_frames = max(1, int(round(float(chunk_s) * fs)))
    overlap_frames = max(0, int(round(float(overlap_s) * fs)))
    pad_frames = max(0, int(round(float(analysis_pad_s) * fs)))
    if overlap_frames * 2 >= chunk_frames:
        raise ValueError("overlap_s must be less than half of chunk_s")

    config = {
        "file": str(file_path),
        "dataset": dataset,
        "raw_path": raw_path,
        "fs_raw": fs,
        "band_low": band_low,
        "band_high": band_high,
        "fs_ds": fs_ds,
        "chunk_s": chunk_s,
        "overlap_s": overlap_s,
        "start_s": start_s,
        "stop_s": stop_s,
        "intervals": interval_list,
        "analysis_pad_s": analysis_pad_s,
        "k_grid_n": k_grid_n,
        "lambda_min_mm": lambda_min,
        "lambda_max_mm": lambda_max,
        "pgd_threshold": pgd_threshold,
        "pgd_min_duration_ms": pgd_min_duration_ms,
        "pgd_max_direction_change_deg_per_ms": pgd_max_direction_change_deg_per_ms,
        "gradient_n_neighbors": gradient_n_neighbors,
        "gradient_max_distance_mm": gradient_max_distance_mm,
        "refine": refine,
    }
    _init_pgd_output(out_path, config)
    k_grid = make_k_grid(lambda_min, lambda_max, n_grid=k_grid_n)
    k_grid_prepared = False
    gradient_edges = None
    frequency_hz = 0.5 * (float(band_low) + float(band_high))

    for interval_idx, (interval_start_s, interval_stop_s) in enumerate(interval_list):
        core_start = int(round(interval_start_s * fs))
        stop_frame_global = int(round(interval_stop_s * fs))
        while core_start < stop_frame_global:
            core_stop = min(stop_frame_global, core_start + chunk_frames)
            read_start = max(0, core_start - overlap_frames - pad_frames)
            read_stop = min(total_frames, core_stop + overlap_frames + pad_frames)
            data, coords_mm, fs_loaded, _ = load_chunk(file_path, dataset, read_start, read_stop - read_start)
            if abs(float(fs_loaded) - fs) > 1e-6:
                raise ValueError(f"fs_raw={fs} does not match file sampling={fs_loaded}")
            if not k_grid_prepared:
                k_grid = prepare_k_grid_basis(k_grid, coords_mm)
                gradient_edges = make_neighbor_edges(
                    coords_mm,
                    n_neighbors=gradient_n_neighbors,
                    max_distance_mm=gradient_max_distance_mm,
                )
                k_grid_prepared = True
            data_ds = bandpass_downsample(
                data,
                fs,
                band_low=band_low,
                band_high=band_high,
                fs_ds=fs_ds,
                max_channels_per_block=max_channels_per_block,
            )
            u, amp = extract_phasors(data_ds)
            rows = _pgd_rows(
                u,
                amp,
                coords_mm,
                fs_ds=fs_ds,
                chunk_start_s=read_start / fs,
                keep_start_s=core_start / fs,
                keep_stop_s=core_stop / fs,
                frequency_hz=frequency_hz,
                k_grid=k_grid,
                gradient_edges=gradient_edges if gradient_edges is not None else np.empty((0, 2), dtype=np.int64),
                pgd_threshold=pgd_threshold,
                pgd_min_duration_ms=pgd_min_duration_ms,
                pgd_max_direction_change_deg_per_ms=pgd_max_direction_change_deg_per_ms,
                refine=refine,
            )
            for row in rows:
                row["interval_idx"] = int(interval_idx)
            _append_pgd_rows(out_path, rows)
            core_start = core_stop
    return Path(out_path)


def process_file(
    file_path: str | Path,
    dataset: str,
    out_path: str | Path,
    *,
    fs_raw: float | None = None,
    band_low: float = 30.0,
    band_high: float = 80.0,
    fs_ds: float = 500.0,
    chunk_s: float = 30.0,
    overlap_s: float = 1.0,
    win_ms: float = 100.0,
    hop_ms: float = 50.0,
    lambda_min: float = 0.5,
    lambda_max: float = 5.0,
    k_grid_n: int = 41,
    refine: bool = True,
    fit_radial: bool = False,
    radial_center_grid_n: int = 15,
    radial_center_margin_mm: float = 0.0,
    radial_delta_r_min: float = 0.05,
    speed_min_cycles: float = 1.0,
    lambda_edge_fraction: float = 0.98,
    speed_min_r: float = 0.18,
    r_min: float = 0.2,
    coherence_min: float = 0.0,
    start_s: float = 0.0,
    stop_s: float | None = None,
    intervals: Iterable[tuple[float, float]] | None = None,
    max_channels_per_block: int = 128,
) -> Path:
    """Stream a MaxTwo LFP recording and append phase-wave windows to HDF5."""
    info = inspect_file(file_path, dataset)
    inferred_fs = float(info["fs_raw"])
    fs = inferred_fs if fs_raw is None or float(fs_raw) <= 0 else float(fs_raw)
    raw_path, _ = _resolve_dataset_paths(dataset, file_path)
    total_frames = int(info["raw_shape"][1])
    recording_stop_s = total_frames / fs
    if intervals is None:
        interval_list = [(float(start_s), recording_stop_s if stop_s is None else float(stop_s))]
    else:
        interval_list = [(float(a), float(b)) for a, b in intervals if np.isfinite(a) and np.isfinite(b) and float(b) > float(a)]
        if stop_s is not None:
            interval_list = [(max(float(start_s), a), min(float(stop_s), b)) for a, b in interval_list]
        else:
            interval_list = [(max(float(start_s), a), b) for a, b in interval_list]
        interval_list = [(max(0.0, a), min(recording_stop_s, b)) for a, b in interval_list if b > a]
    chunk_frames = max(1, int(round(float(chunk_s) * fs)))
    overlap_frames = max(0, int(round(float(overlap_s) * fs)))
    if overlap_frames * 2 >= chunk_frames:
        raise ValueError("overlap_s must be less than half of chunk_s")

    config = {
        "file": str(file_path),
        "dataset": dataset,
        "raw_path": raw_path,
        "fs_raw": fs,
        "band_low": band_low,
        "band_high": band_high,
        "fs_ds": fs_ds,
        "chunk_s": chunk_s,
        "overlap_s": overlap_s,
        "win_ms": win_ms,
        "hop_ms": hop_ms,
        "lambda_min_mm": lambda_min,
        "lambda_max_mm": lambda_max,
        "k_grid_n": k_grid_n,
        "fit_radial": fit_radial,
        "radial_center_grid_n": radial_center_grid_n,
        "radial_center_margin_mm": radial_center_margin_mm,
        "radial_delta_r_min": radial_delta_r_min,
        "speed_min_cycles": speed_min_cycles,
        "lambda_edge_fraction": lambda_edge_fraction,
        "speed_min_r": speed_min_r,
        "r_min": r_min,
        "coherence_min": coherence_min,
        "start_s": start_s,
        "stop_s": stop_s,
        "intervals": interval_list,
    }
    _init_output(out_path, config)
    k_grid = make_k_grid(lambda_min, lambda_max, n_grid=k_grid_n)
    radial_grid = None
    k_grid_prepared = False

    for interval_idx, (interval_start_s, interval_stop_s) in enumerate(interval_list):
        core_start = int(round(interval_start_s * fs))
        stop_frame_global = int(round(interval_stop_s * fs))
        while core_start < stop_frame_global:
            core_stop = min(stop_frame_global, core_start + chunk_frames)
            read_start = max(0, core_start - overlap_frames)
            read_stop = min(total_frames, core_stop + overlap_frames)
            data, coords_mm, fs_loaded, _ = load_chunk(file_path, dataset, read_start, read_stop - read_start)
            if abs(float(fs_loaded) - fs) > 1e-6:
                raise ValueError(f"fs_raw={fs} does not match file sampling={fs_loaded}")
            if not k_grid_prepared:
                k_grid = prepare_k_grid_basis(k_grid, coords_mm)
                k_grid_prepared = True
            if fit_radial and radial_grid is None:
                radial_grid = make_radial_grid(
                    coords_mm,
                    center_grid_n=radial_center_grid_n,
                    center_margin_mm=radial_center_margin_mm,
                    lambda_min_mm=lambda_min,
                    lambda_max_mm=lambda_max,
                    k_grid_n=k_grid_n,
                )

            data_ds = bandpass_downsample(
                data,
                fs,
                band_low=band_low,
                band_high=band_high,
                fs_ds=fs_ds,
                max_channels_per_block=max_channels_per_block,
            )
            u, amp = extract_phasors(data_ds)
            rows = _window_rows(
                data_ds,
                u,
                amp,
                coords_mm,
                fs_ds=fs_ds,
                chunk_start_s=read_start / fs,
                keep_start_s=core_start / fs,
                keep_stop_s=core_stop / fs,
                win_ms=win_ms,
                hop_ms=hop_ms,
                k_grid=k_grid,
                radial_grid=radial_grid,
                refine=refine,
                fit_radial=fit_radial,
                band_low=band_low,
                band_high=band_high,
                r_min=r_min,
                coherence_min=coherence_min,
                radial_delta_r_min=radial_delta_r_min,
                speed_min_cycles=speed_min_cycles,
                lambda_edge_fraction=lambda_edge_fraction,
                speed_min_r=speed_min_r,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
            )
            for row in rows:
                row["interval_idx"] = int(interval_idx)
            _append_rows(out_path, rows)
            core_start = core_stop

    return Path(out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate traveling phase waves from MaxTwo HD-MEA LFP.")
    parser.add_argument("--file", required=True, help="Input MaxTwo .raw.h5 file")
    parser.add_argument(
        "--dataset",
        required=True,
        help="data_store alias like data0000, a well number like 0/well000, or a full raw dataset path",
    )
    parser.add_argument("--out_path", required=True, help="Output HDF5 path")
    parser.add_argument("--fs_raw", type=float, default=0.0, help="Raw sampling rate; 0 infers from file")
    parser.add_argument("--band_low", type=float, default=30.0)
    parser.add_argument("--band_high", type=float, default=80.0)
    parser.add_argument("--fs_ds", type=float, default=500.0)
    parser.add_argument("--chunk_s", type=float, default=30.0)
    parser.add_argument("--overlap_s", type=float, default=1.0)
    parser.add_argument("--win_ms", type=float, default=100.0)
    parser.add_argument("--hop_ms", type=float, default=50.0)
    parser.add_argument("--lambda_min", type=float, default=0.5)
    parser.add_argument("--lambda_max", type=float, default=5.0)
    parser.add_argument("--k_grid_n", type=int, default=41)
    parser.add_argument("--no_refine", action="store_true")
    parser.add_argument("--fit_radial", action="store_true")
    parser.add_argument("--radial_center_grid_n", type=int, default=15)
    parser.add_argument("--radial_center_margin_mm", type=float, default=0.0)
    parser.add_argument("--radial_delta_r_min", type=float, default=0.05)
    parser.add_argument("--speed_min_cycles", type=float, default=1.0)
    parser.add_argument("--lambda_edge_fraction", type=float, default=0.98)
    parser.add_argument("--speed_min_r", type=float, default=0.18)
    parser.add_argument("--r_min", type=float, default=0.2)
    parser.add_argument("--coherence_min", type=float, default=0.0)
    parser.add_argument("--pgd", action="store_true", help="Run instantaneous planar PGD analysis into /pgd_waves")
    parser.add_argument("--pgd_analysis_pad_s", type=float, default=0.05)
    parser.add_argument("--pgd_threshold", type=float, default=0.5)
    parser.add_argument("--pgd_min_duration_ms", type=float, default=5.0)
    parser.add_argument("--pgd_max_direction_change_deg_per_ms", type=float, default=3.0)
    parser.add_argument("--pgd_gradient_n_neighbors", type=int, default=6)
    parser.add_argument("--pgd_gradient_max_distance_mm", type=float, default=None)
    parser.add_argument("--start_s", type=float, default=0.0)
    parser.add_argument("--stop_s", type=float, default=None)
    parser.add_argument("--max_channels_per_block", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    if args.pgd:
        return process_pgd_file(
            args.file,
            args.dataset,
            args.out_path,
            fs_raw=args.fs_raw,
            band_low=args.band_low,
            band_high=args.band_high,
            fs_ds=args.fs_ds,
            chunk_s=args.chunk_s,
            overlap_s=args.overlap_s,
            start_s=args.start_s,
            stop_s=args.stop_s,
            analysis_pad_s=args.pgd_analysis_pad_s,
            k_grid_n=args.k_grid_n,
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max,
            pgd_threshold=args.pgd_threshold,
            pgd_min_duration_ms=args.pgd_min_duration_ms,
            pgd_max_direction_change_deg_per_ms=args.pgd_max_direction_change_deg_per_ms,
            gradient_n_neighbors=args.pgd_gradient_n_neighbors,
            gradient_max_distance_mm=args.pgd_gradient_max_distance_mm,
            refine=not args.no_refine,
            max_channels_per_block=args.max_channels_per_block,
        )
    return process_file(
        args.file,
        args.dataset,
        args.out_path,
        fs_raw=args.fs_raw,
        band_low=args.band_low,
        band_high=args.band_high,
        fs_ds=args.fs_ds,
        chunk_s=args.chunk_s,
        overlap_s=args.overlap_s,
        win_ms=args.win_ms,
        hop_ms=args.hop_ms,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        k_grid_n=args.k_grid_n,
        refine=not args.no_refine,
        fit_radial=args.fit_radial,
        radial_center_grid_n=args.radial_center_grid_n,
        radial_center_margin_mm=args.radial_center_margin_mm,
        radial_delta_r_min=args.radial_delta_r_min,
        speed_min_cycles=args.speed_min_cycles,
        lambda_edge_fraction=args.lambda_edge_fraction,
        speed_min_r=args.speed_min_r,
        r_min=args.r_min,
        coherence_min=args.coherence_min,
        start_s=args.start_s,
        stop_s=args.stop_s,
        max_channels_per_block=args.max_channels_per_block,
    )


if __name__ == "__main__":
    main()
