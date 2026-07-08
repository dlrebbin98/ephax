from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt
from scipy.spatial import cKDTree

from ephax.metrics.burst import (
    assign_max_population_ifr_burst_anchors,
    build_highres_traces,
    build_participation_activity_state,
    build_population_ifr,
    detect_high_activity_epochs,
    detect_participation_burst_epochs,
)
from ephax.preprocessing.dataset import Recording


def dataset_for_well(well: int) -> str:
    return f"data{int(well):04d}"

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


def load_data_store_spikes(file_path: str | Path, dataset: str, *, min_amp: float = 0.0):
    raw_path, rec_group_path = _resolve_dataset_paths(dataset, file_path)
    with h5py.File(file_path, "r") as h5:
        rec_group = h5[rec_group_path]
        settings = rec_group["settings"]
        mapping_obj = settings["mapping"]
        if getattr(mapping_obj, "dtype", None) is not None and mapping_obj.dtype.fields is not None:
            mapping_arr = mapping_obj[:]
            layout = {name: np.asarray(mapping_arr[name]) for name in ("channel", "electrode", "x", "y")}
        else:
            layout = {name: np.asarray(mapping_obj[name])[:] for name in ("channel", "electrode", "x", "y")}
        sf = float(settings["sampling"][0])
        spikes_arr = rec_group["spikes"][:]
        frame_nos = h5[raw_path].parent.get("frame_nos")
        first_frame = (
            int(frame_nos[0])
            if frame_nos is not None and frame_nos.shape[0]
            else int(np.nanmin(spikes_arr["frameno"]))
            if spikes_arr.size
            else 0
        )
        spikes = {
            "time": (np.asarray(spikes_arr["frameno"], dtype=float) - first_frame) / sf,
            "channel": np.asarray(spikes_arr["channel"]),
            "amplitude": np.asarray(spikes_arr["amplitude"]),
        }
        channel_to_electrode = {int(ch): int(el) for ch, el in zip(layout["channel"], layout["electrode"])}
        spikes["electrode"] = np.asarray([channel_to_electrode.get(int(ch), -1) for ch in spikes["channel"]])
        keep = (
            (spikes["electrode"] >= 0)
            & (spikes["time"] >= 0)
            & (np.abs(spikes["amplitude"]) >= float(min_amp))
        )
        spikes = {key: np.asarray(value)[keep] for key, value in spikes.items()}
    return spikes, layout, sf
