from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt, welch


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
    "wave_model_class",
    "radial_valid",
    "R_fit",
    "mean_weight",
    "n_good",
    "valid",
    "interval_idx",
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
    if not (0 < band_low < band_high < fs_raw / 2):
        raise ValueError("band limits must satisfy 0 < band_low < band_high < fs_raw/2")
    if not (0 < fs_ds <= fs_raw):
        raise ValueError("fs_ds must be positive and <= fs_raw")

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


def make_radial_grid(
    coords_mm: np.ndarray,
    *,
    center_grid_n: int = 15,
    center_margin_mm: float = 0.0,
    lambda_min_mm: float = 0.5,
    lambda_max_mm: float = 5.0,
    k_grid_n: int = 41,
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
    return {
        "centers": centers,
        "k": k_values,
        "k_min": k_min,
        "k_max": k_max,
        "center_margin_mm": float(center_margin_mm),
    }


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
    ubar = np.asarray(ubar, dtype=np.complex128)
    w = np.asarray(w, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape != (ubar.size, 2):
        raise ValueError("coords must have shape (n_channels, 2)")

    finite = np.isfinite(w) & (w > min_weight) & np.isfinite(ubar.real) & np.isfinite(ubar.imag)
    finite &= np.isfinite(coords).all(axis=1)
    if np.count_nonzero(finite) < 3:
        return {"kx": np.nan, "ky": np.nan, "R_fit": np.nan, "n_good": int(np.count_nonzero(finite))}

    ubar = ubar[finite]
    w = w[finite]
    coords = coords[finite]
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
    ubar = np.asarray(ubar, dtype=np.complex128)
    w = np.asarray(w, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape != (ubar.size, 2):
        raise ValueError("coords must have shape (n_channels, 2)")

    finite = np.isfinite(w) & (w > min_weight) & np.isfinite(ubar.real) & np.isfinite(ubar.imag)
    finite &= np.isfinite(coords).all(axis=1)
    if np.count_nonzero(finite) < 3:
        return {
            "radial_x0_mm": np.nan,
            "radial_y0_mm": np.nan,
            "radial_k": np.nan,
            "radial_sign": 0,
            "R_radial": np.nan,
            "radial_n_good": int(np.count_nonzero(finite)),
        }

    ubar = ubar[finite]
    w = w[finite]
    coords = coords[finite]
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

    best_score = -np.inf
    best_center = np.array([np.nan, np.nan], dtype=np.float64)
    best_k = np.nan
    best_sign = 0
    weighted_ubar = w * ubar
    for center in centers:
        distances = np.linalg.norm(coords - center[None, :], axis=1)
        for k_abs in k_values:
            phase = float(k_abs) * distances
            for sign in (-1.0, 1.0):
                score = float(np.abs(np.sum(weighted_ubar * np.exp(-1j * sign * phase))))
                if score > best_score:
                    best_score = score
                    best_center = center.astype(np.float64, copy=True)
                    best_k = float(k_abs)
                    best_sign = int(sign)

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
            "radial_sign",
            "wave_model_class",
            "radial_valid",
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
                "wave_model_class": int(wave_model_class),
                "radial_valid": radial_valid,
                "R_fit": float(fit["R_fit"]),
                "mean_weight": mean_weight,
                "n_good": int(fit["n_good"]),
                "valid": valid,
            }
        )
    return rows


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
        "r_min": r_min,
        "coherence_min": coherence_min,
        "start_s": start_s,
        "stop_s": stop_s,
        "intervals": interval_list,
    }
    _init_output(out_path, config)
    k_grid = make_k_grid(lambda_min, lambda_max, n_grid=k_grid_n)
    radial_grid = None

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
    parser.add_argument("--r_min", type=float, default=0.2)
    parser.add_argument("--coherence_min", type=float, default=0.0)
    parser.add_argument("--start_s", type=float, default=0.0)
    parser.add_argument("--stop_s", type=float, default=None)
    parser.add_argument("--max_channels_per_block", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
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
        r_min=args.r_min,
        coherence_min=args.coherence_min,
        start_s=args.start_s,
        stop_s=args.stop_s,
        max_channels_per_block=args.max_channels_per_block,
    )


if __name__ == "__main__":
    main()
