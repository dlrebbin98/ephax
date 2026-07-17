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
from ephax.metrics.lfp import (
    bandpass_downsample,
    dataset_for_well,
    extract_phasors,
    inspect_file,
    load_chunk,
    load_data_store_spikes,
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

REQUIRED_WAVEFRONT_GROUPS = frozenset(("wavefront_calibration", "wavefront_events", "wavefront_local"))


@dataclass(frozen=True)
class LFPWavefrontCacheConfig:
    source: str
    profile: str
    div: int
    band_low_hz: float
    band_high_hz: float
    fs_hz: float
    lambda_min_mm: float
    lambda_max_mm: float
    radial: bool
    interval_mode: str = "burst"

    def dataset_for_well(self, well: int) -> str:
        return dataset_for_well(well)

    @staticmethod
    def _num_token(value: float) -> str:
        return f"{float(value):g}"

    @property
    def band_token(self) -> str:
        return f"{self._num_token(self.band_low_hz)}-{self._num_token(self.band_high_hz)}Hz"

    @property
    def fs_token(self) -> str:
        return f"fs{self._num_token(self.fs_hz)}"

    @property
    def lambda_token(self) -> str:
        return f"lambda{self._num_token(self.lambda_min_mm)}-{self._num_token(self.lambda_max_mm)}"

    def wavefront_filename(self, well: int) -> str:
        well = int(well)
        dataset = self.dataset_for_well(well)
        return (
            f"wavefront_{self.source}_{self.profile}_well{well}_DIV{int(self.div)}_"
            f"{dataset}_{self.interval_mode}_{self.band_token}_{self.fs_token}_"
            f"{self.lambda_token}_radial{int(bool(self.radial))}.h5"
        )

    def wavefront_path(self, cache_dir: str | Path, well: int) -> Path:
        return Path(cache_dir) / self.wavefront_filename(well)

    def phasor_feature_filename(self, well: int) -> str:
        well = int(well)
        dataset = self.dataset_for_well(well)
        return (
            f"phasor_features_{self.source}_{self.profile}_well{well}_DIV{int(self.div)}_"
            f"{dataset}_{self.band_token}_{self.fs_token}.h5"
        )

    def phasor_feature_path(self, cache_dir: str | Path, well: int) -> Path:
        return Path(cache_dir) / self.phasor_feature_filename(well)


def lfp_wavefront_cache_status(path: str | Path) -> tuple[bool, str]:
    path = Path(path)
    if not path.exists():
        return False, "missing"
    try:
        with h5py.File(path, "r") as h5:
            missing = REQUIRED_WAVEFRONT_GROUPS - set(h5.keys())
            if missing:
                return False, f"missing groups {sorted(missing)}"
            if "complete" in h5.attrs and not bool(h5.attrs["complete"]):
                return False, "marked incomplete"
    except OSError as exc:
        return False, str(exc)
    return True, "ok"


def require_lfp_wavefront_caches(
    config: LFPWavefrontCacheConfig,
    cache_dir: str | Path,
    wells: list[int] | tuple[int, ...],
    *,
    require_all: bool = True,
) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    diagnostics: list[str] = []
    for well in wells:
        well = int(well)
        path = config.wavefront_path(cache_dir, well)
        ok, reason = lfp_wavefront_cache_status(path)
        if ok:
            paths[well] = path
        else:
            diagnostics.append(f"well {well}: {path} [{reason}]")
    if (require_all and len(paths) != len(wells)) or not paths:
        raise FileNotFoundError("Missing compatible canonical wavefront caches:\n" + "\n".join(diagnostics))
    return paths


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


def read_lfp_wavefront_cache(path: str | Path) -> dict[str, dict[str, object]]:
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


# Public LFP-prefixed cache API; the HDF5 layout is unchanged for compatibility.
initialize_lfp_wavefront_cache = initialize_wavefront_cache
append_lfp_wavefront_rows = append_wavefront_rows

@dataclass(frozen=True)
class LFPEventFrontBuildConfig:
    raw_file: Path
    cache_dir: Path
    cache: LFPWavefrontCacheConfig
    wells: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    profile_calibration_events: int | None = None
    profile_scoring_events: int = 200
    min_amp: float = 0.0
    top_start: int = 0
    top_stop: int = 1000
    ifr_grid_hz: float = 50.0
    smooth_sigma_sec: float = 0.15
    high_activity_mad_scale: float = 3.0
    high_activity_min_duration_ms: float = 30.0
    high_activity_max_gap_bins: int = 0
    highres_bin_ms: float = 1.0
    highres_smooth_sigma_ms: float = 3.0
    network_bin_ms: float = 10.0
    network_min_participation_fraction: float = 0.05
    network_min_duration_ms: float = 20.0
    phase_pad_s: float = 0.25
    search_pre_ms: float = 20.0
    search_post_ms: float = 40.0
    phase_bins: int = 36
    pseudocount: float = 1.0
    theta_override_rad: float | None = None
    min_electrodes: int = 50
    match_max_ms: float = 8.0
    radius_mm: float = 0.30
    min_neighbors: int = 8
    max_radius_mm: float = 0.45
    max_neighbors: int = 24
    min_geometry_ratio: float = 0.10
    max_residual_ms: float = 4.0
    min_arrival_span_ms: float = 1.0
    min_distance_span_mm: float = 0.20
    max_speed_mm_per_s: float = 500.0
    min_arrival_amplitude: float = 0.0
    min_arrival_amplitude_percentile: float = 0.0
    max_channels_per_block: int = 256
    radial_center_grid_n: int = 15


def ensure_lfp_wavefront_caches(config: LFPEventFrontBuildConfig, *, build_missing: bool = True) -> dict[int, Path]:
    try:
        return require_lfp_wavefront_caches(config.cache, config.cache_dir, config.wells, require_all=True)
    except FileNotFoundError:
        if not build_missing:
            raise
    status = build_lfp_wavefront_caches(config)
    missing = status.loc[status["cache_state"].isin(["missing", "incompatible", "partial"])]
    if not missing.empty:
        print("Processed cache status:")
        print(status.to_string(index=False))
    return require_lfp_wavefront_caches(config.cache, config.cache_dir, config.wells, require_all=True)


def load_completed_lfp_wavefront_cache(config: LFPEventFrontBuildConfig, well: int):
    status, cache = lfp_wavefront_cache_state(config, well)
    if status != "complete":
        return None
    return cache


def build_lfp_wavefront_caches(config: LFPEventFrontBuildConfig) -> pd.DataFrame:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    contexts = {}
    rows = []
    for well in config.wells:
        context = _detect_context(config, int(well))
        contexts[int(well)] = context
        if context["bursts"].empty:
            rows.append(_status_row(config, int(well), context, "no_bursts"))
            continue
        state, _ = lfp_wavefront_cache_state(config, int(well), context)
        rows.append(_status_row(config, int(well), context, state))
    status = pd.DataFrame(rows)
    print(status.round(3).to_string(index=False))
    for well in config.wells:
        context = contexts[int(well)]
        if context["bursts"].empty:
            continue
        _process_well(config, int(well), context)
    return status


def lfp_wavefront_cache_state(config: LFPEventFrontBuildConfig, well: int, context: dict | None = None):
    path = config.cache.wavefront_path(config.cache_dir, well)
    if not path.exists():
        return "missing", None
    cache = read_lfp_wavefront_cache(path)
    stored = json.loads(cache.get("wavefront_calibration", {}).get("config", "{}"))
    if not _summary_config_matches(config, stored):
        return "incompatible", None
    if context is not None and stored != _front_config(config, well, context):
        return "incompatible", None
    with h5py.File(path, "r") as h5:
        complete = bool(h5.attrs.get("complete", False))
    return ("complete" if complete else "partial"), cache


def _status_row(config: LFPEventFrontBuildConfig, well: int, context: dict, state: str) -> dict:
    if context["bursts"].empty:
        return {
            "well": well,
            "cache_state": state,
            "detected_bursts": 0,
            "calibration_events": 0,
            "scored_events": 0,
            "retained_lfp_s": 0.0,
            "burst_detection_s": context["detection_elapsed_s"],
        }
    feature_path = config.cache.phasor_feature_path(config.cache_dir, well)
    cached_segments = 0
    if feature_path.exists():
        metadata = read_phasor_feature_metadata(feature_path)
        if metadata["config"] == _feature_config(config, well, context):
            cached_segments = len(metadata["segments"])
    return {
        "well": well,
        "cache_state": state,
        "detected_bursts": len(context["bursts"]),
        "calibration_events": len(context["calibration_events"]),
        "scored_events": len(context["scoring_events"]),
        "overlap_events": context["overlap_count"],
        "cached_segments": cached_segments,
        "pending_segments": len(context["merged_feature_intervals"]) - cached_segments,
        "retained_lfp_s": context["retained_lfp_s"],
        "burst_detection_s": context["detection_elapsed_s"],
    }


def _detect_context(config: LFPEventFrontBuildConfig, well: int) -> dict:
    started = perf_counter()
    dataset = dataset_for_well(well)
    info = inspect_file(config.raw_file, dataset)
    spikes, layout, sf = load_data_store_spikes(config.raw_file, dataset, min_amp=config.min_amp)
    stop_s = min(
        info["raw_shape"][1] / info["fs_raw"],
        float(np.nanmax(spikes["time"])) if len(spikes["time"]) else np.inf,
    )
    recording = Recording(spikes=spikes, layout=layout, start_time=0.0, end_time=stop_s, sf=float(sf))
    refs = recording.refs_top(start=config.top_start, stop=config.top_stop)
    population = build_population_ifr(
        recording, refs, grid_hz=config.ifr_grid_hz, smooth_sigma_sec=config.smooth_sigma_sec
    )
    high_epochs, _ = detect_high_activity_epochs(
        population.time_grid,
        population.mean_ifr_smooth,
        mad_scale=config.high_activity_mad_scale,
        min_duration_ms=config.high_activity_min_duration_ms,
        max_gap_bins=config.high_activity_max_gap_bins,
    )
    highres = build_highres_traces(
        recording, refs, bin_ms=config.highres_bin_ms, smooth_sigma_ms=config.highres_smooth_sigma_ms
    )
    network = build_participation_activity_state(highres, aggregation_ms=config.network_bin_ms)
    bursts = detect_participation_burst_epochs(
        network,
        high_epochs,
        min_participation_fraction=config.network_min_participation_fraction,
        min_duration_ms=config.network_min_duration_ms,
    )
    bursts = assign_max_population_ifr_burst_anchors(highres, bursts).sort_values("anchor_time_s").reset_index(drop=True)
    if bursts.empty:
        return {"well": well, "bursts": bursts, "spikes": spikes, "stop_s": stop_s, "detection_elapsed_s": perf_counter() - started}
    selected = select_representative_event_indices(
        len(bursts),
        n_calibration=config.profile_calibration_events,
        n_scoring=config.profile_scoring_events,
    )
    calibration_events = bursts.iloc[selected["calibration_indices"]].copy()
    scoring_events = bursts.iloc[selected["scoring_indices"]].copy()
    calibration_intervals = [
        (
            max(0.0, float(row.start_time_s) - config.phase_pad_s),
            min(stop_s, float(row.end_time_s) + config.phase_pad_s),
        )
        for row in calibration_events.itertuples(index=False)
    ]
    scoring_intervals = [
        (
            max(0.0, float(row.anchor_time_s) - config.search_pre_ms / 1000.0 - config.phase_pad_s),
            min(stop_s, float(row.anchor_time_s) + config.search_post_ms / 1000.0 + config.phase_pad_s),
        )
        for row in scoring_events.itertuples(index=False)
    ]
    merged = merge_time_intervals(calibration_intervals + scoring_intervals)
    return {
        "well": well,
        "bursts": bursts,
        "spikes": spikes,
        "stop_s": stop_s,
        "calibration_events": calibration_events,
        "scoring_events": scoring_events,
        "calibration_intervals": calibration_intervals,
        "scoring_intervals": scoring_intervals,
        "merged_feature_intervals": merged,
        "overlap_count": int(selected["overlap_count"]),
        "retained_lfp_s": float(sum(stop - start for start, stop in merged)),
        "detection_elapsed_s": perf_counter() - started,
    }


def _front_config(config: LFPEventFrontBuildConfig, well: int, context: dict) -> dict:
    result = _summary_config(config)
    result.update(
        {
            "file": str(config.raw_file),
            "dataset": dataset_for_well(well),
            "well": int(well),
            "div": int(config.cache.div),
            "calibration_event_ids": _event_ids(context["calibration_events"]),
            "scoring_event_ids": _event_ids(context["scoring_events"]),
            "calibration_scoring_overlap": int(context["overlap_count"]),
        }
    )
    return result


def _summary_config(config: LFPEventFrontBuildConfig) -> dict:
    result = {
        "profile": config.cache.profile,
        "band_low": float(config.cache.band_low_hz),
        "band_high": float(config.cache.band_high_hz),
        "fs_ds": float(config.cache.fs_hz),
        "filter_pad_s": float(config.phase_pad_s),
        "theta_override_rad": None if config.theta_override_rad is None else float(config.theta_override_rad),
        "phase_bins": int(config.phase_bins),
        "pseudocount": float(config.pseudocount),
        "min_electrodes": int(config.min_electrodes),
        "match_max_ms": float(config.match_max_ms),
        "radius_mm": float(config.radius_mm),
        "min_neighbors": int(config.min_neighbors),
        "max_radius_mm": float(config.max_radius_mm),
        "max_neighbors": int(config.max_neighbors),
        "min_geometry_ratio": float(config.min_geometry_ratio),
        "max_residual_ms": float(config.max_residual_ms),
        "min_arrival_span_ms": float(config.min_arrival_span_ms),
        "min_distance_span_mm": float(config.min_distance_span_mm),
        "max_speed_mm_per_s": float(config.max_speed_mm_per_s),
    }
    if float(config.min_arrival_amplitude) > 0.0:
        result["min_arrival_amplitude"] = float(config.min_arrival_amplitude)
    if float(config.min_arrival_amplitude_percentile) > 0.0:
        result["min_arrival_amplitude_percentile"] = float(config.min_arrival_amplitude_percentile)
    return result


def _summary_config_matches(config: LFPEventFrontBuildConfig, stored: dict) -> bool:
    expected = _summary_config(config)
    return all(stored.get(key) == value for key, value in expected.items())


def _feature_config(config: LFPEventFrontBuildConfig, well: int, context: dict) -> dict:
    return {
        "file": str(config.raw_file),
        "dataset": dataset_for_well(well),
        "well": int(well),
        "div": int(config.cache.div),
        "profile": config.cache.profile,
        "calibration_event_ids": _event_ids(context["calibration_events"]),
        "scoring_event_ids": _event_ids(context["scoring_events"]),
        "band_low": float(config.cache.band_low_hz),
        "band_high": float(config.cache.band_high_hz),
        "fs_ds": float(config.cache.fs_hz),
        "filter_pad_s": float(config.phase_pad_s),
        "intervals": [[float(start), float(stop)] for start, stop in context["merged_feature_intervals"]],
    }


def _event_ids(frame: pd.DataFrame) -> list[int]:
    return [int(value) for value in frame["event_idx"].to_numpy(int)]


def _load_lfp_interval(config: LFPEventFrontBuildConfig, well: int, interval):
    dataset = dataset_for_well(well)
    info = inspect_file(config.raw_file, dataset)
    start_s, stop_s = map(float, interval)
    start_frame = int(round(start_s * info["fs_raw"]))
    n_frames = max(1, int(round((stop_s - start_s) * info["fs_raw"])))
    raw, coords, fs_raw, mapping = load_chunk(config.raw_file, dataset, start_frame, n_frames)
    band = bandpass_downsample(
        raw,
        fs_raw,
        config.cache.band_low_hz,
        config.cache.band_high_hz,
        config.cache.fs_hz,
        max_channels_per_block=config.max_channels_per_block,
    )
    time = start_s + np.arange(band.shape[0]) / config.cache.fs_hz
    return band, time, coords, mapping


def _build_feature_cache(config: LFPEventFrontBuildConfig, well: int, context: dict) -> Path:
    path = config.cache.phasor_feature_path(config.cache_dir, well)
    feature_config = _feature_config(config, well, context)
    merged = context["merged_feature_intervals"]
    if phasor_feature_cache_matches(path, feature_config):
        print(f"Well {well}: using phasor feature cache {path.name}")
        return path
    completed_segments = 0
    initialized = False
    if path.exists():
        metadata = read_phasor_feature_metadata(path)
        if metadata["config"] == feature_config:
            completed_segments = len(metadata["segments"])
            initialized = True
            print(f"Well {well}: resuming phasor features after {completed_segments}/{len(merged)} segments")
    print(f"Well {well}: retained LFP context {context['retained_lfp_s']:.2f}s across {len(merged)} merged segments")
    for segment_idx, interval in enumerate(merged[completed_segments:], start=completed_segments + 1):
        print(f"Well {well}: feature segment {segment_idx}/{len(merged)}")
        band, time, coords, mapping = _load_lfp_interval(config, well, interval)
        phasors, amplitude = extract_phasors(band)
        electrodes = np.asarray(mapping["electrode"], dtype=int)
        if not initialized:
            initialize_phasor_feature_cache(path, coords, electrodes, config=feature_config)
            initialized = True
        append_phasor_feature_segment(path, interval[0], time, phasors, amplitude)
    if not initialized:
        raise ValueError(f"Well {well}: no feature intervals were available")
    finalize_phasor_feature_cache(path)
    return path


def _process_well(config: LFPEventFrontBuildConfig, well: int, context: dict):
    state, cache = lfp_wavefront_cache_state(config, well, context)
    if state == "complete":
        print(f"Well {well}: completed profile cache already exists; skipped")
        return cache
    path = config.cache.wavefront_path(config.cache_dir, well)
    feature_path = _build_feature_cache(config, well, context)
    feature_meta = read_phasor_feature_metadata(feature_path)
    coords_static = np.asarray(feature_meta["coords_mm"], dtype=float)
    neighborhoods = make_local_phase_neighborhoods(
        coords_static,
        radius_mm=config.radius_mm,
        min_neighbors=config.min_neighbors,
        max_radius_mm=config.max_radius_mm,
        max_neighbors=config.max_neighbors,
        min_geometry_ratio=config.min_geometry_ratio,
    )
    if state != "partial":
        _initialize_wavefront_result(config, well, context, path)
        cache = read_lfp_wavefront_cache(path)
    calibration = cache["wavefront_calibration"]
    existing_ids = set(map(int, cache.get("wavefront_events", {}).get("event_idx", [])))
    pending = [
        (int(row.event_idx), row)
        for row in context["scoring_events"].itertuples(index=False)
        if int(row.event_idx) not in existing_ids
    ]
    for pending_idx, (event_idx, row) in enumerate(pending, start=1):
        print(f"Well {well}: scoring event {pending_idx}/{len(pending)}")
        _append_event_front(config, well, context, feature_path, path, neighborhoods, calibration, event_idx, row)
    with h5py.File(path, "a") as h5:
        h5.attrs["complete"] = True
        h5.attrs["completed_event_count"] = int(len(context["scoring_events"]))
    print(f"Well {well}: completed {path.name}")
    return read_lfp_wavefront_cache(path)


def _initialize_wavefront_result(config: LFPEventFrontBuildConfig, well: int, context: dict, path: Path) -> None:
    calibration_parts = []
    feature_path = config.cache.phasor_feature_path(config.cache_dir, well)
    for idx, interval in enumerate(context["calibration_intervals"], start=1):
        print(f"Well {well}: calibration slice {idx}/{len(context['calibration_intervals'])}")
        phasors, time, _, electrodes = read_phasor_feature_interval(feature_path, *interval)
        calibration_parts.append(
            estimate_excitable_phase(
                phasors,
                time,
                electrodes,
                np.asarray(context["spikes"]["time"], dtype=float),
                np.asarray(context["spikes"]["electrode"], dtype=int),
                n_bins=config.phase_bins,
                pseudocount=config.pseudocount,
            )
        )
    calibration = combine_excitable_phase_calibrations(calibration_parts, pseudocount=config.pseudocount)
    thetas = np.asarray([part["theta_excitable_rad"] for part in calibration_parts], dtype=float)
    resultant = float(abs(np.mean(np.exp(1j * thetas))))
    calibration["calibration_theta_resultant"] = resultant
    calibration["calibration_theta_circular_std_rad"] = float(
        np.sqrt(max(0.0, -2.0 * np.log(max(resultant, np.finfo(float).tiny))))
    )
    calibration["calibration_interval_count"] = int(len(calibration_parts))
    calibration["calibration_stable"] = int(resultant >= 0.50)
    if config.theta_override_rad is not None:
        calibration["theta_excitable_rad"] = float(config.theta_override_rad)
    initialize_wavefront_cache(path, calibration, config=_front_config(config, well, context))
    with h5py.File(path, "a") as h5:
        h5.attrs["complete"] = False


def _append_event_front(
    config: LFPEventFrontBuildConfig,
    well: int,
    context: dict,
    feature_path: Path,
    cache_path: Path,
    neighborhoods,
    calibration,
    event_idx: int,
    row,
) -> None:
    anchor_s = float(row.anchor_time_s)
    read_interval = (
        max(0.0, anchor_s - config.search_pre_ms / 1000.0 - config.phase_pad_s),
        min(context["stop_s"], anchor_s + config.search_post_ms / 1000.0 + config.phase_pad_s),
    )
    phasors, time, coords, electrodes, amplitude = read_phasor_feature_interval(
        feature_path, *read_interval, include_amplitude=True
    )
    analysis = analyze_excitable_phase_front(
        phasors,
        time,
        coords,
        neighborhoods,
        amplitude=amplitude,
        min_arrival_amplitude=config.min_arrival_amplitude,
        min_arrival_amplitude_percentile=config.min_arrival_amplitude_percentile,
        theta_excitable_rad=float(calibration["theta_excitable_rad"]),
        anchor_time_s=anchor_s,
        frequency_hz=0.5 * (config.cache.band_low_hz + config.cache.band_high_hz),
        match_max_ms=config.match_max_ms,
        min_electrodes=config.min_electrodes,
        search_pre_ms=config.search_pre_ms,
        search_post_ms=config.search_post_ms,
        min_neighbors=config.min_neighbors,
        max_residual_ms=config.max_residual_ms,
        min_arrival_span_ms=config.min_arrival_span_ms,
        min_distance_span_mm=config.min_distance_span_mm,
        max_speed_mm_per_s=config.max_speed_mm_per_s,
        radial_center_grid_n=config.radial_center_grid_n,
    )
    front = analysis["front"]
    local = analysis["local"]
    append_wavefront_rows(
        cache_path,
        "wavefront_events",
        [{
            "event_idx": int(event_idx),
            "anchor_time_s": anchor_s,
            "front_time_s": float(front["front_time_s"]),
            "front_cycle_offset": int(front["cycle_offset"]),
            "front_n_electrodes": int(front["n_electrodes"]),
            "front_valid": int(front["valid"]),
            **analysis["planar"],
            **analysis["radial"],
        }],
    )
    electrodes = np.asarray(electrodes, dtype=int)
    arrival = np.asarray(front["arrival_time_s"], dtype=float)
    append_wavefront_rows(
        cache_path,
        "wavefront_local",
        {
            "event_idx": np.full(electrodes.size, event_idx),
            "anchor_time_s": np.full(electrodes.size, anchor_s),
            "electrode": electrodes,
            "x_mm": coords[:, 0],
            "y_mm": coords[:, 1],
            "arrival_time_s": arrival,
            "arrival_rel_anchor_ms": (arrival - anchor_s) * 1000.0,
            "arrival_amplitude": local["arrival_amplitude"],
            "arrival_amplitude_threshold": local["arrival_amplitude_threshold"],
            "velocity_x_mm_per_s": local["velocity_x_mm_per_s"],
            "velocity_y_mm_per_s": local["velocity_y_mm_per_s"],
            "speed_mm_per_s": local["speed_mm_per_s"],
            "residual_rms_ms": local["residual_rms_ms"],
            "arrival_span_ms": local["arrival_span_ms"],
            "distance_span_mm": local["distance_span_mm"],
            "arrival_gradient_norm_s_per_mm": local["arrival_gradient_norm_s_per_mm"],
            "speed_censored": local["speed_censored"].astype(np.uint8),
            "n_good": local["n_good"],
            "valid": local["valid"].astype(np.uint8),
        },
    )


# =====================================================================================
# Event selection: burst detection, activity ranking, and null-corrected front subsetting
# =====================================================================================


@dataclass(frozen=True)
class WellBurstConfig:
    """Parameters for :func:`detect_well_bursts` (defaults match the Figure 3 pipeline)."""
    min_amp: float = 0.0
    top_start: int = 0
    top_stop: int = 1000
    ifr_grid_hz: float = 50.0
    smooth_sigma_sec: float = 0.15
    high_activity_mad_scale: float = 3.0
    high_activity_min_duration_ms: float = 30.0
    high_activity_max_gap_bins: int = 0
    highres_bin_ms: float = 1.0
    highres_smooth_sigma_ms: float = 3.0
    network_bin_ms: float = 10.0
    network_min_participation_fraction: float = 0.05
    network_min_duration_ms: float = 20.0


@dataclass(frozen=True)
class FrontNullConfig:
    """Local arrival-plane refit parameters for the spatial-null front support."""
    radius_mm: float = 0.30
    min_neighbors: int = 8
    max_radius_mm: float = 0.45
    max_neighbors: int = 24
    min_geometry_ratio: float = 0.10
    match_max_ms: float = 8.0
    max_residual_ms: float = 4.0
    min_arrival_span_ms: float = 1.0
    min_distance_span_mm: float = 0.20
    max_speed_mm_per_s: float = 500.0


def detect_well_bursts(raw_file, dataset, *, cfg: WellBurstConfig = WellBurstConfig(),
                       cache_dir=None, force_recompute: bool = False):
    """Detect participation bursts for one well; returns ``(bursts_df, info)``.

    Thin composition of the existing burst API (``build_population_ifr`` ->
    ``detect_high_activity_epochs`` -> ``build_participation_activity_state`` ->
    ``detect_participation_burst_epochs`` -> ``assign_max_population_ifr_burst_anchors``).
    ``bursts_df`` has ``start_time_s``, ``end_time_s``, ``anchor_time_s`` sorted by anchor.

    Burst detection is deterministic and (over a full recording) slow, so when ``cache_dir`` is
    given the ``bursts_df`` is pickled there keyed by ``(raw_file, dataset, cfg)`` and reused on
    later calls / other notebooks. ``info`` is always re-read (cheap) so it stays consistent.
    """
    info = inspect_file(raw_file, dataset)
    cache_path = None
    if cache_dir is not None:
        import hashlib
        from dataclasses import astuple
        key = hashlib.md5(repr((str(raw_file), str(dataset), astuple(cfg))).encode()).hexdigest()[:16]
        cache_path = Path(cache_dir) / f"well_bursts_{key}.pkl"
        if cache_path.exists() and not force_recompute:
            return pd.read_pickle(cache_path), info
    spikes, layout, sf = load_data_store_spikes(raw_file, dataset, min_amp=cfg.min_amp)
    stop_s = min(
        info["raw_shape"][1] / info["fs_raw"],
        float(np.nanmax(spikes["time"])) if len(spikes["time"]) else np.inf,
    )
    recording = Recording(spikes=spikes, layout=layout, start_time=0.0, end_time=stop_s, sf=float(sf))
    refs = recording.refs_top(start=cfg.top_start, stop=cfg.top_stop)
    population = build_population_ifr(recording, refs, grid_hz=cfg.ifr_grid_hz, smooth_sigma_sec=cfg.smooth_sigma_sec)
    high_epochs, _ = detect_high_activity_epochs(
        population.time_grid, population.mean_ifr_smooth,
        mad_scale=cfg.high_activity_mad_scale,
        min_duration_ms=cfg.high_activity_min_duration_ms,
        max_gap_bins=cfg.high_activity_max_gap_bins,
    )
    highres = build_highres_traces(recording, refs, bin_ms=cfg.highres_bin_ms, smooth_sigma_ms=cfg.highres_smooth_sigma_ms)
    network = build_participation_activity_state(highres, aggregation_ms=cfg.network_bin_ms)
    bursts = detect_participation_burst_epochs(
        network, high_epochs,
        min_participation_fraction=cfg.network_min_participation_fraction,
        min_duration_ms=cfg.network_min_duration_ms,
    )
    bursts = assign_max_population_ifr_burst_anchors(highres, bursts).sort_values("anchor_time_s").reset_index(drop=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        bursts.to_pickle(cache_path)
    return bursts, info


def peri_anchor_spike_counts(spike_times, anchors, *, pre_s: float, post_s: float) -> np.ndarray:
    """Spike count in ``[anchor - pre_s, anchor + post_s]`` for each anchor (burst-peak activity rank)."""
    st = np.asarray(spike_times, dtype=float)
    anchors = np.asarray(anchors, dtype=float)
    return np.array(
        [int(np.count_nonzero((st >= a - pre_s) & (st <= a + post_s))) for a in anchors],
        dtype=int,
    )


def event_null_arrival_fit(group, rng, *, null_type: str, cfg: FrontNullConfig = FrontNullConfig()) -> dict:
    """Refit one event's local arrival planes under a coordinate/arrival-time shuffle null."""
    coords = group[["x_mm", "y_mm"]].to_numpy(float)
    arrival = group["arrival_time_s"].to_numpy(float)
    amp = group["arrival_amplitude"].to_numpy(float) if "arrival_amplitude" in group else None
    if "arrival_amplitude_threshold" in group:
        amp_threshold = group["arrival_amplitude_threshold"].to_numpy(float)
        amp_threshold = np.where(np.isfinite(amp_threshold), amp_threshold, 0.0)
    else:
        amp_threshold = 0.0
    if null_type == "coordinate_shuffle":
        coords = coords[rng.permutation(coords.shape[0])]
    elif null_type == "arrival_time_shuffle":
        arrival = arrival[rng.permutation(arrival.size)]
    else:
        raise ValueError("null_type must be 'coordinate_shuffle' or 'arrival_time_shuffle'")
    neighborhoods = make_local_phase_neighborhoods(
        coords,
        radius_mm=cfg.radius_mm,
        min_neighbors=cfg.min_neighbors,
        max_radius_mm=cfg.max_radius_mm,
        max_neighbors=cfg.max_neighbors,
        min_geometry_ratio=cfg.min_geometry_ratio,
    )
    fitted = fit_local_arrival_velocity(
        arrival,
        coords,
        neighborhoods,
        arrival_amplitude=amp,
        min_arrival_amplitude=amp_threshold,
        match_max_ms=cfg.match_max_ms,
        min_neighbors=cfg.min_neighbors,
        max_residual_ms=cfg.max_residual_ms,
        min_arrival_span_ms=cfg.min_arrival_span_ms,
        min_distance_span_mm=cfg.min_distance_span_mm,
        max_speed_mm_per_s=cfg.max_speed_mm_per_s,
    )
    valid = np.asarray(fitted["valid"], dtype=bool)
    speeds = np.asarray(fitted["speed_mm_per_s"], dtype=float)
    return {
        "median_speed_mm_per_s": float(np.nanmedian(speeds[valid])) if np.any(valid) else np.nan,
        "n_valid": int(np.count_nonzero(valid)),
    }


def compute_front_null_support(
    local_df,
    event_speed_df,
    *,
    null_types=("coordinate_shuffle", "arrival_time_shuffle"),
    reps: int = 30,
    seed: int = 1,
    max_events_per_well=25,
    cfg: FrontNullConfig = FrontNullConfig(),
    cache_path=None,
    force_recompute: bool = False,
):
    """Per-event spatial-null support table (``well, event_idx, null_type, null_rep, n_valid, ...``).

    Optionally cached to ``cache_path`` (CSV). Reuses the cached per-event local arrival tables in
    ``local_df`` (columns ``well, event_idx, x_mm, y_mm, arrival_time_s`` and optional amplitude).
    """
    if cache_path is not None and Path(cache_path).exists() and not force_recompute:
        return pd.read_csv(cache_path)

    rng = np.random.default_rng(int(seed))
    observed = event_speed_df.rename(columns={"n_valid_local_fits": "observed_n_valid"})[
        ["well", "event_idx", "median_speed_mm_per_s", "observed_n_valid"]
    ].rename(columns={"median_speed_mm_per_s": "observed_median_speed_mm_per_s"})
    selected_parts = []
    for _well, group in observed.sort_values(["well", "event_idx"]).groupby("well", sort=True):
        if max_events_per_well is None or len(group) <= int(max_events_per_well):
            selected_parts.append(group)
        else:
            pick = np.round(np.linspace(0, len(group) - 1, int(max_events_per_well))).astype(int)
            selected_parts.append(group.iloc[pick])
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else observed.iloc[:0].copy()
    selected_keys = set(zip(selected["well"].astype(int), selected["event_idx"].astype(int)))
    grouped_events = [
        (key, group)
        for key, group in local_df.groupby(["well", "event_idx"], sort=False)
        if (int(key[0]), int(key[1])) in selected_keys
    ]
    rows = []
    for null_type in null_types:
        for rep in range(int(reps)):
            for (well, event_idx), group in grouped_events:
                fit = event_null_arrival_fit(group, rng, null_type=null_type, cfg=cfg)
                rows.append({
                    "null_type": null_type,
                    "null_rep": int(rep),
                    "well": int(well),
                    "event_idx": int(event_idx),
                    **fit,
                })
    front_null_df = pd.DataFrame(rows)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        front_null_df.to_csv(cache_path, index=False)
    return front_null_df


def front_significant_events(event_speed_df, front_null_df, *, support_quantile: float = 0.95):
    """Annotate ``event_speed_df`` with the null-corrected ``front_significant`` flag.

    Per-well support threshold = max over null types of the ``support_quantile`` quantile of the null
    ``n_valid``; an event is front-significant when its observed ``n_valid_local_fits`` >= that threshold.
    Returns ``event_speed_df`` with added ``n_valid``, ``n_valid_significance_threshold`` and
    ``front_significant`` columns (caller applies any further finite-speed filter).
    """
    support_thresholds = (
        front_null_df
        .groupby(["well", "null_type"], as_index=False)["n_valid"]
        .quantile(support_quantile)
        .rename(columns={"n_valid": "n_valid_q"})
    )
    support_threshold_by_well = (
        support_thresholds
        .groupby("well", as_index=False)
        .agg(n_valid_significance_threshold=("n_valid_q", "max"))
    )
    corrected = event_speed_df.rename(columns={"n_valid_local_fits": "n_valid"}).merge(
        support_threshold_by_well, on="well", how="left"
    )
    corrected["front_significant"] = corrected["n_valid"] >= corrected["n_valid_significance_threshold"]
    return corrected


def _as_dataframe(payload, *, well, path, group_name):
    columns = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            columns[key] = value
    if not columns:
        return pd.DataFrame()
    lengths = {key: len(value) for key, value in columns.items()}
    first_len = next(iter(lengths.values()))
    keep = {key: value for key, value in columns.items() if len(value) == first_len}
    frame = pd.DataFrame(keep)
    frame.insert(0, "well", int(well))
    frame.insert(1, "cache_path", str(path))
    frame.insert(2, "group_name", group_name)
    return frame


def _calibration_frames(payload, *, well, path):
    attrs = {key: value for key, value in payload.items() if not isinstance(value, np.ndarray)}
    config = json.loads(attrs.get("config", "{}")) if attrs.get("config") else {}
    centers = np.asarray(payload["phase_bin_centers_rad"], dtype=float)
    rel_prob = np.asarray(payload["relative_spike_probability"], dtype=float)
    occupancy = np.asarray(payload["occupancy_counts"], dtype=float)
    spike_counts = np.asarray(payload["spike_phase_counts"], dtype=float)
    curve = pd.DataFrame({
        "well": int(well),
        "cache_path": str(path),
        "phase_rad": centers,
        "relative_spike_probability": rel_prob,
        "occupancy_counts": occupancy,
        "spike_phase_counts": spike_counts,
    })
    summary = {
        "well": int(well),
        "cache_path": str(path),
        "theta_excitable_rad": float(attrs.get("theta_excitable_rad", np.nan)),
        "n_spikes_sampled": int(attrs.get("n_spikes_sampled", 0)),
        "calibration_theta_resultant": float(attrs.get("calibration_theta_resultant", np.nan)),
        "calibration_theta_circular_std_rad": float(attrs.get("calibration_theta_circular_std_rad", np.nan)),
        "calibration_stable": bool(attrs.get("calibration_stable", False)),
        "calibration_interval_count": int(attrs.get("calibration_interval_count", 0)),
        "config": config,
    }
    return curve, summary


def load_all_wavefront_caches(cache_paths):
    """Read per-well wavefront caches into ``(calibration_df, calibration_summary_df, event_df, local_df)``."""
    calibration_curves = []
    calibration_summaries = []
    events = []
    locals_ = []
    for well, path in cache_paths.items():
        cache = read_lfp_wavefront_cache(path)
        curve, summary = _calibration_frames(cache["wavefront_calibration"], well=well, path=path)
        calibration_curves.append(curve)
        calibration_summaries.append(summary)
        event = _as_dataframe(cache["wavefront_events"], well=well, path=path, group_name="wavefront_events")
        local = _as_dataframe(cache["wavefront_local"], well=well, path=path, group_name="wavefront_local")
        events.append(event)
        locals_.append(local)
    calibration_df = pd.concat(calibration_curves, ignore_index=True)
    calibration_summary_df = pd.DataFrame(calibration_summaries)
    event_df = pd.concat(events, ignore_index=True)
    local_df = pd.concat(locals_, ignore_index=True)
    return calibration_df, calibration_summary_df, event_df, local_df
