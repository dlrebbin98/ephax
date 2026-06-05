from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import h5py
import numpy as np
import pandas as pd


def normalize_interval_array(intervals) -> np.ndarray:
    """Return sorted ``(start, end)`` intervals as a finite float array."""
    if intervals is None:
        return np.empty((0, 2), dtype=float)
    if isinstance(intervals, pd.DataFrame):
        if not {"start_time_s", "end_time_s"}.issubset(intervals.columns):
            raise ValueError("intervals DataFrame requires start_time_s and end_time_s columns.")
        arr = intervals[["start_time_s", "end_time_s"]].to_numpy(dtype=float)
    else:
        arr = np.asarray(intervals, dtype=float)
        if arr.size == 0:
            return np.empty((0, 2), dtype=float)
        arr = arr.reshape(-1, 2)
    finite = np.isfinite(arr).all(axis=1) & (arr[:, 1] > arr[:, 0])
    arr = arr[finite]
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    return arr[np.argsort(arr[:, 0], kind="mergesort")]


def concatenate_spike_times_over_intervals(spike_times_s, intervals) -> tuple[np.ndarray, float]:
    """Restrict spike times to intervals and concatenate intervals without gaps."""
    spike_times = np.sort(np.asarray(spike_times_s, dtype=float))
    intervals_arr = normalize_interval_array(intervals)
    if spike_times.size == 0 or intervals_arr.size == 0:
        return np.empty(0, dtype=float), float(np.sum(intervals_arr[:, 1] - intervals_arr[:, 0]))

    chunks: list[np.ndarray] = []
    offset = 0.0
    for start_s, end_s in intervals_arr:
        left = np.searchsorted(spike_times, start_s, side="left")
        right = np.searchsorted(spike_times, end_s, side="left")
        if right > left:
            chunks.append((spike_times[left:right] - start_s) + offset)
        offset += float(end_s - start_s)
    if not chunks:
        return np.empty(0, dtype=float), offset
    return np.concatenate(chunks).astype(float, copy=False), offset


def spike_times_by_electrode_in_intervals(recording, electrodes, intervals) -> tuple[dict[int, np.ndarray], float]:
    """Return concatenated interval-relative spike times for each requested electrode."""
    electrodes = np.asarray(electrodes, dtype=int)
    spike_times = np.asarray(recording.spikes["time"], dtype=float)
    spike_electrodes = np.asarray(recording.spikes["electrode"], dtype=int)
    intervals_arr = normalize_interval_array(intervals)
    total_duration_s = float(np.sum(intervals_arr[:, 1] - intervals_arr[:, 0])) if intervals_arr.size else 0.0
    by_electrode: dict[int, np.ndarray] = {}
    for electrode in electrodes:
        times, _ = concatenate_spike_times_over_intervals(spike_times[spike_electrodes == int(electrode)], intervals_arr)
        by_electrode[int(electrode)] = times
    return by_electrode, total_duration_s


def sttc_tiling_proportion(spike_times_s, duration_s: float, dt_s: float) -> float:
    """Fraction of recording time within dt_s of at least one spike."""
    times = np.sort(np.asarray(spike_times_s, dtype=float))
    duration_s = float(duration_s)
    dt_s = float(dt_s)
    if duration_s <= 0 or times.size == 0 or dt_s < 0:
        return 0.0
    starts = np.maximum(0.0, times - dt_s)
    ends = np.minimum(duration_s, times + dt_s)
    order = np.argsort(starts, kind="mergesort")
    starts = starts[order]
    ends = ends[order]
    covered = 0.0
    current_start = float(starts[0])
    current_end = float(ends[0])
    for start, end in zip(starts[1:], ends[1:]):
        start = float(start)
        end = float(end)
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += max(0.0, current_end - current_start)
            current_start = start
            current_end = end
    covered += max(0.0, current_end - current_start)
    return min(1.0, covered / duration_s)


def sttc_spike_proportion(spike_times_a_s, spike_times_b_s, dt_s: float) -> float:
    """Proportion of spikes in A within dt_s of any spike in B."""
    a = np.sort(np.asarray(spike_times_a_s, dtype=float))
    b = np.sort(np.asarray(spike_times_b_s, dtype=float))
    if a.size == 0 or b.size == 0:
        return 0.0
    pos = np.searchsorted(b, a)
    nearest = np.full(a.shape, np.inf, dtype=float)
    right = pos < b.size
    nearest[right] = np.minimum(nearest[right], np.abs(b[pos[right]] - a[right]))
    left = pos > 0
    nearest[left] = np.minimum(nearest[left], np.abs(a[left] - b[pos[left] - 1]))
    return float(np.mean(nearest <= float(dt_s)))


def spike_time_tiling_coefficient(spike_times_a_s, spike_times_b_s, duration_s: float, dt_s: float) -> float:
    """Compute Cutts-Eglen spike time tiling coefficient for two spike trains."""
    a = np.sort(np.asarray(spike_times_a_s, dtype=float))
    b = np.sort(np.asarray(spike_times_b_s, dtype=float))
    if duration_s <= 0 or a.size == 0 or b.size == 0:
        return np.nan
    pa = sttc_spike_proportion(a, b, dt_s)
    pb = sttc_spike_proportion(b, a, dt_s)
    ta = sttc_tiling_proportion(a, duration_s, dt_s)
    tb = sttc_tiling_proportion(b, duration_s, dt_s)

    denom_a = 1.0 - pa * tb
    denom_b = 1.0 - pb * ta
    term_a = 0.0 if np.isclose(denom_a, 0.0) else (pa - tb) / denom_a
    term_b = 0.0 if np.isclose(denom_b, 0.0) else (pb - ta) / denom_b
    return float(0.5 * (term_a + term_b))


def compute_sttc_adjacency(
    spike_times_by_electrode: Mapping[int, np.ndarray],
    duration_s: float,
    *,
    dt_ms: float = 10.0,
    min_rate_hz: float = 0.01,
    keep_negative: bool = False,
) -> dict[str, np.ndarray]:
    """Compute a weighted undirected STTC adjacency matrix."""
    electrodes_all = np.asarray(list(spike_times_by_electrode.keys()), dtype=int)
    rates_all = np.asarray(
        [len(spike_times_by_electrode[int(e)]) / duration_s if duration_s > 0 else 0.0 for e in electrodes_all],
        dtype=float,
    )
    keep = np.isfinite(rates_all) & (rates_all >= float(min_rate_hz))
    electrodes = electrodes_all[keep]
    rates = rates_all[keep]
    n = electrodes.size
    adjacency = np.zeros((n, n), dtype=np.float32)
    dt_s = float(dt_ms) / 1000.0

    tiling = {
        int(e): sttc_tiling_proportion(spike_times_by_electrode[int(e)], duration_s, dt_s)
        for e in electrodes
    }
    for i in range(n):
        a = spike_times_by_electrode[int(electrodes[i])]
        ta = tiling[int(electrodes[i])]
        for j in range(i + 1, n):
            b = spike_times_by_electrode[int(electrodes[j])]
            tb = tiling[int(electrodes[j])]
            pa = sttc_spike_proportion(a, b, dt_s)
            pb = sttc_spike_proportion(b, a, dt_s)
            denom_a = 1.0 - pa * tb
            denom_b = 1.0 - pb * ta
            term_a = 0.0 if np.isclose(denom_a, 0.0) else (pa - tb) / denom_a
            term_b = 0.0 if np.isclose(denom_b, 0.0) else (pb - ta) / denom_b
            value = 0.5 * (term_a + term_b)
            if not keep_negative:
                value = max(0.0, value)
            adjacency[i, j] = adjacency[j, i] = np.float32(value)
    return {"electrodes": electrodes, "rates_hz": rates, "adjacency": adjacency}


def thresholded_edge_table(adjacency, electrodes, coords=None, *, top_fraction: float = 0.05) -> pd.DataFrame:
    """Return strongest undirected edges from a weighted adjacency matrix."""
    adjacency = np.asarray(adjacency, dtype=float)
    electrodes = np.asarray(electrodes, dtype=int)
    if adjacency.shape[0] != adjacency.shape[1] or adjacency.shape[0] != electrodes.size:
        raise ValueError("adjacency must be square and match electrodes.")
    i, j = np.triu_indices(electrodes.size, k=1)
    weights = adjacency[i, j]
    finite = np.isfinite(weights) & (weights > 0)
    i = i[finite]
    j = j[finite]
    weights = weights[finite]
    if weights.size == 0:
        return pd.DataFrame(columns=["source_electrode", "target_electrode", "weight", "distance_um"])
    n_keep = max(1, int(np.ceil(float(top_fraction) * weights.size)))
    order = np.argsort(weights)[::-1][:n_keep]
    out = {
        "source_electrode": electrodes[i[order]].astype(int),
        "target_electrode": electrodes[j[order]].astype(int),
        "weight": weights[order].astype(float),
    }
    if coords is not None:
        coords_arr = np.asarray(coords, dtype=float)
        if coords_arr.shape[0] == electrodes.size:
            delta = coords_arr[i[order]] - coords_arr[j[order]]
            out["distance_um"] = np.sqrt(np.sum(delta * delta, axis=1))
    return pd.DataFrame(out)


def write_sttc_network_cache(
    path,
    *,
    adjacency,
    electrodes,
    coords_um,
    rates_hz,
    intervals,
    config: Optional[Mapping[str, object]] = None,
) -> Path:
    """Write an STTC network cache to HDF5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    intervals_arr = normalize_interval_array(intervals)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("adjacency", data=np.asarray(adjacency, dtype=np.float32), compression="gzip")
        h5.create_dataset("electrodes", data=np.asarray(electrodes, dtype=np.int64))
        h5.create_dataset("coords_um", data=np.asarray(coords_um, dtype=np.float32))
        h5.create_dataset("rates_hz", data=np.asarray(rates_hz, dtype=np.float32))
        h5.create_dataset("intervals_s", data=intervals_arr.astype(np.float64))
        if config:
            for key, value in config.items():
                if value is None:
                    continue
                h5.attrs[str(key)] = value
    return path


def read_sttc_network_cache(path) -> dict[str, Any]:
    """Read an STTC network cache from HDF5."""
    with h5py.File(path, "r") as h5:
        return {
            "adjacency": h5["adjacency"][:],
            "electrodes": h5["electrodes"][:],
            "coords_um": h5["coords_um"][:],
            "rates_hz": h5["rates_hz"][:],
            "intervals_s": h5["intervals_s"][:],
            "attrs": dict(h5.attrs),
        }
