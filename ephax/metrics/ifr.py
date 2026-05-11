from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from ..models import IFRPeaks, IFRTimeSeriesPanel


def calculate_ifr(spikes_data, selected_electrodes, start_time=None, end_time=None):
    """Compute instantaneous firing rate per electrode as a step function."""
    electrode_spikes = {int(el): [] for el in selected_electrodes}
    for t, el in zip(spikes_data["time"], spikes_data["electrode"]):
        if int(el) in electrode_spikes:
            electrode_spikes[int(el)].append(float(t))

    for el in electrode_spikes:
        electrode_spikes[el] = np.asarray(electrode_spikes[el], dtype=float)

    if start_time is None:
        start_time = float(np.min(spikes_data["time"])) if len(spikes_data["time"]) else 0.0
    if end_time is None:
        end_time = float(np.max(spikes_data["time"])) if len(spikes_data["time"]) else 0.0

    def _calc(times):
        if times.size < 2:
            return np.array([start_time, end_time], dtype=float), np.array([0.0, 0.0], dtype=float)
        ifr_times = [start_time]
        ifr_values = [0.0]
        first = times[0]
        ifr_times.append(first)
        ifr_values.append(0.0)
        for i in range(times.size - 1):
            a = times[i]
            b = times[i + 1]
            interval = max(1e-12, b - a)
            val = 1.0 / interval
            ifr_times.extend([a, b])
            ifr_values.extend([val, val])
        last = times[-1]
        ifr_times.append(last)
        ifr_values.append(0.0)
        ifr_times.append(end_time)
        ifr_values.append(0.0)
        return np.asarray(ifr_times, dtype=float), np.asarray(ifr_values, dtype=float)

    ifr_data = {}
    total_firing = {}
    all_ifr_values = []
    for el, times in electrode_spikes.items():
        sel = (times >= float(start_time)) & (times <= float(end_time))
        times_sel = times[sel]
        if times_sel.size > 0:
            t_arr, v_arr = _calc(times_sel)
            ifr_data[el] = (t_arr, v_arr)
            duration = max(1e-12, float(end_time) - float(start_time))
            total_firing[el] = float(times_sel.size) / duration
            all_ifr_values.extend(v_arr.tolist())
    return ifr_data, total_firing, np.asarray(all_ifr_values, dtype=float)


def prepare_ifr_timeseries_panel(
    spikes_data: dict,
    selected_electrodes: Iterable[int],
    start_time: float,
    end_time: float,
    *,
    recording_index: int = 0,
    log_scale: bool = True,
    time_grid_hz: float = 100.0,
    max_time_points: int = 5000,
) -> IFRTimeSeriesPanel | None:
    """Prepare one recording's IFR heatmap and histogram values for plotting."""
    selected_electrodes = list(selected_electrodes)
    if len(selected_electrodes) == 0:
        return None

    ifr_data, _, all_ifr_values = calculate_ifr(spikes_data, selected_electrodes, start_time, end_time)
    duration = max(0.0, float(end_time) - float(start_time))
    target_hz = float(time_grid_hz) if time_grid_hz and time_grid_hz > 0 else 100.0
    n_points = int(max(1, min(duration * target_hz, float(max_time_points))))
    time_points = np.linspace(float(start_time), float(end_time), n_points, dtype=np.float32)

    valid_electrodes = []
    rows = []
    for electrode in selected_electrodes:
        electrode = int(electrode)
        if electrode not in ifr_data:
            continue

        times, values = ifr_data[electrode]
        if log_scale:
            values = np.where(values == 0, 1e-3, values)
            values = np.log10(values)
            values = np.where(np.isinf(values), -3, values)
        rows.append(np.interp(time_points, times.astype(np.float32), values).astype(np.float32))
        valid_electrodes.append(electrode)

    if len(rows) == 0:
        return None

    histogram_values = all_ifr_values.copy()
    if log_scale:
        histogram_values = histogram_values[histogram_values > 1e-3]
        histogram_values = np.log10(histogram_values)

    return IFRTimeSeriesPanel(
        recording_index=int(recording_index),
        start_time=float(start_time),
        end_time=float(end_time),
        electrodes=np.asarray(valid_electrodes, dtype=int),
        time_points=time_points,
        heatmap=np.vstack(rows).astype(np.float32),
        histogram_values=np.asarray(histogram_values, dtype=float),
        log_scale=bool(log_scale),
    )


def prepare_ifr_timeseries_panels(
    spikes_data_list: Iterable[dict],
    start_times: Iterable[float],
    end_times: Iterable[float],
    selected_electrodes_per_recording: Iterable[Iterable[int]],
    *,
    log_scale: bool = True,
    time_grid_hz: float = 100.0,
    max_time_points: int = 5000,
) -> list[IFRTimeSeriesPanel]:
    """Prepare IFR time-series panels for all recordings with valid IFR data."""
    panels = []
    for idx, (spikes_data, start_time, end_time, selected_electrodes) in enumerate(
        zip(spikes_data_list, start_times, end_times, selected_electrodes_per_recording)
    ):
        panel = prepare_ifr_timeseries_panel(
            spikes_data,
            selected_electrodes,
            start_time,
            end_time,
            recording_index=idx,
            log_scale=log_scale,
            time_grid_hz=time_grid_hz,
            max_time_points=max_time_points,
        )
        if panel is not None:
            panels.append(panel)
    return panels


def ifr_peaks(
    spikes_data_list: Iterable[dict],
    start_times: Iterable[float],
    end_times: Iterable[float],
    log_scale: bool = True,
    selected_refs_per_recording: Optional[Iterable[Iterable[int]]] = None,
) -> IFRPeaks:
    """Compute IFR values across datasets and estimate peaks via KDE."""
    spikes_data_list = list(spikes_data_list)
    start_times = list(start_times)
    end_times = list(end_times)
    assert len(spikes_data_list) == len(start_times) == len(end_times), "Mismatch in lengths of input lists."

    values = []
    for idx, (spikes_data, start_time, end_time) in enumerate(zip(spikes_data_list, start_times, end_times)):
        if selected_refs_per_recording is None:
            raise ValueError("selected_refs_per_recording must be provided for ifr_peaks")
        try:
            selected_electrodes = list(selected_refs_per_recording[idx])
        except Exception:
            selected_electrodes = []
        _, _, ifr_vals = calculate_ifr(spikes_data, selected_electrodes, start_time, end_time)
        values.extend(ifr_vals)

    values = np.asarray(values)
    if log_scale:
        values = values[values > 1e-3]
        values = np.log10(values)

    kde = gaussian_kde(values)
    x_grid = np.linspace(values.min(), values.max(), 1000)
    kde_values = kde.evaluate(x_grid)
    peaks_indices, _ = find_peaks(kde_values)
    peaks_x = x_grid[peaks_indices]
    peaks_y = kde_values[peaks_indices]
    peaks_hz = 10 ** peaks_x if log_scale else peaks_x

    return IFRPeaks(values=values, kde_x=x_grid, kde_y=kde_values, peaks_x=peaks_x, peaks_y=peaks_y, peaks_hz=peaks_hz)
