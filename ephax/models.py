from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BinnedSeries:
    centers: np.ndarray
    mean: np.ndarray
    stderr: np.ndarray


@dataclass
class FRDistanceResult:
    distances: np.ndarray
    rates: np.ndarray
    bins: np.ndarray
    binned: BinnedSeries


@dataclass
class CofiringDistanceResult:
    distances: np.ndarray
    proportions: np.ndarray
    bins: np.ndarray
    binned: BinnedSeries


@dataclass
class PopulationIFR:
    time_grid: np.ndarray
    electrodes: np.ndarray
    ifr_matrix: np.ndarray
    mean_ifr: np.ndarray
    mean_ifr_smooth: np.ndarray
    per_electrode_mean_hz: np.ndarray


@dataclass
class HighResTraces:
    bin_edges_s: np.ndarray
    time_centers_s: np.ndarray
    electrodes: np.ndarray
    per_electrode_rate_hz: np.ndarray
    population_rate_hz: np.ndarray
    spikes_by_electrode: dict[int, np.ndarray]
    spike_presence: np.ndarray


@dataclass
class NetworkActivityState:
    bin_edges_s: np.ndarray
    time_centers_s: np.ndarray
    electrodes: np.ndarray
    per_electrode_rate_hz: np.ndarray
    active_mask: np.ndarray
    spike_counts: np.ndarray
    active_electrode_counts: np.ndarray
    total_spike_counts: np.ndarray
    participation_fraction: np.ndarray
    population_activity_hz: np.ndarray
    network_score: np.ndarray
    electrode_thresholds_hz: np.ndarray


@dataclass
class AlignedBurstEvents:
    relative_time_ms: np.ndarray
    population_windows: np.ndarray
    aligned_rate: np.ndarray
    aligned_spikes: np.ndarray
    valid_anchors: object
    electrodes: np.ndarray


@dataclass
class WaveAnalysisResult:
    event_direction: object
    trace: object
    peaks: object
    bin_summary: object
    fit_summary: object
    heatmap: object
    bootstrap_speeds: np.ndarray


@dataclass
class DiscreteTEResult:
    delay_centers_ms: np.ndarray
    signed_dx_centers_um: np.ndarray
    conditional_probability: np.ndarray
    raw_te_bits: np.ndarray
    bias_corrected_te_bits: np.ndarray
    te_pvalue: np.ndarray
    effective_observations: np.ndarray
    trigger_summary: object
    observation_summary: object
    ridge_summary: object
    fit_summary: object
    score: float


@dataclass
class IFRPeaks:
    values: np.ndarray           # possibly log10-transformed
    kde_x: np.ndarray
    kde_y: np.ndarray
    peaks_x: np.ndarray          # x positions in same domain as values
    peaks_y: np.ndarray          # densities at peaks
    peaks_hz: np.ndarray         # peaks converted to Hz if values are log10


@dataclass
class IFRTimeSeriesPanel:
    recording_index: int
    start_time: float
    end_time: float
    electrodes: np.ndarray
    time_points: np.ndarray
    heatmap: np.ndarray
    histogram_values: np.ndarray
    log_scale: bool


@dataclass
class GMMFit:
    means_hz: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    p_value: float | None = None


@dataclass
class CofiringHeatmap:
    Z: np.ndarray                # shape: (len(delays)-1, len(distance_bins)-1)
    distance_bins: np.ndarray
    delays: np.ndarray


@dataclass
class ModelCurve:
    r_um: np.ndarray
    curve: np.ndarray
    upper: np.ndarray | None = None
    lower: np.ndarray | None = None


# New core data models for refactor
@dataclass
class Layout:
    channel: np.ndarray
    electrode: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @staticmethod
    def from_legacy(layout: dict) -> "Layout":
        return Layout(
            channel=np.asarray(layout["channel"]),
            electrode=np.asarray(layout["electrode"]),
            x=np.asarray(layout["x"]),
            y=np.asarray(layout["y"]),
        )

    def to_legacy(self) -> dict:
        return {
            "channel": np.asarray(self.channel),
            "electrode": np.asarray(self.electrode),
            "x": np.asarray(self.x),
            "y": np.asarray(self.y),
        }


@dataclass
class Recording:
    time: np.ndarray
    channel: np.ndarray
    amplitude: np.ndarray
    electrode: np.ndarray

    # Optional metadata fields can be added later as needed

    @staticmethod
    def from_legacy(spikes_data: dict, layout: dict | None = None) -> "Recording":
        # layout not strictly required here but included for symmetry/validation later
        return Recording(
            time=np.asarray(spikes_data["time"]),
            channel=np.asarray(spikes_data["channel"]),
            amplitude=np.asarray(spikes_data["amplitude"]),
            electrode=np.asarray(spikes_data["electrode"]),
        )

    def to_legacy(self) -> dict:
        return {
            "time": np.asarray(self.time),
            "channel": np.asarray(self.channel),
            "amplitude": np.asarray(self.amplitude),
            "electrode": np.asarray(self.electrode),
        }
