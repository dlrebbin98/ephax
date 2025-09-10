from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BinnedSeries:
    centers: np.ndarray
    mean: np.ndarray
    stderr: np.ndarray


@dataclass
class IFRPeaks:
    values: np.ndarray           # possibly log10-transformed
    kde_x: np.ndarray
    kde_y: np.ndarray
    peaks_x: np.ndarray          # x positions in same domain as values
    peaks_y: np.ndarray          # densities at peaks
    peaks_hz: np.ndarray         # peaks converted to Hz if values are log10


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
