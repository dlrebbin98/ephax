from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message="Intel MKL WARNING")
warnings.filterwarnings("ignore", message="RuntimeWarning: overflow encountered")

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Literal, Optional

import numpy as np

from .models import Layout

# Reuse existing loaders for now; centralize here
from ephax.helper_functions import load_spikes_data as _load_spikes_data
from ephax.helper_functions import load_spikes_npz as _load_spikes_npz
from ephax.helper_functions import get_activity_sorted_electrodes as _get_activity_sorted_electrodes


@dataclass
class PrepConfig:
    # Common
    min_amp: float = 0.0             # passed to loaders when applicable
    # Mode: 'threshold' uses per-bin activity; 'top' uses get_activity_sorted_electrodes; 'selected' uses provided ids
    mode: Literal["threshold", "top", "selected"] = "threshold"
    # Threshold mode
    bin_width: float = 0.01          # seconds
    active_threshold: float = 0.1    # percent of bins with activity
    # Top mode
    top_start: int = 0
    top_stop: Optional[int] = None
    top_use_recording_window: bool = True
    top_start_time: float = 0.0
    top_end_time: float = 600.0
    # Logging
    verbose: bool = True
    # Selected mode
    selected_refs: Optional[List[int]] = None


@dataclass
class Recording:
    """Windowed recording with spikes, layout, and sampling/window metadata."""
    spikes: dict
    layout: dict
    start_time: float
    end_time: float
    sf: float

    def to_legacy(self) -> Tuple[dict, dict, float, float]:
        return self.spikes, self.layout, self.start_time, self.end_time

    def windowed_spikes(self) -> dict:
        """Return spikes dict restricted to [start_time, end_time]."""
        import numpy as np
        mask = (self.spikes["time"] >= self.start_time) & (self.spikes["time"] <= self.end_time)
        return {k: v[mask] for k, v in self.spikes.items()}

    def refs_top(self, start: int = 10, stop: Optional[int] = 210) -> np.ndarray:
        """Legacy-like top-active electrodes within [start_time, end_time]."""
        s = float(self.start_time)
        e = float(self.end_time)
        sel = _get_activity_sorted_electrodes([self.spikes], start=start, stop=stop, start_time=s, end_time=e)
        return np.asarray(sel, dtype=int)

    def active_threshold(self, bin_width: float = 0.01, pct: float = 0.1) -> np.ndarray:
        """Lyapunov-like activity filter within [start_time, end_time]."""
        import numpy as np
        spikes = self.windowed_spikes()
        electrodes = np.unique(spikes["electrode"]) if len(spikes["electrode"]) else np.array([], dtype=int)
        if electrodes.size == 0:
            return electrodes
        bins = np.arange(self.start_time, self.end_time + bin_width, bin_width)
        selected = []
        for e in electrodes:
            t = spikes["time"][spikes["electrode"] == e]
            if t.size == 0:
                continue
            counts, _ = np.histogram(t, bins=bins)
            rates = counts / bin_width
            active_bins = np.sum(rates > 0)
            if rates.size and (active_bins / rates.size * 100) > pct:
                selected.append(int(e))
        return np.asarray(selected, dtype=int)

    def randomize_electrode_mapping(
        self,
        seed: Optional[int] = None,
        rng: np.random.Generator | None = None,
        inplace: bool = False,
        shuffle_layout_coordinates: bool = False,
    ) -> "Recording":
        """Shuffle channel→electrode assignments and optionally scramble layout coordinates.

        A permutation is applied so each channel keeps its spike train but gets
        reassigned to a random electrode label. When ``shuffle_layout_coordinates``
        is True, the electrode labels are additionally attached to random
        ``(x, y)`` coordinate pairs, breaking the original spatial arrangement.
        The spikes' ``electrode`` labels are regenerated from the shuffled
        layout to ensure consistency. If ``inplace`` is False (default) a new
        :class:`Recording` is returned.
        """
        import numpy as np

        if rng is not None and seed is not None:
            raise ValueError("Provide either 'seed' or 'rng', not both.")
        if rng is None:
            rng = np.random.default_rng(seed)

        required_layout_keys = {"channel", "electrode", "x", "y"}
        missing_layout = required_layout_keys - set(self.layout.keys())
        if missing_layout:
            raise KeyError(f"Recording layout missing keys: {sorted(missing_layout)}")
        if "channel" not in self.spikes:
            raise KeyError("Recording spikes missing 'channel' field.")

        layout = self.layout if inplace else self.layout.copy()
        spikes = self.spikes if inplace else self.spikes.copy()

        channels = np.asarray(layout["channel"])
        if channels.size == 0:
            return self if inplace else Recording(spikes=spikes, layout=layout, start_time=self.start_time, end_time=self.end_time, sf=self.sf)

        perm_indices = rng.permutation(channels.size)
        for key in ("electrode", "x", "y"):
            layout[key] = np.asarray(layout[key])[perm_indices]

        if shuffle_layout_coordinates and channels.size > 1:
            coord_perm = rng.permutation(channels.size)
            layout["x"] = np.asarray(layout["x"])[coord_perm]
            layout["y"] = np.asarray(layout["y"])[coord_perm]

        channel_to_electrode = dict(zip(channels, layout["electrode"]))
        try:
            spikes["electrode"] = np.asarray([channel_to_electrode[ch] for ch in np.asarray(spikes["channel"])], dtype=layout["electrode"].dtype)
        except KeyError as exc:
            raise KeyError(f"Spike channel {exc.args[0]} not present in layout after randomization.") from exc

        if inplace:
            self.layout = layout
            self.spikes = spikes
            return self

        return Recording(
            spikes=spikes,
            layout=layout,
            start_time=self.start_time,
            end_time=self.end_time,
            sf=self.sf,
        )


@dataclass
class RestingActivityDataset:
    recordings: List[Recording]
    sf: float | None = None

    @classmethod
    def from_file_info(
        cls,
        file_info: Iterable[tuple],
        source: Literal["h5", "npz"] = "h5",
        min_amp: float = 0.0,
    ) -> "RestingActivityDataset":
        if source == "h5":
            sf, spikes_data_list, layout_list, start_times, end_times = _load_spikes_data(file_info, min_amp=min_amp)
        elif source == "npz":
            sf, spikes_data_list, layout_list, start_times, end_times = _load_spikes_npz(file_info, min_amp=min_amp)
        else:
            raise ValueError("source must be 'h5' or 'npz'")

        recs: List[Recording] = []
        for sd, l, st, et in zip(spikes_data_list, layout_list, start_times, end_times):
            recs.append(Recording(spikes=sd, layout=l, start_time=float(st), end_time=float(et), sf=float(sf)))
        return cls(recordings=recs, sf=sf)

    def to_legacy(self) -> Tuple[list[dict], list[dict], list[float], list[float]]:
        spikes_list: List[dict] = []
        layout_list: List[dict] = []
        starts: List[float] = []
        ends: List[float] = []
        for r in self.recordings:
            s, l, st, et = r.to_legacy()
            spikes_list.append(s)
            layout_list.append(l)
            starts.append(st)
            ends.append(et)
        return spikes_list, layout_list, starts, ends

    def select_ref_electrodes(self, cfg: PrepConfig) -> List[np.ndarray]:
        """Compute selected reference electrodes per recording without filtering spikes.

        Returns a list of arrays of electrode ids, one per recording.
        """
        refs: List[np.ndarray] = []
        for rec in self.recordings:
            if cfg.mode == "threshold":
                self._verbose_flag = bool(cfg.verbose)
                sel = rec.active_threshold(bin_width=cfg.bin_width, pct=cfg.active_threshold)
                self._verbose_flag = False
            elif cfg.mode == "top":
                s_time, e_time = (rec.start_time, rec.end_time) if cfg.top_use_recording_window else (cfg.top_start_time, cfg.top_end_time)
                sel = rec.refs_top(start=cfg.top_start, stop=cfg.top_stop)
                if cfg.verbose:
                    total_elec = np.unique(rec.spikes["electrode"]).size
                    print("\nSelecting top electrodes by activity (refs only)...")
                    print(f"Total electrodes: {total_elec}")
                    print(f"Top range: start={cfg.top_start}, stop={cfg.top_stop}")
                    print(f"Window: [{s_time}, {e_time}] s")
                    print(f"Selected electrodes: {sel.size}")
            elif cfg.mode == "selected":
                provided = np.asarray(cfg.selected_refs or [], dtype=int)
                if provided.size == 0:
                    sel = np.unique(rec.spikes.get('electrode', np.array([], dtype=int)))
                else:
                    present = np.unique(rec.spikes.get('electrode', np.array([], dtype=int)))
                    sel = np.intersect1d(provided, present, assume_unique=False)
                if cfg.verbose:
                    print("\nUsing provided selected electrodes (refs only)...")
                    print(f"Provided: {len(cfg.selected_refs or [])}, Present in recording: {sel.size}")
            else:
                raise ValueError("Unsupported mode in PrepConfig: %r" % (cfg.mode,))
            refs.append(sel.astype(int))
        return refs

    def filter_active(self, cfg: PrepConfig) -> "RestingActivityDataset":
        filtered: List[Recording] = []
        for rec in self.recordings:
            if cfg.mode == "threshold":
                # Enable verbose prints inside selection
                self._verbose_flag = bool(cfg.verbose)
                sel = rec.active_threshold(bin_width=cfg.bin_width, pct=cfg.active_threshold)
                self._verbose_flag = False
            elif cfg.mode == "top":
                sel = rec.refs_top(start=cfg.top_start, stop=cfg.top_stop)
            elif cfg.mode == "selected":
                provided = np.asarray(cfg.selected_refs or [], dtype=int)
                if provided.size == 0:
                    sel = np.unique(rec.spikes.get('electrode', np.array([], dtype=int)))
                else:
                    present = np.unique(rec.spikes.get('electrode', np.array([], dtype=int)))
                    sel = np.intersect1d(provided, present, assume_unique=False)
            if cfg.verbose:
                total_elec = np.unique(rec.spikes["electrode"]).size
                print("\nSelecting top electrodes by activity...")
                print(f"Total electrodes: {total_elec}")
                if cfg.mode == "top":
                    print(f"Top range: start={cfg.top_start}, stop={cfg.top_stop}")
                print(f"Selected electrodes: {sel.size}")
            else:
                raise ValueError("Unsupported mode in PrepConfig: %r" % (cfg.mode,))

            if sel.size == 0:
                mask = np.zeros_like(rec.spikes["electrode"], dtype=bool)
            else:
                mask = np.isin(rec.spikes["electrode"], sel)
            spikes_f = {k: v[mask] for k, v in rec.spikes.items()}
            filtered.append(Recording(spikes=spikes_f, layout=rec.layout, start_time=rec.start_time, end_time=rec.end_time, sf=rec.sf))
        return RestingActivityDataset(
            recordings=filtered,
            sf=self.sf,
        )
