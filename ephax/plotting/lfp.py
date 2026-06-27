from __future__ import annotations

from io import BytesIO
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def save_lfp_phase_gif(
    path: str | Path,
    phase: np.ndarray,
    time_s: np.ndarray,
    coords_mm: np.ndarray,
    *,
    anchor_time_s: float | None = None,
    frame_indices: np.ndarray | None = None,
    frame_step: int = 1,
    electrodes: np.ndarray | None = None,
    spike_times_s: np.ndarray | None = None,
    spike_electrodes: np.ndarray | None = None,
    spike_half_window_s: float | None = None,
    amplitude_alpha: np.ndarray | None = None,
    dot_size: float = 12.0,
    duration_s: float = 0.14,
    dpi: int = 140,
    title_prefix: str = "LFP phase",
) -> Path:
    """Render an HD-MEA LFP phase animation from precomputed phase values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    phase = np.asarray(phase, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    coords_mm = np.asarray(coords_mm, dtype=float)
    if phase.ndim != 2:
        raise ValueError("phase must have shape (n_times, n_channels)")
    if coords_mm.shape != (phase.shape[1], 2):
        raise ValueError("coords_mm must have shape (n_channels, 2)")
    if time_s.shape != (phase.shape[0],):
        raise ValueError("time_s must have shape (n_times,)")
    if frame_indices is None:
        frame_indices = np.arange(0, phase.shape[0], max(1, int(frame_step)), dtype=int)
    else:
        frame_indices = np.asarray(frame_indices, dtype=int)
    frame_indices = frame_indices[(frame_indices >= 0) & (frame_indices < phase.shape[0])]
    if frame_indices.size == 0:
        raise ValueError("No valid frame indices were provided")

    x_um = coords_mm[:, 0] * 1000.0
    y_um = coords_mm[:, 1] * 1000.0
    x_pad = max(50.0, 0.03 * float(np.ptp(x_um))) if x_um.size else 50.0
    y_pad = max(50.0, 0.03 * float(np.ptp(y_um))) if y_um.size else 50.0
    alpha = np.ones(phase.shape[1], dtype=float) if amplitude_alpha is None else np.asarray(amplitude_alpha, dtype=float)
    alpha = np.clip(np.nan_to_num(alpha, nan=0.2), 0.2, 1.0)
    if alpha.shape != (phase.shape[1],):
        raise ValueError("amplitude_alpha must have shape (n_channels,)")

    electrodes = np.arange(phase.shape[1]) if electrodes is None else np.asarray(electrodes, dtype=int)
    electrode_to_idx = {int(electrode): idx for idx, electrode in enumerate(electrodes)}
    spike_times_s = np.asarray([] if spike_times_s is None else spike_times_s, dtype=float)
    spike_electrodes = np.asarray([] if spike_electrodes is None else spike_electrodes, dtype=int)
    if spike_half_window_s is None:
        if time_s.size > 1:
            spike_half_window_s = 0.5 * float(np.nanmedian(np.diff(time_s)))
        else:
            spike_half_window_s = 0.001

    norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = plt.get_cmap("twilight")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    with imageio.get_writer(path, mode="I", duration=float(duration_s)) as writer:
        for frame_idx in frame_indices:
            frame_time = float(time_s[int(frame_idx)])
            spike_mask = np.zeros(phase.shape[1], dtype=bool)
            if spike_times_s.size and spike_electrodes.size:
                in_frame = (spike_times_s >= frame_time - spike_half_window_s) & (
                    spike_times_s < frame_time + spike_half_window_s
                )
                for electrode in np.unique(spike_electrodes[in_frame]):
                    idx = electrode_to_idx.get(int(electrode))
                    if idx is not None:
                        spike_mask[idx] = True
            edge_colors = np.tile(np.array([[0.06, 0.06, 0.06, 0.55]]), (phase.shape[1], 1))
            edge_colors[spike_mask] = np.array([1.0, 1.0, 1.0, 1.0])
            line_widths = np.full(phase.shape[1], 0.25)
            line_widths[spike_mask] = 1.3

            fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=False)
            fig.subplots_adjust(left=0.10, right=0.84, bottom=0.13, top=0.88)
            cax = fig.add_axes([0.87, 0.18, 0.025, 0.64])
            ax.scatter(
                x_um,
                y_um,
                c=np.angle(np.exp(1j * phase[int(frame_idx)])),
                s=float(dot_size),
                cmap=cmap,
                norm=norm,
                edgecolors=edge_colors,
                linewidths=line_widths,
                alpha=alpha,
            )
            ax.set_facecolor("black")
            ax.set_xlim(float(np.nanmin(x_um) - x_pad), float(np.nanmax(x_um) + x_pad))
            ax.set_ylim(float(np.nanmin(y_um) - y_pad), float(np.nanmax(y_um) + y_pad))
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x (um)")
            ax.set_ylabel("y (um)")
            if anchor_time_s is None:
                ax.set_title(f"{title_prefix} | t = {frame_time:.3f} s")
            else:
                rel_ms = (frame_time - float(anchor_time_s)) * 1000.0
                ax.set_title(f"{title_prefix} | t = {rel_ms:+.0f} ms | spiking electrodes = {np.count_nonzero(spike_mask)}")
            cbar = fig.colorbar(mappable, cax=cax, ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            cbar.ax.set_yticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
            cbar.set_label("Hilbert phase (rad)")
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=int(dpi))
            buf.seek(0)
            writer.append_data(imageio.imread(buf))
            buf.close()
            plt.close(fig)
    return path


__all__ = ["save_lfp_phase_gif"]
