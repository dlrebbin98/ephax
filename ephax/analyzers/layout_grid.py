from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata

from ..prep import RestingActivityDataset


@dataclass
class GridResult:
    grid: np.ndarray
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    vmin: float
    vmax: float


class LayoutGridPlotter:
    """Plots average firing-rate grids on the 2D electrode layout."""

    def __init__(self, dataset: RestingActivityDataset) -> None:
        self.ds = dataset

    @staticmethod
    def _interpolate_grid(
        grid_avg: np.ndarray,
        counts: np.ndarray,
        x_min: float,
        y_min: float,
        grid_size: float,
        method: str = "linear",
    ) -> np.ndarray:
        """Fill missing grid cells via spatial interpolation.

        ``grid_avg`` holds averaged rates per bin; ``counts`` marks how many
        electrodes contributed to each bin. Missing cells (counts == 0) are
        interpolated in physical coordinate space. A nearest-neighbour fallback
        is used when the primary method cannot provide a value.
        """

        if grid_avg.size == 0:
            return grid_avg

        missing = counts == 0
        if not np.any(missing):
            return grid_avg

        x_centers = x_min + (np.arange(grid_avg.shape[0]) + 0.5) * grid_size
        y_centers = y_min + (np.arange(grid_avg.shape[1]) + 0.5) * grid_size
        grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")

        known_mask = ~missing
        points = np.column_stack((grid_x[known_mask], grid_y[known_mask]))
        values = grid_avg[known_mask]

        # Require at least three points to interpolate in 2D
        if points.shape[0] < 3:
            return grid_avg

        targets = np.column_stack((grid_x[missing], grid_y[missing]))

        interp_vals = griddata(points, values, targets, method=method)
        if np.any(np.isnan(interp_vals)):
            nearest = griddata(points, values, targets, method="nearest")
            nan_mask = np.isnan(interp_vals)
            interp_vals[nan_mask] = nearest[nan_mask]

        filled = grid_avg.copy()
        filled[missing] = interp_vals
        return filled

    # --------- pooled across recordings ---------
    def compute_grid_avghz_pooled(self, grid_size: float = 50.0, interpolate: bool = False) -> GridResult:
        # Accumulate per-electrode average Hz across recordings
        pooled: dict[int, float] = {}
        counts: dict[int, int] = {}
        # Assume consistent layout geometry across recordings; use the first
        layout_df_all = [pd.DataFrame(rec.layout) for rec in self.ds.recordings]
        layout_df = pd.concat(layout_df_all, ignore_index=True).drop_duplicates(subset=["electrode"])

        for rec in self.ds.recordings:
            s, e = rec.start_time, rec.end_time
            dur = float(e - s)
            if dur <= 0:
                continue
            sdf = pd.DataFrame(rec.spikes)
            mask = (sdf["time"] >= s) & (sdf["time"] <= e)
            sdf = sdf[mask]
            # Average Hz per electrode during window
            fr = sdf["electrode"].value_counts() / dur
            for elec, rate in fr.items():
                elec_i = int(elec)
                pooled[elec_i] = pooled.get(elec_i, 0.0) + float(rate)
                counts[elec_i] = counts.get(elec_i, 0) + 1

        if not pooled:
            raise ValueError("No firing data to pool.")

        avg_fr = {k: pooled[k] / counts[k] for k in pooled.keys()}

        # Filter layout to electrodes with rates
        valid_electrodes = set(avg_fr.keys())
        layout_df = layout_df[layout_df["electrode"].isin(valid_electrodes)].copy()
        # Build lists aligned to layout_df rows
        map_x = layout_df["x"].to_numpy()
        map_y = layout_df["y"].to_numpy()
        map_electrode = layout_df["electrode"].astype(int).to_numpy()
        map_rates = np.array([avg_fr[int(e)] for e in map_electrode], dtype=float)

        # Define grid extents
        x_n = int(np.ceil(float(map_x.max()) / grid_size)) - 1
        y_n = int(np.ceil(float(map_y.max()) / grid_size)) - 1
        x_min, x_max = 1.0, x_n * grid_size
        y_min, y_max = 1.0, y_n * grid_size
        x_bins = np.linspace(x_min, x_max, x_n + 1)
        y_bins = np.linspace(y_min, y_max, y_n + 1)

        grid = np.zeros((x_n, y_n), dtype=float)
        gc = np.zeros((x_n, y_n), dtype=float)
        # Bin electrode rates into grid cells
        for xi, yi, rate in zip(map_x, map_y, map_rates):
            ix = int(np.digitize(xi, x_bins) - 1)
            iy = int(np.digitize(yi, y_bins) - 1)
            if 0 <= ix < x_n and 0 <= iy < y_n:
                grid[ix, iy] += rate
                gc[ix, iy] += 1
        with np.errstate(invalid='ignore', divide='ignore'):
            grid_avg = np.divide(grid, gc, where=gc != 0)
        if interpolate:
            grid_avg = self._interpolate_grid(grid_avg, gc, x_min, y_min, grid_size)
        # Mark remaining empty cells as NaN to avoid LogNorm issues and visual blocks
        mask_empty = gc == 0
        if not interpolate:
            grid_avg[mask_empty] = np.nan
        # LogNorm safety
        valid_vals = grid_avg[np.isfinite(grid_avg) & (grid_avg > 1e-6)]
        if valid_vals.size == 0:
            vmin, vmax = 1e-6, 1e-6
        else:
            vmin, vmax = float(valid_vals.min()), float(valid_vals.max())
        return GridResult(grid=grid_avg, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, vmin=vmin, vmax=vmax)

    def plot_grid_avghz_pooled(self, grid_size: float = 50.0, interpolate: bool = False):
        res = self.compute_grid_avghz_pooled(grid_size=grid_size, interpolate=interpolate)
        fig, ax = plt.subplots(figsize=(10, 6))
        vmin = max(res.vmin, 1e-6)
        vmax = max(res.vmax, vmin)
        if vmax <= vmin:
            vmax = vmin * 1.01
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('magma').copy()
        cmap.set_bad('black')
        cax = ax.imshow(res.grid.T, origin='lower', cmap=cmap, norm=norm,
                        extent=[res.x_min, res.x_max, res.y_min, res.y_max])
        ax.set_xlabel('X-coordinate ($\\mu m$)')
        ax.set_ylabel('Y-coordinate ($\\mu m$)')
        ax.set_title('Average Firing Rate (pooled)')
        ax.set_facecolor('black')
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
        cbar.set_label('Average Firing Rate (Hz)')
        ax.grid(True)
        ax.set_aspect('equal')
        plt.show()
        return fig, ax

    # --------- per-recording panel ---------
    def compute_grid_avghz_per_recording(self, grid_size: float = 50.0, interpolate: bool = False) -> List[GridResult]:
        results: List[GridResult] = []
        # Establish extents from combined layout to keep consistent axes
        layout_df_all = [pd.DataFrame(rec.layout) for rec in self.ds.recordings]
        layout_all = pd.concat(layout_df_all, ignore_index=True)
        x_max_all = float(layout_all['x'].max())
        y_max_all = float(layout_all['y'].max())
        x_n_all = int(np.ceil(x_max_all / grid_size)) - 1
        y_n_all = int(np.ceil(y_max_all / grid_size)) - 1
        x_min, x_max = 1.0, x_n_all * grid_size
        y_min, y_max = 1.0, y_n_all * grid_size
        x_bins_all = np.linspace(x_min, x_max, x_n_all + 1)
        y_bins_all = np.linspace(y_min, y_max, y_n_all + 1)

        for rec in self.ds.recordings:
            s, e = rec.start_time, rec.end_time
            dur = float(e - s)
            if dur <= 0:
                continue
            sdf = pd.DataFrame(rec.spikes)
            ldf = pd.DataFrame(rec.layout)
            mask = (sdf['time'] >= s) & (sdf['time'] <= e)
            sdf = sdf[mask]
            fr = sdf['electrode'].value_counts() / dur
            # Map electrodes to positions and rates
            ldf_idx = ldf.set_index('electrode')
            xs, ys, rates = [], [], []
            for elec, rate in fr.items():
                elec = int(elec)
                if elec in ldf_idx.index:
                    pos = ldf_idx.loc[elec]
                    xs.append(float(pos['x']))
                    ys.append(float(pos['y']))
                    rates.append(float(rate))
            xs = np.asarray(xs); ys = np.asarray(ys); rates = np.asarray(rates)
            grid = np.zeros((x_n_all, y_n_all), dtype=float)
            gc = np.zeros((x_n_all, y_n_all), dtype=float)
            for xi, yi, r in zip(xs, ys, rates):
                ix = int(np.digitize(xi, x_bins_all) - 1)
                iy = int(np.digitize(yi, y_bins_all) - 1)
                if 0 <= ix < x_n_all and 0 <= iy < y_n_all:
                    grid[ix, iy] += r
                    gc[ix, iy] += 1
            with np.errstate(invalid='ignore', divide='ignore'):
                grid_avg = np.divide(grid, gc, where=gc != 0)
            if interpolate:
                grid_avg = self._interpolate_grid(grid_avg, gc, x_min, y_min, grid_size)
            mask_empty = gc == 0
            if not interpolate:
                grid_avg[mask_empty] = np.nan
            valid_vals = grid_avg[np.isfinite(grid_avg) & (grid_avg > 1e-6)]
            if valid_vals.size == 0:
                vmin, vmax = 1e-6, 1e-6
            else:
                vmin, vmax = float(valid_vals.min()), float(valid_vals.max())
            results.append(GridResult(grid=grid_avg, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, vmin=vmin, vmax=vmax))
        return results

    def plot_grid_avghz_panel(self, grid_size: float = 50.0, ncols: int = 3, interpolate: bool = False):
        grids = self.compute_grid_avghz_per_recording(grid_size=grid_size, interpolate=interpolate)
        if not grids:
            print("No data to plot.")
            return None, None
        n = len(grids)
        ncols = max(1, int(ncols))
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        # Compute global vmin/vmax across all grids for consistent LogNorm
        all_vals = []
        for res in grids:
            vals = res.grid[np.isfinite(res.grid) & (res.grid > 1e-6)]
            if vals.size:
                all_vals.append(vals)
        if all_vals:
            concat = np.concatenate(all_vals)
            global_vmin = float(max(1e-6, np.min(concat)))
            global_vmax = float(max(global_vmin, np.max(concat)))
        else:
            global_vmin = global_vmax = 1e-6
        # Ensure proper bounds for LogNorm
        if global_vmax <= global_vmin:
            global_vmax = global_vmin * 1.01
        norm = LogNorm(vmin=global_vmin, vmax=global_vmax)
        idx = 0
        last_im = None
        for r in range(nrows):
            for c in range(ncols):
                ax = axes[r, c]
                if idx < n:
                    res = grids[idx]
                    cmap = plt.get_cmap('magma').copy()
                    cmap.set_bad('black')
                    last_im = ax.imshow(res.grid.T, origin='lower', cmap=cmap, norm=norm,
                                        extent=[res.x_min, res.x_max, res.y_min, res.y_max])
                    ax.set_title(f'Recording {idx+1}')
                    # Match averaged plot background
                    ax.set_facecolor('black')
                else:
                    ax.axis('off')
                ax.set_aspect('equal')
                ax.set_xlabel('X ($\\mu m$)')
                ax.set_ylabel('Y ($\\mu m$)')
                idx += 1
        fig.tight_layout()
        # Single shared colorbar for all panels
        if last_im is not None:
            fig.colorbar(last_im, ax=axes.ravel().tolist(), orientation='vertical', label='Average Firing Rate (Hz)')
        plt.show()
        return fig, axes
