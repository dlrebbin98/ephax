from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
from scipy.fftpack import dct, idct

from ..prep import RestingActivityDataset


@dataclass
class InterpolatedGrid:
    grid: np.ndarray
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class DCTAnalyzer:
    """
    Discrete Cosine Transform (DCT) analysis on interpolated firing-rate grids.

    - Builds per-recording interpolated grids of average firing rate
    - Computes DCT per grid; can average DCTs across recordings
    - Visualizes grids, DCT magnitudes, and reconstructions
    """

    def __init__(self, dataset: RestingActivityDataset) -> None:
        self.ds = dataset

    # ---------- Grid computation ----------
    @staticmethod
    def _grid_extents(layout_df: pd.DataFrame, grid_size: float) -> Tuple[int, int, float, float, float, float, np.ndarray, np.ndarray]:
        x_n = int(np.ceil(float(layout_df['x'].max()) / grid_size)) - 1
        y_n = int(np.ceil(float(layout_df['y'].max()) / grid_size)) - 1
        x_min, x_max = 1.0, x_n * grid_size
        y_min, y_max = 1.0, y_n * grid_size
        x_bins = np.linspace(x_min, x_max, x_n + 1)
        y_bins = np.linspace(y_min, y_max, y_n + 1)
        return x_n, y_n, x_min, x_max, y_min, y_max, x_bins, y_bins

    def compute_interpolated_grid_for_recording(self, rec_idx: int, grid_size: float = 50.0) -> InterpolatedGrid:
        rec = self.ds.recordings[rec_idx]
        s, e = rec.start_time, rec.end_time
        dur = float(e - s)
        sdf = pd.DataFrame(rec.spikes)
        ldf = pd.DataFrame(rec.layout)
        mask = (sdf['time'] >= s) & (sdf['time'] <= e)
        sdf = sdf[mask]
        # Average firing rate per electrode
        fr = sdf['electrode'].value_counts() / max(dur, 1e-9)
        # Grid extents
        x_n, y_n, x_min, x_max, y_min, y_max, x_bins, y_bins = self._grid_extents(ldf, grid_size)
        grid = np.zeros((x_n, y_n), dtype=float)
        gc = np.zeros((x_n, y_n), dtype=float)
        ldf_idx = ldf.set_index('electrode')
        for elec, rate in fr.items():
            elec = int(elec)
            if elec not in ldf_idx.index:
                continue
            x, y = float(ldf_idx.loc[elec, 'x']), float(ldf_idx.loc[elec, 'y'])
            ix = int(np.digitize(x, x_bins) - 1)
            iy = int(np.digitize(y, y_bins) - 1)
            if 0 <= ix < x_n and 0 <= iy < y_n:
                grid[ix, iy] += float(rate)
                gc[ix, iy] += 1.0
        with np.errstate(invalid='ignore', divide='ignore'):
            grid_avg = np.divide(grid, gc, where=gc != 0)
        # Interpolate missing cells in index space
        known_points = []
        known_values = []
        for ix in range(x_n):
            for iy in range(y_n):
                if gc[ix, iy] > 0:
                    known_points.append((ix, iy))
                    known_values.append(grid_avg[ix, iy])
        grid_x, grid_y = np.meshgrid(range(x_n), range(y_n), indexing='ij')
        if known_points:
            interpolated = griddata(np.array(known_points), np.array(known_values), (grid_x, grid_y), method='cubic', fill_value=0.0)
            grid_avg = np.where(gc > 0, grid_avg, interpolated)
        else:
            grid_avg[:] = 0.0
        return InterpolatedGrid(grid=grid_avg, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    def compute_interpolated_grids(self, grid_size: float = 50.0) -> List[InterpolatedGrid]:
        out: List[InterpolatedGrid] = []
        for i in range(len(self.ds.recordings)):
            out.append(self.compute_interpolated_grid_for_recording(i, grid_size=grid_size))
        return out

    # ---------- Visualization ----------
    @staticmethod
    def plot_grid(res: InterpolatedGrid):
        grid = np.nan_to_num(res.grid, nan=0.0, posinf=0.0, neginf=0.0)
        grid[grid == 0] = 1e-6
        valid = grid[grid > 1e-6]
        if valid.size == 0:
            vmin = vmax = 1e-6
        else:
            vmin = float(np.nanmin(valid)); vmax = float(np.nanmax(valid))
        vmin = max(vmin, 1e-6)
        vmax = max(vmax, vmin * 1.01)
        fig, ax = plt.subplots(figsize=(10, 6))
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('magma')
        cax = ax.imshow(grid.T, origin='lower', cmap=cmap, norm=norm, extent=[res.x_min, res.x_max, res.y_min, res.y_max])
        ax.set_xlabel(r'X-coordinate ($\mu m$)')
        ax.set_ylabel(r'Y-coordinate ($\mu m$)')
        ax.set_title('Average Firing Rate (Interpolated)')
        ax.set_facecolor('black')
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
        cbar.set_label('Average Firing Rate (Hz)')
        ax.grid(True)
        ax.set_aspect('equal')
        plt.show()
        return fig, ax

    # ---------- DCT ----------
    @staticmethod
    def dct2(grid: np.ndarray) -> np.ndarray:
        return dct(dct(grid.T, norm='ortho').T, norm='ortho')

    @staticmethod
    def plot_dct(grid: np.ndarray):
        coeff = DCTAnalyzer.dct2(grid)
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.imshow(np.abs(coeff), origin='lower', cmap='magma', aspect='auto')
        ax.set_xlabel('DCT X-coefficient')
        ax.set_ylabel('DCT Y-coefficient')
        ax.set_title('Magnitude of DCT Coefficients')
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
        cbar.set_label('Magnitude')
        plt.grid(True)
        ax.set_aspect('auto')
        plt.show()
        return fig, ax

    @staticmethod
    def average_dct(grids: List[np.ndarray]) -> np.ndarray:
        if not grids:
            raise ValueError("No grids supplied")
        mats = []
        for g in grids:
            M = DCTAnalyzer.dct2(g)
            M[0, 0] = 0.0
            M = M / max(np.max(np.abs(M)), 1e-9)
            mats.append(M)
        dct_sum = np.sum(mats, axis=0)
        dct_avg = dct_sum / len(mats)
        # normalize
        denom = max(np.max(np.abs(dct_avg)), 1e-9)
        return dct_avg / denom

    @staticmethod
    def reconstruct_from_top_components(dct_transform: np.ndarray, stop_rank: int, start_rank: int = 0, plot_distribution: bool = False) -> np.ndarray:
        if start_rank > stop_rank - 1:
            raise ValueError("start_rank must be < stop_rank")
        flat_indices = np.dstack(np.unravel_index(np.argsort(-np.abs(dct_transform).ravel()), dct_transform.shape))[0]
        if plot_distribution:
            sorted_magnitudes = np.sort(np.abs(dct_transform).ravel())[::-1]
            plt.figure(figsize=(8, 5))
            plt.plot(sorted_magnitudes, marker='o', linestyle='-', color='b', alpha=0.7)
            plt.xlabel('Sorted Coefficient Rank')
            plt.ylabel('Coefficient Magnitude')
            plt.title('Distribution of DCT Coefficients\' Magnitudes')
            plt.yscale('log')
            plt.grid(True, which="both", ls="--")
            plt.show()
        top = np.zeros_like(dct_transform)
        for i in range(start_rank, min(stop_rank, len(flat_indices))):
            x_idx, y_idx = flat_indices[i]
            top[x_idx, y_idx] = dct_transform[x_idx, y_idx]
        reconstructed = idct(idct(top.T, norm='ortho').T, norm='ortho')
        return reconstructed

    @staticmethod
    def plot_reconstructed_grid(reconstructed: np.ndarray, res: InterpolatedGrid):
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.imshow(reconstructed.T, origin='lower', cmap='magma', extent=[res.x_min, res.x_max, res.y_min, res.y_max])
        ax.set_xlabel(r'X-coordinate ($\mu m$)')
        ax.set_ylabel(r'Y-coordinate ($\mu m$)')
        ax.set_title('Reconstructed Firing Rate Grid')
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
        cbar.set_label('Average Firing Rate (Hz)')
        plt.grid(True)
        ax.set_aspect('equal')
        plt.show()
        return fig, ax

    @staticmethod
    def extract_and_plot_spatial_frequencies_from_dct(dct_transform: np.ndarray, n_components_stop: int, n_components_start: int = 0, array_dims: Tuple[int, int] = (3800, 2100), axis: str = 'x', num_points: int = 1000):
        flat_indices = np.dstack(np.unravel_index(np.argsort(-np.abs(dct_transform).ravel()), dct_transform.shape))[0]
        num_x, num_y = dct_transform.shape
        length_x, length_y = array_dims
        signal = np.zeros(num_points)
        x_space = np.linspace(0, length_x if axis == 'x' else length_y, num_points)
        for i in range(n_components_start, n_components_stop):
            x_idx, y_idx = flat_indices[i]
            coeff_value = dct_transform[x_idx, y_idx]
            freq_x = x_idx / num_x * (1 / (2 * (length_x / num_x)))
            freq_y = y_idx / num_y * (1 / (2 * (length_y / num_y)))
            amplitude = np.abs(coeff_value)
            phase = 0.0  # DCT is real; treat as zero-phase for visualization
            if axis == 'x':
                wave = amplitude * np.cos(2 * np.pi * freq_x * x_space + phase)
            else:
                wave = amplitude * np.cos(2 * np.pi * freq_y * x_space + phase)
            signal += wave
        max_value = np.max(signal)
        max_positions = x_space[np.where(signal == max_value)[0]]
        print(f'Peaks at {max_positions}')
        plt.figure(figsize=(10, 6))
        plt.plot(x_space, signal, 'b-', label=f'Reconstructed Signal along {axis}-axis')
        plt.xlabel(f'{axis}-axis Position (μm)')
        plt.ylabel('Amplitude (a.u.)')
        plt.title('Reconstructed Spatial Signal from Top DCT Components')
        plt.grid(True)
        plt.legend()
        plt.show()

