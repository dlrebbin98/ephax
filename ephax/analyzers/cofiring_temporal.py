from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from io import BytesIO

from ..prep import RestingActivityDataset, PrepConfig
from ..analyzers.ifr import plot_cofiring_heatmap as _plot_heatmap
from ..compute import aggregate_cofiring_heatmap as _aggregate_heatmap
from ..helper_functions import assign_r_theta_distance, assign_r_distance
from ..compute import cofiring_proportions as _cofiring_proportions
from matplotlib.cm import get_cmap
from scipy.stats import binned_statistic_2d

# Local legacy-equivalent helpers (integrated from resting_activity)
def _norm_t0(heatmap_data: np.ndarray, delays: np.ndarray) -> np.ndarray:
    t0_index = np.where(delays == 0)[0]
    if len(t0_index) == 0:
        return heatmap_data
    t0_index = t0_index[0]
    base = heatmap_data[t0_index, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(heatmap_data, base, out=np.zeros_like(heatmap_data), where=base != 0)


def _proc_theta(spikes_data_list, layout_list, ref_electrode, start_times, end_times, window_size, delays, verbose: bool = False):
    """Legacy-equivalent: bin mean co-firing proportions per (distance, theta)."""
    heatmap_data_sum = np.zeros((len(delays), 30, 30))
    count_data = np.zeros((len(delays), 30, 30))
    distance_bins = None
    theta_bins = np.linspace(-np.pi, np.pi, num=31)
    for spikes_data, layout, st, et in zip(spikes_data_list, layout_list, start_times, end_times):
        spikes_df = pd.DataFrame(spikes_data)
        layout_df = pd.DataFrame(layout)
        # Skip this recording if ref not present in layout
        if "electrode" not in layout_df.columns or int(ref_electrode) not in set(layout_df["electrode"].tolist()):
            if verbose:
                print(f"[theta] Recording skipped: ref {int(ref_electrode)} not in layout")
            continue
        spikes_df, layout_df = assign_r_theta_distance(spikes_df, layout_df, ref_electrode)
        spikes_df_during = spikes_df[(spikes_df['time'] >= float(st)) & (spikes_df['time'] <= float(et))]
        firing_times = spikes_df_during['time'][spikes_df_during['electrode'] == ref_electrode]
        coords = layout_df.set_index('electrode')[['x', 'y']]
        for i, delay in enumerate(delays):
            delay_sec = float(delay) / 1000.0
            props = _cofiring_proportions(
                spikes_df_during,
                firing_times,
                window_size=float(window_size) / 10000.0,
                delay=delay_sec,
                ref_electrode=int(ref_electrode),
            )
            if not props:
                continue
            elecs = np.array(list(props.keys()), dtype=int)
            vals = np.array([props[e] for e in elecs], dtype=float)
            elecs = elecs[elecs != int(ref_electrode)]
            if elecs.size == 0:
                continue
            lmap = layout_df.set_index('electrode')
            dists = lmap.loc[elecs, 'distance'].to_numpy()
            thetas = lmap.loc[elecs, 'theta'].to_numpy()
            db = np.linspace(float(np.nanmin(dists)), float(np.nanmax(dists)), num=31)
            distance_bins = db
            bin_means, _, _, _ = binned_statistic_2d(dists, thetas, vals[: len(dists)], statistic='mean', bins=[db, theta_bins])
            valid = ~np.isnan(bin_means)
            heatmap_data_sum[i, valid] += bin_means[valid]
            count_data[i, valid] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_avg = np.true_divide(heatmap_data_sum, count_data)
        heatmap_avg[~np.isfinite(heatmap_avg)] = 0
    return heatmap_avg, distance_bins, theta_bins


def _proc_grid(spikes_data_list, layout_list, ref_electrode, start_times, end_times, window_size, delays, verbose: bool = False):
    """Legacy-equivalent: bin mean co-firing proportions per spatial grid cell."""
    grid_size = None
    heatmap_data_sum = None
    count_data = None
    x_bins = y_bins = None
    for spikes_data, layout, st, et in zip(spikes_data_list, layout_list, start_times, end_times):
        spikes_df = pd.DataFrame(spikes_data)
        layout_df = pd.DataFrame(layout)
        # Skip this recording if ref not present in layout
        if "electrode" not in layout_df.columns or int(ref_electrode) not in set(layout_df["electrode"].tolist()):
            if verbose:
                print(f"[grid] Recording skipped: ref {int(ref_electrode)} not in layout")
            continue
        spikes_df, layout_df = assign_r_distance(spikes_df, layout_df, ref_electrode)
        spikes_df_during = spikes_df[(spikes_df['time'] >= float(st)) & (spikes_df['time'] <= float(et))]
        firing_times = spikes_df_during['time'][spikes_df_during['electrode'] == ref_electrode]
        # grid bins
        x_min, x_max = layout_df['x'].min(), layout_df['x'].max()
        y_min, y_max = layout_df['y'].min(), layout_df['y'].max()
        x_bins = np.arange(x_min, x_max + 100, 100)
        y_bins = np.arange(y_min, y_max + 100, 100)
        if grid_size is None:
            grid_size = (len(y_bins) - 1, len(x_bins) - 1)
            heatmap_data_sum = np.zeros((len(delays), *grid_size))
            count_data = np.zeros((len(delays), *grid_size))
        # Precompute coordinate lookup once per recording
        coords = layout_df.set_index('electrode')[['x', 'y']]
        for i, delay in enumerate(delays):
            delay_sec = float(delay) / 1000.0
            props = _cofiring_proportions(
                spikes_df_during,
                firing_times,
                window_size=float(window_size) / 10000.0,
                delay=delay_sec,
                ref_electrode=int(ref_electrode),
            )
            if not props:
                continue
            # Vectorize: build arrays of electrode coords and values
            elecs = np.array([e for e in props.keys() if e != int(ref_electrode) and e in coords.index], dtype=int)
            if elecs.size == 0:
                continue
            vals = np.array([props[int(e)] for e in elecs], dtype=float)
            xs = coords.loc[elecs, 'x'].to_numpy(dtype=float)
            ys = coords.loc[elecs, 'y'].to_numpy(dtype=float)
            # Bin mean per cell using binned_statistic_2d; transpose to (y, x)
            bm, _, _, _ = binned_statistic_2d(xs, ys, vals, statistic='mean', bins=[x_bins, y_bins])
            bm = bm.T
            valid = ~np.isnan(bm)
            heatmap_data_sum[i, valid] += bm[valid]
            count_data[i, valid] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_avg = np.true_divide(heatmap_data_sum, count_data)
        heatmap_avg[~np.isfinite(heatmap_avg)] = 0
    return heatmap_avg, x_bins, y_bins


@dataclass
class CofiringTemporalConfig:
    """Configuration for temporal co-firing analyses and GIFs.

    - start_ms/stop_ms/step_ms: delay axis in milliseconds.
    - normalize: whether to t0-normalize frames when plotting.
    - verbose: print per-ref/per-recording skips and stats.
    """
    start_ms: float = -20.0
    stop_ms: float = 20.0
    step_ms: float = 10.0
    normalize: bool = False
    verbose: bool = False


class CofiringTemporalAnalyzer:
    """Temporal co-firing analyses: averaged heatmap, theta GIF, grid GIF.

    Uses per-recording windows from the dataset and stored refs (from PrepConfig
    or provided explicitly). Provides methods to compute and visualize
    co-firing as a function of distance and delay.
    """

    def __init__(
        self,
        dataset: RestingActivityDataset,
        config: CofiringTemporalConfig,
        refs_per_recording: Optional[List[np.ndarray]] = None,
        selection_prep_config: Optional[PrepConfig] = None,
    ) -> None:
        self.ds = dataset
        self.cfg = config
        self._stored_refs = refs_per_recording or getattr(dataset, 'selected_refs', None)
        if self._stored_refs is None and selection_prep_config is not None:
            self._stored_refs = dataset.select_ref_electrodes(selection_prep_config)
        if self._stored_refs is None:
            # fallback: use all electrodes present
            self._stored_refs = [np.unique(pd.DataFrame(rec.spikes)["electrode"].to_numpy()) for rec in self.ds.recordings]

        # Precompute delays (ms) array
        start_ms, stop_ms, step_ms = self.cfg.start_ms, self.cfg.stop_ms, self.cfg.step_ms
        self.delays = np.linspace(start_ms, stop_ms, int((stop_ms - start_ms) / step_ms) + 1)

    # --------- Averaged co-firing heatmap ---------
    def plot_avg_cofiring_heatmap(self):
        """Aggregate per-ref co-firing into a distance×delay heatmap and plot it."""
        spikes_list, layout_list, start_times, end_times = self.ds.to_legacy()
        # Window size expected by aggregate_cofiring_heatmap is in samples for 10kHz; emulate legacy by using ds.sf
        sf = float(self.ds.sf or 10000.0)
        window_size_samples = int(self.cfg.step_ms * sf / 1000.0)
        # aggregate_cofiring_heatmap expects a flat list of reference electrodes across recordings
        flat_refs: List[int] = []
        for refs in self._stored_refs:
            if refs is None:
                continue
            try:
                iter(refs)
            except TypeError:
                flat_refs.append(int(refs))
                continue
            for r in refs:
                try:
                    flat_refs.append(int(r))
                except Exception:
                    pass
        # Deduplicate while preserving order
        seen = set()
        flat_refs = [x for x in flat_refs if not (x in seen or seen.add(x))]
        heatmap = _aggregate_heatmap(
            spikes_list,
            layout_list,
            flat_refs,
            start_times,
            end_times,
            window_size=window_size_samples,
            delays=self.delays,
        )
        return _plot_heatmap(heatmap, normalize=self.cfg.normalize, show=True)

    # --------- Theta GIF ---------
    def _compute_theta_cube(self, rec_idx: int, refs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rec = self.ds.recordings[rec_idx]
        s, e = rec.start_time, rec.end_time
        sf = float(rec.sf)
        sdf = pd.DataFrame(rec.spikes)
        ldf = pd.DataFrame(rec.layout)
        mask_t = (sdf['time'] >= s) & (sdf['time'] <= e)
        sdf = sdf[mask_t].copy()

        # bin definitions
        # distances based on layout extent
        dmin = 0.0
        dmax = float(max(ldf['x'].max() - ldf['x'].min(), ldf['y'].max() - ldf['y'].min()))
        distance_bins = np.linspace(dmin, dmax, num=31)
        theta_bins = np.linspace(-np.pi, np.pi, num=31)

        cube = np.zeros((len(self.delays), len(distance_bins) - 1, len(theta_bins) - 1), dtype=float)
        count = np.zeros_like(cube)

        pm_sec = float(self.cfg.step_ms) / 1000.0
        window_size_sec = 2.0 * pm_sec

        for ref in refs:
            sdf_theta, ldf_theta = assign_r_theta_distance(sdf.copy(), ldf.copy(), int(ref))
            firing_times = sdf_theta['time'][sdf_theta['electrode'] == int(ref)]
            # For each delay, compute proportions and bin 2D
            for i, delay in enumerate(self.delays):
                delay_sec = float(delay) / 1000.0
                # Vectorized window membership
                t = sdf_theta['time'].to_numpy()[:, None]
                starts = (firing_times.to_numpy() + delay_sec)[None, :]
                ends = (firing_times.to_numpy() + delay_sec + window_size_sec)[None, :]
                mask = (t >= starts) & (t <= ends)
                coinciding = sdf_theta[np.any(mask, axis=1)]
                # Exclude ref electrode
                coinciding = coinciding[coinciding['electrode'] != int(ref)]
                if coinciding.empty:
                    continue
                # Bin by (distance, theta)
                d = coinciding['distance'].to_numpy()
                th = coinciding['theta'].to_numpy()
                H, d_edges, t_edges = np.histogram2d(d, th, bins=[distance_bins, theta_bins])
                cube[i] += H
                count[i] += (H > 0)
        with np.errstate(invalid='ignore'):
            cube_avg = np.divide(cube, count, where=count != 0)
        return cube_avg, distance_bins, theta_bins

    def create_theta_gif(self, output_filename: str = 'cofiring_theta.gif'):
        """Strict legacy logic: build theta heatmaps per ref using process_electrode_theta, then average."""
        spikes_list, layout_list, start_times, end_times = self.ds.to_legacy()
        # Window size (samples) as in legacy main: step_ms scaled by sf
        sf = float(self.ds.sf or 10000.0)
        window_size_samples = int(self.cfg.step_ms * sf / 1000.0)
        # Flatten refs across recordings
        flat_refs: List[int] = []
        for refs in self._stored_refs:
            if refs is None:
                continue
            flat_refs.extend([int(r) for r in np.atleast_1d(refs)])
        # Deduplicate preserve order
        seen = set(); flat_refs = [r for r in flat_refs if not (r in seen or seen.add(r))]
        if not flat_refs:
            print('No data for theta GIF')
            return
        # Use full timespan across recordings
        # Use per-recording windows directly
        sts = start_times
        ets = end_times
        if self.cfg.verbose:
            print(f"[theta] Using {len(flat_refs)} unique refs over {len(spikes_list)} recordings")
        # Use threads to avoid multiprocessing memory duplication of large inputs
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_proc_theta)(spikes_list, layout_list, ref, sts, ets, window_size_samples, self.delays, self.cfg.verbose)
            for ref in flat_refs
        )
        avg_heatmap_data = np.mean([res[0] for res in results], axis=0)
        distance_bins = results[0][1]
        theta_bins = results[0][2][:-1]
        if self.cfg.normalize:
            avg_heatmap_data = _norm_t0(avg_heatmap_data, self.delays)
        vmin = float(np.nanmin(avg_heatmap_data))
        vmax = float(np.nanmax(avg_heatmap_data))
        theta_edges = np.linspace(-np.pi, np.pi, avg_heatmap_data.shape[2] + 1)
        with imageio.get_writer(output_filename, mode='I', duration=0.2) as writer:
            for i, delay in enumerate(self.delays):
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
                theta_grid, r_grid = np.meshgrid(theta_edges, distance_bins)
                ax.grid(False)
                cax = ax.pcolormesh(theta_grid, r_grid, avg_heatmap_data[i], cmap=get_cmap('magma'), vmin=vmin, vmax=vmax)
                ax.set_theta_zero_location('N')
                ax.set_theta_direction(-1)
                ax.tick_params(colors='white')
                ax.set_title(f'Time Delay: {delay:.2f} ms')
                fig.colorbar(cax, ax=ax, label='Probability')
                buf = BytesIO(); fig.savefig(buf, format='png', dpi=150, bbox_inches='tight'); buf.seek(0)
                writer.append_data(imageio.v2.imread(buf)); buf.close(); plt.close(fig)

    # --------- Grid GIF ---------
    def _compute_grid_cube(self, rec_idx: int, refs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rec = self.ds.recordings[rec_idx]
        s, e = rec.start_time, rec.end_time
        sdf = pd.DataFrame(rec.spikes)
        ldf = pd.DataFrame(rec.layout)
        mask_t = (sdf['time'] >= s) & (sdf['time'] <= e)
        sdf = sdf[mask_t].copy()
        # Grid bins
        x_min, x_max = float(ldf['x'].min()), float(ldf['x'].max())
        y_min, y_max = float(ldf['y'].min()), float(ldf['y'].max())
        x_bins = np.arange(x_min, x_max + 100, 100)
        y_bins = np.arange(y_min, y_max + 100, 100)
        cube = np.zeros((len(self.delays), len(y_bins) - 1, len(x_bins) - 1), dtype=float)
        count = np.zeros_like(cube)
        pm_sec = float(self.cfg.step_ms) / 1000.0
        window_size_sec = 2.0 * pm_sec
        for ref in refs:
            sdf_r, ldf_r = assign_r_distance(sdf.copy(), ldf.copy(), int(ref))
            firing_times = sdf_r['time'][sdf_r['electrode'] == int(ref)]
            for i, delay in enumerate(self.delays):
                delay_sec = float(delay) / 1000.0
                t = sdf_r['time'].to_numpy()[:, None]
                starts = (firing_times.to_numpy() + delay_sec)[None, :]
                ends = (firing_times.to_numpy() + delay_sec + window_size_sec)[None, :]
                mask = (t >= starts) & (t <= ends)
                coinc = sdf_r[np.any(mask, axis=1)]
                coinc = coinc[coinc['electrode'] != int(ref)]
                if coinc.empty:
                    continue
                # Bin into spatial grid
                # Ensure x,y exist by merging with layout on electrode
                coords = ldf[['electrode', 'x', 'y']]
                coinc_xy = coinc.merge(coords, on='electrode', how='left')
                xs = coinc_xy['x'].to_numpy()
                ys = coinc_xy['y'].to_numpy()
                xi = np.clip(np.digitize(xs, x_bins) - 1, 0, len(x_bins) - 2)
                yi = np.clip(np.digitize(ys, y_bins) - 1, 0, len(y_bins) - 2)
                for xx, yy in zip(xi, yi):
                    cube[i, yy, xx] += 1
                    count[i, yy, xx] += 1
        with np.errstate(invalid='ignore'):
            cube_avg = np.divide(cube, count, where=count != 0)
        return cube_avg, x_bins, y_bins

    def create_grid_gif(self, output_filename: str = 'cofiring_grid.gif'):
        """Strict legacy logic using process_electrode_grid averaged across flattened refs."""
        spikes_list, layout_list, start_times, end_times = self.ds.to_legacy()
        sf = float(self.ds.sf or 10000.0)
        window_size_samples = int(self.cfg.step_ms * sf / 1000.0)
        flat_refs: List[int] = []
        for refs in self._stored_refs:
            if refs is None:
                continue
            flat_refs.extend([int(r) for r in np.atleast_1d(refs)])
        seen = set(); flat_refs = [r for r in flat_refs if not (r in seen or seen.add(r))]
        if not flat_refs:
            print('No data for grid GIF')
            return
        sts = start_times; ets = end_times
        if self.cfg.verbose:
            print(f"[grid] Using {len(flat_refs)} unique refs over {len(spikes_list)} recordings")
        # Use threads to avoid duplicating spikes/layout across processes
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_proc_grid)(spikes_list, layout_list, ref, sts, ets, window_size_samples, self.delays, self.cfg.verbose)
            for ref in flat_refs
        )
        avg_cube = np.mean([r[0] for r in results], axis=0)
        x_bins = results[0][1]; y_bins = results[0][2]
        vmin, vmax = float(np.nanmin(avg_cube)), float(np.nanmax(avg_cube))
        with imageio.get_writer(output_filename, mode='I', duration=0.2) as writer:
            for i, delay in enumerate(self.delays):
                fig, ax = plt.subplots(figsize=(8, 8))
                cax = ax.imshow(avg_cube[i], cmap=get_cmap('magma'), extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]], vmin=vmin, vmax=vmax, origin='lower')
                ax.set_xlabel('X Coordinate ($\\mu m$)')
                ax.set_ylabel('Y Coordinate ($\\mu m$)')
                ax.set_title(f'Time Delay: {delay:.2f} ms')
                fig.colorbar(cax, ax=ax, label='Probability')
                ax.set_aspect('equal')
                buf = BytesIO(); fig.savefig(buf, format='png', dpi=150, bbox_inches='tight'); buf.seek(0)
                writer.append_data(imageio.v2.imread(buf)); buf.close(); plt.close(fig)
