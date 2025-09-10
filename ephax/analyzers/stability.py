from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import scipy
from scipy.stats import gaussian_kde, binned_statistic

from ..prep import RestingActivityDataset
from ..spikes import calculate_ifr


@dataclass
class StabilityConfig:
    bin_width: float = 0.01  # seconds for activity analysis grid and smoothing
    sigma: float = 0.05      # seconds for Gaussian smoothing in smooth_ifr_trajectory


class StabilityAnalyzer:
    """
    Lyapunov-style stability analysis on smoothed IFR trajectories.

    - Uses dataset recording windows (start_time, end_time) per recording
    - Selects active electrodes per recording based on activity threshold (like legacy)
    - Builds electrode pairs with distances and smoothed IFR trajectories
    - Computes Lyapunov exponents per distance bin with optional stratified sampling
    - Provides plotting helpers to visualize distributions and CI with significance
    """

    def __init__(self, dataset: RestingActivityDataset, config: Optional[StabilityConfig] = None):
        self.ds = dataset
        self.cfg = config or StabilityConfig()

    # -------- Activity selection (legacy-compatible) --------
    def _analyze_firing_activity(self, spikes_data: dict, layout: dict, start_time: float, end_time: float, bin_width: float = 0.01) -> np.ndarray:
        # Get active electrodes
        spikes_df = pd.DataFrame(spikes_data)
        active_electrodes = np.unique(spikes_df['electrode'])

        # Calculate firing rates per bin
        bins = np.arange(start_time, end_time + bin_width, bin_width)
        activity_stats = []
        for electrode in active_electrodes:
            spike_times = spikes_df['time'][spikes_df['electrode'] == electrode]
            spike_times = spike_times[(spike_times >= start_time) & (spike_times <= end_time)]
            if len(spike_times) > 0:
                counts, _ = np.histogram(spike_times, bins=bins)
                rates = counts / bin_width
                active_bins = np.sum(rates > 0)
                mean_rate = np.mean(rates)
                max_rate = np.max(rates)
                activity_stats.append({
                    'electrode': electrode,
                    'n_spikes': len(spike_times),
                    'active_bins': active_bins,
                    'percent_active': (active_bins / len(rates)) * 100,
                    'mean_rate': mean_rate,
                    'max_rate': max_rate,
                })

        stats_df = pd.DataFrame(activity_stats)
        active_threshold = 0.1  # percent of bins with activity
        sufficiently_active = stats_df[stats_df['percent_active'] > active_threshold]
        return sufficiently_active['electrode'].to_numpy(dtype=int)

    # -------- IFR smoothing (legacy-compatible) --------
    @staticmethod
    def _smooth_ifr_trajectory(ifr_times: np.ndarray, ifr_values: np.ndarray, start_time: float, end_time: float, bin_width: float = 0.01, sigma: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        bins = np.arange(start_time, end_time + bin_width, bin_width)
        min_rate = 1e-5
        log_values = np.log10(ifr_values + min_rate)
        interpolated = np.interp(bins, ifr_times, log_values)
        kernel_width = int(4 * sigma / bin_width)
        kernel_times = np.arange(-kernel_width, kernel_width + 1) * bin_width
        kernel = np.exp(-(kernel_times ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        smoothed = np.convolve(interpolated, kernel, mode='same')
        return bins, smoothed

    # -------- Prepare pairs --------
    def prepare(self, plot_heatmap: bool = False) -> List[Dict]:
        all_pairs: List[Dict] = []
        for idx, rec in enumerate(self.ds.recordings):
            spikes_data = rec.spikes
            layout = rec.layout
            start_time = rec.start_time
            end_time = rec.end_time

            # Activity selection per recording
            active_electrodes = self._analyze_firing_activity(spikes_data, layout, start_time, end_time, bin_width=self.cfg.bin_width)

            # IFR per active electrode
            ifr_data, _, _ = calculate_ifr(spikes_data, active_electrodes, start_time, end_time)

            # Smooth IFR trajectories
            smoothed: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            heatmap_data = []
            time_bins = None
            for el, (ifr_t, ifr_v) in ifr_data.items():
                bins, smooth = self._smooth_ifr_trajectory(ifr_t, ifr_v, start_time, end_time, bin_width=self.cfg.bin_width, sigma=self.cfg.sigma)
                smoothed[el] = (bins, smooth)
                heatmap_data.append(smooth)
                if time_bins is None:
                    time_bins = bins

            if plot_heatmap and heatmap_data:
                plt.figure(figsize=(12, 8))
                plt.imshow(
                    np.array(heatmap_data),
                    aspect='auto',
                    extent=[time_bins[0], time_bins[-1], 0, len(heatmap_data)],
                    cmap='hot',
                    interpolation='nearest',
                    origin='lower',
                )
                plt.colorbar(label="IFR (log Hz)")
                plt.ylabel("Electrodes")
                plt.xlabel("Time (s)")
                plt.title(f"Heatmap of IFR Rates (Recording {idx+1})")
                plt.show()

            # Build electrode coordinate map
            layout_df = pd.DataFrame(layout)
            coords = {int(row['electrode']): (float(row['x']), float(row['y'])) for _, row in layout_df.iterrows()}

            # Build pairs with distances and smoothed IFR
            for i, ref_el in enumerate(active_electrodes):
                if ref_el not in smoothed:
                    continue
                rx, ry = coords.get(int(ref_el), (None, None))
                if rx is None:
                    continue
                for tgt_el in active_electrodes[i + 1:]:
                    if tgt_el not in smoothed:
                        continue
                    tx, ty = coords.get(int(tgt_el), (None, None))
                    if tx is None:
                        continue
                    distance = float(np.sqrt((rx - tx) ** 2 + (ry - ty) ** 2))
                    all_pairs.append({
                        'recording_idx': idx,
                        'ref_electrode': int(ref_el),
                        'target_electrode': int(tgt_el),
                        'distance': distance,
                        'ref_ifr': smoothed[int(ref_el)],
                        'target_ifr': smoothed[int(tgt_el)],
                    })
        return all_pairs

    # -------- Lyapunov helpers --------
    @staticmethod
    def _create_delay_vectors(time_series: np.ndarray, tau: int, m: int) -> np.ndarray:
        N = len(time_series) - (m - 1) * tau
        if N <= 0:
            return np.array([])
        vectors = np.zeros((N, m))
        for i in range(m):
            vectors[:, i] = time_series[i * tau: i * tau + N]
        return vectors

    @staticmethod
    def _compute_lyapunov(embed1: np.ndarray, embed2: np.ndarray, max_points: int = 1000, min_neighbors: int = 5, max_neighbors: int = 10, max_dt: int = 10) -> float:
        min_length = min(len(embed1), len(embed2))
        if min_length <= 0:
            return np.nan
        # Random sampling
        if min_length > max_points:
            indices = np.random.choice(min_length - 5, max_points, replace=False)
            indices = np.sort(indices)
            embed1 = embed1[indices]
            embed2 = embed2[indices]
            min_length = max_points

        # Normalize safely
        joint = np.hstack([embed1, embed2])
        means = np.mean(joint, axis=0)
        stds = np.std(joint, axis=0)
        stds[stds == 0] = 1
        joint = (joint - means) / stds

        # Distances
        distances = scipy.spatial.distance.pdist(joint)
        distances = scipy.spatial.distance.squareform(distances)
        if np.all(distances == 0):
            return np.nan

        epsilon = 0.2
        divergence_rates = []
        for i in range(min_length - max_dt):
            neighbors = np.where((distances[i] > 0) & (distances[i] < epsilon))[0]
            neighbors = neighbors[neighbors < min_length - max_dt]
            if len(neighbors) >= min_neighbors:
                if len(neighbors) > max_neighbors:
                    neighbors = neighbors[np.argsort(distances[i, neighbors])[:max_neighbors]]
                initial = distances[i, neighbors]
                future = np.array([distances[i + dt, neighbors] for dt in range(1, max_dt)])
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratios = future / initial[None, :]
                    valid_mask = (ratios > 0)
                    if np.any(valid_mask):
                        logs = np.log(ratios + 1e-10)
                        dt_array = np.arange(1, max_dt)[:, None]
                        divs = np.mean(logs * valid_mask, axis=1) / dt_array
                        valid_divs = divs[~np.isnan(divs)]
                        if len(valid_divs) > 0:
                            divergence_rates.extend(valid_divs)
        if divergence_rates:
            return float(np.mean(divergence_rates))
        return np.nan

    # -------- Binning and analysis --------
    def bin_and_analyze(
        self,
        all_pairs: List[Dict],
        bin_size: int = 200,
        max_distance: int = 3200,
        tau: int = 10,
        m: int = 5,
        max_pairs_per_bin: int = 20000,
        random_state: Optional[int] = 42,
        stratified: bool = True,
    ) -> Tuple[Dict[int, Dict], np.ndarray]:
        print("\nStarting stability analysis...")
        distances = [pair['distance'] for pair in all_pairs]
        bins = np.arange(0, max_distance, bin_size)
        print(f"Created {len(bins)-1} distance bins")

        print("Binning pairs...")
        binned_pairs: Dict[int, List[Dict]] = {i: [] for i in range(len(bins)-1)}
        for pair in all_pairs:
            bin_idx = int(np.digitize(pair['distance'], bins) - 1)
            if bin_idx in binned_pairs:
                binned_pairs[bin_idx].append(pair)

        print("Computing stability metrics...")
        stability_metrics: Dict[int, Dict] = {}
        total_bins = len(binned_pairs)
        rng = np.random.default_rng(random_state)

        for bin_idx, pairs in binned_pairs.items():
            if len(pairs) == 0:
                continue
            print(f"\nProcessing bin {bin_idx+1}/{total_bins} ({len(pairs)} pairs)")
            # Sampling
            if len(pairs) > max_pairs_per_bin:
                if stratified:
                    by_rec: Dict[int, List[Dict]] = {}
                    for p in pairs:
                        by_rec.setdefault(int(p.get('recording_idx', -1)), []).append(p)
                    n_rec = len(by_rec)
                    quota = max(1, max_pairs_per_bin // max(1, n_rec))
                    sampled: List[Dict] = []
                    for _, plist in by_rec.items():
                        k = min(quota, len(plist))
                        idxs = rng.choice(len(plist), size=k, replace=False)
                        sampled.extend([plist[i] for i in idxs])
                    if len(sampled) < max_pairs_per_bin:
                        remaining = [p for p in pairs if p not in sampled]
                        k = min(max_pairs_per_bin - len(sampled), len(remaining))
                        if k > 0:
                            idxs = rng.choice(len(remaining), size=k, replace=False)
                            sampled.extend([remaining[i] for i in idxs])
                    pairs_to_process = sampled[:max_pairs_per_bin]
                else:
                    idxs = rng.choice(len(pairs), size=max_pairs_per_bin, replace=False)
                    pairs_to_process = [pairs[i] for i in idxs]
            else:
                pairs_to_process = list(pairs)

            # Lyapunov per pair
            bin_lyap: List[float] = []
            for pair in pairs_to_process:
                ref_times, ref_values = pair['ref_ifr']
                tgt_times, tgt_values = pair['target_ifr']
                if len(ref_values) >= m * tau + 30 and len(tgt_values) >= m * tau + 30:
                    ref_emb = self._create_delay_vectors(ref_values, tau, m)
                    tgt_emb = self._create_delay_vectors(tgt_values, tau, m)
                    if ref_emb.size > 0 and tgt_emb.size > 0:
                        lyap = self._compute_lyapunov(ref_emb, tgt_emb, max_points=100)
                        if not np.isnan(lyap):
                            bin_lyap.append(lyap)

            if bin_lyap:
                stability_metrics[bin_idx] = {
                    'distance': int((bins[bin_idx] + bins[bin_idx+1]) / 2),
                    'mean_lyap': float(np.mean(bin_lyap)),
                    'std_lyap': float(np.std(bin_lyap)),
                    'n_pairs': int(len(bin_lyap)),
                    'raw_lyap': list(bin_lyap),
                }
                ci = 1.96 * np.std(bin_lyap) / np.sqrt(len(bin_lyap))
                print(f"Bin {bin_idx+1}: Mean Lyapunov = {np.mean(bin_lyap):.4f} ± {ci:.4f} (n={len(bin_lyap)})")
            else:
                print(f"No valid Lyapunov exponents for bin {bin_idx+1}")

        print("Analysis complete!")
        return stability_metrics, bins

    # -------- Plotting helpers --------
    @staticmethod
    def plot_exponent_difference_heatmap(stability_metrics: Dict[int, Dict], bins: np.ndarray, max_bins: int = 35, exponent_cap: float = 2.0, kde_points: int = 200) -> None:
        # Prepare data
        sorted_indices = sorted(stability_metrics.keys())
        selected_indices = sorted_indices[:max_bins]
        distances = [stability_metrics[i]['distance'] for i in selected_indices]
        raw_exponents = [stability_metrics[i]['raw_lyap'] for i in selected_indices]
        all_exps = np.concatenate([np.asarray(r) for r in raw_exponents if len(r) > 0])
        exp_min = float(np.min(all_exps))
        exp_max = float(min(np.max(all_exps), exponent_cap))
        exp_grid = np.linspace(exp_min, exp_max, kde_points)
        kde_distributions = []
        for exps in raw_exponents:
            if len(exps) > 1:
                kde = gaussian_kde(exps)
                kde_pdf = kde(exp_grid)
            else:
                kde_pdf = np.zeros_like(exp_grid)
            kde_distributions.append(kde_pdf)
        ref = kde_distributions[0] if kde_distributions else np.zeros_like(exp_grid)
        diffs = np.array([d - ref for d in kde_distributions])

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            pd.DataFrame(diffs.T, index=exp_grid, columns=distances),
            cmap='RdBu_r', center=0, xticklabels=np.round(distances, 2), yticklabels=False, cbar_kws={'label': 'Difference in Probability Density'}, ax=ax
        )
        ax.set_xlabel('Distance (μm)')
        ax.set_ylabel('Lyapunov Exponent')
        ax.set_title('KDE-based Distribution Differences from First Bin')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
        p = np.array(pvals, dtype=float)
        mask = np.isfinite(p)
        m = np.sum(mask)
        if m == 0:
            return p
        order = np.argsort(p[mask])
        ranked = p[mask][order]
        adj = np.empty_like(ranked)
        prev = 1.0
        for i in range(m - 1, -1, -1):
            adj_i = ranked[i] * m / (i + 1)
            prev = min(prev, adj_i)
            adj[i] = prev
        out = np.full_like(p, np.nan)
        out_idx = np.where(mask)[0][order]
        out[out_idx] = adj
        return out

    @staticmethod
    def plot_stability_distributions(stability_metrics: Dict[int, Dict], bins: np.ndarray) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        distances = [metrics['distance'] for metrics in stability_metrics.values()]
        raw = [metrics['raw_lyap'] for metrics in stability_metrics.values()]
        data = []
        for distance, raw_values in zip(distances, raw):
            for value in raw_values:
                data.append({"Distance": distance, "Lyapunov": value})
        raw_df = pd.DataFrame(data)
        max_lyap = 2
        sns.violinplot(data=raw_df, x="Distance", y="Lyapunov", ax=ax1)
        ax1.set_title('Raw Lyapunov Exponent Distributions')
        ax1.set_xlabel('Distance (μm)')
        ax1.set_ylabel('Lyapunov Exponent')
        ax1.set_ylim(top=max_lyap)

        lyap_grid = np.linspace(min(raw_df['Lyapunov']), max(raw_df['Lyapunov']), 100)
        avg_kde = scipy.stats.gaussian_kde(raw_df['Lyapunov'])
        avg_density = avg_kde(lyap_grid)
        mean = np.mean(raw_df['Lyapunov'], axis=0)
        ci = 1.96 * np.std(raw_df['Lyapunov'], axis=0) / np.sqrt(len(raw_df['Lyapunov']))
        print(f'Average LE: {mean:.4} ± {ci:.4f}')

        diffs = []
        for values in raw:
            if len(values) > 1:
                kde = scipy.stats.gaussian_kde(values)
                density = kde(lyap_grid)
                diff = density - avg_density
            else:
                diff = np.zeros_like(lyap_grid)
            diffs.append(diff)
        diffs_array = np.array(diffs).T
        vmax = np.max(np.abs(diffs_array))
        im = ax2.imshow(
            diffs_array,
            aspect='auto',
            origin='lower',
            extent=[min(distances), max(distances), min(lyap_grid), max_lyap],
            cmap='RdBu_r', vmin=-vmax, vmax=vmax,
        )
        plt.colorbar(im, ax=ax2, label='Difference from Average KDE')
        ax2.set_title('Difference from Average Distribution (KDE)')
        ax2.set_xlabel('Distance (μm)')
        ax2.set_ylabel('Lyapunov Exponent')
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_ci_with_significance(
        cls,
        stability_metrics: Dict[int, Dict],
        bins: np.ndarray,
        alpha: float = 0.05,
        correction: str = "fdr_bh",
        test: str = "welch",
    ) -> None:
        bin_indices = sorted(stability_metrics.keys())
        distances = [stability_metrics[i]["distance"] for i in bin_indices]
        arrays = [np.asarray(stability_metrics[i]["raw_lyap"], dtype=float) for i in bin_indices]
        means = np.array([np.nan if len(a) == 0 else np.mean(a) for a in arrays], dtype=float)
        ses = np.array([np.nan if len(a) < 2 else np.std(a, ddof=1) / np.sqrt(len(a)) for a in arrays], dtype=float)
        cis = 1.96 * ses

        # Global pool excluding each bin for testing
        pvals = []
        for i, a in enumerate(arrays):
            if len(a) < 2:
                pvals.append(np.nan)
                continue
            others = [arrays[j] for j in range(len(arrays)) if j != i and len(arrays[j]) > 1]
            if not others:
                pvals.append(np.nan)
                continue
            pooled = np.concatenate(others)
            if test == "welch":
                from scipy.stats import ttest_ind
                _, p = ttest_ind(a, pooled, equal_var=False)
            elif test == "ks":
                from scipy.stats import ks_2samp
                _, p = ks_2samp(a, pooled, alternative="two-sided", mode="auto")
            else:
                raise ValueError("Unsupported test; use 'welch' or 'ks'.")
            pvals.append(float(p))
        pvals = np.array(pvals, dtype=float)

        # Multiple testing correction
        if correction == "fdr_bh":
            adj_p = cls._fdr_bh(pvals)
        elif correction == "bonferroni":
            m = np.sum(np.isfinite(pvals))
            adj_p = np.minimum(1.0, pvals * m)
        else:
            adj_p = pvals

        def star(p):
            if not np.isfinite(p):
                return ""
            return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < alpha else ""

        stars = [star(p) for p in adj_p]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.errorbar(distances, means, yerr=cis, fmt="o-", capsize=3, color="C0", label="Mean ± 95% CI")
        ax.set_xlabel("Distance (μm)")
        ax.set_ylabel("Lyapunov Exponent")
        ax.set_title(
            f"Lyapunov Mean ± 95% CI per Distance Bin (test={test}, correction={correction})"
        )
        if np.any(np.isfinite(means)):
            y_hi = means + cis
            y_min = np.nanmin(means - cis)
            y_max = np.nanmax(means + cis)
            y_off = 0.05 * (y_max - y_min if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0)
            for x, y, s in zip(distances, y_hi, stars):
                if s:
                    ax.text(x, y + y_off, s, ha="center", va="bottom", fontsize=10, color="k")
        ax.legend()
        plt.tight_layout()
        plt.show()

