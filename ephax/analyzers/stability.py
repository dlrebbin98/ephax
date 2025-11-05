from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import scipy
from scipy.stats import gaussian_kde, binned_statistic

from ..prep import RestingActivityDataset
from ..helper_functions import calculate_ifr


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

    def __init__(
        self,
        dataset: RestingActivityDataset,
        config: Optional[StabilityConfig] = None,
        dataset_perm: RestingActivityDataset | List[RestingActivityDataset] | None = None,
    ):
        self.ds = dataset
        if dataset_perm is None:
            self.ds_perm: List[RestingActivityDataset] = []
        elif isinstance(dataset_perm, list):
            self.ds_perm = dataset_perm
        else:
            self.ds_perm = [dataset_perm]
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
    def prepare(
        self,
        plot_heatmap: bool = False,
        dataset: RestingActivityDataset | None = None,
    ) -> List[Dict]:
        ds_use = dataset or self.ds
        all_pairs: List[Dict] = []
        for idx, rec in enumerate(ds_use.recordings):
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
        max_distance: int = 3500,
        tau: int = 10,
        m: int = 5,
        max_pairs_per_bin: int = 20000,
        random_state: Optional[int] = 42,
        stratified: bool = True,
        bin_edges: np.ndarray | None = None,
    ) -> Tuple[Dict[int, Dict], np.ndarray]:
        print("\nStarting stability analysis...")
        distances = [pair['distance'] for pair in all_pairs]
        if bin_edges is not None:
            bins = np.asarray(bin_edges, dtype=float)
        else:
            bins = np.arange(0, max_distance + bin_size, bin_size)
        if bins.ndim != 1 or bins.size < 2:
            raise ValueError("bin_edges must define at least two boundaries.")
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
    def _fit_correlation_curve(
        corr_x: np.ndarray,
        corr_y: np.ndarray,
        x_target: np.ndarray,
        y_target: np.ndarray,
        allow_offset: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        """Scale/shift a correlation template so it best matches ``y_target``."""
        if corr_x is None or corr_y is None:
            return None

        corr_x = np.asarray(corr_x, dtype=float)
        corr_y = np.asarray(corr_y, dtype=float)
        x_target = np.asarray(x_target, dtype=float)
        y_target = np.asarray(y_target, dtype=float)

        if corr_x.size == 0 or corr_y.size == 0 or x_target.size == 0:
            return None

        order = np.argsort(corr_x)
        corr_x_sorted = corr_x[order]
        corr_y_sorted = corr_y[order]

        interp = np.interp(
            x_target,
            corr_x_sorted,
            corr_y_sorted,
            left=np.nan,
            right=np.nan,
        )

        mask = np.isfinite(interp) & np.isfinite(y_target)
        if np.count_nonzero(mask) < (2 if allow_offset else 1):
            return None

        base = interp[mask]
        target = y_target[mask]

        try:
            if allow_offset:
                A = np.column_stack([base, np.ones_like(base)])
                coeffs, *_ = np.linalg.lstsq(A, target, rcond=None)
                amp = float(coeffs[0])
                offset = float(coeffs[1])
                if amp < 0:
                    amp = 0.0
                    offset = float(np.nanmean(target))
            else:
                denom = float(np.dot(base, base))
                if denom <= 0:
                    return None
                amp = float(np.dot(base, target) / denom)
                if amp < 0:
                    amp = 0.0
                offset = 0.0
        except np.linalg.LinAlgError:
            return None

        if not np.isfinite(amp) or not np.isfinite(offset):
            return None

        fitted = amp * corr_y_sorted + offset
        return corr_x_sorted, fitted, amp, offset

    @staticmethod
    def plot_stability_distributions(
        stability_metrics: Dict[int, Dict],
        bins: np.ndarray,
        stability_metrics_perm: Optional[List[Dict[int, Dict]]] = None,
    ) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        def _metrics_to_records(metrics: Dict[int, Dict], label: str) -> list[dict]:
            records: list[dict] = []
            for data in metrics.values():
                dist = data.get('distance')
                for value in data.get('raw_lyap', []):
                    records.append({"Distance": dist, "Lyapunov": value, "Condition": label})
            return records

        records = _metrics_to_records(stability_metrics, "Original")
        if stability_metrics_perm:
            for idx, metrics_perm in enumerate(stability_metrics_perm):
                label = "Permuted" if len(stability_metrics_perm) == 1 else f"Permuted #{idx+1}"
                records.extend(_metrics_to_records(metrics_perm, label))

        if not records:
            print("No Lyapunov exponents available to plot.")
            return

        raw_df = pd.DataFrame(records)
        dist_order = sorted(raw_df['Distance'].dropna().unique())
        raw_df['Distance'] = pd.Categorical(raw_df['Distance'], categories=dist_order, ordered=True)

        if raw_df.empty:
            max_lyap = 0.0
        else:
            max_series = raw_df['Lyapunov'].max(skipna=True)
            max_lyap = float(max_series) if np.isfinite(max_series) else 0.0
        sns.violinplot(
            data=raw_df,
            x="Distance",
            y="Lyapunov",
            hue="Condition" if stability_metrics_perm is not None else None,
            dodge=True,
            cut=0,
            ax=ax1,
        )
        if stability_metrics_perm is not None:
            ax1.legend(title="Condition", frameon=False)
        else:
            legend = ax1.get_legend()
            if legend is not None:
                legend.remove()
        ax1.set_title('Raw Lyapunov Exponent Distributions')
        ax1.set_xlabel('Distance (μm)')
        ax1.set_ylabel('Lyapunov Exponent')
        ax1.set_ylim(top=max_lyap if max_lyap > 0 else 1.0)

        # KDE difference heatmap (Original minus Permuted when available, else against overall mean)
        distance_grid = dist_order
        exp_values = raw_df['Lyapunov'].to_numpy()
        lyap_min = float(np.nanmin(exp_values)) if exp_values.size else 0.0
        lyap_max = float(np.nanmax(exp_values)) if exp_values.size else 1.0
        if lyap_min == lyap_max:
            lyap_max = lyap_min + 1.0
        lyap_grid = np.linspace(lyap_min, lyap_max, 200)

        def _kde(values: np.ndarray) -> np.ndarray:
            values = values[np.isfinite(values)]
            if values.size > 1:
                return gaussian_kde(values)(lyap_grid)
            return np.zeros_like(lyap_grid)

        orig_by_distance = {
            data['distance']: np.asarray(data.get('raw_lyap', []), dtype=float)
            for data in stability_metrics.values()
        }
        perm_by_distance: Dict[float, List[np.ndarray]] = {}
        if stability_metrics_perm:
            for metrics_perm in stability_metrics_perm:
                for data in metrics_perm.values():
                    dist = data.get('distance')
                    perm_by_distance.setdefault(dist, []).append(
                        np.asarray(data.get('raw_lyap', []), dtype=float)
                    )

        diff_columns = []
        for dist in distance_grid:
            orig_vals = orig_by_distance.get(dist, np.array([], dtype=float))
            kde_orig = _kde(orig_vals)
            if stability_metrics_perm:
                perm_vals_list = perm_by_distance.get(dist, [])
                if perm_vals_list:
                    perm_concat = np.concatenate([vals for vals in perm_vals_list if vals.size], axis=0) if any(vals.size for vals in perm_vals_list) else np.array([], dtype=float)
                else:
                    perm_concat = np.array([], dtype=float)
                kde_perm = _kde(perm_concat)
                diff = kde_orig - kde_perm
            else:
                avg_density = _kde(exp_values)
                diff = kde_orig - avg_density
            diff_columns.append(diff)

        diff_matrix = np.column_stack(diff_columns) if diff_columns else np.zeros((len(lyap_grid), 0))
        vmax = float(np.max(np.abs(diff_matrix))) if diff_matrix.size else 1.0
        sns.heatmap(
            diff_matrix,
            cmap='RdBu_r',
            center=0,
            vmin=-vmax,
            vmax=vmax,
            xticklabels=[f"{d:.0f}" for d in distance_grid],
            yticklabels=False,
            cbar_kws={'label': 'Density Difference'},
            ax=ax2,
        )
        ax2.set_xlabel('Distance (μm)')
        ax2.set_ylabel('Lyapunov Exponent')
        title_suffix = 'Original - Permuted' if stability_metrics_perm is not None else 'Difference from Global KDE'
        ax2.set_title(f'Lyapunov KDE Difference ({title_suffix})')

        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_ci_with_significance(
        cls,
        stability_metrics: Dict[int, Dict],
        bins: np.ndarray,
        stability_metrics_perm: Optional[List[Dict[int, Dict]]] = None,
        correlation_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        alpha: float = 0.05,
        correction: str = "fdr_bh",
        test: str = "welch",
    ) -> None:
        if not stability_metrics_perm:
            bin_indices = sorted(stability_metrics.keys())
            distances = [stability_metrics[i]["distance"] for i in bin_indices]
            arrays = [np.asarray(stability_metrics[i]["raw_lyap"], dtype=float) for i in bin_indices]
            means = np.array([np.nan if len(a) == 0 else np.mean(a) for a in arrays], dtype=float)
            ses = np.array([np.nan if len(a) < 2 else np.std(a, ddof=1) / np.sqrt(len(a)) for a in arrays], dtype=float)
            cis = 1.96 * ses

            pooled_all = np.concatenate([a for a in arrays if a.size > 0]) if arrays else np.array([])
            pooled_all = pooled_all[np.isfinite(pooled_all)]
            overall_mean = float(np.nan) if pooled_all.size == 0 else float(np.mean(pooled_all))

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

            if correction == "fdr_bh":
                adj_p = cls._fdr_bh(pvals)
            elif correction == "bonferroni":
                m = np.sum(np.isfinite(pvals))
                adj_p = np.minimum(1.0, pvals * m)
            else:
                adj_p = pvals

            def star_fn(p):
                if not np.isfinite(p):
                    return ""
                return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < alpha else ""

            stars = [star_fn(p) for p in adj_p]

            order = np.argsort(np.asarray(distances, dtype=float))
            x = np.asarray(distances, dtype=float)[order]
            y = np.asarray(means, dtype=float)[order]
            y_ci = np.asarray(cis, dtype=float)[order]

            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_ci)
            x, y, y_ci = x[valid], y[valid], y_ci[valid]

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.fill_between(x, y - y_ci, y + y_ci, color="C0", alpha=0.2, linewidth=0, label="95% CI")
            ax.plot(x, y, color="C0", lw=2, label="Mean")
            ax.scatter(x, y, color="C0", edgecolor="white", zorder=3)

            if np.isfinite(overall_mean):
                ax.axhline(overall_mean, color="C3", lw=1.5, ls="--", alpha=0.8,
                           label=f"Overall mean = {overall_mean:.3f}")

            if x.size:
                y_hi = y + y_ci
                y_min = float(np.nanmin(y - y_ci))
                y_max = float(np.nanmax(y + y_ci))
                span = y_max - y_min if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0
                y_off = 0.04 * span
                stars_arr = np.array(stars, dtype=object)[order][valid]
                for xi, yi, s in zip(x, y_hi, stars_arr):
                    if s:
                        ax.text(xi, yi + y_off, s, ha="center", va="bottom", fontsize=10, color="k")

            ax.set_xlabel("Distance (μm)")
            ax.set_ylabel("Lyapunov Exponent")
            ax.set_title(
                f"Lyapunov Mean ± 95% CI per Distance Bin (test={test}, correction={correction})"
            )
            ax.grid(True, which="both", alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if correlation_curve is not None:
                corr_x, corr_y = correlation_curve
                fit = cls._fit_correlation_curve(corr_x, corr_y, x, y, allow_offset=True)
                if fit is not None:
                    corr_x_fit, corr_y_fit, amp, offset = fit
                    label = f'Correlation fit (amp={amp:.2g}, b={offset:.2g})'
                else:
                    corr_x_fit, corr_y_fit = np.asarray(corr_x, dtype=float), np.asarray(corr_y, dtype=float)
                    label = 'Correlation model'
                ax_corr = ax.twinx()
                ax_corr.plot(corr_x_fit, corr_y_fit, color='crimson', linestyle='--', label=label)
                ax_corr.set_ylabel('Synergy model (a.u.)')
                if np.any(np.isfinite(corr_y_fit)):
                    ax_corr.set_ylim(float(np.nanmin(corr_y_fit)), float(np.nanmax(corr_y_fit)))
                lines_data, labels_data = ax.get_legend_handles_labels()
                lines_model, labels_model = ax_corr.get_legend_handles_labels()
                ax.legend(lines_data + lines_model, labels_data + labels_model, loc='upper right')
            else:
                ax.legend(frameon=False)
            plt.tight_layout()
            plt.show()
            return

        # Comparison against permuted/null dataset
        orig_metrics = {data['distance']: np.asarray(data.get('raw_lyap', []), dtype=float) for data in stability_metrics.values()}
        perm_metrics: Dict[float, List[np.ndarray]] = {}
        for metrics_perm in stability_metrics_perm:
            for data in metrics_perm.values():
                dist = data.get('distance')
                perm_metrics.setdefault(dist, []).append(np.asarray(data.get('raw_lyap', []), dtype=float))
        distances = sorted(set(orig_metrics.keys()) | set(perm_metrics.keys()))

        orig_arrays = [orig_metrics.get(dist, np.array([], dtype=float)) for dist in distances]
        perm_arrays = []
        for dist in distances:
            vals_list = perm_metrics.get(dist, [])
            if vals_list:
                concatenated = np.concatenate([arr for arr in vals_list if arr.size], axis=0) if any(arr.size for arr in vals_list) else np.array([], dtype=float)
            else:
                concatenated = np.array([], dtype=float)
            perm_arrays.append(concatenated)

        def _mean_ci(arr: np.ndarray) -> tuple[float, float]:
            if arr.size == 0:
                return np.nan, np.nan
            mean = float(np.mean(arr))
            if arr.size < 2:
                return mean, np.nan
            se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
            return mean, 1.96 * se

        means_orig, cis_orig = zip(*[_mean_ci(arr) for arr in orig_arrays]) if distances else ([], [])
        means_perm, cis_perm = zip(*[_mean_ci(arr) for arr in perm_arrays]) if distances else ([], [])

        summary_entries: list[tuple] = []
        orig_counts = {
            data['distance']: int(data.get('n_pairs', len(data.get('raw_lyap', []))))
            for data in stability_metrics.values()
        }
        perm_counts: Dict[float, int] = {}
        for metrics_perm in stability_metrics_perm:
            for data in metrics_perm.values():
                dist = data.get('distance')
                count = int(data.get('n_pairs', len(data.get('raw_lyap', []))))
                perm_counts[dist] = perm_counts.get(dist, 0) + count

        x = np.asarray(distances, dtype=float)
        means_orig = np.asarray(means_orig, dtype=float)
        cis_orig = np.asarray(cis_orig, dtype=float)
        means_perm = np.asarray(means_perm, dtype=float)
        cis_perm = np.asarray(cis_perm, dtype=float)

        fig, ax = plt.subplots(figsize=(12, 6))

        mask_orig = np.isfinite(means_orig)
        if np.any(mask_orig):
            ax.fill_between(x[mask_orig], means_orig[mask_orig] - cis_orig[mask_orig], means_orig[mask_orig] + cis_orig[mask_orig], color="C0", alpha=0.2, linewidth=0)
            ax.plot(x[mask_orig], means_orig[mask_orig], color="C0", lw=2, label="Original mean")
            ax.scatter(x[mask_orig], means_orig[mask_orig], color="C0", edgecolor="white", zorder=3)

        mask_perm = np.isfinite(means_perm)
        if np.any(mask_perm):
            ax.fill_between(x[mask_perm], means_perm[mask_perm] - cis_perm[mask_perm], means_perm[mask_perm] + cis_perm[mask_perm], color="C1", alpha=0.2, linewidth=0)
            ax.plot(x[mask_perm], means_perm[mask_perm], color="C1", lw=2, label="Permuted mean")
            ax.scatter(x[mask_perm], means_perm[mask_perm], color="C1", edgecolor="white", zorder=3)

        # Significance tests per bin (original vs permuted)
        from scipy.stats import ttest_ind

        pvals = []
        idxs = []
        test_stats = []
        for idx, (orig_arr, perm_arr) in enumerate(zip(orig_arrays, perm_arrays)):
            if orig_arr.size >= 2 and perm_arr.size >= 2:
                stat, p = ttest_ind(orig_arr, perm_arr, equal_var=False)
                pvals.append(float(p))
                idxs.append(idx)
                test_stats.append(float(stat))

        adj = np.array([])
        if pvals:
            pvals_array = np.array(pvals, dtype=float)
            if correction == "fdr_bh":
                adj = cls._fdr_bh(pvals_array)
            elif correction == "bonferroni":
                adj = np.minimum(1.0, pvals_array * len(pvals_array))
            else:
                adj = pvals_array

        def star_fn(p):
            if not np.isfinite(p):
                return ""
            return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < alpha else ""

        if adj.size:
            upper_orig = means_orig + cis_orig
            upper_perm = means_perm + cis_perm
            lower_orig = means_orig - cis_orig
            lower_perm = means_perm - cis_perm
            parts = [
                upper_orig[np.isfinite(upper_orig)],
                upper_perm[np.isfinite(upper_perm)],
                lower_orig[np.isfinite(lower_orig)],
                lower_perm[np.isfinite(lower_perm)],
            ]
            non_empty_parts = [part for part in parts if part.size]
            cat = np.concatenate(non_empty_parts) if non_empty_parts else np.array([])
            if cat.size == 0:
                y_off_base = 1.0
                span = 1.0
            else:
                y_min = np.nanmin(cat)
                y_max = np.nanmax(cat)
                span = y_max - y_min if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0
                y_off_base = y_max

            y_offset = 0.04 * span
            for seq_idx, (bin_idx, p_adj, stat) in enumerate(zip(idxs, adj, test_stats)):
                star = star_fn(p_adj)
                raw_p = pvals[seq_idx]
                local_hi = np.nanmax([
                    upper_orig[bin_idx] if np.isfinite(upper_orig[bin_idx]) else np.nan,
                    upper_perm[bin_idx] if np.isfinite(upper_perm[bin_idx]) else np.nan,
                ])
                if not np.isfinite(local_hi):
                    local_hi = y_off_base
                if star:
                    ax.text(
                        x[bin_idx],
                        local_hi + y_offset,
                        star,
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color="k",
                    )
                dist = x[bin_idx]
                summary_entries.append(
                    (
                        dist,
                        orig_counts.get(dist, orig_arr.size),
                        float(means_orig[bin_idx]),
                        means_orig[bin_idx] - cis_orig[bin_idx] if np.isfinite(cis_orig[bin_idx]) else float('nan'),
                        means_orig[bin_idx] + cis_orig[bin_idx] if np.isfinite(cis_orig[bin_idx]) else float('nan'),
                        perm_counts.get(dist, perm_arr.size),
                        float(means_perm[bin_idx]),
                        means_perm[bin_idx] - cis_perm[bin_idx] if np.isfinite(cis_perm[bin_idx]) else float('nan'),
                        means_perm[bin_idx] + cis_perm[bin_idx] if np.isfinite(cis_perm[bin_idx]) else float('nan'),
                        stat,
                        raw_p,
                        p_adj,
                    )
                )
        # Add rows for bins without valid comparison
        compared_indices = set(idxs)
        for bin_idx, dist in enumerate(x):
            if bin_idx in compared_indices:
                continue
            orig_arr = orig_arrays[bin_idx]
            perm_arr = perm_arrays[bin_idx]
            summary_entries.append(
                (
                    dist,
                    orig_counts.get(dist, orig_arr.size),
                    float(means_orig[bin_idx]),
                    means_orig[bin_idx] - cis_orig[bin_idx] if np.isfinite(cis_orig[bin_idx]) else float('nan'),
                    means_orig[bin_idx] + cis_orig[bin_idx] if np.isfinite(cis_orig[bin_idx]) else float('nan'),
                    perm_counts.get(dist, perm_arr.size),
                    float(means_perm[bin_idx]),
                    means_perm[bin_idx] - cis_perm[bin_idx] if np.isfinite(cis_perm[bin_idx]) else float('nan'),
                    means_perm[bin_idx] + cis_perm[bin_idx] if np.isfinite(cis_perm[bin_idx]) else float('nan'),
                    float('nan'),
                    float('nan'),
                    float('nan'),
                )
            )

        ax.set_xlabel("Distance (μm)")
        ax.set_ylabel("Lyapunov Exponent")
        ax.set_title("Lyapunov Mean ± 95% CI (Original vs Permuted)")
        ax.grid(True, which="both", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if correlation_curve is not None:
            corr_x, corr_y = correlation_curve
            fit = cls._fit_correlation_curve(
                corr_x,
                corr_y,
                x[mask_orig],
                means_orig[mask_orig],
                allow_offset=True,
            ) if np.any(mask_orig) else None
            if fit is not None:
                corr_x_fit, corr_y_fit, amp, offset = fit
                label = f'Correlation fit (amp={amp:.2g}, b={offset:.2g})'
            else:
                corr_x_fit, corr_y_fit = np.asarray(corr_x, dtype=float), np.asarray(corr_y, dtype=float)
                label = 'Correlation model'
            ax_corr = ax.twinx()
            ax_corr.plot(corr_x_fit, corr_y_fit, color='crimson', linestyle='--', label=label)
            ax_corr.set_ylabel('Synergy model (a.u.)')
            if np.any(np.isfinite(corr_y_fit)):
                ax_corr.set_ylim(float(np.nanmin(corr_y_fit)), float(np.nanmax(corr_y_fit)))
            lines_data, labels_data = ax.get_legend_handles_labels()
            lines_model, labels_model = ax_corr.get_legend_handles_labels()
            ax.legend(lines_data + lines_model, labels_data + labels_model, loc='upper right')
        else:
            ax.legend(frameon=False)
        plt.tight_layout()
        plt.show()

        if summary_entries:
            summary_entries.sort(key=lambda row: row[0])
            print("\nLyapunov per-bin summary (Original vs Permuted, Welch t-test):")
            header = (
                "Distance μm | n_orig | mean_orig | ci_orig_low | ci_orig_high | "
                "n_perm | mean_perm | ci_perm_low | ci_perm_high | t-stat | p_raw | p_adj"
            )
            print(header)
            for (
                dist,
                n_o,
                mean_o,
                ci_o_low,
                ci_o_high,
                n_p,
                mean_p,
                ci_p_low,
                ci_p_high,
                stat,
                raw_p,
                p_adj,
            ) in summary_entries:
                print(
                    f"{dist:10.0f} | {n_o:6d} | {mean_o:9.4f} | {ci_o_low:11.4f} | {ci_o_high:11.4f} | "
                    f"{n_p:6d} | {mean_p:9.4f} | {ci_p_low:11.4f} | {ci_p_high:11.4f} | {stat:6.3f} | {raw_p:5.2e} | {p_adj:5.2e}"
                )
