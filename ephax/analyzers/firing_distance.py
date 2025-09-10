from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, f_oneway, tukey_hsd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.feature_selection import mutual_info_regression

from ..prep import RestingActivityDataset, PrepConfig
from ..models import BinnedSeries
from .ifr import IFRAnalyzer, IFRConfig
from ephax.helper_functions import assign_r_distance_all
from ephax.helper_functions import log_likelihood, likelihood_ratio_test
from ephax.r_function import correlation_function
from ephax.helper_functions import assign_r_distance
from ..compute import cofiring_proportions


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


class FiringDistanceAnalyzer:
    """Analyze firing/co-firing statistics versus distance between electrodes.

    Provides averaged rate vs distance, co-firing vs distance, weighted pairwise
    distance histograms, synergy overlays using correlation_function, and model
    comparison utilities for reduced models.
    """

    def __init__(
        self,
        dataset: RestingActivityDataset,
        refs_per_recording: Optional[List[np.ndarray]] = None,
        selection_prep_config: Optional[PrepConfig] = None,
        # Default model parameters for synergy overlays
        v_eph: float = 0.1,
        v_ax: float = 0.45,
        std: float = 0.15,
        lambda_eph: float = 100000.0,
    ) -> None:
        self.ds = dataset
        self._stored_refs: Optional[List[np.ndarray]] = refs_per_recording
        self._selection_cfg: Optional[PrepConfig] = selection_prep_config
        # Store default model parameters
        self.v_eph = float(v_eph)
        self.v_ax = float(v_ax)
        self.std = float(std)
        self.lambda_eph = float(lambda_eph)
        # If no refs provided, compute defaults immediately for determinism
        if self._stored_refs is None:
            self._stored_refs = self._compute_default_refs()

    # ----- Firing rate vs distance -----
    def avg_rate_vs_distance(
        self,
        refs_per_recording: Optional[List[np.ndarray]] = None,
        log: bool = False,
        min_distance: float = 50,
        max_distance: float = 3500,
    ) -> FRDistanceResult:
        """Average firing rate vs distance, binned and aggregated across recordings."""
        refs = self._ensure_refs(refs_per_recording)
        all_rates: List[float] = []
        all_dists: List[float] = []

        for rec, rlist in zip(self.ds.recordings, refs):
            if rlist is None or len(rlist) == 0:
                continue
            spikes_df = pd.DataFrame(rec.spikes)
            layout_df = pd.DataFrame(rec.layout)
            spikes_df, distances_df = assign_r_distance_all(spikes_df, layout_df, rlist)

            mask_t = (spikes_df["time"] >= rec.start_time) & (spikes_df["time"] <= rec.end_time)
            spikes_df_during = spikes_df[mask_t]
            duration = float(rec.end_time - rec.start_time)
            if duration <= 0:
                continue
            counts = spikes_df_during["electrode"].value_counts().reset_index()
            counts.columns = ["electrode", "counts"]
            counts["firing_rate"] = counts["counts"] / duration
            merged = pd.merge(counts, distances_df, on="electrode", how="inner")
            merged = merged[merged["electrode"] != merged["ref_electrode"]]
            all_rates.extend(merged["firing_rate"].astype(float).tolist())
            all_dists.extend(merged["distance"].astype(float).tolist())

        if len(all_dists) == 0:
            centers = np.array([])
            return FRDistanceResult(
                distances=np.array([]), rates=np.array([]), bins=np.array([]),
                binned=BinnedSeries(centers=centers, mean=centers, stderr=centers)
            )

        all_rates = np.asarray(all_rates, dtype=float)
        all_dists = np.asarray(all_dists, dtype=float)

        if log:
            bins = np.logspace(np.log10(max(min_distance, all_dists.min())), np.log10(max_distance), num=100)
        else:
            bins = np.linspace(max(min_distance, all_dists.min()), max_distance, num=100)

        bin_means, bin_edges, _ = binned_statistic(all_dists, all_rates, statistic=np.nanmean, bins=bins)
        bin_std_err, _, _ = binned_statistic(
            all_dists, all_rates, statistic=lambda x: np.std(x) / np.sqrt(len(x)), bins=bins
        )
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        valid = ~np.isnan(bin_means)
        binned = BinnedSeries(centers=centers[valid], mean=bin_means[valid], stderr=bin_std_err[valid])
        return FRDistanceResult(distances=all_dists, rates=all_rates, bins=bins, binned=binned)

    def plot_rate_with_synergy(
        self,
        result: FRDistanceResult,
        v_eph: Optional[float] = None,
        v_ax: Optional[float] = None,
        lambda_eph: Optional[float] = None,
        std: Optional[float] = None,
        peak_min_hz: float = 30.0,
        peak_max_hz: float = 1000.0,
        title: str = "Averaged Firing Rate vs. Distance with First Synergy Coefficient Peaks",
    ):
        """Plot rate vs distance with synergy overlays and model diagnostics."""
        if result.binned.centers.size == 0:
            print("No data to plot.")
            return None, None
        # Primary axis (FR)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(result.binned.centers, result.binned.mean, color='blue', label='Mean Firing Rate')
        ax1.fill_between(result.binned.centers, result.binned.mean - result.binned.stderr, result.binned.mean + result.binned.stderr, alpha=0.4, color='blue')
        ax1.set_xlabel('Distance from Electrode ($\\mu m$)')
        ax1.set_ylabel('Mean Firing Rate (Hz)')
        ax1.set_title(title)

        # IFR peaks
        ok, gamma_hz_list, weights = self._compute_ifr_peaks_weights(peak_min_hz, peak_max_hz)
        if not ok:
            return fig, ax1

        # Build model
        v_eph_val = self.v_eph if v_eph is None else float(v_eph)
        v_ax_val = self.v_ax if v_ax is None else float(v_ax)
        lambda_eph = self.lambda_eph if lambda_eph is None else float(lambda_eph)
        std_val = self.std if std is None else float(std)
        v_eph_um_s = v_eph_val * 1e6
        v_ax_um_s = v_ax_val * 1e6
        min_distance = 50.0
        max_distance = 3500.0
        r_um = np.linspace(min_distance, max_distance, 1000)
        total_r_full = np.zeros_like(r_um)
        for hz, w in zip(gamma_hz_list, weights):
            total_r_full += correlation_function(r_um, hz, v_eph_um_s, v_ax_um_s, lambda_eph) * w

        # Fit/diagnostics
        y = np.interp(r_um, result.binned.centers, result.binned.mean)
        full_residuals = y - total_r_full
        logL_full = log_likelihood(full_residuals, len(gamma_hz_list))
        reduced_models = []
        upper_cis = []
        lower_cis = []
        bic_values = []
        mi_values = []
        lrt_values = []
        residuals_dict = {}
        current_gamma_list = list(gamma_hz_list)
        steps = 50
        while len(current_gamma_list) > 0:
            total_r_reduced = np.zeros_like(r_um)
            total_r_reduced_list = [np.zeros_like(r_um) for _ in range(steps)]
            current_weights = weights[:len(current_gamma_list)]
            for hz, w in zip(current_gamma_list, current_weights):
                total_r_reduced += correlation_function(r_um, hz, v_eph_um_s, v_ax_um_s, lambda_eph) * w
                for idx, v in enumerate(np.linspace(v_ax_um_s - std_val * 1e6, v_ax_um_s + std_val * 1e6, steps)):
                    total_r_reduced_list[idx] += correlation_function(r_um, hz, v_eph_um_s, v, lambda_eph) * w
            reduced_residuals = y - total_r_reduced
            logL_reduced = log_likelihood(reduced_residuals, len(current_gamma_list))
            print(f"Log-likelihood of full model: {logL_full}")
            print(f"Log-likelihood of reduced model: {logL_reduced}")
            df = len(gamma_hz_list) - len(current_gamma_list)
            LRT_stat, p_value = likelihood_ratio_test(logL_full, logL_reduced, df)
            lrt_values.append((LRT_stat, p_value))
            residuals = y - total_r_reduced
            print(len(residuals[np.isfinite(residuals)]))
            bic = (len(y) * np.log(np.sum(residuals ** 2) / len(y))) + (len(current_gamma_list) * np.log(len(y)))
            bic_values.append(bic)
            mi = mutual_info_regression(total_r_reduced.reshape(-1, 1), y)
            mi_values.append(mi[0])
            reduced_models.append((current_gamma_list.copy(), total_r_reduced))
            stacked = np.stack(total_r_reduced_list)
            upper_cis.append(np.max(stacked, axis=0))
            lower_cis.append(np.min(stacked, axis=0))
            # Store residuals for model comparison
            residuals_dict[f'Reduced Model {len(current_gamma_list)} Hz'] = np.array(residuals.tolist(), dtype=float)
            current_gamma_list = []

        # Compare models using statistical tests (only if more than one model)
        if len(residuals_dict) > 1:
            FiringDistanceAnalyzer.compare_model_fits(residuals_dict)
        else:
            print("Skipped statistical model comparison: need at least two models.")

        # Twin axis + shading
        ax2 = ax1.twinx()
        ax2.set_ylabel('Synergy Coefficient')
        inv_velocity_diff = (1 / v_eph_um_s) - (1 / v_ax_um_s)
        velocity_factor = 1 / inv_velocity_diff if inv_velocity_diff != 0 else 0.0
        ymin, ymax = ax1.get_ylim()
        try:
            from ..helper_functions import truncate_colormap as _truncate
            base_cmap = _truncate(plt.get_cmap('viridis'), 0.4, 0.9)
        except Exception:
            base_cmap = plt.get_cmap('viridis')
        self._shade_first_peaks(ax1, r_um, gamma_hz_list, v_eph_um_s, v_ax_um_s, lambda_eph, ymin, ymax, base_cmap)
        for idx, (frequencies, model_curve) in enumerate(reduced_models):
            color = base_cmap(0.5)
            ax2.plot(r_um, model_curve, linestyle='--', color=color, label=f'Model with {[round(f) for f in frequencies]} Hz')
            ax2.fill_between(r_um, lower_cis[idx], upper_cis[idx], alpha=0.2, color=color)
        ax2.set_ylim(np.min(total_r_full), np.max(total_r_full))
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        return fig, (ax1, ax2)

    # ----- Shared helpers -----
    def _compute_ifr_peaks_weights(self, peak_min_hz: float, peak_max_hz: float):
        spikes_list, layout_list, start_times, end_times = self.ds.to_legacy()
        # Build IFR analyzer that respects the same electrode selection policy
        if self._selection_cfg is not None:
            ifr = IFRAnalyzer.from_dataset(self.ds, config=IFRConfig(log_scale=True), selection_prep_config=self._selection_cfg)
        else:
            ifr = IFRAnalyzer(spikes_list, start_times, end_times, config=IFRConfig(log_scale=True))
            # Fall back to stored refs computed in __init__
            ifr._refs_per_recording = self._stored_refs
        peaks = ifr.peaks()
        fit = ifr.fit_gmm(peaks.values)
        means = np.asarray(fit.means_hz, dtype=float)
        weights = np.asarray(fit.weights, dtype=float)
        mask = (means > peak_min_hz) & (means < peak_max_hz)
        gamma_hz_list = np.sort(means[mask])
        weights = weights[mask]
        if weights.size == 0 or gamma_hz_list.size == 0:
            return False, np.array([]), np.array([])
        weights = weights / max(1e-9, weights.min())
        print(f'Included IFR Peaks: {gamma_hz_list}')
        return True, gamma_hz_list, weights

    def _shade_first_peaks(
        self,
        ax1,
        r_um: np.ndarray,
        gamma_hz_list: np.ndarray,
        v_eph_um_s: float,
        v_ax_um_s: float,
        lambda_eph: float,
        ymin: float,
        ymax: float,
        base_cmap,
    ):
        inv_velocity_diff = (1 / v_eph_um_s) - (1 / v_ax_um_s)
        velocity_factor = 1 / inv_velocity_diff if inv_velocity_diff != 0 else 0.0
        for i, hz in enumerate(gamma_hz_list):
            delta_t_start = (3 / (4 * hz))
            delta_t_end = (5 / (4 * hz))
            r_start = delta_t_start * velocity_factor
            r_end = delta_t_end * velocity_factor
            r0 = max(r_start, r_um.min())
            r1 = min(r_end, r_um.max())
            if r1 <= r0:
                continue
            sel = (r_um >= r0) & (r_um <= r1)
            if not np.any(sel):
                continue
            R = correlation_function(r_um[sel], hz, v_eph_um_s, v_ax_um_s, lambda_eph)
            Rn = (R - R.min()) / (R.max() - R.min() + 1e-9)
            Z = np.tile(Rn, (2, 1))
            color = base_cmap(i / max(1, len(gamma_hz_list)-1))
            extent = [r0, r1, ymin, ymax]
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', color])
            ax1.imshow(Z, extent=extent, origin='lower', aspect='auto', cmap=custom_cmap, alpha=0.5)
            try:
                max_idx = np.where(Rn == np.max(Rn))[0]
                max_positions = r_um[sel][max_idx]
                print(f'Peaks for {hz} Hz at {max_positions}')
            except Exception:
                pass

    # ----- Cofiring vs distance (no temporal dimension) -----
    def cofiring_avg_vs_distance(
        self,
        refs_per_recording: Optional[List[np.ndarray]] = None,
        plusminus_ms: float = 2.0,
        log: bool = False,
        min_distance: float = 50,
        max_distance: float = 3500,
    ) -> CofiringDistanceResult:
        """Average co-firing probability vs distance for a ±window around delay 0."""
        refs = self._ensure_refs(refs_per_recording)
        all_props: List[float] = []
        all_dists: List[float] = []
        # Convert symmetric ±window (in milliseconds) to seconds for cofiring_proportions API
        pm_sec = float(plusminus_ms) / 1000.0
        window_size_sec = 2.0 * pm_sec
        delay_sec = -pm_sec

        for rec, rlist in zip(self.ds.recordings, refs):
            if rlist is None or len(rlist) == 0:
                continue
            spikes_df_full = pd.DataFrame(rec.spikes)
            layout_df_full = pd.DataFrame(rec.layout)
            # Restrict to window once
            mask_t = (spikes_df_full["time"] >= rec.start_time) & (spikes_df_full["time"] <= rec.end_time)
            spikes_df_window = spikes_df_full[mask_t].copy()

            for ref in rlist:
                # Assign distances relative to ref
                spikes_df, layout_df = assign_r_distance(spikes_df_window.copy(), layout_df_full.copy(), int(ref))
                firing_times = spikes_df["time"][spikes_df["electrode"] == int(ref)]
                props = cofiring_proportions(
                    spikes_df,
                    firing_times,
                    window_size=window_size_sec,
                    delay=delay_sec,
                    ref_electrode=int(ref),
                )
                for electrode, proportion in props.items():
                    if electrode == int(ref):
                        continue
                    all_props.append(float(proportion))
                    d = layout_df.loc[layout_df["electrode"] == electrode, "distance"].values[0]
                    all_dists.append(float(d))

        if not all_dists:
            centers = np.array([])
            return CofiringDistanceResult(
                distances=np.array([]), proportions=np.array([]), bins=np.array([]),
                binned=BinnedSeries(centers=centers, mean=centers, stderr=centers)
            )

        all_props = np.asarray(all_props, dtype=float)
        all_dists = np.asarray(all_dists, dtype=float)

        if log:
            bins = np.logspace(np.log10(max(min_distance, all_dists.min())), np.log10(max_distance), num=50)
        else:
            bins = np.linspace(max(min_distance, all_dists.min()), max_distance, num=50)

        bin_means, bin_edges, _ = binned_statistic(all_dists, all_props, statistic=np.nanmean, bins=bins)
        bin_std_err, _, _ = binned_statistic(
            all_dists, all_props, statistic=lambda x: np.std(x) / np.sqrt(len(x)), bins=bins
        )
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        valid = ~np.isnan(bin_means)
        binned = BinnedSeries(centers=centers[valid], mean=bin_means[valid], stderr=bin_std_err[valid])
        return CofiringDistanceResult(distances=all_dists, proportions=all_props, bins=bins, binned=binned)

    # ----- Model comparison utilities -----
    @staticmethod
    def compare_model_fits(residuals_dict: dict) -> None:
        """Compare models via ANOVA and (if significant) Tukey's HSD on residuals.

        residuals_dict: mapping of model label -> iterable of residuals.
        Prints ANOVA stats and Tukey HSD table when applicable.
        """
        residuals_combined = []
        group_labels = []
        for label, residuals in residuals_dict.items():
            residuals_combined.extend(residuals)
            group_labels.extend([label] * len(residuals))
        residuals_combined = np.asarray(residuals_combined, dtype=float)
        group_labels = np.asarray(group_labels)
        valid = np.isfinite(residuals_combined)
        residuals_combined = residuals_combined[valid]
        group_labels = group_labels[valid]
        try:
            groups = [residuals_combined[group_labels == label] for label in residuals_dict.keys()]
            f_stat, p_value = f_oneway(*groups)
            print(f"ANOVA F-statistic: {f_stat}, p-value: {p_value}")
        except Exception as e:
            print(f"Error during ANOVA test: {e}")
            return
        if p_value < 0.05:
            try:
                labels_numeric = pd.Categorical(group_labels).codes
                result = tukey_hsd(residuals_combined, labels_numeric)
                print("Tukey's HSD Results:")
                print(result)
            except Exception as e:
                print(f"Error running Tukey's HSD: {e}")

    def plot_cofiring_with_synergy(
        self,
        result: CofiringDistanceResult,
        v_eph: Optional[float] = None,
        v_ax: Optional[float] = None,
        lambda_eph: Optional[float] = None,
        std: Optional[float] = None,
        peak_min_hz: float = 30.0,
        peak_max_hz: float = 1000.0,
    ):
        """Plot co-firing vs distance with synergy overlays and model diagnostics."""
        if result.binned.centers.size == 0:
            print("No data to plot.")
            return None, None

        # Primary axis: mean ± stderr
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(result.binned.centers, result.binned.mean, color='blue', label='Mean Co-Firing Rate')
        ax1.fill_between(
            result.binned.centers,
            result.binned.mean - result.binned.stderr,
            result.binned.mean + result.binned.stderr,
            alpha=0.4,
            color='blue'
        )
        ax1.set_xlabel('Distance from Electrode ($\\mu m$)')
        ax1.set_ylabel('Mean Co-Firing Probability')
        ax1.set_title('Averaged Co-Firing Probability vs. Distance')

        # IFR peaks (respect same selection policy as analyzer)
        ok, gamma_hz_list, weights = self._compute_ifr_peaks_weights(peak_min_hz, peak_max_hz)
        if not ok:
            return fig, ax1

        # Model
        v_eph_val = self.v_eph if v_eph is None else float(v_eph)
        v_ax_val = self.v_ax if v_ax is None else float(v_ax)
        lambda_eph = self.lambda_eph if lambda_eph is None else float(lambda_eph)
        std_val = self.std if std is None else float(std)
        v_eph_um_s = v_eph_val * 1e6
        v_ax_um_s = v_ax_val * 1e6
        min_distance = 50.0
        max_distance = 3500.0
        r_um = np.linspace(min_distance, max_distance, 1000)
        total_r_full = np.zeros_like(r_um)
        for hz, w in zip(gamma_hz_list, weights):
            total_r_full += correlation_function(r_um, hz, v_eph_um_s, v_ax_um_s, lambda_eph) * w

        # Fit and diagnostics
        y = np.interp(r_um, result.binned.centers, result.binned.mean)
        full_residuals = y - total_r_full
        logL_full = log_likelihood(full_residuals, len(gamma_hz_list))
        reduced_models = []
        upper_cis = []
        lower_cis = []
        bic_values = []
        mi_values = []
        lrt_values = []
        residuals_dict = {}
        current_gamma_list = list(gamma_hz_list)
        steps = 50
        while len(current_gamma_list) > 0:
            total_r_reduced = np.zeros_like(r_um)
            total_r_reduced_list = [np.zeros_like(r_um) for _ in range(steps)]
            current_weights = weights[:len(current_gamma_list)]
            for hz, w in zip(current_gamma_list, current_weights):
                total_r_reduced += correlation_function(r_um, hz, v_eph_um_s, v_ax_um_s, lambda_eph) * w
                for idx, v in enumerate(np.linspace(v_ax_um_s - std_val * 1e6, v_ax_um_s + std_val * 1e6, steps)):
                    total_r_reduced_list[idx] += correlation_function(r_um, hz, v_eph_um_s, v, lambda_eph) * w
            reduced_residuals = y - total_r_reduced
            logL_reduced = log_likelihood(reduced_residuals, len(current_gamma_list))
            print(f"Log-likelihood of full model: {logL_full}")
            print(f"Log-likelihood of reduced model: {logL_reduced}")
            df = len(gamma_hz_list) - len(current_gamma_list)
            LRT_stat, p_value = likelihood_ratio_test(logL_full, logL_reduced, df)
            lrt_values.append((LRT_stat, p_value))
            residuals = y - total_r_reduced
            print(len(residuals[np.isfinite(residuals)]))
            bic = (len(y) * np.log(np.sum(residuals ** 2) / len(y))) + (len(current_gamma_list) * np.log(len(y)))
            bic_values.append(bic)
            mi = mutual_info_regression(total_r_reduced.reshape(-1, 1), y)
            mi_values.append(mi[0])
            reduced_models.append((current_gamma_list.copy(), total_r_reduced))
            stacked = np.stack(total_r_reduced_list)
            upper_cis.append(np.max(stacked, axis=0))
            lower_cis.append(np.min(stacked, axis=0))
            residuals_dict[f'Reduced Model {len(current_gamma_list)} Hz'] = np.array(residuals.tolist(), dtype=float)
            current_gamma_list = []

        # Compare models using statistical tests (only if more than one model)
        if len(residuals_dict) > 1:
            FiringDistanceAnalyzer.compare_model_fits(residuals_dict)
        else:
            print("Skipped statistical model comparison: need at least two models.")

        # Secondary axis and shading
        ax2 = ax1.twinx()
        ax2.set_ylabel('Synergy Correlation Function (a.u.)')
        inv_velocity_diff = (1 / v_eph_um_s) - (1 / v_ax_um_s)
        velocity_factor = 1 / inv_velocity_diff if inv_velocity_diff != 0 else 0.0
        ymin, ymax = ax1.get_ylim()
        try:
            from ..helper_functions import truncate_colormap as _truncate
            base_cmap = _truncate(plt.get_cmap('viridis'), 0.4, 0.9)
        except Exception:
            base_cmap = plt.get_cmap('viridis')

        self._shade_first_peaks(ax1, r_um, gamma_hz_list, v_eph_um_s, v_ax_um_s, lambda_eph, ymin, ymax, base_cmap)

        # Plot reduced model on ax2
        for idx, (frequencies, model_curve) in enumerate(reduced_models):
            color = base_cmap(0.5)
            ax2.plot(r_um, model_curve, linestyle='--', color=color, label=f'Model with {[round(f) for f in frequencies]} Hz')
            ax2.fill_between(r_um, lower_cis[idx], upper_cis[idx], alpha=0.2, color=color)

        ax2.set_ylim(np.min(total_r_full), np.max(total_r_full))
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        return fig, (ax1, ax2)

    # ----- Pairwise distance histogram of active electrodes -----
    @staticmethod
    def _boundary_weight(x1, y1, x2, y2, max_x, max_y) -> float:
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if dist == 0:
            return 1.0
        radius = dist
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        num_points = 100
        theta = np.linspace(0, 2 * np.pi, num_points)
        circle_x = mid_x + radius * np.cos(theta)
        circle_y = mid_y + radius * np.sin(theta)
        outside_count = np.sum((circle_x < 0) | (circle_x > max_x) | (circle_y < 0) | (circle_y > max_y))
        proportion_outside = outside_count / num_points
        proportion_inside = max(0.01, 1 - proportion_outside)
        weight = 1 / proportion_inside
        weight /= (2 * np.pi * radius)
        weight /= 2
        return weight

    def distance_histogram(
        self,
        refs_per_recording: Optional[List[np.ndarray]] = None,
        finite_size_correction: bool = True,
        min_distance: float = 50,
        max_distance: float = 3500,
        bins: int = 74,
    ):
        """Compute pairwise distances between refs (per recording) with boundary weights."""
        refs = self._ensure_refs(refs_per_recording)
        all_distances: List[float] = []
        all_weights: List[float] = []
        for rec, rlist in zip(self.ds.recordings, refs):
            if rlist is None or len(rlist) == 0:
                continue
            layout_df = pd.DataFrame(rec.layout)
            # Coordinates for refs
            coords = layout_df.set_index('electrode').loc[rlist, ['x', 'y']].dropna().to_numpy()
            max_x = layout_df['x'].max()
            max_y = layout_df['y'].max()
            n = coords.shape[0]
            for i in range(n):
                xi, yi = coords[i]
                for j in range(i + 1, n):
                    xj, yj = coords[j]
                    dist = float(np.linalg.norm([xj - xi, yj - yi]))
                    all_distances.append(dist)
                    if finite_size_correction:
                        all_weights.append(self._boundary_weight(xi, yi, xj, yj, max_x, max_y))
                    else:
                        all_weights.append(1.0)
        return np.asarray(all_distances, dtype=float), np.asarray(all_weights, dtype=float)

    def plot_distance_hist_with_synergy(
        self,
        distances: np.ndarray,
        weights: np.ndarray | None,
        v_eph: Optional[float] = None,
        v_ax: Optional[float] = None,
        lambda_eph: Optional[float] = None,
        peak_min_hz: float = 30.0,
        peak_max_hz: float = 1000.0,
        bins: int = 74,
    ):
        """Plot weighted histogram of pairwise distances with synergy shading."""
        fig, ax1 = plt.subplots(figsize=(10, 6))
        if weights is not None and weights.size == distances.size:
            ax1.hist(distances, bins=bins, weights=weights, alpha=0.4)
        else:
            ax1.hist(distances, bins=bins, alpha=0.4)
        ymin, ymax = ax1.get_ylim()
        ax1.set_xlim(0, 3500)
        ax1.set_xlabel('Distance between Most Active Electrodes ($\\mu m$)')
        ax1.set_ylabel('Weighted Count')
        ax1.set_title('Weighted Histogram of Distances Between Most Active Electrodes Across Recordings')

        ok, gamma_hz_list, weights_pk = self._compute_ifr_peaks_weights(peak_min_hz, peak_max_hz)
        if not ok:
            return fig, ax1
        v_eph_val = self.v_eph if v_eph is None else float(v_eph)
        v_ax_val = self.v_ax if v_ax is None else float(v_ax)
        lambda_eph = self.lambda_eph if lambda_eph is None else float(lambda_eph)
        v_eph_um_s = v_eph_val * 1e6
        v_ax_um_s = v_ax_val * 1e6
        r_um = np.linspace(50.0, 3500.0, 1000)
        try:
            from ..helper_functions import truncate_colormap as _truncate
            base_cmap = _truncate(plt.get_cmap('viridis'), 0.4, 0.9)
        except Exception:
            base_cmap = plt.get_cmap('viridis')
        self._shade_first_peaks(ax1, r_um, gamma_hz_list, v_eph_um_s, v_ax_um_s, lambda_eph, ymin, ymax, base_cmap)
        return fig, ax1

    # ----- Internal: refs management -----
    def _compute_default_refs(self) -> List[np.ndarray]:
        # Prefer provided selection config; else default to legacy-like top selection
        cfg = self._selection_cfg or PrepConfig(mode='top', top_start=10, top_stop=210, top_use_recording_window=True, verbose=True)
        try:
            return self.ds.select_ref_electrodes(cfg)
        except Exception:
            # Fallback: use unique electrodes per recording
            out: List[np.ndarray] = []
            for rec in self.ds.recordings:
                import numpy as np
                out.append(np.unique(rec.spikes.get('electrode', np.array([], dtype=int))))
            return out

    def _ensure_refs(self, refs_per_recording: Optional[List[np.ndarray]]) -> List[np.ndarray]:
        if refs_per_recording is not None:
            return refs_per_recording
        if self._stored_refs is not None:
            return self._stored_refs
        self._stored_refs = self._compute_default_refs()
        return self._stored_refs
