from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="Intel MKL WARNING")
warnings.filterwarnings("ignore", message="RuntimeWarning: overflow encountered")

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, f_oneway, tukey_hsd, pearsonr
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.feature_selection import mutual_info_regression

from ..prep import RestingActivityDataset, PrepConfig
from ..models import BinnedSeries
from .ifr import IFRAnalyzer, IFRConfig
from ephax.helper_functions import assign_r_distance_all, assign_r_distance, log_likelihood, likelihood_ratio_test
from ephax.r_function import correlation_function
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
        dataset_perm: RestingActivityDataset | None = None,
        refs_per_recording: Optional[List[np.ndarray]] = None,
        selection_prep_config: Optional[PrepConfig] = None,
        # Default model parameters for synergy overlays
        v_eph: float = 0.1,
        v_ax: float = 0.45,
        std: float = 0.15,
        lambda_eph: float = 100000.0,
    ) -> None:
        self.ds = dataset
        self.ds_perm = dataset_perm
        self._stored_refs: Optional[List[np.ndarray]] = refs_per_recording
        self._stored_refs_perm: Optional[List[np.ndarray]] = None
        self._selection_cfg: Optional[PrepConfig] = selection_prep_config
        # Store default model parameters
        self.v_eph = float(v_eph)
        self.v_ax = float(v_ax)
        self.std = float(std)
        self.lambda_eph = float(lambda_eph)
        # If no refs provided, compute defaults immediately for determinism
        if self._stored_refs is None:
            self._stored_refs = self._compute_default_refs(dataset=self.ds)
        if self.ds_perm is not None and self._stored_refs_perm is None:
            self._stored_refs_perm = self._compute_default_refs(dataset=self.ds_perm)

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
            bins = np.logspace(np.log10(max(min_distance, all_dists.min())), np.log10(max_distance), num=74)
        else:
            bins = np.linspace(max(min_distance, all_dists.min()), max_distance, num=74)

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
        title: str = "Averaged Firing Rate vs. Distance with First Signal Synergy Peaks",
        show_exponential_fit: bool = True,
        allow_offset: bool = False,
    ):
        """Plot rate vs distance with synergy overlays and model diagnostics."""
        if result.binned.centers.size == 0:
            return None, None
        # Three rows: top=data+fit+shading, middle=residuals+model, bottom=scatter of predicted vs actual residuals
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.35)
        ax1 = fig.add_subplot(gs[0])
        ax_model = fig.add_subplot(gs[1], sharex=ax1)
        ax_scatter = fig.add_subplot(gs[2])
        ax1.plot(result.binned.centers, result.binned.mean, color='blue', label='Mean Firing Rate')
        ax1.fill_between(result.binned.centers, result.binned.mean - result.binned.stderr, result.binned.mean + result.binned.stderr, alpha=0.4, color='blue')
        ax1.set_xlabel('Distance from Electrode ($\\mu m$)')
        ax1.set_ylabel('Mean Firing Rate (Hz)')
        ax1.set_title(title)

        # Exponential fit: y ≈ A * exp(-k x) + C on binned means (TOP subplot)
        exp_popt = None
        try:
            x_centers = np.asarray(result.binned.centers, dtype=float)
            y_mean = np.asarray(result.binned.mean, dtype=float)
            y_err = np.asarray(result.binned.stderr, dtype=float)
            valid = np.isfinite(x_centers) & np.isfinite(y_mean)
            x_valid = x_centers[valid]
            y_valid = y_mean[valid]
            if x_valid.size >= 3:
                def _exp_fun(x, A, k, C):
                    return A * np.exp(-k * x) + C
                A0 = float(max(1e-12, y_valid[0] - y_valid[-1]))
                k0 = 1.0 / max(1e-9, (x_valid.max() - x_valid.min()))
                C0 = float(y_valid[-1])
                exp_popt, _ = curve_fit(_exp_fun, x_valid, y_valid, p0=[A0, k0, C0], maxfev=10000)
                if show_exponential_fit:
                    # Plot fit evaluated at the same bin centers for visual consistency
                    ax1.plot(x_valid, _exp_fun(x_valid, *exp_popt), linestyle='-.', color='orange', label=f'Exp fit (A={exp_popt[0]:.2e}, k={exp_popt[1]:.2e})')
        except Exception:
            pass
        ax1.legend(loc='upper left')
        ax1.legend(loc='upper left')

        synergy = self._prepare_synergy_components(
            peak_min_hz=peak_min_hz,
            peak_max_hz=peak_max_hz,
            v_eph=v_eph,
            v_ax=v_ax,
            lambda_eph=lambda_eph,
        )
        if synergy is None:
            return fig, ax1

        r_um, total_r_full, gamma_hz_list, weights, v_eph_um_s, v_ax_um_s, lambda_val = synergy

        ymin_top, ymax_top = ax1.get_ylim()
        try:
            from ..helper_functions import truncate_colormap as _truncate
            base_cmap = _truncate(plt.get_cmap('viridis'), 0.4, 0.9)
        except Exception:
            base_cmap = plt.get_cmap('viridis')
        self._shade_first_peaks(ax1, r_um, gamma_hz_list, v_eph_um_s, v_ax_um_s, lambda_val, ymin_top, ymax_top, base_cmap)

        y = np.interp(r_um, result.binned.centers, result.binned.mean)

        model_curves: List[tuple[list[float], np.ndarray]] = []
        current_gamma = list(gamma_hz_list)
        weight_list = list(weights)
        while current_gamma:
            curve = np.zeros_like(r_um)
            for hz, w in zip(current_gamma, weight_list[: len(current_gamma)]):
                curve += correlation_function(r_um, hz, v_eph_um_s, v_ax_um_s, lambda_val) * w
            model_curves.append((current_gamma.copy(), curve))
            current_gamma = current_gamma[:-1]

        # Model comparisons against reduced components
        residuals_dict: Dict[str, np.ndarray] = {}
        log_l_map: Dict[str, float] = {}
        bic_map: Dict[str, float] = {}
        mi_map: Dict[str, float] = {}
        labels: List[str] = []

        for gammas, curve in model_curves:
            label = f"Model ({', '.join(f'{hz:.1f}' for hz in gammas)} Hz)"
            residuals = np.asarray(y - curve, dtype=float)
            residuals_dict[label] = residuals
            labels.append(label)
            n_params = max(len(gammas), 1)
            log_l_map[label] = log_likelihood(residuals, n_params)
            rss = float(np.sum(residuals ** 2))
            bic_map[label] = (len(y) * np.log(rss / len(y))) + (n_params * np.log(len(y)))
            try:
                mi = mutual_info_regression(curve.reshape(-1, 1), y)[0]
            except Exception:
                mi = np.nan
            mi_map[label] = mi

        if labels:
            print("\n[Rate] Synergy model summary:")
            for label in labels:
                print(
                    f"  {label}: logL={log_l_map[label]:.4f}, BIC={bic_map[label]:.4f}, MI={mi_map[label]:.5f}"
                )

        if len(labels) >= 2:
            full_label = labels[0]
            full_gamma_count = len(model_curves[0][0])
            log_full = log_l_map[full_label]
            print("[Rate] Likelihood-ratio tests vs full model:")
            for label, (gammas, _) in zip(labels[1:], model_curves[1:]):
                df = full_gamma_count - len(gammas)
                stat, p_value = likelihood_ratio_test(log_full, log_l_map[label], df)
                print(f"  {label}: LRT={stat:.4f}, p={p_value:.3e}")

        if len(residuals_dict) > 1:
            self.compare_model_fits(residuals_dict)

        ax_model.set_xlabel('Distance from Electrode ($\\mu m$)')
        ax_model.set_ylabel('Residual (Hz)')
        residual_series = None
        if exp_popt is not None and x_valid.size:
            def _exp_fun(x, A, k, C):
                return A * np.exp(-k * x) + C

            y_fit_valid = _exp_fun(x_valid, *exp_popt)
            err_valid = y_err[valid]
            residual_series = y_valid - y_fit_valid
            ax_model.plot(x_valid, residual_series, color='purple', label='Residual (data − exp fit)')
            ax_model.fill_between(x_valid, residual_series - err_valid, residual_series + err_valid, alpha=0.2, color='purple')
            if residual_series.size:
                max_abs = float(np.nanmax(np.abs(residual_series)))
                if max_abs == 0:
                    max_abs = 1e-9
                ax_model.set_ylim(-1.05 * max_abs, 1.05 * max_abs)
            ax_model.axhline(0.0, color='gray', lw=1.0, ls='--')

            resid_mask = np.isfinite(residual_series)
            color_positions = np.linspace(0.2, 0.9, num=len(model_curves)) if model_curves else []
            for color_frac, (freqs, curve) in zip(color_positions, model_curves):
                curve_interp = np.interp(x_valid, r_um, curve)
                valid_mask = resid_mask & np.isfinite(curve_interp)
                if np.count_nonzero(valid_mask) < (2 if allow_offset else 1):
                    continue
                amp, offset = self._fit_cosine_curve(
                    curve_interp[valid_mask],
                    residual_series[valid_mask],
                    allow_offset=allow_offset,
                )
                fitted_curve = amp * curve + offset
                label = f"Model {'+'.join(str(int(round(f))) for f in freqs)} Hz"
                ax_model.plot(
                    r_um,
                    fitted_curve,
                    linestyle='--',
                    color=base_cmap(color_frac),
                    label=label,
                )

        ax_model.legend(loc='upper left')

        # Bottom-most subplot handled once below; remove duplicated block

        # Bottom-most subplot: scatter of predicted (model at centers) vs actual residuals
        try:
            # Predicted proxy: model full evaluated at bin centers
            model_at_centers = np.interp(x_valid, r_um, total_r_full)
            # Residual series computed above if fit succeeded
            if exp_popt is not None:
                def _exp_fun(x, A, k, C):
                    return A * np.exp(-k * x) + C
                y_fit_valid = _exp_fun(x_valid, *exp_popt)
                residual_series = y_valid - y_fit_valid
                ax_scatter.scatter(model_at_centers, residual_series, s=20, alpha=0.7, color='teal')
                ax_scatter.set_xlabel('Predicted (model at centers)')
                ax_scatter.set_ylabel('Actual residual (data − exp fit)')
                finite = np.isfinite(model_at_centers) & np.isfinite(residual_series)
                r, p = np.nan, np.nan
                if np.any(finite):
                    x_sc = model_at_centers[finite]
                    y_sc = residual_series[finite]
                    r, p = pearsonr(x_sc, y_sc)
                    ax_scatter.set_title(f'Predicted vs Actual Residuals (r={r:.3f}, p={p:.2e})')
                    # Regression line
                    m, b = np.polyfit(x_sc, y_sc, 1)
                else:
                    m, b = 1.0, 0.0
                # Regression; make square axes
                xmin, xmax = ax_scatter.get_xlim()
                xs = np.linspace(xmin, xmax, 100)
                ax_scatter.plot(xs, m * xs + b, color='crimson', lw=1.5, label='Linear fit')
                ax_scatter.set_box_aspect(1)
                ax_scatter.legend(loc='best')
        except Exception:
            pass

        # End of rate plot
        return fig, ax1

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
        # Keep peaks silent; caller prints concise model summaries
        return True, gamma_hz_list, weights

    def _prepare_synergy_components(
        self,
        peak_min_hz: float,
        peak_max_hz: float,
        v_eph: Optional[float],
        v_ax: Optional[float],
        lambda_eph: Optional[float],
        min_distance: float = 50.0,
        max_distance: float = 3500.0,
        num_points: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float] | None:
        ok, gamma_hz_list, weights = self._compute_ifr_peaks_weights(peak_min_hz, peak_max_hz)
        if not ok:
            return None

        v_eph_val = self.v_eph if v_eph is None else float(v_eph)
        v_ax_val = self.v_ax if v_ax is None else float(v_ax)
        lambda_val = self.lambda_eph if lambda_eph is None else float(lambda_eph)
        v_eph_um_s = v_eph_val * 1e6
        v_ax_um_s = v_ax_val * 1e6

        r_um = np.linspace(min_distance, max_distance, num_points)
        total_curve = np.zeros_like(r_um)
        for hz, w in zip(gamma_hz_list, weights):
            total_curve += correlation_function(r_um, hz, v_eph_um_s, v_ax_um_s, lambda_val) * w

        return r_um, total_curve, gamma_hz_list, weights, v_eph_um_s, v_ax_um_s, lambda_val

    def correlation_curve(
        self,
        v_eph: Optional[float] = None,
        v_ax: Optional[float] = None,
        lambda_eph: Optional[float] = None,
        peak_min_hz: float = 30.0,
        peak_max_hz: float = 1000.0,
        min_distance: float = 50.0,
        max_distance: float = 3500.0,
        num_points: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray] | None:
        prepared = self._prepare_synergy_components(
            peak_min_hz=peak_min_hz,
            peak_max_hz=peak_max_hz,
            v_eph=v_eph,
            v_ax=v_ax,
            lambda_eph=lambda_eph,
            min_distance=min_distance,
            max_distance=max_distance,
            num_points=num_points,
        )
        if prepared is None:
            return None
        r_um, total_curve, *_ = prepared
        return r_um, total_curve

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
                # peak position debug removed
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
            bins = np.logspace(np.log10(max(min_distance, all_dists.min())), np.log10(max_distance), num=74)
        else:
            bins = np.linspace(max(min_distance, all_dists.min()), max_distance, num=74)

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
        show_exponential_fit: bool = True,
        allow_offset: bool = False,
    ):
        """Plot co-firing vs distance with synergy overlays and model diagnostics."""
        if result.binned.centers.size == 0:
            return None, None

        # Three rows: top=data+fit+shading, middle=residuals+model (no shading), bottom=scatter
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.35)
        ax1 = fig.add_subplot(gs[0])
        ax_model = fig.add_subplot(gs[1], sharex=ax1)
        ax_scatter = fig.add_subplot(gs[2])
        ax1.plot(result.binned.centers, result.binned.mean, color='blue', label='Mean Co-Firing Rate')
        ax1.fill_between(
            result.binned.centers,
            result.binned.mean - result.binned.stderr,
            result.binned.mean + result.binned.stderr,
            alpha=0.4,
            color='blue'
        )
        ax1.set_xlabel('Distance from Electrode ($\\mu m$)')
        ax1.set_ylabel('Mean Co-Firing Expectation (spikes per ref-spike)')
        ax1.set_title('Averaged Co-Firing Expectation vs. Distance')

        # Exponential fit: compute on binned means, draw in TOP subplot
        popt = None
        try:
            x_centers = np.asarray(result.binned.centers, dtype=float)
            y_mean = np.asarray(result.binned.mean, dtype=float)
            y_err = np.asarray(result.binned.stderr, dtype=float)
            valid = np.isfinite(x_centers) & np.isfinite(y_mean)
            x_valid = x_centers[valid]
            y_valid = y_mean[valid]
            if x_valid.size >= 3:
                def _exp_fun(x, A, k, C):
                    return A * np.exp(-k * x) + C
                # Initial guesses: A ~ y0 - yN, k ~ 1/Δx, C ~ yN
                A0 = float(max(1e-12, y_valid[0] - y_valid[-1]))
                k0 = 1.0 / max(1e-9, (x_valid.max() - x_valid.min()))
                C0 = float(y_valid[-1])
                popt, _ = curve_fit(_exp_fun, x_valid, y_valid, p0=[A0, k0, C0], maxfev=10000)
                if show_exponential_fit:
                    ax1.plot(x_valid, _exp_fun(x_valid, *popt), linestyle='-.', color='orange', label=f'Exp fit (A={popt[0]:.2e}, k={popt[1]:.2e})')
        except Exception as e:
            print(f"Exponential fit failed: {e}")
        ax1.legend(loc='upper left')
        # Background shading on TOP subplot only
        ymin_top, ymax_top = ax1.get_ylim()
        try:
            from ..helper_functions import truncate_colormap as _truncate
            base_cmap = _truncate(plt.get_cmap('viridis'), 0.4, 0.9)
        except Exception:
            base_cmap = plt.get_cmap('viridis')
        # compute parameters for shading
        ok_synergy = self._prepare_synergy_components(
            peak_min_hz=peak_min_hz,
            peak_max_hz=peak_max_hz,
            v_eph=v_eph,
            v_ax=v_ax,
            lambda_eph=lambda_eph,
        )
        if ok_synergy is None:
            return fig, ax1
        r_um, total_r_full, gamma_hz_list, weights, v_eph_um_s, v_ax_um_s, lambda_val = ok_synergy

        ymin_top, ymax_top = ax1.get_ylim()
        try:
            from ..helper_functions import truncate_colormap as _truncate
            base_cmap = _truncate(plt.get_cmap('viridis'), 0.4, 0.9)
        except Exception:
            base_cmap = plt.get_cmap('viridis')
        self._shade_first_peaks(ax1, r_um, gamma_hz_list, v_eph_um_s, v_ax_um_s, lambda_val, ymin_top, ymax_top, base_cmap)

        # ----- Build ephaptic–axonal model on the bottom subplot -----
        y = np.interp(r_um, result.binned.centers, result.binned.mean)

        model_curves: List[tuple[list[float], np.ndarray]] = []
        current_gamma = list(gamma_hz_list)
        weight_list = list(weights)
        while current_gamma:
            curve = np.zeros_like(r_um)
            for hz, w in zip(current_gamma, weight_list[: len(current_gamma)]):
                curve += correlation_function(r_um, hz, v_eph_um_s, v_ax_um_s, lambda_val) * w
            model_curves.append((current_gamma.copy(), curve))
            current_gamma = current_gamma[:-1]

        residuals_dict: Dict[str, np.ndarray] = {}
        log_l_map: Dict[str, float] = {}
        bic_map: Dict[str, float] = {}
        mi_map: Dict[str, float] = {}
        labels: List[str] = []

        for gammas, curve in model_curves:
            label = f"Model ({', '.join(f'{hz:.1f}' for hz in gammas)} Hz)"
            residuals = np.asarray(y - curve, dtype=float)
            residuals_dict[label] = residuals
            labels.append(label)
            n_params = max(len(gammas), 1)
            log_l_map[label] = log_likelihood(residuals, n_params)
            rss = float(np.sum(residuals ** 2))
            bic_map[label] = (len(y) * np.log(rss / len(y))) + (n_params * np.log(len(y)))
            try:
                mi = mutual_info_regression(curve.reshape(-1, 1), y)[0]
            except Exception:
                mi = np.nan
            mi_map[label] = mi

        if labels:
            print("\n[Cofiring] Synergy model summary:")
            for label in labels:
                print(
                    f"  {label}: logL={log_l_map[label]:.4f}, BIC={bic_map[label]:.4f}, MI={mi_map[label]:.5f}"
                )

        if len(labels) >= 2:
            full_label = labels[0]
            full_gamma_count = len(model_curves[0][0])
            log_full = log_l_map[full_label]
            print("[Cofiring] Likelihood-ratio tests vs full model:")
            for label, (gammas, _) in zip(labels[1:], model_curves[1:]):
                df = full_gamma_count - len(gammas)
                stat, p_value = likelihood_ratio_test(log_full, log_l_map[label], df)
                print(f"  {label}: LRT={stat:.4f}, p={p_value:.3e}")

        if len(residuals_dict) > 1:
            self.compare_model_fits(residuals_dict)

        # Middle subplot: residuals with fitted model curves
        ax_model.set_xlabel('Distance from Electrode ($\\mu m$)')
        ax_model.set_ylabel('Residual')
        residual_series = None
        if popt is not None and x_valid.size:
            def _exp_fun(x, A, k, C):
                return A * np.exp(-k * x) + C

            y_fit_valid = _exp_fun(x_valid, *popt)
            err_valid = y_err[valid]
            residual_series = y_valid - y_fit_valid
            ax_model.plot(x_valid, residual_series, color='purple', label='Residual (data − exp fit)')
            ax_model.fill_between(x_valid, residual_series - err_valid, residual_series + err_valid, alpha=0.2, color='purple')
            if residual_series.size:
                max_abs = float(np.nanmax(np.abs(residual_series)))
                if max_abs == 0:
                    max_abs = 1e-9
                ax_model.set_ylim(-1.05 * max_abs, 1.05 * max_abs)
            ax_model.axhline(0.0, color='gray', lw=1.0, ls='--')

            resid_mask = np.isfinite(residual_series)
            color_positions = np.linspace(0.2, 0.9, num=len(model_curves)) if model_curves else []
            for color_frac, (freqs, curve) in zip(color_positions, model_curves):
                curve_interp = np.interp(x_valid, r_um, curve)
                valid_mask = resid_mask & np.isfinite(curve_interp)
                if np.count_nonzero(valid_mask) < (2 if allow_offset else 1):
                    continue
                amp, offset = self._fit_cosine_curve(
                    curve_interp[valid_mask],
                    residual_series[valid_mask],
                    allow_offset=allow_offset,
                )
                fitted_curve = amp * curve + offset
                label = f"Model {'+'.join(str(int(round(f))) for f in freqs)} Hz"
                ax_model.plot(
                    r_um,
                    fitted_curve,
                    linestyle='--',
                    color=base_cmap(color_frac),
                    label=label,
                )

        ax_model.legend(loc='upper left')

        # Bottom-most subplot: scatter predicted (model at centers) vs actual residuals
        try:
            model_at_centers = np.interp(x_valid, r_um, total_r_full)
            if popt is not None:
                def _exp_fun(x, A, k, C):
                    return A * np.exp(-k * x) + C
                y_fit_valid = _exp_fun(x_valid, *popt)
                residual_series = y_valid - y_fit_valid
                ax_scatter.scatter(model_at_centers, residual_series, s=20, alpha=0.7, color='teal')
                ax_scatter.set_xlabel('Predicted (model at centers)')
                ax_scatter.set_ylabel('Actual residual (data − exp fit)')
                finite = np.isfinite(model_at_centers) & np.isfinite(residual_series)
                r, p = np.nan, np.nan
                if np.any(finite):
                    x_sc = model_at_centers[finite]
                    y_sc = residual_series[finite]
                    r, p = pearsonr(x_sc, y_sc)
                    ax_scatter.set_title(f'Predicted vs Actual Residuals (r={r:.3f}, p={p:.2e})')
                    m, b = np.polyfit(x_sc, y_sc, 1)
                else:
                    m, b = 1.0, 0.0
                xmin, xmax = ax_scatter.get_xlim()
                xs = np.linspace(xmin, xmax, 100)
                ax_scatter.plot(xs, m * xs + b, color='crimson', lw=1.5, label='Linear fit')
                ax_scatter.set_box_aspect(1)
                ax_scatter.legend(loc='best')
        except Exception:
            pass

        return fig, ax1

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
        dataset: RestingActivityDataset | None = None,
    ):
        """Compute pairwise distances between refs (per recording) with boundary weights."""
        ds_use = dataset or self.ds
        if refs_per_recording is not None:
            refs = refs_per_recording
        elif ds_use is self.ds_perm:
            refs = self._ensure_perm_refs()
        elif ds_use is self.ds:
            refs = self._ensure_refs(refs_per_recording)
        else:
            refs = self._compute_default_refs(dataset=ds_use)
        all_distances: List[float] = []
        all_weights: List[float] = []
        for rec, rlist in zip(ds_use.recordings, refs):
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

    @staticmethod
    def _fit_cosine_curve(curve: np.ndarray, target: np.ndarray, allow_offset: bool) -> Tuple[float, float]:
        curve = np.asarray(curve, dtype=float)
        target = np.asarray(target, dtype=float)
        valid = np.isfinite(curve) & np.isfinite(target)
        if np.count_nonzero(valid) < (2 if allow_offset else 1):
            if allow_offset and np.count_nonzero(valid) > 0:
                return 0.0, float(np.nanmean(target[valid]))
            return 0.0, 0.0

        x = curve[valid]
        y = target[valid]
        if allow_offset:
            A = np.column_stack([x, np.ones_like(x)])
            coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
            amp = float(coeffs[0])
            offset = float(coeffs[1])
            if amp < 0:
                amp = 0.0
                offset = float(np.nanmean(y))
            return amp, offset

        denom = float(np.dot(x, x))
        if denom <= 0:
            return 0.0, 0.0
        amp = float(np.dot(x, y) / denom)
        if amp < 0:
            amp = 0.0
        return amp, 0.0

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
        allow_offset: bool = True,
    ):
        """Plot original/permuted distance ratio with fitted synergy model curves."""
        fig, ax_ratio = plt.subplots(figsize=(10, 6))
        finite_size_correction = weights is not None and weights.size == distances.size
        weights_use = weights if finite_size_correction else None
        if np.isscalar(bins):
            edges = np.linspace(0.0, 3500.0, int(bins) + 1)
        else:
            edges = np.asarray(bins, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts, _ = np.histogram(distances, bins=edges, weights=weights_use)

        ratio = np.full_like(counts, np.nan, dtype=float)
        perm_counts = None
        if self.ds_perm is not None:
            perm_distances, perm_weights = self.distance_histogram(
                finite_size_correction=finite_size_correction,
                min_distance=edges[0],
                max_distance=edges[-1],
                bins=bins,
                dataset=self.ds_perm,
            )
            perm_weights_use = perm_weights if finite_size_correction else None
            perm_counts, _ = np.histogram(perm_distances, bins=edges, weights=perm_weights_use)
            ratio = np.divide(
                counts,
                perm_counts,
                out=np.full_like(counts, np.nan, dtype=float),
                where=(perm_counts if perm_counts is not None else 0) > 0,
            )

        finite_mask = np.isfinite(ratio)
        if np.any(finite_mask):
            ax_ratio.plot(centers[finite_mask], ratio[finite_mask], color='crimson', marker='o', label='Original / Permuted')
        ax_ratio.axhline(1.0, color='gray', linestyle='--', linewidth=1.0)
        ax_ratio.set_xlim(edges[0], edges[-1])
        ax_ratio.set_ylabel('Original / Permuted ratio')
        ax_ratio.set_xlabel('Distance between Most Active Electrodes ($\\mu m$)')
        finite_vals = ratio[np.isfinite(ratio)]
        if finite_vals.size:
            ymin_ratio = float(np.nanmin(finite_vals))
            ymax_ratio = float(np.nanmax(finite_vals))
            pad = 0.05 * (ymax_ratio - ymin_ratio if ymax_ratio > ymin_ratio else 1.0)
            ax_ratio.set_ylim(max(0.0, ymin_ratio - pad), ymax_ratio + pad)
        else:
            ax_ratio.set_ylim(0.0, 1.1)

        # Model curves aligned to ratio axis
        ok, gamma_hz_list, weights_pk = self._compute_ifr_peaks_weights(peak_min_hz, peak_max_hz)
        if not ok:
            return fig, ax_ratio

        v_eph_val = self.v_eph if v_eph is None else float(v_eph)
        v_ax_val = self.v_ax if v_ax is None else float(v_ax)
        lambda_eph = self.lambda_eph if lambda_eph is None else float(lambda_eph)
        v_eph_um_s = v_eph_val * 1e6
        v_ax_um_s = v_ax_val * 1e6
        r_um = np.linspace(edges[0], edges[-1], 1000)

        try:
            from ..helper_functions import truncate_colormap as _truncate
            base_cmap = _truncate(plt.get_cmap('viridis'), 0.4, 0.9)
        except Exception:
            base_cmap = plt.get_cmap('viridis')

        if finite_vals.size:
            # Build cumulative model and progressive reductions similar to other plots
            model_curves: List[tuple[list[float], np.ndarray]] = []
            current_gamma = list(gamma_hz_list)
            current_weights = list(weights_pk)
            while current_gamma:
                curve = np.zeros_like(r_um)
                for hz, w in zip(current_gamma, current_weights[: len(current_gamma)]):
                    curve += correlation_function(r_um, hz, v_eph_um_s, v_ax_um_s, lambda_eph) * w
                model_curves.append((current_gamma.copy(), curve))
                current_gamma = current_gamma[:-1]

            ratio_valid = np.isfinite(ratio)
            model_colors = np.linspace(0.2, 0.9, num=len(model_curves)) if model_curves else []

            residuals_dict: Dict[str, np.ndarray] = {}
            log_l_map: Dict[str, float] = {}
            bic_map: Dict[str, float] = {}
            labels: List[str] = []

            for color_frac, (freqs, curve) in zip(model_colors, model_curves):
                fitted_curve = None
                interp_curve = np.interp(centers, r_um, curve)
                valid_mask = ratio_valid & np.isfinite(interp_curve)
                if np.count_nonzero(valid_mask) < (2 if allow_offset else 1):
                    continue
                amp, offset = self._fit_cosine_curve(
                    interp_curve[valid_mask],
                    ratio[valid_mask],
                    allow_offset=allow_offset,
                )
                fitted_curve = amp * curve + offset
                label = f"Model {'+'.join(str(int(round(f))) for f in freqs)} Hz"
                ax_ratio.plot(
                    r_um,
                    fitted_curve,
                    linestyle='--',
                    color=base_cmap(color_frac),
                    label=label,
                )

                labels.append(label)
                fitted_at_centers = np.interp(centers, r_um, fitted_curve)
                residuals = ratio - fitted_at_centers
                finite = np.isfinite(residuals)
                residuals = residuals[finite]
                residuals_dict[label] = residuals
                n_params = max(len(freqs), 1) + (1 if allow_offset else 0)
                if residuals.size > 0:
                    log_l_map[label] = log_likelihood(residuals, n_params)
                    rss = float(np.sum(residuals ** 2))
                    n_eff = residuals.size
                    bic_map[label] = (n_eff * np.log(rss / max(n_eff, 1))) + (n_params * np.log(max(n_eff, 1)))
                else:
                    log_l_map[label] = float('nan')
                    bic_map[label] = float('nan')

            if labels:
                print("\n[Distance Ratio] Synergy model summary:")
                for label in labels:
                    print(
                        f"  {label}: logL={log_l_map[label]:.4f}, BIC={bic_map[label]:.4f}"
                    )

            if len(labels) >= 2 and all(np.isfinite(log_l_map[label]) for label in labels):
                full_label = labels[0]
                log_full = log_l_map[full_label]
                full_gamma_count = len(model_curves[0][0])
                print("[Distance Ratio] Likelihood-ratio tests vs full model:")
                for label, (freqs, _) in zip(labels[1:], model_curves[1:]):
                    df = full_gamma_count - len(freqs)
                    stat, p_value = likelihood_ratio_test(log_full, log_l_map[label], df)
                    print(f"  {label}: LRT={stat:.4f}, p={p_value:.3e}")

            if len(residuals_dict) > 1:
                self.compare_model_fits(residuals_dict)

        ax_ratio.set_title('Original / Permuted Distance Ratio with Synergy Model Fits')
        if ax_ratio.legend_ is not None:
            ax_ratio.legend_.remove()
        handles, labels = ax_ratio.get_legend_handles_labels()
        if handles:
            ax_ratio.legend(handles, labels, loc='upper left')
        return fig, ax_ratio

    # ----- Internal: refs management -----
    def _compute_default_refs(self, dataset: RestingActivityDataset | None = None) -> List[np.ndarray]:
        ds = dataset or self.ds
        cfg = self._selection_cfg or PrepConfig(mode='top', top_start=10, top_stop=210, top_use_recording_window=True, verbose=True)
        try:
            return ds.select_ref_electrodes(cfg)
        except Exception:
            out: List[np.ndarray] = []
            for rec in ds.recordings:
                import numpy as np
                out.append(np.unique(rec.spikes.get('electrode', np.array([], dtype=int))))
            return out

    def _ensure_refs(self, refs_per_recording: Optional[List[np.ndarray]]) -> List[np.ndarray]:
        if refs_per_recording is not None:
            return refs_per_recording
        if self._stored_refs is not None:
            return self._stored_refs
        self._stored_refs = self._compute_default_refs(dataset=self.ds)
        return self._stored_refs

    def _ensure_perm_refs(self) -> List[np.ndarray]:
        if self.ds_perm is None:
            return []
        if self._stored_refs_perm is not None:
            return self._stored_refs_perm
        self._stored_refs_perm = self._compute_default_refs(dataset=self.ds_perm)
        return self._stored_refs_perm
