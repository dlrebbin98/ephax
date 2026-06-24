from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .burst import AlignedBurstEvents, HighResTraces


@dataclass
class DiscreteTEResult:
    delay_centers_ms: np.ndarray
    signed_dx_centers_um: np.ndarray
    conditional_probability: np.ndarray
    raw_te_bits: np.ndarray
    bias_corrected_te_bits: np.ndarray
    te_pvalue: np.ndarray
    effective_observations: np.ndarray
    trigger_summary: object
    observation_summary: object
    ridge_summary: object
    fit_summary: object
    score: float


def blocks_to_bins(blocks_ms, *, bin_ms: float) -> list[tuple[int, int]]:
    """Convert history blocks in milliseconds into inclusive bin offsets."""
    out = []
    for start_ms, stop_ms in blocks_ms:
        start_bin = max(1, int(round(float(start_ms) / float(bin_ms))))
        stop_bin = max(start_bin, int(round(float(stop_ms) / float(bin_ms))))
        out.append((start_bin, stop_bin))
    return out


def build_trigger_summary(
    *,
    scope: str,
    highres: HighResTraces,
    selected_electrodes,
    aligned: AlignedBurstEvents | None = None,
    coarse_epochs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build source-trigger rows for gamma-aligned or macro-burst TE."""
    rows = []
    selected_electrodes = np.asarray(selected_electrodes, dtype=int)
    if scope == "gamma":
        if aligned is None or aligned.valid_anchors is None or len(aligned.valid_anchors) == 0:
            return pd.DataFrame()
        valid_anchors = pd.DataFrame(aligned.valid_anchors)
        pre_bins = int(np.count_nonzero(aligned.relative_time_ms < 0))
        post_bins = int(np.count_nonzero(aligned.relative_time_ms > 0))
        for window_idx, row in enumerate(valid_anchors.itertuples(index=False)):
            anchor_idx = int(np.argmin(np.abs(highres.time_centers_s - float(row.anchor_time_s))))
            window_start_idx = anchor_idx - pre_bins
            window_stop_idx = anchor_idx + post_bins
            if window_start_idx < 0 or window_stop_idx >= highres.spike_presence.shape[1] or window_idx >= aligned.aligned_spikes.shape[0]:
                continue
            window_spikes = aligned.aligned_spikes[window_idx]
            active_sources = np.flatnonzero(window_spikes.any(axis=1))
            for src_idx in active_sources:
                first_rel_idx = int(np.flatnonzero(window_spikes[src_idx])[0])
                source_abs_idx = window_start_idx + first_rel_idx
                rows.append(_trigger_row(scope, row.coarse_event_idx, window_idx, src_idx, selected_electrodes, source_abs_idx, highres, window_start_idx, window_stop_idx))
    elif scope == "macro_burst":
        if coarse_epochs is None:
            return pd.DataFrame()
        for row in coarse_epochs.itertuples(index=False):
            window_start_idx = int(np.searchsorted(highres.time_centers_s, float(row.start_time_s), side="left"))
            window_stop_idx = int(np.searchsorted(highres.time_centers_s, float(row.end_time_s), side="right")) - 1
            if window_start_idx < 0 or window_stop_idx >= highres.spike_presence.shape[1] or window_stop_idx <= window_start_idx:
                continue
            epoch_spikes = highres.spike_presence[:, window_start_idx : window_stop_idx + 1]
            active_sources = np.flatnonzero(epoch_spikes.any(axis=1))
            for src_idx in active_sources:
                first_rel_idx = int(np.flatnonzero(epoch_spikes[src_idx])[0])
                source_abs_idx = window_start_idx + first_rel_idx
                rows.append(_trigger_row(scope, row.event_idx, -1, src_idx, selected_electrodes, source_abs_idx, highres, window_start_idx, window_stop_idx))
    else:
        raise ValueError("scope must be 'gamma' or 'macro_burst'")
    return pd.DataFrame(rows)


def build_observation_summary(
    trigger_summary: pd.DataFrame,
    spike_presence: np.ndarray,
    delay_offsets_bins: np.ndarray,
    target_history_bins,
    source_history_bins,
    *,
    control_exclusion_bins: int = 3,
    controls_per_trigger: int = 1,
    max_triggers: int | None = None,
    target_present_half_width_bins: int = 0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Pair each trigger with one or more control observations."""
    if trigger_summary.empty:
        return pd.DataFrame()
    rng = rng or np.random.default_rng(0)
    max_hist_back = max(max(stop for _, stop in target_history_bins), max(stop for _, stop in source_history_bins))
    max_delay = int(np.asarray(delay_offsets_bins).max())
    present_half_width = max(0, int(target_present_half_width_bins))
    rows = []
    trigger_summary_use = trigger_summary.copy()
    if max_triggers is not None and len(trigger_summary_use) > int(max_triggers):
        keep_idx = np.sort(rng.choice(len(trigger_summary_use), size=int(max_triggers), replace=False))
        trigger_summary_use = trigger_summary_use.iloc[keep_idx].reset_index(drop=True)

    for pair_id, row in enumerate(trigger_summary_use.itertuples(index=False)):
        center_idx = int(row.source_abs_idx)
        min_center = int(row.window_start_idx) + max_hist_back
        max_center = int(row.window_stop_idx) - (max_delay + present_half_width)
        if center_idx < min_center or center_idx > max_center:
            continue

        src_idx = int(row.source_index)
        source_trace = spike_presence[src_idx].astype(bool)
        forbidden = source_trace.copy()
        if int(control_exclusion_bins) > 0:
            kernel = np.ones(2 * int(control_exclusion_bins) + 1, dtype=int)
            forbidden = np.convolve(source_trace.astype(int), kernel, mode="same") > 0
        valid_positions = np.zeros_like(source_trace, dtype=bool)
        valid_positions[min_center : max_center + 1] = True
        candidate_control_idx = np.flatnonzero(valid_positions & ~forbidden)
        if candidate_control_idx.size == 0:
            continue
        rows.append(_observation_row(pair_id, "trigger", row, center_idx))
        n_controls = max(1, int(controls_per_trigger))
        replace = candidate_control_idx.size < n_controls
        for control_idx in rng.choice(candidate_control_idx, size=n_controls, replace=replace):
            rows.append(_observation_row(pair_id, "control", row, int(control_idx)))
    return pd.DataFrame(rows)


def build_signed_target_map(layout: dict | pd.DataFrame, selected_electrodes, *, signed_dx_bin_um: float):
    """Map each source electrode to target indices grouped by signed dx bins."""
    layout_df = pd.DataFrame(layout).drop_duplicates("electrode")
    selected_electrodes = np.asarray(selected_electrodes, dtype=int)
    layout_df = layout_df[layout_df["electrode"].isin(selected_electrodes)].set_index("electrode").loc[selected_electrodes].reset_index()
    x = layout_df["x"].to_numpy(dtype=float)
    signed_dx_matrix = x[None, :] - x[:, None]
    max_abs_dx = float(np.max(np.abs(signed_dx_matrix[~np.eye(len(x), dtype=bool)])))
    n_side_bins = int(np.ceil(max_abs_dx / float(signed_dx_bin_um)))
    centers = np.arange(-n_side_bins, n_side_bins + 1, dtype=float) * float(signed_dx_bin_um)
    edges = np.concatenate((centers - 0.5 * float(signed_dx_bin_um), [centers[-1] + 0.5 * float(signed_dx_bin_um)]))
    pair_bin_idx = np.digitize(signed_dx_matrix, edges, right=False) - 1
    pair_bin_idx = np.clip(pair_bin_idx, 0, len(centers) - 1)
    np.fill_diagonal(pair_bin_idx, -1)
    source_bin_targets = []
    for src_idx in range(len(selected_electrodes)):
        bin_map = {}
        for bin_idx in range(len(centers)):
            target_idx = np.flatnonzero(pair_bin_idx[src_idx] == bin_idx)
            target_idx = target_idx[target_idx != src_idx]
            if target_idx.size:
                bin_map[int(bin_idx)] = target_idx.astype(int)
        source_bin_targets.append(bin_map)
    return centers, source_bin_targets


def transfer_entropy_bits_from_counts(counts_yhist_xhist_y) -> float:
    counts = np.asarray(counts_yhist_xhist_y, dtype=float)
    total = float(counts.sum())
    if total <= 0:
        return np.nan
    p = counts / total
    p_yhist_xhist = p.sum(axis=2, keepdims=True)
    p_yhist_y = p.sum(axis=1, keepdims=True)
    p_yhist = p.sum(axis=(1, 2), keepdims=True)
    valid = (p > 0) & (p_yhist_xhist > 0) & (p_yhist_y > 0) & (p_yhist > 0)
    if not np.any(valid):
        return 0.0
    ratio = np.divide(p * p_yhist, p_yhist_xhist * p_yhist_y, out=np.ones_like(p), where=valid)
    return float(np.sum(p[valid] * np.log2(ratio[valid])))


def counts_from_te_states(yhist_state, xhist_state, y_present, n_yhist_states: int, n_xhist_states: int) -> np.ndarray:
    codes = ((yhist_state.astype(np.int64) * n_xhist_states) + xhist_state.astype(np.int64)) * 2 + y_present.astype(np.int64)
    return np.bincount(codes, minlength=n_yhist_states * n_xhist_states * 2).reshape(n_yhist_states, n_xhist_states, 2)


def run_discrete_te(
    *,
    highres: HighResTraces,
    layout: dict | pd.DataFrame,
    selected_electrodes,
    trigger_summary: pd.DataFrame,
    target_history_blocks_ms,
    source_history_blocks_ms,
    temporal_bin_ms: float = 1.0,
    signed_dx_bin_um: float = 200.0,
    delay_start_ms: float = 0.0,
    delay_stop_ms: float = 40.0,
    control_exclusion_ms: float = 3.0,
    controls_per_trigger: int = 1,
    n_surrogates: int = 100,
    min_observations: int = 800,
    min_effect_bits: float = 0.001,
    alpha: float = 0.05,
    delay_smooth_sigma_bins: float = 1.0,
    local_delay_tolerance_bins: int = 2,
    bootstrap_reps: int = 1000,
    random_seed: int = 0,
    max_triggers: int | None = None,
) -> DiscreteTEResult:
    """Run one discrete-time transfer entropy configuration."""
    highres_bin_ms = float(np.median(np.diff(highres.time_centers_s)) * 1000.0)
    temporal_bin_ms = float(temporal_bin_ms)
    target_present_half_width_bins = max(0, int(round(0.5 * temporal_bin_ms / highres_bin_ms)))
    delay_centers_ms = np.arange(float(delay_start_ms), float(delay_stop_ms) + 0.5 * temporal_bin_ms, temporal_bin_ms, dtype=float)
    delay_offsets_bins = np.rint(delay_centers_ms / highres_bin_ms).astype(int)
    target_bins = blocks_to_bins(target_history_blocks_ms, bin_ms=highres_bin_ms)
    source_bins = blocks_to_bins(source_history_blocks_ms, bin_ms=highres_bin_ms)
    rng = np.random.default_rng(int(random_seed))
    obs_summary = build_observation_summary(
        trigger_summary,
        highres.spike_presence,
        delay_offsets_bins,
        target_bins,
        source_bins,
        control_exclusion_bins=max(0, int(round(float(control_exclusion_ms) / highres_bin_ms))),
        controls_per_trigger=controls_per_trigger,
        max_triggers=max_triggers,
        target_present_half_width_bins=target_present_half_width_bins,
        rng=rng,
    )
    signed_dx_centers, source_bin_targets = build_signed_target_map(layout, selected_electrodes, signed_dx_bin_um=signed_dx_bin_um)
    arrays = _compute_te_arrays(
        highres.spike_presence,
        obs_summary,
        delay_offsets_bins,
        signed_dx_centers,
        source_bin_targets,
        target_bins,
        source_bins,
        n_surrogates=n_surrogates,
        random_seed=random_seed,
        target_present_half_width_bins=target_present_half_width_bins,
    )
    ridge = extract_first_stable_ridge(
        delay_centers_ms,
        signed_dx_centers,
        arrays["bias_corrected_te_bits"],
        arrays["te_pvalue"],
        arrays["effective_observations"],
        arrays["conditional_probability"],
        min_obs=min_observations,
        min_effect=min_effect_bits,
        alpha=alpha,
        smooth_sigma_bins=delay_smooth_sigma_bins,
        tolerance_bins=local_delay_tolerance_bins,
    )
    fit = fit_speed_with_bootstrap(ridge, bootstrap_reps=bootstrap_reps, rng_seed=int(random_seed) + 23)
    if not fit.empty and {"fit_speed_um_per_ms", "direction"}.issubset(fit.columns):
        n_fit = int(fit.loc[np.isfinite(fit["fit_speed_um_per_ms"]), "direction"].nunique())
    else:
        n_fit = 0
    mean_te = float(ridge["te_bits"].mean()) if not ridge.empty else 0.0
    score = 20.0 * n_fit + float(len(ridge)) + 1000.0 * mean_te
    return DiscreteTEResult(
        delay_centers_ms=delay_centers_ms,
        signed_dx_centers_um=signed_dx_centers,
        conditional_probability=arrays["conditional_probability"],
        raw_te_bits=arrays["raw_te_bits"],
        bias_corrected_te_bits=arrays["bias_corrected_te_bits"],
        te_pvalue=arrays["te_pvalue"],
        effective_observations=arrays["effective_observations"],
        trigger_summary=trigger_summary,
        observation_summary=obs_summary,
        ridge_summary=ridge,
        fit_summary=fit,
        score=float(score),
    )


def extract_first_stable_ridge(
    delay_centers_ms,
    signed_dx_centers_um,
    bias_corrected_te,
    pvalue,
    obs_counts,
    cond_prob,
    *,
    min_obs: int,
    min_effect: float,
    alpha: float,
    smooth_sigma_bins: float,
    tolerance_bins: int,
) -> pd.DataFrame:
    if np.asarray(bias_corrected_te).size == 0:
        return _empty_ridge()
    signed_dx_centers_um = np.asarray(signed_dx_centers_um, dtype=float)
    delay_centers_ms = np.asarray(delay_centers_ms, dtype=float)
    base = np.nan_to_num(bias_corrected_te, nan=0.0)
    if float(smooth_sigma_bins) > 0:
        smooth = gaussian_filter1d(base, sigma=float(smooth_sigma_bins), axis=0, mode="nearest")
    else:
        smooth = base
    valid_mask = (
        np.isfinite(bias_corrected_te)
        & np.isfinite(pvalue)
        & (obs_counts >= float(min_obs))
        & (bias_corrected_te >= float(min_effect))
        & (pvalue < float(alpha))
    )
    provisional = np.full(len(signed_dx_centers_um), -1, dtype=int)
    for bin_idx, dx_um in enumerate(signed_dx_centers_um):
        if dx_um == 0:
            continue
        valid = valid_mask[:, bin_idx]
        if not np.any(valid):
            continue
        col = smooth[:, bin_idx].copy()
        col[~valid] = -np.inf
        peaks, _ = find_peaks(col)
        peaks = peaks[np.isfinite(col[peaks])]
        if valid[0] and (peaks.size == 0 or col[0] >= np.nanmax(col[peaks])):
            provisional[bin_idx] = 0
        elif peaks.size:
            provisional[bin_idx] = int(np.min(peaks))
        else:
            provisional[bin_idx] = int(np.flatnonzero(valid)[0])

    keep_mask = np.zeros_like(provisional, dtype=bool)
    for bin_idx, peak_idx in enumerate(provisional):
        if peak_idx < 0 or signed_dx_centers_um[bin_idx] == 0:
            continue
        neighbors = []
        if bin_idx > 0 and np.sign(signed_dx_centers_um[bin_idx - 1]) == np.sign(signed_dx_centers_um[bin_idx]):
            neighbors.append(bin_idx - 1)
        if bin_idx < len(provisional) - 1 and np.sign(signed_dx_centers_um[bin_idx + 1]) == np.sign(signed_dx_centers_um[bin_idx]):
            neighbors.append(bin_idx + 1)
        if any(provisional[nb] >= 0 and abs(int(provisional[nb]) - int(peak_idx)) <= int(tolerance_bins) for nb in neighbors):
            keep_mask[bin_idx] = True

    rows = []
    for bin_idx in np.flatnonzero(keep_mask):
        peak_idx = int(provisional[bin_idx])
        rows.append(
            {
                "signed_dx_um": float(signed_dx_centers_um[bin_idx]),
                "abs_dx_um": float(abs(signed_dx_centers_um[bin_idx])),
                "peak_delay_ms": float(delay_centers_ms[peak_idx]),
                "te_bits": float(bias_corrected_te[peak_idx, bin_idx]),
                "pvalue": float(pvalue[peak_idx, bin_idx]),
                "effective_observations": float(obs_counts[peak_idx, bin_idx]),
                "conditional_probability": float(cond_prob[peak_idx, bin_idx]),
                "direction": "rightward" if signed_dx_centers_um[bin_idx] > 0 else "leftward",
            }
        )
    if not rows:
        return _empty_ridge()
    return pd.DataFrame(rows).sort_values(["direction", "signed_dx_um"]).reset_index(drop=True)


def fit_speed_with_bootstrap(speed_df: pd.DataFrame, *, bootstrap_reps: int, rng_seed: int) -> pd.DataFrame:
    columns = [
        "direction",
        "n_bins",
        "slope_ms_per_um",
        "intercept_ms",
        "fit_speed_um_per_ms",
        "bootstrap_speed_median_um_per_ms",
        "bootstrap_speed_ci_low_um_per_ms",
        "bootstrap_speed_ci_high_um_per_ms",
    ]
    rows = []
    rng = np.random.default_rng(int(rng_seed))
    min_slope = 1e-4
    if speed_df.empty or "direction" not in speed_df.columns:
        return pd.DataFrame(columns=columns)
    for direction, side_df in speed_df.groupby("direction"):
        if len(side_df) < 3:
            continue
        x = side_df["abs_dx_um"].to_numpy(dtype=float)
        y = side_df["peak_delay_ms"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        fit_speed = 1.0 / slope if np.isfinite(slope) and slope > min_slope else np.nan
        boot = []
        for _ in range(int(bootstrap_reps)):
            idx = rng.integers(0, len(side_df), size=len(side_df))
            if np.unique(x[idx]).size < 2:
                continue
            boot_slope, _ = np.polyfit(x[idx], y[idx], 1)
            if np.isfinite(boot_slope) and boot_slope > min_slope:
                boot.append(1.0 / boot_slope)
        if boot:
            boot = np.asarray(boot, dtype=float)
            ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
            median = float(np.median(boot))
        else:
            ci_low = ci_high = median = np.nan
        rows.append(
            {
                "direction": direction,
                "n_bins": int(len(side_df)),
                "slope_ms_per_um": float(slope),
                "intercept_ms": float(intercept),
                "fit_speed_um_per_ms": float(fit_speed),
                "bootstrap_speed_median_um_per_ms": float(median),
                "bootstrap_speed_ci_low_um_per_ms": float(ci_low),
                "bootstrap_speed_ci_high_um_per_ms": float(ci_high),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _compute_te_arrays(
    spike_presence,
    obs_summary,
    delay_offsets_bins,
    signed_dx_centers,
    source_bin_targets,
    target_history_bins,
    source_history_bins,
    *,
    n_surrogates: int,
    random_seed: int,
    target_present_half_width_bins: int = 0,
):
    n_delay = len(delay_offsets_bins)
    n_bins = len(signed_dx_centers)
    conditional_probability = np.full((n_delay, n_bins), np.nan, dtype=float)
    raw_te_bits = np.full((n_delay, n_bins), np.nan, dtype=float)
    bias_corrected_te_bits = np.full((n_delay, n_bins), np.nan, dtype=float)
    te_pvalue = np.full((n_delay, n_bins), np.nan, dtype=float)
    effective_observations = np.zeros((n_delay, n_bins), dtype=float)
    if obs_summary.empty:
        return {
            "conditional_probability": conditional_probability,
            "raw_te_bits": raw_te_bits,
            "bias_corrected_te_bits": bias_corrected_te_bits,
            "te_pvalue": te_pvalue,
            "effective_observations": effective_observations,
        }

    max_hist_back = max(max(stop for _, stop in target_history_bins), max(stop for _, stop in source_history_bins))
    max_delay = int(np.asarray(delay_offsets_bins).max())
    present_half_width = max(0, int(target_present_half_width_bins))
    n_obs = int(len(obs_summary))
    n_yhist = 2 ** len(target_history_bins)
    n_xhist = 2 ** len(source_history_bins)
    x_state = np.full((n_obs, n_delay, n_bins), 255, dtype=np.uint8)
    y_state = np.full((n_obs, n_delay, n_bins), 255, dtype=np.uint8)
    y_present = np.zeros((n_obs, n_delay, n_bins), dtype=bool)
    obs_valid = np.zeros((n_obs, n_bins), dtype=bool)
    is_trigger = obs_summary["kind"].to_numpy(dtype=str) == "trigger"

    for obs_idx, row in enumerate(obs_summary.itertuples(index=False)):
        src_idx = int(row.source_index)
        center_idx = int(row.center_idx)
        seg_start = center_idx - max_hist_back
        seg_stop = center_idx + max_delay + present_half_width + 1
        if seg_start < 0 or seg_stop > spike_presence.shape[1]:
            continue
        source_seg = spike_presence[src_idx, seg_start:seg_stop].astype(bool)
        for bin_idx, target_idx in source_bin_targets[src_idx].items():
            target_seg = spike_presence[target_idx, seg_start:seg_stop].any(axis=0)
            obs_valid[obs_idx, int(bin_idx)] = True
            for delay_i, offset_bins in enumerate(delay_offsets_bins):
                center_local = max_hist_back + int(offset_bins)
                y_present[obs_idx, delay_i, int(bin_idx)] = bool(
                    target_seg[center_local - present_half_width : center_local + present_half_width + 1].any()
                )
                yhist = 0
                for hist_idx, (start_bin, stop_bin) in enumerate(target_history_bins):
                    if target_seg[center_local - stop_bin : center_local - start_bin + 1].any():
                        yhist |= 1 << hist_idx
                xhist = 0
                for hist_idx, (start_bin, stop_bin) in enumerate(source_history_bins):
                    if source_seg[center_local - stop_bin : center_local - start_bin + 1].any():
                        xhist |= 1 << hist_idx
                y_state[obs_idx, delay_i, int(bin_idx)] = yhist
                x_state[obs_idx, delay_i, int(bin_idx)] = xhist

    rng = np.random.default_rng(int(random_seed) + 11)
    for bin_idx in range(n_bins):
        valid_obs = np.flatnonzero(obs_valid[:, bin_idx])
        if valid_obs.size == 0 or signed_dx_centers[bin_idx] == 0:
            continue
        trigger_idx = valid_obs[is_trigger[valid_obs]]
        for delay_i in range(n_delay):
            x_vals = x_state[valid_obs, delay_i, bin_idx]
            yh_vals = y_state[valid_obs, delay_i, bin_idx]
            y_vals = y_present[valid_obs, delay_i, bin_idx].astype(np.uint8)
            valid = (x_vals < 255) & (yh_vals < 255)
            if not np.any(valid):
                continue
            x_vals = x_vals[valid]
            yh_vals = yh_vals[valid]
            y_vals = y_vals[valid]
            effective_observations[delay_i, bin_idx] = float(len(x_vals))
            counts = counts_from_te_states(yh_vals, x_vals, y_vals, n_yhist, n_xhist)
            raw_te = transfer_entropy_bits_from_counts(counts)
            raw_te_bits[delay_i, bin_idx] = raw_te
            if trigger_idx.size:
                trig_x = x_state[trigger_idx, delay_i, bin_idx]
                trig_yh = y_state[trigger_idx, delay_i, bin_idx]
                trig_y = y_present[trigger_idx, delay_i, bin_idx].astype(float)
                trig_valid = (trig_x < 255) & (trig_yh < 255)
                if np.any(trig_valid):
                    conditional_probability[delay_i, bin_idx] = float(np.mean(trig_y[trig_valid]))
            if int(n_surrogates) > 0 and np.isfinite(raw_te):
                surrogate = np.empty(int(n_surrogates), dtype=float)
                for surr_idx in range(int(n_surrogates)):
                    perm_x = x_vals[rng.permutation(len(x_vals))]
                    surrogate[surr_idx] = transfer_entropy_bits_from_counts(counts_from_te_states(yh_vals, perm_x, y_vals, n_yhist, n_xhist))
                finite = surrogate[np.isfinite(surrogate)]
                if finite.size:
                    bias_corrected_te_bits[delay_i, bin_idx] = raw_te - float(finite.mean())
                    te_pvalue[delay_i, bin_idx] = (np.sum(finite >= raw_te) + 1.0) / (len(finite) + 1.0)
                else:
                    bias_corrected_te_bits[delay_i, bin_idx] = raw_te
            else:
                bias_corrected_te_bits[delay_i, bin_idx] = raw_te
    return {
        "conditional_probability": conditional_probability,
        "raw_te_bits": raw_te_bits,
        "bias_corrected_te_bits": bias_corrected_te_bits,
        "te_pvalue": te_pvalue,
        "effective_observations": effective_observations,
    }


def _trigger_row(scope, event_idx, window_idx, src_idx, selected_electrodes, source_abs_idx, highres, window_start_idx, window_stop_idx):
    return {
        "scope": scope,
        "scope_event_idx": int(event_idx),
        "window_idx": int(window_idx),
        "source_index": int(src_idx),
        "source_electrode": int(selected_electrodes[src_idx]),
        "source_abs_idx": int(source_abs_idx),
        "source_time_s": float(highres.time_centers_s[source_abs_idx]),
        "window_start_idx": int(window_start_idx),
        "window_stop_idx": int(window_stop_idx),
    }


def _observation_row(pair_id: int, kind: str, row, center_idx: int):
    return {
        "pair_id": int(pair_id),
        "kind": kind,
        "scope": row.scope,
        "scope_event_idx": int(row.scope_event_idx),
        "window_idx": int(row.window_idx),
        "source_index": int(row.source_index),
        "source_electrode": int(row.source_electrode),
        "center_idx": int(center_idx),
        "window_start_idx": int(row.window_start_idx),
        "window_stop_idx": int(row.window_stop_idx),
    }


def _empty_ridge():
    return pd.DataFrame(
        columns=[
            "signed_dx_um",
            "abs_dx_um",
            "peak_delay_ms",
            "te_bits",
            "pvalue",
            "effective_observations",
            "conditional_probability",
            "direction",
        ]
    )
