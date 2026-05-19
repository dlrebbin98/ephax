from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from .ephaptic import correlation_function


@dataclass(frozen=True)
class HDMEADifferentiationConfig:
    """Parameters for the HD-MEA differentiation/apoptosis simulation."""

    grid_columns: int = 220
    grid_rows: int = 120
    pitch_um: float = 17.5
    hz: float = 45.0
    v_eph_m_s: float = 0.1
    v_ax_m_s: float = 0.8
    lambda_eph_um: float = 1e4
    axonal_decay_um: float = 100.0
    ephaptic_correlation_gain: float | None = None
    subsample_size: int = 20
    random_seed: int = 0
    plating_loss_fraction: float = 0.50
    target_fraction: float = 0.10
    milestone_step_fraction: float = 0.10

    @property
    def initial_count(self) -> int:
        return int(self.grid_columns) * int(self.grid_rows)

    @property
    def plating_loss_count(self) -> int:
        return int(round(self.initial_count * float(self.plating_loss_fraction)))

    @property
    def plated_count(self) -> int:
        return self.initial_count - self.plating_loss_count

    @property
    def target_count(self) -> int:
        return int(round(self.plated_count * float(self.target_fraction)))

    @property
    def v_eph_um_s(self) -> float:
        return float(self.v_eph_m_s) * 1e6

    @property
    def v_ax_um_s(self) -> float:
        return float(self.v_ax_m_s) * 1e6

    @property
    def correlation_gain(self) -> float:
        return float(self.hz if self.ephaptic_correlation_gain is None else self.ephaptic_correlation_gain)

    @property
    def wavelength_um(self) -> float:
        return 1.0 / (float(self.hz) * abs((1.0 / self.v_eph_um_s) - (1.0 / self.v_ax_um_s)))

    def validate(self) -> None:
        if self.plated_count < self.target_count:
            raise ValueError("plating_loss_fraction leaves fewer neurons than target_fraction requires.")
        if self.subsample_size < 2:
            raise ValueError("subsample_size must be at least 2.")


def make_hd_mea_grid(config: HDMEADifferentiationConfig) -> pd.DataFrame:
    col_idx, row_idx = np.meshgrid(np.arange(config.grid_columns), np.arange(config.grid_rows))
    electrode = np.arange(config.initial_count, dtype=int)
    return pd.DataFrame(
        {
            "electrode": electrode,
            "col": col_idx.reshape(-1).astype(int),
            "row": row_idx.reshape(-1).astype(int),
            "x_um": col_idx.reshape(-1).astype(float) * float(config.pitch_um),
            "y_um": row_idx.reshape(-1).astype(float) * float(config.pitch_um),
        }
    )


def raw_correlation_wave(r_um, config: HDMEADifferentiationConfig):
    r_um = np.asarray(r_um, dtype=float)
    delta_t = r_um * ((1.0 / config.v_eph_um_s) - (1.0 / config.v_ax_um_s))
    return np.cos(2.0 * np.pi * float(config.hz) * delta_t) / float(config.hz)


def ephaptic_correlation_kernel(r_um, config: HDMEADifferentiationConfig):
    return config.correlation_gain * raw_correlation_wave(r_um, config) * np.exp(-np.asarray(r_um, dtype=float) / float(config.lambda_eph_um)) ** 2


def axonal_kernel(r_um, config: HDMEADifferentiationConfig):
    return np.exp(-np.asarray(r_um, dtype=float) / float(config.axonal_decay_um))


def build_kernel_curves(grid: pd.DataFrame, config: HDMEADifferentiationConfig, *, n_points: int = 2000) -> dict[str, np.ndarray | float]:
    coords_um = grid[["x_um", "y_um"]].to_numpy(dtype=float)
    max_distance_um = float(np.hypot(coords_um[:, 0].max(), coords_um[:, 1].max()))
    r_um = np.linspace(0.0, max_distance_um, int(n_points))
    ephaptic = ephaptic_correlation_kernel(r_um, config)
    expected = config.correlation_gain * correlation_function(
        r_um,
        config.hz,
        config.v_eph_um_s,
        config.v_ax_um_s,
        config.lambda_eph_um,
    )
    if not np.allclose(ephaptic, expected):
        raise AssertionError("ephaptic kernel no longer matches ephax.modeling.ephaptic.correlation_function.")
    axonal = axonal_kernel(r_um, config)
    return {
        "r_um": r_um,
        "axonal": axonal,
        "ephaptic": ephaptic,
        "ephaptic_axonal": axonal + ephaptic,
        "wavelength_um": config.wavelength_um,
    }


def format_fraction_label(frac: float) -> str:
    pct = 100.0 * float(frac)
    return f"{pct:.0f}%" if np.isclose(pct, round(pct)) else f"{pct:.1f}%"


def build_pruning_milestones(config: HDMEADifferentiationConfig) -> dict[str, int]:
    fractions = []
    current = 1.0 - float(config.milestone_step_fraction)
    while current > float(config.target_fraction) + 1e-9:
        fractions.append(round(current, 10))
        current -= float(config.milestone_step_fraction)
    fractions.append(float(config.target_fraction))

    milestones = {}
    for frac in fractions:
        label = format_fraction_label(frac)
        threshold = config.target_count if np.isclose(frac, config.target_fraction) else int(np.floor(config.plated_count * frac))
        milestones[label] = int(threshold)
    return milestones


def snapshot_active_nodes(grid: pd.DataFrame, active_mask) -> pd.DataFrame:
    return grid.loc[active_mask, ["electrode", "col", "row", "x_um", "y_um"]].copy()


def make_plating_mask(grid: pd.DataFrame, config: HDMEADifferentiationConfig, random_seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(random_seed))
    active = np.ones(len(grid), dtype=bool)
    if config.plating_loss_count > 0:
        eliminated_by_plating = rng.choice(np.arange(len(grid)), size=config.plating_loss_count, replace=False)
        active[eliminated_by_plating] = False
    return active


def interaction_scores(sample_coords, config: HDMEADifferentiationConfig, *, include_correlation: bool):
    distances_um = cdist(sample_coords, sample_coords)
    np.fill_diagonal(distances_um, np.nan)
    axonal_scores = np.nansum(axonal_kernel(distances_um, config), axis=1)
    if not include_correlation:
        return axonal_scores, axonal_scores, np.zeros_like(axonal_scores)
    corr_scores = np.nansum(ephaptic_correlation_kernel(distances_um, config), axis=1)
    return axonal_scores + corr_scores, axonal_scores, corr_scores


def run_apoptosis_simulation(
    grid: pd.DataFrame,
    config: HDMEADifferentiationConfig,
    model_name: str,
    *,
    include_correlation: bool,
    random_seed: int | None = None,
    milestones: dict[str, int] | None = None,
    show_progress: bool = True,
) -> dict[str, object]:
    config.validate()
    seed = config.random_seed if random_seed is None else int(random_seed)
    coords_um = grid[["x_um", "y_um"]].to_numpy(dtype=float)
    milestones = build_pruning_milestones(config) if milestones is None else dict(milestones)
    plating_mask = make_plating_mask(grid, config, seed)
    rng = np.random.default_rng(seed + 1)
    active = plating_mask.copy()
    history = []
    milestone_snapshots = {}
    round_idx = 0
    eliminated_total = 0

    for label, threshold in milestones.items():
        if int(active.sum()) <= threshold:
            milestone_snapshots[label] = snapshot_active_nodes(grid, active)

    progress_total = int(active.sum()) - config.target_count
    progress = tqdm(total=progress_total, desc=f"{model_name} apoptosis", unit="neurons", disable=not show_progress)
    try:
        while int(active.sum()) > config.target_count:
            round_idx += 1
            active_before = int(active.sum())
            active_indices = np.flatnonzero(active)
            sample_size = min(int(config.subsample_size), active_before)
            sample_indices = rng.choice(active_indices, size=sample_size, replace=False)
            scores, axonal_scores, corr_scores = interaction_scores(
                coords_um[sample_indices],
                config,
                include_correlation=include_correlation,
            )
            eliminate_count = min(sample_size // 2, active_before - config.target_count)
            order = np.lexsort((rng.random(sample_size), scores))
            active[sample_indices[order[:eliminate_count]]] = False

            active_after = int(active.sum())
            eliminated = active_before - active_after
            eliminated_total += eliminated
            progress.update(eliminated)

            for label, threshold in milestones.items():
                if label not in milestone_snapshots and active_after <= threshold:
                    milestone_snapshots[label] = snapshot_active_nodes(grid, active)

            history.append(
                {
                    "model": model_name,
                    "round": round_idx,
                    "plating_loss_fraction": float(config.plating_loss_fraction),
                    "plated_starting_count": config.plated_count,
                    "active_before": active_before,
                    "active_after": active_after,
                    "sampled": sample_size,
                    "eliminated": eliminated,
                    "score_mean": float(np.mean(scores)),
                    "score_median": float(np.median(scores)),
                    "score_max": float(np.max(scores)),
                    "axonal_score_mean": float(np.mean(axonal_scores)),
                    "correlation_score_mean": float(np.mean(corr_scores)),
                }
            )
    finally:
        progress.close()

    if int(active.sum()) != config.target_count:
        raise AssertionError("simulation ended at an unexpected active-node count.")
    if eliminated_total != progress_total:
        raise AssertionError("simulation eliminated an unexpected number of nodes.")
    if not set(milestones).issubset(milestone_snapshots):
        raise AssertionError("simulation did not capture all requested milestones.")

    return {
        "model": model_name,
        "history": pd.DataFrame(history),
        "plating_mask": plating_mask.copy(),
        "plated_initial": snapshot_active_nodes(grid, plating_mask),
        "active_mask": active.copy(),
        "final_active": snapshot_active_nodes(grid, active),
        "milestone_snapshots": milestone_snapshots,
    }


def build_differentiation_simulation(
    config: HDMEADifferentiationConfig | None = None,
    *,
    model_labels: dict[str, str] | None = None,
    show_progress: bool = True,
) -> dict[str, object]:
    config = HDMEADifferentiationConfig() if config is None else config
    config.validate()
    model_labels = {
        "axonal_only": "Axonal only",
        "ephaptic_axonal": "Ephaptic-axonal",
    } if model_labels is None else dict(model_labels)
    grid = make_hd_mea_grid(config)
    kernels = build_kernel_curves(grid, config)
    milestones = build_pruning_milestones(config)
    simulation_results = {
        "axonal_only": run_apoptosis_simulation(
            grid,
            config,
            model_labels["axonal_only"],
            include_correlation=False,
            milestones=milestones,
            show_progress=show_progress,
        ),
        "ephaptic_axonal": run_apoptosis_simulation(
            grid,
            config,
            model_labels["ephaptic_axonal"],
            include_correlation=True,
            milestones=milestones,
            show_progress=show_progress,
        ),
    }
    if not np.array_equal(
        simulation_results["axonal_only"]["plating_mask"],
        simulation_results["ephaptic_axonal"]["plating_mask"],
    ):
        raise AssertionError("model simulations do not share the same plating mask.")
    history_df = pd.concat([result["history"] for result in simulation_results.values()], ignore_index=True)
    return {
        "grid": grid,
        "kernels": kernels,
        "milestones": milestones,
        "ordered_labels": list(milestones),
        "simulation_results": simulation_results,
        "history_df": history_df,
        "parameters": {
            **asdict(config),
            "initial_count": config.initial_count,
            "plating_loss_count": config.plating_loss_count,
            "plated_count": config.plated_count,
            "target_count": config.target_count,
            "v_eph_um_s": config.v_eph_um_s,
            "v_ax_um_s": config.v_ax_um_s,
            "wavelength_um": config.wavelength_um,
        },
    }
