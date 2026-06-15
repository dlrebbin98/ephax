import numpy as np

from ephax.modeling import (
    HDMEADifferentiationConfig,
    build_differentiation_simulation,
    build_kernel_curves,
    make_hd_mea_grid,
)
from ephax.modeling.ephaptic import correlation_function


def test_hd_mea_differentiation_shared_simulation_runs_on_small_grid():
    cfg = HDMEADifferentiationConfig(
        grid_columns=8,
        grid_rows=6,
        pitch_um=17.5,
        subsample_size=6,
        random_seed=3,
        plating_loss_fraction=0.25,
        target_fraction=0.50,
        milestone_step_fraction=0.25,
    )

    payload = build_differentiation_simulation(cfg, show_progress=False)

    assert len(payload["grid"]) == cfg.initial_count
    assert payload["parameters"]["target_count"] == cfg.target_count
    assert set(payload["simulation_results"]) == {"axonal_only", "ephaptic_axonal"}
    assert np.array_equal(
        payload["simulation_results"]["axonal_only"]["plating_mask"],
        payload["simulation_results"]["ephaptic_axonal"]["plating_mask"],
    )
    assert len(payload["simulation_results"]["axonal_only"]["final_active"]) == cfg.target_count
    assert len(payload["simulation_results"]["ephaptic_axonal"]["final_active"]) == cfg.target_count
    assert set(payload["milestones"]).issubset(payload["simulation_results"]["axonal_only"]["milestone_snapshots"])
    assert set(payload["milestones"]).issubset(payload["simulation_results"]["ephaptic_axonal"]["milestone_snapshots"])


def test_hd_mea_differentiation_kernels_match_ephaptic_reference():
    cfg = HDMEADifferentiationConfig(grid_columns=4, grid_rows=3)
    grid = make_hd_mea_grid(cfg)

    kernels = build_kernel_curves(grid, cfg, n_points=32)

    expected_ephaptic = cfg.correlation_gain * correlation_function(
        kernels["r_um"],
        cfg.hz,
        cfg.v_eph_um_s,
        cfg.v_ax_um_s,
        cfg.lambda_eph_um,
    )
    expected_near_field = cfg.near_field_relative_amplitude * np.exp(-kernels["r_um"] / cfg.near_field_decay_um)
    expected_axonal_component = np.exp(-kernels["r_um"] / cfg.axonal_decay_um)
    assert np.allclose(kernels["ephaptic"], expected_ephaptic)
    assert np.allclose(kernels["near_field"], expected_near_field)
    assert np.allclose(kernels["axonal_component"], expected_axonal_component)
    assert np.allclose(kernels["axonal"], kernels["near_field"] + kernels["axonal_component"])
    assert np.allclose(kernels["ephaptic_axonal"], kernels["near_field"] + kernels["ephaptic"])
