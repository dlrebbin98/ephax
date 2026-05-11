from __future__ import annotations

import numpy as np


def velocity_decay(x, v0=0.9, v_min=0.4, k=0.002):
    return v_min + (v0 - v_min) * np.exp(-k * x)


def correlation_function(r_um, hz, v_eph, v_ax, lambda_eph):
    """Compute the ephaptic-axonal correlation function."""
    r_um = np.asarray(r_um, dtype=float)
    if hz <= 0:
        return np.zeros_like(r_um)
    delta_t = r_um * ((1 / v_eph) - (1 / v_ax))
    return (np.cos(2 * np.pi * hz * delta_t) * np.exp(-r_um / lambda_eph) ** 2) / hz


def correlation_function_w_sum(r_um, hz_list, v_eph, v_ax, lambda_eph):
    r_um = np.asarray(r_um, dtype=float)
    total = np.zeros_like(r_um)
    for hz in hz_list:
        total += correlation_function(r_um, hz, v_eph, v_ax, lambda_eph)
    return total
