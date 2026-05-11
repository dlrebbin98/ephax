from __future__ import annotations

import numpy as np
from scipy.stats import binned_statistic

from ..models import BinnedSeries


def bin_series(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> BinnedSeries:
    """Bin y by x into fixed bins and compute mean and stderr per bin."""
    bin_means, bin_edges, _ = binned_statistic(x, y, statistic=np.nanmean, bins=bins)
    bin_std_err, _, _ = binned_statistic(
        x,
        y,
        statistic=lambda v: np.nanstd(v) / np.sqrt(np.sum(np.isfinite(v))),
        bins=bins,
    )
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid = ~np.isnan(bin_means)
    return BinnedSeries(centers=centers[valid], mean=bin_means[valid], stderr=bin_std_err[valid])
