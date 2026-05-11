from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def log_likelihood(residuals, n_params):
    sigma2 = np.var(residuals)
    n = len(residuals)
    return -0.5 * n * np.log(2 * np.pi * sigma2) - np.sum(residuals**2) / (2 * sigma2)


def likelihood_ratio_test(logL_full, logL_reduced, df):
    """Compute likelihood-ratio statistic and p-value."""
    if df is None or df <= 0:
        return np.nan, np.nan
    lrt_stat = 2 * (logL_full - logL_reduced)
    p_value = chi2.sf(lrt_stat, df)
    return lrt_stat, p_value
