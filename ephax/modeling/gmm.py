from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture


@dataclass
class GMMFit:
    means_hz: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    p_value: float | None = None


def fit_ifr_gmm(
    values: np.ndarray,
    log_scale: bool = True,
    n_components: int | None = None,
) -> GMMFit:
    """Fit a Gaussian mixture model to IFR values."""
    X = values.reshape(-1, 1)
    if n_components is None:
        bics = []
        rng = range(1, 9)
        for n in rng:
            gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=0)
            gmm.fit(X)
            bics.append(gmm.bic(X))
        n_components = list(rng)[int(np.argmin(bics))]

    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=0)
    gmm.fit(X)

    means = gmm.means_.flatten()
    std = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()
    means_hz = 10 ** means if log_scale else means

    single = GaussianMixture(n_components=1, covariance_type="full", random_state=0)
    single.fit(X)
    ll_gmm = gmm.score(X) * len(values)
    ll_single = single.score(X) * len(values)
    stat = 2 * (ll_gmm - ll_single)
    df = max(1, n_components - 1)
    p_value = float(1 - chi2.cdf(stat, df))

    return GMMFit(means_hz=np.asarray(means_hz), std=np.asarray(std), weights=np.asarray(weights), p_value=p_value)
