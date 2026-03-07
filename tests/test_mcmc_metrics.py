from __future__ import annotations

import numpy as np

from src.utils.mcmc_metrics import (
    autocorrelation_1d,
    effective_sample_size_1d,
    integrated_autocorr_time,
)


def _ar1(phi: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=float)
    eps = rng.normal(size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def test_acf_properties_on_white_noise() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=4000)
    acf = autocorrelation_1d(x, max_lag=20)
    assert abs(acf[0] - 1.0) < 1e-12
    assert abs(acf[1]) < 0.1


def test_ess_white_noise_higher_than_ar1() -> None:
    rng = np.random.default_rng(7)
    white = rng.normal(size=3000)
    ar = _ar1(phi=0.9, n=3000, seed=7)
    ess_white = effective_sample_size_1d(white, max_lag=200)
    ess_ar = effective_sample_size_1d(ar, max_lag=200)
    assert ess_white > ess_ar
    assert ess_white > 1500
    assert ess_ar < 800


def test_iact_consistent_with_ess_relation() -> None:
    x = _ar1(phi=0.75, n=2500, seed=11)
    acf = autocorrelation_1d(x, max_lag=200)
    tau = integrated_autocorr_time(acf)
    ess = effective_sample_size_1d(x, max_lag=200)
    expected = len(x) / tau
    assert np.isfinite(tau)
    assert tau >= 1.0
    assert abs(ess - expected) / expected < 1e-6
