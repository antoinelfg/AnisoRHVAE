from __future__ import annotations

from typing import Any

import numpy as np


def _as_1d(x: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("input array must be non-empty")
    return arr


def autocorrelation_1d(
    x: np.ndarray | list[float],
    max_lag: int | None = None,
) -> np.ndarray:
    """Compute normalized autocorrelation for lags [0, max_lag]."""
    arr = _as_1d(x)
    n = arr.size
    if max_lag is None:
        max_lag = max(1, min(n - 1, 200))
    max_lag = int(max(1, min(max_lag, n - 1)))

    centered = arr - arr.mean()
    var = np.dot(centered, centered)
    out = np.zeros(max_lag + 1, dtype=float)
    out[0] = 1.0
    if var <= 1e-14:
        return out

    for lag in range(1, max_lag + 1):
        out[lag] = np.dot(centered[:-lag], centered[lag:]) / var
    return out


def integrated_autocorr_time(acf: np.ndarray | list[float]) -> float:
    """Estimate IACT using initial positive sequence truncation."""
    arr = _as_1d(acf)
    if arr[0] <= 0:
        return 1.0

    tau = 1.0
    for k in range(1, arr.size):
        if arr[k] <= 0:
            break
        tau += 2.0 * arr[k]
    return float(max(1.0, tau))


def effective_sample_size_1d(
    x: np.ndarray | list[float],
    max_lag: int | None = None,
) -> float:
    arr = _as_1d(x)
    acf = autocorrelation_1d(arr, max_lag=max_lag)
    tau = integrated_autocorr_time(acf)
    return float(arr.size / tau)


def ess_iact_per_dim(
    chains: np.ndarray,
    burn_in: int = 0,
    max_lag: int | None = None,
    normalize_to_1000: bool = True,
) -> dict[str, Any]:
    """
    Compute ESS/IACT per latent dimension.

    Accepts chains shaped [C, T, D] or [T, D].
    """
    arr = np.asarray(chains, dtype=float)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError("chains must have shape [C,T,D] or [T,D]")
    if burn_in < 0 or burn_in >= arr.shape[1]:
        raise ValueError("burn_in must be in [0, T-1]")

    post = arr[:, burn_in:, :]
    c, t, d = post.shape
    flat_n = c * t

    ess_vals: list[float] = []
    ess_norm_vals: list[float] = []
    iact_vals: list[float] = []
    acf_by_dim: list[np.ndarray] = []

    for dim in range(d):
        flat = post[:, :, dim].reshape(-1)
        acf = autocorrelation_1d(flat, max_lag=max_lag)
        tau = integrated_autocorr_time(acf)
        ess = flat.size / tau
        ess_vals.append(float(ess))
        iact_vals.append(float(tau))
        if normalize_to_1000:
            ess_norm_vals.append(float(ess * 1000.0 / max(1, flat_n)))
        else:
            ess_norm_vals.append(float(ess))
        acf_by_dim.append(acf)

    ess_arr = np.asarray(ess_vals, dtype=float)
    ess_norm_arr = np.asarray(ess_norm_vals, dtype=float)
    iact_arr = np.asarray(iact_vals, dtype=float)
    return {
        "ess_per_dim": ess_arr.tolist(),
        "ess_norm_per_dim": ess_norm_arr.tolist(),
        "iact_per_dim": iact_arr.tolist(),
        "ess_min": float(np.min(ess_arr)),
        "ess_norm_min": float(np.min(ess_norm_arr)),
        "ess_median": float(np.median(ess_arr)),
        "iact_median": float(np.median(iact_arr)),
        "acf_per_dim": [a.tolist() for a in acf_by_dim],
        "n_effective_input": int(flat_n),
    }


def linear_drift_slope(x: np.ndarray | list[float]) -> float:
    arr = _as_1d(x)
    if arr.size < 2:
        return 0.0
    t = np.arange(arr.size, dtype=float)
    slope, _ = np.polyfit(t, arr, deg=1)
    return float(slope)


def aggregate_hamiltonian_metrics(
    hamiltonians: list[np.ndarray],
    proposal_dh: list[np.ndarray],
) -> dict[str, float]:
    """
    Aggregate stability indicators across chains.
    """
    slopes = [abs(linear_drift_slope(h)) for h in hamiltonians if len(h) >= 2]
    dh_abs = []
    for d in proposal_dh:
        if d.size:
            dh_abs.append(np.abs(np.asarray(d, dtype=float)))
    if dh_abs:
        merged = np.concatenate(dh_abs)
        dh_p95 = float(np.percentile(merged, 95))
    else:
        dh_p95 = float("nan")
    return {
        "dh_p95_abs": dh_p95,
        "h_drift_slope_abs_mean": float(np.mean(slopes)) if slopes else float("nan"),
    }
