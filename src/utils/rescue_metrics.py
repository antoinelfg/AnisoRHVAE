from __future__ import annotations

from typing import Any

import numpy as np
import torch


def compute_hit_steps(min_distances: np.ndarray, threshold: float) -> np.ndarray:
    """
    First timestep where the trajectory reaches the manifold threshold.

    Returns -1 for never-hit trajectories.
    """
    d = np.asarray(min_distances, dtype=float)
    if d.ndim != 2:
        raise ValueError("min_distances must be [n_traj, n_steps]")
    hits = d <= float(threshold)
    hit_steps = np.full(d.shape[0], -1, dtype=int)
    for i in range(d.shape[0]):
        idx = np.flatnonzero(hits[i])
        if idx.size > 0:
            hit_steps[i] = int(idx[0])
    return hit_steps


def rescue_rate_at_horizon(hit_steps: np.ndarray, horizon: int) -> float:
    hs = np.asarray(hit_steps, dtype=int).reshape(-1)
    return float(np.mean((hs >= 0) & (hs <= int(horizon))))


def median_steps_to_hit(hit_steps: np.ndarray, default_if_none: float = np.nan) -> float:
    hs = np.asarray(hit_steps, dtype=int).reshape(-1)
    valid = hs[hs >= 0]
    if valid.size == 0:
        return float(default_if_none)
    return float(np.median(valid))


def summarize_rescue_from_distances(
    min_distances: np.ndarray,
    threshold: float,
    horizon: int,
) -> dict[str, float | list[int]]:
    hit_steps = compute_hit_steps(min_distances, threshold=threshold)
    return {
        "rescue_rate": rescue_rate_at_horizon(hit_steps, horizon=horizon),
        "median_steps_to_hit": median_steps_to_hit(hit_steps, default_if_none=float(horizon + 1)),
        "hit_steps": hit_steps.tolist(),
    }


def alignment_cosine(a: np.ndarray, b: np.ndarray, abs_value: bool = False) -> np.ndarray:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    if va.shape != vb.shape:
        raise ValueError("a and b must have the same shape")
    norm = np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1)
    cos = np.sum(va * vb, axis=1) / np.clip(norm, 1e-12, None)
    cos = np.clip(cos, -1.0, 1.0)
    if abs_value:
        cos = np.abs(cos)
    return cos


def plateau_fraction(
    grad_norms: np.ndarray,
    outside_mask: np.ndarray,
    threshold: float = 1e-3,
) -> float:
    g = np.asarray(grad_norms, dtype=float).reshape(-1)
    m = np.asarray(outside_mask, dtype=bool).reshape(-1)
    if g.size != m.size:
        raise ValueError("grad_norms and outside_mask must have the same length")
    if not np.any(m):
        return 0.0
    return float(np.mean(g[m] < float(threshold)))


def model_r0(model: Any, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> float:
    """
    Recover the manifold threshold radius used by alpha transition.
    """
    if device is None:
        device = next(model.parameters()).device
    probe = torch.zeros(1, device=device, dtype=dtype)
    if hasattr(model, "_compute_r0"):
        with torch.no_grad():
            r0 = model._compute_r0(probe)
        return float(r0.reshape(-1)[0].item())

    temperature = float(getattr(model, "temperature", torch.tensor(1.0)).item())
    void_weight_threshold = float(getattr(model, "void_weight_threshold", -1.0))
    if 0.0 < void_weight_threshold < 1.0:
        return float(temperature * np.sqrt(-np.log(void_weight_threshold)))
    void_threshold = float(getattr(model, "void_threshold", 1.5))
    return float(void_threshold * temperature)


def nearest_centroid_vectors(
    points: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns vectors from points to nearest centroid and associated distances.
    """
    d = torch.cdist(points, centroids)
    idx = torch.argmin(d, dim=1)
    nearest = centroids[idx]
    vec = nearest - points
    min_d = d[torch.arange(points.shape[0], device=points.device), idx]
    return vec, min_d


def gradient_logdet_inv(
    model: Any,
    points: torch.Tensor,
    batch_size: int = 1024,
) -> torch.Tensor:
    """
    Compute grad of 0.5 * log det G^{-1}(z) for each point.
    """
    grads = []
    for i in range(0, points.shape[0], batch_size):
        z = points[i : i + batch_size].clone().detach().requires_grad_(True)
        g_inv = model.G_inv(z)
        logdet_inv = torch.linalg.slogdet(g_inv).logabsdet
        obj = 0.5 * logdet_inv
        grad = torch.autograd.grad(obj.sum(), z, create_graph=False)[0]
        grads.append(grad.detach())
    return torch.cat(grads, dim=0)


def compute_plateau_alignment_metrics(
    model: Any,
    points: torch.Tensor,
    centroids: torch.Tensor,
    r0: float,
    grad_threshold: float = 1e-3,
) -> dict[str, float]:
    """
    Plateau fraction and rescue-gradient alignment outside manifold.
    """
    if points.numel() == 0:
        return {
            "plateau_fraction": 0.0,
            "alignment_mean": float("nan"),
            "rescue_directionality_mean": float("nan"),
        }

    with torch.no_grad():
        to_centroid, min_d = nearest_centroid_vectors(points, centroids)
        outside = min_d > float(r0)
    outside_np = outside.detach().cpu().numpy()

    grads = gradient_logdet_inv(model, points)
    grad_norm = torch.linalg.norm(grads, dim=1).detach().cpu().numpy()
    plateau = plateau_fraction(grad_norm, outside_np, threshold=grad_threshold)

    with torch.no_grad():
        cos = alignment_cosine(
            grads[:, : to_centroid.shape[1]].detach().cpu().numpy(),
            to_centroid.detach().cpu().numpy(),
            abs_value=False,
        )
    valid = outside_np & np.isfinite(cos)
    align = float(np.mean(cos[valid])) if np.any(valid) else float("nan")

    return {
        "plateau_fraction": float(plateau),
        "alignment_mean": align,
        "rescue_directionality_mean": align,
    }


def _local_tangent(neighbors: torch.Tensor) -> torch.Tensor:
    """
    Principal local direction (largest eigenvector of neighborhood covariance).

    neighbors: [B, K, D]
    returns: [B, D]
    """
    centered = neighbors - neighbors.mean(dim=1, keepdim=True)
    denom = max(int(neighbors.shape[1]) - 1, 1)
    cov = centered.transpose(1, 2) @ centered / float(denom)
    _, eigvecs = torch.linalg.eigh(cov)
    return eigvecs[:, :, -1]


def _sample_near_manifold_band(
    centroids: torch.Tensor,
    r0: float,
    n_samples: int,
    band_min: float,
    band_max: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points around centroids and keep those in a min-distance band.
    """
    device = centroids.device
    dtype = centroids.dtype
    latent_dim = centroids.shape[1]

    c_min = centroids.min(dim=0).values
    c_max = centroids.max(dim=0).values
    span = torch.clamp(c_max - c_min, min=1e-3)
    lo = c_min - 0.5 * span
    hi = c_max + 0.5 * span

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    draws = max(int(n_samples), 512)
    rand = torch.rand(draws, latent_dim, generator=generator, device=device, dtype=dtype)
    points = lo.unsqueeze(0) + rand * (hi - lo).unsqueeze(0)

    dists = torch.cdist(points, centroids)
    min_d = dists.min(dim=1).values
    band = (min_d >= float(band_min * r0)) & (min_d <= float(band_max * r0))

    if not torch.any(band):
        # Fallback: if no point lands in the requested shell, use closest ones to r0.
        idx = torch.argsort(torch.abs(min_d - float(r0)))[: min(draws, 256)]
        return points[idx], dists[idx]

    chosen = points[band]
    chosen_dists = dists[band]
    if chosen.shape[0] > n_samples:
        perm = torch.randperm(chosen.shape[0], generator=generator, device=device)[:n_samples]
        chosen = chosen[perm]
        chosen_dists = chosen_dists[perm]
    return chosen, chosen_dists


def compute_tangent_alignment_metric(
    model: Any,
    centroids: torch.Tensor,
    r0: float,
    n_samples: int = 1024,
    k_neighbors: int = 8,
    band_min: float = 0.6,
    band_max: float = 1.2,
    seed: int = 0,
) -> dict[str, float]:
    """
    A_tan = mean(|cos(e_cheap, tangent_local)|) on a near-manifold band.

    - e_cheap: direction of maximal eigenvalue of G_base^{-1}
    - tangent_local: principal direction from local centroid neighborhood
    """
    if centroids.numel() == 0:
        return {"tangent_alignment_mean": float("nan"), "tangent_alignment_count": 0.0}

    points, dists = _sample_near_manifold_band(
        centroids=centroids,
        r0=r0,
        n_samples=n_samples,
        band_min=band_min,
        band_max=band_max,
        seed=seed,
    )
    if points.numel() == 0:
        return {"tangent_alignment_mean": float("nan"), "tangent_alignment_count": 0.0}

    k = min(int(k_neighbors), int(centroids.shape[0]))
    if k <= 0:
        return {"tangent_alignment_mean": float("nan"), "tangent_alignment_count": 0.0}
    nn_idx = torch.topk(dists, k=k, largest=False).indices
    neighbors = centroids[nn_idx]  # [B, K, D]
    tangent = _local_tangent(neighbors)
    tangent = tangent / torch.clamp(torch.linalg.norm(tangent, dim=1, keepdim=True), min=1e-8)

    with torch.no_grad():
        if hasattr(model, "_compute_base_inverse_metric"):
            g_inv_base = model._compute_base_inverse_metric(points)
        else:
            g_inv_base = model.G_inv(points)
        _, eigvecs = torch.linalg.eigh(g_inv_base)
        cheap = eigvecs[:, :, -1]
        cheap = cheap / torch.clamp(torch.linalg.norm(cheap, dim=1, keepdim=True), min=1e-8)
        cos = torch.abs(torch.sum(cheap * tangent, dim=1))

    valid = torch.isfinite(cos)
    if not torch.any(valid):
        return {"tangent_alignment_mean": float("nan"), "tangent_alignment_count": 0.0}
    return {
        "tangent_alignment_mean": float(cos[valid].mean().item()),
        "tangent_alignment_count": float(valid.sum().item()),
    }


def compute_border_tunnel_metrics(
    model: Any,
    points: torch.Tensor,
    centroids: torch.Tensor,
    r0: float,
) -> dict[str, float]:
    """
    Border overshoot and tunnel-risk proxy metrics.

    border_overshoot_index:
      median(logdet_inv on transition) - max(median(manifold), median(far_void))

    tunnel proxy:
      N_eff = 1 / sum_k pi_k^2 for attractor weights in transition/far-void.
    """
    if points.numel() == 0:
        return {
            "border_overshoot_index": float("nan"),
            "logdet_manifold_median": float("nan"),
            "logdet_transition_median": float("nan"),
            "logdet_far_void_median": float("nan"),
            "tunnel_neff_transition_median": float("nan"),
            "tunnel_neff_far_void_median": float("nan"),
        }

    with torch.no_grad():
        d = torch.cdist(points, centroids)
        min_d = d.min(dim=1).values
        if hasattr(model, "_compute_alpha"):
            alpha = model._compute_alpha(min_d)
        else:
            # Fallback if alpha helper is unavailable.
            alpha = torch.sigmoid(min_d - float(r0))
        logdet_inv = torch.linalg.slogdet(model.G_inv(points)).logabsdet

    manifold_mask = alpha <= 0.1
    transition_mask = (alpha >= 0.2) & (alpha <= 0.8)
    far_void_mask = alpha >= 0.9

    def _median(vals: torch.Tensor, mask: torch.Tensor) -> float:
        use = vals[mask]
        if use.numel() == 0:
            return float("nan")
        return float(torch.median(use).item())

    med_manifold = _median(logdet_inv, manifold_mask)
    med_transition = _median(logdet_inv, transition_mask)
    med_far_void = _median(logdet_inv, far_void_mask)

    border = float("nan")
    if np.isfinite(med_transition) and (np.isfinite(med_manifold) or np.isfinite(med_far_void)):
        baseline = max(v for v in (med_manifold, med_far_void) if np.isfinite(v))
        border = float(med_transition - baseline)

    # Tunnel proxy via attractor concentration.
    neff_transition = float("nan")
    neff_far_void = float("nan")
    if bool(getattr(model, "use_attractor", False)):
        if str(getattr(model, "attractor_smoothness", "soft")) == "hard":
            neff = torch.ones(points.shape[0], device=points.device, dtype=points.dtype)
        else:
            precisions = None
            metric_kind = str(getattr(model, "attractor_metric", "euclidean")).lower()
            use_det = bool(getattr(model, "attractor_use_det", False))
            if metric_kind == "mahalanobis" or use_det:
                precisions = model._get_attractor_precisions(None, points.device)
            weights = model._compute_soft_attractor_weights(points, centroids, precisions)
            neff = 1.0 / torch.clamp(torch.sum(weights * weights, dim=1), min=1e-12)
        neff_transition = _median(neff, transition_mask)
        neff_far_void = _median(neff, far_void_mask)

    return {
        "border_overshoot_index": float(border),
        "logdet_manifold_median": float(med_manifold),
        "logdet_transition_median": float(med_transition),
        "logdet_far_void_median": float(med_far_void),
        "tunnel_neff_transition_median": float(neff_transition),
        "tunnel_neff_far_void_median": float(neff_far_void),
    }
