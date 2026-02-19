from __future__ import annotations

import numpy as np
import torch

from src.utils.rescue_metrics import (
    alignment_cosine,
    compute_border_tunnel_metrics,
    compute_hit_steps,
    compute_tangent_alignment_metric,
    median_steps_to_hit,
    plateau_fraction,
    rescue_rate_at_horizon,
    summarize_rescue_from_distances,
)


def test_hit_steps_and_summary() -> None:
    dists = np.array(
        [
            [2.0, 1.5, 0.9, 0.8],
            [1.8, 1.7, 1.6, 1.5],
            [3.0, 2.0, 1.0, 0.5],
        ],
        dtype=float,
    )
    hit = compute_hit_steps(dists, threshold=1.0)
    assert hit.tolist() == [2, -1, 2]
    assert rescue_rate_at_horizon(hit, horizon=3) == 2 / 3
    assert median_steps_to_hit(hit) == 2.0

    summary = summarize_rescue_from_distances(dists, threshold=1.0, horizon=3)
    assert abs(float(summary["rescue_rate"]) - (2 / 3)) < 1e-12
    assert float(summary["median_steps_to_hit"]) == 2.0


def test_alignment_cosine_bounds() -> None:
    a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 1.0]])
    cos = alignment_cosine(a, b)
    assert np.all(cos <= 1.0 + 1e-12)
    assert np.all(cos >= -1.0 - 1e-12)


def test_plateau_fraction_outside_mask() -> None:
    grad_norms = np.array([1e-4, 2e-3, 5e-4], dtype=float)
    outside = np.array([True, True, False], dtype=bool)
    frac = plateau_fraction(grad_norms, outside, threshold=1e-3)
    assert abs(frac - 0.5) < 1e-12


class _DummyMetricModel:
    use_attractor = True
    attractor_smoothness = "soft"
    attractor_metric = "euclidean"
    attractor_use_det = False

    def _compute_base_inverse_metric(self, z: torch.Tensor) -> torch.Tensor:
        b, d = z.shape
        out = torch.zeros(b, d, d, device=z.device, dtype=z.dtype)
        out[:, 0, 0] = 2.0
        out[:, 1, 1] = 1.0
        return out

    def G_inv(self, z: torch.Tensor) -> torch.Tensor:
        return self._compute_base_inverse_metric(z)

    def _compute_alpha(self, min_d: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((min_d - 1.0) * 5.0)

    def _compute_soft_attractor_weights(
        self,
        z: torch.Tensor,
        centroids: torch.Tensor,
        precisions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del precisions
        d = torch.cdist(z, centroids)
        return torch.softmax(-d, dim=1)

    def _get_attractor_precisions(self, *_args, **_kwargs):
        return None


def test_tangent_alignment_metric_is_finite() -> None:
    model = _DummyMetricModel()
    centroids = torch.tensor(
        [
            [-1.0, 0.0],
            [-0.5, 0.0],
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    out = compute_tangent_alignment_metric(
        model=model,
        centroids=centroids,
        r0=0.5,
        n_samples=256,
        k_neighbors=3,
        band_min=0.2,
        band_max=1.5,
        seed=7,
    )
    assert np.isfinite(out["tangent_alignment_mean"])
    assert out["tangent_alignment_count"] > 0


def test_border_tunnel_metrics_shapes() -> None:
    model = _DummyMetricModel()
    centroids = torch.tensor(
        [
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    points = torch.tensor(
        [
            [0.0, 0.0],
            [0.3, 0.2],
            [1.0, 0.0],
            [1.3, 0.4],
            [2.2, 0.0],
            [-2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    out = compute_border_tunnel_metrics(model=model, points=points, centroids=centroids, r0=1.0)
    assert np.isfinite(out["border_overshoot_index"])
    assert np.isfinite(out["tunnel_neff_transition_median"])
