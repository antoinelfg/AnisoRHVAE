from __future__ import annotations

import torch

from src.utils.low_data_metrics import (
    compute_d_rmse,
    compute_geo_euc_ratio,
    compute_interp_smoothness,
    compute_prd,
)


def test_d_rmse_zero_on_identical_points() -> None:
    real = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    path = real.clone()
    assert compute_d_rmse(path, real) == 0.0


def test_geo_euc_ratio_linear_path_is_one() -> None:
    path = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    ratio = compute_geo_euc_ratio(path)
    assert abs(ratio - 1.0) < 1e-6


def test_interp_smoothness_linear_is_zero() -> None:
    # Sequence linear in pixel space.
    x0 = torch.zeros(1, 4, 4)
    x1 = torch.ones(1, 4, 4)
    x2 = 2 * torch.ones(1, 4, 4)
    seq = torch.stack([x0, x1, x2], dim=0)
    smooth = compute_interp_smoothness(seq)
    assert abs(smooth) < 1e-8


def test_prd_identical_clouds_high_scores() -> None:
    cloud = torch.randn(200, 8)
    out = compute_prd(cloud, cloud, k=5)
    assert out["precision"] > 0.8
    assert out["recall"] > 0.8
