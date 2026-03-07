from __future__ import annotations

import pytest

from scripts.wandb_metric_core4_trial import (
    PRIMARY7_ORDERED_KEYS,
    PRIMARY7_WEIGHTS,
    compute_primary7_weighted_objective,
)


def test_primary7_weights_sum_to_one() -> None:
    total = sum(PRIMARY7_WEIGHTS[k] for k in PRIMARY7_ORDERED_KEYS)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_primary7_weighted_objective_penalizes_nonfinite() -> None:
    metrics = {k: 0.0 for k in PRIMARY7_ORDERED_KEYS}
    metrics["coverage_local"] = float("nan")
    out = compute_primary7_weighted_objective(metrics)
    assert out["objective_primary7_weighted_v1"] == -5.0
    assert out["primary7_green_count"] == 0


def test_primary7_weighted_objective_positive_for_greenish_metrics() -> None:
    metrics = {
        "tangent_alignment_mean": 0.88,
        "rescue_directionality_mean": 0.68,
        "border_overshoot_index": -0.03,
        "plateau_fraction": 0.12,
        "rescue_rate": 0.80,
        "median_steps_to_manifold": 24.0,
        "coverage_local": 0.60,
    }
    out = compute_primary7_weighted_objective(metrics)
    assert out["objective_primary7_weighted_v1"] > 0.0
    assert out["primary7_green_count"] >= 5
