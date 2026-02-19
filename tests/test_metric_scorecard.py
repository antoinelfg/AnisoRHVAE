from __future__ import annotations

import pytest

from src.utils.metric_scorecard import (
    ABSOLUTE_RULES_BY_NAME,
    evaluate_absolute,
    evaluate_claim,
    evaluate_relative_gains,
    score_margin,
)


def test_absolute_threshold_statuses() -> None:
    metrics = {
        "acceptance_mean": 0.8,
        "dh_p95_abs": 0.7,
        "h_drift_slope_abs_mean": 0.005,
        "ess_norm_min": 180.0,
        "iact_median": 12.0,
        "coverage_local": 0.62,
        "rescue_rate": 0.80,
        "median_steps_to_manifold": 22.0,
        "plateau_fraction": 0.10,
    }
    status = evaluate_absolute(metrics)
    assert status["acceptance_mean"] == "green"
    assert status["dh_p95_abs"] == "green"
    assert status["ess_norm_min"] == "green"


def test_relative_gains_logic() -> None:
    baseline = {
        "ess_norm_min": 100.0,
        "coverage_local": 0.40,
        "rescue_rate": 0.55,
        "median_steps_to_manifold": 50.0,
    }
    aniso = {
        "ess_norm_min": 130.0,  # +30%
        "coverage_local": 0.51,  # +0.11
        "rescue_rate": 0.71,  # +0.16
        "median_steps_to_manifold": 38.0,  # -24%
    }
    rel = evaluate_relative_gains(baseline, aniso)
    assert rel["n_pass"] == 4
    assert rel["checks"]["ess_norm_min"] is True
    assert rel["checks"]["coverage_local"] is True


def test_claim_rule_requires_no_red_and_3_of_4_gains() -> None:
    strict_baseline = {
        "acceptance_mean": 0.8,
        "dh_p95_abs": 1.0,
        "h_drift_slope_abs_mean": 0.01,
        "ess_norm_min": 100.0,
        "coverage_local": 0.40,
        "rescue_rate": 0.55,
        "median_steps_to_manifold": 50.0,
    }
    strict_aniso = {
        "acceptance_mean": 0.82,
        "dh_p95_abs": 0.9,
        "h_drift_slope_abs_mean": 0.01,
        "ess_norm_min": 130.0,
        "coverage_local": 0.50,  # +0.10
        "rescue_rate": 0.70,  # +0.15
        "median_steps_to_manifold": 45.0,  # no 20% drop
    }
    matched_baseline = dict(strict_baseline)
    matched_aniso = {
        "acceptance_mean": 0.80,
        "dh_p95_abs": 0.8,
        "h_drift_slope_abs_mean": 0.009,
        "ess_norm_min": 125.0,
        "coverage_local": 0.54,
        "rescue_rate": 0.73,
        "median_steps_to_manifold": 39.0,
    }
    claim = evaluate_claim(strict_baseline, strict_aniso, matched_baseline, matched_aniso)
    assert claim["claim_validated"] is True


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("rescue_rate", 0.60, 0.0),
        ("rescue_rate", 0.75, 1.0),
        ("plateau_fraction", 0.35, 0.0),
        ("plateau_fraction", 0.20, 1.0),
        ("acceptance_mean", 0.50, 0.0),
        ("acceptance_mean", 0.60, 1.0),
        ("acceptance_mean", 0.90, 1.0),
        ("acceptance_mean", 0.95, 0.0),
    ],
)
def test_score_margin_hits_boundary_targets(name: str, value: float, expected: float) -> None:
    margin = score_margin(value, ABSOLUTE_RULES_BY_NAME[name])
    assert margin == pytest.approx(expected, abs=1e-9)


def test_score_margin_clips_out_of_range_values() -> None:
    high_margin = score_margin(2.0, ABSOLUTE_RULES_BY_NAME["rescue_rate"])
    low_margin = score_margin(-1.0, ABSOLUTE_RULES_BY_NAME["rescue_rate"])
    assert high_margin == 2.0
    assert low_margin == -1.0
