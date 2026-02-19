from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class ThresholdRule:
    name: str
    mode: str  # "range", "max", "min"
    green: tuple[float, float] | float
    yellow: tuple[float, float] | float


ABSOLUTE_RULES: tuple[ThresholdRule, ...] = (
    ThresholdRule("acceptance_mean", "range", (0.60, 0.90), (0.50, 0.95)),
    ThresholdRule("dh_p95_abs", "max", 1.0, 2.0),
    ThresholdRule("h_drift_slope_abs_mean", "max", 0.01, 0.03),
    ThresholdRule("ess_norm_min", "min", 150.0, 80.0),
    ThresholdRule("iact_median", "max", 20.0, 40.0),
    ThresholdRule("coverage_local", "min", 0.55, 0.40),
    ThresholdRule("rescue_rate", "min", 0.75, 0.60),
    ThresholdRule("median_steps_to_manifold", "max", 30.0, 50.0),
    ThresholdRule("plateau_fraction", "max", 0.20, 0.35),
    ThresholdRule("tangent_alignment_mean", "min", 0.85, 0.75),
    ThresholdRule("rescue_directionality_mean", "min", 0.60, 0.45),
    ThresholdRule("border_overshoot_index", "max", 0.0, 0.15),
)
ABSOLUTE_RULES_BY_NAME: dict[str, ThresholdRule] = {rule.name: rule for rule in ABSOLUTE_RULES}


def classify_value(value: float, rule: ThresholdRule) -> str:
    if value is None:
        return "red"
    if rule.mode == "range":
        lo_g, hi_g = rule.green  # type: ignore[misc]
        lo_y, hi_y = rule.yellow  # type: ignore[misc]
        if lo_g <= value <= hi_g:
            return "green"
        if lo_y <= value <= hi_y:
            return "yellow"
        return "red"

    if rule.mode == "max":
        g = float(rule.green)  # type: ignore[arg-type]
        y = float(rule.yellow)  # type: ignore[arg-type]
        if value <= g:
            return "green"
        if value <= y:
            return "yellow"
        return "red"

    if rule.mode == "min":
        g = float(rule.green)  # type: ignore[arg-type]
        y = float(rule.yellow)  # type: ignore[arg-type]
        if value >= g:
            return "green"
        if value >= y:
            return "yellow"
        return "red"

    raise ValueError(f"Unknown rule mode: {rule.mode}")


def score_margin(
    value: float,
    rule: ThresholdRule,
    clip_low: float = -1.0,
    clip_high: float = 2.0,
) -> float:
    """
    Normalized smooth margin score relative to yellow->green transition.

    - min mode: (x - yellow) / (green - yellow)
    - max mode: (yellow - x) / (yellow - green)
    - range mode: min(lower_side_margin, upper_side_margin)
    """
    if value is None:
        return float(clip_low)
    x = float(value)
    if not math.isfinite(x):
        return float(clip_low)

    if rule.mode == "min":
        g = float(rule.green)  # type: ignore[arg-type]
        y = float(rule.yellow)  # type: ignore[arg-type]
        denom = max(1e-12, g - y)
        raw = (x - y) / denom
    elif rule.mode == "max":
        g = float(rule.green)  # type: ignore[arg-type]
        y = float(rule.yellow)  # type: ignore[arg-type]
        denom = max(1e-12, y - g)
        raw = (y - x) / denom
    elif rule.mode == "range":
        lo_g, hi_g = rule.green  # type: ignore[misc]
        lo_y, hi_y = rule.yellow  # type: ignore[misc]
        lo_denom = max(1e-12, lo_g - lo_y)
        hi_denom = max(1e-12, hi_y - hi_g)
        lower = (x - lo_y) / lo_denom
        upper = (hi_y - x) / hi_denom
        raw = min(lower, upper)
    else:
        raise ValueError(f"Unknown rule mode: {rule.mode}")

    return float(min(max(raw, clip_low), clip_high))


def evaluate_absolute(metrics: dict[str, float]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rule in ABSOLUTE_RULES:
        value = float(metrics.get(rule.name, float("nan")))
        out[rule.name] = classify_value(value, rule)
    return out


def _safe_ratio(a: float, b: float) -> float:
    if b == 0:
        return float("inf") if a > 0 else 0.0
    return a / b


def evaluate_relative_gains(
    baseline: dict[str, float],
    aniso: dict[str, float],
) -> dict[str, Any]:
    """
    Primary gains:
    - ess_norm_min: +20%
    - coverage_local: +0.10 absolute
    - rescue_rate: +0.15 absolute
    - median_steps_to_manifold: -20%
    """
    ess_gain = _safe_ratio(aniso["ess_norm_min"], baseline["ess_norm_min"]) >= 1.20
    cov_gain = (aniso["coverage_local"] - baseline["coverage_local"]) >= 0.10
    rescue_gain = (aniso["rescue_rate"] - baseline["rescue_rate"]) >= 0.15
    steps_gain = _safe_ratio(aniso["median_steps_to_manifold"], baseline["median_steps_to_manifold"]) <= 0.80

    checks = {
        "ess_norm_min": bool(ess_gain),
        "coverage_local": bool(cov_gain),
        "rescue_rate": bool(rescue_gain),
        "median_steps_to_manifold": bool(steps_gain),
    }
    n_pass = int(sum(1 for v in checks.values() if v))
    return {"checks": checks, "n_pass": n_pass, "n_total": 4}


def has_red_stability(statuses: dict[str, str]) -> bool:
    stability_keys = ("acceptance_mean", "dh_p95_abs", "h_drift_slope_abs_mean")
    return any(statuses.get(k) == "red" for k in stability_keys)


def evaluate_claim(
    strict_baseline: dict[str, float],
    strict_aniso: dict[str, float],
    matched_baseline: dict[str, float],
    matched_aniso: dict[str, float],
) -> dict[str, Any]:
    strict_abs = evaluate_absolute(strict_aniso)
    matched_abs = evaluate_absolute(matched_aniso)
    strict_rel = evaluate_relative_gains(strict_baseline, strict_aniso)
    matched_rel = evaluate_relative_gains(matched_baseline, matched_aniso)

    strict_ok = (not has_red_stability(strict_abs)) and (strict_rel["n_pass"] >= 3)
    matched_ok = (not has_red_stability(matched_abs)) and (matched_rel["n_pass"] >= 3)
    claim_validated = bool(strict_ok or matched_ok)

    return {
        "claim_validated": claim_validated,
        "strict": {
            "absolute": strict_abs,
            "relative": strict_rel,
            "pass": strict_ok,
        },
        "matched": {
            "absolute": matched_abs,
            "relative": matched_rel,
            "pass": matched_ok,
        },
    }
