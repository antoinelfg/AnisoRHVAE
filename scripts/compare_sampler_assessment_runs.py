#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PRIMARY_METRICS = [
    "coverage_local",
    "rescue_rate",
    "median_steps_to_manifold",
    "plateau_fraction",
    "tangent_alignment_mean",
    "rescue_directionality_mean",
    "border_overshoot_index",
]
STABILITY_METRICS = [
    "acceptance_mean",
    "dh_p95_abs",
    "h_drift_slope_abs_mean",
]


def _resolve_summary(path: Path) -> Path:
    if path.is_file() and path.name == "summary_by_protocol.json":
        return path
    candidate = path / "summary_by_protocol.json"
    if candidate.exists():
        return candidate
    nested = sorted(path.rglob("summary_by_protocol.json"))
    if not nested:
        raise FileNotFoundError(f"No summary_by_protocol.json found under {path}")
    return nested[-1]


def _load(summary_file: Path, protocol: str, model_key: str) -> dict[str, Any]:
    obj = json.loads(summary_file.read_text())
    rank = obj.get("ranking", {}).get(protocol, {}).get(model_key, {})
    if not rank:
        raise KeyError(f"Missing ranking for protocol={protocol} model_key={model_key} in {summary_file}")

    def median(metric: str) -> float | None:
        item = rank.get("metrics", {}).get(metric, {})
        val = item.get("median") if isinstance(item, dict) else None
        try:
            return float(val) if val is not None else None
        except Exception:
            return None

    data = {
        "summary_file": str(summary_file),
        "hard_reject": bool(rank.get("hard_reject", True)),
        "primary_green_count": rank.get("primary_green_count"),
        "iqr_noise_penalty": rank.get("iqr_noise_penalty"),
        "objective_full_kpi_v1": obj.get("sweep_objectives", {}).get("objective_full_kpi_v1"),
        "statuses": rank.get("statuses", {}),
        "medians": {m: median(m) for m in (STABILITY_METRICS + PRIMARY_METRICS)},
    }
    return data


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _line(metric: str, a: dict[str, Any], b: dict[str, Any]) -> str:
    av = a["medians"].get(metric)
    bv = b["medians"].get(metric)
    as_ = a["statuses"].get(metric, "-")
    bs_ = b["statuses"].get(metric, "-")
    return f"{metric:28s} | {_fmt(av):>8s} ({as_:>6s}) | {_fmt(bv):>8s} ({bs_:>6s})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two metric_assessment_suite run summaries.")
    parser.add_argument("--run_a", required=True, type=Path)
    parser.add_argument("--run_b", required=True, type=Path)
    parser.add_argument("--label_a", default="A")
    parser.add_argument("--label_b", default="B")
    parser.add_argument("--protocol", default="matched")
    parser.add_argument("--model_key", default="aniso")
    args = parser.parse_args()

    summary_a = _resolve_summary(args.run_a)
    summary_b = _resolve_summary(args.run_b)
    a = _load(summary_a, args.protocol, args.model_key)
    b = _load(summary_b, args.protocol, args.model_key)

    print(f"{args.label_a}: {summary_a}")
    print(f"{args.label_b}: {summary_b}")
    print()
    print("Global ranking fields")
    print(f"- {args.label_a}: hard_reject={a['hard_reject']} objective_full_kpi_v1={_fmt(a['objective_full_kpi_v1'], 5)} primary_green_count={a['primary_green_count']} iqr_noise_penalty={_fmt(a['iqr_noise_penalty'], 5)}")
    print(f"- {args.label_b}: hard_reject={b['hard_reject']} objective_full_kpi_v1={_fmt(b['objective_full_kpi_v1'], 5)} primary_green_count={b['primary_green_count']} iqr_noise_penalty={_fmt(b['iqr_noise_penalty'], 5)}")
    print()

    print(f"{'metric':28s} | {args.label_a:>20s} | {args.label_b:>20s}")
    print("-" * 82)
    for m in STABILITY_METRICS:
        print(_line(m, a, b))
    print("-" * 82)
    for m in PRIMARY_METRICS:
        print(_line(m, a, b))


if __name__ == "__main__":
    main()
