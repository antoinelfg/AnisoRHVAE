#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _extract_median(rank: dict[str, Any], metric: str) -> float | None:
    metrics = rank.get("metrics", {})
    item = metrics.get(metric, {}) if isinstance(metrics, dict) else {}
    value = item.get("median") if isinstance(item, dict) else None
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _fmt(value: Any, ndigits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "1" if value else "0"
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return str(value)


def _collect_row(summary_file: Path, protocol: str, model_key: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(summary_file.read_text())
    except Exception:
        return None

    ranking = payload.get("ranking", {}).get(protocol, {}).get(model_key, {})
    if not ranking:
        return None

    summary_block = payload.get("summary", {}).get(protocol, {}).get(model_key, {})
    sampler_cfg = summary_block.get("sampler_config", {}) if isinstance(summary_block, dict) else {}
    sweep_objectives = payload.get("sweep_objectives", {})

    path_parts = summary_file.parts
    eps_dir = None
    for p in path_parts:
        if p.startswith("eps_"):
            eps_dir = p
            break

    row = {
        "summary_file": str(summary_file),
        "run_dir": str(summary_file.parent),
        "eps_dir": eps_dir,
        "eps_lf": sampler_cfg.get("eps_lf"),
        "n_lf": sampler_cfg.get("n_lf"),
        "fp_steps": sampler_cfg.get("fp_steps"),
        "fp_damping": sampler_cfg.get("fp_damping"),
        "volume_power": sampler_cfg.get("volume_power"),
        "hard_reject": bool(ranking.get("hard_reject", True)),
        "primary_green_count": ranking.get("primary_green_count"),
        "iqr_noise_penalty": ranking.get("iqr_noise_penalty"),
        "objective_full_kpi_v1": sweep_objectives.get("objective_full_kpi_v1"),
        "objective_metric_core4_v1": sweep_objectives.get("objective_metric_core4_v1"),
        "acceptance_mean": _extract_median(ranking, "acceptance_mean"),
        "dh_p95_abs": _extract_median(ranking, "dh_p95_abs"),
        "h_drift_slope_abs_mean": _extract_median(ranking, "h_drift_slope_abs_mean"),
        "coverage_local": _extract_median(ranking, "coverage_local"),
        "rescue_rate": _extract_median(ranking, "rescue_rate"),
        "median_steps_to_manifold": _extract_median(ranking, "median_steps_to_manifold"),
        "plateau_fraction": _extract_median(ranking, "plateau_fraction"),
        "tangent_alignment_mean": _extract_median(ranking, "tangent_alignment_mean"),
        "rescue_directionality_mean": _extract_median(ranking, "rescue_directionality_mean"),
        "border_overshoot_index": _extract_median(ranking, "border_overshoot_index"),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize eps candidate metric_assessment_suite runs under a root directory."
    )
    parser.add_argument("--root", required=True, type=Path, help="Root directory containing eps_* run folders.")
    parser.add_argument("--protocol", default="matched", type=str)
    parser.add_argument("--model_key", default="aniso", type=str)
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--csv_out", type=Path, default=None)
    args = parser.parse_args()

    files = sorted(args.root.rglob("summary_by_protocol.json"))
    rows: list[dict[str, Any]] = []
    for f in files:
        row = _collect_row(f, protocol=args.protocol, model_key=args.model_key)
        if row is not None:
            rows.append(row)

    if not rows:
        print(f"No matching summary files found under {args.root} for {args.protocol}/{args.model_key}.")
        return

    def _safe_float(x: Any, default: float) -> float:
        try:
            v = float(x)
            if v != v:  # NaN
                return default
            return v
        except Exception:
            return default

    rows.sort(
        key=lambda r: (
            int(bool(r.get("hard_reject", True))),
            -_safe_float(r.get("objective_full_kpi_v1"), -1e9),
            -_safe_float(r.get("primary_green_count"), -1e9),
            _safe_float(r.get("iqr_noise_penalty"), 1e9),
        )
    )

    header = [
        "rank",
        "eps_lf",
        "objective_full_kpi_v1",
        "hard_reject",
        "primary_green_count",
        "iqr_noise_penalty",
        "acceptance_mean",
        "dh_p95_abs",
        "h_drift_slope_abs_mean",
        "coverage_local",
        "rescue_rate",
        "median_steps_to_manifold",
        "plateau_fraction",
        "tangent_alignment_mean",
        "rescue_directionality_mean",
        "border_overshoot_index",
        "run_dir",
    ]

    print("\nTop candidates:")
    print(" | ".join(header))
    for i, row in enumerate(rows[: args.top_k], start=1):
        vals = [
            str(i),
            _fmt(row.get("eps_lf"), 5),
            _fmt(row.get("objective_full_kpi_v1"), 5),
            str(bool(row.get("hard_reject", True))),
            _fmt(row.get("primary_green_count"), 0),
            _fmt(row.get("iqr_noise_penalty"), 5),
            _fmt(row.get("acceptance_mean"), 4),
            _fmt(row.get("dh_p95_abs"), 4),
            _fmt(row.get("h_drift_slope_abs_mean"), 4),
            _fmt(row.get("coverage_local"), 4),
            _fmt(row.get("rescue_rate"), 4),
            _fmt(row.get("median_steps_to_manifold"), 2),
            _fmt(row.get("plateau_fraction"), 4),
            _fmt(row.get("tangent_alignment_mean"), 4),
            _fmt(row.get("rescue_directionality_mean"), 4),
            _fmt(row.get("border_overshoot_index"), 4),
            row.get("run_dir", "-"),
        ]
        print(" | ".join(vals))

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
