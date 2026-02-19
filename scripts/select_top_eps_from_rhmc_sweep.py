#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

import wandb


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _narrow_grid(center: float) -> list[float]:
    lo = max(0.001, center - 0.004)
    hi = min(0.050, center + 0.004)
    vals = [round(lo + 0.0005 * i, 4) for i in range(17)]
    vals = [min(max(v, 0.001), 0.050) for v in vals]
    # Keep unique while preserving order in case clipping created duplicates.
    out: list[float] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def _compute_score_from_summary(summary: Any, target_accept: float) -> float | None:
    c_step = _to_float(summary.get("centroid/step_len_mean"))
    v_step = _to_float(summary.get("void/step_len_mean"))
    c_dh = _to_float(summary.get("centroid/proposal_dh_std"))
    v_dh = _to_float(summary.get("void/proposal_dh_std"))
    c_acc = _to_float(summary.get("centroid/accept"))
    v_acc = _to_float(summary.get("void/accept"))
    if None in (c_step, v_step, c_dh, v_dh, c_acc, v_acc):
        return None
    step_mean = 0.5 * (c_step + v_step)
    dh_std = 0.5 * (c_dh + v_dh)
    acc = 0.5 * (c_acc + v_acc)
    return float(step_mean - 0.5 * dh_std - abs(acc - target_accept))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select top epsilon values from an RHMC W&B sweep using acceptance filtering "
            "and diag/score ranking."
        )
    )
    parser.add_argument(
        "--sweep_path",
        type=str,
        required=True,
        help="W&B sweep path: entity/project/sweep_id",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of eps values to keep after filtering/ranking.",
    )
    parser.add_argument(
        "--min_accept",
        type=float,
        default=0.70,
        help="Minimum centroid and void acceptance threshold.",
    )
    args = parser.parse_args()

    api = wandb.Api()
    sweep = api.sweep(args.sweep_path)

    rows: list[dict[str, Any]] = []
    for run in sweep.runs:
        score = _to_float(run.summary.get("diag/score"))
        if score is None:
            target_accept = _to_float(run.config.get("score_target_accept"))
            if target_accept is None:
                target_accept = 0.8
            score = _compute_score_from_summary(run.summary, target_accept)
        eps = _to_float(run.config.get("eps_lf"))
        acc_c = _to_float(run.summary.get("centroid/accept"))
        acc_v = _to_float(run.summary.get("void/accept"))
        if score is None or eps is None or acc_c is None or acc_v is None:
            continue
        rows.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "score": score,
                "eps": eps,
                "acc_c": acc_c,
                "acc_v": acc_v,
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    kept = [r for r in rows if r["acc_c"] >= args.min_accept and r["acc_v"] >= args.min_accept]
    top = kept[: max(1, int(args.top_k))]

    print(f"sweep: {args.sweep_path}")
    print(f"total_scored_runs: {len(rows)}")
    print(f"filtered_runs (acc >= {args.min_accept:.2f} both centroid/void): {len(kept)}")
    print()
    print("top runs:")
    for r in top:
        print(
            f"- run={r['run_id']} name={r['run_name']} eps={r['eps']:.6f} "
            f"score={r['score']:.6f} acc_c={r['acc_c']:.3f} acc_v={r['acc_v']:.3f}"
        )

    if top:
        best_eps = top[0]["eps"]
        narrow = _narrow_grid(best_eps)
        print()
        print(f"best_eps: {best_eps:.6f}")
        print("narrow_eps_values:")
        print(", ".join(f"{v:.4f}" for v in narrow))


if __name__ == "__main__":
    main()
