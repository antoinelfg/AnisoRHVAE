#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from typing import Any, Iterable

import wandb


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _compute_score(
    c_step: float,
    v_step: float,
    c_dh: float,
    v_dh: float,
    c_acc: float,
    v_acc: float,
    target_accept: float,
) -> dict[str, float]:
    step_mean = 0.5 * (c_step + v_step)
    dh_std = 0.5 * (c_dh + v_dh)
    acc_mean = 0.5 * (c_acc + v_acc)
    accept_penalty = abs(acc_mean - target_accept)
    score = step_mean - 0.5 * dh_std - accept_penalty
    return {
        "diag/score_recomputed": float(score),
        "diag/step_len_mean_avg": float(step_mean),
        "diag/dh_std_avg": float(dh_std),
        "diag/accept_avg": float(acc_mean),
        "diag/accept_penalty": float(accept_penalty),
        "diag/score_target_accept_used": float(target_accept),
    }


def _iter_runs(api: wandb.Api, sweep_path: str | None, run_path: str | None) -> Iterable[Any]:
    if sweep_path:
        sweep = api.sweep(sweep_path)
        for run in sweep.runs:
            yield run
    elif run_path:
        yield api.run(run_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill diag/score and component summaries for RHMC W&B runs from existing"
            " centroid/void metrics."
        )
    )
    parser.add_argument("--sweep_path", type=str, default=None, help="entity/project/sweep_id")
    parser.add_argument("--run_path", type=str, default=None, help="entity/project/run_id")
    parser.add_argument(
        "--target_accept",
        type=float,
        default=None,
        help="Override score target acceptance. Defaults to run config score_target_accept or 0.8.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing diag/score summary value.",
    )
    args = parser.parse_args()

    if bool(args.sweep_path) == bool(args.run_path):
        raise SystemExit("Provide exactly one of --sweep_path or --run_path")

    api = wandb.Api()
    updated = 0
    skipped = 0

    for run in _iter_runs(api, args.sweep_path, args.run_path):
        s = run.summary

        c_step = _to_float(s.get("centroid/step_len_mean"))
        v_step = _to_float(s.get("void/step_len_mean"))
        c_dh = _to_float(s.get("centroid/proposal_dh_std"))
        v_dh = _to_float(s.get("void/proposal_dh_std"))
        c_acc = _to_float(s.get("centroid/accept"))
        v_acc = _to_float(s.get("void/accept"))

        if None in (c_step, v_step, c_dh, v_dh, c_acc, v_acc):
            skipped += 1
            print(f"[skip] {run.path[-1]} missing one or more centroid/void summary metrics")
            continue

        target_accept = (
            float(args.target_accept)
            if args.target_accept is not None
            else float(run.config.get("score_target_accept", 0.8))
        )

        payload = _compute_score(
            c_step=c_step,
            v_step=v_step,
            c_dh=c_dh,
            v_dh=v_dh,
            c_acc=c_acc,
            v_acc=v_acc,
            target_accept=target_accept,
        )

        existing = _to_float(s.get("diag/score"))
        if args.overwrite or existing is None:
            s["diag/score"] = payload["diag/score_recomputed"]

        for k, v in payload.items():
            s[k] = v

        s.update()
        updated += 1
        print(
            f"[ok] {run.path[-1]} diag/score={float(s.get('diag/score')):.6f} "
            f"(recomputed={payload['diag/score_recomputed']:.6f})"
        )

    print(f"done: updated={updated}, skipped={skipped}")


if __name__ == "__main__":
    main()
