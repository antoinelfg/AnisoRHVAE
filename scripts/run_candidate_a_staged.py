#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], cwd: Path) -> str:
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")
    return f"{completed.stdout}\n{completed.stderr}"


def _list_subdirs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {p.name for p in path.iterdir() if p.is_dir()}


def _latest_subdir(path: Path) -> Path:
    candidates = [p for p in path.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No subdirectories found in {path}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_saved_path(log_text: str, prefix: str) -> Path | None:
    pattern = re.compile(re.escape(prefix) + r"\s*(.+)")
    for line in log_text.splitlines():
        match = pattern.search(line.strip())
        if match:
            return Path(match.group(1).strip())
    return None


def _find_created_subdir(base_dir: Path, before: set[str]) -> Path:
    after = _list_subdirs(base_dir)
    created = sorted(after - before)
    if created:
        return base_dir / created[-1]
    return _latest_subdir(base_dir)


def _build_override_args(overrides: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in overrides.items():
        if isinstance(value, bool):
            out.append(f"{key}={str(value).lower()}")
        else:
            out.append(f"{key}={value}")
    return out


def train_candidate(
    python_exec: str,
    config_path: Path,
    model_out_root: Path,
    overrides: dict[str, Any],
) -> Path:
    model_out_root.mkdir(parents=True, exist_ok=True)
    before = _list_subdirs(model_out_root)
    full_overrides = dict(overrides)
    full_overrides["output_dir"] = str(model_out_root)
    cmd = [
        python_exec,
        "scripts/run_with_config.py",
        "--config",
        str(config_path),
        *_build_override_args(full_overrides),
    ]
    logs = _run(cmd, cwd=ROOT_DIR)
    logged = _extract_saved_path(logs, "[RHVAE BASELINE] All outputs saved to:")
    if logged is not None and logged.exists():
        return logged
    return _find_created_subdir(model_out_root, before)


def assess_candidate(
    python_exec: str,
    baseline_model_path: Path,
    candidate_model_path: Path,
    assessment_out_root: Path,
    seeds: list[int],
    sampler_name: str,
    strict_volume_power: float,
    run_quality: bool,
) -> tuple[Path, dict[str, Any]]:
    assessment_out_root.mkdir(parents=True, exist_ok=True)
    before = _list_subdirs(assessment_out_root)
    cmd = [
        python_exec,
        "scripts/metric_assessment_suite.py",
        "--baseline_model_path",
        str(baseline_model_path),
        "--ours_model_path",
        str(candidate_model_path),
        "--profile",
        "smoke",
        "--protocols",
        "strict",
        "--sampling_seeds",
        *[str(s) for s in seeds],
        "--sampler_name",
        sampler_name,
        "--strict_volume_power",
        str(strict_volume_power),
        "--skip_fid",
        "--output_dir",
        str(assessment_out_root),
    ]
    if run_quality:
        cmd.append("--run_quality")
    logs = _run(cmd, cwd=ROOT_DIR)
    logged = _extract_saved_path(logs, "Saved assessment outputs to:")
    if logged is not None and logged.exists():
        assessment_dir = logged
    else:
        assessment_dir = _find_created_subdir(assessment_out_root, before)

    ranking_path = assessment_dir / "ranking_summary.json"
    if not ranking_path.exists():
        raise FileNotFoundError(f"Expected ranking summary at {ranking_path}")
    payload = json.loads(ranking_path.read_text())
    strict_aniso = payload.get("strict", {}).get("aniso", {})
    if not strict_aniso:
        raise ValueError(f"Missing strict/aniso ranking block in {ranking_path}")
    return assessment_dir, strict_aniso


def _score_tuple(rank: dict[str, Any], candidate_id: str) -> tuple[Any, ...]:
    def _neg(value: Any) -> float:
        try:
            val = float(value)
        except Exception:
            return float("inf")
        if not math.isfinite(val):
            return float("inf")
        return -val

    def _pos(value: Any, default: float = float("inf")) -> float:
        try:
            val = float(value)
        except Exception:
            return default
        if not math.isfinite(val):
            return default
        return val

    hard_reject = 1 if bool(rank.get("hard_reject", True)) else 0
    return (
        hard_reject,
        -int(rank.get("primary_green_count", 0)),
        _neg(rank.get("tiebreak_tangent_alignment_mean")),
        _neg(rank.get("tiebreak_rescue_rate")),
        _neg(rank.get("tiebreak_coverage_local")),
        _pos(rank.get("iqr_noise_penalty")),
        candidate_id,
    )


def _pick_winner(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No candidates to rank.")
    ranked = sorted(candidates, key=lambda row: _score_tuple(row["ranking"], row["candidate_id"]))
    return ranked[0]


def _candidate_record(
    candidate_id: str,
    stage: str,
    overrides: dict[str, Any],
    model_dir: Path,
    assessment_dir: Path,
    ranking: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "overrides": overrides,
        "model_dir": str(model_dir),
        "assessment_dir": str(assessment_dir),
        "ranking": ranking,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage Candidate A with one-knob sweep and 3-seed smoke ranking.")
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/run_geometry_default_soft_no_tunnels_v2.yaml",
    )
    parser.add_argument(
        "--baseline_model_path",
        type=str,
        default="outputs/pythae_rhvae_baseline/2026-02-09_15-04-06",
    )
    parser.add_argument(
        "--atom_power_values",
        nargs="+",
        type=float,
        default=[1.05, 1.10, 1.15, 1.20],
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 29, 47])
    parser.add_argument("--sampler_name", type=str, default="volume")
    parser.add_argument("--strict_volume_power", type=float, default=1.0)
    parser.add_argument("--run_quality", dest="run_quality", action="store_true")
    parser.add_argument("--skip_quality", dest="run_quality", action="store_false")
    parser.set_defaults(run_quality=True)
    parser.add_argument("--python_exec", type=str, default=sys.executable)
    parser.add_argument("--runs_root", type=str, default="outputs/candidate_a_staged")
    parser.add_argument("--assessment_root", type=str, default="results/metric_assessment_smoke_rank")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_assessment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 3:
        raise ValueError("Use at least 3 seeds for final ranking (recommended: 13 29 47).")

    base_config = Path(args.base_config)
    baseline_model = Path(args.baseline_model_path)
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")
    if not baseline_model.exists():
        raise FileNotFoundError(f"Baseline model path not found: {baseline_model}")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = Path(args.runs_root) / f"candidate_a_staged_{stamp}"
    assess_root = Path(args.assessment_root) / f"candidate_a_staged_{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    assess_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "created_at": datetime.datetime.now().isoformat(),
        "base_config": str(base_config),
        "baseline_model_path": str(baseline_model),
        "epochs": int(args.epochs),
        "seeds": [int(s) for s in args.seeds],
        "stages": {},
    }

    # Stage A0 (reference only, no mutation)
    results["stages"]["A0"] = {
        "description": "Base config unchanged",
        "config": str(base_config),
    }

    # Stage A1: one-knob atom_power sweep
    a1_candidates: list[dict[str, Any]] = []
    for atom_power in args.atom_power_values:
        atom_label = f"{atom_power:.2f}".replace(".", "p")
        candidate_id = f"A1_atom_power_{atom_label}"
        overrides = {
            "epochs": int(args.epochs),
            "atom_power": float(atom_power),
        }
        model_parent = run_root / "A1" / candidate_id
        assess_parent = assess_root / "A1" / candidate_id

        if args.skip_training:
            model_dir = _latest_subdir(model_parent)
        else:
            model_dir = train_candidate(
                python_exec=args.python_exec,
                config_path=base_config,
                model_out_root=model_parent,
                overrides=overrides,
            )

        if args.skip_assessment:
            assessment_dir = _latest_subdir(assess_parent)
            ranking_payload = json.loads((assessment_dir / "ranking_summary.json").read_text())
            ranking = ranking_payload["strict"]["aniso"]
        else:
            assessment_dir, ranking = assess_candidate(
                python_exec=args.python_exec,
                baseline_model_path=baseline_model,
                candidate_model_path=model_dir,
                assessment_out_root=assess_parent,
                seeds=[int(s) for s in args.seeds],
                sampler_name=args.sampler_name,
                strict_volume_power=float(args.strict_volume_power),
                run_quality=bool(args.run_quality),
            )
        a1_candidates.append(
            _candidate_record(
                candidate_id=candidate_id,
                stage="A1",
                overrides=overrides,
                model_dir=model_dir,
                assessment_dir=assessment_dir,
                ranking=ranking,
            )
        )

    a1_winner = _pick_winner(a1_candidates)
    results["stages"]["A1"] = {
        "candidates": a1_candidates,
        "winner": a1_winner,
    }

    # Stage A2: attractor hardening only on A1 winner.
    a2_overrides = dict(a1_winner["overrides"])
    a2_overrides.update(
        attractor_smoothness="hard",
        attractor_k_nearest=1,
        attractor_gamma=8.0,
        attractor_bias_energy=25.0,
    )
    a2_id = f"A2_from_{a1_winner['candidate_id']}"
    a2_model_parent = run_root / "A2" / a2_id
    a2_assess_parent = assess_root / "A2" / a2_id
    if args.skip_training:
        a2_model_dir = _latest_subdir(a2_model_parent)
    else:
        a2_model_dir = train_candidate(
            python_exec=args.python_exec,
            config_path=base_config,
            model_out_root=a2_model_parent,
            overrides=a2_overrides,
        )
    if args.skip_assessment:
        a2_assessment_dir = _latest_subdir(a2_assess_parent)
        a2_rank_payload = json.loads((a2_assessment_dir / "ranking_summary.json").read_text())
        a2_ranking = a2_rank_payload["strict"]["aniso"]
    else:
        a2_assessment_dir, a2_ranking = assess_candidate(
            python_exec=args.python_exec,
            baseline_model_path=baseline_model,
            candidate_model_path=a2_model_dir,
            assessment_out_root=a2_assess_parent,
            seeds=[int(s) for s in args.seeds],
            sampler_name=args.sampler_name,
            strict_volume_power=float(args.strict_volume_power),
            run_quality=bool(args.run_quality),
        )
    a2_record = _candidate_record(
        candidate_id=a2_id,
        stage="A2",
        overrides=a2_overrides,
        model_dir=a2_model_dir,
        assessment_dir=a2_assessment_dir,
        ranking=a2_ranking,
    )
    results["stages"]["A2"] = {"candidate": a2_record}

    # Stage A3: blend/void-shape only on top of A2.
    a3_overrides = dict(a2_overrides)
    a3_overrides.update(
        transition_steepness=6.0,
        radial_stretch=10.0,
        void_decay_scale=10.0,
        void_decay_power=1.2,
    )
    a3_id = f"A3_from_{a2_id}"
    a3_model_parent = run_root / "A3" / a3_id
    a3_assess_parent = assess_root / "A3" / a3_id
    if args.skip_training:
        a3_model_dir = _latest_subdir(a3_model_parent)
    else:
        a3_model_dir = train_candidate(
            python_exec=args.python_exec,
            config_path=base_config,
            model_out_root=a3_model_parent,
            overrides=a3_overrides,
        )
    if args.skip_assessment:
        a3_assessment_dir = _latest_subdir(a3_assess_parent)
        a3_rank_payload = json.loads((a3_assessment_dir / "ranking_summary.json").read_text())
        a3_ranking = a3_rank_payload["strict"]["aniso"]
    else:
        a3_assessment_dir, a3_ranking = assess_candidate(
            python_exec=args.python_exec,
            baseline_model_path=baseline_model,
            candidate_model_path=a3_model_dir,
            assessment_out_root=a3_assess_parent,
            seeds=[int(s) for s in args.seeds],
            sampler_name=args.sampler_name,
            strict_volume_power=float(args.strict_volume_power),
            run_quality=bool(args.run_quality),
        )
    a3_record = _candidate_record(
        candidate_id=a3_id,
        stage="A3",
        overrides=a3_overrides,
        model_dir=a3_model_dir,
        assessment_dir=a3_assessment_dir,
        ranking=a3_ranking,
    )
    results["stages"]["A3"] = {"candidate": a3_record}

    summary_path = assess_root / "candidate_a_staged_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"[done] Wrote staged summary: {summary_path}")


if __name__ == "__main__":
    main()
