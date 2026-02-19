#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import wandb
except Exception:
    wandb = None

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from scripts.sampling_diagnostics import load_model_and_centroids
from scripts.metric_assessment_suite import (
    SamplerConfig,
    compute_local_coverage,
    run_rescue_assessment,
    sample_latents,
    set_seed,
)
from src.utils.metric_scorecard import ABSOLUTE_RULES_BY_NAME, classify_value, score_margin
from src.utils.rescue_metrics import (
    compute_border_tunnel_metrics,
    compute_plateau_alignment_metrics,
    compute_tangent_alignment_metric,
    model_r0,
)


CORE4_KEYS = (
    "tangent_alignment_mean",
    "rescue_directionality_mean",
    "border_overshoot_index",
    "plateau_fraction",
)
PRIMARY7_ORDERED_KEYS = (
    "tangent_alignment_mean",
    "rescue_directionality_mean",
    "border_overshoot_index",
    "plateau_fraction",
    "rescue_rate",
    "median_steps_to_manifold",
    "coverage_local",
)
PRIMARY7_WEIGHTS = {
    "tangent_alignment_mean": 0.22,
    "rescue_directionality_mean": 0.20,
    "border_overshoot_index": 0.17,
    "plateau_fraction": 0.15,
    "rescue_rate": 0.11,
    "median_steps_to_manifold": 0.09,
    "coverage_local": 0.06,
}


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


def _build_train_overrides(args: argparse.Namespace, model_out_root: Path) -> list[str]:
    overrides = {
        "epochs": int(args.epochs),
        "seed": int(args.train_seed),
        "output_dir": str(model_out_root),
    }
    optional_fields = (
        "atom_power",
        "temperature",
        "transition_steepness",
        "radial_stretch",
        "void_decay_scale",
        "void_decay_power",
        "attractor_gamma",
        "attractor_k_nearest",
    )
    for key in optional_fields:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    return [f"{key}={value}" for key, value in overrides.items()]


def train_model_from_config(args: argparse.Namespace, model_root: Path) -> Path:
    model_root.mkdir(parents=True, exist_ok=True)
    before = _list_subdirs(model_root)
    cmd = [
        args.python_exec,
        "scripts/run_with_config.py",
        "--config",
        str(args.base_config),
        *_build_train_overrides(args, model_root),
    ]
    logs = _run(cmd, cwd=ROOT_DIR)
    logged = _extract_saved_path(logs, "[RHVAE BASELINE] All outputs saved to:")
    if logged is not None and logged.exists():
        return logged
    return _find_created_subdir(model_root, before)


def sample_probe_points(
    centroids: torch.Tensor,
    n_points: int,
    bbox_scale: float,
    seed: int,
) -> torch.Tensor:
    n_points = max(int(n_points), 256)
    generator = torch.Generator(device=centroids.device)
    generator.manual_seed(int(seed))

    center = centroids.mean(dim=0)
    c_min = centroids.min(dim=0).values
    c_max = centroids.max(dim=0).values
    span = torch.clamp(c_max - c_min, min=1e-3)
    lo = center - float(bbox_scale) * span
    hi = center + float(bbox_scale) * span

    n_uniform = max(1, n_points // 2)
    n_local = n_points - n_uniform

    rand = torch.rand(
        n_uniform,
        centroids.shape[1],
        generator=generator,
        device=centroids.device,
        dtype=centroids.dtype,
    )
    uniform = lo.unsqueeze(0) + rand * (hi - lo).unsqueeze(0)

    if n_local > 0:
        idx = torch.randint(
            0,
            centroids.shape[0],
            (n_local,),
            generator=generator,
            device=centroids.device,
        )
        noise = 0.35 * span.unsqueeze(0) * torch.randn(
            n_local,
            centroids.shape[1],
            generator=generator,
            device=centroids.device,
            dtype=centroids.dtype,
        )
        local = centroids[idx] + noise
        return torch.cat([uniform, local], dim=0)
    return uniform


def compute_core4_metrics(
    model: Any,
    centroids: torch.Tensor,
    probe_points: torch.Tensor,
    probe_seed: int,
) -> dict[str, float]:
    r0 = model_r0(model, device=centroids.device)
    pa = compute_plateau_alignment_metrics(model, probe_points, centroids, r0=r0, grad_threshold=1e-3)
    tan = compute_tangent_alignment_metric(
        model,
        centroids,
        r0=r0,
        n_samples=1024,
        k_neighbors=8,
        band_min=0.6,
        band_max=1.2,
        seed=int(probe_seed) + 17,
    )
    border = compute_border_tunnel_metrics(
        model,
        points=probe_points,
        centroids=centroids,
        r0=r0,
    )
    return {
        "tangent_alignment_mean": float(tan["tangent_alignment_mean"]),
        "rescue_directionality_mean": float(pa["rescue_directionality_mean"]),
        "border_overshoot_index": float(border["border_overshoot_index"]),
        "plateau_fraction": float(pa["plateau_fraction"]),
        "r0_model": float(r0),
        "logdet_manifold_median": float(border["logdet_manifold_median"]),
        "logdet_transition_median": float(border["logdet_transition_median"]),
        "logdet_far_void_median": float(border["logdet_far_void_median"]),
        "tunnel_neff_transition_median": float(border["tunnel_neff_transition_median"]),
        "tunnel_neff_far_void_median": float(border["tunnel_neff_far_void_median"]),
    }


def compute_primary7_metrics(
    model: Any,
    centroids: torch.Tensor,
    sampler_cfg: SamplerConfig,
    sampling_mcmc_steps: int,
    coverage_samples: int,
    rescue_horizon: int,
    rescue_ring_starts: int,
    rescue_gaussian_starts: int,
    seed: int,
) -> dict[str, float]:
    set_seed(int(seed))
    sampled, sample_acc = sample_latents(
        model,
        sampler_cfg,
        n_samples=max(1, int(coverage_samples)),
        mcmc_steps=max(1, int(sampling_mcmc_steps)),
    )
    coverage = compute_local_coverage(sampled[: max(1, int(coverage_samples))], centroids)
    rescue = run_rescue_assessment(
        model,
        centroids,
        sampler_cfg,
        horizon=max(2, int(rescue_horizon)),
        ring_starts=max(1, int(rescue_ring_starts)),
        gaussian_starts=max(1, int(rescue_gaussian_starts)),
        seed=int(seed) + 37,
    )
    return {
        "coverage_local": float(coverage),
        "rescue_rate": float(rescue["rescue_rate"]),
        "median_steps_to_manifold": float(rescue["median_steps_to_manifold"]),
        "plateau_fraction": float(rescue["plateau_fraction"]),
        "tangent_alignment_mean": float(rescue["tangent_alignment_mean"]),
        "rescue_directionality_mean": float(rescue["rescue_directionality_mean"]),
        "border_overshoot_index": float(rescue["border_overshoot_index"]),
        "sample_acceptance_rate": float(sample_acc),
        "r0_model": float(rescue["r0_model"]),
        "radius": float(rescue["radius"]),
    }


def compute_core4_objective(metrics: dict[str, float]) -> dict[str, Any]:
    core4_values = [metrics.get(key, float("nan")) for key in CORE4_KEYS]
    if not all(math.isfinite(float(v)) for v in core4_values):
        return {
            "objective_metric_core4_v1": -5.0,
            "core4_green_count": 0,
            "core4_n_total": len(CORE4_KEYS),
            "margins": {key: float("nan") for key in CORE4_KEYS},
            "statuses": {key: "red" for key in CORE4_KEYS},
        }

    margins: dict[str, float] = {}
    statuses: dict[str, str] = {}
    for key in CORE4_KEYS:
        value = float(metrics[key])
        rule = ABSOLUTE_RULES_BY_NAME[key]
        margins[key] = float(score_margin(value, rule))
        statuses[key] = classify_value(value, rule)

    objective = float(np.mean([margins[key] for key in CORE4_KEYS]))
    green_count = int(sum(1 for key in CORE4_KEYS if statuses[key] == "green"))
    return {
        "objective_metric_core4_v1": objective,
        "core4_green_count": green_count,
        "core4_n_total": len(CORE4_KEYS),
        "margins": margins,
        "statuses": statuses,
    }


def compute_primary7_weighted_objective(metrics: dict[str, float]) -> dict[str, Any]:
    values = [metrics.get(key, float("nan")) for key in PRIMARY7_ORDERED_KEYS]
    if not all(math.isfinite(float(v)) for v in values):
        return {
            "objective_primary7_weighted_v1": -5.0,
            "primary7_green_count": 0,
            "primary7_n_total": len(PRIMARY7_ORDERED_KEYS),
            "margins": {key: float("nan") for key in PRIMARY7_ORDERED_KEYS},
            "statuses": {key: "red" for key in PRIMARY7_ORDERED_KEYS},
            "weighted_terms": {key: float("nan") for key in PRIMARY7_ORDERED_KEYS},
        }

    margins: dict[str, float] = {}
    statuses: dict[str, str] = {}
    weighted_terms: dict[str, float] = {}
    objective = 0.0
    for key in PRIMARY7_ORDERED_KEYS:
        value = float(metrics[key])
        rule = ABSOLUTE_RULES_BY_NAME[key]
        margin = float(score_margin(value, rule))
        weight = float(PRIMARY7_WEIGHTS[key])
        margins[key] = margin
        statuses[key] = classify_value(value, rule)
        weighted_terms[key] = weight * margin
        objective += weighted_terms[key]

    green_count = int(sum(1 for key in PRIMARY7_ORDERED_KEYS if statuses[key] == "green"))
    return {
        "objective_primary7_weighted_v1": float(objective),
        "primary7_green_count": green_count,
        "primary7_n_total": len(PRIMARY7_ORDERED_KEYS),
        "margins": margins,
        "statuses": statuses,
        "weighted_terms": weighted_terms,
    }


def init_wandb_run(args: argparse.Namespace, run_stamp: str, config: dict[str, Any]) -> Any | None:
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    if wandb is None:
        return None

    should_init = bool(
        args.wandb_project
        or os.environ.get("WANDB_PROJECT")
        or os.environ.get("WANDB_SWEEP_ID")
        or os.environ.get("WANDB_RUN_ID")
    )
    if not should_init:
        return None

    project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
    run_name = args.wandb_run_name
    if args.wandb_name_mode == "timestamp":
        run_name = run_name or f"metric_core4_trial_{run_stamp}"
    elif args.wandb_name_mode == "auto" and not run_name:
        run_name = f"metric_core4_trial_seed{args.train_seed}_{run_stamp}"
    tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None

    kwargs: dict[str, Any] = {"config": config, "job_type": "metric_core4_trial"}
    if project:
        kwargs["project"] = project
    if entity:
        kwargs["entity"] = entity
    if args.wandb_group:
        kwargs["group"] = args.wandb_group
    if run_name:
        kwargs["name"] = run_name
    if tags:
        kwargs["tags"] = tags
    try:
        return wandb.init(**kwargs)
    except Exception as exc:
        print(f"[warning] WandB init failed: {exc}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single W&B trial for metric-only Core-4 objective.")
    parser.add_argument("--base_config", type=str, default="configs/run_geometry_default_soft_no_tunnels_v2.yaml")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train_seed", type=int, default=13)
    parser.add_argument("--runs_root", type=str, default="outputs/metric_core4_sweep")
    parser.add_argument("--analysis_root", type=str, default="results/metric_core4_sweep")
    parser.add_argument("--existing_model_path", type=str, default=None)
    parser.add_argument("--python_exec", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--probe_points", type=int, default=2048)
    parser.add_argument("--probe_bbox_scale", type=float, default=2.5)
    parser.add_argument("--probe_seed", type=int, default=13)
    parser.add_argument(
        "--eval_sampler_name",
        type=str,
        default="volume",
        choices=["volume", "volume_riemannian", "riemannian", "geodesic", "volume_det", "volume_riemannian_det", "dual_riemannian"],
    )
    parser.add_argument("--eval_n_lf", type=int, default=30)
    parser.add_argument("--eval_eps_lf", type=float, default=0.30)
    parser.add_argument("--eval_volume_power", type=float, default=1.0)
    parser.add_argument("--eval_sampling_mcmc_steps", type=int, default=12)
    parser.add_argument("--eval_coverage_samples", type=int, default=200)
    parser.add_argument("--eval_rescue_horizon", type=int, default=25)
    parser.add_argument("--eval_rescue_ring_starts", type=int, default=4)
    parser.add_argument("--eval_rescue_gaussian_starts", type=int, default=4)
    parser.add_argument("--eval_seed", type=int, default=13)

    parser.add_argument("--atom_power", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--transition_steepness", type=float, default=None)
    parser.add_argument("--radial_stretch", type=float, default=None)
    parser.add_argument("--void_decay_scale", type=float, default=None)
    parser.add_argument("--void_decay_power", type=float, default=None)
    parser.add_argument("--attractor_gamma", type=float, default=None)
    parser.add_argument("--attractor_k_nearest", type=int, default=None)

    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_name_mode", type=str, default="auto", choices=["timestamp", "auto", "manual"])
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_n_lf <= 0:
        raise ValueError("--eval_n_lf must be > 0")
    if args.eval_eps_lf <= 0:
        raise ValueError("--eval_eps_lf must be > 0")
    run_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_id = os.environ.get("WANDB_RUN_ID")
    if run_id:
        token = run_id
    else:
        token = f"pid{os.getpid()}"
    run_token = f"{run_stamp}_{token}"

    analysis_dir = Path(args.analysis_root) / run_token
    analysis_dir.mkdir(parents=True, exist_ok=True)

    base_config = Path(args.base_config)
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    if args.existing_model_path:
        model_dir = Path(args.existing_model_path)
    else:
        model_root = Path(args.runs_root) / run_token
        model_dir = train_model_from_config(args, model_root=model_root)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model output directory not found: {model_dir}")

    device = torch.device(args.device)
    model, centroids = load_model_and_centroids(model_dir, device=device, aggressive_void=False)
    probe_points = sample_probe_points(
        centroids=centroids,
        n_points=args.probe_points,
        bbox_scale=args.probe_bbox_scale,
        seed=args.probe_seed,
    )
    metrics = compute_core4_metrics(model, centroids, probe_points=probe_points, probe_seed=args.probe_seed)
    objective = compute_core4_objective(metrics)
    eval_cfg = SamplerConfig(
        sampler_name=str(args.eval_sampler_name),
        exact=True,
        volume_power=float(args.eval_volume_power),
        n_lf=int(args.eval_n_lf),
        eps_lf=float(args.eval_eps_lf),
        momentum_persist=0.0,
        fp_steps=15,
        fp_damping=0.7,
    )
    primary7_metrics = compute_primary7_metrics(
        model=model,
        centroids=centroids,
        sampler_cfg=eval_cfg,
        sampling_mcmc_steps=int(args.eval_sampling_mcmc_steps),
        coverage_samples=int(args.eval_coverage_samples),
        rescue_horizon=int(args.eval_rescue_horizon),
        rescue_ring_starts=int(args.eval_rescue_ring_starts),
        rescue_gaussian_starts=int(args.eval_rescue_gaussian_starts),
        seed=int(args.eval_seed),
    )
    primary7_objective = compute_primary7_weighted_objective(primary7_metrics)

    summary = {
        "created_at": datetime.datetime.now().isoformat(),
        "model_dir": str(model_dir),
        "base_config": str(base_config),
        "device": str(device),
        "probe_points": int(args.probe_points),
        "probe_bbox_scale": float(args.probe_bbox_scale),
        "probe_seed": int(args.probe_seed),
        "core4_metrics": metrics,
        "core4_objective": objective,
        "primary7_metrics": primary7_metrics,
        "primary7_objective": primary7_objective,
        "train_params": {
            "epochs": int(args.epochs),
            "seed": int(args.train_seed),
            "atom_power": args.atom_power,
            "temperature": args.temperature,
            "transition_steepness": args.transition_steepness,
            "radial_stretch": args.radial_stretch,
            "void_decay_scale": args.void_decay_scale,
            "void_decay_power": args.void_decay_power,
            "attractor_gamma": args.attractor_gamma,
            "attractor_k_nearest": args.attractor_k_nearest,
        },
        "eval_sampler": {
            "sampler_name": str(args.eval_sampler_name),
            "n_lf": int(args.eval_n_lf),
            "eps_lf": float(args.eval_eps_lf),
            "volume_power": float(args.eval_volume_power),
            "sampling_mcmc_steps": int(args.eval_sampling_mcmc_steps),
            "coverage_samples": int(args.eval_coverage_samples),
            "rescue_horizon": int(args.eval_rescue_horizon),
            "rescue_ring_starts": int(args.eval_rescue_ring_starts),
            "rescue_gaussian_starts": int(args.eval_rescue_gaussian_starts),
            "eval_seed": int(args.eval_seed),
        },
    }
    summary_path = analysis_dir / "trial_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    config_payload = vars(args).copy()
    config_payload["model_dir"] = str(model_dir)
    run = init_wandb_run(args, run_stamp=run_stamp, config=config_payload)
    if run is not None:
        scalar_payload: dict[str, Any] = {
            "sweep/objective_metric_core4_v1": float(objective["objective_metric_core4_v1"]),
            "sweep/core4_green_count": int(objective["core4_green_count"]),
            "sweep/core4_n_total": int(objective["core4_n_total"]),
            "sweep/objective_primary7_weighted_v1": float(primary7_objective["objective_primary7_weighted_v1"]),
            "sweep/primary7_green_count": int(primary7_objective["primary7_green_count"]),
            "sweep/primary7_n_total": int(primary7_objective["primary7_n_total"]),
        }
        for key in CORE4_KEYS:
            value = metrics.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                scalar_payload[f"core4/{key}"] = float(value)
            margin = objective["margins"].get(key)
            if isinstance(margin, (int, float)) and math.isfinite(float(margin)):
                scalar_payload[f"core4_margin/{key}"] = float(margin)
        for key in PRIMARY7_ORDERED_KEYS:
            value = primary7_metrics.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                scalar_payload[f"primary7/{key}"] = float(value)
            margin = primary7_objective["margins"].get(key)
            if isinstance(margin, (int, float)) and math.isfinite(float(margin)):
                scalar_payload[f"primary7_margin/{key}"] = float(margin)
            term = primary7_objective["weighted_terms"].get(key)
            if isinstance(term, (int, float)) and math.isfinite(float(term)):
                scalar_payload[f"primary7_weighted_term/{key}"] = float(term)
        run.log(scalar_payload)
        run.summary["model_dir"] = str(model_dir)
        run.summary["trial_summary_json"] = str(summary_path)
        run.finish()

    print(f"[done] model_dir={model_dir}")
    print(f"[done] sweep/objective_metric_core4_v1={objective['objective_metric_core4_v1']:.6f}")
    print(f"[done] sweep/objective_primary7_weighted_v1={primary7_objective['objective_primary7_weighted_v1']:.6f}")
    print(f"[done] summary_json={summary_path}")


if __name__ == "__main__":
    main()
