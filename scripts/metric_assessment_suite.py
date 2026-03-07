#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
try:
    import wandb
except Exception:
    wandb = None

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from scripts.rhmc_chain_demo import _build_sampler, run_hmc_chain as run_chain_with_energy
from scripts.sampling_diagnostics import (
    compute_fid_score,
    compute_precision_recall,
    decode_latents,
    load_model_and_centroids,
)
from src.utils.mcmc_metrics import aggregate_hamiltonian_metrics, ess_iact_per_dim
from src.utils.metric_scorecard import (
    ABSOLUTE_RULES,
    ABSOLUTE_RULES_BY_NAME,
    evaluate_absolute,
    evaluate_claim,
    evaluate_relative_gains,
    score_margin,
)
from src.utils.rescue_metrics import (
    compute_border_tunnel_metrics,
    compute_plateau_alignment_metrics,
    compute_tangent_alignment_metric,
    model_r0,
    summarize_rescue_from_distances,
)


def _git_text(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return f"<unavailable: {exc}>"
    return proc.stdout


def write_provenance_snapshot(out_dir: Path, *, args_payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_args.json").write_text(json.dumps(args_payload, indent=2), encoding="utf-8")
    (out_dir / "git_commit.txt").write_text(_git_text(["rev-parse", "HEAD"]), encoding="utf-8")
    (out_dir / "git_status.txt").write_text(_git_text(["status", "--short"]), encoding="utf-8")


DEFAULT_BASELINE_MODEL = ROOT_DIR / "outputs/pythae_rhvae_baseline/2026-02-09_15-04-06"
DEFAULT_ANISO_MODEL = ROOT_DIR / "outputs/pythae_rhvae_baseline/2026-02-09_18-16-00"
MODEL_KEYS = ("baseline", "aniso")
DEFAULT_NUMERIC_ARGS = {
    "n_chains": 4,
    "chain_length": 500,
    "burn_in": 100,
    "sampling_mcmc_steps": 100,
    "coverage_samples": 2000,
    "rescue_ring_starts": 32,
    "rescue_gaussian_starts": 32,
    "rescue_horizon": 100,
    "quality_samples": 1000,
    "fid_samples": 1000,
    "pilot_n_chains": 2,
    "pilot_chain_length": 150,
    "bootstrap_samples": 1000,
}
PROFILE_PRESETS = {
    "full": {},
    "short": {
        "n_chains": 2,
        "chain_length": 220,
        "burn_in": 50,
        "sampling_mcmc_steps": 40,
        "coverage_samples": 700,
        "rescue_ring_starts": 10,
        "rescue_gaussian_starts": 10,
        "rescue_horizon": 55,
        "quality_samples": 400,
        "fid_samples": 400,
        "pilot_n_chains": 2,
        "pilot_chain_length": 90,
        "bootstrap_samples": 400,
    },
    "smoke": {
        "n_chains": 1,
        "chain_length": 80,
        "burn_in": 20,
        "sampling_mcmc_steps": 8,
        "coverage_samples": 80,
        "rescue_ring_starts": 2,
        "rescue_gaussian_starts": 2,
        "rescue_horizon": 20,
        "quality_samples": 80,
        "fid_samples": 80,
        "pilot_n_chains": 1,
        "pilot_chain_length": 40,
        "bootstrap_samples": 200,
    },
}
REFERENCE_VISUAL_SPECS = {
    "training": [
        ("training_curves", "training_curves.png"),
        ("reconstructions", "reconstructions_epoch_{latest}.png"),
        ("latent_space", "latent_space_epoch_{latest}.png"),
        ("metric_field", "metric_field_epoch_{latest}.png"),
        ("metric_tissot", "metric_tissot_epoch_{latest}.png"),
    ],
    "analysis": [
        ("metric_surface_3d", "metric_surface_3d.png"),
        ("metric_surface_3d_wide", "metric_surface_3d_wide.png"),
        ("volume_rescue_quiver", "volume_rescue_quiver.png"),
        ("rhmc_chain", "rhmc_chain.png"),
        ("geodesic_batch", "geodesic_batch.png"),
        ("geodesic_image_sequences", "geodesic_image_sequences.png"),
        ("prior_generation", "prior_generation.png"),
        ("curvature_histograms", "curvature_histograms.png"),
    ],
    "sampling": [
        ("interpolation_paths", "interpolation_paths.png"),
        ("interpolation_geodesic", "interpolation_geodesic.png"),
        ("multi_start_sampling", "multi_start_sampling.png"),
        ("rhmc_diagnostics", "rhmc_diagnostics.png"),
        ("quality_metrics", "quality_metrics.png"),
        ("fid_comparison", "fid_comparison.png"),
    ],
}
SUMMARY_METRIC_KEYS = (
    "acceptance_mean",
    "dh_p95_abs",
    "h_drift_slope_abs_mean",
    "ess_norm_min",
    "iact_median",
    "coverage_local",
    "rescue_rate",
    "median_steps_to_manifold",
    "plateau_fraction",
    "tangent_alignment_mean",
    "rescue_directionality_mean",
    "border_overshoot_index",
    "alignment_mean",
    "quality_precision",
    "quality_recall",
    "quality_diversity",
    "fid",
)
RANK_STABILITY_KEYS = ("acceptance_mean", "dh_p95_abs", "h_drift_slope_abs_mean")
RANK_PRIMARY_KEYS = (
    "coverage_local",
    "rescue_rate",
    "median_steps_to_manifold",
    "plateau_fraction",
    "tangent_alignment_mean",
    "rescue_directionality_mean",
    "border_overshoot_index",
)
RANK_TIE_BREAKER_KEYS = (
    "tangent_alignment_mean",
    "rescue_rate",
    "coverage_local",
)
CORE4_OBJECTIVE_KEYS = (
    "tangent_alignment_mean",
    "rescue_directionality_mean",
    "border_overshoot_index",
    "plateau_fraction",
)
FULL_OBJECTIVE_STABILITY_KEYS = ("acceptance_mean", "dh_p95_abs", "h_drift_slope_abs_mean")
FULL_OBJECTIVE_PRIMARY_KEYS = RANK_PRIMARY_KEYS


@dataclass(frozen=True)
class SamplerConfig:
    sampler_name: str = "volume_riemannian"
    exact: bool = True
    volume_power: float = 1.0
    n_lf: int = 30
    eps_lf: float = 0.01
    momentum_persist: float = 0.0
    eps_jitter: float = 0.0
    n_lf_jitter: int = 0
    fp_steps: int = 15
    fp_damping: float = 0.7
    radial_prior_weight: float = 0.0
    mass_mode: str = "standard"
    adaptive_dual_step: bool = False
    adaptive_max_dual_displacement: float = 0.75
    adaptive_min_step_scale: float = 0.05
    dynamic_jitter_scale: float = 0.0
    hybrid_explore_steps: int = 3
    hybrid_rescue_steps: int = 1


def effective_volume_exponent(sampler_name: str, volume_power: float) -> float:
    """Effective exponent on det(G_inv) in the induced target density."""
    name = str(sampler_name).strip().lower()
    vp = float(volume_power)
    if name in {"volume", "volume_riemannian"}:
        return vp + 0.5
    return float("nan")


def normalize_sampler_name(name: str) -> str:
    raw = str(name).strip().lower()
    aliases = {
        "volume": "volume_riemannian",
        "volume_riemannian": "volume_riemannian",
        "riemannian": "riemannian",
        "geodesic": "geodesic",
        "volume_det": "volume_det",
        "volume_riemannian_det": "volume_riemannian_det",
        "dual_riemannian": "dual_riemannian",
        "hybrid_volume_mix": "hybrid_volume_mix",
        "hybrid_mix": "hybrid_volume_mix",
        "mixed": "hybrid_volume_mix",
    }
    if raw in aliases:
        return aliases[raw]
    raise ValueError(f"Unsupported sampler_name: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_model_artifacts(model_dir: Path) -> None:
    required = ("rhvae_metric.pt", "rhvae_model.pt")
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {model_dir}: {missing}")


def load_real_data_if_available(model_dir: Path) -> torch.Tensor | None:
    data_path = model_dir / "train_data.pt"
    if not data_path.exists():
        return None
    payload = torch.load(data_path, map_location="cpu")
    if isinstance(payload, dict):
        data = payload.get("data", payload.get("images"))
    else:
        data = payload
    if data is None:
        return None
    return torch.as_tensor(data, dtype=torch.float32)


def build_output_dir(base_dir: Path) -> Path:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _parse_wandb_tags(raw_tags: str | None) -> list[str] | None:
    if raw_tags is None:
        return None
    tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
    return tags or None


def _build_wandb_run_name(args: argparse.Namespace, run_stamp: str) -> str | None:
    if args.wandb_name_mode == "manual":
        return args.wandb_run_name
    if args.wandb_name_mode == "timestamp":
        return args.wandb_run_name or f"metric_assessment_{run_stamp}"
    if args.wandb_run_name:
        return args.wandb_run_name
    protocol_key = "-".join(args.protocols)
    return f"metric_assessment_{protocol_key}_seeds{len(args.sampling_seeds)}_{run_stamp}"


def init_wandb_run(args: argparse.Namespace, run_stamp: str) -> Any | None:
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    if wandb is None:
        if args.wandb_project:
            print("[warning] wandb not installed; skipping wandb logging.")
        else:
            print("[info] wandb disabled (wandb not installed).")
        return None

    should_init = bool(
        args.wandb_project
        or os.environ.get("WANDB_PROJECT")
        or os.environ.get("WANDB_SWEEP_ID")
        or os.environ.get("WANDB_RUN_ID")
    )
    if not should_init:
        print("[info] wandb disabled (no --wandb_project and no WANDB_* sweep/run env).")
        return None

    project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
    tags = _parse_wandb_tags(args.wandb_tags)
    run_name = _build_wandb_run_name(args, run_stamp)

    config = vars(args).copy()
    config.pop("current_protocol", None)

    wandb_kwargs: dict[str, Any] = {
        "config": config,
        "job_type": "metric_assessment",
    }
    if project:
        wandb_kwargs["project"] = project
    if entity:
        wandb_kwargs["entity"] = entity
    if args.wandb_group:
        wandb_kwargs["group"] = args.wandb_group
    if run_name:
        wandb_kwargs["name"] = run_name
    if tags:
        wandb_kwargs["tags"] = tags

    try:
        run = wandb.init(**wandb_kwargs)
        print(f"[info] wandb enabled: project={project or '<default>'}, entity={entity or '<default>'}, mode={os.environ.get('WANDB_MODE', 'online')}")
        return run
    except Exception as exc:
        print(f"[warning] WandB init failed: {exc}")
        return None


def _to_wandb_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (list, tuple)):
        return [_to_wandb_value(v) for v in value]
    return value


def log_seed_metrics_to_wandb(wandb_run: Any | None, row: dict[str, Any], step: int | None = None) -> None:
    if wandb_run is None:
        return
    key_prefix = f"seed/{row['protocol']}/{row['model_key']}"
    metric_keys = (
        "acceptance_mean",
        "dh_p95_abs",
        "h_drift_slope_abs_mean",
        "ess_norm_min",
        "iact_median",
        "coverage_local",
        "rescue_rate",
        "median_steps_to_manifold",
        "plateau_fraction",
        "tangent_alignment_mean",
        "rescue_directionality_mean",
        "border_overshoot_index",
        "alignment_mean",
        "quality_precision",
        "quality_recall",
        "quality_diversity",
        "fid",
    )
    payload: dict[str, Any] = {
        f"{key_prefix}/seed": int(row["seed"]),
    }
    for key in metric_keys:
        value = _to_wandb_value(row.get(key))
        if value is not None:
            payload[f"{key_prefix}/{key}"] = value

    try:
        if step is None:
            wandb_run.log(payload)
        else:
            wandb_run.log(payload, step=int(step))
    except Exception as exc:
        print(f"[warning] WandB seed logging failed: {exc}")


def safe_wandb_log(wandb_run: Any | None, payload: dict[str, Any], step: int | None = None) -> None:
    if wandb_run is None:
        return
    try:
        if step is None:
            wandb_run.log(payload)
        else:
            wandb_run.log(payload, step=int(step))
    except Exception as exc:
        print(f"[warning] WandB logging failed: {exc}")


def rows_to_wandb_table(rows: list[dict[str, Any]]) -> Any | None:
    if wandb is None:
        return None
    columns = sorted({key for row in rows for key in row.keys()})
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*[_to_wandb_value(row.get(col)) for col in columns])
    return table


def summary_to_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol, per_model in summary.items():
        for model_key, model_payload in per_model.items():
            metrics = model_payload.get("metrics", {})
            for metric_name, metric_payload in metrics.items():
                if not isinstance(metric_payload, dict):
                    continue
                rows.append(
                    {
                        "protocol": protocol,
                        "model_key": model_key,
                        "metric": metric_name,
                        "mean": _to_wandb_value(metric_payload.get("mean")),
                        "std": _to_wandb_value(metric_payload.get("std")),
                        "ci95_low": _to_wandb_value(metric_payload.get("ci95_low")),
                        "ci95_high": _to_wandb_value(metric_payload.get("ci95_high")),
                        "n": _to_wandb_value(metric_payload.get("n")),
                    }
                )
    return rows


def log_summary_to_wandb(
    wandb_run: Any | None,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    claim: dict[str, Any] | None,
) -> None:
    if wandb_run is None or wandb is None:
        return

    scalar_payload: dict[str, Any] = {}
    for protocol in summary.keys():
        for model_key in MODEL_KEYS:
            mean_metrics = _mean_map(summary, protocol, model_key)
            for metric_name, metric_value in mean_metrics.items():
                value = _to_wandb_value(metric_value)
                if value is not None:
                    scalar_payload[f"summary/{protocol}/{model_key}/{metric_name}"] = value

    if claim is not None:
        scalar_payload["scorecard/claim_validated"] = float(bool(claim.get("claim_validated", False)))
        for block_name in ("strict", "matched"):
            block = claim.get(block_name, {})
            scalar_payload[f"scorecard/{block_name}/pass"] = float(bool(block.get("pass", False)))
            rel = block.get("relative", {})
            if isinstance(rel, dict):
                if "n_pass" in rel:
                    scalar_payload[f"scorecard/{block_name}/relative_n_pass"] = int(rel["n_pass"])
                if "n_total" in rel:
                    scalar_payload[f"scorecard/{block_name}/relative_n_total"] = int(rel["n_total"])

    if scalar_payload:
        try:
            wandb_run.log(scalar_payload)
        except Exception as exc:
            print(f"[warning] WandB summary logging failed: {exc}")

    tables_payload: dict[str, Any] = {}
    raw_table = rows_to_wandb_table(rows)
    if raw_table is not None:
        tables_payload["tables/raw_metrics"] = raw_table
    score_table = rows_to_wandb_table(score_rows)
    if score_table is not None:
        tables_payload["tables/scorecard"] = score_table
    summary_table = rows_to_wandb_table(summary_to_rows(summary))
    if summary_table is not None:
        tables_payload["tables/summary"] = summary_table
    if tables_payload:
        try:
            wandb_run.log(tables_payload)
        except Exception as exc:
            print(f"[warning] WandB table logging failed: {exc}")

    chart_metrics = (
        "acceptance_mean",
        "ess_norm_min",
        "coverage_local",
        "rescue_rate",
        "median_steps_to_manifold",
        "plateau_fraction",
        "tangent_alignment_mean",
        "rescue_directionality_mean",
        "border_overshoot_index",
    )
    for metric_name in chart_metrics:
        chart_table = wandb.Table(columns=["protocol_model", "value"])
        count = 0
        for protocol in summary.keys():
            for model_key in MODEL_KEYS:
                value = _mean_map(summary, protocol, model_key).get(metric_name)
                value = _to_wandb_value(value)
                if value is None:
                    continue
                chart_table.add_data(f"{protocol}/{model_key}", value)
                count += 1
        if count == 0:
            continue
        chart = wandb.plot.bar(
            chart_table,
            "protocol_model",
            "value",
            title=f"{metric_name} by protocol/model",
        )
        try:
            wandb_run.log({f"dashboard/{metric_name}_bar": chart})
        except Exception as exc:
            print(f"[warning] WandB chart logging failed for {metric_name}: {exc}")


def log_plot_images_to_wandb(wandb_run: Any | None, plots_dir: Path, max_images: int) -> None:
    if wandb_run is None or wandb is None:
        return
    if not plots_dir.exists():
        return
    pngs = sorted(plots_dir.rglob("*.png"))
    if not pngs:
        return

    limit = len(pngs) if max_images <= 0 else min(len(pngs), max_images)
    for png in pngs[:limit]:
        rel = png.relative_to(plots_dir).as_posix().replace("/", "_")
        key = f"plots/{rel.replace('.png', '')}"
        try:
            wandb_run.log({key: wandb.Image(str(png))})
        except Exception as exc:
            print(f"[warning] WandB image logging failed for {png}: {exc}")
            break

    wandb_run.summary["plots/available_png"] = len(pngs)
    wandb_run.summary["plots/logged_png"] = limit


def log_output_artifact_to_wandb(wandb_run: Any | None, out_dir: Path) -> None:
    if wandb_run is None or wandb is None:
        return
    artifact_name = f"metric-assessment-{out_dir.name}"
    artifact = wandb.Artifact(name=artifact_name, type="metric_assessment")

    files = [
        out_dir / "raw_metrics.csv",
        out_dir / "summary_by_protocol.json",
        out_dir / "scorecard.csv",
        out_dir / "scorecard.md",
        out_dir / "ranking_summary.csv",
        out_dir / "ranking_summary.json",
        out_dir / "model_run_bilan.csv",
        out_dir / "protocol_configs.json",
    ]
    for path in files:
        if path.exists():
            artifact.add_file(str(path), name=path.name)
    plots_dir = out_dir / "plots"
    if plots_dir.exists():
        artifact.add_dir(str(plots_dir), name="plots")
    try:
        wandb_run.log_artifact(artifact)
    except Exception as exc:
        print(f"[warning] WandB artifact logging failed: {exc}")


def sampler_from_config(
    model: Any,
    cfg: SamplerConfig,
    mcmc_steps: int,
    beta_zero: float = 1.0,
) -> Any:
    sampler = _build_sampler(
        cfg.sampler_name,
        model,
        mcmc_steps=mcmc_steps,
        n_lf=cfg.n_lf,
        eps_lf=cfg.eps_lf,
        beta_zero=beta_zero,
        volume_power=cfg.volume_power,
        radial_prior_weight=float(cfg.radial_prior_weight),
        radial_prior_center=None,
        mass_mode=str(cfg.mass_mode),
        adaptive_dual_step=bool(cfg.adaptive_dual_step),
        adaptive_max_dual_displacement=float(cfg.adaptive_max_dual_displacement),
        adaptive_min_step_scale=float(cfg.adaptive_min_step_scale),
        dynamic_jitter_scale=float(cfg.dynamic_jitter_scale),
        hybrid_explore_steps=int(cfg.hybrid_explore_steps),
        hybrid_rescue_steps=int(cfg.hybrid_rescue_steps),
    )
    sampler.exact = bool(cfg.exact)
    sampler.momentum_persist = float(cfg.momentum_persist)
    if hasattr(sampler, "fp_steps"):
        sampler.fp_steps = int(cfg.fp_steps)
    if hasattr(sampler, "fp_damping"):
        sampler.fp_damping = float(cfg.fp_damping)
    if hasattr(sampler, "adaptive_dual_step"):
        sampler.adaptive_dual_step = bool(cfg.adaptive_dual_step)
    if hasattr(sampler, "adaptive_max_dual_displacement"):
        sampler.adaptive_max_dual_displacement = float(cfg.adaptive_max_dual_displacement)
    if hasattr(sampler, "adaptive_min_step_scale"):
        sampler.adaptive_min_step_scale = float(cfg.adaptive_min_step_scale)
    if hasattr(sampler, "dynamic_jitter_scale"):
        sampler.dynamic_jitter_scale = float(cfg.dynamic_jitter_scale)
    return sampler


def run_chain_batch(
    model: Any,
    centroids: torch.Tensor,
    cfg: SamplerConfig,
    n_chains: int,
    chain_length: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=centroids.device)
    generator.manual_seed(int(seed))

    chains: list[np.ndarray] = []
    energies: list[dict[str, np.ndarray]] = []
    accept_rates: list[float] = []
    wall_time_start = time.perf_counter()
    for _ in range(n_chains):
        start_idx = torch.randint(
            0,
            centroids.shape[0],
            (1,),
            generator=generator,
            device=centroids.device,
        ).item()
        start = centroids[start_idx : start_idx + 1].clone()
        sampler = sampler_from_config(model, cfg, mcmc_steps=chain_length)
        chain, energy = run_chain_with_energy(
            start,
            sampler,
            chain_length,
            cfg.n_lf,
            cfg.eps_lf,
            eps_jitter=cfg.eps_jitter,
            n_lf_jitter=cfg.n_lf_jitter,
        )
        chains.append(chain)
        energies.append(energy)
        accept_arr = np.asarray(energy.get("accept", np.array([])), dtype=float)
        accept_rates.append(float(np.mean(accept_arr)) if accept_arr.size else float("nan"))

    return {
        "chains": chains,
        "energies": energies,
        "acceptance_rates": accept_rates,
        "wall_time_sec": float(max(0.0, time.perf_counter() - wall_time_start)),
    }


def evaluate_mixing_stability(
    chains: list[np.ndarray],
    energies: list[dict[str, np.ndarray]],
    burn_in: int,
    wall_time_sec: float | None = None,
) -> dict[str, Any]:
    chain_stack = np.stack(chains, axis=0)
    accept = [np.asarray(e.get("accept", np.array([])), dtype=float) for e in energies]
    hamiltonians = [np.asarray(e.get("hamiltonian", np.array([])), dtype=float) for e in energies]
    proposal_dh = [np.asarray(e.get("proposal_dh", np.array([])), dtype=float) for e in energies]

    acceptance_rates = [float(np.mean(a)) for a in accept if a.size > 0]
    acceptance_mean = float(np.mean(acceptance_rates)) if acceptance_rates else float("nan")

    ham_stats = aggregate_hamiltonian_metrics(hamiltonians, proposal_dh)
    burn = min(burn_in, max(0, chain_stack.shape[1] - 2))
    ess_stats = ess_iact_per_dim(chain_stack, burn_in=burn, max_lag=200, normalize_to_1000=True)

    return {
        "acceptance_mean": acceptance_mean,
        "dh_p95_abs": ham_stats["dh_p95_abs"],
        "h_drift_slope_abs_mean": ham_stats["h_drift_slope_abs_mean"],
        "ess_min": ess_stats["ess_min"],
        "ess_median": ess_stats["ess_median"],
        "ess_norm_min": ess_stats["ess_norm_min"],
        "ess_median_per_sec": float(ess_stats["ess_median"] / max(1e-9, float(wall_time_sec)))
        if wall_time_sec is not None
        else float("nan"),
        "iact_median": ess_stats["iact_median"],
        "ess_per_dim": ess_stats["ess_per_dim"],
        "ess_norm_per_dim": ess_stats["ess_norm_per_dim"],
        "iact_per_dim": ess_stats["iact_per_dim"],
        "acf_per_dim": ess_stats["acf_per_dim"],
        "chains_array": chain_stack,
        "energies": energies,
        "wall_time_sec": float(wall_time_sec) if wall_time_sec is not None else float("nan"),
    }


def sample_latents(
    model: Any,
    cfg: SamplerConfig,
    n_samples: int,
    mcmc_steps: int,
) -> tuple[torch.Tensor, float]:
    sampler = sampler_from_config(model, cfg, mcmc_steps=mcmc_steps)
    use_jitter = cfg.eps_jitter > 0.0 or cfg.n_lf_jitter > 0
    if use_jitter:
        try:
            samples = sampler.sample(
                n_samples,
                eps_jitter=float(cfg.eps_jitter),
                n_lf_jitter=int(cfg.n_lf_jitter),
            )
        except TypeError:
            samples = sampler.sample(n_samples)
    else:
        samples = sampler.sample(n_samples)
    acceptance = float(getattr(sampler, "last_acceptance_rate", float("nan")))
    return samples, acceptance


def compute_local_coverage(samples: torch.Tensor, centroids: torch.Tensor) -> float:
    """
    A centroid is covered if at least one sample is within its local NN radius.
    """
    d_sc = torch.cdist(centroids, samples)
    min_sample_dist = d_sc.min(dim=1).values
    d_cc = torch.cdist(centroids, centroids)
    eye = torch.eye(d_cc.shape[0], device=d_cc.device, dtype=torch.bool)
    d_cc = d_cc.masked_fill(eye, float("inf"))
    local_radius = d_cc.min(dim=1).values
    covered = (min_sample_dist <= local_radius).float().mean().item()
    return float(covered)


def _make_ring_starts(
    centroids: torch.Tensor,
    n_points: int,
    radius: float,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=centroids.device)
    generator.manual_seed(int(seed))

    latent_dim = centroids.shape[1]
    center = centroids.mean(dim=0)
    phase = float(torch.rand(1, generator=generator, device=centroids.device).item()) * (2.0 * math.pi)
    theta = torch.linspace(0.0, 2.0 * math.pi, n_points + 1, device=centroids.device)[:-1] + phase
    starts = center.unsqueeze(0).repeat(n_points, 1)
    starts[:, 0] = center[0] + radius * torch.cos(theta)
    if latent_dim > 1:
        starts[:, 1] = center[1] + radius * torch.sin(theta)
    return starts


def _make_gaussian_starts(
    latent_dim: int,
    n_points: int,
    std: float,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(n_points, latent_dim, generator=generator, device=device) * float(std)


def run_rescue_assessment(
    model: Any,
    centroids: torch.Tensor,
    cfg: SamplerConfig,
    horizon: int,
    ring_starts: int,
    gaussian_starts: int,
    seed: int,
) -> dict[str, Any]:
    center = centroids.mean(dim=0, keepdim=True)
    radius = float(2.5 * torch.linalg.norm(centroids - center, dim=1).max().item())
    ring = _make_ring_starts(centroids, ring_starts, radius, seed=seed)
    gauss = _make_gaussian_starts(
        latent_dim=centroids.shape[1],
        n_points=gaussian_starts,
        std=2.0,
        device=centroids.device,
        seed=seed + 17,
    )
    starts = torch.cat([ring, gauss], dim=0)

    all_min_dists: list[np.ndarray] = []
    all_points: list[torch.Tensor] = []
    for idx in range(starts.shape[0]):
        sampler = sampler_from_config(model, cfg, mcmc_steps=horizon)
        start = starts[idx : idx + 1].clone()
        chain, _ = run_chain_with_energy(
            start,
            sampler,
            horizon,
            cfg.n_lf,
            cfg.eps_lf,
            eps_jitter=cfg.eps_jitter,
            n_lf_jitter=cfg.n_lf_jitter,
        )
        points = torch.as_tensor(chain, dtype=torch.float32, device=centroids.device)
        with torch.no_grad():
            min_d = torch.cdist(points, centroids).min(dim=1).values
        all_min_dists.append(min_d.detach().cpu().numpy())
        all_points.append(points)

    dists = np.stack(all_min_dists, axis=0)
    r0 = model_r0(model, device=centroids.device)
    summary = summarize_rescue_from_distances(dists, threshold=r0, horizon=horizon)

    hit_steps = np.asarray(summary["hit_steps"], dtype=int)
    rescue_curve = [(hit_steps >= 0) & (hit_steps <= t) for t in range(horizon + 1)]
    rescue_curve = np.mean(np.stack(rescue_curve, axis=0), axis=1).tolist()

    points = torch.cat(all_points, dim=0)
    pa = compute_plateau_alignment_metrics(model, points, centroids, r0=r0, grad_threshold=1e-3)
    tan = compute_tangent_alignment_metric(
        model,
        centroids,
        r0=r0,
        n_samples=1024,
        k_neighbors=8,
        band_min=0.6,
        band_max=1.2,
        seed=seed + 53,
    )
    border = compute_border_tunnel_metrics(
        model,
        points=points,
        centroids=centroids,
        r0=r0,
    )

    return {
        "rescue_rate": float(summary["rescue_rate"]),
        "median_steps_to_manifold": float(summary["median_steps_to_hit"]),
        "plateau_fraction": float(pa["plateau_fraction"]),
        "alignment_mean": float(pa["alignment_mean"]),
        "rescue_directionality_mean": float(pa["rescue_directionality_mean"]),
        "tangent_alignment_mean": float(tan["tangent_alignment_mean"]),
        "border_overshoot_index": float(border["border_overshoot_index"]),
        "logdet_manifold_median": float(border["logdet_manifold_median"]),
        "logdet_transition_median": float(border["logdet_transition_median"]),
        "logdet_far_void_median": float(border["logdet_far_void_median"]),
        "tunnel_neff_transition_median": float(border["tunnel_neff_transition_median"]),
        "tunnel_neff_far_void_median": float(border["tunnel_neff_far_void_median"]),
        "rescue_curve": rescue_curve,
        "r0_model": float(r0),
        "radius": float(radius),
    }


def compute_diversity(samples: torch.Tensor, max_points: int = 1000) -> float:
    if samples.shape[0] < 2:
        return 0.0
    use = samples[: min(max_points, samples.shape[0])]
    d = torch.cdist(use, use)
    triu = torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)
    return float(d[triu].mean().item())


def prepare_quality_cache(model: Any, real_data: torch.Tensor | None, device: torch.device) -> dict[str, Any]:
    cache: dict[str, Any] = {"real_data": real_data, "real_latents": None}
    if real_data is None:
        return cache
    with torch.no_grad():
        flat = real_data.view(real_data.shape[0], -1).to(device)
        enc = model.encoder(flat)
        cache["real_latents"] = enc.embedding.detach()
    return cache


def compute_generation_metrics(
    model: Any,
    samples: torch.Tensor,
    quality_cache: dict[str, Any],
    device: torch.device,
    run_quality: bool,
    run_fid: bool,
    quality_samples: int,
    fid_samples: int,
) -> dict[str, float]:
    out: dict[str, float] = {
        "quality_precision": float("nan"),
        "quality_recall": float("nan"),
        "quality_diversity": float("nan"),
        "fid": float("nan"),
    }
    real_data = quality_cache.get("real_data")
    real_latents = quality_cache.get("real_latents")

    if run_quality and real_latents is not None:
        gen = samples[: min(quality_samples, samples.shape[0])]
        real = real_latents[: min(quality_samples, real_latents.shape[0])]
        precision, recall = compute_precision_recall(real, gen)
        out["quality_precision"] = float(precision)
        out["quality_recall"] = float(recall)
        out["quality_diversity"] = float(compute_diversity(gen))

    if run_fid and real_data is not None:
        gen_fid = samples[: min(fid_samples, samples.shape[0])]
        with torch.no_grad():
            decoded = decode_latents(model, gen_fid)
        fid = compute_fid_score(real_data[: decoded.shape[0]].to(device), decoded.to(device), device)
        out["fid"] = float(fid)

    return out


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(max(1, n_boot)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(
    rows: list[dict[str, Any]],
    protocols: list[str],
    bootstrap_samples: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for protocol in protocols:
        summary[protocol] = {}
        for model_key in MODEL_KEYS:
            subset = [r for r in rows if r["protocol"] == protocol and r["model_key"] == model_key]
            model_summary: dict[str, Any] = {"n_rows": len(subset), "metrics": {}}
            if subset:
                model_summary["sampler_config"] = {
                    "sampler_name": subset[0]["sampler_name"],
                    "exact": subset[0]["exact"],
                    "volume_power": subset[0]["volume_power"],
                    "effective_volume_exponent": subset[0].get("effective_volume_exponent"),
                    "n_lf": subset[0]["n_lf"],
                    "eps_lf": subset[0]["eps_lf"],
                    "momentum_persist": subset[0]["momentum_persist"],
                    "eps_jitter": subset[0]["eps_jitter"],
                    "n_lf_jitter": subset[0]["n_lf_jitter"],
                    "fp_steps": subset[0]["fp_steps"],
                    "fp_damping": subset[0]["fp_damping"],
                    "radial_prior_weight": subset[0]["radial_prior_weight"],
                    "hybrid_explore_steps": subset[0]["hybrid_explore_steps"],
                    "hybrid_rescue_steps": subset[0]["hybrid_rescue_steps"],
                }
            for key in SUMMARY_METRIC_KEYS:
                vals = [float(r[key]) for r in subset if isinstance(r.get(key), (float, int)) and np.isfinite(float(r[key]))]
                if not vals:
                    model_summary["metrics"][key] = None
                    continue
                ci_low, ci_high = bootstrap_ci(vals, n_boot=bootstrap_samples, seed=123)
                model_summary["metrics"][key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "n": len(vals),
                }
            summary[protocol][model_key] = model_summary
    return summary


def _median_iqr(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    return med, float(q75 - q25)


def summarize_candidate_ranking(
    rows: list[dict[str, Any]],
    protocol: str,
    model_key: str = "aniso",
) -> dict[str, Any]:
    subset = [r for r in rows if r.get("protocol") == protocol and r.get("model_key") == model_key]
    if not subset:
        return {
            "protocol": protocol,
            "model_key": model_key,
            "n_seeds": 0,
            "hard_reject": True,
            "reject_reasons": ["no_rows"],
            "primary_green_count": 0,
            "primary_n_total": len(RANK_PRIMARY_KEYS),
            "tiebreak_tangent_alignment_mean": float("nan"),
            "tiebreak_rescue_rate": float("nan"),
            "tiebreak_coverage_local": float("nan"),
            "iqr_noise_penalty": float("nan"),
            "metrics": {},
            "statuses": {},
        }

    metric_keys = sorted(set(RANK_STABILITY_KEYS) | set(RANK_PRIMARY_KEYS) | set(RANK_TIE_BREAKER_KEYS))
    metrics: dict[str, dict[str, float]] = {}
    medians: dict[str, float] = {}
    for key in metric_keys:
        vals = [
            float(r[key])
            for r in subset
            if isinstance(r.get(key), (float, int)) and np.isfinite(float(r[key]))
        ]
        med, iqr = _median_iqr(vals)
        metrics[key] = {
            "median": med,
            "iqr": iqr,
            "n": float(len(vals)),
        }
        medians[key] = med

    statuses = evaluate_absolute(medians)
    reject_reasons = [key for key in RANK_STABILITY_KEYS if statuses.get(key) == "red"]
    hard_reject = len(reject_reasons) > 0

    primary_green = sum(1 for key in RANK_PRIMARY_KEYS if statuses.get(key) == "green")
    primary_iqrs = [metrics[key]["iqr"] for key in RANK_PRIMARY_KEYS if np.isfinite(metrics[key]["iqr"])]
    iqr_noise = float(np.mean(primary_iqrs)) if primary_iqrs else float("nan")

    return {
        "protocol": protocol,
        "model_key": model_key,
        "n_seeds": len(subset),
        "hard_reject": hard_reject,
        "reject_reasons": reject_reasons,
        "primary_green_count": int(primary_green),
        "primary_n_total": len(RANK_PRIMARY_KEYS),
        "tiebreak_tangent_alignment_mean": float(medians.get("tangent_alignment_mean", float("nan"))),
        "tiebreak_rescue_rate": float(medians.get("rescue_rate", float("nan"))),
        "tiebreak_coverage_local": float(medians.get("coverage_local", float("nan"))),
        "iqr_noise_penalty": float(iqr_noise),
        "metrics": metrics,
        "statuses": {key: statuses.get(key) for key in metric_keys},
    }


def _finite_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _rank_medians(rank: dict[str, Any]) -> dict[str, float]:
    metrics = rank.get("metrics", {})
    out: dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out
    for key, payload in metrics.items():
        if not isinstance(payload, dict):
            continue
        value = payload.get("median")
        if isinstance(value, (int, float, np.floating, np.integer)):
            out[str(key)] = float(value)
    return out


def build_sweep_objectives(
    ranking_payload: dict[str, dict[str, Any]],
    protocol: str = "matched",
    model_key: str = "aniso",
) -> dict[str, float]:
    rank = ranking_payload.get(protocol, {}).get(model_key, {})
    medians = _rank_medians(rank)
    statuses = rank.get("statuses", {})
    if not isinstance(statuses, dict):
        statuses = {}

    core4_values = [medians.get(key, float("nan")) for key in CORE4_OBJECTIVE_KEYS]
    core4_finite = all(np.isfinite(v) for v in core4_values)
    core4_margins = [
        score_margin(medians.get(key, float("nan")), ABSOLUTE_RULES_BY_NAME[key]) for key in CORE4_OBJECTIVE_KEYS
    ]
    if core4_finite:
        objective_core4 = _finite_mean(core4_margins)
    else:
        objective_core4 = -5.0
    core4_green_count = sum(1 for key in CORE4_OBJECTIVE_KEYS if statuses.get(key) == "green")

    stability_margins = [
        score_margin(medians.get(key, float("nan")), ABSOLUTE_RULES_BY_NAME[key]) for key in FULL_OBJECTIVE_STABILITY_KEYS
    ]
    primary_margins = [
        score_margin(medians.get(key, float("nan")), ABSOLUTE_RULES_BY_NAME[key]) for key in FULL_OBJECTIVE_PRIMARY_KEYS
    ]
    required_full_values = [medians.get(key, float("nan")) for key in (FULL_OBJECTIVE_STABILITY_KEYS + FULL_OBJECTIVE_PRIMARY_KEYS)]
    if all(np.isfinite(v) for v in required_full_values):
        objective_full = 2.0 * _finite_mean(stability_margins) + _finite_mean(primary_margins)
    else:
        objective_full = -5.0
    if any(statuses.get(key) == "red" for key in FULL_OBJECTIVE_STABILITY_KEYS):
        objective_full -= 5.0

    return {
        "objective_metric_core4_v1": float(objective_core4),
        "objective_full_kpi_v1": float(objective_full),
        "core4_green_count": float(core4_green_count),
        "core4_n_total": float(len(CORE4_OBJECTIVE_KEYS)),
    }


def build_ranking_wandb_scalars(
    ranking_payload: dict[str, dict[str, Any]],
    sweep_objectives: dict[str, float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for protocol, per_model in ranking_payload.items():
        if not isinstance(per_model, dict):
            continue
        for model_key, rank in per_model.items():
            if not isinstance(rank, dict):
                continue
            base = f"ranking/{protocol}/{model_key}"
            payload[f"{base}/primary_green_count"] = int(rank.get("primary_green_count", 0))
            payload[f"{base}/hard_reject"] = float(bool(rank.get("hard_reject", True)))
            iqr_penalty = _to_wandb_value(rank.get("iqr_noise_penalty"))
            if iqr_penalty is not None:
                payload[f"{base}/iqr_noise_penalty"] = iqr_penalty

    if sweep_objectives:
        val = _to_wandb_value(sweep_objectives.get("objective_metric_core4_v1"))
        if val is not None:
            payload["sweep/objective_metric_core4_v1"] = val
        val = _to_wandb_value(sweep_objectives.get("objective_full_kpi_v1"))
        if val is not None:
            payload["sweep/objective_full_kpi_v1"] = val
        val = _to_wandb_value(sweep_objectives.get("core4_green_count"))
        if val is not None:
            payload["sweep/core4_green_count"] = val
        val = _to_wandb_value(sweep_objectives.get("core4_n_total"))
        if val is not None:
            payload["sweep/core4_n_total"] = val

    return payload


def _mean_map(summary: dict[str, Any], protocol: str, model_key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    model = summary.get(protocol, {}).get(model_key, {})
    metrics = model.get("metrics", {})
    for key, value in metrics.items():
        if isinstance(value, dict) and "mean" in value:
            out[key] = float(value["mean"])
    return out


def build_scorecard(summary: dict[str, Any], protocols: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []

    for protocol in protocols:
        baseline = _mean_map(summary, protocol, "baseline")
        aniso = _mean_map(summary, protocol, "aniso")
        if not baseline or not aniso:
            continue
        baseline_abs = evaluate_absolute(baseline)
        aniso_abs = evaluate_absolute(aniso)
        rel = evaluate_relative_gains(baseline, aniso)

        for rule in ABSOLUTE_RULES:
            key = rule.name
            rows.append(
                {
                    "protocol": protocol,
                    "metric": key,
                    "baseline_value": baseline.get(key),
                    "aniso_value": aniso.get(key),
                    "baseline_status": baseline_abs.get(key),
                    "aniso_status": aniso_abs.get(key),
                    "relative_pass": None,
                    "relative_target": None,
                }
            )

        for key, passed in rel["checks"].items():
            target = {
                "ess_norm_min": ">= +20%",
                "coverage_local": ">= +0.10",
                "rescue_rate": ">= +0.15",
                "median_steps_to_manifold": "<= -20%",
            }[key]
            rows.append(
                {
                    "protocol": protocol,
                    "metric": f"relative_{key}",
                    "baseline_value": baseline.get(key),
                    "aniso_value": aniso.get(key),
                    "baseline_status": None,
                    "aniso_status": None,
                    "relative_pass": bool(passed),
                    "relative_target": target,
                }
            )

    claim = None
    if "strict" in protocols and "matched" in protocols:
        strict_b = _mean_map(summary, "strict", "baseline")
        strict_a = _mean_map(summary, "strict", "aniso")
        matched_b = _mean_map(summary, "matched", "baseline")
        matched_a = _mean_map(summary, "matched", "aniso")
        if strict_b and strict_a and matched_b and matched_a:
            claim = evaluate_claim(strict_b, strict_a, matched_b, matched_a)
            rows.append(
                {
                    "protocol": "overall",
                    "metric": "claim_validated",
                    "baseline_value": None,
                    "aniso_value": None,
                    "baseline_status": None,
                    "aniso_status": None,
                    "relative_pass": bool(claim["claim_validated"]),
                    "relative_target": "Aniso no red stability + >=3/4 gains in strict or matched",
                }
            )
    return rows, claim


def write_scorecard_md(path: Path, rows: list[dict[str, Any]], claim: dict[str, Any] | None) -> None:
    def _cell(value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    lines = ["# Metric Assessment Scorecard", ""]
    if claim is not None:
        lines.append(f"- `claim_validated`: **{bool(claim['claim_validated'])}**")
        lines.append("")
    lines.extend(
        [
            "| protocol | metric | baseline | aniso | baseline_status | aniso_status | relative_pass | target |",
            "|---|---|---:|---:|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {protocol} | {metric} | {baseline_value} | {aniso_value} | {baseline_status} | {aniso_status} | {relative_pass} | {relative_target} |".format(
                protocol=_cell(row.get("protocol", "")),
                metric=_cell(row.get("metric", "")),
                baseline_value=_cell(row.get("baseline_value", "")),
                aniso_value=_cell(row.get("aniso_value", "")),
                baseline_status=_cell(row.get("baseline_status", "")),
                aniso_status=_cell(row.get("aniso_status", "")),
                relative_pass=_cell(row.get("relative_pass", "")),
                relative_target=_cell(row.get("relative_target", "")),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def plot_chain_diagnostics(energy: dict[str, np.ndarray], out_path: Path, title: str) -> None:
    h = np.asarray(energy.get("hamiltonian", np.array([])), dtype=float)
    accept = np.asarray(energy.get("accept", np.array([])), dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    if h.size:
        ax.plot(np.arange(h.size), h, label="H", linewidth=1.4)
    if accept.size:
        step = np.arange(1, accept.size + 1)
        run_mean = np.cumsum(accept) / np.arange(1, accept.size + 1)
        ax.plot(step, run_mean, label="accept_rate_running", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_acf(acf_per_dim: list[list[float]], out_path: Path, max_dims: int = 2, max_lag: int = 60) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for dim in range(min(max_dims, len(acf_per_dim))):
        acf = np.asarray(acf_per_dim[dim], dtype=float)
        acf = acf[: max_lag + 1]
        ax.plot(np.arange(acf.size), acf, label=f"dim {dim}")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("ACF (post burn-in)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ess_bar(ess_norm_per_dim: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x = np.arange(len(ess_norm_per_dim))
    ax.bar(x, ess_norm_per_dim, color="steelblue", edgecolor="black")
    ax.set_title("ESS per dim (normalized to 1000)")
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("ESS/1000")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rescue_curve(curve: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(np.arange(len(curve)), curve, color="crimson", linewidth=1.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Rescue CDF (fraction rescued by step)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Rescue fraction")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_scatter(samples: torch.Tensor, centroids: torch.Tensor, out_path: Path) -> None:
    if samples.shape[1] < 2:
        return
    s = samples[:, :2].detach().cpu().numpy()
    c = centroids[:, :2].detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(c[:, 0], c[:, 1], c="gray", alpha=0.35, s=18, label="centroids")
    ax.scatter(s[:, 0], s[:, 1], c="royalblue", alpha=0.18, s=8, label="samples")
    ax.set_aspect("equal")
    ax.set_title("Centroid coverage view")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pilot_acceptance(
    model: Any,
    centroids: torch.Tensor,
    cfg: SamplerConfig,
    n_chains: int,
    chain_length: int,
    seed: int = 101,
) -> float:
    batch = run_chain_batch(model, centroids, cfg, n_chains=n_chains, chain_length=chain_length, seed=seed)
    vals = [v for v in batch["acceptance_rates"] if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def select_matched_eps(
    model: Any,
    centroids: torch.Tensor,
    sampler_name: str,
    n_lf: int,
    volume_power: float,
    momentum_persist: float,
    eps_jitter: float,
    n_lf_jitter: int,
    fp_steps: int,
    fp_damping: float,
    radial_prior_weight: float,
    mass_mode: str,
    adaptive_dual_step: bool,
    adaptive_max_dual_displacement: float,
    adaptive_min_step_scale: float,
    dynamic_jitter_scale: float,
    hybrid_explore_steps: int,
    hybrid_rescue_steps: int,
    eps_grid: list[float],
    target_acceptance: float,
    pilot_n_chains: int,
    pilot_chain_length: int,
    on_trial: Callable[[dict[str, float]], None] | None = None,
) -> float:
    best_eps = eps_grid[0]
    best_delta = float("inf")
    for eps in eps_grid:
        cfg = SamplerConfig(
            sampler_name=sampler_name,
            n_lf=n_lf,
            eps_lf=float(eps),
            volume_power=volume_power,
            exact=True,
            momentum_persist=float(momentum_persist),
            eps_jitter=float(eps_jitter),
            n_lf_jitter=int(n_lf_jitter),
            fp_steps=fp_steps,
            fp_damping=fp_damping,
            radial_prior_weight=float(radial_prior_weight),
            mass_mode=str(mass_mode),
            adaptive_dual_step=bool(adaptive_dual_step),
            adaptive_max_dual_displacement=float(adaptive_max_dual_displacement),
            adaptive_min_step_scale=float(adaptive_min_step_scale),
            dynamic_jitter_scale=float(dynamic_jitter_scale),
            hybrid_explore_steps=int(hybrid_explore_steps),
            hybrid_rescue_steps=int(hybrid_rescue_steps),
        )
        acc = pilot_acceptance(
            model,
            centroids,
            cfg,
            n_chains=pilot_n_chains,
            chain_length=pilot_chain_length,
            seed=101,
        )
        delta = abs(acc - target_acceptance)
        if on_trial is not None:
            on_trial(
                {
                    "eps_lf": float(eps),
                    "acceptance": float(acc),
                    "delta_to_target": float(delta),
                }
            )
        if delta < best_delta:
            best_delta = delta
            best_eps = float(eps)
    return float(best_eps)


def select_tuned_config(
    model: Any,
    centroids: torch.Tensor,
    n_lf_grid: list[int],
    eps_grid: list[float],
    volume_power_grid: list[float],
    sampler_name: str,
    momentum_persist: float,
    eps_jitter: float,
    n_lf_jitter: int,
    fp_steps: int,
    fp_damping: float,
    radial_prior_weight: float,
    mass_mode: str,
    adaptive_dual_step: bool,
    adaptive_max_dual_displacement: float,
    adaptive_min_step_scale: float,
    dynamic_jitter_scale: float,
    hybrid_explore_steps: int,
    hybrid_rescue_steps: int,
    target_acceptance: float,
    pilot_n_chains: int,
    pilot_chain_length: int,
    burn_in: int,
    on_trial: Callable[[dict[str, float]], None] | None = None,
) -> SamplerConfig:
    best_cfg = SamplerConfig(
        sampler_name=sampler_name,
        momentum_persist=float(momentum_persist),
        eps_jitter=float(eps_jitter),
        n_lf_jitter=int(n_lf_jitter),
        fp_steps=int(fp_steps),
        fp_damping=float(fp_damping),
        radial_prior_weight=float(radial_prior_weight),
        mass_mode=str(mass_mode),
        adaptive_dual_step=bool(adaptive_dual_step),
        adaptive_max_dual_displacement=float(adaptive_max_dual_displacement),
        adaptive_min_step_scale=float(adaptive_min_step_scale),
        dynamic_jitter_scale=float(dynamic_jitter_scale),
        hybrid_explore_steps=int(hybrid_explore_steps),
        hybrid_rescue_steps=int(hybrid_rescue_steps),
    )
    best_score = -float("inf")
    for n_lf in n_lf_grid:
        for eps in eps_grid:
            for vp in volume_power_grid:
                cfg = SamplerConfig(
                    sampler_name=sampler_name,
                    n_lf=int(n_lf),
                    eps_lf=float(eps),
                    volume_power=float(vp),
                    exact=True,
                    momentum_persist=float(momentum_persist),
                    eps_jitter=float(eps_jitter),
                    n_lf_jitter=int(n_lf_jitter),
                    fp_steps=fp_steps,
                    fp_damping=fp_damping,
                    radial_prior_weight=float(radial_prior_weight),
                    mass_mode=str(mass_mode),
                    adaptive_dual_step=bool(adaptive_dual_step),
                    adaptive_max_dual_displacement=float(adaptive_max_dual_displacement),
                    adaptive_min_step_scale=float(adaptive_min_step_scale),
                    dynamic_jitter_scale=float(dynamic_jitter_scale),
                    hybrid_explore_steps=int(hybrid_explore_steps),
                    hybrid_rescue_steps=int(hybrid_rescue_steps),
                )
                batch = run_chain_batch(
                    model,
                    centroids,
                    cfg,
                    n_chains=pilot_n_chains,
                    chain_length=pilot_chain_length,
                    seed=101,
                )
                mix = evaluate_mixing_stability(
                    batch["chains"],
                    batch["energies"],
                    burn_in=min(burn_in, pilot_chain_length // 4),
                    wall_time_sec=float(batch["wall_time_sec"]),
                )
                acc = mix["acceptance_mean"]
                dh = mix["dh_p95_abs"]
                ess = mix["ess_norm_min"]
                # Composite score (pilot): acceptance targeting + stability + mixing.
                score = -abs(acc - target_acceptance) - 0.05 * dh + 0.005 * ess
                if on_trial is not None:
                    on_trial(
                        {
                            "n_lf": float(n_lf),
                            "eps_lf": float(eps),
                            "volume_power": float(vp),
                            "acceptance": float(acc),
                            "dh_p95_abs": float(dh),
                            "ess_norm_min": float(ess),
                            "score": float(score),
                        }
                    )
                if score > best_score:
                    best_score = score
                    best_cfg = cfg
    return best_cfg


def _override_if_default(args: argparse.Namespace, key: str, value: int) -> None:
    if not hasattr(args, key):
        return
    if getattr(args, key) == DEFAULT_NUMERIC_ARGS.get(key):
        setattr(args, key, value)


def apply_profile_overrides(args: argparse.Namespace) -> None:
    preset = PROFILE_PRESETS.get(args.profile, {})
    for key, value in preset.items():
        _override_if_default(args, key, value)


def maybe_apply_quick_mode(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    for key, value in PROFILE_PRESETS["smoke"].items():
        if hasattr(args, key):
            setattr(args, key, min(getattr(args, key), value))


def maybe_apply_cpu_fallback(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cpu":
        return
    args.chain_length = min(args.chain_length, 250)
    args.burn_in = min(args.burn_in, 60)
    args.coverage_samples = min(args.coverage_samples, 1000)
    args.rescue_ring_starts = min(args.rescue_ring_starts, 16)
    args.rescue_gaussian_starts = min(args.rescue_gaussian_starts, 16)
    args.rescue_horizon = min(args.rescue_horizon, 60)
    args.quality_samples = min(args.quality_samples, 400)
    args.fid_samples = min(args.fid_samples, 400)
    args.pilot_chain_length = min(args.pilot_chain_length, 100)


def _latest_subdir(path: Path) -> Path | None:
    if not path.exists():
        return None
    candidates = [p for p in path.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_epoch(path: Path) -> int:
    match = re.search(r"_epoch_(\d+)\.png$", path.name)
    return int(match.group(1)) if match else -1


def _latest_epoch_plot(model_dir: Path, stem: str) -> Path | None:
    candidates = list(model_dir.glob(f"{stem}_epoch_*.png"))
    if not candidates:
        return None
    return max(candidates, key=_extract_epoch)


def _copy_if_exists(src: Path | None, dst: Path) -> Path | None:
    if src is None or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def collect_reference_visuals(model_key: str, model_dir: Path, out_base: Path) -> list[Path]:
    out_dir = out_base / model_key
    copied: list[Path] = []

    # Training visuals
    for logical_name, pattern in REFERENCE_VISUAL_SPECS["training"]:
        src = None
        if "{latest}" in pattern:
            stem = pattern.split("_epoch_")[0]
            src = _latest_epoch_plot(model_dir, stem)
        else:
            candidate = model_dir / pattern
            if candidate.exists():
                src = candidate
        dst = out_dir / "training" / f"{logical_name}.png"
        copied_path = _copy_if_exists(src, dst)
        if copied_path is not None:
            copied.append(copied_path)

    # Analysis visuals (latest subdir if available)
    analysis_root = model_dir / "analysis_results"
    analysis_dir = _latest_subdir(analysis_root) or (analysis_root if analysis_root.exists() else None)
    if analysis_dir is not None:
        for logical_name, filename in REFERENCE_VISUAL_SPECS["analysis"]:
            src = analysis_dir / filename
            dst = out_dir / "analysis" / f"{logical_name}.png"
            copied_path = _copy_if_exists(src if src.exists() else None, dst)
            if copied_path is not None:
                copied.append(copied_path)

    # Sampling visuals (latest subdir if available)
    sampling_root = model_dir / "sampling_diagnostics"
    sampling_dir = _latest_subdir(sampling_root) or (sampling_root if sampling_root.exists() else None)
    if sampling_dir is not None:
        for logical_name, filename in REFERENCE_VISUAL_SPECS["sampling"]:
            src = sampling_dir / filename
            dst = out_dir / "sampling" / f"{logical_name}.png"
            copied_path = _copy_if_exists(src if src.exists() else None, dst)
            if copied_path is not None:
                copied.append(copied_path)

    return copied


def log_image_paths_to_wandb(
    wandb_run: Any | None,
    image_paths: list[Path],
    key_prefix: str,
    max_images: int,
) -> None:
    if wandb_run is None or wandb is None or not image_paths:
        return
    limit = len(image_paths) if max_images <= 0 else min(len(image_paths), max_images)
    for image_path in image_paths[:limit]:
        rel_name = "_".join(image_path.parts[-4:]).replace(".png", "")
        key = f"{key_prefix}/{rel_name}"
        try:
            wandb_run.log({key: wandb.Image(str(image_path))})
        except Exception as exc:
            print(f"[warning] WandB image logging failed for {image_path}: {exc}")
            break


def build_reference_comparison_panels(reference_dir: Path, out_dir: Path) -> list[Path]:
    panels: list[Path] = []
    baseline_root = reference_dir / "baseline"
    aniso_root = reference_dir / "aniso"
    if not baseline_root.exists() or not aniso_root.exists():
        return panels

    for category in ("training", "analysis", "sampling"):
        left_cat = baseline_root / category
        right_cat = aniso_root / category
        if not left_cat.exists() or not right_cat.exists():
            continue
        shared_names = sorted(
            {
                p.name
                for p in left_cat.glob("*.png")
                if (right_cat / p.name).exists()
            }
        )
        if not shared_names:
            continue
        fig, axes = plt.subplots(len(shared_names), 2, figsize=(14, 4.5 * len(shared_names)))
        if len(shared_names) == 1:
            axes = np.array([axes])
        for idx, name in enumerate(shared_names):
            left = plt.imread(str(left_cat / name))
            right = plt.imread(str(right_cat / name))
            ax_l, ax_r = axes[idx]
            ax_l.imshow(left)
            ax_r.imshow(right)
            ax_l.axis("off")
            ax_r.axis("off")
            if idx == 0:
                ax_l.set_title("Baseline")
                ax_r.set_title("AnisoRHVAE")
            ax_l.set_ylabel(name.replace(".png", ""), rotation=90, fontsize=9)
        fig.suptitle(f"{category.title()} Comparison", fontsize=14)
        fig.tight_layout()
        out_path = out_dir / f"reference_{category}_panel.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        panels.append(out_path)

    return panels


def extract_training_summary(model_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    history_path = model_dir / "training_history.pt"
    if not history_path.exists():
        return out
    try:
        history = torch.load(history_path, map_location="cpu")
    except Exception:
        return out
    if not isinstance(history, dict):
        return out
    for key in ("train_loss", "train_recon", "val_loss", "val_recon"):
        values = history.get(key)
        if isinstance(values, (list, tuple)) and values:
            arr = np.asarray(values, dtype=float)
            if arr.size and np.isfinite(arr).any():
                out[f"{key}_last"] = float(arr[-1])
                out[f"{key}_best"] = float(np.nanmin(arr))
    out["epochs"] = float(max(len(history.get("train_loss", [])), len(history.get("val_loss", []))))
    return out


def extract_sampling_summary(model_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    sampling_root = model_dir / "sampling_diagnostics"
    sampling_dir = _latest_subdir(sampling_root) or (sampling_root if sampling_root.exists() else None)
    if sampling_dir is None:
        return out
    summary_path = sampling_dir / "metrics_summary.json"
    if not summary_path.exists():
        return out
    try:
        payload = json.loads(summary_path.read_text())
    except Exception:
        return out
    rhmc = payload.get("rhmc", {})
    quality = payload.get("quality", {})
    fid = payload.get("fid", {})
    if isinstance(rhmc, dict) and "mean_acceptance" in rhmc:
        out["sampling_rhmc_mean_acceptance"] = float(rhmc["mean_acceptance"])
    if isinstance(quality, dict):
        for sampler in ("volume", "gaussian"):
            vals = quality.get(sampler, {})
            if isinstance(vals, dict):
                for metric in ("precision", "recall", "coverage", "diversity"):
                    if metric in vals:
                        out[f"sampling_quality_{sampler}_{metric}"] = float(vals[metric])
    if isinstance(fid, dict):
        for sampler in ("volume", "gaussian"):
            vals = fid.get(sampler, {})
            if isinstance(vals, dict) and "fid" in vals:
                out[f"sampling_fid_{sampler}"] = float(vals["fid"])
    return out


def build_model_bilan_rows(model_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key, model_dir in model_dirs.items():
        row: dict[str, Any] = {
            "model_key": model_key,
            "model_path": str(model_dir),
        }
        row.update(extract_training_summary(model_dir))
        row.update(extract_sampling_summary(model_dir))
        rows.append(row)
    return rows


def evaluate_model_protocol_seed(
    model_key: str,
    model_dir: Path,
    model: Any,
    centroids: torch.Tensor,
    sampler_cfg: SamplerConfig,
    seed: int,
    args: argparse.Namespace,
    quality_cache: dict[str, Any],
    plots_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    mix_batch = run_chain_batch(
        model,
        centroids,
        sampler_cfg,
        n_chains=args.n_chains,
        chain_length=args.chain_length,
        seed=seed + 11,
    )
    mix_metrics = evaluate_mixing_stability(
        mix_batch["chains"],
        mix_batch["energies"],
        burn_in=args.burn_in,
        wall_time_sec=float(mix_batch["wall_time_sec"]),
    )

    n_samples_needed = max(
        args.coverage_samples,
        args.quality_samples if args.run_quality else 0,
        args.fid_samples if args.run_fid else 0,
    )
    n_samples_needed = max(n_samples_needed, 1)
    set_seed(seed + 23)
    sampled, sample_acc = sample_latents(
        model,
        sampler_cfg,
        n_samples=n_samples_needed,
        mcmc_steps=args.sampling_mcmc_steps,
    )
    coverage = compute_local_coverage(sampled[: args.coverage_samples], centroids)

    rescue = run_rescue_assessment(
        model,
        centroids,
        sampler_cfg,
        horizon=args.rescue_horizon,
        ring_starts=args.rescue_ring_starts,
        gaussian_starts=args.rescue_gaussian_starts,
        seed=seed + 37,
    )

    gen_metrics = compute_generation_metrics(
        model,
        sampled,
        quality_cache=quality_cache,
        device=centroids.device,
        run_quality=args.run_quality,
        run_fid=args.run_fid,
        quality_samples=args.quality_samples,
        fid_samples=args.fid_samples,
    )

    seed_plot_paths: list[Path] = []
    if args.save_plots and mix_batch["energies"]:
        model_plot_dir = plots_dir / args.current_protocol / model_key
        energy_path = model_plot_dir / f"chain_energy_accept_seed{seed}.png"
        plot_chain_diagnostics(
            mix_batch["energies"][0],
            energy_path,
            title=f"{model_key} / {args.current_protocol} / seed {seed}",
        )
        seed_plot_paths.append(energy_path)
        acf_path = model_plot_dir / f"acf_seed{seed}.png"
        plot_acf(
            mix_metrics["acf_per_dim"],
            acf_path,
        )
        seed_plot_paths.append(acf_path)
        ess_path = model_plot_dir / f"ess_seed{seed}.png"
        plot_ess_bar(
            mix_metrics["ess_norm_per_dim"],
            ess_path,
        )
        seed_plot_paths.append(ess_path)
        rescue_path = model_plot_dir / f"rescue_curve_seed{seed}.png"
        plot_rescue_curve(
            rescue["rescue_curve"],
            rescue_path,
        )
        seed_plot_paths.append(rescue_path)
        coverage_path = model_plot_dir / f"coverage_seed{seed}.png"
        plot_coverage_scatter(
            sampled[: args.coverage_samples],
            centroids,
            coverage_path,
        )
        seed_plot_paths.append(coverage_path)

    row = {
        "model_key": model_key,
        "model_path": str(model_dir),
        "protocol": args.current_protocol,
        "seed": int(seed),
        "sampler_name": sampler_cfg.sampler_name,
        "exact": bool(sampler_cfg.exact),
        "volume_power": float(sampler_cfg.volume_power),
        "effective_volume_exponent": float(
            effective_volume_exponent(sampler_cfg.sampler_name, sampler_cfg.volume_power)
        ),
        "n_lf": int(sampler_cfg.n_lf),
        "eps_lf": float(sampler_cfg.eps_lf),
        "momentum_persist": float(sampler_cfg.momentum_persist),
        "eps_jitter": float(sampler_cfg.eps_jitter),
        "n_lf_jitter": int(sampler_cfg.n_lf_jitter),
        "fp_steps": int(sampler_cfg.fp_steps),
        "fp_damping": float(sampler_cfg.fp_damping),
        "radial_prior_weight": float(sampler_cfg.radial_prior_weight),
        "mass_mode": str(sampler_cfg.mass_mode),
        "adaptive_dual_step": bool(sampler_cfg.adaptive_dual_step),
        "adaptive_max_dual_displacement": float(sampler_cfg.adaptive_max_dual_displacement),
        "adaptive_min_step_scale": float(sampler_cfg.adaptive_min_step_scale),
        "dynamic_jitter_scale": float(sampler_cfg.dynamic_jitter_scale),
        "hybrid_explore_steps": int(sampler_cfg.hybrid_explore_steps),
        "hybrid_rescue_steps": int(sampler_cfg.hybrid_rescue_steps),
        "n_chains": int(args.n_chains),
        "chain_length": int(args.chain_length),
        "burn_in": int(args.burn_in),
        "sampling_mcmc_steps": int(args.sampling_mcmc_steps),
        "coverage_samples": int(args.coverage_samples),
        "rescue_ring_starts": int(args.rescue_ring_starts),
        "rescue_gaussian_starts": int(args.rescue_gaussian_starts),
        "rescue_horizon": int(args.rescue_horizon),
        "acceptance_mean": float(mix_metrics["acceptance_mean"]),
        "dh_p95_abs": float(mix_metrics["dh_p95_abs"]),
        "h_drift_slope_abs_mean": float(mix_metrics["h_drift_slope_abs_mean"]),
        "ess_min": float(mix_metrics["ess_min"]),
        "ess_median": float(mix_metrics["ess_median"]),
        "ess_median_per_sec": float(mix_metrics["ess_median_per_sec"]),
        "ess_norm_min": float(mix_metrics["ess_norm_min"]),
        "iact_median": float(mix_metrics["iact_median"]),
        "mix_wall_time_sec": float(mix_metrics["wall_time_sec"]),
        "coverage_local": float(coverage),
        "rescue_rate": float(rescue["rescue_rate"]),
        "median_steps_to_manifold": float(rescue["median_steps_to_manifold"]),
        "plateau_fraction": float(rescue["plateau_fraction"]),
        "alignment_mean": float(rescue["alignment_mean"]),
        "rescue_directionality_mean": float(rescue["rescue_directionality_mean"]),
        "tangent_alignment_mean": float(rescue["tangent_alignment_mean"]),
        "border_overshoot_index": float(rescue["border_overshoot_index"]),
        "logdet_manifold_median": float(rescue["logdet_manifold_median"]),
        "logdet_transition_median": float(rescue["logdet_transition_median"]),
        "logdet_far_void_median": float(rescue["logdet_far_void_median"]),
        "tunnel_neff_transition_median": float(rescue["tunnel_neff_transition_median"]),
        "tunnel_neff_far_void_median": float(rescue["tunnel_neff_far_void_median"]),
        "r0_model": float(rescue["r0_model"]),
        "sample_acceptance_rate": float(sample_acc),
        "quality_precision": float(gen_metrics["quality_precision"]),
        "quality_recall": float(gen_metrics["quality_recall"]),
        "quality_diversity": float(gen_metrics["quality_diversity"]),
        "fid": float(gen_metrics["fid"]),
        "__seed_plot_paths": [str(path) for path in seed_plot_paths],
    }
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metric-centric assessment suite for baseline vs AnisoRHVAE.")
    parser.add_argument("--baseline_model_path", type=str, default=str(DEFAULT_BASELINE_MODEL))
    parser.add_argument("--ours_model_path", type=str, default=str(DEFAULT_ANISO_MODEL))
    parser.add_argument(
        "--aniso_only",
        action="store_true",
        help="Evaluate only ours_model_path (aniso) and skip baseline model evaluation.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=["full", "short", "smoke"],
        help="Preset budget profile. 'short' is recommended for iterative benchmark loops.",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["strict", "matched", "tuned"],
        choices=["strict", "matched", "tuned"],
    )
    parser.add_argument(
        "--protocol_profile",
        type=str,
        default=None,
        choices=["all", "strict", "matched", "tuned"],
        help="Convenience selector. When set, overrides --protocols.",
    )
    parser.add_argument("--sampling_seeds", nargs="+", type=int, default=[13, 29, 47])
    parser.add_argument(
        "--allow_two_seed_prefilter",
        action="store_true",
        help="Allow exactly 2 seeds for quick elimination. Final ranking should still use >=3 seeds.",
    )
    parser.add_argument("--output_dir", type=str, default="results/metric_assessment")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--sampler_name",
        type=str,
        default="volume_riemannian",
        choices=["volume", "volume_riemannian", "riemannian", "geodesic", "volume_det", "volume_riemannian_det", "dual_riemannian", "hybrid_volume_mix"],
        help="Sampler family used for strict protocol (and matched/tuned initialization).",
    )
    parser.add_argument(
        "--strict_volume_power",
        type=float,
        default=2.0,
        help="Volume exponent used in strict protocol.",
    )
    parser.add_argument(
        "--strict_n_lf",
        type=int,
        default=10,
        help="Leapfrog step count used by strict protocol (and matched/tuned pilot initialization).",
    )
    parser.add_argument(
        "--strict_eps_lf",
        type=float,
        default=0.05,
        help="Leapfrog step size used by strict protocol.",
    )
    parser.add_argument(
        "--strict_fp_steps",
        type=int,
        default=15,
        help="Fixed-point iterations used by Riemannian generalized leapfrog samplers.",
    )
    parser.add_argument(
        "--strict_fp_damping",
        type=float,
        default=0.72,
        help="Damping factor for fixed-point updates in Riemannian generalized leapfrog samplers.",
    )
    parser.add_argument(
        "--strict_momentum_persist",
        type=float,
        default=0.0,
        help="GHMC momentum persistence in [0, 1). Higher values can improve exploration.",
    )
    parser.add_argument(
        "--strict_eps_jitter",
        type=float,
        default=0.0,
        help="Relative jitter amplitude for leapfrog step size (0 disables jitter).",
    )
    parser.add_argument(
        "--strict_n_lf_jitter",
        type=int,
        default=0,
        help="Integer jitter applied to leapfrog count: local_n_lf in [n_lf-j, n_lf+j].",
    )
    parser.add_argument(
        "--strict_radial_prior_weight",
        type=float,
        default=0.1,
        help="Optional radial prior strength for volume_riemannian to improve manifold rescue.",
    )
    parser.add_argument(
        "--strict_mass_mode",
        type=str,
        default="standard",
        choices=["standard", "dual"],
        help="Mass convention for strict/matched/tuned volume_riemannian protocols.",
    )
    parser.add_argument(
        "--strict_use_dual_metric",
        type=str,
        nargs="?",
        const="True",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--strict_adaptive_dual_step",
        dest="strict_adaptive_dual_step",
        action="store_true",
        help="Enable dual-step local adaptation (caps displacement in dual regions).",
    )
    parser.add_argument(
        "--strict_no_adaptive_dual_step",
        dest="strict_adaptive_dual_step",
        action="store_false",
        help="Disable dual-step local adaptation.",
    )
    parser.set_defaults(strict_adaptive_dual_step=False)
    parser.add_argument(
        "--strict_adaptive_max_dual_displacement",
        type=float,
        default=0.75,
        help="Maximum allowed local displacement (in latent units) before scaling eps down.",
    )
    parser.add_argument(
        "--strict_adaptive_min_step_scale",
        type=float,
        default=0.05,
        help="Lower bound for local eps scaling in adaptive dual mode.",
    )
    parser.add_argument(
        "--strict_dynamic_jitter_scale",
        type=float,
        default=0.0,
        help="Trace-scaled covariance jitter used during momentum refresh in strict protocols.",
    )
    parser.add_argument(
        "--strict_hybrid_explore_steps",
        type=int,
        default=3,
        help="For hybrid sampler: number of `volume` transitions per cycle.",
    )
    parser.add_argument(
        "--strict_hybrid_rescue_steps",
        type=int,
        default=1,
        help="For hybrid sampler: number of `volume_riemannian` transitions per cycle.",
    )

    parser.add_argument("--target_acceptance", type=float, default=0.80)
    parser.add_argument("--n_chains", type=int, default=DEFAULT_NUMERIC_ARGS["n_chains"])
    parser.add_argument("--chain_length", type=int, default=DEFAULT_NUMERIC_ARGS["chain_length"])
    parser.add_argument("--burn_in", type=int, default=DEFAULT_NUMERIC_ARGS["burn_in"])
    parser.add_argument("--sampling_mcmc_steps", type=int, default=DEFAULT_NUMERIC_ARGS["sampling_mcmc_steps"])
    parser.add_argument("--coverage_samples", type=int, default=DEFAULT_NUMERIC_ARGS["coverage_samples"])
    parser.add_argument("--rescue_ring_starts", type=int, default=DEFAULT_NUMERIC_ARGS["rescue_ring_starts"])
    parser.add_argument("--rescue_gaussian_starts", type=int, default=DEFAULT_NUMERIC_ARGS["rescue_gaussian_starts"])
    parser.add_argument("--rescue_horizon", type=int, default=DEFAULT_NUMERIC_ARGS["rescue_horizon"])

    parser.add_argument("--run_quality", action="store_true")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--run_fid", action="store_true")
    parser.add_argument("--quality_samples", type=int, default=DEFAULT_NUMERIC_ARGS["quality_samples"])
    parser.add_argument("--fid_samples", type=int, default=DEFAULT_NUMERIC_ARGS["fid_samples"])

    parser.add_argument("--pilot_n_chains", type=int, default=DEFAULT_NUMERIC_ARGS["pilot_n_chains"])
    parser.add_argument("--pilot_chain_length", type=int, default=DEFAULT_NUMERIC_ARGS["pilot_chain_length"])
    parser.add_argument("--matched_eps_grid", nargs="+", type=float, default=[0.003, 0.005, 0.008, 0.010, 0.015, 0.020])
    parser.add_argument(
        "--skip_matched_pilot",
        action="store_true",
        help="Skip matched epsilon pilot selection and use the provided --matched_eps_grid value directly.",
    )
    parser.add_argument("--tuned_n_lf_grid", nargs="+", type=int, default=[20, 30, 40])
    parser.add_argument("--tuned_eps_grid", nargs="+", type=float, default=[0.003, 0.005, 0.008, 0.010, 0.015])
    parser.add_argument("--tuned_volume_power_grid", nargs="+", type=float, default=[0.5, 1.0, 1.5])

    parser.add_argument("--bootstrap_samples", type=int, default=DEFAULT_NUMERIC_ARGS["bootstrap_samples"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument(
        "--collect_reference_visuals",
        action="store_true",
        help="Collect and log existing training/analysis/sampling visuals from model folders.",
    )

    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--wandb_name_mode",
        type=str,
        default="auto",
        choices=["timestamp", "auto", "manual"],
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--wandb_max_plot_images",
        type=int,
        default=24,
        help="Maximum number of PNG plot files to log as W&B images (<=0 means all).",
    )

    args = parser.parse_args()
    if args.strict_use_dual_metric is not None:
        strict_dual_raw = str(args.strict_use_dual_metric).strip().lower()
        if strict_dual_raw in {"true", "1"}:
            args.strict_mass_mode = "dual"
        elif strict_dual_raw in {"false", "0"}:
            args.strict_mass_mode = "standard"
        else:
            parser.error("Deprecated --strict_use_dual_metric only accepts True/False or 1/0.")
    if args.protocol_profile:
        if args.protocol_profile == "all":
            args.protocols = ["strict", "matched", "tuned"]
        else:
            args.protocols = [args.protocol_profile]
    args.sampler_name = normalize_sampler_name(args.sampler_name)
    args.sampling_seeds = list(dict.fromkeys(int(s) for s in args.sampling_seeds))
    if len(args.sampling_seeds) <= 1:
        parser.error("Ranking requires multiple seeds; use at least 2 seeds (recommended: 13 29 47).")
    if len(args.sampling_seeds) == 2 and not args.allow_two_seed_prefilter:
        parser.error(
            "Two-seed runs are only allowed for prefilter mode. Set --allow_two_seed_prefilter or provide >=3 seeds."
        )
    if args.strict_n_lf <= 0:
        parser.error("--strict_n_lf must be > 0")
    if args.strict_eps_lf <= 0:
        parser.error("--strict_eps_lf must be > 0")
    if args.strict_fp_steps <= 0:
        parser.error("--strict_fp_steps must be > 0")
    if args.strict_fp_damping <= 0:
        parser.error("--strict_fp_damping must be > 0")
    if not (0.0 <= args.strict_momentum_persist < 1.0):
        parser.error("--strict_momentum_persist must satisfy 0 <= value < 1.")
    if not (0.0 <= args.strict_eps_jitter < 1.0):
        parser.error("--strict_eps_jitter must satisfy 0 <= value < 1.")
    if args.strict_n_lf_jitter < 0:
        parser.error("--strict_n_lf_jitter must be >= 0.")
    if args.strict_radial_prior_weight < 0.0:
        parser.error("--strict_radial_prior_weight must be >= 0.")
    if args.strict_adaptive_max_dual_displacement <= 0.0:
        parser.error("--strict_adaptive_max_dual_displacement must be > 0.")
    if not (0.0 < args.strict_adaptive_min_step_scale <= 1.0):
        parser.error("--strict_adaptive_min_step_scale must satisfy 0 < value <= 1.")
    if args.strict_hybrid_explore_steps <= 0:
        parser.error("--strict_hybrid_explore_steps must be > 0.")
    if args.strict_hybrid_rescue_steps <= 0:
        parser.error("--strict_hybrid_rescue_steps must be > 0.")
    if args.skip_matched_pilot and "matched" in args.protocols and len(args.matched_eps_grid) != 1:
        parser.error("--skip_matched_pilot requires exactly one value in --matched_eps_grid.")
    return args


def main() -> None:
    args = parse_args()
    args.run_fid = bool(args.run_fid and not args.skip_fid)
    apply_profile_overrides(args)
    maybe_apply_quick_mode(args)
    if args.quick and args.profile != "smoke":
        print("[info] --quick enabled; effective profile is smoke-like (strict cap).")
    if args.save_plots:
        print("[info] plotting enabled (--save_plots).")
    else:
        print("[info] plotting disabled (use --save_plots to save PNG outputs).")
    if args.profile != "full":
        print(f"[info] profile='{args.profile}' active for shorter benchmark runtime.")
    if len(args.sampling_seeds) == 2:
        print("[warning] Running 2-seed prefilter mode. Final ranking should be rerun with 3 seeds.")
    if args.aniso_only:
        print("[info] aniso-only mode enabled: baseline model will be skipped.")
    if args.skip_matched_pilot and "matched" in args.protocols:
        print("[info] matched pilot selection disabled: using provided matched_eps_grid directly.")

    device = torch.device(args.device)
    maybe_apply_cpu_fallback(args, device)

    baseline_dir = Path(args.baseline_model_path)
    aniso_dir = Path(args.ours_model_path)
    model_dirs = {"aniso": aniso_dir} if args.aniso_only else {"baseline": baseline_dir, "aniso": aniso_dir}
    active_model_keys = tuple(model_dirs.keys())

    for model_dir in model_dirs.values():
        ensure_model_artifacts(model_dir)
    if args.run_quality or args.run_fid:
        # train_data.pt is optional in robustness mode; warn only.
        for model_dir in model_dirs.values():
            if not (model_dir / "train_data.pt").exists():
                print(f"[warning] {model_dir}/train_data.pt missing -> quality/FID may be skipped.")

    out_dir = build_output_dir(Path(args.output_dir))
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    write_provenance_snapshot(out_dir, args_payload=dict(vars(args)))

    if args.collect_reference_visuals or args.save_plots:
        reference_root = plots_dir / "reference"
        collected_images: list[Path] = []
        for model_key, model_path in model_dirs.items():
            collected_images.extend(collect_reference_visuals(model_key, model_path, reference_root))
        panel_images = build_reference_comparison_panels(reference_root, plots_dir)
        collected_images.extend(panel_images)
        print(f"[info] collected {len(collected_images)} reference visual(s) in {reference_root}.")
    else:
        collected_images = []

    wandb_run = init_wandb_run(args, run_stamp=out_dir.name)
    if wandb_run is not None:
        wandb_run.summary["output_dir"] = str(out_dir)
        if "baseline" in model_dirs:
            wandb_run.summary["baseline_model_path"] = str(model_dirs["baseline"])
        wandb_run.summary["aniso_model_path"] = str(model_dirs["aniso"])
        wandb_run.summary["aniso_only"] = bool(args.aniso_only)
        wandb_run.summary["skip_matched_pilot"] = bool(args.skip_matched_pilot)
        if collected_images:
            log_image_paths_to_wandb(
                wandb_run,
                collected_images,
                key_prefix="reference",
                max_images=max(0, int(args.wandb_max_plot_images)),
            )
    total_seed_jobs = len(args.protocols) * len(active_model_keys) * len(args.sampling_seeds)
    safe_wandb_log(
        wandb_run,
        {
            "progress/state": "started",
            "progress/total_seed_jobs": int(total_seed_jobs),
            "progress/completed_seed_jobs": 0,
            "progress/fraction": 0.0,
        },
    )

    try:
        models: dict[str, Any] = {}
        centroids_map: dict[str, torch.Tensor] = {}
        quality_cache: dict[str, dict[str, Any]] = {}

        for key in active_model_keys:
            model, centroids = load_model_and_centroids(model_dirs[key], device=device, aggressive_void=False)
            models[key] = model
            centroids_map[key] = centroids
            real_data = load_real_data_if_available(model_dirs[key])
            quality_cache[key] = prepare_quality_cache(model, real_data, device=device)

        # Build protocol configs (per model for matched/tuned).
        protocol_cfgs: dict[str, dict[str, SamplerConfig]] = {}
        strict_n_lf = int(args.strict_n_lf)
        if device.type == "cpu":
            strict_n_lf = min(strict_n_lf, 20)
        if args.quick:
            strict_n_lf = min(strict_n_lf, 10)
        strict_fp_steps = int(args.strict_fp_steps)
        if args.quick:
            strict_fp_steps = min(strict_fp_steps, 3)
        strict_fp_damping = float(args.strict_fp_damping)
        strict_cfg = SamplerConfig(
            sampler_name=args.sampler_name,
            exact=True,
            volume_power=float(args.strict_volume_power),
            n_lf=strict_n_lf,
            eps_lf=float(args.strict_eps_lf),
            momentum_persist=float(args.strict_momentum_persist),
            eps_jitter=float(args.strict_eps_jitter),
            n_lf_jitter=int(args.strict_n_lf_jitter),
            fp_steps=strict_fp_steps,
            fp_damping=strict_fp_damping,
            radial_prior_weight=float(args.strict_radial_prior_weight),
            mass_mode=str(args.strict_mass_mode),
            adaptive_dual_step=bool(args.strict_adaptive_dual_step),
            adaptive_max_dual_displacement=float(args.strict_adaptive_max_dual_displacement),
            adaptive_min_step_scale=float(args.strict_adaptive_min_step_scale),
            dynamic_jitter_scale=float(args.strict_dynamic_jitter_scale),
            hybrid_explore_steps=int(args.strict_hybrid_explore_steps),
            hybrid_rescue_steps=int(args.strict_hybrid_rescue_steps),
        )
        if "strict" in args.protocols:
            protocol_cfgs["strict"] = {key: strict_cfg for key in active_model_keys}

        if "matched" in args.protocols:
            protocol_cfgs["matched"] = {}
            if args.skip_matched_pilot:
                eps = float(args.matched_eps_grid[0])
                print(f"[pilot][matched] skipped -> using eps_lf={eps:.5f} for all active models", flush=True)
                for key in active_model_keys:
                    safe_wandb_log(
                        wandb_run,
                        {f"pilot/matched/{key}/selected_eps_lf": float(eps)},
                    )
                    protocol_cfgs["matched"][key] = SamplerConfig(
                        sampler_name=args.sampler_name,
                        exact=True,
                        volume_power=float(strict_cfg.volume_power),
                        n_lf=strict_cfg.n_lf,
                        eps_lf=eps,
                        momentum_persist=float(strict_cfg.momentum_persist),
                        eps_jitter=float(strict_cfg.eps_jitter),
                        n_lf_jitter=int(strict_cfg.n_lf_jitter),
                        fp_steps=strict_cfg.fp_steps,
                        fp_damping=strict_cfg.fp_damping,
                        radial_prior_weight=float(strict_cfg.radial_prior_weight),
                        mass_mode=str(strict_cfg.mass_mode),
                        adaptive_dual_step=bool(strict_cfg.adaptive_dual_step),
                        adaptive_max_dual_displacement=float(strict_cfg.adaptive_max_dual_displacement),
                        adaptive_min_step_scale=float(strict_cfg.adaptive_min_step_scale),
                        dynamic_jitter_scale=float(strict_cfg.dynamic_jitter_scale),
                        hybrid_explore_steps=int(strict_cfg.hybrid_explore_steps),
                        hybrid_rescue_steps=int(strict_cfg.hybrid_rescue_steps),
                    )
            else:
                for key in active_model_keys:
                    print(f"[pilot][matched] selecting eps for {key}...", flush=True)

                    def _matched_cb(payload: dict[str, float], model_key: str = key) -> None:
                        safe_wandb_log(
                            wandb_run,
                            {
                                f"pilot/matched/{model_key}/eps_lf": payload["eps_lf"],
                                f"pilot/matched/{model_key}/acceptance": payload["acceptance"],
                                f"pilot/matched/{model_key}/delta_to_target": payload["delta_to_target"],
                            },
                        )

                    eps = select_matched_eps(
                        models[key],
                        centroids_map[key],
                        sampler_name=args.sampler_name,
                        n_lf=strict_cfg.n_lf,
                        volume_power=float(strict_cfg.volume_power),
                        momentum_persist=float(strict_cfg.momentum_persist),
                        eps_jitter=float(strict_cfg.eps_jitter),
                        n_lf_jitter=int(strict_cfg.n_lf_jitter),
                        fp_steps=strict_cfg.fp_steps,
                        fp_damping=strict_cfg.fp_damping,
                        radial_prior_weight=float(strict_cfg.radial_prior_weight),
                        mass_mode=str(strict_cfg.mass_mode),
                        adaptive_dual_step=bool(strict_cfg.adaptive_dual_step),
                        adaptive_max_dual_displacement=float(strict_cfg.adaptive_max_dual_displacement),
                        adaptive_min_step_scale=float(strict_cfg.adaptive_min_step_scale),
                        dynamic_jitter_scale=float(strict_cfg.dynamic_jitter_scale),
                        hybrid_explore_steps=int(strict_cfg.hybrid_explore_steps),
                        hybrid_rescue_steps=int(strict_cfg.hybrid_rescue_steps),
                        eps_grid=[float(v) for v in args.matched_eps_grid],
                        target_acceptance=float(args.target_acceptance),
                        pilot_n_chains=args.pilot_n_chains,
                        pilot_chain_length=args.pilot_chain_length,
                        on_trial=_matched_cb,
                    )
                    print(f"[pilot][matched] {key} -> eps_lf={eps:.5f}", flush=True)
                    safe_wandb_log(
                        wandb_run,
                        {f"pilot/matched/{key}/selected_eps_lf": float(eps)},
                    )
                    protocol_cfgs["matched"][key] = SamplerConfig(
                        sampler_name=args.sampler_name,
                        exact=True,
                        volume_power=float(strict_cfg.volume_power),
                        n_lf=strict_cfg.n_lf,
                        eps_lf=eps,
                        momentum_persist=float(strict_cfg.momentum_persist),
                        eps_jitter=float(strict_cfg.eps_jitter),
                        n_lf_jitter=int(strict_cfg.n_lf_jitter),
                        fp_steps=strict_cfg.fp_steps,
                        fp_damping=strict_cfg.fp_damping,
                        radial_prior_weight=float(strict_cfg.radial_prior_weight),
                        mass_mode=str(strict_cfg.mass_mode),
                        adaptive_dual_step=bool(strict_cfg.adaptive_dual_step),
                        adaptive_max_dual_displacement=float(strict_cfg.adaptive_max_dual_displacement),
                        adaptive_min_step_scale=float(strict_cfg.adaptive_min_step_scale),
                        dynamic_jitter_scale=float(strict_cfg.dynamic_jitter_scale),
                        hybrid_explore_steps=int(strict_cfg.hybrid_explore_steps),
                        hybrid_rescue_steps=int(strict_cfg.hybrid_rescue_steps),
                    )

        if "tuned" in args.protocols:
            protocol_cfgs["tuned"] = {}
            n_lf_grid = [min(20, int(v)) for v in args.tuned_n_lf_grid] if device.type == "cpu" else [int(v) for v in args.tuned_n_lf_grid]
            for key in active_model_keys:
                print(f"[pilot][tuned] grid search for {key}...", flush=True)
                def _tuned_cb(payload: dict[str, float], model_key: str = key) -> None:
                    safe_wandb_log(
                        wandb_run,
                        {
                            f"pilot/tuned/{model_key}/n_lf": payload["n_lf"],
                            f"pilot/tuned/{model_key}/eps_lf": payload["eps_lf"],
                            f"pilot/tuned/{model_key}/volume_power": payload["volume_power"],
                            f"pilot/tuned/{model_key}/acceptance": payload["acceptance"],
                            f"pilot/tuned/{model_key}/dh_p95_abs": payload["dh_p95_abs"],
                            f"pilot/tuned/{model_key}/ess_norm_min": payload["ess_norm_min"],
                            f"pilot/tuned/{model_key}/score": payload["score"],
                        },
                    )
                cfg = select_tuned_config(
                    models[key],
                    centroids_map[key],
                    n_lf_grid=n_lf_grid,
                    eps_grid=[float(v) for v in args.tuned_eps_grid],
                    volume_power_grid=[float(v) for v in args.tuned_volume_power_grid],
                    sampler_name=args.sampler_name,
                    momentum_persist=float(strict_cfg.momentum_persist),
                    eps_jitter=float(strict_cfg.eps_jitter),
                    n_lf_jitter=int(strict_cfg.n_lf_jitter),
                    fp_steps=strict_cfg.fp_steps,
                    fp_damping=strict_cfg.fp_damping,
                    radial_prior_weight=float(strict_cfg.radial_prior_weight),
                    mass_mode=str(strict_cfg.mass_mode),
                    adaptive_dual_step=bool(strict_cfg.adaptive_dual_step),
                    adaptive_max_dual_displacement=float(strict_cfg.adaptive_max_dual_displacement),
                    adaptive_min_step_scale=float(strict_cfg.adaptive_min_step_scale),
                    dynamic_jitter_scale=float(strict_cfg.dynamic_jitter_scale),
                    hybrid_explore_steps=int(strict_cfg.hybrid_explore_steps),
                    hybrid_rescue_steps=int(strict_cfg.hybrid_rescue_steps),
                    target_acceptance=float(args.target_acceptance),
                    pilot_n_chains=args.pilot_n_chains,
                    pilot_chain_length=args.pilot_chain_length,
                    burn_in=args.burn_in,
                    on_trial=_tuned_cb,
                )
                print(
                    f"[pilot][tuned] {key} -> n_lf={cfg.n_lf}, eps_lf={cfg.eps_lf:.5f}, volume_power={cfg.volume_power:.3f}",
                    flush=True,
                )
                safe_wandb_log(
                    wandb_run,
                    {
                        f"pilot/tuned/{key}/selected_n_lf": int(cfg.n_lf),
                        f"pilot/tuned/{key}/selected_eps_lf": float(cfg.eps_lf),
                        f"pilot/tuned/{key}/selected_volume_power": float(cfg.volume_power),
                    },
                )
                protocol_cfgs["tuned"][key] = cfg

        rows: list[dict[str, Any]] = []
        run_start = time.time()
        completed_seed_jobs = 0
        for protocol in args.protocols:
            args.current_protocol = protocol
            print(f"\n=== Protocol: {protocol} ===", flush=True)
            safe_wandb_log(wandb_run, {"progress/current_protocol": protocol})
            for key in active_model_keys:
                cfg = protocol_cfgs[protocol][key]
                print(
                    f"  {key}: n_lf={cfg.n_lf}, eps_lf={cfg.eps_lf:.5f}, volume_power={cfg.volume_power:.3f}",
                    flush=True,
                )
                for seed in args.sampling_seeds:
                    print(f"    seed={seed}", flush=True)
                    safe_wandb_log(
                        wandb_run,
                        {
                            "progress/current_model": key,
                            "progress/current_seed": int(seed),
                            "progress/completed_seed_jobs": int(completed_seed_jobs),
                            "progress/fraction": float(completed_seed_jobs / max(1, total_seed_jobs)),
                        },
                    )
                    row = evaluate_model_protocol_seed(
                        model_key=key,
                        model_dir=model_dirs[key],
                        model=models[key],
                        centroids=centroids_map[key],
                        sampler_cfg=cfg,
                        seed=int(seed),
                        args=args,
                        quality_cache=quality_cache[key],
                        plots_dir=plots_dir,
                    )
                    seed_plot_paths = [Path(p) for p in row.pop("__seed_plot_paths", [])]
                    rows.append(row)
                    write_csv(out_dir / "raw_metrics.csv", rows)
                    log_seed_metrics_to_wandb(wandb_run, row)
                    if seed_plot_paths:
                        log_image_paths_to_wandb(
                            wandb_run,
                            seed_plot_paths,
                            key_prefix=f"seed_plots/{protocol}/{key}/seed_{seed}",
                            max_images=max(0, int(args.wandb_max_plot_images)),
                        )
                    completed_seed_jobs += 1
                    elapsed = max(1e-6, time.time() - run_start)
                    avg_job_sec = elapsed / max(1, completed_seed_jobs)
                    remaining_jobs = max(0, total_seed_jobs - completed_seed_jobs)
                    eta_seconds = avg_job_sec * remaining_jobs
                    safe_wandb_log(
                        wandb_run,
                        {
                            "progress/completed_seed_jobs": int(completed_seed_jobs),
                            "progress/fraction": float(completed_seed_jobs / max(1, total_seed_jobs)),
                            "progress/elapsed_minutes": float(elapsed / 60.0),
                            "progress/eta_minutes": float(eta_seconds / 60.0),
                        },
                    )
                    print(
                        f"      done ({completed_seed_jobs}/{total_seed_jobs}) "
                        f"elapsed={elapsed/60.0:.1f}min eta={eta_seconds/60.0:.1f}min",
                        flush=True,
                    )

        write_csv(out_dir / "raw_metrics.csv", rows)

        summary = summarize_rows(rows, protocols=args.protocols, bootstrap_samples=args.bootstrap_samples)
        score_rows, claim = build_scorecard(summary, protocols=args.protocols)
        write_csv(out_dir / "scorecard.csv", score_rows)
        write_scorecard_md(out_dir / "scorecard.md", score_rows, claim)
        ranking_payload: dict[str, dict[str, Any]] = {}
        ranking_rows: list[dict[str, Any]] = []
        for protocol in args.protocols:
            ranking_payload[protocol] = {}
            for model_key in active_model_keys:
                rank = summarize_candidate_ranking(rows, protocol=protocol, model_key=model_key)
                ranking_payload[protocol][model_key] = rank
                ranking_rows.append(
                    {
                        "protocol": protocol,
                        "model_key": model_key,
                        "n_seeds": rank["n_seeds"],
                        "hard_reject": rank["hard_reject"],
                        "reject_reasons": ",".join(rank["reject_reasons"]),
                        "primary_green_count": rank["primary_green_count"],
                        "primary_n_total": rank["primary_n_total"],
                        "tiebreak_tangent_alignment_mean": rank["tiebreak_tangent_alignment_mean"],
                        "tiebreak_rescue_rate": rank["tiebreak_rescue_rate"],
                        "tiebreak_coverage_local": rank["tiebreak_coverage_local"],
                        "iqr_noise_penalty": rank["iqr_noise_penalty"],
                    }
                )
        sweep_objectives = build_sweep_objectives(ranking_payload, protocol="matched", model_key="aniso")
        write_csv(out_dir / "ranking_summary.csv", ranking_rows)
        (out_dir / "ranking_summary.json").write_text(json.dumps(ranking_payload, indent=2))
        model_bilan_rows = build_model_bilan_rows(model_dirs)
        write_csv(out_dir / "model_run_bilan.csv", model_bilan_rows)

        protocol_cfg_json: dict[str, dict[str, dict[str, Any]]] = {}
        for protocol, mapping in protocol_cfgs.items():
            protocol_cfg_json[protocol] = {}
            for key, cfg in mapping.items():
                cfg_payload = asdict(cfg)
                cfg_payload["effective_volume_exponent"] = float(
                    effective_volume_exponent(cfg.sampler_name, cfg.volume_power)
                )
                protocol_cfg_json[protocol][key] = cfg_payload
        (out_dir / "protocol_configs.json").write_text(json.dumps(protocol_cfg_json, indent=2))

        payload = {
            "metadata": {
                "created_at": datetime.datetime.now().isoformat(),
                "device": str(device),
                "protocols": args.protocols,
                "sampling_seeds": args.sampling_seeds,
                "profile": args.profile,
                "run_quality": bool(args.run_quality),
                "run_fid": bool(args.run_fid),
                "aniso_only": bool(args.aniso_only),
                "skip_matched_pilot": bool(args.skip_matched_pilot),
                "quick_mode": bool(args.quick),
                "save_plots": bool(args.save_plots),
                "wandb_enabled": bool(wandb_run is not None),
            },
            "summary": summary,
            "ranking": ranking_payload,
            "sweep_objectives": sweep_objectives,
            "model_run_bilan": model_bilan_rows,
            "claim": claim,
        }
        (out_dir / "summary_by_protocol.json").write_text(json.dumps(payload, indent=2))

        log_summary_to_wandb(
            wandb_run,
            summary=summary,
            rows=rows,
            score_rows=score_rows,
            claim=claim,
        )
        if args.save_plots:
            log_plot_images_to_wandb(wandb_run, plots_dir, max_images=args.wandb_max_plot_images)
        if wandb_run is not None and wandb is not None and model_bilan_rows:
            bilan_table = rows_to_wandb_table(model_bilan_rows)
            if bilan_table is not None:
                safe_wandb_log(wandb_run, {"tables/model_run_bilan": bilan_table})
        ranking_scalars = build_ranking_wandb_scalars(ranking_payload, sweep_objectives=sweep_objectives)
        if ranking_scalars:
            safe_wandb_log(wandb_run, ranking_scalars)
        log_output_artifact_to_wandb(wandb_run, out_dir)
        if wandb_run is not None and claim is not None:
            wandb_run.summary["claim_validated"] = bool(claim.get("claim_validated", False))

        print(f"\nSaved assessment outputs to: {out_dir}")
        print(f"- raw_metrics.csv")
        print(f"- summary_by_protocol.json")
        print(f"- scorecard.csv")
        print(f"- scorecard.md")
        print(f"- ranking_summary.csv")
        print(f"- ranking_summary.json")
        print(f"- model_run_bilan.csv")
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as exc:
                print(f"[warning] WandB finish failed: {exc}")


if __name__ == "__main__":
    main()
