#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _is_valid_run(model_id: str, run_dir: Path) -> bool:
    if model_id not in {"rhvae_standard", "aniso"}:
        return True
    metric_path = run_dir / "rhvae_metric.pt"
    if not metric_path.exists():
        return False
    try:
        payload = torch.load(metric_path, map_location="cpu")
        centroids = payload.get("centroids")
        mats = payload.get("metric_matrices")
        if not isinstance(centroids, torch.Tensor) or not isinstance(mats, torch.Tensor):
            return False
        if centroids.ndim != 2 or mats.ndim != 3:
            return False
        if int(centroids.shape[0]) < 2 or int(mats.shape[0]) < 2:
            return False
        if int(mats.shape[0]) != int(centroids.shape[0]):
            return False
        return True
    except Exception:
        return False


def _latest_run(parent: Path, required_files: list[str], model_id: str) -> Path | None:
    if not parent.exists() or not parent.is_dir():
        return None
    runs = sorted([p for p in parent.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        if all((r / f).exists() for f in required_files) and _is_valid_run(model_id, r):
            return r
    return None


def _run(cmd: list[str]) -> None:
    print("[train_missing]", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def _extend_wandb_args(
    cmd: list[str],
    args: argparse.Namespace,
    model_id: str,
    n: int,
    seed: int,
    *,
    include_wandb_mode: bool = True,
) -> list[str]:
    out = list(cmd)

    if args.wandb_project:
        out.extend(["--wandb_project", args.wandb_project])
    if args.wandb_entity:
        out.extend(["--wandb_entity", args.wandb_entity])
    if args.wandb_group:
        out.extend(["--wandb_group", args.wandb_group])
    if args.wandb_tags:
        out.extend(["--wandb_tags", args.wandb_tags])
    if include_wandb_mode and args.wandb_mode:
        out.extend(["--wandb_mode", args.wandb_mode])

    out.extend(["--wandb_name_mode", args.wandb_name_mode])
    if args.wandb_run_name:
        run_name = f"{args.wandb_run_name}_{model_id}_N{int(n):03d}_seed{int(seed):03d}"
    else:
        run_name = f"low_data_{model_id}_N{int(n):03d}_seed{int(seed):03d}"
    out.extend(["--wandb_run_name", run_name])
    return out


def _core4_spatial_geometry_args() -> list[str]:
    return [
        "--rhvae_variant",
        "geometry",
        "--use_attractor",
        "--kernel_type",
        "mahalanobis",
        "--atom_norm",
        "trace",
        "--atom_power",
        "1.0959864113997024",
        "--kernel_power",
        "1.0",
        "--precision_jitter",
        "0.01",
        "--attractor_smoothness",
        "soft",
        "--attractor_metric",
        "mahalanobis",
        "--attractor_use_det",
        "--attractor_gamma",
        "7.624554217806352",
        "--attractor_k_nearest",
        "1",
        "--attractor_bias_energy",
        "18.0",
        "--void_threshold",
        "1.2",
        "--void_weight_threshold",
        "-1.0",
        "--void_decay_type",
        "invquad",
        "--void_decay_scale",
        "8.87131893173153",
        "--void_decay_power",
        "1.7447710342008058",
        "--void_decay_softplus_k",
        "5.0",
        "--radial_stretch",
        "9.252860449508804",
        "--transition_steepness",
        "7.276725930229776",
        "--void_eigshape_mode",
        "det_preserving_spectral",
        "--void_eigshape_alpha_min",
        "0.8",
        "--void_eigshape_power",
        "-1.2",
        "--void_eigshape_eig_floor",
        "1e-8",
    ]


def _n_to_num_sequences(n: int, train_ratio: float = 0.8) -> int:
    safe_ratio = max(1e-6, min(float(train_ratio), 0.999999))
    return max(1, int(math.ceil(float(n) / safe_ratio)))


def _resolve_num_sequences(n: int, mode: str, fixed: int) -> int:
    if str(mode) == "fixed":
        return max(1, int(fixed))
    return _n_to_num_sequences(int(n))


def _resolve_max_frames(n: int, mode: str, fixed: int) -> int | None:
    if str(mode) == "fixed":
        return max(1, int(fixed))
    if str(mode) == "n":
        return max(1, int(n))
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train missing low-data models for all N/seed combinations.")
    parser.add_argument("--model_registry", type=str, default="configs/assets/model_registry.yaml")
    parser.add_argument("--output_root", type=str, default="outputs/low_data_models/rotmnist_v1")
    parser.add_argument("--processed_dir", type=str, default="data/processed/rotmnist/v1")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--subset_ns", nargs="+", type=int, default=[50, 100, 500])
    parser.add_argument("--subset_seeds", nargs="+", type=int, default=[13, 29, 47, 71, 89])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--materialize_aliases", action="store_true")
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--rhvae_drop_last", type=str, default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--rhvae_auto_temperature", action="store_true")
    parser.add_argument(
        "--rhvae_auto_temperature_stat",
        type=str,
        default="median_nn",
        choices=["median_nn", "mean_nn", "mean_pairwise", "median_pairwise", "silverman", "mean_knn_5"],
    )
    parser.add_argument("--rhvae_auto_temperature_every", type=int, default=0)
    parser.add_argument("--rhvae_temperature_scale", type=float, default=1.0)
    parser.add_argument("--rhvae_plot_heatmaps_during_training", action="store_true")
    parser.add_argument("--rhvae_vis_every", type=int, default=10)
    parser.add_argument("--rhvae_metric_grid_res", type=int, default=40)
    parser.add_argument("--rhvae_metric_tissot_grid_res", type=int, default=16)
    parser.add_argument(
        "--rhvae_standard_backend",
        type=str,
        default="run_with_config",
        choices=["tensor", "run_with_config"],
        help="Training backend for model_id=rhvae_standard.",
    )
    parser.add_argument(
        "--rhvae_standard_num_sequences_mode",
        type=str,
        default="from_n",
        choices=["from_n", "fixed"],
        help="How to set num_sequences when using run_with_config backend.",
    )
    parser.add_argument(
        "--rhvae_standard_num_sequences",
        type=int,
        default=200,
        help="Used when --rhvae_standard_num_sequences_mode=fixed.",
    )
    parser.add_argument(
        "--rhvae_standard_max_frames_mode",
        type=str,
        default="n",
        choices=["n", "fixed", "none"],
        help="How to set max_frames when using run_with_config backend.",
    )
    parser.add_argument(
        "--rhvae_standard_max_frames",
        type=int,
        default=3000,
        help="Used when --rhvae_standard_max_frames_mode=fixed.",
    )
    parser.add_argument("--rhvae_standard_frame_mode", type=str, default="t0", choices=["t0", "all"])
    parser.add_argument("--rhvae_standard_lr", type=float, default=5e-4)
    parser.add_argument("--rhvae_standard_n_centroids", type=int, default=100)
    parser.add_argument("--rhvae_standard_temperature", type=float, default=0.5)
    parser.add_argument("--rhvae_standard_regularization", type=float, default=1e-2)
    parser.add_argument(
        "--rhvae_standard_analysis_sampler",
        type=str,
        default="volume",
        choices=["riemannian", "geodesic", "volume"],
    )
    parser.add_argument("--rhvae_standard_sampling_fid_samples", type=int, default=1000)
    parser.add_argument("--rhvae_standard_sampling_quality_samples", type=int, default=500)
    parser.add_argument("--rhvae_standard_sampling_n_chains", type=int, default=4)
    parser.add_argument("--rhvae_standard_sampling_chain_length", type=int, default=100)
    parser.add_argument("--rhvae_standard_skip_sampling_diagnostics", action="store_true")
    parser.add_argument(
        "--rhvae_aniso_profile",
        type=str,
        default="legacy",
        choices=["legacy", "gravity_well", "core4_spatial"],
    )
    parser.add_argument(
        "--rhvae_aniso_backend",
        type=str,
        default="run_with_config",
        choices=["tensor", "run_with_config"],
        help="Training backend for model_id=aniso.",
    )
    parser.add_argument(
        "--rhvae_aniso_num_sequences_mode",
        type=str,
        default="from_n",
        choices=["from_n", "fixed"],
        help="How to set num_sequences when using run_with_config backend.",
    )
    parser.add_argument(
        "--rhvae_aniso_num_sequences",
        type=int,
        default=200,
        help="Used when --rhvae_aniso_num_sequences_mode=fixed.",
    )
    parser.add_argument(
        "--rhvae_aniso_max_frames_mode",
        type=str,
        default="n",
        choices=["n", "fixed", "none"],
        help="How to set max_frames when using run_with_config backend.",
    )
    parser.add_argument(
        "--rhvae_aniso_max_frames",
        type=int,
        default=3000,
        help="Used when --rhvae_aniso_max_frames_mode=fixed.",
    )
    parser.add_argument("--rhvae_aniso_frame_mode", type=str, default="t0", choices=["t0", "all"])
    parser.add_argument("--rhvae_aniso_lr", type=float, default=5e-4)
    parser.add_argument("--rhvae_aniso_n_centroids", type=int, default=100)
    parser.add_argument("--rhvae_aniso_temperature", type=float, default=0.7960961952783896)
    parser.add_argument("--rhvae_aniso_regularization", type=float, default=0.6)
    parser.add_argument(
        "--rhvae_aniso_analysis_sampler",
        type=str,
        default="volume",
        choices=["riemannian", "geodesic", "volume"],
    )
    parser.add_argument("--rhvae_aniso_sampling_fid_samples", type=int, default=1000)
    parser.add_argument("--rhvae_aniso_sampling_quality_samples", type=int, default=500)
    parser.add_argument("--rhvae_aniso_sampling_n_chains", type=int, default=4)
    parser.add_argument("--rhvae_aniso_sampling_chain_length", type=int, default=100)
    parser.add_argument("--rhvae_aniso_skip_sampling_diagnostics", action="store_true")
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
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = str(args.wandb_mode)

    import yaml

    registry_path = (ROOT / args.model_registry).resolve()
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    model_entries = list(registry.get("models", []))
    if args.models:
        selected = {str(m).strip() for m in args.models if str(m).strip()}
        model_entries = [e for e in model_entries if str(e.get("model_id")) in selected]
        if not model_entries:
            raise RuntimeError(f"No model entries matched --models={sorted(selected)}")

    if args.quick:
        args.subset_ns = [50]
        args.subset_seeds = [13]
        args.epochs = min(args.epochs, 2)

    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for n in args.subset_ns:
        for seed in args.subset_seeds:
            for entry in model_entries:
                model_id = str(entry["model_id"])
                required = [str(x) for x in entry.get("required_files", [])]
                parent = output_root / model_id / f"N{int(n):03d}" / f"seed{int(seed):03d}"
                if (not args.force_retrain) and _latest_run(parent, required, model_id=model_id) is not None:
                    print(f"[train_missing] skip existing {model_id} N={n} seed={seed}")
                    continue

                if model_id == "vanilla_vae":
                    cmd = _extend_wandb_args(
                        [
                            sys.executable,
                            "scripts/train_vanilla_vae.py",
                            "--processed_dir",
                            args.processed_dir,
                            "--n",
                            str(n),
                            "--seed",
                            str(seed),
                            "--epochs",
                            str(args.epochs),
                            "--batch_size",
                            str(args.batch_size),
                            "--latent_dim",
                            str(args.latent_dim),
                            "--output_root",
                            args.output_root,
                            "--model_id",
                            model_id,
                        ],
                        args,
                        model_id=model_id,
                        n=n,
                        seed=seed,
                    )
                    _run(cmd)
                elif model_id == "rhvae_standard":
                    if args.rhvae_standard_backend == "tensor":
                        rhvae_extra = [
                            "--batch_size",
                            str(args.batch_size),
                            "--drop_last",
                            args.rhvae_drop_last,
                            "--vis_every",
                            str(args.rhvae_vis_every),
                            "--metric_grid_res",
                            str(args.rhvae_metric_grid_res),
                            "--metric_tissot_grid_res",
                            str(args.rhvae_metric_tissot_grid_res),
                        ]
                        if args.rhvae_auto_temperature:
                            rhvae_extra.extend(
                                [
                                    "--auto_temperature",
                                    "--auto_temperature_stat",
                                    str(args.rhvae_auto_temperature_stat),
                                    "--auto_temperature_every",
                                    str(args.rhvae_auto_temperature_every),
                                    "--temperature_scale",
                                    str(args.rhvae_temperature_scale),
                                ]
                            )
                        if args.rhvae_plot_heatmaps_during_training:
                            rhvae_extra.append("--plot_heatmaps_during_training")
                        cmd = _extend_wandb_args(
                            [
                                sys.executable,
                                "scripts/train_rhvae_tensor.py",
                                "--mode",
                                "standard",
                                "--processed_dir",
                                args.processed_dir,
                                "--n",
                                str(n),
                                "--seed",
                                str(seed),
                                "--epochs",
                                str(args.epochs),
                                "--latent_dim",
                                str(args.latent_dim),
                                "--output_root",
                                args.output_root,
                                "--model_id",
                                model_id,
                            ]
                            + rhvae_extra,
                            args,
                            model_id=model_id,
                            n=n,
                            seed=seed,
                        )
                        _run(cmd)
                    elif args.rhvae_standard_backend == "run_with_config":
                        num_sequences = _resolve_num_sequences(
                            n=int(n),
                            mode=str(args.rhvae_standard_num_sequences_mode),
                            fixed=int(args.rhvae_standard_num_sequences),
                        )
                        max_frames = _resolve_max_frames(
                            n=int(n),
                            mode=str(args.rhvae_standard_max_frames_mode),
                            fixed=int(args.rhvae_standard_max_frames),
                        )

                        output_parent = output_root / model_id / f"N{int(n):03d}" / f"seed{int(seed):03d}"
                        output_parent.mkdir(parents=True, exist_ok=True)

                        standard_cmd = [
                            sys.executable,
                            "scripts/run_pythae_rhvae_baseline.py",
                            "--rhvae_variant",
                            "standard",
                            "--epochs",
                            str(args.epochs),
                            "--batch_size",
                            str(args.batch_size),
                            "--lr",
                            str(args.rhvae_standard_lr),
                            "--seed",
                            str(seed),
                            "--output_dir",
                            str(output_parent),
                            "--num_sequences",
                            str(num_sequences),
                            "--frame_mode",
                            str(args.rhvae_standard_frame_mode),
                            "--latent_dim",
                            str(args.latent_dim),
                            "--n_centroids",
                            str(args.rhvae_standard_n_centroids),
                            "--regularization",
                            str(args.rhvae_standard_regularization),
                            "--vis_every",
                            str(args.rhvae_vis_every),
                            "--analysis_rhmc_sampler",
                            str(args.rhvae_standard_analysis_sampler),
                            "--sampling_fid_samples",
                            str(args.rhvae_standard_sampling_fid_samples),
                            "--sampling_quality_samples",
                            str(args.rhvae_standard_sampling_quality_samples),
                            "--sampling_n_chains",
                            str(args.rhvae_standard_sampling_n_chains),
                            "--sampling_chain_length",
                            str(args.rhvae_standard_sampling_chain_length),
                        ]
                        if max_frames is not None:
                            standard_cmd.extend(["--max_frames", str(max_frames)])
                        if args.rhvae_auto_temperature:
                            standard_cmd.extend(
                                [
                                    "--auto_temperature",
                                    "--auto_temperature_stat",
                                    str(args.rhvae_auto_temperature_stat),
                                    "--auto_temperature_every",
                                    str(args.rhvae_auto_temperature_every),
                                    "--temperature_scale",
                                    str(args.rhvae_temperature_scale),
                                ]
                            )
                        else:
                            standard_cmd.extend(["--temperature", str(args.rhvae_standard_temperature)])
                        if args.rhvae_standard_skip_sampling_diagnostics:
                            standard_cmd.append("--skip_sampling_diagnostics")

                        cmd = _extend_wandb_args(
                            standard_cmd,
                            args,
                            model_id=model_id,
                            n=n,
                            seed=seed,
                            include_wandb_mode=False,
                        )
                        _run(cmd)
                    else:
                        raise RuntimeError(f"Unsupported --rhvae_standard_backend={args.rhvae_standard_backend}")
                elif model_id == "aniso":
                    if args.rhvae_aniso_backend == "tensor":
                        rhvae_extra = [
                            "--batch_size",
                            str(args.batch_size),
                            "--drop_last",
                            args.rhvae_drop_last,
                            "--vis_every",
                            str(args.rhvae_vis_every),
                            "--metric_grid_res",
                            str(args.rhvae_metric_grid_res),
                            "--metric_tissot_grid_res",
                            str(args.rhvae_metric_tissot_grid_res),
                        ]
                        if args.rhvae_auto_temperature:
                            rhvae_extra.extend(
                                [
                                    "--auto_temperature",
                                    "--auto_temperature_stat",
                                    str(args.rhvae_auto_temperature_stat),
                                    "--auto_temperature_every",
                                    str(args.rhvae_auto_temperature_every),
                                    "--temperature_scale",
                                    str(args.rhvae_temperature_scale),
                                ]
                            )
                        if args.rhvae_plot_heatmaps_during_training:
                            rhvae_extra.append("--plot_heatmaps_during_training")
                        cmd = _extend_wandb_args(
                            [
                                sys.executable,
                                "scripts/train_rhvae_tensor.py",
                                "--mode",
                                "aniso",
                                "--aniso_profile",
                                str(args.rhvae_aniso_profile),
                                "--processed_dir",
                                args.processed_dir,
                                "--n",
                                str(n),
                                "--seed",
                                str(seed),
                                "--epochs",
                                str(args.epochs),
                                "--latent_dim",
                                str(args.latent_dim),
                                "--output_root",
                                args.output_root,
                                "--model_id",
                                model_id,
                            ]
                            + rhvae_extra,
                            args,
                            model_id=model_id,
                            n=n,
                            seed=seed,
                        )
                        _run(cmd)
                    elif args.rhvae_aniso_backend == "run_with_config":
                        if str(args.rhvae_aniso_profile) != "core4_spatial":
                            raise RuntimeError(
                                "run_with_config backend currently supports only --rhvae_aniso_profile core4_spatial. "
                                "Use --rhvae_aniso_backend tensor for legacy/gravity_well."
                            )
                        num_sequences = _resolve_num_sequences(
                            n=int(n),
                            mode=str(args.rhvae_aniso_num_sequences_mode),
                            fixed=int(args.rhvae_aniso_num_sequences),
                        )
                        max_frames = _resolve_max_frames(
                            n=int(n),
                            mode=str(args.rhvae_aniso_max_frames_mode),
                            fixed=int(args.rhvae_aniso_max_frames),
                        )

                        output_parent = output_root / model_id / f"N{int(n):03d}" / f"seed{int(seed):03d}"
                        output_parent.mkdir(parents=True, exist_ok=True)

                        aniso_cmd = [
                            sys.executable,
                            "scripts/run_pythae_rhvae_baseline.py",
                            "--epochs",
                            str(args.epochs),
                            "--batch_size",
                            str(args.batch_size),
                            "--lr",
                            str(args.rhvae_aniso_lr),
                            "--seed",
                            str(seed),
                            "--output_dir",
                            str(output_parent),
                            "--num_sequences",
                            str(num_sequences),
                            "--frame_mode",
                            str(args.rhvae_aniso_frame_mode),
                            "--latent_dim",
                            str(args.latent_dim),
                            "--n_centroids",
                            str(args.rhvae_aniso_n_centroids),
                            "--regularization",
                            str(args.rhvae_aniso_regularization),
                            "--vis_every",
                            str(args.rhvae_vis_every),
                            "--analysis_rhmc_sampler",
                            str(args.rhvae_aniso_analysis_sampler),
                            "--sampling_fid_samples",
                            str(args.rhvae_aniso_sampling_fid_samples),
                            "--sampling_quality_samples",
                            str(args.rhvae_aniso_sampling_quality_samples),
                            "--sampling_n_chains",
                            str(args.rhvae_aniso_sampling_n_chains),
                            "--sampling_chain_length",
                            str(args.rhvae_aniso_sampling_chain_length),
                        ]
                        if max_frames is not None:
                            aniso_cmd.extend(["--max_frames", str(max_frames)])
                        if args.rhvae_auto_temperature:
                            aniso_cmd.extend(
                                [
                                    "--auto_temperature",
                                    "--auto_temperature_stat",
                                    str(args.rhvae_auto_temperature_stat),
                                    "--auto_temperature_every",
                                    str(args.rhvae_auto_temperature_every),
                                    "--temperature_scale",
                                    str(args.rhvae_temperature_scale),
                                ]
                            )
                        else:
                            aniso_cmd.extend(["--temperature", str(args.rhvae_aniso_temperature)])
                        if args.rhvae_aniso_skip_sampling_diagnostics:
                            aniso_cmd.append("--skip_sampling_diagnostics")

                        aniso_cmd.extend(_core4_spatial_geometry_args())
                        cmd = _extend_wandb_args(
                            aniso_cmd,
                            args,
                            model_id=model_id,
                            n=n,
                            seed=seed,
                            include_wandb_mode=False,
                        )
                        _run(cmd)
                    else:
                        raise RuntimeError(f"Unsupported --rhvae_aniso_backend={args.rhvae_aniso_backend}")
                elif model_id == "ebm_conformal":
                    cmd = _extend_wandb_args(
                        [
                            sys.executable,
                            "scripts/train_ebm_conformal.py",
                            "--processed_dir",
                            args.processed_dir,
                            "--n",
                            str(n),
                            "--seed",
                            str(seed),
                            "--latent_dim",
                            str(args.latent_dim),
                            "--ae_epochs",
                            str(max(2, args.epochs // 2)),
                            "--ebm_epochs",
                            str(max(2, args.epochs // 2)),
                            "--batch_size",
                            str(args.batch_size),
                            "--output_root",
                            args.output_root,
                            "--model_id",
                            model_id,
                        ],
                        args,
                        model_id=model_id,
                        n=n,
                        seed=seed,
                    )
                    _run(cmd)
                else:
                    print(f"[train_missing] WARNING unsupported model_id={model_id}, skipping")

    if args.materialize_aliases:
        alias_root = ROOT / "outputs/reference_models"
        alias_root.mkdir(parents=True, exist_ok=True)
        preferred_n = 500 if 500 in args.subset_ns else args.subset_ns[-1]
        preferred_seed = 13 if 13 in args.subset_seeds else args.subset_seeds[0]

        for entry in model_entries:
            model_id = str(entry["model_id"])
            required = [str(x) for x in entry.get("required_files", [])]
            parent = output_root / model_id / f"N{int(preferred_n):03d}" / f"seed{int(preferred_seed):03d}"
            chosen = _latest_run(parent, required, model_id=model_id)
            if chosen is None:
                continue
            alias = alias_root / model_id
            if alias.is_symlink() or alias.exists():
                try:
                    alias.unlink()
                except IsADirectoryError:
                    # Keep pre-existing directories untouched.
                    continue
            alias.symlink_to(chosen)
            print(f"[train_missing] alias {alias} -> {chosen}")

    print("[train_missing] complete")


if __name__ == "__main__":
    main()
