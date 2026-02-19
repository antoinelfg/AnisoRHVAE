#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def _extend_wandb_args(cmd: list[str], args: argparse.Namespace, model_id: str, n: int, seed: int) -> list[str]:
    out = list(cmd)

    if args.wandb_project:
        out.extend(["--wandb_project", args.wandb_project])
    if args.wandb_entity:
        out.extend(["--wandb_entity", args.wandb_entity])
    if args.wandb_group:
        out.extend(["--wandb_group", args.wandb_group])
    if args.wandb_tags:
        out.extend(["--wandb_tags", args.wandb_tags])
    if args.wandb_mode:
        out.extend(["--wandb_mode", args.wandb_mode])

    out.extend(["--wandb_name_mode", args.wandb_name_mode])
    if args.wandb_run_name:
        run_name = f"{args.wandb_run_name}_{model_id}_N{int(n):03d}_seed{int(seed):03d}"
    else:
        run_name = f"low_data_{model_id}_N{int(n):03d}_seed{int(seed):03d}"
    out.extend(["--wandb_run_name", run_name])
    return out


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
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--materialize_aliases", action="store_true")
    parser.add_argument("--rhvae_drop_last", type=str, default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--rhvae_auto_temperature", action="store_true")
    parser.add_argument(
        "--rhvae_auto_temperature_stat",
        type=str,
        default="median_nn",
        choices=["median_nn", "mean_nn", "mean_pairwise"],
    )
    parser.add_argument("--rhvae_auto_temperature_every", type=int, default=0)
    parser.add_argument("--rhvae_temperature_scale", type=float, default=1.0)
    parser.add_argument("--rhvae_plot_heatmaps_during_training", action="store_true")
    parser.add_argument("--rhvae_vis_every", type=int, default=10)
    parser.add_argument("--rhvae_metric_grid_res", type=int, default=40)
    parser.add_argument("--rhvae_metric_tissot_grid_res", type=int, default=16)
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
                if _latest_run(parent, required, model_id=model_id) is not None:
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
                elif model_id == "aniso":
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
                            "--processed_dir",
                            args.processed_dir,
                            "--n",
                            str(n),
                            "--seed",
                            str(seed),
                            "--epochs",
                            str(args.epochs),
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
