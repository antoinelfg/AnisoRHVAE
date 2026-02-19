#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.utils.asset_manifest import dump_json, ensure_dir, sha256_file, utc_now_iso


@dataclass
class SplitBundle:
    images: torch.Tensor  # [N, 2], float32
    labels: torch.Tensor  # [N], int64


def _balanced_labels(n_samples: int, n_classes: int) -> np.ndarray:
    counts = [n_samples // n_classes for _ in range(n_classes)]
    for i in range(n_samples % n_classes):
        counts[i] += 1
    labels = np.concatenate([np.full(c, cls, dtype=np.int64) for cls, c in enumerate(counts)], axis=0)
    return labels


def _sample_moons(n_samples: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(int(n_samples), 2)
    x = np.zeros((int(n_samples), 2), dtype=np.float32)

    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    t0 = rng.uniform(0.0, math.pi, size=idx0.shape[0])
    x[idx0, 0] = np.cos(t0)
    x[idx0, 1] = np.sin(t0)

    t1 = rng.uniform(0.0, math.pi, size=idx1.shape[0])
    x[idx1, 0] = 1.0 - np.cos(t1)
    x[idx1, 1] = -np.sin(t1) - 0.5

    x += rng.normal(0.0, float(noise), size=x.shape).astype(np.float32)
    perm = rng.permutation(x.shape[0])
    return x[perm], labels[perm]


def _sample_circles(
    n_samples: int,
    noise: float,
    factor: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(int(n_samples), 2)
    x = np.zeros((int(n_samples), 2), dtype=np.float32)
    angles = rng.uniform(0.0, 2.0 * math.pi, size=x.shape[0])
    radii = np.where(labels == 0, 1.0, float(factor)).astype(np.float32)
    x[:, 0] = radii * np.cos(angles)
    x[:, 1] = radii * np.sin(angles)
    x += rng.normal(0.0, float(noise), size=x.shape).astype(np.float32)
    perm = rng.permutation(x.shape[0])
    return x[perm], labels[perm]


def _sample_ring_gaussians(
    n_samples: int,
    n_classes: int,
    radius: float,
    std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(int(n_samples), int(n_classes))
    x = np.zeros((int(n_samples), 2), dtype=np.float32)
    centers = []
    for k in range(int(n_classes)):
        a = 2.0 * math.pi * float(k) / float(n_classes)
        centers.append(np.array([radius * math.cos(a), radius * math.sin(a)], dtype=np.float32))
    centers_np = np.stack(centers, axis=0)
    for k in range(int(n_classes)):
        idx = np.where(labels == k)[0]
        mu = centers_np[k]
        x[idx] = mu + rng.normal(0.0, float(std), size=(idx.shape[0], 2)).astype(np.float32)
    perm = rng.permutation(x.shape[0])
    return x[perm], labels[perm]


def _sample_blobs(
    n_samples: int,
    n_classes: int,
    std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(int(n_samples), int(n_classes))
    x = np.zeros((int(n_samples), 2), dtype=np.float32)
    radius = 2.5
    centers = []
    for k in range(int(n_classes)):
        a = 2.0 * math.pi * float(k) / float(n_classes)
        centers.append(np.array([radius * math.cos(a), radius * math.sin(a)], dtype=np.float32))
    centers_np = np.stack(centers, axis=0)
    for k in range(int(n_classes)):
        idx = np.where(labels == k)[0]
        mu = centers_np[k]
        x[idx] = mu + rng.normal(0.0, float(std), size=(idx.shape[0], 2)).astype(np.float32)
    perm = rng.permutation(x.shape[0])
    return x[perm], labels[perm]


def _build_split(
    dataset: str,
    n_samples: int,
    seed: int,
    noise: float,
    circles_factor: float,
    n_classes: int,
    ring_radius: float,
    ring_std: float,
    standardize_mean: np.ndarray | None = None,
    standardize_std: np.ndarray | None = None,
) -> tuple[SplitBundle, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))

    if dataset == "moons":
        x_np, y_np = _sample_moons(n_samples=int(n_samples), noise=float(noise), rng=rng)
    elif dataset == "circles":
        x_np, y_np = _sample_circles(
            n_samples=int(n_samples),
            noise=float(noise),
            factor=float(circles_factor),
            rng=rng,
        )
    elif dataset == "8gaussians":
        x_np, y_np = _sample_ring_gaussians(
            n_samples=int(n_samples),
            n_classes=8,
            radius=float(ring_radius),
            std=float(ring_std),
            rng=rng,
        )
    elif dataset == "blobs":
        x_np, y_np = _sample_blobs(
            n_samples=int(n_samples),
            n_classes=int(n_classes),
            std=float(ring_std),
            rng=rng,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if standardize_mean is not None and standardize_std is not None:
        x_np = (x_np - standardize_mean) / np.clip(standardize_std, 1e-6, None)

    bundle = SplitBundle(
        images=torch.from_numpy(x_np.astype(np.float32)),
        labels=torch.from_numpy(y_np.astype(np.int64)),
    )
    return bundle, x_np, y_np


def _subset_indices_stratified(
    labels: torch.Tensor,
    n_samples: int,
    seed: int,
    num_classes: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    labels_np = labels.cpu().numpy()

    counts = [n_samples // num_classes for _ in range(num_classes)]
    for k in range(n_samples % num_classes):
        counts[k] += 1

    all_idx: list[int] = []
    per_class_counts: dict[str, int] = {}
    for cls in range(num_classes):
        cls_idx = np.where(labels_np == cls)[0]
        need = counts[cls]
        if need > len(cls_idx):
            raise ValueError(f"Class {cls} has only {len(cls_idx)} samples, need {need}.")
        chosen = rng.choice(cls_idx, size=need, replace=False)
        all_idx.extend(int(i) for i in chosen.tolist())
        per_class_counts[str(cls)] = int(need)

    rng.shuffle(all_idx)
    return {
        "indices": [int(i) for i in all_idx],
        "class_counts": per_class_counts,
    }


def _save_split(path: Path, bundle: SplitBundle, split_name: str, dataset_id: str) -> None:
    payload = {
        "images": bundle.images,
        "labels": bundle.labels,
        "split": split_name,
        "dataset_id": dataset_id,
    }
    torch.save(payload, path)


def _plot_overview(
    train_xy: np.ndarray,
    train_y: np.ndarray,
    val_xy: np.ndarray,
    val_y: np.ndarray,
    test_xy: np.ndarray,
    test_y: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, xy, yy, title in [
        (axes[0], train_xy, train_y, "train"),
        (axes[1], val_xy, val_y, "val"),
        (axes[2], test_xy, test_y, "test"),
    ]:
        ax.scatter(xy[:, 0], xy[:, 1], c=yy, s=6, alpha=0.8, cmap="tab10")
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic toy-2D low-data benchmark assets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="moons",
        choices=["moons", "circles", "8gaussians", "blobs"],
    )
    parser.add_argument("--processed_dir", type=str, default="data/processed/toy2d/moons_v1")
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--val_size", type=int, default=400)
    parser.add_argument("--test_size", type=int, default=400)
    parser.add_argument("--split_seed", type=int, default=13)
    parser.add_argument("--subset_ns", nargs="+", type=int, default=[50, 100, 500])
    parser.add_argument("--subset_seeds", nargs="+", type=int, default=[13, 29, 47, 71, 89])
    parser.add_argument("--num_classes", type=int, default=6, help="Only used for --dataset blobs.")
    parser.add_argument("--noise", type=float, default=0.08, help="Noise for moons/circles.")
    parser.add_argument("--circles_factor", type=float, default=0.5, help="Inner radius factor for circles.")
    parser.add_argument("--ring_radius", type=float, default=2.0, help="Radius for 8gaussians.")
    parser.add_argument("--ring_std", type=float, default=0.18, help="Cluster std for 8gaussians/blobs.")
    parser.add_argument("--standardize", action="store_true", help="Standardize with train mean/std.")
    parser.add_argument("--data_manifest_out", type=str, default="assets/manifests/data_manifest_toy2d.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.split_seed))
    np.random.seed(int(args.split_seed))
    torch.manual_seed(int(args.split_seed))

    processed_dir = (ROOT / args.processed_dir).resolve()
    subset_dir = processed_dir / "subsets"
    ensure_dir(processed_dir)
    ensure_dir(subset_dir)

    if args.dataset == "moons":
        n_classes = 2
    elif args.dataset == "circles":
        n_classes = 2
    elif args.dataset == "8gaussians":
        n_classes = 8
    else:
        n_classes = int(args.num_classes)

    dataset_id = f"toy2d_{args.dataset}_v1"

    train_bundle_raw, train_xy_raw, train_y = _build_split(
        dataset=args.dataset,
        n_samples=int(args.train_size),
        seed=int(args.split_seed),
        noise=float(args.noise),
        circles_factor=float(args.circles_factor),
        n_classes=int(n_classes),
        ring_radius=float(args.ring_radius),
        ring_std=float(args.ring_std),
    )
    standardize_mean = None
    standardize_std = None
    if args.standardize:
        standardize_mean = train_xy_raw.mean(axis=0, keepdims=False)
        standardize_std = train_xy_raw.std(axis=0, keepdims=False)

    if standardize_mean is not None and standardize_std is not None:
        train_xy = (train_xy_raw - standardize_mean) / np.clip(standardize_std, 1e-6, None)
        train_bundle = SplitBundle(
            images=torch.from_numpy(train_xy.astype(np.float32)),
            labels=train_bundle_raw.labels.clone(),
        )
    else:
        train_xy = train_xy_raw
        train_bundle = train_bundle_raw

    val_bundle, val_xy, val_y = _build_split(
        dataset=args.dataset,
        n_samples=int(args.val_size),
        seed=int(args.split_seed) + 1,
        noise=float(args.noise),
        circles_factor=float(args.circles_factor),
        n_classes=int(n_classes),
        ring_radius=float(args.ring_radius),
        ring_std=float(args.ring_std),
        standardize_mean=standardize_mean,
        standardize_std=standardize_std,
    )
    test_bundle, test_xy, test_y = _build_split(
        dataset=args.dataset,
        n_samples=int(args.test_size),
        seed=int(args.split_seed) + 2,
        noise=float(args.noise),
        circles_factor=float(args.circles_factor),
        n_classes=int(n_classes),
        ring_radius=float(args.ring_radius),
        ring_std=float(args.ring_std),
        standardize_mean=standardize_mean,
        standardize_std=standardize_std,
    )

    train_path = processed_dir / "train.pt"
    val_path = processed_dir / "val.pt"
    test_path = processed_dir / "test.pt"
    _save_split(train_path, train_bundle, "train", dataset_id=dataset_id)
    _save_split(val_path, val_bundle, "val", dataset_id=dataset_id)
    _save_split(test_path, test_bundle, "test", dataset_id=dataset_id)

    subset_files: list[str] = []
    for n in args.subset_ns:
        if int(n) < int(n_classes):
            raise ValueError(f"subset N={n} must be >= num_classes={n_classes} for stratified split.")
        for seed in args.subset_seeds:
            subset = _subset_indices_stratified(
                labels=train_bundle.labels,
                n_samples=int(n),
                seed=int(seed),
                num_classes=int(n_classes),
            )
            out = {
                "dataset_id": dataset_id,
                "split": "train",
                "n": int(n),
                "seed": int(seed),
                "num_classes": int(n_classes),
                "indices": subset["indices"],
                "class_counts": subset["class_counts"],
            }
            p = subset_dir / f"N{int(n):03d}_seed{int(seed):03d}.json"
            p.write_text(json.dumps(out, indent=2), encoding="utf-8")
            subset_files.append(str(p))

    overview_path = processed_dir / "overview.png"
    _plot_overview(train_xy, train_y, val_xy, val_y, test_xy, test_y, out_path=overview_path)

    metadata = {
        "dataset_id": dataset_id,
        "created_at": utc_now_iso(),
        "processed_dir": str(processed_dir),
        "dataset": str(args.dataset),
        "split_seed": int(args.split_seed),
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "test_size": int(args.test_size),
        "num_classes": int(n_classes),
        "subset_ns": [int(x) for x in args.subset_ns],
        "subset_seeds": [int(x) for x in args.subset_seeds],
        "standardize": bool(args.standardize),
        "standardize_mean": [float(x) for x in standardize_mean.tolist()] if standardize_mean is not None else None,
        "standardize_std": [float(x) for x in standardize_std.tolist()] if standardize_std is not None else None,
        "noise": float(args.noise),
        "circles_factor": float(args.circles_factor),
        "ring_radius": float(args.ring_radius),
        "ring_std": float(args.ring_std),
        "files": {
            "train_pt": str(train_path),
            "val_pt": str(val_path),
            "test_pt": str(test_path),
            "subset_dir": str(subset_dir),
            "subset_files": subset_files,
            "overview_png": str(overview_path),
        },
    }
    metadata_path = processed_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    data_manifest = {
        "created_at": utc_now_iso(),
        "datasets": [
            {
                "dataset_id": dataset_id,
                "status": "ready",
                "processed_dir": str(processed_dir),
                "files": [
                    {
                        "relative_path": str(Path(args.processed_dir) / "train.pt"),
                        "absolute_path": str(train_path),
                        "sha256": sha256_file(train_path),
                        "size_bytes": train_path.stat().st_size,
                    },
                    {
                        "relative_path": str(Path(args.processed_dir) / "val.pt"),
                        "absolute_path": str(val_path),
                        "sha256": sha256_file(val_path),
                        "size_bytes": val_path.stat().st_size,
                    },
                    {
                        "relative_path": str(Path(args.processed_dir) / "test.pt"),
                        "absolute_path": str(test_path),
                        "sha256": sha256_file(test_path),
                        "size_bytes": test_path.stat().st_size,
                    },
                    {
                        "relative_path": str(Path(args.processed_dir) / "metadata.json"),
                        "absolute_path": str(metadata_path),
                        "sha256": sha256_file(metadata_path),
                        "size_bytes": metadata_path.stat().st_size,
                    },
                    {
                        "relative_path": str(Path(args.processed_dir) / "overview.png"),
                        "absolute_path": str(overview_path),
                        "sha256": sha256_file(overview_path),
                        "size_bytes": overview_path.stat().st_size,
                    },
                ],
                "split_definition": {
                    "dataset": str(args.dataset),
                    "train_size": int(args.train_size),
                    "val_size": int(args.val_size),
                    "test_size": int(args.test_size),
                    "num_classes": int(n_classes),
                    "split_seed": int(args.split_seed),
                    "standardize": bool(args.standardize),
                    "noise": float(args.noise),
                    "circles_factor": float(args.circles_factor),
                    "ring_radius": float(args.ring_radius),
                    "ring_std": float(args.ring_std),
                },
                "subset_definition": {
                    "subset_ns": [int(x) for x in args.subset_ns],
                    "subset_seeds": [int(x) for x in args.subset_seeds],
                    "stratified": True,
                    "subset_count": len(subset_files),
                },
            }
        ],
    }
    dump_json((ROOT / args.data_manifest_out).resolve(), data_manifest)

    print("[prepare_toy2d_lowdata] completed")
    print(f"  dataset_id={dataset_id}")
    print(f"  processed_dir={processed_dir}")
    print(f"  metadata={metadata_path}")
    print(f"  subsets={len(subset_files)}")
    print(f"  overview={overview_path}")


if __name__ == "__main__":
    main()
