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

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.utils.asset_manifest import dump_json, ensure_dir, sha256_file, utc_now_iso


@dataclass
class SplitBundle:
    images: torch.Tensor  # [N, 1, H, W], float32 [0,1]
    labels: torch.Tensor  # [N]
    angles: torch.Tensor  # [N], degrees


def _rotate_tensor_images(images_u8: torch.Tensor, angles: np.ndarray) -> torch.Tensor:
    n = images_u8.shape[0]
    out = torch.empty((n, 1, images_u8.shape[1], images_u8.shape[2]), dtype=torch.float32)
    for i in range(n):
        img = images_u8[i].unsqueeze(0).float() / 255.0
        out[i] = TF.rotate(
            img,
            angle=float(angles[i]),
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
    return out


def _build_split(data: torch.Tensor, labels: torch.Tensor, angles_seed: int) -> SplitBundle:
    rng = np.random.default_rng(int(angles_seed))
    angles = rng.uniform(0.0, 360.0, size=(data.shape[0],)).astype(np.float32)
    rotated = _rotate_tensor_images(data, angles)
    return SplitBundle(
        images=rotated,
        labels=labels.clone().long(),
        angles=torch.from_numpy(angles),
    )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic RotMNIST low-data benchmark assets.")
    parser.add_argument("--raw_dir", type=str, default="data/external/mnist/raw")
    parser.add_argument("--processed_dir", type=str, default="data/processed/rotmnist/v1")
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--val_size", type=int, default=10000)
    parser.add_argument("--test_size", type=int, default=10000)
    parser.add_argument("--angles_seed", type=int, default=13)
    parser.add_argument("--split_seed", type=int, default=13)
    parser.add_argument("--subset_ns", nargs="+", type=int, default=[50, 100, 500])
    parser.add_argument("--subset_seeds", nargs="+", type=int, default=[13, 29, 47, 71, 89])
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--data_manifest_out", type=str, default="assets/manifests/data_manifest.json")
    return parser.parse_args()


def _save_split(path: Path, bundle: SplitBundle, split_name: str) -> None:
    payload = {
        "images": bundle.images,
        "labels": bundle.labels,
        "angles": bundle.angles,
        "split": split_name,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()

    raw_dir = (ROOT / args.raw_dir).resolve()
    processed_dir = (ROOT / args.processed_dir).resolve()
    subset_dir = processed_dir / "subsets"

    ensure_dir(raw_dir)
    ensure_dir(processed_dir)
    ensure_dir(subset_dir)

    torch.manual_seed(args.split_seed)
    random.seed(args.split_seed)
    np.random.seed(args.split_seed)

    ds_train = MNIST(root=str(raw_dir), train=True, download=True)
    ds_test = MNIST(root=str(raw_dir), train=False, download=True)

    data_train = ds_train.data  # [60000, 28, 28] uint8
    labels_train = ds_train.targets.long()
    data_test_all = ds_test.data
    labels_test_all = ds_test.targets.long()

    total_train = int(data_train.shape[0])
    need = int(args.train_size) + int(args.val_size)
    if need > total_train:
        raise ValueError(f"Requested train+val={need} exceeds available {total_train}")

    perm = torch.randperm(total_train)
    train_idx = perm[: args.train_size]
    val_idx = perm[args.train_size : args.train_size + args.val_size]

    split_train = _build_split(data_train[train_idx], labels_train[train_idx], angles_seed=args.angles_seed)
    split_val = _build_split(data_train[val_idx], labels_train[val_idx], angles_seed=args.angles_seed + 1)
    if int(args.test_size) > int(data_test_all.shape[0]):
        raise ValueError(f"Requested test_size={args.test_size} exceeds available {int(data_test_all.shape[0])}")
    if int(args.test_size) < int(data_test_all.shape[0]):
        test_perm = torch.randperm(int(data_test_all.shape[0]))
        test_idx = test_perm[: int(args.test_size)]
        data_test = data_test_all[test_idx]
        labels_test = labels_test_all[test_idx]
    else:
        data_test = data_test_all
        labels_test = labels_test_all

    split_test = _build_split(data_test, labels_test, angles_seed=args.angles_seed + 2)

    train_path = processed_dir / "train.pt"
    val_path = processed_dir / "val.pt"
    test_path = processed_dir / "test.pt"

    _save_split(train_path, split_train, "train")
    _save_split(val_path, split_val, "val")
    _save_split(test_path, split_test, "test")

    subset_files: list[str] = []
    for n in args.subset_ns:
        for seed in args.subset_seeds:
            subset = _subset_indices_stratified(
                labels=split_train.labels,
                n_samples=int(n),
                seed=int(seed),
                num_classes=int(args.num_classes),
            )
            out = {
                "dataset_id": "rotmnist_v1",
                "split": "train",
                "n": int(n),
                "seed": int(seed),
                "num_classes": int(args.num_classes),
                "indices": subset["indices"],
                "class_counts": subset["class_counts"],
            }
            p = subset_dir / f"N{int(n):03d}_seed{int(seed):03d}.json"
            p.write_text(json.dumps(out, indent=2), encoding="utf-8")
            subset_files.append(str(p))

    metadata = {
        "dataset_id": "rotmnist_v1",
        "created_at": utc_now_iso(),
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "split_seed": int(args.split_seed),
        "angles_seed": int(args.angles_seed),
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "test_size": int(split_test.images.shape[0]),
        "num_classes": int(args.num_classes),
        "subset_ns": [int(x) for x in args.subset_ns],
        "subset_seeds": [int(x) for x in args.subset_seeds],
        "files": {
            "train_pt": str(train_path),
            "val_pt": str(val_path),
            "test_pt": str(test_path),
            "subset_dir": str(subset_dir),
            "subset_files": subset_files,
        },
    }
    metadata_path = processed_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    data_manifest = {
        "created_at": utc_now_iso(),
        "datasets": [
            {
                "dataset_id": "rotmnist_v1",
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
                ],
                "split_definition": {
                    "train_size": int(args.train_size),
                    "val_size": int(args.val_size),
                    "test_size": int(split_test.images.shape[0]),
                    "num_classes": int(args.num_classes),
                    "angles_seed": int(args.angles_seed),
                    "split_seed": int(args.split_seed),
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

    print("[prepare_rotmnist_lowdata] completed")
    print(f"  processed_dir={processed_dir}")
    print(f"  metadata={metadata_path}")
    print(f"  subsets={len(subset_files)}")


if __name__ == "__main__":
    main()
