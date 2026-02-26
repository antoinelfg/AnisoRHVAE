#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf  # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.data.ellipse_datamodule import EllipseSequenceDataModule
from src.utils.asset_manifest import dump_json, ensure_dir, sha256_file, utc_now_iso


@dataclass
class SplitBundle:
    images: torch.Tensor  # [N, 1, H, W]
    labels: torch.Tensor  # [N]


def _extract_frames(dataset: Any, frame_mode: str, batch_size: int, max_frames: int | None, seed: int) -> torch.Tensor:
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    chunks: list[torch.Tensor] = []
    for batch in loader:
        x = None
        if isinstance(batch, dict):
            x = batch.get("data", batch.get("x", batch.get("images")))
        elif isinstance(batch, (tuple, list)) and len(batch) > 0:
            x = batch[0]
        else:
            x = batch

        if not isinstance(x, torch.Tensor):
            continue

        if x.dim() == 5:
            if frame_mode == "all":
                x = x.reshape(-1, *x.shape[2:])
            else:
                x = x[:, 0]
        elif x.dim() != 4:
            continue
        chunks.append(x.detach().cpu())

    if not chunks:
        raise RuntimeError("No frames extracted from ellipse dataset.")

    frames = torch.cat(chunks, dim=0).float()
    if max_frames is not None and int(max_frames) > 0 and int(frames.shape[0]) > int(max_frames):
        g = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(int(frames.shape[0]), generator=g)[: int(max_frames)]
        frames = frames.index_select(0, idx)
    return frames


def _save_split(path: Path, bundle: SplitBundle, split_name: str, dataset_id: str, frame_mode: str) -> None:
    payload = {
        "images": bundle.images,
        "labels": bundle.labels,
        "split": split_name,
        "dataset_id": dataset_id,
        "frame_mode": frame_mode,
    }
    torch.save(payload, path)


def _subset_indices_uniform(n_total: int, n_samples: int, seed: int) -> dict[str, Any]:
    if int(n_samples) > int(n_total):
        raise ValueError(f"Requested subset n={n_samples} but only {n_total} train samples are available.")
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(int(n_total), size=int(n_samples), replace=False)
    idx = [int(i) for i in idx.tolist()]
    rng.shuffle(idx)
    return {"indices": idx}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare ellipse low-data benchmark assets (train/val/test + subsets).")
    p.add_argument("--processed_dir", type=str, default="data/processed/ellipses_lowdata_v1")
    p.add_argument("--dataset_id", type=str, default="ellipses_lowdata_v1")
    p.add_argument("--num_sequences", type=int, default=1600)
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--image_size", nargs=2, type=int, default=[64, 64])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--min_radius", type=int, default=8)
    p.add_argument("--max_radius", type=int, default=20)
    p.add_argument("--min_eccentricity", type=float, default=0.0)
    p.add_argument("--max_eccentricity", type=float, default=0.9)
    p.add_argument("--fix_center", action="store_true", default=True)
    p.add_argument("--no_fix_center", action="store_false", dest="fix_center")
    p.add_argument("--fix_theta", action="store_true", default=True)
    p.add_argument("--no_fix_theta", action="store_false", dest="fix_theta")
    p.add_argument("--fix_intensity", action="store_true", default=True)
    p.add_argument("--no_fix_intensity", action="store_false", dest="fix_intensity")
    p.add_argument("--keep_major_axis_constant", action="store_true", default=True)
    p.add_argument("--no_keep_major_axis_constant", action="store_false", dest="keep_major_axis_constant")
    p.add_argument("--keep_area_constant", action="store_true", default=False)
    p.add_argument("--outline_only", action="store_true", default=False)
    p.add_argument("--outline_width", type=int, default=2)
    p.add_argument("--antialias", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--frame_mode", type=str, choices=["t0", "all"], default="t0")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--subset_ns", nargs="+", type=int, default=[50, 100, 500, 1000])
    p.add_argument("--subset_seeds", nargs="+", type=int, default=[13, 29, 47, 71, 89])
    p.add_argument("--data_manifest_out", type=str, default="assets/manifests/data_manifest_ellipses.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    processed_dir = (ROOT / args.processed_dir).resolve()
    subset_dir = processed_dir / "subsets"
    ensure_dir(processed_dir)
    ensure_dir(subset_dir)

    dm_cfg = OmegaConf.create(
        {
            "num_sequences": int(args.num_sequences),
            "seq_len": int(args.seq_len),
            "sequence_length": int(args.seq_len),
            "image_size": [int(args.image_size[0]), int(args.image_size[1])],
            "batch_size": int(args.batch_size),
            "num_workers": 0,
            "min_radius": int(args.min_radius),
            "max_radius": int(args.max_radius),
            "min_eccentricity": float(args.min_eccentricity),
            "max_eccentricity": float(args.max_eccentricity),
            "fix_center": bool(args.fix_center),
            "fix_theta": bool(args.fix_theta),
            "fix_intensity": bool(args.fix_intensity),
            "keep_major_axis_constant": bool(args.keep_major_axis_constant),
            "keep_area_constant": bool(args.keep_area_constant),
            "outline_only": bool(args.outline_only),
            "outline_width": int(args.outline_width),
            "antialias": bool(args.antialias),
            "seed": int(args.seed),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
        }
    )

    dm = EllipseSequenceDataModule(dm_cfg)
    dm.setup("fit")

    train_frames = _extract_frames(
        dm.train_dataset,  # pyright: ignore[reportArgumentType]
        frame_mode=str(args.frame_mode),
        batch_size=int(args.batch_size),
        max_frames=args.max_frames,
        seed=int(args.seed),
    )
    val_frames = _extract_frames(
        dm.val_dataset,  # pyright: ignore[reportArgumentType]
        frame_mode=str(args.frame_mode),
        batch_size=int(args.batch_size),
        max_frames=args.max_frames,
        seed=int(args.seed) + 1,
    )
    test_frames = _extract_frames(
        dm.test_dataset,  # pyright: ignore[reportArgumentType]
        frame_mode=str(args.frame_mode),
        batch_size=int(args.batch_size),
        max_frames=args.max_frames,
        seed=int(args.seed) + 2,
    )

    train_bundle = SplitBundle(
        images=train_frames,
        labels=torch.zeros((train_frames.shape[0],), dtype=torch.long),
    )
    val_bundle = SplitBundle(
        images=val_frames,
        labels=torch.zeros((val_frames.shape[0],), dtype=torch.long),
    )
    test_bundle = SplitBundle(
        images=test_frames,
        labels=torch.zeros((test_frames.shape[0],), dtype=torch.long),
    )

    train_path = processed_dir / "train.pt"
    val_path = processed_dir / "val.pt"
    test_path = processed_dir / "test.pt"

    _save_split(train_path, train_bundle, "train", dataset_id=str(args.dataset_id), frame_mode=str(args.frame_mode))
    _save_split(val_path, val_bundle, "val", dataset_id=str(args.dataset_id), frame_mode=str(args.frame_mode))
    _save_split(test_path, test_bundle, "test", dataset_id=str(args.dataset_id), frame_mode=str(args.frame_mode))

    subset_files: list[str] = []
    n_train = int(train_bundle.images.shape[0])
    for n in args.subset_ns:
        for seed in args.subset_seeds:
            subset = _subset_indices_uniform(n_total=n_train, n_samples=int(n), seed=int(seed))
            payload = {
                "dataset_id": str(args.dataset_id),
                "split": "train",
                "n": int(n),
                "seed": int(seed),
                "indices": subset["indices"],
            }
            p = subset_dir / f"N{int(n):03d}_seed{int(seed):03d}.json"
            p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            subset_files.append(str(p))

    metadata = {
        "dataset_id": str(args.dataset_id),
        "created_at": utc_now_iso(),
        "processed_dir": str(processed_dir),
        "seed": int(args.seed),
        "frame_mode": str(args.frame_mode),
        "num_sequences": int(args.num_sequences),
        "seq_len": int(args.seq_len),
        "image_size": [int(args.image_size[0]), int(args.image_size[1])],
        "train_size": int(train_bundle.images.shape[0]),
        "val_size": int(val_bundle.images.shape[0]),
        "test_size": int(test_bundle.images.shape[0]),
        "subset_ns": [int(x) for x in args.subset_ns],
        "subset_seeds": [int(x) for x in args.subset_seeds],
        "files": {
            "train_pt": str(train_path),
            "val_pt": str(val_path),
            "test_pt": str(test_path),
            "subset_dir": str(subset_dir),
            "subset_files": subset_files,
        },
        "generation_config": {
            "min_radius": int(args.min_radius),
            "max_radius": int(args.max_radius),
            "min_eccentricity": float(args.min_eccentricity),
            "max_eccentricity": float(args.max_eccentricity),
            "fix_center": bool(args.fix_center),
            "fix_theta": bool(args.fix_theta),
            "fix_intensity": bool(args.fix_intensity),
            "keep_major_axis_constant": bool(args.keep_major_axis_constant),
            "keep_area_constant": bool(args.keep_area_constant),
            "outline_only": bool(args.outline_only),
            "outline_width": int(args.outline_width),
            "antialias": bool(args.antialias),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
        },
    }
    metadata_path = processed_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    data_manifest = {
        "created_at": utc_now_iso(),
        "datasets": [
            {
                "dataset_id": str(args.dataset_id),
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
                    "frame_mode": str(args.frame_mode),
                    "num_sequences": int(args.num_sequences),
                    "train_size": int(train_bundle.images.shape[0]),
                    "val_size": int(val_bundle.images.shape[0]),
                    "test_size": int(test_bundle.images.shape[0]),
                },
                "subset_definition": {
                    "subset_ns": [int(x) for x in args.subset_ns],
                    "subset_seeds": [int(x) for x in args.subset_seeds],
                    "subset_count": len(subset_files),
                    "sampling": "uniform_without_replacement",
                },
            }
        ],
    }
    dump_json((ROOT / args.data_manifest_out).resolve(), data_manifest)

    print("[prepare_ellipses_lowdata] completed")
    print(f"  processed_dir={processed_dir}")
    print(f"  train={tuple(train_bundle.images.shape)}")
    print(f"  val={tuple(val_bundle.images.shape)}")
    print(f"  test={tuple(test_bundle.images.shape)}")
    print(f"  subsets={len(subset_files)}")


if __name__ == "__main__":
    main()
