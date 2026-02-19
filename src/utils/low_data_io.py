from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import torch


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def load_split_tensor(processed_dir: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    path = processed_dir / f"{split}.pt"
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        images = payload.get("images")
        labels = payload.get("labels")
    else:
        raise TypeError(f"Unsupported split payload format: {path}")
    if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError(f"Invalid split payload in {path}")
    return images.float(), labels.long()


def load_subset_indices(processed_dir: Path, n: int, seed: int) -> list[int]:
    p = processed_dir / "subsets" / f"N{int(n):03d}_seed{int(seed):03d}.json"
    payload = json.loads(p.read_text(encoding="utf-8"))
    idx = payload.get("indices", [])
    return [int(i) for i in idx]


def subset_from_indices(images: torch.Tensor, labels: torch.Tensor, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.as_tensor(indices, dtype=torch.long)
    return images.index_select(0, idx), labels.index_select(0, idx)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def model_output_dir(base: Path, model_id: str, n: int, seed: int) -> Path:
    out = base / model_id / f"N{int(n):03d}" / f"seed{int(seed):03d}" / timestamp()
    out.mkdir(parents=True, exist_ok=True)
    return out
