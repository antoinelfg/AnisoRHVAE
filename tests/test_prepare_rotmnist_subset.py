from __future__ import annotations

import torch

from scripts.prepare_rotmnist_lowdata import _subset_indices_stratified


def test_subset_indices_stratified_counts() -> None:
    # 10 classes x 20 samples each.
    labels = torch.tensor([c for c in range(10) for _ in range(20)], dtype=torch.long)
    out = _subset_indices_stratified(labels=labels, n_samples=50, seed=13, num_classes=10)
    idx = out["indices"]
    counts = out["class_counts"]

    assert len(idx) == 50
    assert sum(int(v) for v in counts.values()) == 50
    # 50 over 10 classes => exactly 5/class.
    assert all(int(counts[str(c)]) == 5 for c in range(10))


def test_subset_indices_are_unique() -> None:
    labels = torch.tensor([c for c in range(10) for _ in range(20)], dtype=torch.long)
    out = _subset_indices_stratified(labels=labels, n_samples=100, seed=29, num_classes=10)
    idx = out["indices"]
    assert len(idx) == len(set(idx))
