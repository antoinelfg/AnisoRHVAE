from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

try:
    from cleanfid import fid as cleanfid

    HAS_CLEANFID = True
except Exception:
    HAS_CLEANFID = False

try:
    import torchvision

    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False


def _as_2d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        return t.unsqueeze(0)
    if t.dim() > 2:
        return t.reshape(t.shape[0], -1)
    return t


def compute_d_rmse(path_points: torch.Tensor, real_points: torch.Tensor) -> float:
    """Root mean squared distance to the nearest real sample along a path."""
    p = _as_2d(path_points).float()
    r = _as_2d(real_points).float()
    if p.numel() == 0 or r.numel() == 0:
        return float("nan")
    d = torch.cdist(p, r)
    min_d = d.min(dim=1).values
    return float(torch.sqrt(torch.mean(min_d * min_d)).item())


def compute_interp_smoothness(images: torch.Tensor) -> float:
    """Normalized second finite-difference smoothness over interpolation frames."""
    x = images.float()
    if x.dim() < 3 or x.shape[0] < 3:
        return float("nan")
    flat = x.reshape(x.shape[0], -1)
    first = flat[1:] - flat[:-1]
    second = flat[2:] - 2.0 * flat[1:-1] + flat[:-2]
    numer = torch.linalg.vector_norm(second, dim=1).mean()
    denom = torch.linalg.vector_norm(first, dim=1).mean().clamp_min(1e-8)
    return float((numer / denom).item())


def compute_geo_euc_ratio(path_latents: torch.Tensor) -> float:
    """Geodesic-to-Euclidean proxy ratio based on path length over endpoint distance."""
    p = _as_2d(path_latents).float()
    if p.shape[0] < 2:
        return float("nan")
    seg = torch.linalg.norm(p[1:] - p[:-1], dim=1)
    geo = seg.sum()
    euc = torch.linalg.norm(p[-1] - p[0])
    if float(euc.item()) <= 1e-12:
        return float("nan")
    return float((geo / euc).item())


def compute_prd(real_samples: torch.Tensor, generated_samples: torch.Tensor, k: int = 5) -> dict[str, float]:
    """Improved precision/recall proxy using k-NN manifolds."""
    real = _as_2d(real_samples).float()
    gen = _as_2d(generated_samples).float()
    if real.shape[0] < (k + 1) or gen.shape[0] < (k + 1):
        return {"precision": float("nan"), "recall": float("nan")}

    k_real = min(int(k), int(real.shape[0]) - 1)
    k_gen = min(int(k), int(gen.shape[0]) - 1)
    if k_real <= 0 or k_gen <= 0:
        return {"precision": float("nan"), "recall": float("nan")}

    real_d = torch.cdist(real, real)
    real_d.fill_diagonal_(float("inf"))
    real_r = torch.topk(real_d, k_real, largest=False, dim=1).values[:, -1]

    gen_d = torch.cdist(gen, gen)
    gen_d.fill_diagonal_(float("inf"))
    gen_r = torch.topk(gen_d, k_gen, largest=False, dim=1).values[:, -1]

    cross = torch.cdist(gen, real)

    # Precision: generated sample lies inside the manifold of at least one real sample.
    precision = (cross <= real_r.unsqueeze(0)).any(dim=1).float().mean()

    # Recall: real sample lies inside the manifold of at least one generated sample.
    recall = (cross.t() <= gen_r.unsqueeze(0)).any(dim=1).float().mean()
    return {"precision": float(precision.item()), "recall": float(recall.item())}


def _to_nchw(images: torch.Tensor) -> torch.Tensor:
    x = images
    if x.dim() == 2:
        side = int(math.sqrt(x.shape[1]))
        x = x.view(-1, 1, side, side)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


def compute_fid(real_images: torch.Tensor, generated_images: torch.Tensor, device: torch.device) -> float:
    """FID with clean-fid if available; otherwise mean/variance approximation."""
    real = _to_nchw(real_images).to(device).float().clamp(0, 1)
    gen = _to_nchw(generated_images).to(device).float().clamp(0, 1)

    disable_cleanfid = os.environ.get("LOW_DATA_DISABLE_CLEANFID", "0") == "1"
    if HAS_CLEANFID and not disable_cleanfid:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            real_dir = Path(tmp) / "real"
            gen_dir = Path(tmp) / "gen"
            real_dir.mkdir(parents=True, exist_ok=True)
            gen_dir.mkdir(parents=True, exist_ok=True)

            real_use = F.interpolate(real, size=(299, 299), mode="bilinear", align_corners=False)
            gen_use = F.interpolate(gen, size=(299, 299), mode="bilinear", align_corners=False)

            n = min(int(real_use.shape[0]), int(gen_use.shape[0]), 2048)
            for i in range(n):
                rimg = (real_use[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gimg = (gen_use[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(rimg).save(real_dir / f"{i:05d}.png")
                Image.fromarray(gimg).save(gen_dir / f"{i:05d}.png")
            return float(cleanfid.compute_fid(str(real_dir), str(gen_dir), device=device))

    real_small = F.interpolate(real, size=(32, 32), mode="bilinear", align_corners=False)
    gen_small = F.interpolate(gen, size=(32, 32), mode="bilinear", align_corners=False)
    real_f = real_small.reshape(real_small.shape[0], -1)
    gen_f = gen_small.reshape(gen_small.shape[0], -1)

    mu_r = real_f.mean(dim=0)
    mu_g = gen_f.mean(dim=0)
    mean_diff = torch.sum((mu_r - mu_g) ** 2)

    var_r = real_f.var(dim=0)
    var_g = gen_f.var(dim=0)
    var_diff = torch.sum((torch.sqrt(var_r + 1e-8) - torch.sqrt(var_g + 1e-8)) ** 2)
    return float((mean_diff + var_diff).item())


class _TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_classifier(num_classes: int) -> nn.Module:
    if HAS_TORCHVISION:
        m = torchvision.models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    return _TinyCNN(num_classes=num_classes)


@dataclass
class AugmentationEvalResult:
    balanced_accuracy: float
    macro_f1: float


def augmentation_eval(
    real_train_x: torch.Tensor,
    real_train_y: torch.Tensor,
    synthetic_x: torch.Tensor,
    synthetic_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict[str, float]:
    """Train classifier on real+synthetic and evaluate Balanced Accuracy and Macro-F1."""
    x_train = torch.cat([real_train_x, synthetic_x], dim=0).float()
    y_train = torch.cat([real_train_y, synthetic_y], dim=0).long()

    num_classes = int(torch.max(torch.cat([y_train, test_y.long()], dim=0)).item()) + 1
    model = _build_classifier(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(max(1, int(epochs))):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(test_x.float().to(device))
        pred = torch.argmax(logits, dim=1).cpu().numpy()

    y_true = test_y.long().cpu().numpy()
    bal = float(balanced_accuracy_score(y_true, pred))
    macro = float(f1_score(y_true, pred, average="macro", zero_division=0))
    return {
        "balanced_accuracy": bal,
        "macro_f1": macro,
    }
