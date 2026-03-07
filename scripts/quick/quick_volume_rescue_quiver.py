#!/usr/bin/env python3
"""Quick volume-rescue quiver: arrows = ∇(0.5 log det G^{-1})."""

from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.analyze_metric_full import load_metric_geometry


# === CONFIG (edit if needed) ===
MODEL_PATH = Path("outputs/pythae_rhvae_baseline/2026-01-29_17-11-14")
BOUNDS = 6.0
RESOLUTION = 140
QUIVER_STRIDE = 6
ARROW_SCALE = 0.85  # relative to grid spacing * stride
SHOW_CENTROIDS = True


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, centroids = load_metric_geometry(MODEL_PATH, device)

    axis = np.linspace(-BOUNDS, BOUNDS, RESOLUTION)
    X, Y = np.meshgrid(axis, axis)
    grid_2d = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)

    latent_dim = model.latent_dim
    if latent_dim > 2:
        padded = np.zeros((grid_2d.shape[0], latent_dim), dtype=np.float32)
        padded[:, :2] = grid_2d
        grid = padded
    else:
        grid = grid_2d

    tensor_grid = torch.from_numpy(grid).to(device)

    batch = 1024
    grads = []
    for i in range(0, tensor_grid.shape[0], batch):
        z = tensor_grid[i : i + batch].clone().detach().requires_grad_(True)
        G_inv = model.G_inv(z)
        logdet_inv = torch.linalg.slogdet(G_inv).logabsdet
        obj = 0.5 * logdet_inv
        grad = torch.autograd.grad(obj.sum(), z, create_graph=False)[0]
        grads.append(grad[:, :2].detach().cpu().numpy())

    grad_all = np.concatenate(grads, axis=0)
    U = grad_all[:, 0].reshape(RESOLUTION, RESOLUTION)
    V = grad_all[:, 1].reshape(RESOLUTION, RESOLUTION)

    # Normalize arrows for direction-only quiver
    norm = np.sqrt(U**2 + V**2)
    U = U / np.clip(norm, 1e-8, None)
    V = V / np.clip(norm, 1e-8, None)

    step = QUIVER_STRIDE
    Xq = X[::step, ::step]
    Yq = Y[::step, ::step]
    Uq = U[::step, ::step]
    Vq = V[::step, ::step]

    grid_spacing = (2 * BOUNDS) / (RESOLUTION - 1)
    arrow_scale = grid_spacing * step * ARROW_SCALE
    Uq = Uq * arrow_scale
    Vq = Vq * arrow_scale

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.quiver(
        Xq,
        Yq,
        Uq,
        Vq,
        color="crimson",
        alpha=0.9,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
    )
    if SHOW_CENTROIDS:
        centroids_np = centroids[:, :2].detach().cpu().numpy()
        ax.scatter(
            centroids_np[:, 0],
            centroids_np[:, 1],
            s=14,
            c="gray",
            alpha=0.35,
        )
    ax.set_title("Volume-Element Rescue Field (Quiver)")
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.set_aspect("equal")
    ax.set_xlim(-BOUNDS, BOUNDS)
    ax.set_ylim(-BOUNDS, BOUNDS)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_base = MODEL_PATH / "analysis_results"
    out_base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = out_base / f"volume_rescue_quiver_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "volume_rescue_quiver.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(out_path)


if __name__ == "__main__":
    main()
