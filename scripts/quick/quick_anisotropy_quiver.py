#!/usr/bin/env python3
"""Minimal anisotropy heatmap + eigenvector quiver for a trained metric."""

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
RESOLUTION = 160
QUIVER_STRIDE = 8
EIGEN_MODE = "min"  # "min" or "max"
HEATMAP = "logcond"  # "logcond", "lambda_min", "lambda_max"
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
    lam_min_list = []
    lam_max_list = []
    vec_list = []

    with torch.no_grad():
        for i in range(0, tensor_grid.shape[0], batch):
            z = tensor_grid[i : i + batch]
            G = model.G(z)
            eigvals, eigvecs = torch.linalg.eigh(G)  # ascending eigenvalues
            lam_min = eigvals[:, 0]
            lam_max = eigvals[:, -1]
            if EIGEN_MODE == "max":
                vec = eigvecs[:, :, -1]
            else:
                vec = eigvecs[:, :, 0]
            lam_min_list.append(lam_min.detach().cpu().numpy())
            lam_max_list.append(lam_max.detach().cpu().numpy())
            vec_list.append(vec.detach().cpu().numpy())

    lam_min_all = np.concatenate(lam_min_list)
    lam_max_all = np.concatenate(lam_max_list)
    vec_all = np.concatenate(vec_list, axis=0)

    lam_min_grid = lam_min_all.reshape(RESOLUTION, RESOLUTION)
    lam_max_grid = lam_max_all.reshape(RESOLUTION, RESOLUTION)

    if HEATMAP == "lambda_min":
        heat = lam_min_grid
        heat_title = "lambda_min(G)"
    elif HEATMAP == "lambda_max":
        heat = lam_max_grid
        heat_title = "lambda_max(G)"
    else:
        cond = lam_max_grid / np.clip(lam_min_grid, 1e-12, None)
        heat = np.log10(cond)
        heat_title = "log10 cond(G)"

    # Quiver (use first two components)
    vec_2d = vec_all[:, :2]
    norm = np.linalg.norm(vec_2d, axis=1, keepdims=True)
    vec_2d = vec_2d / np.clip(norm, 1e-8, None)
    U = vec_2d[:, 0].reshape(RESOLUTION, RESOLUTION)
    V = vec_2d[:, 1].reshape(RESOLUTION, RESOLUTION)

    step = QUIVER_STRIDE
    Xq = X[::step, ::step]
    Yq = Y[::step, ::step]
    Uq = U[::step, ::step]
    Vq = V[::step, ::step]

    # Scale arrows to the grid spacing
    grid_spacing = (2 * BOUNDS) / (RESOLUTION - 1)
    arrow_scale = grid_spacing * step * 0.85
    Uq = Uq * arrow_scale
    Vq = Vq * arrow_scale

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(
        heat,
        origin="lower",
        extent=[-BOUNDS, BOUNDS, -BOUNDS, BOUNDS],
        cmap="magma",
    )
    ax.quiver(
        Xq,
        Yq,
        Uq,
        Vq,
        color="white",
        alpha=0.8,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0025,
    )
    if SHOW_CENTROIDS:
        centroids_np = centroids[:, :2].detach().cpu().numpy()
        ax.scatter(
            centroids_np[:, 0],
            centroids_np[:, 1],
            s=14,
            c="cyan",
            edgecolor="white",
            linewidth=0.3,
            alpha=0.9,
        )
    ax.set_title(f"{heat_title} + eigvec({EIGEN_MODE})")
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    fig.colorbar(im, ax=ax, label=heat_title)
    fig.tight_layout()

    out_base = MODEL_PATH / "analysis_results"
    out_base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = out_base / f"anisotropy_quiver_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "anisotropy_quiver.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(out_path)


if __name__ == "__main__":
    main()
