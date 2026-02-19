#!/usr/bin/env python3
"""Quick visualization of the blend coefficient alpha (void vs base) across latent space."""

from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.analyze_metric_full import load_metric_geometry


# === CONFIG (edit as needed) ===
MODEL_PATH = Path("outputs/pythae_rhvae_baseline/2026-02-09_17-27-43/")
BOUNDS = 6.0
RESOLUTION = 180
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

    # Compute min distance to centroids (euclidean)
    with torch.no_grad():
        centroids_t = centroids.to(device)
        diff = centroids_t.unsqueeze(0) - tensor_grid.unsqueeze(1)
        dists_sq = torch.einsum("bkd,bkd->bk", diff, diff)
        min_dists_sq = dists_sq.min(dim=1).values
        min_dists = torch.sqrt(min_dists_sq + 1e-10)
        alpha = model._compute_alpha(min_dists)

    alpha_grid = alpha.detach().cpu().numpy().reshape(RESOLUTION, RESOLUTION)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(
        alpha_grid,
        origin="lower",
        extent=[-BOUNDS, BOUNDS, -BOUNDS, BOUNDS],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    if SHOW_CENTROIDS:
        centroids_np = centroids[:, :2].detach().cpu().numpy()
        ax.scatter(
            centroids_np[:, 0],
            centroids_np[:, 1],
            s=12,
            c="white",
            edgecolor="black",
            linewidth=0.3,
            alpha=0.9,
        )
    ax.set_title("Blend Coefficient α (void vs base)")
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    fig.colorbar(im, ax=ax, label="alpha")
    fig.tight_layout()

    out_base = MODEL_PATH / "analysis_results"
    out_base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = out_base / f"alpha_map_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "alpha_map.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(out_path)


if __name__ == "__main__":
    main()
