#!/usr/bin/env python3
"""Check alignment between base-metric cheap direction and local tangent of centroids."""

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
RESOLUTION = 160
K_NEIGHBORS = 8
MAX_DIST = 1.2   # mask points farther than this from any centroid
SHOW_CENTROIDS = True


def _local_tangent(pts: torch.Tensor) -> torch.Tensor:
    """Compute principal direction (tangent) from local neighbor cloud.

    pts: [B, K, 2]
    returns: [B, 2]
    """
    mean = pts.mean(dim=1, keepdim=True)
    centered = pts - mean
    cov = centered.transpose(1, 2) @ centered / max(pts.shape[1] - 1, 1)
    # eigvals ascending; take largest eigenvector
    _, eigvecs = torch.linalg.eigh(cov)
    tangent = eigvecs[:, :, -1]
    return tangent


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, centroids = load_metric_geometry(MODEL_PATH, device)

    if model.latent_dim < 2:
        raise ValueError("Latent dim must be >= 2 for alignment plot.")

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
    centroids_2d = centroids[:, :2].to(device)

    batch = 1024
    cos_list = []
    min_dist_list = []

    for i in range(0, tensor_grid.shape[0], batch):
        z = tensor_grid[i : i + batch]

        # Base metric (no void blending)
        G_inv_base = model._compute_base_inverse_metric(z)
        G_base = torch.linalg.inv(G_inv_base)
        eigvals, eigvecs = torch.linalg.eigh(G_base)  # ascending
        e_min = eigvecs[:, :, 0]
        e_min_2d = e_min[:, :2]
        e_min_2d = e_min_2d / (torch.linalg.norm(e_min_2d, dim=1, keepdim=True) + 1e-8)

        # Local tangent from nearest centroids
        z2 = z[:, :2]
        dists = torch.cdist(z2, centroids_2d)
        min_dists = dists.min(dim=1).values
        _, idx = torch.topk(dists, k=min(K_NEIGHBORS, centroids_2d.shape[0]), largest=False)
        neighbors = centroids_2d[idx]  # [B, K, 2]
        tangent = _local_tangent(neighbors)
        tangent = tangent / (torch.linalg.norm(tangent, dim=1, keepdim=True) + 1e-8)

        cos = (e_min_2d * tangent).sum(dim=1).abs()
        cos_list.append(cos.detach().cpu().numpy())
        min_dist_list.append(min_dists.detach().cpu().numpy())

    cos_all = np.concatenate(cos_list)
    min_dist_all = np.concatenate(min_dist_list)

    alignment_map = cos_all.reshape(RESOLUTION, RESOLUTION)
    dist_map = min_dist_all.reshape(RESOLUTION, RESOLUTION)

    # Mask points too far from manifold
    alignment_map = alignment_map.copy()
    alignment_map[dist_map > MAX_DIST] = np.nan

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(
        alignment_map,
        origin="lower",
        extent=[-BOUNDS, BOUNDS, -BOUNDS, BOUNDS],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    if SHOW_CENTROIDS:
        centroids_np = centroids[:, :2].detach().cpu().numpy()
        ax.scatter(
            centroids_np[:, 0],
            centroids_np[:, 1],
            s=12,
            c="black",
            alpha=0.5,
        )
    ax.set_title("Base Metric Alignment vs Local Tangent (|cos|)")
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    fig.colorbar(im, ax=ax, label="|cos| (eigvec_min vs tangent)")
    fig.tight_layout()

    out_base = MODEL_PATH / "analysis_results"
    out_base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = out_base / f"base_tangent_alignment_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "base_tangent_alignment.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(out_path)


if __name__ == "__main__":
    main()
