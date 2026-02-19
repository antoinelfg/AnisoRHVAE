#!/usr/bin/env python3
"""Overlay rescue gradient quiver + cheap eigenvector directions on a heatmap."""

from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from scripts.analyze_metric_full import load_metric_geometry


# === CONFIG (edit if needed) ===
MODEL_PATH = Path("outputs/pythae_rhvae_baseline/2026-02-09_17-27-43/")
BOUNDS = 5.0
RESOLUTION = 160
QUIVER_STRIDE = 8
ARROW_SCALE = 0.9  # relative to grid spacing * stride
SHOW_CENTROIDS = True
HEATMAP = "logcond"  # "logcond", "lambda_min", "lambda_max"
EIGEN_MODE = "min"  # "min" or "max"
GRAD_MIN = 1e-4


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

    # ---- Rescue gradient (volume-element) ----
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
    U_res = grad_all[:, 0].reshape(RESOLUTION, RESOLUTION)
    V_res = grad_all[:, 1].reshape(RESOLUTION, RESOLUTION)
    grad_norm = np.sqrt(U_res ** 2 + V_res ** 2)
    U_res = U_res / np.clip(grad_norm, 1e-8, None)
    V_res = V_res / np.clip(grad_norm, 1e-8, None)

    # ---- Eigenvectors + cond(G) ----
    lam_min_list = []
    lam_max_list = []
    vec_list = []
    with torch.no_grad():
        for i in range(0, tensor_grid.shape[0], batch):
            z = tensor_grid[i : i + batch]
            G = model.G(z)
            eigvals, eigvecs = torch.linalg.eigh(G)  # ascending
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

    vec_2d = vec_all[:, :2]
    vec_norm = np.linalg.norm(vec_2d, axis=1, keepdims=True)
    vec_2d = vec_2d / np.clip(vec_norm, 1e-8, None)
    U_eig = vec_2d[:, 0].reshape(RESOLUTION, RESOLUTION)
    V_eig = vec_2d[:, 1].reshape(RESOLUTION, RESOLUTION)

    # ---- Downsample for quiver ----
    step = QUIVER_STRIDE
    Xq = X[::step, ::step]
    Yq = Y[::step, ::step]

    U_res_q = U_res[::step, ::step].copy()
    V_res_q = V_res[::step, ::step].copy()
    grad_norm_q = grad_norm[::step, ::step]
    mask = grad_norm_q >= GRAD_MIN
    U_res_q = np.where(mask, U_res_q, 0.0)
    V_res_q = np.where(mask, V_res_q, 0.0)

    U_eig_q = U_eig[::step, ::step]
    V_eig_q = V_eig[::step, ::step]

    grid_spacing = (2 * BOUNDS) / (RESOLUTION - 1)
    arrow_scale = grid_spacing * step * ARROW_SCALE
    U_res_q *= arrow_scale
    V_res_q *= arrow_scale
    U_eig_q *= arrow_scale
    V_eig_q *= arrow_scale

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(
        heat,
        origin="lower",
        extent=[-BOUNDS, BOUNDS, -BOUNDS, BOUNDS],
        cmap="magma",
    )

    ax.quiver(
        Xq, Yq, U_eig_q, V_eig_q,
        color="white", alpha=0.6, angles="xy",
        scale_units="xy", scale=1.0, width=0.0025,
    )
    ax.quiver(
        Xq, Yq, U_res_q, V_res_q,
        color="crimson", alpha=0.85, angles="xy",
        scale_units="xy", scale=1.0, width=0.003,
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

    ax.set_title(f"{heat_title} + eigvec({EIGEN_MODE}) vs rescue")
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label=heat_title)

    legend_items = [
        Line2D([0], [0], color="white", lw=2, label="cheap direction (eigvec)"),
        Line2D([0], [0], color="crimson", lw=2, label="rescue gradient"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=True, fontsize=8)

    fig.tight_layout()

    out_base = MODEL_PATH / "analysis_results"
    out_base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = out_base / f"overlay_rescue_eigvec_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rescue_vs_cheap.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(out_path)


if __name__ == "__main__":
    main()
