#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "lib" / "src"))

from pythae.models.rhvae import RHVAE, RHVAEConfig
from src.models.rhvae_geometry import GeometryRHVAE, GeometryRHVAEConfig
from src.utils.low_data_io import (
    load_split_tensor,
    load_subset_indices,
    model_output_dir,
    subset_from_indices,
    write_json,
)
from src.utils.wandb_logging import (
    init_wandb_run,
    make_wandb_image,
    safe_wandb_finish,
    safe_wandb_log,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class RHVAETrainConfig:
    mode: str
    aniso_profile: str
    temperature_policy: str
    n: int
    seed: int
    latent_dim: int
    epochs: int
    batch_size: int
    lr: float
    temperature: float
    regularization: float
    drop_last: str
    auto_temperature: bool
    auto_temperature_stat: str
    auto_temperature_every: int
    temperature_scale: float
    n_lf: int
    eps_lf: float
    vis_every: int
    plot_heatmaps_during_training: bool
    metric_grid_res: int
    metric_tissot_grid_res: int
    radial_stretch: float


def _get_centroids_tensor(model: Any) -> torch.Tensor | None:
    if hasattr(model, "centroids_tens") and isinstance(model.centroids_tens, torch.Tensor):
        if model.centroids_tens.numel() > 0:
            return model.centroids_tens
    if hasattr(model, "centroids"):
        c = model.centroids
        if isinstance(c, torch.Tensor):
            if c.numel() > 0:
                return c
        elif hasattr(c, "__iter__"):
            c_list = list(c)
            if c_list:
                return torch.stack([x for x in c_list])
    return None


def _get_metric_mats(model: Any) -> torch.Tensor | None:
    if hasattr(model, "M_tens") and isinstance(model.M_tens, torch.Tensor) and model.M_tens.numel() > 0:
        return model.M_tens
    if hasattr(model, "M"):
        m = model.M
        if isinstance(m, torch.Tensor):
            return m
        if hasattr(m, "__iter__"):
            m_list = list(m)
            if m_list:
                return torch.stack([x for x in m_list])
    return None


def _build_model(args: argparse.Namespace, input_dim_flat: int) -> tuple[Any, dict[str, Any]]:
    if args.mode == "standard":
        cfg = RHVAEConfig(
            input_dim=(input_dim_flat,),
            latent_dim=args.latent_dim,
            n_lf=args.n_lf,
            eps_lf=args.eps_lf,
            beta_zero=0.3,
            temperature=args.temperature,
            regularization=args.regularization,
        )
        model = RHVAE(cfg)
        cfg_dict = cfg.__dict__
        return model, cfg_dict

    if args.aniso_profile == "gravity_well":
        # Exact parity with run_pythae_rhvae_baseline.py geometry_case=gravity_well
        radial_stretch = 10.0 if args.radial_stretch is None else float(args.radial_stretch)
        cfg = GeometryRHVAEConfig(
            input_dim=(input_dim_flat,),
            latent_dim=args.latent_dim,
            n_lf=args.n_lf,
            eps_lf=args.eps_lf,
            beta_zero=0.3,
            temperature=args.temperature,
            regularization=args.regularization,
            use_attractor=True,
            kernel_type="mahalanobis",
            atom_norm="trace",
            atom_power=1.0,
            kernel_power=1.0,
            precision_jitter=1e-6,
            void_threshold=1.5,
            void_weight_threshold=-1.0,
            transition_steepness=5.0,
            radial_stretch=radial_stretch,
            void_decay_type="invquad",
            void_decay_scale=1.0,
            void_decay_power=2.0,
            void_decay_softplus_k=5.0,
            attractor_metric="euclidean",
            attractor_smoothness="soft",
            attractor_use_det=True,
            attractor_k_nearest=10,
            attractor_gamma=5.0,
            attractor_bias_energy=15.0,
        )
    elif args.aniso_profile == "core4_spatial":
        # Match the selected core4 run (metric_core4_sweep/2026-02-17_11-07-08).
        radial_stretch = 9.252860449508804 if args.radial_stretch is None else float(args.radial_stretch)
        cfg = GeometryRHVAEConfig(
            input_dim=(input_dim_flat,),
            latent_dim=args.latent_dim,
            n_lf=args.n_lf,
            eps_lf=args.eps_lf,
            beta_zero=0.3,
            temperature=args.temperature,
            regularization=args.regularization,
            use_attractor=True,
            kernel_type="mahalanobis",
            atom_norm="trace",
            atom_power=1.0959864113997024,
            kernel_power=1.0,
            precision_jitter=0.01,
            void_threshold=1.2,
            void_weight_threshold=-1.0,
            transition_steepness=7.276725930229776,
            radial_stretch=radial_stretch,
            void_decay_type="invquad",
            void_decay_scale=8.87131893173153,
            void_decay_power=1.7447710342008058,
            void_decay_softplus_k=5.0,
            void_eigshape_mode="none",
            void_eigshape_alpha_min=1.0,
            void_eigshape_power=-1.0,
            void_eigshape_eig_floor=1e-8,
            attractor_metric="mahalanobis",
            attractor_smoothness="soft",
            attractor_use_det=True,
            attractor_k_nearest=1,
            attractor_gamma=7.624554217806352,
            attractor_bias_energy=18.0,
        )
    else:
        radial_stretch = 6.0 if args.radial_stretch is None else float(args.radial_stretch)
        cfg = GeometryRHVAEConfig(
            input_dim=(input_dim_flat,),
            latent_dim=args.latent_dim,
            n_lf=args.n_lf,
            eps_lf=args.eps_lf,
            beta_zero=0.3,
            temperature=args.temperature,
            regularization=args.regularization,
            use_attractor=True,
            kernel_type="mahalanobis",
            atom_norm="trace",
            atom_power=1.1,
            kernel_power=1.0,
            precision_jitter=0.01,
            void_threshold=1.2,
            void_weight_threshold=-1.0,
            transition_steepness=12.0,
            radial_stretch=radial_stretch,
            void_decay_type="invquad",
            void_decay_scale=4.5,
            void_decay_power=1.5,
            void_decay_softplus_k=5.0,
            attractor_metric="mahalanobis",
            attractor_smoothness="soft",
            attractor_use_det=True,
            attractor_k_nearest=2,
            attractor_gamma=25.0,
            attractor_bias_energy=18.0,
        )
    model = GeometryRHVAE(cfg)
    cfg_dict = cfg.__dict__
    return model, cfg_dict


def _train_epoch(model: Any, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for xb, _ in loader:
        xb = xb.to(device)
        optimizer.zero_grad()
        out = model({"data": xb})
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = int(xb.shape[0])
        total_loss += float(loss.item()) * bs
        n_samples += bs
    return float(total_loss / max(1, n_samples))


def _eval_epoch(model: Any, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.enable_grad():
        for xb, _ in loader:
            xb = xb.to(device).requires_grad_(True)
            out = model({"data": xb})
            bs = int(xb.shape[0])
            total_loss += float(out.loss.item()) * bs
            n_samples += bs
    return float(total_loss / max(1, n_samples))


def _suggest_temperature(
    centroids: torch.Tensor,
    sample_cap: int = 2000,
    stat: str = "median_nn",
) -> float | None:
    """Centroid-distance heuristic for RHVAE temperature."""
    if not isinstance(centroids, torch.Tensor) or centroids.shape[0] < 2:
        return None
    c = centroids.detach().cpu()
    if c.shape[0] > int(sample_cap):
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(c.shape[0], generator=g)[: int(sample_cap)]
        c = c.index_select(0, idx)
    dists = torch.cdist(c, c)
    dists.fill_diagonal_(float("inf"))
    if stat == "mean_pairwise":
        finite = dists[torch.isfinite(dists)]
        if finite.numel() == 0:
            return None
        return float(finite.mean().item())
    nn = dists.min(dim=1).values
    if nn.numel() == 0:
        return None
    if stat == "mean_nn":
        return float(nn.mean().item())
    return float(nn.median().item())


def _pad_grid_to_latent(grid_pts: np.ndarray, latent_dim: int, device: torch.device) -> torch.Tensor:
    zs = torch.tensor(grid_pts, dtype=torch.float32, device=device)
    if latent_dim > zs.shape[1]:
        pad = torch.zeros((zs.shape[0], latent_dim - zs.shape[1]), device=device, dtype=zs.dtype)
        zs = torch.cat([zs, pad], dim=1)
    elif latent_dim < zs.shape[1]:
        zs = zs[:, :latent_dim]
    return zs


def _metric_bounds_from_centroids(model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    centroids_2d = None
    c_tensor = _get_centroids_tensor(model)
    if isinstance(c_tensor, torch.Tensor):
        c_np = c_tensor.detach().cpu().numpy()
        if c_np.ndim == 2 and c_np.shape[1] >= 2:
            centroids_2d = c_np[:, :2]

    if centroids_2d is None or centroids_2d.size == 0:
        x_range = np.linspace(-5.0, 5.0, 40)
        y_range = np.linspace(-5.0, 5.0, 40)
        return x_range, y_range, np.array([[-5.0, 5.0], [-5.0, 5.0]]), None

    c_min = centroids_2d.min(axis=0)
    c_max = centroids_2d.max(axis=0)
    x_span = max(1e-6, float(c_max[0] - c_min[0]))
    y_span = max(1e-6, float(c_max[1] - c_min[1]))
    x_margin = max(0.5, 0.3 * x_span)
    y_margin = max(0.5, 0.3 * y_span)
    x_min = float(c_min[0] - x_margin)
    x_max = float(c_max[0] + x_margin)
    y_min = float(c_min[1] - y_margin)
    y_max = float(c_max[1] + y_margin)
    x_range = np.linspace(x_min, x_max, 40)
    y_range = np.linspace(y_min, y_max, 40)
    return x_range, y_range, np.array([[x_min, x_max], [y_min, y_max]]), centroids_2d


def _save_metric_field_plot(
    model: Any,
    epoch: int,
    output_dir: Path,
    wandb_run: Any | None,
    device: torch.device,
    grid_res: int,
) -> None:
    if int(getattr(model, "latent_dim", 0)) < 2:
        return

    model.eval()
    x_range, y_range, extent_arr, centroids_2d = _metric_bounds_from_centroids(model)
    x_range = np.linspace(float(extent_arr[0, 0]), float(extent_arr[0, 1]), int(grid_res))
    y_range = np.linspace(float(extent_arr[1, 0]), float(extent_arr[1, 1]), int(grid_res))
    X, Y = np.meshgrid(x_range, y_range)
    grid_pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    with torch.no_grad():
        zs = _pad_grid_to_latent(grid_pts, int(model.latent_dim), device)
        G_grid = model.G(zs)

    G_np = G_grid.detach().cpu().numpy()
    if G_np.ndim == 4:
        G_np = G_np.mean(axis=1)
    if G_np.shape[-1] > 2:
        G_np = G_np[:, :2, :2]

    if G_np.shape[0] != int(grid_res) * int(grid_res):
        model.train()
        return

    # Symmetrize in case of tiny numeric asymmetries before eigendecomposition.
    G_np = 0.5 * (G_np + np.transpose(G_np, (0, 2, 1)))
    eigvals = np.linalg.eigvalsh(G_np)
    eig_min = np.clip(eigvals[:, 0], 1e-8, None)
    eig_max = np.clip(eigvals[:, -1], 1e-8, None)
    cond = eig_max / eig_min
    logdet = np.log(np.abs(np.linalg.det(G_np)) + 1e-12)

    cond_grid = cond.reshape(int(grid_res), int(grid_res))
    logdet_grid = logdet.reshape(int(grid_res), int(grid_res))
    eig_min_grid = eig_min.reshape(int(grid_res), int(grid_res))
    eig_max_grid = eig_max.reshape(int(grid_res), int(grid_res))
    extent = [
        float(extent_arr[0, 0]),
        float(extent_arr[0, 1]),
        float(extent_arr[1, 0]),
        float(extent_arr[1, 1]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Epoch {int(epoch)}: RHVAE Metric Field", fontsize=14)

    im0 = axes[0, 0].imshow(cond_grid, extent=extent, origin="lower", cmap="viridis", aspect="auto")
    if centroids_2d is not None:
        axes[0, 0].scatter(centroids_2d[:, 0], centroids_2d[:, 1], c="red", s=15, alpha=0.7)
    axes[0, 0].set_title(f"Condition Number\nrange: [{cond.min():.2f}, {cond.max():.2f}]")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(logdet_grid, extent=extent, origin="lower", cmap="RdBu_r", aspect="auto")
    if centroids_2d is not None:
        axes[0, 1].scatter(centroids_2d[:, 0], centroids_2d[:, 1], c="black", s=15, alpha=0.7)
    axes[0, 1].set_title(f"log|det(G)|\nrange: [{logdet.min():.2f}, {logdet.max():.2f}]")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(eig_min_grid, extent=extent, origin="lower", cmap="plasma", aspect="auto")
    if centroids_2d is not None:
        axes[1, 0].scatter(centroids_2d[:, 0], centroids_2d[:, 1], c="white", s=15, alpha=0.7)
    axes[1, 0].set_title(f"Min Eigenvalue\nrange: [{eig_min.min():.3f}, {eig_min.max():.3f}]")
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(eig_max_grid, extent=extent, origin="lower", cmap="plasma", aspect="auto")
    if centroids_2d is not None:
        axes[1, 1].scatter(centroids_2d[:, 0], centroids_2d[:, 1], c="white", s=15, alpha=0.7)
    axes[1, 1].set_title(f"Max Eigenvalue\nrange: [{eig_max.min():.3f}, {eig_max.max():.3f}]")
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"metric_field_epoch_{int(epoch):03d}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    payload: dict[str, Any] = {
        "metric/condition_mean": float(np.mean(cond)),
        "metric/condition_max": float(np.max(cond)),
        "metric/logdet_mean": float(np.mean(logdet)),
        "metric/eig_min_mean": float(np.mean(eig_min)),
        "metric/eig_max_mean": float(np.mean(eig_max)),
    }
    wandb_img = make_wandb_image(save_path)
    if wandb_img is not None:
        payload["metric/field_image"] = wandb_img
    safe_wandb_log(wandb_run, payload, step=epoch)
    model.train()


def _save_metric_tissot_plot(
    model: Any,
    epoch: int,
    output_dir: Path,
    wandb_run: Any | None,
    device: torch.device,
    grid_res: int,
) -> None:
    if int(getattr(model, "latent_dim", 0)) < 2:
        return

    model.eval()
    x_range, y_range, extent_arr, centroids_2d = _metric_bounds_from_centroids(model)
    x_range = np.linspace(float(extent_arr[0, 0]), float(extent_arr[0, 1]), int(grid_res))
    y_range = np.linspace(float(extent_arr[1, 0]), float(extent_arr[1, 1]), int(grid_res))
    X, Y = np.meshgrid(x_range, y_range)
    grid_pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    with torch.no_grad():
        zs = _pad_grid_to_latent(grid_pts, int(model.latent_dim), device)
        G_inv = model.G_inv(zs)

    covs = G_inv[:, :2, :2].detach().cpu().numpy()
    covs = 0.5 * (covs + np.transpose(covs, (0, 2, 1)))

    ratios: list[float] = []
    ellipses: list[tuple[int, float, float, float]] = []
    for idx, cov in enumerate(covs):
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-8, None)
        ratio = float(eigvals[-1] / eigvals[0])
        ratios.append(ratio)
        width = 2.0 * float(np.sqrt(eigvals[-1]))
        height = 2.0 * float(np.sqrt(eigvals[0]))
        angle = float(np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1])))
        ellipses.append((idx, width, height, angle))

    ratios_arr = np.asarray(ratios, dtype=float)
    norm = mcolors.Normalize(vmin=float(np.min(ratios_arr)), vmax=float(np.max(ratios_arr)) + 1e-8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array(ratios_arr)

    fig, ax = plt.subplots(figsize=(8, 8))
    if centroids_2d is not None:
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c="black", alpha=0.4, s=18, label="Centroids")

    for idx, width, height, angle in ellipses:
        e = patches.Ellipse(
            (float(grid_pts[idx, 0]), float(grid_pts[idx, 1])),
            width,
            height,
            angle=angle,
            edgecolor=sm.to_rgba(ratios[idx]),
            facecolor="none",
            linewidth=1.0,
        )
        ax.add_patch(e)

    ax.set_xlim(float(x_range[0]), float(x_range[-1]))
    ax.set_ylim(float(y_range[0]), float(y_range[-1]))
    ax.set_aspect("equal")
    ax.set_title(f"Epoch {int(epoch)}: Metric Tissot Indicatrices")
    fig.colorbar(sm, ax=ax, label="Condition Number")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"metric_tissot_epoch_{int(epoch):03d}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    payload: dict[str, Any] = {
        "metric/tissot_condition_mean": float(np.mean(ratios_arr)),
        "metric/tissot_condition_max": float(np.max(ratios_arr)),
    }
    wandb_img = make_wandb_image(save_path)
    if wandb_img is not None:
        payload["metric/tissot_image"] = wandb_img
    safe_wandb_log(wandb_run, payload, step=epoch)
    model.train()


def _update_metric_epoch(model: Any) -> tuple[int, int]:
    """Refresh the RHVAE global metric from buffered batch statistics."""
    m_buf = getattr(model, "M", None)
    c_buf = getattr(model, "centroids", None)
    if m_buf is not None and c_buf is not None and len(m_buf) == 0 and len(c_buf) == 0:
        centroids = _get_centroids_tensor(model)
        mats = _get_metric_mats(model)
        n_centroids = int(centroids.shape[0]) if isinstance(centroids, torch.Tensor) else 0
        n_atoms = int(mats.shape[0]) if isinstance(mats, torch.Tensor) else 0
        return n_centroids, n_atoms

    if hasattr(model, "update"):
        with torch.no_grad():
            model.update()
    centroids = _get_centroids_tensor(model)
    mats = _get_metric_mats(model)
    n_centroids = int(centroids.shape[0]) if isinstance(centroids, torch.Tensor) else 0
    n_atoms = int(mats.shape[0]) if isinstance(mats, torch.Tensor) else 0
    return n_centroids, n_atoms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RHVAE variants from RotMNIST tensor splits.")
    parser.add_argument("--mode", type=str, required=True, choices=["standard", "aniso"])
    parser.add_argument(
        "--aniso_profile",
        type=str,
        default="legacy",
        choices=["legacy", "gravity_well", "core4_spatial"],
        help="Geometry preset for --mode aniso.",
    )
    parser.add_argument("--processed_dir", type=str, default="data/processed/rotmnist/v1")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--drop_last", type=str, default="auto", choices=["auto", "true", "false"])
    parser.add_argument(
        "--temperature_policy",
        type=str,
        default="fixed",
        choices=["fixed", "auto_once", "auto_every"],
        help=(
            "Temperature schedule policy. "
            "'fixed' keeps T constant for the full run; "
            "'auto_once' sets T once from centroid distances; "
            "'auto_every' refreshes T every N epochs."
        ),
    )
    parser.add_argument("--auto_temperature", action="store_true")
    parser.add_argument(
        "--auto_temperature_stat",
        type=str,
        default="median_nn",
        choices=["median_nn", "mean_nn", "mean_pairwise"],
    )
    parser.add_argument(
        "--auto_temperature_every",
        type=int,
        default=0,
        help="0 sets temperature once after metric update; >0 updates every N epochs.",
    )
    parser.add_argument("--temperature_scale", type=float, default=1.0)
    parser.add_argument("--vis_every", type=int, default=10)
    parser.add_argument("--plot_heatmaps_during_training", action="store_true")
    parser.add_argument("--metric_grid_res", type=int, default=40)
    parser.add_argument("--metric_tissot_grid_res", type=int, default=16)
    parser.add_argument("--n_lf", type=int, default=3)
    parser.add_argument("--eps_lf", type=float, default=1e-3)
    parser.add_argument("--radial_stretch", type=float, default=None)
    parser.add_argument("--output_root", type=str, default="outputs/low_data_models/rotmnist_v1")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_name_mode", type=str, default="auto", choices=["timestamp", "auto", "manual"])
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    effective_temperature_policy = str(args.temperature_policy)
    if bool(args.auto_temperature):
        # Backward compatibility with previous flag.
        effective_temperature_policy = "auto_every" if int(args.auto_temperature_every) > 0 else "auto_once"
    effective_auto_every = int(args.auto_temperature_every)
    if effective_temperature_policy == "auto_every" and effective_auto_every <= 0:
        effective_auto_every = 1

    model_id = args.model_id
    if model_id is None:
        model_id = "rhvae_standard" if args.mode == "standard" else "aniso"

    wandb_run = init_wandb_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        tags=args.wandb_tags,
        run_name=args.wandb_run_name,
        name_mode=args.wandb_name_mode,
        mode=args.wandb_mode,
        config=vars(args),
        name_prefix=f"lowdata_{model_id}_N{int(args.n):03d}_seed{int(args.seed):03d}",
    )

    try:
        processed_dir = (ROOT / args.processed_dir).resolve()
        train_images, train_labels = load_split_tensor(processed_dir, "train")
        val_images, val_labels = load_split_tensor(processed_dir, "val")

        subset_idx = load_subset_indices(processed_dir, args.n, args.seed)
        train_images, train_labels = subset_from_indices(train_images, train_labels, subset_idx)

        x_train = train_images.reshape(train_images.shape[0], -1)
        x_val = val_images.reshape(val_images.shape[0], -1)

        if args.drop_last == "true":
            drop_last = True
        elif args.drop_last == "false":
            drop_last = False
        else:
            drop_last = int(x_train.shape[0]) >= int(args.batch_size)

        if drop_last and int(x_train.shape[0]) < int(args.batch_size):
            drop_last = False

        print(
            f"[train_rhvae_tensor] loader: train_n={int(x_train.shape[0])} batch_size={int(args.batch_size)} "
            f"drop_last={drop_last}"
        )
        train_loader = DataLoader(
            TensorDataset(x_train, train_labels),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=drop_last,
        )
        val_loader = DataLoader(TensorDataset(x_val, val_labels), batch_size=args.batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg_dict = _build_model(args, input_dim_flat=x_train.shape[1])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        out_dir = model_output_dir((ROOT / args.output_root).resolve(), model_id, args.n, args.seed)

        history: list[dict[str, float]] = []
        best_val = float("inf")
        auto_temperature_set = False
        print(
            "[train_rhvae_tensor] temperature_policy="
            f"{effective_temperature_policy} initial_T={float(args.temperature):.6f}"
        )

        for epoch in range(1, args.epochs + 1):
            train_loss = _train_epoch(model, train_loader, device, optimizer)
            n_centroids, n_atoms = _update_metric_epoch(model)
            auto_suggested: float | None = None
            if effective_temperature_policy == "fixed":
                should_update_temp = False
            elif effective_temperature_policy == "auto_once":
                should_update_temp = not auto_temperature_set
            elif effective_temperature_policy == "auto_every":
                should_update_temp = (epoch % max(1, int(effective_auto_every))) == 0
            else:
                raise ValueError(f"Unsupported temperature_policy={effective_temperature_policy}")
            if should_update_temp:
                centroids_now = _get_centroids_tensor(model)
                auto_suggested = _suggest_temperature(
                    centroids_now,
                    stat=str(args.auto_temperature_stat),
                )
                if auto_suggested is not None:
                    new_t = max(1e-6, float(args.temperature_scale) * auto_suggested)
                    with torch.no_grad():
                        model.temperature.fill_(new_t)
                    auto_temperature_set = True
                    print(
                        "[train_rhvae_tensor] auto temperature set: "
                        f"T={new_t:.4f} (stat={args.auto_temperature_stat}, base={auto_suggested:.4f})"
                    )

            val_loss = _eval_epoch(model, val_loader, device)
            best_val = min(best_val, val_loss)
            scheduler.step(val_loss)
            model.train()
            current_temp = (
                float(model.temperature.detach().cpu().item())
                if hasattr(model, "temperature")
                else float(args.temperature)
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "n_centroids": float(n_centroids),
                    "n_atoms": float(n_atoms),
                    "temperature": current_temp,
                }
            )
            print(f"[train_rhvae_tensor] mode={args.mode} epoch={epoch} train={train_loss:.4f} val={val_loss:.4f}")
            wandb_payload: dict[str, Any] = {
                "train/loss": float(train_loss),
                "val/loss": float(val_loss),
                "metric/n_centroids": int(n_centroids),
                "metric/n_atoms": int(n_atoms),
                "metric/temperature": float(current_temp),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "train/epoch": int(epoch),
                "meta/n": int(args.n),
                "meta/seed": int(args.seed),
                "meta/model_id": str(model_id),
                "meta/mode": str(args.mode),
            }
            if auto_suggested is not None:
                wandb_payload["metric/temperature_suggested"] = float(auto_suggested)
            safe_wandb_log(wandb_run, wandb_payload, step=epoch)

            if args.plot_heatmaps_during_training and (
                epoch == 1 or epoch % max(1, int(args.vis_every)) == 0 or epoch == int(args.epochs)
            ):
                _save_metric_field_plot(
                    model=model,
                    epoch=epoch,
                    output_dir=out_dir,
                    wandb_run=wandb_run,
                    device=device,
                    grid_res=int(args.metric_grid_res),
                )
                _save_metric_tissot_plot(
                    model=model,
                    epoch=epoch,
                    output_dir=out_dir,
                    wandb_run=wandb_run,
                    device=device,
                    grid_res=int(args.metric_tissot_grid_res),
                )

        torch.save(model.state_dict(), out_dir / "rhvae_model.pt")

        centroids = _get_centroids_tensor(model)
        mats = _get_metric_mats(model)
        if centroids is None or mats is None:
            raise RuntimeError("Failed to extract RHVAE metric tensors after training.")
        if int(centroids.shape[0]) < 2 or int(mats.shape[0]) < 2:
            raise RuntimeError(
                "Metric update failed: expected >=2 centroids/atoms, got "
                f"{tuple(centroids.shape)} and {tuple(mats.shape)}. "
                "This usually indicates missing/failed model.update() during training."
            )
        if int(mats.shape[0]) != int(centroids.shape[0]):
            raise RuntimeError(
                "Invalid metric payload: centroid and atom counts differ "
                f"({int(centroids.shape[0])} vs {int(mats.shape[0])})."
            )

        temp = float(model.temperature.detach().cpu().item()) if hasattr(model, "temperature") else float(args.temperature)
        reg = float(model.lbd.detach().cpu().item()) if hasattr(model, "lbd") else float(args.regularization)

        metric_payload = {
            "centroids": centroids.detach().cpu(),
            "metric_matrices": mats.detach().cpu(),
            "temperature": temp,
            "regularization": reg,
            "config": cfg_dict,
        }
        torch.save(metric_payload, out_dir / "rhvae_metric.pt")

        torch.save(
            {
                "images": train_images,
                "labels": train_labels,
                "subset_indices": subset_idx,
            },
            out_dir / "train_data.pt",
        )

        cfg_out = RHVAETrainConfig(
            mode=args.mode,
            aniso_profile=str(args.aniso_profile),
            temperature_policy=str(effective_temperature_policy),
            n=args.n,
            seed=args.seed,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            regularization=args.regularization,
            drop_last=args.drop_last,
            auto_temperature=bool(args.auto_temperature),
            auto_temperature_stat=str(args.auto_temperature_stat),
            auto_temperature_every=int(args.auto_temperature_every),
            temperature_scale=float(args.temperature_scale),
            n_lf=args.n_lf,
            eps_lf=args.eps_lf,
            vis_every=int(args.vis_every),
            plot_heatmaps_during_training=bool(args.plot_heatmaps_during_training),
            metric_grid_res=int(args.metric_grid_res),
            metric_tissot_grid_res=int(args.metric_tissot_grid_res),
            radial_stretch=float(cfg_dict.get("radial_stretch", 0.0)),
        )
        write_json(out_dir / "config.json", asdict(cfg_out))

        metrics = {
            "final_train_loss": history[-1]["train_loss"] if history else float("nan"),
            "final_val_loss": history[-1]["val_loss"] if history else float("nan"),
            "best_val_loss": best_val,
            "history": history,
        }
        write_json(out_dir / "metrics.json", metrics)

        safe_wandb_log(
            wandb_run,
            {
                "final/train_loss": metrics["final_train_loss"],
                "final/val_loss": metrics["final_val_loss"],
                "final/best_val_loss": metrics["best_val_loss"],
                "meta/output_dir": str(out_dir),
                "meta/model_id": str(model_id),
                "meta/mode": str(args.mode),
            },
        )
        if wandb_run is not None:
            wandb_run.summary["output_dir"] = str(out_dir)
            wandb_run.summary["n"] = int(args.n)
            wandb_run.summary["seed"] = int(args.seed)
            wandb_run.summary["model_id"] = str(model_id)
            wandb_run.summary["mode"] = str(args.mode)

        print(f"[train_rhvae_tensor] output_dir={out_dir}")
    finally:
        safe_wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
