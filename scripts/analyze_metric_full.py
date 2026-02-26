#!/usr/bin/env python3
"""Comprehensive RHVAE metric analysis suite.

Visualizes the learned geometry, compares paths, inspects field dynamics,
and reports quantitative statistics that describe how the metric distorts the
latent space.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import matplotlib.colors as mcolors  # pyright: ignore[reportMissingImports]
import matplotlib.patches as patches  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import seaborn as sns  # pyright: ignore[reportMissingModuleSource]
import torch  # pyright: ignore[reportMissingImports]
from mpl_toolkits.mplot3d import Axes3D  # pyright: ignore[reportMissingImports]

try:
    import wandb
except ImportError:
    wandb = None
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.rhvae_geometry import GeometryRHVAE, GeometryRHVAEConfig
from src.models.samplers.hmc_sampler import (
    RiemannianHMCSampler,
    GeodesicHMCSampler,
    RHVAEVolumeElementHMCSampler,
    VolumeElementRiemannianHMCSampler,
    RHVAELogDetHMCSampler,
    DualRiemannianHMCSampler,
)
from src.utils.metric_helpers import load_metric_bundle

sns.set_theme(style="whitegrid")


def build_output_dir(base: Path | str) -> Path:
    base_path = Path(base)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base_path / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def pad_latent(points: torch.Tensor, latent_dim: int) -> torch.Tensor:
    if latent_dim == points.shape[1]:
        return points
    padded = torch.zeros(points.shape[0], latent_dim, device=points.device, dtype=points.dtype)
    padded[:, : points.shape[1]] = points
    return padded


def encode_support_images(
    model: GeometryRHVAE,
    images: torch.Tensor,
    max_support: int = 600,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = images[:max_support].clone()
    device = next(model.parameters()).device
    with torch.no_grad():
        flat = images.view(images.shape[0], -1).to(device)
        enc_out = model.encoder(flat)
        latents = enc_out.embedding
    return images.cpu(), latents


def plot_geodesic_image_sequences(
    batch_data: list[dict[str, np.ndarray]],
    support_images: torch.Tensor,
    support_latents: torch.Tensor,
    out_dir: Path,
    wandb_run: Optional[Any] = None,
    max_paths: int = 4,
    frames_per_path: int = 5,
) -> None:
    if not batch_data or support_images is None or support_latents is None:
        return
    rows = min(len(batch_data), max_paths)
    cols = frames_per_path
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.atleast_2d(axes)
    if axes.shape[0] != rows or axes.shape[1] != cols:
        axes = axes.reshape(rows, cols)
    support_images_cpu = support_images
    sup_latents = support_latents
    for row in range(rows):
        data = batch_data[row]
        geodesic_path = data["geodesic"]
        if geodesic_path.shape[0] == 0:
            continue
        indices = np.linspace(0, geodesic_path.shape[0] - 1, cols, dtype=int)
        for col, idx in enumerate(indices):
            point = torch.tensor(geodesic_path[idx], device=sup_latents.device)
            dists = torch.norm(point - sup_latents, dim=1)
            best = int(dists.argmin().item())
            img = support_images_cpu[best]
            ax = axes[row, col]
            img_plot = img.permute(1, 2, 0).numpy()
            ax.imshow(np.clip(img_plot, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(data.get("description", "pair").capitalize())
    plt.tight_layout()
    save_plot(
        fig,
        out_dir / "geodesic_image_sequences.png",
        "analysis/geodesic_images",
        wandb_run,
    )
def save_plot(
    fig: plt.Figure,
    path: Path,
    tag: str,
    wandb_run: Optional[Any] = None,
) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if wandb_run is not None and wandb is not None:
        wandb_run.log({tag: wandb.Image(fig)})
    plt.close(fig)


def init_wandb(args) -> Optional[Any]:
    if wandb is None or not args.wandb_project:
        return None
    tags = [tag.strip() for tag in args.wandb_tags.split(",")] if args.wandb_tags else None
    run_name = args.wandb_run_name
    if args.wandb_name_mode == "timestamp" or run_name is None:
        run_name = f"analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {
        "model_path": args.model_path,
        "device": args.device,
        "skip_distortion": args.skip_distortion,
        "far_pairs": args.far_pairs,
        "random_pairs": args.random_pairs,
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_name,
        tags=tags,
        config=config,
    )
    return run


def load_metric_geometry(model_path: Path | str, device: torch.device) -> tuple[GeometryRHVAE, torch.Tensor]:
    folder = Path(model_path)
    if folder.suffix == ".pt":
        folder = folder.parent
    if not folder.exists():
        raise FileNotFoundError(f"Model path {folder} does not exist")

    centroids, atoms, temperature, regularization, cfg_dict = load_metric_bundle(folder / "rhvae_metric.pt")
    cfg_kwargs = dict(cfg_dict) if cfg_dict is not None else {}
    cfg_kwargs.setdefault("input_dim", (centroids.shape[1],))
    cfg_kwargs.setdefault("latent_dim", centroids.shape[1])
    cfg_kwargs.setdefault("temperature", float(temperature))
    cfg_kwargs.setdefault("regularization", float(regularization))
    if not isinstance(cfg_kwargs["input_dim"], tuple):
        cfg_kwargs["input_dim"] = tuple(cfg_kwargs["input_dim"])

    allowed_fields = {
        name
        for name, field in GeometryRHVAEConfig.__dataclass_fields__.items()
        if field.init
    }
    filtered_kwargs = {k: v for k, v in cfg_kwargs.items() if k in allowed_fields}
    config = GeometryRHVAEConfig(**filtered_kwargs)
    model = GeometryRHVAE(config).to(device)

    if atoms is None:
        raise ValueError("Metric atoms missing in metric bundle.")

    model.set_centroids(centroids.to(device))
    model.set_atoms(torch.as_tensor(atoms, dtype=torch.float32).to(device))
    model.temperature.data = torch.tensor(float(temperature), device=device)
    model.lbd.data = torch.tensor(float(regularization), device=device)
    model._refresh_metric_hooks()
    model_pt = folder / "rhvae_model.pt"
    if model_pt.exists():
        state_dict = torch.load(model_pt, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, centroids.to(device)


def compute_path_length(model: GeometryRHVAE, path: torch.Tensor) -> float:
    with torch.no_grad():
        deltas = path[1:] - path[:-1]
        mids = 0.5 * (path[1:] + path[:-1])
        G_mid = model.G(mids)
        integrand = torch.einsum("ni,nij,nj->n", deltas, G_mid, deltas)
        lengths = torch.sqrt(torch.clamp(integrand, min=0.0))
        return lengths.sum().item()


def compute_path_energy(model: GeometryRHVAE, path: torch.Tensor) -> torch.Tensor:
    """Per-segment metric energy along a discrete path."""
    with torch.no_grad():
        deltas = path[1:] - path[:-1]
        mids = 0.5 * (path[1:] + path[:-1])
        G_mid = model.G(mids)
        energy = torch.einsum("ni,nij,nj->n", deltas, G_mid, deltas)
    return energy


def optimize_geodesic(
    model: GeometryRHVAE,
    start: torch.Tensor,
    end: torch.Tensor,
    n_points: int = 48,
    iterations: int = 800,
    lr: float = 0.02,
) -> torch.Tensor:
    device = start.device
    start = start.reshape(-1)
    end = end.reshape(-1)
    t = torch.linspace(0, 1, n_points, device=device).unsqueeze(1)
    init_path = start + t * (end - start)
    inner = init_path[1:-1].clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([inner], lr=lr)

    for _ in range(iterations):
        optimizer.zero_grad()
        full = torch.cat([start.unsqueeze(0), inner, end.unsqueeze(0)], dim=0)
        deltas = full[1:] - full[:-1]
        mids = 0.5 * (full[1:] + full[:-1])
        G_mid = model.G(mids)
        energy = torch.einsum("ni,nij,nj->n", deltas, G_mid, deltas).sum()
        energy.backward()
        optimizer.step()

    with torch.no_grad():
        full = torch.cat([start.unsqueeze(0), inner.detach(), end.unsqueeze(0)], dim=0)
    return full


def metric_field_tissot(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    out_dir: Path,
    grid_bounds: float = 4.0,
    resolution: int = 18,
    wandb_run: Optional[Any] = None,
    step: int = 0,
) -> None:
    print("-> Metric Field Tissot Indicatrices")
    latent_dim = model.latent_dim
    c_np = centroids[:, :2].detach().cpu().numpy()
    bounds = max(grid_bounds, np.max(np.abs(c_np)) + 2.0)
    axis = np.linspace(-bounds, bounds, resolution)
    X, Y = np.meshgrid(axis, axis)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

    with torch.no_grad():
        tensor_grid = torch.from_numpy(grid_points.astype(np.float32)).to(centroids.device)
        tensor_full = pad_latent(tensor_grid, latent_dim)
        G_inv = model.G_inv(tensor_full)

    covariance = G_inv[:, :2, :2].cpu().numpy()
    ellipses = []
    ratios = []
    for idx, cov in enumerate(covariance):
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-8, None)
        ratio = float(eigvals[1] / eigvals[0])
        ratios.append(ratio)
        width = 2 * np.sqrt(eigvals[1])
        height = 2 * np.sqrt(eigvals[0])
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        ellipses.append(
            {
                "center": (grid_points[idx, 0], grid_points[idx, 1]),
                "width": width,
                "height": height,
                "angle": angle,
                "ratio": ratio,
            }
        )

    ratios_arr = np.array(ratios)
    norm = mcolors.Normalize(vmin=ratios_arr.min(), vmax=ratios_arr.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array(ratios_arr)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(c_np[:, 0], c_np[:, 1], c="black", alpha=0.15, s=12, label="Centroids")

    for ellipse in ellipses:
        color = sm.to_rgba(ellipse["ratio"])
        patch = patches.Ellipse(
            ellipse["center"],
            ellipse["width"],
            ellipse["height"],
            angle=ellipse["angle"],
            edgecolor=color,
            facecolor="none",
            linewidth=1.2,
        )
        ax.add_patch(patch)

    if resolution > 6:
        arrow_step = max(1, resolution // 6)
        grid_idx = np.arange(0, resolution, arrow_step)
        iy, ix = np.meshgrid(grid_idx, grid_idx, indexing="xy")
        arrow_indices = (iy * resolution + ix).ravel()
        arrow_points = grid_points[arrow_indices]
        arrow_tensor = torch.from_numpy(arrow_points.astype(np.float32)).to(centroids.device)
        arrow_full = pad_latent(arrow_tensor, latent_dim)
        with torch.no_grad():
            G_inv_arrows = model.G_inv(arrow_full)[:, :2, :2]
            eigvals, eigvecs = torch.linalg.eigh(G_inv_arrows)
        principal_dirs = eigvecs[:, :, -1].cpu().numpy()
        principal_lens = np.sqrt(np.clip(eigvals[:, -1].cpu().numpy(), 0.0, None))
        arrow_scale = 0.5 * (bounds / resolution)
        arrow_u = principal_dirs[:, 0] * principal_lens * arrow_scale
        arrow_v = principal_dirs[:, 1] * principal_lens * arrow_scale
        ax.quiver(
            arrow_points[:, 0],
            arrow_points[:, 1],
            arrow_u,
            arrow_v,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="gray",
            width=0.003,
            alpha=0.6,
        )

    ax.set_title("Metric Field (Tissot Indicatrices)")
    ax.set_xlim(axis[0], axis[-1])
    ax.set_ylim(axis[0], axis[-1])
    ax.set_aspect("equal")
    fig.colorbar(sm, ax=ax, label="Condition Number")
    save_plot(
        fig,
        out_dir / "metric_field_tissot.png",
        "analysis/metric_tissot",
        wandb_run,
    )


def logdet_landscape(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    out_dir: Path,
    grid_bounds: float = 4.0,
    resolution: int = 140,
    wandb_run: Optional[Any] = None,
    step: int = 0,
) -> None:
    print("-> Log-Determinant Landscape")
    latent_dim = model.latent_dim
    c_np = centroids[:, :2].detach().cpu().numpy()
    bounds = max(grid_bounds, np.max(np.abs(c_np)) * 1.2)
    axis = np.linspace(-bounds, bounds, resolution)
    X, Y = np.meshgrid(axis, axis)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

    with torch.no_grad():
        tensor_grid = torch.from_numpy(grid_points.astype(np.float32)).to(centroids.device)
        tensor_full = pad_latent(tensor_grid, latent_dim)
        G = model.G(tensor_full)
        sign, logabsdet = torch.linalg.slogdet(G)
        logdet = logabsdet

    logdet_grid = logdet.cpu().numpy().reshape(resolution, resolution)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        logdet_grid,
        origin='lower',
        extent=[-bounds, bounds, -bounds, bounds],
        aspect='auto',
        cmap='magma',
    )
    ax.scatter(c_np[:, 0], c_np[:, 1], c='cyan', edgecolor='white', s=25, label='Centroids')
    ax.set_title('Log-Determinant Landscape (Volume Element)')
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    fig.colorbar(im, ax=ax, label='log(det G)')
    save_plot(
        fig,
        out_dir / 'logdet_landscape.png',
        'analysis/logdet_landscape',
        wandb_run,
    )


def metric_surface_3d(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    out_dir: Path,
    grid_bounds: float = 4.0,
    resolution: int = 160,
    tag_suffix: str = "",
    file_suffix: str = "",
    wandb_run: Optional[Any] = None,
) -> None:
    print("-> Metric Surface (3D + contours)")
    latent_dim = model.latent_dim
    c_np = centroids[:, :2].detach().cpu().numpy()
    bounds = max(grid_bounds, np.max(np.abs(c_np)) + 3.0)
    axis = np.linspace(-bounds, bounds, resolution)
    X, Y = np.meshgrid(axis, axis)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

    with torch.no_grad():
        tensor_grid = torch.from_numpy(grid_points.astype(np.float32)).to(centroids.device)
        tensor_full = pad_latent(tensor_grid, latent_dim)
        G = model.G(tensor_full)
        sign, logabsdet = torch.linalg.slogdet(G)
        logdet = logabsdet.cpu().numpy()
        cent_full = pad_latent(centroids[:, :2], latent_dim).to(centroids.device)
        G_cent = model.G(cent_full)
        _, logabsdet_cent = torch.linalg.slogdet(G_cent)
        cent_logdet = logabsdet_cent.cpu().numpy()

    logdet_grid = logdet.reshape(resolution, resolution)
    z_min = float(logdet_grid.min())
    z_max = float(logdet_grid.max())
    z_offset = z_min - 0.25 * (z_max - z_min)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X,
        Y,
        logdet_grid,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.92,
        zorder=2,
    )
    ax.contour(
        X,
        Y,
        logdet_grid,
        zdir="z",
        levels=15,
        offset=z_offset,
        cmap="viridis",
        linestyles="solid",
    )
    ax.scatter(
        c_np[:, 0],
        c_np[:, 1],
        cent_logdet,
        color="black",
        s=30,
        label="Centroids",
        depthshade=True,
    )
    arrow_grid = min(16, max(3, resolution // 16))
    indices = np.linspace(0, resolution - 1, arrow_grid, dtype=int)
    idx_X, idx_Y = np.meshgrid(indices, indices, indexing="xy")
    arrow_points = np.stack([axis[idx_X].ravel(), axis[idx_Y].ravel()], axis=1)
    arrow_logdet = logdet_grid[idx_Y, idx_X].ravel()
    if arrow_points.size > 0:
        arrow_tensor = torch.from_numpy(arrow_points.astype(np.float32)).to(centroids.device)
        arrow_full = pad_latent(arrow_tensor, latent_dim)
        with torch.no_grad():
            G_inv_arrow = model.G_inv(arrow_full)[:, :2, :2]
        eigvals, eigvecs = torch.linalg.eigh(G_inv_arrow)
        principal = eigvecs[:, :, -1].cpu().numpy()
        principal_energy = np.sqrt(np.clip(eigvals[:, -1].cpu().numpy(), 0.0, None))
        arrow_scale = 0.9 * (bounds / resolution)
        arrow_u = principal[:, 0] * principal_energy
        arrow_v = principal[:, 1] * principal_energy
        ax.quiver(
            arrow_points[:, 0],
            arrow_points[:, 1],
            z_offset + 0.05,
            arrow_u,
            arrow_v,
            np.zeros_like(arrow_u),
            length=arrow_scale,
            normalize=True,
            color="black",
            linewidth=1.5,
            arrow_length_ratio=0.45,
            alpha=0.85,
            pivot="tail",
        )
    ax.set_title("Metric Landscape (3D log det G with Contours)")
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.set_zlabel("log det G")
    ax.view_init(elev=30, azim=-65)
    fig.colorbar(surf, ax=ax, pad=0.12, label="log det G")

    save_plot(
        fig,
        out_dir / f"metric_surface_3d{file_suffix}.png",
        f"analysis/metric_surface_3d{tag_suffix}",
        wandb_run,
    )


def _sample_random_manifold_pairs(
    centroids: torch.Tensor,
    num_pairs: int,
    jitter: float,
    device: torch.device,
    used_pairs: set[tuple[int, int]],
    seed: int = 42,
) -> list[dict[str, torch.Tensor]]:
    if centroids.shape[0] < 1 or num_pairs <= 0:
        return []
    rng = torch.Generator(device=device).manual_seed(seed)
    pairs = []
    attempts = 0
    while len(pairs) < num_pairs and attempts < num_pairs * 12:
        idx = torch.randint(0, centroids.shape[0], (2,), generator=rng, device=device)
        i, j = int(idx[0].item()), int(idx[1].item())
        if i == j:
            attempts += 1
            continue
        key = tuple(sorted((i, j)))
        if key in used_pairs:
            attempts += 1
            continue
        used_pairs.add(key)
        pts = centroids[[i, j]]
        jitter_noise = jitter * torch.randn_like(pts, device=device)
        start = pts[0] + jitter_noise[0]
        end = pts[1] + jitter_noise[1]
        pairs.append(
            {
                "start": start,
                "end": end,
                "description": f"random_{len(pairs) + 1}",
            }
        )
        attempts += 1
    return pairs


def _axis_extreme_pairs(
    centroids: torch.Tensor,
    device: torch.device,
    used_pairs: set[tuple[int, int]],
) -> list[dict[str, torch.Tensor]]:
    pairs: list[dict[str, torch.Tensor]] = []
    dims = min(2, centroids.shape[1])
    for dim in range(dims):
        values = centroids[:, dim]
        min_idx = int(torch.argmin(values).item())
        max_idx = int(torch.argmax(values).item())
        if min_idx == max_idx:
            continue
        key = tuple(sorted((min_idx, max_idx)))
        if key in used_pairs:
            continue
        used_pairs.add(key)
        pairs.append(
            {
                "start": centroids[min_idx].to(device),
                "end": centroids[max_idx].to(device),
                "description": f"axis{dim}_minmax",
            }
        )
    return pairs


def geodesic_vs_linear(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    out_dir: Path,
    device: torch.device,
    far_pairs: int = 0,
    random_pairs: int = 8,
    axis_extremes: bool = False,
    jitter: float = 0.08,
    subplots: tuple[int, int] = (4, 2),
    wandb_run: Optional[Any] = None,
) -> tuple[list[dict[str, float]], list[dict[str, np.ndarray]]]:
    print("-> Geodesic vs Linear Interpolation")
    centroid_2d = centroids[:, :2].detach().cpu().numpy()
    distances = torch.cdist(centroids, centroids)
    mask = torch.tril(torch.ones_like(distances, dtype=torch.bool), diagonal=0)
    distances = distances.masked_fill(mask, -1.0)

    stats: list[dict[str, float]] = []
    pairs_info: list[dict[str, torch.Tensor]] = []

    available_pairs = (distances >= 0).sum().item()
    top_k = min(far_pairs, distances.numel())
    if available_pairs > 0 and far_pairs > 0:
        k = min(far_pairs, available_pairs)
        indices = torch.topk(distances.flatten(), k).indices
        used_pairs: set[tuple[int, int]] = set()
        for idx in indices:
            i = int(idx // distances.shape[1])
            j = int(idx % distances.shape[1])
            key = tuple(sorted((i, j)))
            if key in used_pairs:
                continue
            used_pairs.add(key)
            pairs_info.append(
                {
                    "start": centroids[i],
                    "end": centroids[j],
                    "description": f"far_{len(pairs_info) + 1}",
                }
            )
    else:
        used_pairs = set()

    if random_pairs > 0:
        pairs_info.extend(
            _sample_random_manifold_pairs(
                centroids,
                num_pairs=random_pairs,
                jitter=jitter,
                device=device,
                used_pairs=used_pairs,
            )
        )
    if axis_extremes:
        pairs_info.extend(_axis_extreme_pairs(centroids, device, used_pairs))

    if not pairs_info:
        return stats, []

    batch_data: list[dict[str, np.ndarray]] = []
    for rank, meta in enumerate(pairs_info):
        start = meta["start"]
        end = meta["end"]
        linear_steps = 120
        t = torch.linspace(0, 1, linear_steps, device=device).unsqueeze(1)
        linear_path = start + t * (end - start)
        geodesic_path = optimize_geodesic(
            model,
            start,
            end,
            n_points=64,
            iterations=1200,
            lr=0.015,
        )

        linear_length = compute_path_length(model, linear_path)
        geodesic_length = compute_path_length(model, geodesic_path)
        linear_energy = compute_path_energy(model, linear_path).detach().cpu().numpy()
        geodesic_energy = compute_path_energy(model, geodesic_path).detach().cpu().numpy()
        reduction = linear_length - geodesic_length
        stats.append(
            {
                "linear": linear_length,
                "geodesic": geodesic_length,
                "reduction": reduction,
                "description": meta.get("description", "pair"),
            }
        )
        batch_data.append(
            {
                "linear": linear_path.detach().cpu().numpy(),
                "geodesic": geodesic_path.detach().cpu().numpy(),
                "linear_energy": linear_energy,
                "geodesic_energy": geodesic_energy,
                "description": meta.get("description", "pair"),
            }
        )

    max_subplots = subplots[0] * subplots[1]
    total = min(len(batch_data), max_subplots)
    fig, axes = plt.subplots(
        subplots[0],
        subplots[1],
        figsize=(subplots[1] * 4, subplots[0] * 4),
        squeeze=False,
    )
    fig.suptitle("Geodesic vs Linear Interpolation (Batch View)")
    axes_flat = axes.flatten()
    for idx in range(len(axes_flat)):
        ax = axes_flat[idx]
        if idx < total:
            data = batch_data[idx]
            ax.scatter(centroid_2d[:, 0], centroid_2d[:, 1], c="gray", alpha=0.3, s=12)
            ax.plot(data["linear"][:, 0], data["linear"][:, 1], "--", color="dimgray")
            ax.plot(data["geodesic"][:, 0], data["geodesic"][:, 1], "-", color="royalblue")
            ax.set_title(f"{data['description']} #{idx + 1}")
            if data["description"].startswith("axis"):
                start_pt = data["geodesic"][0]
                end_pt = data["geodesic"][-1]
                ax.text(
                    start_pt[0],
                    start_pt[1],
                    data["description"],
                    fontsize=8,
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
                ax.text(
                    end_pt[0],
                    end_pt[1],
                    data["description"],
                    fontsize=8,
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
        ax.set_aspect("equal")
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_plot(
        fig,
        out_dir / "geodesic_batch.png",
        "analysis/geodesic_batch",
        wandb_run,
    )

    # Energy profiles along each path (metric energy per segment)
    fig, axes = plt.subplots(
        subplots[0],
        subplots[1],
        figsize=(subplots[1] * 4, subplots[0] * 3.2),
        squeeze=False,
    )
    fig.suptitle("Geodesic vs Linear Energy Profiles")
    axes_flat = axes.flatten()
    for idx in range(len(axes_flat)):
        ax = axes_flat[idx]
        if idx < total:
            data = batch_data[idx]
            lin_e = data["linear_energy"]
            geo_e = data["geodesic_energy"]
            t_lin = np.linspace(0, 1, len(lin_e))
            t_geo = np.linspace(0, 1, len(geo_e))
            ax.plot(t_lin, lin_e, "--", color="dimgray", linewidth=1.2, label="linear")
            ax.plot(t_geo, geo_e, "-", color="royalblue", linewidth=1.4, label="geodesic")
            ax.set_title(f"{data['description']} #{idx + 1}")
            if idx == 0:
                ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel("Path progress")
        ax.set_ylabel("Metric energy")
        ax.grid(alpha=0.2)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_plot(
        fig,
        out_dir / "geodesic_energy_profiles.png",
        "analysis/geodesic_energy_profiles",
        wandb_run,
    )

    return stats, batch_data


def rescue_dynamics_quiver(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    out_dir: Path,
    steps: int = 60,
    step_size: float = 0.05,
    wandb_run: Optional[Any] = None,
    step: int = 0,
) -> None:
    print("-> Lost Sampler Rescue Dynamics")
    device = centroids.device
    c_ext = centroids.abs().max().item() * 2.5
    start = torch.tensor([c_ext, c_ext], device=device)
    if model.latent_dim > 2:
        pad = torch.zeros(model.latent_dim - 2, device=device)
        start = torch.cat([start, pad])

    path = [start.clone()]
    z = start.clone()
    for _ in range(steps):
        dists = torch.cdist(z.unsqueeze(0), centroids).squeeze(0)
        nearest = centroids[torch.argmin(dists)]
        grad = 2 * (z - nearest)
        grad = grad / (grad.norm() + 1e-8)
        G_inv = model.G_inv(z.unsqueeze(0)).squeeze(0)
        step_vec = G_inv @ grad
        z = z - step_size * step_vec
        path.append(z.clone())

    path_stack = torch.stack(path)
    deltas = path_stack[1:] - path_stack[:-1]
    deltas_2d = deltas[:, :2].detach().cpu().numpy()
    colors = np.linspace(0, 1, deltas.shape[0])

    fig, ax = plt.subplots(figsize=(10, 8))
    centroids_np = centroids[:, :2].detach().cpu().numpy()
    ax.scatter(centroids_np[:, 0], centroids_np[:, 1], c='gray', alpha=0.3, s=20)
    path_np = path_stack[:, :2].detach().cpu().numpy()
    ax.plot(path_np[:, 0], path_np[:, 1], color='black', linewidth=1.5, alpha=0.7)
    quiver = ax.quiver(
        path_np[:-1, 0],
        path_np[:-1, 1],
        deltas_2d[:, 0],
        deltas_2d[:, 1],
        colors,
        cmap='coolwarm',
        scale=1.0,
        width=0.005,
        headwidth=4,
    )
    fig.colorbar(quiver, ax=ax, label='Step progression')
    ax.set_title('Lost Sampler Rescue Field (Quiver)')
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_aspect('equal')
    save_plot(
        fig,
        out_dir / 'lost_sampler_quiver.png',
        'analysis/rescue_quiver',
        wandb_run,
    )


def rescue_dynamics_quiver_volume(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    out_dir: Path,
    steps: int = 60,
    step_size: float = 0.05,
    wandb_run: Optional[Any] = None,
    step: int = 0,
) -> None:
    """Quiver using the volume-element gradient: ∇ log sqrt det(G^{-1})."""
    print("-> Volume-Element Rescue Dynamics")
    device = centroids.device
    c_ext = centroids.abs().max().item() * 2.5
    start = torch.tensor([c_ext, c_ext], device=device)
    if model.latent_dim > 2:
        pad = torch.zeros(model.latent_dim - 2, device=device)
        start = torch.cat([start, pad])

    path = [start.clone()]
    z = start.clone()
    for _ in range(steps):
        z = z.detach().requires_grad_(True)
        G_inv = model.G_inv(z.unsqueeze(0)).squeeze(0)
        logdet_inv = torch.linalg.slogdet(G_inv).logabsdet
        obj = 0.5 * logdet_inv
        grad = torch.autograd.grad(obj, z, create_graph=False)[0]
        grad = grad / (grad.norm() + 1e-8)
        z = z + step_size * grad
        path.append(z.detach())

    path_stack = torch.stack(path)
    deltas = path_stack[1:] - path_stack[:-1]
    deltas_2d = deltas[:, :2].detach().cpu().numpy()
    colors = np.linspace(0, 1, deltas.shape[0])

    fig, ax = plt.subplots(figsize=(10, 8))
    centroids_np = centroids[:, :2].detach().cpu().numpy()
    ax.scatter(centroids_np[:, 0], centroids_np[:, 1], c='gray', alpha=0.3, s=20)
    path_np = path_stack[:, :2].detach().cpu().numpy()
    ax.plot(path_np[:, 0], path_np[:, 1], color='black', linewidth=1.5, alpha=0.7)
    quiver = ax.quiver(
        path_np[:-1, 0],
        path_np[:-1, 1],
        deltas_2d[:, 0],
        deltas_2d[:, 1],
        colors,
        cmap='coolwarm',
        scale=1.0,
        width=0.005,
        headwidth=4,
    )
    fig.colorbar(quiver, ax=ax, label='Step progression')
    ax.set_title('Volume-Element Rescue Field (Quiver)')
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_aspect('equal')
    save_plot(
        fig,
        out_dir / 'volume_rescue_quiver.png',
        'analysis/volume_rescue_quiver',
        wandb_run,
    )


def curvature_statistics(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    out_dir: Path,
    prior_samples: int = 1000,
    posterior_samples: int = 1000,
    wandb_run: Optional[Any] = None,
) -> dict[str, float]:
    print("-> Curvature & Anisotropy Histograms")
    device = centroids.device
    latent_dim = model.latent_dim
    prior = torch.randn(prior_samples, latent_dim, device=device)
    choices = torch.randint(0, centroids.shape[0], (posterior_samples,), device=device)
    posterior = centroids[choices] + 0.08 * torch.randn(posterior_samples, latent_dim, device=device)

    def metric_measures(points: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            # Condition number still based on G eigenvalues
            G = model.G(points)
            eigvals = torch.linalg.eigvalsh(G)
            eigvals = eigvals.clamp_min(1e-8)
            cond = (eigvals[:, -1] / eigvals[:, 0]).cpu().numpy()
            # Volume element now based on sqrt(det(G_inv))
            G_inv = model.G_inv(points)
            _, logdet_inv = torch.linalg.slogdet(G_inv)
            volume = torch.exp(0.5 * logdet_inv).cpu().numpy()  # sqrt(det(G_inv))
        return cond, volume

    prior_cond, prior_vol = metric_measures(prior)
    post_cond, post_vol = metric_measures(posterior)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(prior_cond, bins=50, color='steelblue', label='Prior', ax=axes[0], stat='density', element='step')
    sns.histplot(post_cond, bins=50, color='coral', label='On-Manifold', ax=axes[0], stat='density', element='step')
    axes[0].set_title('Condition Number Distribution')
    axes[0].legend()

    sns.histplot(prior_vol, bins=50, color='steelblue', label='Prior', ax=axes[1], stat='density', element='step')
    sns.histplot(post_vol, bins=50, color='coral', label='On-Manifold', ax=axes[1], stat='density', element='step')
    axes[1].set_title('Volume Element Distribution')
    axes[1].legend()

    plt.tight_layout()
    save_plot(
        fig,
        out_dir / 'curvature_histograms.png',
        'analysis/curvature_histograms',
        wandb_run,
    )

    stats = {
        'prior_cond_mean': float(prior_cond.mean()),
        'posterior_cond_mean': float(post_cond.mean()),
        'prior_volume_mean': float(prior_vol.mean()),
        'posterior_volume_mean': float(post_vol.mean()),
    }
    return stats


def distortion_heatmap(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    out_dir: Path,
    grid_bounds: float = 4.0,
    resolution: int = 35,
    integration_steps: int = 24,
    wandb_run: Optional[Any] = None,
) -> dict[str, float]:
    print("-> Distortion Map (Riemannian vs Euclidean)")
    device = centroids.device
    latent_dim = model.latent_dim
    reference = centroids[0]
    axis = np.linspace(-grid_bounds, grid_bounds, resolution)
    X, Y = np.meshgrid(axis, axis)
    grid = torch.from_numpy(np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)).to(device)
    grid = pad_latent(grid, latent_dim)

    ratios = []
    for point in grid:
        euclidean = torch.norm(point - reference)
        if euclidean < 1e-3:
            ratios.append(1.0)
            continue
        steps = integration_steps
        t = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
        path = reference + t * (point - reference)
        riemannian = compute_path_length(model, path)
        ratios.append(float(riemannian / (euclidean.item() + 1e-8)))

    ratio_grid = np.array(ratios).reshape(resolution, resolution)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        ratio_grid,
        origin='lower',
        extent=[-grid_bounds, grid_bounds, -grid_bounds, grid_bounds],
        cmap='plasma',
    )
    ax.scatter(reference[0].cpu().item(), reference[1].cpu().item(), c='white', edgecolor='black', s=80, label='Reference')
    ax.set_title('Distortion Map (Riemannian / Euclidean)')
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    fig.colorbar(im, ax=ax, label='Riem / Eucl')
    ax.legend()
    save_plot(
        fig,
        out_dir / 'distortion_map.png',
        'analysis/distortion_map',
        wandb_run,
    )

    return {
        'distortion_mean': float(ratio_grid.mean()),
        'distortion_max': float(ratio_grid.max()),
    }


def generation_prior_experiment(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    out_dir: Path,
    num_samples: int = 600,
    mcmc_steps: int = 50,
    n_lf: int = 10,
    eps_lf: float = 0.03,
    beta_zero: float = 1.0,
    rhmc_sampler: str = "riemannian",
    use_dual_metric: bool = False,
    adaptive_dual_step: bool = False,
    adaptive_max_dual_displacement: float = 0.07,
    adaptive_min_step_scale: float = 0.05,
    volume_power: float = 2.0,
    radial_prior_weight: float = 0.1,
    momentum_persist: float = 0.0,
    fp_steps: int = 15,
    fp_damping: float = 0.72,
    wandb_run: Optional[Any] = None,
) -> dict[str, float]:
    samples, acc = rhmc_prior_samples(
        model,
        sampler_name=rhmc_sampler,
        num_samples=num_samples,
        mcmc_steps=mcmc_steps,
        n_lf=n_lf,
        eps_lf=eps_lf,
        beta_zero=beta_zero,
        use_dual_metric=use_dual_metric,
        adaptive_dual_step=adaptive_dual_step,
        adaptive_max_dual_displacement=adaptive_max_dual_displacement,
        adaptive_min_step_scale=adaptive_min_step_scale,
        volume_power=volume_power,
        radial_prior_weight=radial_prior_weight,
        momentum_persist=momentum_persist,
        fp_steps=fp_steps,
        fp_damping=fp_damping,
    )
    device = centroids.device if centroids is not None else next(model.parameters()).device
    sample_tensor = samples.to(device)
    with torch.no_grad():
        G = model.G(sample_tensor)
        eigvals = torch.linalg.eigvalsh(G)
        cond = (eigvals[:, -1] / eigvals[:, 0]).cpu().numpy()
        _, logabsdet = torch.linalg.slogdet(G)
        logdet = logabsdet.cpu().numpy()

    z_np = samples[:, :2].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        z_np[:, 0], z_np[:, 1], c=logdet, cmap="viridis", s=25, alpha=0.8
    )
    fig.colorbar(sc, ax=ax, label="log det G")
    ax.set_title("RHMC Prior Samples (Riemannian)")
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")

    save_plot(
        fig,
        out_dir / "prior_generation.png",
        "analysis/prior_generation",
        wandb_run,
    )

    return {
        "prior_cond_mean": float(cond.mean()),
        "prior_logdet_mean": float(logdet.mean()),
        "prior_rhmc_accept_rate": acc,
    }


def _build_rhmc_sampler(
    model: GeometryRHVAE,
    sampler_name: str,
    mcmc_steps: int,
    n_lf: int,
    eps_lf: float,
    beta_zero: float,
    use_dual_metric: bool = False,
    adaptive_dual_step: bool = False,
    adaptive_max_dual_displacement: float = 0.07,
    adaptive_min_step_scale: float = 0.05,
    volume_power: float = 2.0,
    radial_prior_weight: float = 0.1,
    momentum_persist: float = 0.0,
    fp_steps: int = 15,
    fp_damping: float = 0.72,
):
    if sampler_name == "geodesic":
        return GeodesicHMCSampler(
            model,
            mcmc_steps_nbr=mcmc_steps,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
            include_metropolis=True,
        )
    if sampler_name == "volume":
        return RHVAEVolumeElementHMCSampler(
            model,
            mcmc_steps_nbr=mcmc_steps,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
        )
    if sampler_name == "volume_riemannian":
        sampler = VolumeElementRiemannianHMCSampler(
            model,
            mcmc_steps_nbr=mcmc_steps,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
            use_dual_metric=use_dual_metric,
            adaptive_dual_step=adaptive_dual_step,
            adaptive_max_dual_displacement=adaptive_max_dual_displacement,
            adaptive_min_step_scale=adaptive_min_step_scale,
            volume_power=volume_power,
            radial_prior_weight=radial_prior_weight,
            fp_steps=fp_steps,
            fp_damping=fp_damping,
        )
        # Not a constructor arg for this sampler class; keep compatibility by setting the attribute.
        sampler.momentum_persist = float(momentum_persist)
        return sampler
    if sampler_name == "volume_det":
        return RHVAELogDetHMCSampler(
            model,
            mcmc_steps_nbr=mcmc_steps,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
        )
    if sampler_name == "volume_riemannian_det":
        return GeodesicHMCSampler(
            model,
            mcmc_steps_nbr=mcmc_steps,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
            include_metropolis=True,
        )
    if sampler_name == "dual_riemannian":
        return DualRiemannianHMCSampler(
            model,
            mcmc_steps_nbr=mcmc_steps,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
        )
    return RiemannianHMCSampler(
        model,
        mcmc_steps_nbr=mcmc_steps,
        n_lf=n_lf,
        eps_lf=eps_lf,
        beta_zero=beta_zero,
        include_volume_grad=True,
    )


def rhmc_prior_samples(
    model: GeometryRHVAE,
    sampler_name: str = "riemannian",
    num_samples: int = 400,
    mcmc_steps: int = 200,
    n_lf: int = 40,
    eps_lf: float = 0.008,
    beta_zero: float = 0.6,
    use_dual_metric: bool = False,
    adaptive_dual_step: bool = False,
    adaptive_max_dual_displacement: float = 0.07,
    adaptive_min_step_scale: float = 0.05,
    volume_power: float = 2.0,
    radial_prior_weight: float = 0.1,
    momentum_persist: float = 0.0,
    fp_steps: int = 15,
    fp_damping: float = 0.72,
) -> tuple[torch.Tensor, float]:
    sampler = _build_rhmc_sampler(
        model, sampler_name, mcmc_steps, n_lf, eps_lf, beta_zero,
        use_dual_metric=use_dual_metric,
        adaptive_dual_step=adaptive_dual_step,
        adaptive_max_dual_displacement=adaptive_max_dual_displacement,
        adaptive_min_step_scale=adaptive_min_step_scale,
        volume_power=volume_power,
        radial_prior_weight=radial_prior_weight,
        momentum_persist=momentum_persist,
        fp_steps=fp_steps,
        fp_damping=fp_damping,
    )
    samples = sampler.sample(num_samples)
    return samples.detach().cpu(), float(getattr(sampler, "last_acceptance_rate", 0.0))


def rhmc_chain(
    model: GeometryRHVAE,
    sampler_name: str = "riemannian",
    steps: int = 120,
    n_lf: int = 40,
    eps_lf: float = 0.008,
    beta_zero: float = 0.6,
    use_dual_metric: bool = False,
    adaptive_dual_step: bool = False,
    adaptive_max_dual_displacement: float = 0.07,
    adaptive_min_step_scale: float = 0.05,
    volume_power: float = 2.0,
    radial_prior_weight: float = 0.1,
    momentum_persist: float = 0.0,
    fp_steps: int = 15,
    fp_damping: float = 0.72,
) -> tuple[np.ndarray, float, np.ndarray]:
    sampler = _build_rhmc_sampler(
        model, sampler_name, steps, n_lf, eps_lf, beta_zero,
        use_dual_metric=use_dual_metric,
        adaptive_dual_step=adaptive_dual_step,
        adaptive_max_dual_displacement=adaptive_max_dual_displacement,
        adaptive_min_step_scale=adaptive_min_step_scale,
        volume_power=volume_power,
        radial_prior_weight=radial_prior_weight,
        momentum_persist=momentum_persist,
        fp_steps=fp_steps,
        fp_damping=fp_damping,
    )
    device = next(model.parameters()).device
    centroids = model.centroids_tens.to(device)
    K = centroids.shape[0]
    idx = torch.randint(K, (1,), device=device)
    z0 = centroids[idx].detach()
    z = z0.clone().detach().requires_grad_(True)
    path = [z.detach().squeeze(0).cpu().numpy()]
    logdet_values = []
    accept_count = 0
    local_n_lf = int(getattr(sampler, "n_lf", n_lf).item() if torch.is_tensor(getattr(sampler, "n_lf", None)) else getattr(sampler, "n_lf", n_lf))
    local_eps = float(getattr(sampler, "eps_lf", eps_lf).item() if torch.is_tensor(getattr(sampler, "eps_lf", None)) else getattr(sampler, "eps_lf", eps_lf))
    for _ in range(getattr(sampler, "mcmc_steps_nbr", steps)):
        use_tempering = (
            (not bool(getattr(sampler, "exact", False)))
            and hasattr(sampler, "_tempering")
            and hasattr(sampler, "beta_zero_sqrt")
        )
        beta_sqrt_old = sampler.beta_zero_sqrt.to(device) if use_tempering else None
        rho = sampler._initialize_momentum(z)
        with torch.no_grad():
            if hasattr(sampler, "_compute_hamiltonian"):
                H0 = sampler._compute_hamiltonian(z, rho)
            else:
                H0 = sampler._hamiltonian(z, rho)
        for k in range(local_n_lf):
            if hasattr(sampler, "_generalized_leapfrog_step"):
                z, rho = sampler._generalized_leapfrog_step(z, rho, local_eps)
            else:
                z, rho = sampler._leapfrog(z, rho, local_eps)
            if use_tempering and beta_sqrt_old is not None:
                beta_sqrt = sampler._tempering(k + 1, local_n_lf, sampler.beta_zero_sqrt)
                # Ensure all tensors are on the same device
                if torch.is_tensor(beta_sqrt):
                    beta_sqrt = beta_sqrt.to(device)
                else:
                    beta_sqrt = torch.tensor(beta_sqrt, device=device)
                rho = (beta_sqrt_old / beta_sqrt) * rho
                beta_sqrt_old = beta_sqrt
        with torch.no_grad():
            if hasattr(sampler, "_compute_hamiltonian"):
                H1 = sampler._compute_hamiltonian(z, rho)
            else:
                H1 = sampler._hamiltonian(z, rho)
            alpha = torch.exp(-(H1 - H0)).clamp(max=1.0)
            u = torch.rand_like(alpha)
            moves = (u < alpha).float().view(-1, 1)
            accept_count += int(moves.sum().item())
            z = ((moves * z + (1 - moves) * z0).detach().requires_grad_(True))
            z0 = z.detach()
            G = model.G(z)
            _, logdet = torch.linalg.slogdet(G)
            logdet_values.append(float(logdet.item()))
            path.append(z.detach().squeeze(0).cpu().numpy())
    accept_rate = accept_count / getattr(sampler, "mcmc_steps_nbr", steps) if getattr(sampler, "mcmc_steps_nbr", steps) else 0.0
    return np.stack(path, axis=0), accept_rate, np.array(logdet_values)


def simulate_rhmc_chain(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    out_dir: Path,
    steps: int = 80,
    n_lf: int = 30,
    eps_lf: float = 0.01,
    beta_zero: float = 1.0,
    rhmc_sampler: str = "riemannian",
    use_dual_metric: bool = False,
    adaptive_dual_step: bool = False,
    adaptive_max_dual_displacement: float = 0.07,
    adaptive_min_step_scale: float = 0.05,
    volume_power: float = 2.0,
    radial_prior_weight: float = 0.1,
    momentum_persist: float = 0.0,
    fp_steps: int = 15,
    fp_damping: float = 0.72,
    wandb_run: Optional[Any] = None,
) -> dict[str, float]:
    path_np, accept_rate, logdet_values = rhmc_chain(
        model,
        sampler_name=rhmc_sampler,
        steps=steps,
        n_lf=n_lf,
        eps_lf=eps_lf,
        beta_zero=beta_zero,
        use_dual_metric=use_dual_metric,
        adaptive_dual_step=adaptive_dual_step,
        adaptive_max_dual_displacement=adaptive_max_dual_displacement,
        adaptive_min_step_scale=adaptive_min_step_scale,
        volume_power=volume_power,
        radial_prior_weight=radial_prior_weight,
        momentum_persist=momentum_persist,
        fp_steps=fp_steps,
        fp_damping=fp_damping,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    centroids_np = centroids[:, :2].detach().cpu().numpy()
    ax.scatter(centroids_np[:, 0], centroids_np[:, 1], c="gray", alpha=0.3, s=12)
    ax.plot(path_np[:, 0], path_np[:, 1], "-o", color="teal", markersize=4)
    title = "RHMC Chain (Riemannian)"
    if rhmc_sampler == "geodesic":
        title = "RHMC Chain (Geodesic Uniform)"
    elif rhmc_sampler == "volume":
        title = "RHMC Chain (Volume Element)"
    ax.set_title(title)
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")

    save_plot(
        fig,
        out_dir / "rhmc_chain.png",
        "analysis/rhmc_chain",
        wandb_run,
    )

    final_val = float(logdet_values[-1]) if logdet_values.size else 0.0
    mean_val = float(np.mean(logdet_values)) if logdet_values.size else 0.0
    return {
        "rhmc_accept_rate": accept_rate,
        "rhmc_final_energy": final_val,
        "rhmc_energy_mean": mean_val,
    }


def run_analysis(
    model_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    wandb_run: Optional[Any] = None,
    skip_distortion: bool = False,
    far_pairs: int = 0,
    random_pairs: int = 8,
    axis_extremes: bool = False,
    support_images: Optional[torch.Tensor] = None,
    support_max_samples: int = 600,
    grid_bounds: float = 4.0,
    force_attractor_metric: Optional[str] = None,
    rhmc_sampler: str = "riemannian",
    use_dual_metric: bool = False,
    adaptive_dual_step: bool = False,
    adaptive_max_dual_displacement: float = 0.07,
    adaptive_min_step_scale: float = 0.05,
    rhmc_mcmc_steps: int = 50,
    rhmc_n_lf: int = 10,
    rhmc_eps_lf: float = 0.03,
    rhmc_beta_zero: float = 1.0,
    rhmc_volume_power: float = 2.0,
    rhmc_radial_prior_weight: float = 0.1,
    rhmc_momentum_persist: float = 0.0,
    rhmc_fp_steps: int = 15,
    rhmc_fp_damping: float = 0.72,
) -> dict[str, float]:
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    out_dir = build_output_dir(output_dir)

    model, centroids = load_metric_geometry(model_path, device)
    if force_attractor_metric is not None:
        model.attractor_metric = force_attractor_metric
        if model.attractor_metric == "mahalanobis" or model.attractor_use_det:
            model._update_attractor_precisions(model.M_tens)
        model._refresh_metric_hooks()

    support_images_cpu: Optional[torch.Tensor] = None
    support_latents: Optional[torch.Tensor] = None
    if support_images is not None:
        support_images_cpu, support_latents = encode_support_images(
            model, support_images, max_support=support_max_samples
        )

    metric_field_tissot(model, centroids, out_dir, grid_bounds=grid_bounds, wandb_run=wandb_run)
    logdet_landscape(model, centroids, out_dir, grid_bounds=grid_bounds, wandb_run=wandb_run)
    metric_surface_3d(model, centroids, out_dir, grid_bounds=grid_bounds, wandb_run=wandb_run)
    metric_surface_3d(
        model,
        centroids,
        out_dir,
        grid_bounds=grid_bounds * 10.0,
        tag_suffix="_wide",
        file_suffix="_wide",
        wandb_run=wandb_run,
    )
    rescue_dynamics_quiver(model, centroids, out_dir, wandb_run=wandb_run)
    rescue_dynamics_quiver_volume(model, centroids, out_dir, wandb_run=wandb_run)
    geodesic_stats, geodesic_paths = geodesic_vs_linear(
        model,
        centroids,
        out_dir,
        device,
        far_pairs=far_pairs,
        random_pairs=random_pairs,
        axis_extremes=axis_extremes,
        wandb_run=wandb_run,
    )
    curvature_stats = curvature_statistics(model, centroids, out_dir, wandb_run=wandb_run)
    distortion_stats: dict[str, float] = {}
    if not skip_distortion:
        distortion_stats = distortion_heatmap(model, centroids, out_dir, wandb_run=wandb_run)
    prior_stats = generation_prior_experiment(
        model, centroids, out_dir, rhmc_sampler=rhmc_sampler, wandb_run=wandb_run,
        mcmc_steps=rhmc_mcmc_steps,
        n_lf=rhmc_n_lf,
        eps_lf=rhmc_eps_lf,
        beta_zero=rhmc_beta_zero,
        use_dual_metric=use_dual_metric,
        adaptive_dual_step=adaptive_dual_step,
        adaptive_max_dual_displacement=adaptive_max_dual_displacement,
        adaptive_min_step_scale=adaptive_min_step_scale,
        volume_power=rhmc_volume_power,
        radial_prior_weight=rhmc_radial_prior_weight,
        momentum_persist=rhmc_momentum_persist,
        fp_steps=rhmc_fp_steps,
        fp_damping=rhmc_fp_damping,
    )
    rhmc_stats = simulate_rhmc_chain(
        model, centroids, out_dir, rhmc_sampler=rhmc_sampler, wandb_run=wandb_run,
        steps=rhmc_mcmc_steps,
        n_lf=rhmc_n_lf,
        eps_lf=rhmc_eps_lf,
        beta_zero=rhmc_beta_zero,
        use_dual_metric=use_dual_metric,
        adaptive_dual_step=adaptive_dual_step,
        adaptive_max_dual_displacement=adaptive_max_dual_displacement,
        adaptive_min_step_scale=adaptive_min_step_scale,
        volume_power=rhmc_volume_power,
        radial_prior_weight=rhmc_radial_prior_weight,
        momentum_persist=rhmc_momentum_persist,
        fp_steps=rhmc_fp_steps,
        fp_damping=rhmc_fp_damping,
    )

    plot_geodesic_image_sequences(
        geodesic_paths,
        support_images_cpu,
        support_latents,
        out_dir,
        wandb_run,
    )

    reduction_vals = [stat["reduction"] for stat in geodesic_stats]
    avg_reduction = float(np.mean(reduction_vals)) if reduction_vals else 0.0

    print("\n=== Quantitative Summary ===")
    print(f"Average geodesic savings (linear - geodesic): {avg_reduction:.3f}")
    print(
        f"Prior condition mean: {curvature_stats['prior_cond_mean']:.3f}, on-manifold: {curvature_stats['posterior_cond_mean']:.3f}"
    )
    if distortion_stats:
        print(
            f"Distortion ratio (mean/max): {distortion_stats['distortion_mean']:.3f} / {distortion_stats['distortion_max']:.3f}"
        )
    print(f"Results saved to {out_dir}")

    summary = {
        "analysis/avg_geodesic_savings": avg_reduction,
        "analysis/prior_condition_mean": prior_stats["prior_cond_mean"],
        "analysis/prior_logdet_mean": prior_stats["prior_logdet_mean"],
        "analysis/rhmc_final_energy": rhmc_stats["rhmc_final_energy"],
        "analysis/rhmc_energy_mean": rhmc_stats["rhmc_energy_mean"],
        "analysis/metric_prior_cond_mean": curvature_stats["prior_cond_mean"],
        "analysis/metric_posterior_cond_mean": curvature_stats["posterior_cond_mean"],
    }
    if distortion_stats:
        summary["analysis/distortion_ratio_mean"] = distortion_stats["distortion_mean"]
    summary["analysis/prior_rhmc_accept"] = prior_stats.get("prior_rhmc_accept_rate", 0.0)
    if wandb_run is not None and wandb is not None:
        wandb_run.log(summary)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RHVAE metric geometry in detail.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained RHVAE folder.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/analysis_results",
        help="Base directory to store plots.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device.",
    )
    parser.add_argument("--skip_distortion", action="store_true", help="Skip the expensive distortion map.")
    parser.add_argument("--far_pairs", type=int, default=0, help="Number of far centroid pairs to include.")
    parser.add_argument("--random_pairs", type=int, default=8, help="Number of random centroid pairs for batch.")
    parser.add_argument(
        "--axis_extremes",
        action="store_true",
        help="Include geodesics between min/max along latent dims 1 and 2.",
    )
    parser.add_argument(
        "--force_attractor_metric",
        type=str,
        choices=["euclidean", "mahalanobis"],
        default=None,
        help="Override attractor_metric after loading the model.",
    )
    parser.add_argument(
        "--analysis_bounds",
        type=float,
        default=4.0,
        help="Extent (+/- value) used for metric field and surface visualizations.",
    )
    parser.add_argument(
        "--rhmc_sampler",
        type=str,
        default="volume",
        choices=[
            "riemannian",
            "geodesic",
            "volume",
            "volume_riemannian",
            "volume_det",
            "volume_riemannian_det",
            "dual_riemannian",
        ],
        help="Sampler for RHMC prior/chain (riemannian, geodesic-uniform, volume-element, volume-riemannian, volume-detG, volume-riemannian-detG, dual-riemannian).",
    )
    parser.add_argument("--wandb_project", type=str, default=None, help="Log to WandB project.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/account.")
    parser.add_argument("--wandb_group", type=str, default=None, help="WandB run group.")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated WandB tags.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Explicit WandB run name.")
    parser.add_argument(
        "--wandb_name_mode",
        type=str,
        choices=["timestamp", "manual"],
        default="timestamp",
        help="WandB run naming mode.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    wandb_run = init_wandb(args)
    run_analysis(
        args.model_path,
        args.output_dir,
        device,
        wandb_run=wandb_run,
        skip_distortion=args.skip_distortion,
        far_pairs=args.far_pairs,
        random_pairs=args.random_pairs,
        axis_extremes=args.axis_extremes,
        grid_bounds=args.analysis_bounds,
        force_attractor_metric=args.force_attractor_metric,
        rhmc_sampler=args.rhmc_sampler,
    )
    if wandb_run is not None and wandb is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
