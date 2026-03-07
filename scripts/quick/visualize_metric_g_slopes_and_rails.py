#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from mpl_toolkits.mplot3d import Axes3D  # pyright: ignore[reportMissingImports,reportUnusedImport]

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from src.models.rhvae_geometry import GeometryRHVAE, GeometryRHVAEConfig
from src.utils.metric_helpers import load_metric_bundle


def load_geometry_model(model_path: Path, device: torch.device) -> GeometryRHVAE:
    metric_path = model_path / "rhvae_metric.pt" if model_path.is_dir() else model_path
    centroids, atoms, temperature, regularization, cfg_dict = load_metric_bundle(metric_path)

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
    model.eval()
    model.set_centroids(centroids.to(device))
    model.set_atoms(atoms.to(device))
    return model


def parse_dims(text: str, latent_dim: int) -> tuple[int, int]:
    if text.strip().lower() == "auto":
        return 0, 1
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError("--dims must be 'i,j' or 'auto'")
    d0, d1 = int(parts[0]), int(parts[1])
    if d0 == d1:
        raise ValueError("--dims indices must differ")
    if not (0 <= d0 < latent_dim and 0 <= d1 < latent_dim):
        raise ValueError(f"--dims out of range for latent_dim={latent_dim}")
    return d0, d1


def build_grid(
    centroids: torch.Tensor,
    dims: tuple[int, int],
    grid_n: int,
    bounds_scale: float,
    min_bounds: float,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    d0, d1 = dims
    c = centroids.detach().cpu().numpy()
    c2 = c[:, [d0, d1]]
    center2 = c2.mean(axis=0)
    radii = np.linalg.norm(c2 - center2[None, :], axis=1)
    bound = max(float(np.percentile(radii, 95) * bounds_scale), float(min_bounds))

    x = np.linspace(center2[0] - bound, center2[0] + bound, int(grid_n))
    y = np.linspace(center2[1] - bound, center2[1] + bound, int(grid_n))
    xx, yy = np.meshgrid(x, y, indexing="xy")
    flat = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)

    z = centroids.mean(dim=0, keepdim=True).repeat(flat.shape[0], 1)
    z[:, d0] = torch.from_numpy(flat[:, 0]).to(z.device)
    z[:, d1] = torch.from_numpy(flat[:, 1]).to(z.device)
    return xx, yy, z


def build_grid_from_bounds(
    centroids: torch.Tensor,
    dims: tuple[int, int],
    grid_n: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    d0, d1 = dims
    x = np.linspace(float(x_min), float(x_max), int(grid_n))
    y = np.linspace(float(y_min), float(y_max), int(grid_n))
    xx, yy = np.meshgrid(x, y, indexing="xy")
    flat = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)

    z = centroids.mean(dim=0, keepdim=True).repeat(flat.shape[0], 1)
    z[:, d0] = torch.from_numpy(flat[:, 0]).to(z.device)
    z[:, d1] = torch.from_numpy(flat[:, 1]).to(z.device)
    return xx, yy, z


def eval_grid_logdet_g_and_rails(
    model: GeometryRHVAE,
    points: torch.Tensor,
    dims: tuple[int, int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d0, d1 = dims
    idx2 = torch.tensor([d0, d1], device=points.device)
    n = points.shape[0]

    logdet_g = np.empty(n, dtype=np.float64)
    cond_g2 = np.empty(n, dtype=np.float64)
    px = np.empty(n, dtype=np.float64)
    py = np.empty(n, dtype=np.float64)

    with torch.no_grad():
        for i in range(0, n, int(batch_size)):
            z = points[i : i + int(batch_size)]
            g = model.G(z)
            logdet = torch.linalg.slogdet(g).logabsdet
            g2 = g.index_select(1, idx2).index_select(2, idx2)
            evals, evecs = torch.linalg.eigh(g2)  # ascending
            evals = torch.clamp(evals, min=1e-12)
            cond = evals[:, 1] / evals[:, 0]
            principal = evecs[:, :, 1]  # largest eigenvalue direction of G

            j = i + z.shape[0]
            logdet_g[i:j] = logdet.cpu().numpy()
            cond_g2[i:j] = cond.cpu().numpy()
            px[i:j] = principal[:, 0].cpu().numpy()
            py[i:j] = principal[:, 1].cpu().numpy()

    return logdet_g, cond_g2, px, py


def downsample_quiver(
    xx: np.ndarray,
    yy: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    grid_n: int,
    quiver_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pxg = px.reshape(grid_n, grid_n)
    pyg = py.reshape(grid_n, grid_n)
    step = max(1, grid_n // max(2, quiver_n))
    qx = xx[::step, ::step]
    qy = yy[::step, ::step]
    qpx = pxg[::step, ::step]
    qpy = pyg[::step, ::step]
    qn = np.sqrt(qpx * qpx + qpy * qpy) + 1e-12
    qpx = qpx / qn
    qpy = qpy / qn
    return qx, qy, qpx, qpy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize metric slopes (log det G) + anisotropy rails (principal directions of G)."
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="results/metric_g_slopes_and_rails")
    p.add_argument("--dims", type=str, default="0,1")
    p.add_argument("--grid_n", type=int, default=180)
    p.add_argument("--quiver_n", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--bounds_scale", type=float, default=2.2)
    p.add_argument("--min_bounds", type=float, default=8.0)
    p.add_argument("--zoom_min", type=float, default=-3.0)
    p.add_argument("--zoom_max", type=float, default=3.0)
    p.add_argument(
        "--zoom_grid_n",
        type=int,
        default=0,
        help="Grid resolution for lower zoom panels. If <=0, use global grid_n.",
    )
    p.add_argument(
        "--zoom_quiver_n",
        type=int,
        default=0,
        help="Quiver density for lower zoom panels. If <=0, use global quiver_n.",
    )

    p.add_argument(
        "--void_eigshape_mode",
        type=str,
        choices=["none", "det_preserving_spectral"],
        default="none",
    )
    p.add_argument("--void_eigshape_alpha_min", type=float, default=1.0)
    p.add_argument("--void_eigshape_power", type=float, default=-1.0)
    p.add_argument("--void_eigshape_eig_floor", type=float, default=1e-8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    model: Any = load_geometry_model(Path(args.model_path), device=device)

    model.void_eigshape_mode = str(args.void_eigshape_mode)
    model.void_eigshape_alpha_min = float(args.void_eigshape_alpha_min)
    model.void_eigshape_power = float(args.void_eigshape_power)
    model.void_eigshape_eig_floor = float(args.void_eigshape_eig_floor)

    d0, d1 = parse_dims(str(args.dims), int(model.latent_dim))
    xx, yy, points = build_grid(
        centroids=model.centroids_tens.detach(),
        dims=(d0, d1),
        grid_n=int(args.grid_n),
        bounds_scale=float(args.bounds_scale),
        min_bounds=float(args.min_bounds),
    )

    logdet_g, cond_g2, px, py = eval_grid_logdet_g_and_rails(
        model=model,
        points=points,
        dims=(d0, d1),
        batch_size=int(args.batch_size),
    )

    grid_n = int(args.grid_n)
    logdet_grid = logdet_g.reshape(grid_n, grid_n)
    cond_grid = np.log10(np.clip(cond_g2.reshape(grid_n, grid_n), 1e-12, None))
    qx, qy, qpx, qpy = downsample_quiver(xx, yy, px, py, grid_n=grid_n, quiver_n=int(args.quiver_n))

    zoom_min = float(args.zoom_min)
    zoom_max = float(args.zoom_max)
    zoom_grid_n = int(args.zoom_grid_n) if int(args.zoom_grid_n) > 0 else grid_n
    zoom_quiver_n = int(args.zoom_quiver_n) if int(args.zoom_quiver_n) > 0 else int(args.quiver_n)
    xxz, yyz, points_z = build_grid_from_bounds(
        centroids=model.centroids_tens.detach(),
        dims=(d0, d1),
        grid_n=zoom_grid_n,
        x_min=zoom_min,
        x_max=zoom_max,
        y_min=zoom_min,
        y_max=zoom_max,
    )
    logdet_g_z, cond_g2_z, px_z, py_z = eval_grid_logdet_g_and_rails(
        model=model,
        points=points_z,
        dims=(d0, d1),
        batch_size=int(args.batch_size),
    )
    logdet_grid_z = logdet_g_z.reshape(zoom_grid_n, zoom_grid_n)
    cond_grid_z = np.log10(np.clip(cond_g2_z.reshape(zoom_grid_n, zoom_grid_n), 1e-12, None))
    qxz, qyz, qpxz, qpyz = downsample_quiver(
        xxz, yyz, px_z, py_z, grid_n=zoom_grid_n, quiver_n=zoom_quiver_n
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    cent_x = model.centroids_tens[:, d0].detach().cpu().numpy()
    cent_y = model.centroids_tens[:, d1].detach().cpu().numpy()
    zmin = float(np.min(logdet_grid))
    zmax = float(np.max(logdet_grid))

    # Top row: global view
    ax1 = fig.add_subplot(2, 3, 1)
    hm1 = ax1.contourf(xx, yy, logdet_grid, levels=36, cmap="viridis")
    fig.colorbar(hm1, ax=ax1, fraction=0.045, pad=0.02, label="log det(G)")
    ax1.quiver(qx, qy, qpx, qpy, color="white", alpha=0.75, width=0.0025, scale=26)
    ax1.scatter(cent_x, cent_y, s=18, c="black", alpha=0.45, edgecolors="none")
    ax1.set_title("Global: log det(G) + Principal Rails of G")
    ax1.set_xlabel(f"z[{d0}]")
    ax1.set_ylabel(f"z[{d1}]")
    ax1.set_aspect("equal")

    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    surf = ax2.plot_surface(xx, yy, logdet_grid, cmap="cividis", linewidth=0, antialiased=True, alpha=0.95)
    fig.colorbar(surf, ax=ax2, fraction=0.045, pad=0.02, label="log det(G)")
    ax2.set_title("Global: 3D Slope Surface (log det G)")
    ax2.set_xlabel(f"z[{d0}]")
    ax2.set_ylabel(f"z[{d1}]")
    ax2.set_zlabel("log det(G)")
    ax2.view_init(elev=34, azim=-60)

    ax3 = fig.add_subplot(2, 3, 3)
    hm3 = ax3.contourf(xx, yy, cond_grid, levels=36, cmap="magma")
    fig.colorbar(hm3, ax=ax3, fraction=0.045, pad=0.02, label="log10 cond(G[2D])")
    ax3.quiver(qx, qy, qpx, qpy, color="cyan", alpha=0.75, width=0.0025, scale=26)
    ax3.scatter(cent_x, cent_y, s=18, c="white", alpha=0.6, edgecolors="black", linewidths=0.3)
    ax3.set_title("Global: Anisotropy of G + Rails")
    ax3.set_xlabel(f"z[{d0}]")
    ax3.set_ylabel(f"z[{d1}]")
    ax3.set_aspect("equal")

    # Bottom row: zoomed view with dedicated resolution
    ax4 = fig.add_subplot(2, 3, 4)
    hm4 = ax4.contourf(xxz, yyz, logdet_grid_z, levels=36, cmap="viridis")
    fig.colorbar(hm4, ax=ax4, fraction=0.045, pad=0.02, label="log det(G)")
    ax4.quiver(qxz, qyz, qpxz, qpyz, color="white", alpha=0.75, width=0.0025, scale=26)
    ax4.scatter(cent_x, cent_y, s=18, c="black", alpha=0.45, edgecolors="none")
    ax4.set_title(f"Zoom [{zoom_min:g}, {zoom_max:g}]: log det(G) + Rails")
    ax4.set_xlabel(f"z[{d0}]")
    ax4.set_ylabel(f"z[{d1}]")
    ax4.set_xlim(zoom_min, zoom_max)
    ax4.set_ylim(zoom_min, zoom_max)
    ax4.set_aspect("equal")

    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    surf2 = ax5.plot_surface(xxz, yyz, logdet_grid_z, cmap="cividis", linewidth=0, antialiased=True, alpha=0.95)
    fig.colorbar(surf2, ax=ax5, fraction=0.045, pad=0.02, label="log det(G)")
    ax5.set_title(f"Zoom [{zoom_min:g}, {zoom_max:g}]: 3D Slope")
    ax5.set_xlabel(f"z[{d0}]")
    ax5.set_ylabel(f"z[{d1}]")
    ax5.set_zlabel("log det(G)")
    ax5.set_xlim(zoom_min, zoom_max)
    ax5.set_ylim(zoom_min, zoom_max)
    ax5.set_zlim(zmin, zmax)
    ax5.view_init(elev=34, azim=-60)

    ax6 = fig.add_subplot(2, 3, 6)
    hm6 = ax6.contourf(xxz, yyz, cond_grid_z, levels=36, cmap="magma")
    fig.colorbar(hm6, ax=ax6, fraction=0.045, pad=0.02, label="log10 cond(G[2D])")
    ax6.quiver(qxz, qyz, qpxz, qpyz, color="cyan", alpha=0.75, width=0.0025, scale=26)
    ax6.scatter(cent_x, cent_y, s=18, c="white", alpha=0.6, edgecolors="black", linewidths=0.3)
    ax6.set_title(f"Zoom [{zoom_min:g}, {zoom_max:g}]: Anisotropy + Rails")
    ax6.set_xlabel(f"z[{d0}]")
    ax6.set_ylabel(f"z[{d1}]")
    ax6.set_xlim(zoom_min, zoom_max)
    ax6.set_ylim(zoom_min, zoom_max)
    ax6.set_aspect("equal")

    run_dir = Path(args.output_dir) / dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_path = run_dir / "metric_g_slopes_and_rails.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "model_path": str(args.model_path),
        "dims": [int(d0), int(d1)],
        "void_eigshape": {
            "mode": str(args.void_eigshape_mode),
            "alpha_min": float(args.void_eigshape_alpha_min),
            "power": float(args.void_eigshape_power),
            "eig_floor": float(args.void_eigshape_eig_floor),
        },
        "zoom_bounds": [zoom_min, zoom_max],
        "zoom_grid_n": int(zoom_grid_n),
        "zoom_quiver_n": int(zoom_quiver_n),
        "logdet_g": {
            "min": float(np.min(logdet_grid)),
            "max": float(np.max(logdet_grid)),
            "median": float(np.median(logdet_grid)),
        },
        "log10_cond_g2d": {
            "min": float(np.min(cond_grid)),
            "max": float(np.max(cond_grid)),
            "median": float(np.median(cond_grid)),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
