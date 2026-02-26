#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import seaborn as sns
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from src.models.rhvae_geometry import GeometryRHVAE, GeometryRHVAEConfig
from src.utils.metric_helpers import load_metric_bundle
from src.utils.wandb_logging import init_wandb_run, make_wandb_image, safe_wandb_finish, safe_wandb_log
from scripts.rhmc_chain_demo import _build_sampler, run_hmc_chain


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


def potential_volume_riemannian(model: GeometryRHVAE, z: torch.Tensor, volume_power: float) -> torch.Tensor:
    eff = float(volume_power) + 0.5
    g_inv = model.G_inv(z)
    logdet_inv = torch.linalg.slogdet(g_inv).logabsdet
    return -(eff * logdet_inv)


def min_dist_to_centroids(z: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    return torch.cdist(z, centroids).amin(dim=1)


def _multi_lag_autocorr(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Compute autocorrelation at lags 1..max_lag for a 1D signal."""
    n = x.size
    if n < 3:
        return np.full(max_lag, float("nan"))
    x0 = x - x.mean()
    var = (x0 * x0).sum()
    if var <= 1e-12:
        return np.full(max_lag, float("nan"))
    acf = np.array([float((x0[:n - k] * x0[k:]).sum() / var) for k in range(1, max_lag + 1)])
    return acf


def simulate_chain(
    sampler: Any,
    model: GeometryRHVAE,
    z_start: torch.Tensor,
    steps: int,
    n_lf_inner: int,
    eps: float,
    max_radius: float,
    eps_jitter: float = 0.0,
    n_lf_jitter: int = 0,
    no_metropolis: bool = False,
    rescue_threshold: float | None = None,
) -> dict[str, np.ndarray]:
    z = z_start.detach().clone().unsqueeze(0) if z_start.ndim == 1 else z_start.detach().clone()
    chain, energies = run_hmc_chain(
        start_z=z,
        sampler=sampler,
        chain_length=int(steps),
        n_lf=int(n_lf_inner),
        eps_lf=float(eps),
        eps_jitter=float(eps_jitter),
        n_lf_jitter=int(n_lf_jitter),
        no_metropolis=bool(no_metropolis),
    )
    path = np.asarray(chain, dtype=np.float64)
    if path.ndim == 1:
        path = path[:, None]

    z_t = torch.from_numpy(path).to(model.centroids_tens.device, dtype=model.centroids_tens.dtype)
    with torch.no_grad():
        d = torch.cdist(z_t, model.centroids_tens).amin(dim=1)
        logdet_inv = torch.linalg.slogdet(model.G_inv(z_t)).logabsdet
    dist = d.detach().cpu().numpy().astype(np.float64)
    logdet_inv = logdet_inv.detach().cpu().numpy().astype(np.float64)

    diffs = np.diff(path, axis=0)
    speed = np.linalg.norm(diffs, axis=1) if diffs.shape[0] > 0 else np.array([], dtype=np.float64)
    accept = np.asarray(energies.get("accept", np.array([])), dtype=np.float64)

    radius = np.linalg.norm(path, axis=1)
    escaped_idx = np.where(radius > float(max_radius))[0]
    escaped = escaped_idx.size > 0
    escape_step = int(escaped_idx[0]) if escaped else -1

    # --- Autocorrelation (per dimension, averaged) ---
    max_lag = min(20, max(2, path.shape[0] // 5))
    dim_acfs = [_multi_lag_autocorr(path[:, d], max_lag) for d in range(path.shape[1])]
    acf_mean = np.mean(dim_acfs, axis=0)  # average across dims

    # --- Coverage: fraction of centroids visited within 1-NN radius ---
    centroids_np = model.centroids_tens.detach().cpu().numpy()
    nn_dists = np.linalg.norm(
        centroids_np[None, :, :] - path[:, None, :], axis=2
    )  # (steps, n_centroids)
    # Typical 1-NN radius: median nearest-neighbor distance among centroids
    c_dists = torch.cdist(model.centroids_tens, model.centroids_tens)
    c_dists.fill_diagonal_(float("inf"))
    nn_radius = float(c_dists.min(dim=1).values.median().item())
    visited = (nn_dists.min(axis=0) < nn_radius)  # which centroids were visited
    coverage = float(visited.sum()) / max(1, centroids_np.shape[0])

    # --- Rescue steps: steps where chain is out-of-manifold ---
    if rescue_threshold is not None:
        out_of_manifold = (dist > rescue_threshold).astype(np.float64)
    else:
        # Default: use 2× median centroid NN distance as boundary
        out_of_manifold = (dist > 2.0 * nn_radius).astype(np.float64)
    rescue_cumsum = np.cumsum(out_of_manifold)

    return {
        "path": path,
        "dist": dist,
        "logdet_inv": logdet_inv,
        "speed": speed.astype(np.float64),
        "accept": accept,
        "escaped": np.asarray([1.0 if escaped else 0.0], dtype=np.float64),
        "escape_step": np.asarray([float(escape_step)], dtype=np.float64),
        "acf": acf_mean,
        "coverage": np.asarray([coverage], dtype=np.float64),
        "out_of_manifold": out_of_manifold,
        "rescue_cumsum": rescue_cumsum,
    }


def choose_manifold_starts(centroids: torch.Tensor, dims: tuple[int, int], n: int) -> list[torch.Tensor]:
    n = max(1, int(n))
    c2 = centroids[:, [dims[0], dims[1]]]
    center2 = c2.mean(dim=0)
    angles = torch.atan2(c2[:, 1] - center2[1], c2[:, 0] - center2[0])
    order = torch.argsort(angles)
    if n >= order.numel():
        idx = order
    else:
        pick = torch.linspace(0, order.numel() - 1, n, device=centroids.device)
        idx = order[pick.round().long()]
    starts = [centroids[int(i)].detach().clone() for i in idx]
    return starts


def _pick_angular_indices(c2: torch.Tensor, n: int) -> torch.Tensor:
    n = max(1, int(n))
    center2 = c2.mean(dim=0)
    shifted = c2 - center2
    angles = torch.atan2(shifted[:, 1], shifted[:, 0])
    order = torch.argsort(angles)
    if n >= order.numel():
        return order
    pick = torch.linspace(0, order.numel() - 1, n, device=c2.device)
    return order[pick.round().long()]


def make_outside_starts(
    centroids: torch.Tensor,
    dims: tuple[int, int],
    near_multiplier: float,
    far_multiplier: float,
    near_points: int = 1,
    far_points: int = 1,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    d0, d1 = dims
    c2 = centroids[:, [d0, d1]]
    center2 = c2.mean(dim=0)
    shifted = c2 - center2
    radii = torch.linalg.norm(shifted, dim=1)
    r_max = float(torch.max(radii).item())

    z_center = centroids.mean(dim=0)
    near_idx = _pick_angular_indices(c2, int(near_points))
    far_idx = _pick_angular_indices(c2, int(far_points))

    near_starts: list[torch.Tensor] = []
    for i in near_idx:
        direction2 = shifted[int(i)] / (radii[int(i)] + 1e-9)
        z_near = z_center.clone()
        z_near[d0] = center2[0] + float(near_multiplier) * r_max * direction2[0]
        z_near[d1] = center2[1] + float(near_multiplier) * r_max * direction2[1]
        near_starts.append(z_near.detach())

    far_starts: list[torch.Tensor] = []
    for i in far_idx:
        direction2 = shifted[int(i)] / (radii[int(i)] + 1e-9)
        z_far = z_center.clone()
        z_far[d0] = center2[0] + float(far_multiplier) * r_max * direction2[0]
        z_far[d1] = center2[1] + float(far_multiplier) * r_max * direction2[1]
        far_starts.append(z_far.detach())

    return near_starts, far_starts


def compute_bounds(
    centroids: np.ndarray,
    starts: np.ndarray,
    trajs: list[dict[str, np.ndarray]],
    dims: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float]]:
    d0, d1 = dims
    c2 = centroids[:, [d0, d1]]
    center = c2.mean(axis=0)
    path_pts = [t["path"][:, [d0, d1]] for t in trajs if t["path"].size > 0]
    all_pts = [c2, starts[:, [d0, d1]]] + path_pts
    pts = np.vstack(all_pts)
    rad = np.linalg.norm(pts - center[None, :], axis=1)
    keep = rad <= max(np.percentile(rad, 99), 8.0)
    pts = pts[keep]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    pad_x = max(0.7, 0.10 * (x_max - x_min))
    pad_y = max(0.7, 0.10 * (y_max - y_min))
    return (float(x_min - pad_x), float(x_max + pad_x)), (float(y_min - pad_y), float(y_max + pad_y))


def evaluate_potential_grid(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    dims: tuple[int, int],
    volume_power: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d0, d1 = dims
    x = np.linspace(xlim[0], xlim[1], int(grid_n))
    y = np.linspace(ylim[0], ylim[1], int(grid_n))
    xx, yy = np.meshgrid(x, y, indexing="xy")
    flat = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)

    z = centroids.mean(dim=0, keepdim=True).repeat(flat.shape[0], 1)
    z[:, d0] = torch.from_numpy(flat[:, 0]).to(z.device)
    z[:, d1] = torch.from_numpy(flat[:, 1]).to(z.device)

    vals = []
    with torch.no_grad():
        for i in range(0, z.shape[0], 4096):
            chunk = z[i : i + 4096]
            vals.append(potential_volume_riemannian(model, chunk, volume_power).cpu().numpy())
    u = np.concatenate(vals, axis=0).reshape(int(grid_n), int(grid_n))
    return xx, yy, u


def summarize_traj(name: str, tr: dict[str, np.ndarray]) -> dict[str, Any]:
    acc = tr["accept"]
    acc_rate = float(np.mean(acc)) if acc.size else float("nan")
    acf = tr.get("acf", np.array([]))
    coverage = tr.get("coverage", np.array([float("nan")]))
    oom = tr.get("out_of_manifold", np.array([]))
    out = {
        "name": name,
        "accept_rate": acc_rate,
        "acceptance_rate": acc_rate,
        "escaped": bool(int(tr["escaped"][0])),
        "escape_step": int(tr["escape_step"][0]),
        "distance_start": float(tr["dist"][0]) if tr["dist"].size else float("nan"),
        "distance_end": float(tr["dist"][-1]) if tr["dist"].size else float("nan"),
        "logdet_inv_start": float(tr["logdet_inv"][0]) if tr["logdet_inv"].size else float("nan"),
        "logdet_inv_end": float(tr["logdet_inv"][-1]) if tr["logdet_inv"].size else float("nan"),
        "logdet_start": float(tr["logdet_inv"][0]) if tr["logdet_inv"].size else float("nan"),
        "logdet_end": float(tr["logdet_inv"][-1]) if tr["logdet_inv"].size else float("nan"),
        "speed_mean": float(np.mean(tr["speed"])) if tr["speed"].size else float("nan"),
        "autocorrelation_lag1": float(acf[0]) if acf.size > 0 else float("nan"),
        "autocorrelation_lag5": float(acf[4]) if acf.size > 4 else float("nan"),
        "coverage": float(coverage[0]),
        "rescue_step_count": int(oom.sum()) if oom.size > 0 else 0,
        "rescue_step_frac": float(oom.mean()) if oom.size > 0 else 0.0,
    }
    return out


def _log_to_wandb(
    wandb_run: Any | None,
    *,
    fig_path: Path,
    summary: dict[str, Any],
    run_dir: Path,
    model_path: Path,
) -> None:
    if wandb_run is None:
        return

    payload: dict[str, Any] = {
        "three_zone/dim_0": int(summary["dims"][0]),
        "three_zone/dim_1": int(summary["dims"][1]),
        "three_zone/num_trajectories": int(len(summary.get("trajectories", []))),
    }
    for row in summary.get("trajectories", []):
        name = str(row.get("name", "unknown"))
        prefix = f"three_zone/{name}"
        for key in (
            "accept_rate",
            "distance_start",
            "distance_end",
            "logdet_inv_start",
            "logdet_inv_end",
            "speed_mean",
            "autocorrelation_lag1",
            "autocorrelation_lag5",
            "coverage",
            "rescue_step_count",
            "rescue_step_frac",
        ):
            value = row.get(key)
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                payload[f"{prefix}/{key}"] = float(value)
        payload[f"{prefix}/escaped"] = bool(row.get("escaped", False))

    safe_wandb_log(wandb_run, payload)
    img = make_wandb_image(fig_path)
    if img is not None:
        safe_wandb_log(wandb_run, {"three_zone/figure": img})

    try:
        wandb_run.summary["three_zone_output_dir"] = str(run_dir)
        wandb_run.summary["three_zone_model_path"] = str(model_path)
    except Exception:
        pass


def make_figure(
    model: GeometryRHVAE,
    dims: tuple[int, int],
    trajectories: list[tuple[str, dict[str, np.ndarray], str]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    volume_power: float,
    grid_n: int,
    out_path: Path,
    sampler_label: str,
) -> None:
    d0, d1 = dims
    sns.set_theme(style="whitegrid", context="talk")
    xx, yy, u = evaluate_potential_grid(
        model=model,
        centroids=model.centroids_tens.detach(),
        dims=dims,
        volume_power=volume_power,
        xlim=xlim,
        ylim=ylim,
        grid_n=grid_n,
    )
    u_shift = u - float(np.min(u)) + 1e-3
    u_plot = float(np.max(u_shift)) - u_shift + 1e-3
    levels = np.geomspace(float(np.min(u_plot)), float(np.max(u_plot)), 42)

    fig, axes = plt.subplots(2, 3, figsize=(22, 11), constrained_layout=True)
    ax_map = axes[0, 0]
    ax_dist = axes[0, 1]
    ax_logdet = axes[0, 2]
    ax_acc = axes[1, 0]
    ax_acf = axes[1, 1]
    ax_rescue = axes[1, 2]

    c = ax_map.contourf(
        xx,
        yy,
        u_plot,
        levels=levels,
        cmap="viridis_r",
        norm=LogNorm(vmin=float(np.min(u_plot)), vmax=float(np.max(u_plot))),
        alpha=0.92,
    )
    cb = fig.colorbar(c, ax=ax_map, pad=0.02)
    cb.set_label("Log-scaled potential field (contrast-inverted)")

    centroids = model.centroids_tens.detach().cpu().numpy()
    ax_map.scatter(
        centroids[:, d0],
        centroids[:, d1],
        marker="o",
        s=20,
        c="white",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
        zorder=7,
        label="Centroids manifold",
    )

    for name, tr, color in trajectories:
        p = tr["path"]
        p2 = p[:, [d0, d1]]
        ax_map.plot(p2[:, 0], p2[:, 1], color=color, linewidth=2.1, alpha=0.95, zorder=8, label=name)
        ax_map.scatter(p2[0, 0], p2[0, 1], color=color, marker="X", s=110, edgecolors="black", linewidths=0.8, zorder=9)
        q = np.arange(0, max(1, p2.shape[0] - 1), max(1, p2.shape[0] // 14))
        if p2.shape[0] > 1:
            q = q[q < p2.shape[0] - 1]
            ax_map.quiver(
                p2[q, 0],
                p2[q, 1],
                p2[q + 1, 0] - p2[q, 0],
                p2[q + 1, 1] - p2[q, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=color,
                width=0.0025,
                alpha=0.45,
                zorder=8,
            )

        steps = np.arange(tr["dist"].size)
        ax_dist.plot(steps, tr["dist"], color=color, linewidth=2.0, label=name)
        ax_logdet.plot(steps, tr["logdet_inv"], color=color, linewidth=2.0, label=name)
        acc = tr["accept"]
        if acc.size:
            run_acc = np.cumsum(acc) / np.arange(1, acc.size + 1)
            ax_acc.plot(np.arange(1, acc.size + 1), run_acc, color=color, linewidth=2.0, label=name)

        # Autocorrelation decay
        acf = tr.get("acf", np.array([]))
        if acf.size > 0:
            lags = np.arange(1, acf.size + 1)
            ax_acf.plot(lags, acf, color=color, linewidth=2.0, marker="o", markersize=3, label=name)

        # Rescue / out-of-manifold cumulative count
        rescue_cum = tr.get("rescue_cumsum", np.array([]))
        if rescue_cum.size > 0:
            ax_rescue.plot(np.arange(rescue_cum.size), rescue_cum, color=color, linewidth=2.0, label=name)

    ax_map.set_title(f"RHMC Three-Zone Trajectories [{sampler_label}]")
    ax_map.set_xlabel(f"z[{d0}]")
    ax_map.set_ylabel(f"z[{d1}]")
    ax_map.set_xlim(*xlim)
    ax_map.set_ylim(*ylim)
    ax_map.set_aspect("equal")
    ax_map.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=10)

    ax_dist.set_title("Distance To Manifold")
    ax_dist.set_xlabel("step")
    ax_dist.set_ylabel("min distance to centroids")
    ax_dist.legend(loc="upper right", fontsize=9)

    ax_logdet.set_title("log det(G_inv) Along Trajectory")
    ax_logdet.set_xlabel("step")
    ax_logdet.set_ylabel("log det(G_inv)")
    ax_logdet.legend(loc="lower right", fontsize=9)

    ax_acc.set_title("Running Acceptance Rate")
    ax_acc.set_xlabel("step")
    ax_acc.set_ylabel("acceptance")
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.legend(loc="lower right", fontsize=9)

    ax_acf.set_title("Autocorrelation Decay")
    ax_acf.set_xlabel("lag")
    ax_acf.set_ylabel("autocorrelation")
    ax_acf.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_acf.legend(loc="upper right", fontsize=9)

    ax_rescue.set_title("Cumulative Out-of-Manifold Steps")
    ax_rescue.set_xlabel("step")
    ax_rescue.set_ylabel("cumulative count")
    ax_rescue.legend(loc="upper left", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Illustrate RHMC from manifold / near / far starts on the real metric.")
    p.add_argument("--model_path", type=str, default="outputs/reference_models/4K")
    p.add_argument("--output_dir", type=str, default="results/rhmc_three_zone_real_metric")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model and sampler execution (e.g. auto, cpu, cuda, cuda:0).",
    )
    p.add_argument("--dims", type=str, default="0,1", help="'i,j' or 'auto'")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument(
        "--sampler_name",
        type=str,
        default="volume_riemannian",
        choices=["volume", "volume_riemannian", "hybrid_volume_mix"],
    )
    p.add_argument("--volume_power", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--n_lf_inner", type=int, default=6)
    p.add_argument("--eps", type=float, default=0.02)
    p.add_argument("--eps_jitter", type=float, default=0.0)
    p.add_argument("--n_lf_jitter", type=int, default=0)
    p.add_argument("--fp_steps", type=int, default=3)
    p.add_argument("--fp_damping", type=float, default=0.72)
    adapt_group = p.add_mutually_exclusive_group()
    adapt_group.add_argument(
        "--adaptive_dual_step",
        dest="adaptive_dual_step",
        action="store_true",
        help="Enable local dual-step adaptation (scales eps when local dual velocity is too large).",
    )
    adapt_group.add_argument(
        "--no_adaptive_dual_step",
        dest="adaptive_dual_step",
        action="store_false",
        help="Disable local dual-step adaptation.",
    )
    p.set_defaults(adaptive_dual_step=None)
    p.add_argument(
        "--adaptive_max_dual_displacement",
        type=float,
        default=None,
        help="Per-leapfrog displacement cap used by dual-step adaptation.",
    )
    p.add_argument(
        "--adaptive_min_step_scale",
        type=float,
        default=None,
        help="Minimum eps scaling factor allowed by dual-step adaptation.",
    )
    p.add_argument("--radial_prior_weight", type=float, default=0.0)
    p.add_argument("--hybrid_explore_steps", type=int, default=2)
    p.add_argument("--hybrid_rescue_steps", type=int, default=1)
    p.add_argument("--hybrid_rescue_warmup_steps", type=int, default=0)
    p.add_argument("--hybrid_rescue_use_dual_metric", action="store_true")
    p.add_argument("--manifold_points", type=int, default=3)
    p.add_argument(
        "--only_outside_starts",
        action="store_true",
        help="If set, skip manifold starts and use only near/far outside starts.",
    )
    p.add_argument("--near_points", type=int, default=1, help="Number of near-outside start points.")
    p.add_argument("--far_points", type=int, default=1, help="Number of far-outside start points.")
    p.add_argument("--near_multiplier", type=float, default=1.20)
    p.add_argument("--far_multiplier", type=float, default=2.40)
    p.add_argument("--max_radius", type=float, default=45.0)
    p.add_argument("--grid_n", type=int, default=180)
    p.add_argument(
        "--void_eigshape_mode",
        type=str,
        choices=["none", "det_preserving_spectral"],
        default="none",
    )
    p.add_argument(
        "--void_eigshape_alpha_min",
        type=float,
        default=1.0,
        help="Apply void eigenshape when alpha >= this threshold.",
    )
    p.add_argument(
        "--void_eigshape_power",
        type=float,
        default=-1.0,
        help="Det-preserving eigenshape power s in λ' = g*(λ/g)^s.",
    )
    p.add_argument(
        "--void_eigshape_eig_floor",
        type=float,
        default=1e-8,
        help="Eigenvalue floor for void eigenshape stability.",
    )
    # Deprecated aliases kept for one transition cycle.
    p.add_argument("--void_eig_flip", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--void_eig_flip_alpha_min", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--void_eig_flip_eig_floor", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--momentum_persist", type=float, default=0.35)
    # Backward-compatible aliases.
    p.add_argument("--use_dual_metric", type=str, default="False", help="True, False, or zone_aware")
    p.add_argument("--atom_scale", type=float, default=1.0)
    p.add_argument("--integrator", type=str, default="implicit", help=argparse.SUPPRESS)
    p.add_argument("--no_metropolis", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_name_mode", type=str, default="auto", choices=["timestamp", "auto", "manual"])
    p.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_job_type", type=str, default="three_zone_sampling")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wandb_run = init_wandb_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        tags=args.wandb_tags,
        run_name=args.wandb_run_name,
        name_mode=args.wandb_name_mode,
        mode=args.wandb_mode,
        config=vars(args),
        name_prefix="three_zone_sampling",
        job_type=args.wandb_job_type,
    )
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    try:
        device_str = str(args.device).lower()
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(str(args.device))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine.")
        print(f"Using device: {device}")
        model_effective: Any = load_geometry_model(Path(args.model_path), device=device)

        eigshape_mode = str(args.void_eigshape_mode)
        eigshape_alpha_min = float(args.void_eigshape_alpha_min)
        eigshape_power = float(args.void_eigshape_power)
        eigshape_floor = float(args.void_eigshape_eig_floor)
        if bool(args.void_eig_flip):
            eigshape_mode = "det_preserving_spectral"
            if args.void_eig_flip_alpha_min is not None:
                eigshape_alpha_min = float(args.void_eig_flip_alpha_min)
            if args.void_eig_flip_eig_floor is not None:
                eigshape_floor = float(args.void_eig_flip_eig_floor)

        model_effective.void_eigshape_mode = eigshape_mode
        model_effective.void_eigshape_alpha_min = eigshape_alpha_min
        model_effective.void_eigshape_power = eigshape_power
        model_effective.void_eigshape_eig_floor = eigshape_floor

        d0, d1 = parse_dims(str(args.dims), int(model_effective.latent_dim))

        model_effective.atom_scale = float(args.atom_scale)

        dual_metric_val = str(args.use_dual_metric).lower()
        if dual_metric_val == "true" or dual_metric_val == "1":
            use_dual_metric = True
        elif dual_metric_val == "false" or dual_metric_val == "0":
            use_dual_metric = False
        else:
            use_dual_metric = str(args.use_dual_metric)

        if str(args.sampler_name) == "hybrid_volume_mix":
            use_dual_metric = False

        sampler = _build_sampler(
            name=str(args.sampler_name),
            model=model_effective,
            mcmc_steps=max(1, int(args.steps)),
            n_lf=max(1, int(args.n_lf_inner)),
            eps_lf=float(args.eps),
            beta_zero=1.0,
            volume_power=float(args.volume_power),
            radial_prior_weight=float(args.radial_prior_weight),
            radial_prior_center=None,
            hybrid_explore_steps=max(1, int(args.hybrid_explore_steps)),
            hybrid_rescue_steps=max(1, int(args.hybrid_rescue_steps)),
            hybrid_rescue_warmup_steps=max(0, int(args.hybrid_rescue_warmup_steps)),
            hybrid_rescue_use_dual_metric=bool(args.hybrid_rescue_use_dual_metric),
            use_dual_metric=bool(use_dual_metric),
        )
        sampler.exact = (str(args.integrator).lower() != "explicit")
        if hasattr(sampler, "fp_steps"):
            sampler.fp_steps = int(args.fp_steps)
        if hasattr(sampler, "fp_damping"):
            sampler.fp_damping = float(args.fp_damping)
        if args.adaptive_dual_step is not None and hasattr(sampler, "adaptive_dual_step"):
            sampler.adaptive_dual_step = bool(args.adaptive_dual_step)
        if args.adaptive_max_dual_displacement is not None and hasattr(sampler, "adaptive_max_dual_displacement"):
            sampler.adaptive_max_dual_displacement = float(args.adaptive_max_dual_displacement)
        if args.adaptive_min_step_scale is not None and hasattr(sampler, "adaptive_min_step_scale"):
            sampler.adaptive_min_step_scale = float(args.adaptive_min_step_scale)
        sampler.momentum_persist = float(args.momentum_persist)

        centroids = model_effective.centroids_tens.detach()
        near_starts, far_starts = make_outside_starts(
            centroids=centroids,
            dims=(d0, d1),
            near_multiplier=float(args.near_multiplier),
            far_multiplier=float(args.far_multiplier),
            near_points=max(1, int(args.near_points)),
            far_points=max(1, int(args.far_points)),
        )

        starts: list[tuple[str, torch.Tensor, str]] = []
        if not bool(args.only_outside_starts):
            starts_manifold = choose_manifold_starts(centroids, (d0, d1), int(args.manifold_points))
            manifold_colors = ["#4c78a8", "#72b7b2", "#9ecae9", "#a0cbe8", "#1f77b4"]
            for i, z0 in enumerate(starts_manifold):
                starts.append((f"manifold_{i + 1}", z0, manifold_colors[i % len(manifold_colors)]))

        near_colors = ["#f58518", "#ff9f40", "#c96d0e", "#ffb869", "#de7d20"]
        far_colors = ["#d62728", "#ff4d4f", "#b5161b", "#ff7f7f", "#8b1a1a"]
        single_near = len(near_starts) == 1
        single_far = len(far_starts) == 1
        for i, z0 in enumerate(near_starts):
            name = "near_outside" if single_near else f"near_outside_{i + 1}"
            starts.append((name, z0, near_colors[i % len(near_colors)]))
        for i, z0 in enumerate(far_starts):
            name = "far_outside" if single_far else f"far_outside_{i + 1}"
            starts.append((name, z0, far_colors[i % len(far_colors)]))

        if len(starts) == 0:
            raise ValueError("No start points were generated; adjust manifold/near/far point counts.")

        trajectories: list[tuple[str, dict[str, np.ndarray], str]] = []
        runtime_diagnostics_by_name: dict[str, dict[str, Any]] = {}
        for name, z0, color in starts:
            if hasattr(sampler, "_reset_runtime_diagnostics"):
                sampler._reset_runtime_diagnostics()
            tr = simulate_chain(
                sampler=sampler,
                model=model_effective,
                z_start=z0,
                steps=int(args.steps),
                n_lf_inner=int(args.n_lf_inner),
                eps=float(args.eps),
                max_radius=float(args.max_radius),
                eps_jitter=float(args.eps_jitter),
                n_lf_jitter=int(args.n_lf_jitter),
                no_metropolis=bool(args.no_metropolis),
            )
            trajectories.append((name, tr, color))
            if hasattr(sampler, "_finalize_runtime_diagnostics"):
                sampler._finalize_runtime_diagnostics()
            if hasattr(sampler, "get_runtime_diagnostics"):
                runtime_diagnostics_by_name[name] = dict(sampler.get_runtime_diagnostics())

        starts_np = np.vstack([z.detach().cpu().numpy() for _, z, _ in starts])
        traj_only = [t for _, t, _ in trajectories]
        xlim, ylim = compute_bounds(
            centroids=centroids.detach().cpu().numpy(),
            starts=starts_np,
            trajs=traj_only,
            dims=(d0, d1),
        )

        run_dir = Path(args.output_dir) / dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fig_path = run_dir / "rhmc_three_zones_real_metric.png"
        make_figure(
            model=model_effective,
            dims=(d0, d1),
            trajectories=trajectories,
            xlim=xlim,
            ylim=ylim,
            volume_power=float(args.volume_power),
            grid_n=int(args.grid_n),
            out_path=fig_path,
            sampler_label=str(args.sampler_name),
        )

        summary_rows: list[dict[str, Any]] = []
        for name, tr, _ in trajectories:
            row = summarize_traj(name, tr)
            diag = runtime_diagnostics_by_name.get(name)
            if diag:
                row["runtime_diagnostics"] = diag
            summary_rows.append(row)

        summary = {
            "args": vars(args),
            "metric_eigshape": {
                "void_eigshape_mode": eigshape_mode,
                "void_eigshape_alpha_min": eigshape_alpha_min,
                "void_eigshape_power": eigshape_power,
                "void_eigshape_eig_floor": eigshape_floor,
            },
            "dims": [int(d0), int(d1)],
            "trajectories": summary_rows,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _log_to_wandb(
            wandb_run,
            fig_path=fig_path,
            summary=summary,
            run_dir=run_dir,
            model_path=Path(args.model_path),
        )

        print(f"Saved figure: {fig_path}")
        print(f"Saved summary: {run_dir / 'summary.json'}")
        print(
            "Void eigenshape: "
            f"mode={eigshape_mode}, "
            f"alpha_min={eigshape_alpha_min:.6f}, "
            f"power={eigshape_power:.4f}, "
            f"eig_floor={eigshape_floor:.2e}"
        )
        if hasattr(sampler, "adaptive_dual_step"):
            print(
                "Dual-step adaptation: "
                f"enabled={bool(getattr(sampler, 'adaptive_dual_step', False))}, "
                f"max_disp={float(getattr(sampler, 'adaptive_max_dual_displacement', float('nan'))):.6f}, "
                f"min_scale={float(getattr(sampler, 'adaptive_min_step_scale', float('nan'))):.6f}"
            )
        for row in summary["trajectories"]:
            print(
                f"{row['name']}: acc={row['accept_rate']:.3f}, "
                f"d0={row['distance_start']:.3f} -> dT={row['distance_end']:.3f}, "
                f"logdet0={row['logdet_inv_start']:.3f} -> logdetT={row['logdet_inv_end']:.3f}, "
                f"escaped={row['escaped']}"
            )
            print(
                f"  acf_lag1={row.get('autocorrelation_lag1', float('nan')):.3f}, "
                f"acf_lag5={row.get('autocorrelation_lag5', float('nan')):.3f}, "
                f"coverage={row.get('coverage', float('nan')):.3f}, "
                f"rescue_steps={row.get('rescue_step_count', 0)} "
                f"({row.get('rescue_step_frac', 0.0):.1%})"
            )
            diag = row.get("runtime_diagnostics")
            if isinstance(diag, dict) and len(diag) > 0:
                print(
                    "  fp:"
                    f" calls={int(diag.get('fp_calls', 0))},"
                    f" sat_m={float(diag.get('momentum_fp_saturation_rate', 0.0)):.2%},"
                    f" sat_z={float(diag.get('position_fp_saturation_rate', 0.0)):.2%},"
                    f" eps_scale_mean={float(diag.get('eps_scale_mean', 1.0)):.3f},"
                    f" chol_fallbacks={int(diag.get('eigh_fallbacks', 0))}"
                )
    finally:
        safe_wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
