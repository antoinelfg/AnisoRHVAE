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

    return {
        "path": path,
        "dist": dist,
        "logdet_inv": logdet_inv,
        "speed": speed.astype(np.float64),
        "accept": accept,
        "escaped": np.asarray([1.0 if escaped else 0.0], dtype=np.float64),
        "escape_step": np.asarray([float(escape_step)], dtype=np.float64),
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


def make_outside_starts(
    centroids: torch.Tensor,
    dims: tuple[int, int],
    near_multiplier: float,
    far_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    d0, d1 = dims
    c2 = centroids[:, [d0, d1]]
    center2 = c2.mean(dim=0)
    shifted = c2 - center2
    radii = torch.linalg.norm(shifted, dim=1)
    idx = int(torch.argmax(radii).item())
    direction2 = shifted[idx] / (radii[idx] + 1e-9)
    r_max = float(radii[idx].item())

    z_center = centroids.mean(dim=0)
    z_near = z_center.clone()
    z_far = z_center.clone()
    z_near[d0] = center2[0] + float(near_multiplier) * r_max * direction2[0]
    z_near[d1] = center2[1] + float(near_multiplier) * r_max * direction2[1]
    z_far[d0] = center2[0] + float(far_multiplier) * r_max * direction2[0]
    z_far[d1] = center2[1] + float(far_multiplier) * r_max * direction2[1]
    return z_near.detach(), z_far.detach()


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
    out = {
        "name": name,
        "accept_rate": acc_rate,
        "escaped": bool(int(tr["escaped"][0])),
        "escape_step": int(tr["escape_step"][0]),
        "distance_start": float(tr["dist"][0]) if tr["dist"].size else float("nan"),
        "distance_end": float(tr["dist"][-1]) if tr["dist"].size else float("nan"),
        "logdet_inv_start": float(tr["logdet_inv"][0]) if tr["logdet_inv"].size else float("nan"),
        "logdet_inv_end": float(tr["logdet_inv"][-1]) if tr["logdet_inv"].size else float("nan"),
        "speed_mean": float(np.mean(tr["speed"])) if tr["speed"].size else float("nan"),
    }
    return out


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

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    ax_map, ax_dist, ax_logdet, ax_acc = axes.ravel()

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Illustrate RHMC from manifold / near / far starts on the real metric.")
    p.add_argument("--model_path", type=str, default="outputs/reference_models/4K")
    p.add_argument("--output_dir", type=str, default="results/rhmc_three_zone_real_metric")
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
    p.add_argument("--radial_prior_weight", type=float, default=0.0)
    p.add_argument("--hybrid_explore_steps", type=int, default=2)
    p.add_argument("--hybrid_rescue_steps", type=int, default=1)
    p.add_argument("--hybrid_rescue_warmup_steps", type=int, default=0)
    p.add_argument("--hybrid_rescue_use_dual_metric", action="store_true")
    p.add_argument("--manifold_points", type=int, default=3)
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
    p.add_argument("--use_dual_metric", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--integrator", type=str, default="implicit", help=argparse.SUPPRESS)
    p.add_argument("--no_metropolis", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device("cpu")
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

    use_dual_metric = bool(args.use_dual_metric)
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
    sampler.exact = True
    if hasattr(sampler, "fp_steps"):
        sampler.fp_steps = int(args.fp_steps)
    if hasattr(sampler, "fp_damping"):
        sampler.fp_damping = float(args.fp_damping)
    sampler.momentum_persist = float(args.momentum_persist)

    centroids = model_effective.centroids_tens.detach()
    starts_manifold = choose_manifold_starts(centroids, (d0, d1), int(args.manifold_points))
    z_near, z_far = make_outside_starts(
        centroids=centroids,
        dims=(d0, d1),
        near_multiplier=float(args.near_multiplier),
        far_multiplier=float(args.far_multiplier),
    )

    starts: list[tuple[str, torch.Tensor, str]] = []
    manifold_colors = ["#4c78a8", "#72b7b2", "#9ecae9", "#a0cbe8", "#1f77b4"]
    for i, z0 in enumerate(starts_manifold):
        starts.append((f"manifold_{i + 1}", z0, manifold_colors[i % len(manifold_colors)]))
    starts.append(("near_outside", z_near, "#f58518"))
    starts.append(("far_outside", z_far, "#d62728"))

    trajectories: list[tuple[str, dict[str, np.ndarray], str]] = []
    for name, z0, color in starts:
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
        )
        trajectories.append((name, tr, color))

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

    summary = {
        "args": vars(args),
        "metric_eigshape": {
            "void_eigshape_mode": eigshape_mode,
            "void_eigshape_alpha_min": eigshape_alpha_min,
            "void_eigshape_power": eigshape_power,
            "void_eigshape_eig_floor": eigshape_floor,
        },
        "dims": [int(d0), int(d1)],
        "trajectories": [summarize_traj(name, tr) for name, tr, _ in trajectories],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {run_dir / 'summary.json'}")
    print(
        "Void eigenshape: "
        f"mode={eigshape_mode}, "
        f"alpha_min={eigshape_alpha_min:.6f}, "
        f"power={eigshape_power:.4f}, "
        f"eig_floor={eigshape_floor:.2e}"
    )
    for row in summary["trajectories"]:
        print(
            f"{row['name']}: acc={row['accept_rate']:.3f}, "
            f"d0={row['distance_start']:.3f} -> dT={row['distance_end']:.3f}, "
            f"logdet0={row['logdet_inv_start']:.3f} -> logdetT={row['logdet_inv_end']:.3f}, "
            f"escaped={row['escaped']}"
        )


if __name__ == "__main__":
    main()
