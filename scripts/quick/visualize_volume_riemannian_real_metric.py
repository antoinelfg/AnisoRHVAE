#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import seaborn as sns
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from src.models.rhvae_geometry import GeometryRHVAE, GeometryRHVAEConfig
from src.models.samplers.hmc_sampler import VolumeElementRiemannianHMCSampler
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


def select_void_start(centroids: torch.Tensor, multiplier: float = 4.6) -> torch.Tensor:
    center = centroids.mean(dim=0)
    shifted = centroids - center
    cov = (shifted.T @ shifted) / max(1, shifted.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)
    v1 = evecs[:, -1]
    v2 = evecs[:, -2] if evecs.shape[1] > 1 else torch.tensor([0.0, 1.0], device=centroids.device)
    direction = v2 - v1
    direction = direction / (torch.linalg.norm(direction) + 1e-9)
    r_max = torch.linalg.norm(shifted, dim=1).max()
    return (center + float(multiplier) * r_max * direction).unsqueeze(0)


def select_dancer_start(model: GeometryRHVAE) -> torch.Tensor:
    c = model.centroids_tens
    center = c.mean(dim=0, keepdim=True)
    idx = torch.argmin(torch.linalg.norm(c - center, dim=1))
    return c[idx : idx + 1].detach().clone()


def potential_volume_riemannian(
    model: GeometryRHVAE,
    z: torch.Tensor,
    volume_power: float,
) -> torch.Tensor:
    eff = float(volume_power) + 0.5
    g_inv = model.G_inv(z)
    logdet_inv = torch.linalg.slogdet(g_inv).logabsdet
    return -eff * logdet_inv


def velocity(sampler: VolumeElementRiemannianHMCSampler, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    return sampler._velocity(z, rho)


def explicit_riemannian_step(
    sampler: VolumeElementRiemannianHMCSampler,
    z: torch.Tensor,
    rho: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate explicit RMHMC step (still metric-aware, non-implicit)."""
    grad_z, _ = sampler._grad_hamiltonian_z(z, rho)
    if not torch.isfinite(grad_z).all():
        return z.detach().requires_grad_(True), rho.detach()
    rho_half = rho - 0.5 * float(eps) * grad_z
    z_new = z + float(eps) * velocity(sampler, z, rho_half)
    if not torch.isfinite(z_new).all():
        return z.detach().requires_grad_(True), rho.detach()
    z_new = z_new.detach().requires_grad_(True)
    grad_z_new, _ = sampler._grad_hamiltonian_z(z_new, rho_half)
    if not torch.isfinite(grad_z_new).all():
        return z.detach().requires_grad_(True), rho.detach()
    rho_new = rho_half - 0.5 * float(eps) * grad_z_new
    return z_new, rho_new.detach()


def riemannian_step(
    sampler: VolumeElementRiemannianHMCSampler,
    z: torch.Tensor,
    rho: torch.Tensor,
    eps: float,
    integrator: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if integrator == "implicit":
        return sampler._generalized_leapfrog_step(z, rho, float(eps))
    if integrator == "explicit_riemannian":
        return explicit_riemannian_step(sampler, z, rho, float(eps))
    raise ValueError(f"Unknown integrator: {integrator}")


def select_momentum_toward_target(
    sampler: VolumeElementRiemannianHMCSampler,
    z: torch.Tensor,
    target: torch.Tensor,
    generator: torch.Generator,
    trials: int = 96,
) -> torch.Tensor:
    best_rho = None
    best_score = -float("inf")
    z_b = z if z.ndim == 2 else z.unsqueeze(0)
    target_b = target if target.ndim == 2 else target.unsqueeze(0)
    direction = target_b - z_b
    dir_norm = torch.linalg.norm(direction).item()
    if dir_norm <= 1e-9:
        return sampler._initialize_momentum(z_b)

    for _ in range(trials):
        # Keep stochastic draw from N(0, G(z)); then keep the most target-aligned.
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()))
        rho = sampler._initialize_momentum(z_b)
        v = velocity(sampler, z_b, rho)
        denom = (torch.linalg.norm(v) * torch.linalg.norm(direction)).item()
        if denom <= 1e-12:
            continue
        cos = float(torch.sum(v * direction).item() / denom)
        speed = float(torch.linalg.norm(v).item())
        score = cos + 0.05 * speed
        if score > best_score:
            best_score = score
            best_rho = rho

    if best_rho is None:
        best_rho = sampler._initialize_momentum(z_b)
    return best_rho


def rollout_score_parachutist(
    sampler: VolumeElementRiemannianHMCSampler,
    model: GeometryRHVAE,
    z_start: torch.Tensor,
    rho_init: torch.Tensor,
    eps_lf: float,
    n_lf_inner: int,
    steps: int,
    integrator: str,
    max_radius: float,
) -> tuple[float, float, bool]:
    z = z_start.clone().detach()
    if z.ndim == 1:
        z = z.unsqueeze(0)
    z = z.requires_grad_(True)
    rho = rho_init.clone().detach()
    if rho.ndim == 1:
        rho = rho.unsqueeze(0)

    with torch.no_grad():
        logdet_start = torch.linalg.slogdet(model.G_inv(z)).logabsdet.item()
    mean_speed = 0.0
    escaped = False
    for _ in range(steps):
        for _ in range(n_lf_inner):
            z, rho = riemannian_step(
                sampler=sampler,
                z=z,
                rho=rho,
                eps=float(eps_lf),
                integrator=integrator,
            )
            z = z.detach().requires_grad_(True)
            rho = rho.detach()
            if not torch.isfinite(z).all() or not torch.isfinite(rho).all():
                escaped = True
                break
        if escaped:
            break
        with torch.no_grad():
            mean_speed += float(torch.linalg.norm(velocity(sampler, z, rho)).item())
            radius = float(torch.linalg.norm(z).item())
            if radius > float(max_radius):
                escaped = True
                break
    mean_speed /= max(1, steps)
    with torch.no_grad():
        logdet_end = torch.linalg.slogdet(model.G_inv(z)).logabsdet.item()
    return logdet_end - logdet_start, mean_speed, escaped


def tune_parachutist_momentum(
    sampler: VolumeElementRiemannianHMCSampler,
    model: GeometryRHVAE,
    z_start: torch.Tensor,
    target: torch.Tensor,
    generator: torch.Generator,
    trials: int,
    eps_lf: float,
    n_lf_inner: int,
    score_steps: int = 72,
    integrator: str = "implicit",
    max_radius: float = 50.0,
) -> torch.Tensor:
    z_b = z_start if z_start.ndim == 2 else z_start.unsqueeze(0)
    target_b = target if target.ndim == 2 else target.unsqueeze(0)
    direction = target_b - z_b
    best_rho = None
    best_score = -float("inf")
    for _ in range(trials):
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()))
        rho = sampler._initialize_momentum(z_b)
        v = velocity(sampler, z_b, rho)
        denom = (torch.linalg.norm(v) * torch.linalg.norm(direction)).item()
        cos_term = 0.0
        if denom > 1e-12:
            cos_term = float(torch.sum(v * direction).item() / denom)
        delta_logdet, mean_speed, escaped = rollout_score_parachutist(
            sampler=sampler,
            model=model,
            z_start=z_b,
            rho_init=rho,
            eps_lf=eps_lf,
            n_lf_inner=n_lf_inner,
            steps=score_steps,
            integrator=integrator,
            max_radius=max_radius,
        )
        if escaped:
            delta_logdet -= 50.0
            mean_speed *= 0.25
        score = 5.0 * delta_logdet + 0.5 * cos_term + 0.03 * mean_speed
        if score > best_score:
            best_score = score
            best_rho = rho
    if best_rho is None:
        return sampler._initialize_momentum(z_b)
    return best_rho


def simulate_no_metropolis(
    sampler: VolumeElementRiemannianHMCSampler,
    model: GeometryRHVAE,
    z_start: torch.Tensor,
    rho_init: torch.Tensor,
    steps: int,
    eps_lf: float,
    n_lf_inner: int,
    volume_power: float,
    integrator: str,
    max_radius: float,
    use_metropolis: bool = False,
) -> dict[str, np.ndarray]:
    z = z_start.clone().detach()
    if z.ndim == 1:
        z = z.unsqueeze(0)
    z = z.requires_grad_(True)
    rho = rho_init.clone().detach()
    if rho.ndim == 1:
        rho = rho.unsqueeze(0)

    path = [z.detach().cpu().numpy().squeeze(0)]
    speed = []
    h_vals = []
    u_vals = []
    logdet_inv_vals = []
    escaped = False
    escape_step = -1
    accept_hist: list[float] = []

    for step_idx in range(steps):
        if use_metropolis:
            # True RHMC transition: fresh momentum + proposal + accept/reject.
            z_curr = z.detach().clone().requires_grad_(True)
            rho_curr = sampler._initialize_momentum(z_curr).detach()
            with torch.no_grad():
                h0 = sampler._compute_hamiltonian(z_curr, rho_curr)
            z_prop, rho_prop = z_curr, rho_curr
            for _ in range(n_lf_inner):
                z_prop, rho_prop = riemannian_step(
                    sampler=sampler,
                    z=z_prop,
                    rho=rho_prop,
                    eps=float(eps_lf),
                    integrator=integrator,
                )
                z_prop = z_prop.detach().requires_grad_(True)
                rho_prop = rho_prop.detach()
                if not torch.isfinite(z_prop).all() or not torch.isfinite(rho_prop).all():
                    escaped = True
                    escape_step = step_idx
                    break
            if escaped:
                break
            with torch.no_grad():
                h1 = sampler._compute_hamiltonian(z_prop, rho_prop)
                alpha = torch.exp(torch.clamp(h0 - h1, max=0.0))
                u = torch.rand_like(alpha)
                accept = float((u < alpha).float().item())
                z = (accept * z_prop + (1.0 - accept) * z_curr).detach().requires_grad_(True)
                rho = rho_prop.detach()
            accept_hist.append(accept)
        else:
            for _ in range(n_lf_inner):
                z, rho = riemannian_step(
                    sampler=sampler,
                    z=z,
                    rho=rho,
                    eps=float(eps_lf),
                    integrator=integrator,
                )
                z = z.detach().requires_grad_(True)
                rho = rho.detach()
                if not torch.isfinite(z).all() or not torch.isfinite(rho).all():
                    escaped = True
                    escape_step = step_idx
                    break
            if escaped:
                break
            accept_hist.append(1.0)
        with torch.no_grad():
            v = velocity(sampler, z, rho)
            speed.append(float(torch.linalg.norm(v).item()))
            h = sampler._compute_hamiltonian(z, rho).item()
            u = potential_volume_riemannian(model, z, volume_power=volume_power).item()
            logdet_inv = torch.linalg.slogdet(model.G_inv(z)).logabsdet.item()
        h_vals.append(h)
        u_vals.append(u)
        logdet_inv_vals.append(logdet_inv)
        path.append(z.detach().cpu().numpy().squeeze(0))
        if float(torch.linalg.norm(z.detach()).item()) > float(max_radius):
            escaped = True
            escape_step = step_idx + 1
            break

    return {
        "path": np.asarray(path, dtype=np.float64),
        "speed": np.asarray(speed, dtype=np.float64),
        "hamiltonian": np.asarray(h_vals, dtype=np.float64),
        "potential": np.asarray(u_vals, dtype=np.float64),
        "logdet_inv": np.asarray(logdet_inv_vals, dtype=np.float64),
        "escaped": np.asarray([1.0 if escaped else 0.0], dtype=np.float64),
        "escape_step": np.asarray([float(escape_step)], dtype=np.float64),
        "accept_rate": np.asarray([float(np.mean(accept_hist)) if accept_hist else np.nan], dtype=np.float64),
    }


def evaluate_potential_grid(
    model: GeometryRHVAE,
    volume_power: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    n_grid: int,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    z_all = torch.as_tensor(points, dtype=torch.float32, device=model.device)

    vals = []
    with torch.no_grad():
        for i in range(0, z_all.shape[0], batch_size):
            chunk = z_all[i : i + batch_size]
            u = potential_volume_riemannian(model, chunk, volume_power=volume_power)
            vals.append(u.detach().cpu().numpy())
    u_np = np.concatenate(vals, axis=0).reshape(n_grid, n_grid)
    return xx, yy, u_np


def generate_void_starts(
    centroids: torch.Tensor,
    n_starts: int,
    multiplier_min: float,
    multiplier_max: float,
) -> list[torch.Tensor]:
    """Place several starts in the far void around the manifold support."""
    if centroids.shape[1] != 2:
        raise ValueError("This visualization expects latent_dim=2.")

    center = centroids.mean(dim=0)
    shifted = centroids - center
    cov = (shifted.T @ shifted) / max(1, shifted.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)
    e1 = evecs[:, -1]
    e2 = evecs[:, -2] if evecs.shape[1] > 1 else torch.tensor([0.0, 1.0], device=centroids.device)
    r_max = torch.linalg.norm(shifted, dim=1).max()

    starts: list[torch.Tensor] = []
    angles = torch.linspace(0.0, 2.0 * torch.pi, int(n_starts) + 1, device=centroids.device)[:-1]
    for a in angles:
        direction = torch.cos(a) * e1 + torch.sin(a) * e2
        direction = direction / (torch.linalg.norm(direction) + 1e-9)
        phase = 0.5 * (1.0 + torch.sin(3.0 * a))
        mult = float(multiplier_min) + (float(multiplier_max) - float(multiplier_min)) * float(phase)
        z0 = center + mult * r_max * direction
        starts.append(z0.unsqueeze(0).detach().clone())
    return starts


def make_plot_multi_void(
    model: GeometryRHVAE,
    traj_void: list[dict[str, np.ndarray]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    n_grid: int,
    volume_power: float,
    out_path: Path,
    integrator: str,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    xx, yy, u = evaluate_potential_grid(
        model=model,
        volume_power=volume_power,
        xlim=xlim,
        ylim=ylim,
        n_grid=n_grid,
    )

    u_shift = u - float(np.min(u)) + 1e-3
    u_plot = float(np.max(u_shift)) - u_shift + 1e-3
    levels = np.geomspace(float(np.min(u_plot)), float(np.max(u_plot)), 42)

    fig, ax = plt.subplots(figsize=(12, 10))
    c = ax.contourf(
        xx,
        yy,
        u_plot,
        levels=levels,
        cmap="viridis_r",
        norm=LogNorm(vmin=float(np.min(u_plot)), vmax=float(np.max(u_plot))),
        alpha=0.92,
    )
    cb = fig.colorbar(c, ax=ax, pad=0.02)
    cb.set_label("Log-scaled potential field (contrast-inverted)")

    centroids = model.centroids_tens.detach().cpu().numpy()
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="o",
        s=20,
        c="white",
        edgecolors="black",
        linewidths=0.55,
        alpha=0.90,
        zorder=7,
        label="Centroids manifold",
    )

    cmap = plt.get_cmap("turbo")
    for i, tr in enumerate(traj_void):
        p = tr["path"]
        color = cmap(i / max(1, len(traj_void) - 1))
        label = "Trajectoires void RHMC" if i == 0 else None
        ax.plot(p[:, 0], p[:, 1], color=color, linewidth=1.8, alpha=0.9, zorder=8, label=label)
        ax.scatter(p[0, 0], p[0, 1], color=color, marker="X", s=80, edgecolors="black", linewidths=0.7, zorder=9)
        q = np.arange(0, max(1, p.shape[0] - 1), max(1, p.shape[0] // 12))
        if p.shape[0] > 1 and q.size > 0:
            q = q[q < p.shape[0] - 1]
            ax.quiver(
                p[q, 0],
                p[q, 1],
                p[q + 1, 0] - p[q, 0],
                p[q + 1, 1] - p[q, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=color,
                width=0.0024,
                alpha=0.45,
                zorder=8,
            )

    c_center = centroids.mean(axis=0, keepdims=True)
    mid = centroids[np.argmin(np.linalg.norm(centroids - c_center, axis=1))]
    far = traj_void[0]["path"][0]
    ax.annotate(
        "VOID starts (RHMC exact)\nmetrique + Hamiltonien dictent la dynamique",
        xy=(float(far[0]), float(far[1])),
        xytext=(xlim[0] + 0.8, ylim[1] - 1.3),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.3},
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "black", "alpha": 0.9},
    )
    ax.annotate(
        "MANIFOLD 4K (aniso)\nzone de fort det(G^-1)",
        xy=(float(mid[0]), float(mid[1])),
        xytext=(xlim[0] + 0.56 * (xlim[1] - xlim[0]), ylim[0] + 0.72 * (ylim[1] - ylim[0])),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.3},
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "black", "alpha": 0.9},
    )

    ax.set_title(f"RHMC Exact Dynamics from Multiple Void Starts on Real 4K Metric [{integrator}]")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", frameon=True, framealpha=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plot(
    model: GeometryRHVAE,
    traj_a: dict[str, np.ndarray],
    traj_b: dict[str, np.ndarray],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    n_grid: int,
    volume_power: float,
    out_path: Path,
    integrator: str,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    xx, yy, u = evaluate_potential_grid(
        model=model,
        volume_power=volume_power,
        xlim=xlim,
        ylim=ylim,
        n_grid=n_grid,
    )

    # Positive log field with inverted contrast so low potential appears dark.
    u_shift = u - float(np.min(u)) + 1e-3
    u_plot = float(np.max(u_shift)) - u_shift + 1e-3
    levels = np.geomspace(float(np.min(u_plot)), float(np.max(u_plot)), 42)

    fig, ax = plt.subplots(figsize=(12, 10))
    c = ax.contourf(
        xx,
        yy,
        u_plot,
        levels=levels,
        cmap="viridis_r",
        norm=LogNorm(vmin=float(np.min(u_plot)), vmax=float(np.max(u_plot))),
        alpha=0.92,
    )
    cb = fig.colorbar(c, ax=ax, pad=0.02)
    cb.set_label("Log-scaled potential field (contrast-inverted)")

    centroids = model.centroids_tens.detach().cpu().numpy()
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="o",
        s=20,
        c="white",
        edgecolors="black",
        linewidths=0.55,
        alpha=0.90,
        zorder=7,
        label="Centroids manifold",
    )

    pa = traj_a["path"]
    pb = traj_b["path"]
    ax.plot(pa[:, 0], pa[:, 1], color="#d62728", linewidth=2.3, label="Trajectoire A: Parachutiste", zorder=8)
    ax.plot(pb[:, 0], pb[:, 1], color="#00bcd4", linewidth=2.1, label="Trajectoire B: Danseur", zorder=8)
    ax.scatter(pa[0, 0], pa[0, 1], c="#d62728", marker="X", s=180, edgecolors="black", linewidths=1.0, zorder=9)
    ax.scatter(pb[0, 0], pb[0, 1], c="#00bcd4", marker="X", s=180, edgecolors="black", linewidths=1.0, zorder=9)

    idx_a = np.arange(0, pa.shape[0], max(1, pa.shape[0] // 36))
    idx_b = np.arange(0, pb.shape[0], max(1, pb.shape[0] // 36))
    ax.scatter(pa[idx_a, 0], pa[idx_a, 1], c=np.linspace(0.2, 1.0, len(idx_a)), cmap="Reds", s=25, zorder=9)
    ax.scatter(pb[idx_b, 0], pb[idx_b, 1], c=np.linspace(0.2, 1.0, len(idx_b)), cmap="winter", s=25, zorder=9)

    qa = np.arange(0, pa.shape[0] - 1, max(1, pa.shape[0] // 28))
    qb = np.arange(0, pb.shape[0] - 1, max(1, pb.shape[0] // 28))
    ax.quiver(
        pa[qa, 0],
        pa[qa, 1],
        pa[qa + 1, 0] - pa[qa, 0],
        pa[qa + 1, 1] - pa[qa, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#d62728",
        width=0.0030,
        alpha=0.55,
        zorder=8,
    )
    ax.quiver(
        pb[qb, 0],
        pb[qb, 1],
        pb[qb + 1, 0] - pb[qb, 0],
        pb[qb + 1, 1] - pb[qb, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#00bcd4",
        width=0.0030,
        alpha=0.65,
        zorder=8,
    )

    ax.annotate(
        "VOID (loin)\n(masse elevee, vitesse faible)",
        xy=(pa[0, 0], pa[0, 1]),
        xytext=(xlim[0] + 0.8, ylim[1] - 1.2),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.3},
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "black", "alpha": 0.9},
    )
    c_center = centroids.mean(axis=0, keepdims=True)
    mid = centroids[np.argmin(np.linalg.norm(centroids - c_center, axis=1))]
    ax.annotate(
        "MANIFOLD 4K (aniso)\n(masse plus faible, exploration agile)",
        xy=(float(mid[0]), float(mid[1])),
        xytext=(xlim[0] + 0.52 * (xlim[1] - xlim[0]), ylim[0] + 0.70 * (ylim[1] - ylim[0])),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.3},
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "black", "alpha": 0.9},
    )

    ax.set_title(f"VolumeElementRiemannianHMC Dynamics on Real 4K Metric (Aniso) [{integrator}]")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", frameon=True, framealpha=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_plot_limits(
    core_pts: np.ndarray,
    starts_pts: np.ndarray,
    traj_list: list[dict[str, np.ndarray]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    center = core_pts.mean(axis=0)
    all_paths = [t["path"] for t in traj_list if t["path"].size > 0]
    if all_paths:
        path_pts = np.vstack(all_paths)
        all_pts = np.vstack([core_pts, starts_pts, path_pts])
    else:
        all_pts = np.vstack([core_pts, starts_pts])

    core_span = max(float(np.max(np.ptp(core_pts, axis=0))), 1e-6)
    start_radius = float(np.max(np.linalg.norm(starts_pts - center[None, :], axis=1)))
    max_plot_radius = max(1.12 * start_radius, 2.8 * core_span, 8.0)
    keep = np.linalg.norm(all_pts - center[None, :], axis=1) <= max_plot_radius
    pts = np.vstack([core_pts, all_pts[keep]])

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    pad_x = max(0.8, 0.10 * (x_max - x_min))
    pad_y = max(0.8, 0.10 * (y_max - y_min))
    xlim = (float(x_min - pad_x), float(x_max + pad_x))
    ylim = (float(y_min - pad_y), float(y_max + pad_y))
    return xlim, ylim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real 4K volume_riemannian dynamics visualization.")
    parser.add_argument("--model_path", type=str, default="outputs/reference_models/4K")
    parser.add_argument("--mode", type=str, choices=["multi_void", "two_paths"], default="multi_void")
    parser.add_argument("--volume_power", type=float, default=0.8)
    parser.add_argument(
        "--integrator",
        type=str,
        choices=["implicit", "explicit_riemannian"],
        default="implicit",
    )
    parser.add_argument("--fp_steps", type=int, default=3)
    parser.add_argument("--fp_damping", type=float, default=0.72)
    parser.add_argument("--steps_a", type=int, default=240)
    parser.add_argument("--steps_b", type=int, default=220)
    parser.add_argument("--steps_void", type=int, default=90)
    parser.add_argument("--n_void_starts", type=int, default=12)
    parser.add_argument("--void_multiplier_min", type=float, default=3.8)
    parser.add_argument("--void_multiplier_max", type=float, default=6.2)
    parser.add_argument("--n_lf_inner_void", type=int, default=3)
    parser.add_argument("--eps_void", type=float, default=0.015)
    parser.add_argument("--n_lf_inner_a", type=int, default=5)
    parser.add_argument("--n_lf_inner_b", type=int, default=5)
    parser.add_argument("--eps_a", type=float, default=0.008)
    parser.add_argument("--eps_b", type=float, default=0.006)
    parser.add_argument("--void_multiplier", type=float, default=4.8)
    parser.add_argument("--momentum_trials_a", type=int, default=28)
    parser.add_argument("--momentum_trials_b", type=int, default=64)
    parser.add_argument("--momentum_score_steps_a", type=int, default=32)
    parser.add_argument("--max_radius", type=float, default=45.0)
    parser.add_argument("--grid_n", type=int, default=180)
    parser.add_argument("--momentum_persist", type=float, default=0.85)
    parser.add_argument("--use_dual_metric", action="store_true")
    parser.add_argument("--no_metropolis", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="results/volume_riemannian_4k_visual/volume_riemannian_4k_multi_void_rhmc.png",
    )
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device("cpu")
    model_path = Path(args.model_path)
    model = load_geometry_model(model_path=model_path, device=device)

    sampler = VolumeElementRiemannianHMCSampler(
        model,
        mcmc_steps_nbr=1,
        n_lf=1,
        eps_lf=float(args.eps_b),
        exact=True,
        fp_steps=int(args.fp_steps),
        fp_damping=float(args.fp_damping),
        volume_power=float(args.volume_power),
        use_dual_metric=bool(args.use_dual_metric),
    )
    sampler.momentum_persist = float(args.momentum_persist)

    centroids = model.centroids_tens.detach()
    integrator = str(args.integrator)

    if str(args.mode) == "multi_void":
        if integrator != "implicit":
            print("For mode=multi_void, forcing implicit generalized leapfrog (exact RHMC dynamics).")
            integrator = "implicit"

        starts = generate_void_starts(
            centroids=centroids,
            n_starts=int(args.n_void_starts),
            multiplier_min=float(args.void_multiplier_min),
            multiplier_max=float(args.void_multiplier_max),
        )
        traj_void: list[dict[str, np.ndarray]] = []
        for z0 in starts:
            rho0 = sampler._initialize_momentum(z0)
            tr = simulate_no_metropolis(
                sampler=sampler,
                model=model,
                z_start=z0,
                rho_init=rho0,
                steps=int(args.steps_void),
                eps_lf=float(args.eps_void),
                n_lf_inner=int(args.n_lf_inner_void),
                volume_power=float(args.volume_power),
                integrator=integrator,
                max_radius=float(args.max_radius),
                use_metropolis=not bool(args.no_metropolis),
            )
            traj_void.append(tr)

        core_pts = centroids.cpu().numpy()
        starts_pts = np.vstack([s.cpu().numpy() for s in starts])
        xlim, ylim = compute_plot_limits(core_pts=core_pts, starts_pts=starts_pts, traj_list=traj_void)

        out_path = Path(args.output)
        make_plot_multi_void(
            model=model,
            traj_void=traj_void,
            xlim=xlim,
            ylim=ylim,
            n_grid=int(args.grid_n),
            volume_power=float(args.volume_power),
            out_path=out_path,
            integrator=integrator,
        )

        escapes = int(sum(int(t["escaped"][0]) for t in traj_void))
        acc_rates = np.asarray([float(t["accept_rate"][0]) for t in traj_void], dtype=np.float64)

        with torch.no_grad():
            z_start = torch.as_tensor(starts_pts, dtype=torch.float32, device=device)
            z_end = torch.as_tensor(
                np.vstack([t["path"][-1] for t in traj_void]),
                dtype=torch.float32,
                device=device,
            )
            ld_start = torch.linalg.slogdet(model.G_inv(z_start)).logabsdet.cpu().numpy()
            ld_end = torch.linalg.slogdet(model.G_inv(z_end)).logabsdet.cpu().numpy()
        delta_ld = ld_end - ld_start

        print(f"Saved figure to: {out_path}")
        print(
            "Multi-void RHMC summary: "
            f"n_traj={len(traj_void)}, escaped={escapes}, "
            f"accept_rate_mean={np.nanmean(acc_rates):.3f}, "
            f"accept_rate_min={np.nanmin(acc_rates):.3f}, "
            f"accept_rate_max={np.nanmax(acc_rates):.3f}, "
            f"delta_logdet_inv_mean={np.mean(delta_ld):.3e}, "
            f"delta_logdet_inv_median={np.median(delta_ld):.3e}, "
            f"delta_logdet_inv_min={np.min(delta_ld):.3e}, "
            f"delta_logdet_inv_max={np.max(delta_ld):.3e}"
        )
        return

    start_a = select_void_start(centroids, multiplier=float(args.void_multiplier))
    start_b = select_dancer_start(model)
    target = centroids[torch.argmin(torch.linalg.norm(centroids - start_a, dim=1))]

    gen_a = torch.Generator(device=device)
    gen_b = torch.Generator(device=device)
    gen_a.manual_seed(int(args.seed) + 17)
    gen_b.manual_seed(int(args.seed) + 29)

    rho_a = tune_parachutist_momentum(
        sampler=sampler,
        model=model,
        z_start=start_a,
        target=target,
        generator=gen_a,
        trials=int(args.momentum_trials_a),
        eps_lf=float(args.eps_a),
        n_lf_inner=int(args.n_lf_inner_a),
        score_steps=int(args.momentum_score_steps_a),
        integrator=integrator,
        max_radius=float(args.max_radius),
    )
    rho_b = select_momentum_toward_target(
        sampler=sampler,
        z=start_b,
        target=centroids.mean(dim=0),
        generator=gen_b,
        trials=int(args.momentum_trials_b),
    )

    traj_a = simulate_no_metropolis(
        sampler=sampler,
        model=model,
        z_start=start_a,
        rho_init=rho_a,
        steps=int(args.steps_a),
        eps_lf=float(args.eps_a),
        n_lf_inner=int(args.n_lf_inner_a),
        volume_power=float(args.volume_power),
        integrator=integrator,
        max_radius=float(args.max_radius),
        use_metropolis=False,
    )
    traj_b = simulate_no_metropolis(
        sampler=sampler,
        model=model,
        z_start=start_b,
        rho_init=rho_b,
        steps=int(args.steps_b),
        eps_lf=float(args.eps_b),
        n_lf_inner=int(args.n_lf_inner_b),
        volume_power=float(args.volume_power),
        integrator=integrator,
        max_radius=float(args.max_radius),
        use_metropolis=False,
    )

    core_pts = centroids.cpu().numpy()
    plotted = np.vstack([core_pts, traj_a["path"], traj_b["path"], start_a.cpu().numpy(), start_b.cpu().numpy()])
    c_center = core_pts.mean(axis=0)
    core_span = np.max(np.ptp(core_pts, axis=0))
    max_plot_radius = max(2.5 * core_span, 8.0)
    keep = np.linalg.norm(plotted - c_center[None, :], axis=1) <= max_plot_radius
    pts = np.vstack([core_pts, plotted[keep]])
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    pad_x = max(0.8, 0.12 * (x_max - x_min))
    pad_y = max(0.8, 0.12 * (y_max - y_min))
    xlim = (float(x_min - pad_x), float(x_max + pad_x))
    ylim = (float(y_min - pad_y), float(y_max + pad_y))

    out_path = Path(args.output)
    make_plot(
        model=model,
        traj_a=traj_a,
        traj_b=traj_b,
        xlim=xlim,
        ylim=ylim,
        n_grid=int(args.grid_n),
        volume_power=float(args.volume_power),
        out_path=out_path,
        integrator=integrator,
    )

    def _summ(name: str, tr: dict[str, np.ndarray]) -> str:
        escaped = bool(int(tr["escaped"][0]))
        escape_step = int(tr["escape_step"][0])
        speed_end = tr["speed"][-1] if tr["speed"].size > 0 else float("nan")
        u_start = tr["potential"][0] if tr["potential"].size > 0 else float("nan")
        u_end = tr["potential"][-1] if tr["potential"].size > 0 else float("nan")
        ld_start = tr["logdet_inv"][0] if tr["logdet_inv"].size > 0 else float("nan")
        ld_end = tr["logdet_inv"][-1] if tr["logdet_inv"].size > 0 else float("nan")
        accept_rate = float(tr["accept_rate"][0]) if tr["accept_rate"].size > 0 else float("nan")
        return (
            f"{name}: speed(mean)={np.mean(tr['speed']):.3e}, speed(end)={speed_end:.3e}, "
            f"logdet_inv(start)={ld_start:.3e}, logdet_inv(end)={ld_end:.3e}, "
            f"U(start)={u_start:.3e}, U(end)={u_end:.3e}, "
            f"accept_rate={accept_rate:.3f}, escaped={escaped}, escape_step={escape_step}"
        )

    print(f"Saved figure to: {out_path}")
    print(_summ("Trajectory A", traj_a))
    print(_summ("Trajectory B", traj_b))


if __name__ == "__main__":
    main()
