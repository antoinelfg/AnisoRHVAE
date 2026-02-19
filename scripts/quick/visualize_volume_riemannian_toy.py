#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import seaborn as sns
import torch


@dataclass(frozen=True)
class ToyMetricConfig:
    manifold: str = "crescent"
    sigma: float = 2.2
    eps_floor: float = 1e-4
    volume_power: float = 0.5
    mu1: tuple[float, float] = (-2.0, -2.0)
    mu2: tuple[float, float] = (2.0, 2.0)
    crescent_center: tuple[float, float] = (0.0, 0.0)
    crescent_radius: float = 4.0
    crescent_y_scale: float = 0.80
    crescent_theta_min_deg: float = 35.0
    crescent_theta_max_deg: float = 325.0
    crescent_points: int = 28


class ToyVolumeElementRiemannianHMC:
    """
    Toy surrogate of VolumeElementRiemannianHMCSampler on R^2.

    We mirror the sampler convention used in this repo:
      - G_inv(z) = D(z) * I
      - G(z)     = (1 / D(z)) * I
      - H(z, rho) = U(z) + K(z, rho)
      - U(z) = -(v + 0.5) * log det(G_inv(z))
      - K(z, rho) = 0.5 * rho^T G_inv(z) rho

    With D(z) scalar, det(G_inv)=D^2 and K=0.5*D*||rho||^2.
    """

    def __init__(self, cfg: ToyMetricConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.sigma2 = float(cfg.sigma) ** 2
        self.manifold = str(cfg.manifold).strip().lower()
        self.mu1 = torch.tensor(cfg.mu1, dtype=torch.float32, device=device)
        self.mu2 = torch.tensor(cfg.mu2, dtype=torch.float32, device=device)
        self.eps_floor = float(cfg.eps_floor)
        self.volume_power = float(cfg.volume_power)
        self.anchors = self._build_manifold_anchors()

    def _build_manifold_anchors(self) -> torch.Tensor:
        if self.manifold == "two_blobs":
            return torch.stack([self.mu1, self.mu2], dim=0)
        if self.manifold != "crescent":
            raise ValueError(f"Unsupported manifold type: {self.manifold}")

        n_pts = max(8, int(self.cfg.crescent_points))
        t0 = np.deg2rad(float(self.cfg.crescent_theta_min_deg))
        t1 = np.deg2rad(float(self.cfg.crescent_theta_max_deg))
        theta = torch.linspace(float(t0), float(t1), n_pts, device=self.device)
        c = torch.tensor(self.cfg.crescent_center, dtype=torch.float32, device=self.device)
        r = float(self.cfg.crescent_radius)
        y_scale = float(self.cfg.crescent_y_scale)
        x = c[0] + r * torch.cos(theta)
        y = c[1] + y_scale * r * torch.sin(theta)
        return torch.stack([x, y], dim=1)

    def manifold_anchors(self) -> torch.Tensor:
        return self.anchors

    def density(self, z: torch.Tensor) -> torch.Tensor:
        # D(z) = sum_k exp(-||z-a_k||^2 / sigma^2) + floor
        single = z.ndim == 1
        z_batch = z.unsqueeze(0) if single else z
        anchors = self.anchors
        diff = z_batch.unsqueeze(1) - anchors.unsqueeze(0)
        sq = torch.sum(diff * diff, dim=-1)
        g = torch.exp(-sq / self.sigma2)
        out = torch.sum(g, dim=1) + self.eps_floor
        if single:
            return out[0]
        return out

    def grad_density(self, z: torch.Tensor) -> torch.Tensor:
        single = z.ndim == 1
        z_batch = z.unsqueeze(0) if single else z
        anchors = self.anchors
        diff = z_batch.unsqueeze(1) - anchors.unsqueeze(0)  # [B,K,2]
        sq = torch.sum(diff * diff, dim=-1)  # [B,K]
        g = torch.exp(-sq / self.sigma2)  # [B,K]
        c = -2.0 / self.sigma2
        out = c * torch.sum(g.unsqueeze(-1) * diff, dim=1)
        if single:
            return out[0]
        return out

    def potential(self, z: torch.Tensor) -> torch.Tensor:
        # log det(G_inv) = 2 * log D
        eff = self.volume_power + 0.5
        d = torch.clamp(self.density(z), min=self.eps_floor)
        return -(2.0 * eff) * torch.log(d)

    def kinetic(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        d = self.density(z)
        return 0.5 * d * torch.sum(rho * rho, dim=-1)

    def hamiltonian(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        return self.potential(z) + self.kinetic(z, rho)

    def grad_hamiltonian_z(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        d = torch.clamp(self.density(z), min=self.eps_floor)
        grad_d = self.grad_density(z)
        eff = self.volume_power + 0.5
        rho2 = torch.sum(rho * rho, dim=-1, keepdim=True)
        grad_u = -(2.0 * eff) * grad_d / d.unsqueeze(-1)
        grad_k = 0.5 * rho2 * grad_d
        return grad_u + grad_k

    def grad_hamiltonian_rho(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        d = self.density(z).unsqueeze(-1)
        return d * rho

    def sample_momentum(self, z: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        # rho ~ N(0, G(z)) with G=(1/D)I => std = 1/sqrt(D).
        d = torch.clamp(self.density(z), min=self.eps_floor).unsqueeze(-1)
        std = torch.rsqrt(d)
        noise = torch.randn(z.shape, generator=generator, device=self.device, dtype=z.dtype)
        return std * noise

    def leapfrog_explicit(self, z: torch.Tensor, rho: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        # Explicit kick-drift-kick with exact gradients of H.
        grad_z = self.grad_hamiltonian_z(z, rho)
        rho_half = rho - 0.5 * eps * grad_z
        z_new = z + eps * self.grad_hamiltonian_rho(z, rho_half)
        grad_z_new = self.grad_hamiltonian_z(z_new, rho_half)
        rho_new = rho_half - 0.5 * eps * grad_z_new
        return z_new, rho_new


def _distance_to_manifold(sampler: ToyVolumeElementRiemannianHMC, z: torch.Tensor) -> float:
    anchors = sampler.manifold_anchors()
    d = torch.linalg.norm(anchors - z.unsqueeze(0), dim=1)
    return float(torch.min(d).item())


def simulate_trajectory(
    sampler: ToyVolumeElementRiemannianHMC,
    z_start: torch.Tensor,
    n_steps: int,
    eps_lf: float,
    generator: torch.Generator,
    rho_init: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    z = z_start.clone()
    if rho_init is not None:
        rho = rho_init.clone()
    else:
        rho = sampler.sample_momentum(z, generator=generator)

    positions = [z.detach().cpu().numpy()]
    speeds = [torch.linalg.norm(sampler.grad_hamiltonian_rho(z, rho)).item()]
    h_vals = [sampler.hamiltonian(z.unsqueeze(0), rho.unsqueeze(0)).item()]
    d_vals = [sampler.density(z.unsqueeze(0)).item()]

    for _ in range(n_steps):
        z, rho = sampler.leapfrog_explicit(z, rho, eps_lf)
        positions.append(z.detach().cpu().numpy())
        speeds.append(torch.linalg.norm(sampler.grad_hamiltonian_rho(z, rho)).item())
        h_vals.append(sampler.hamiltonian(z.unsqueeze(0), rho.unsqueeze(0)).item())
        d_vals.append(sampler.density(z.unsqueeze(0)).item())

    return {
        "positions": np.asarray(positions, dtype=np.float64),
        "speeds": np.asarray(speeds, dtype=np.float64),
        "hamiltonian": np.asarray(h_vals, dtype=np.float64),
        "density": np.asarray(d_vals, dtype=np.float64),
    }


def _rollout_score(
    sampler: ToyVolumeElementRiemannianHMC,
    z_start: torch.Tensor,
    rho_init: torch.Tensor,
    n_steps: int,
    eps_lf: float,
) -> dict[str, float]:
    z = z_start.clone()
    rho = rho_init.clone()
    d0 = float(sampler.density(z.unsqueeze(0)).item())
    dist0 = _distance_to_manifold(sampler, z)
    speeds: list[float] = []

    for _ in range(n_steps):
        vel = sampler.grad_hamiltonian_rho(z, rho)
        speeds.append(float(torch.linalg.norm(vel).item()))
        z, rho = sampler.leapfrog_explicit(z, rho, eps_lf)

    d1 = float(sampler.density(z.unsqueeze(0)).item())
    dist1 = _distance_to_manifold(sampler, z)
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    max_speed = float(np.max(speeds)) if speeds else 0.0
    return {
        "d0": d0,
        "d1": d1,
        "dist0": dist0,
        "dist1": dist1,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
    }


def select_parachutist_momentum(
    sampler: ToyVolumeElementRiemannianHMC,
    z_start: torch.Tensor,
    generator: torch.Generator,
    n_steps: int,
    eps_lf: float,
    trials: int = 640,
) -> torch.Tensor:
    best_rho: torch.Tensor | None = None
    best_score = -float("inf")
    anchors = sampler.manifold_anchors()
    nearest_idx = torch.argmin(torch.linalg.norm(anchors - z_start.unsqueeze(0), dim=1))
    target = anchors[int(nearest_idx.item())]
    local_steps = min(int(n_steps), 70)

    for _ in range(trials):
        rho = sampler.sample_momentum(z_start, generator=generator)
        vel = sampler.grad_hamiltonian_rho(z_start, rho)
        denom = (torch.linalg.norm(vel) * torch.linalg.norm(target - z_start)).item()
        cos_to_center = 0.0
        if denom > 1e-10:
            cos_to_center = float(torch.dot(vel, target - z_start).item() / denom)
        stats = _rollout_score(sampler, z_start, rho, n_steps=local_steps, eps_lf=eps_lf)
        score = (
            2.0 * (stats["dist0"] - stats["dist1"])
            + 42.0 * (stats["d1"] - stats["d0"])
            + 0.65 * cos_to_center
            + 0.08 * stats["mean_speed"]
        )
        if score > best_score:
            best_score = score
            best_rho = rho

    if best_rho is None:
        return sampler.sample_momentum(z_start, generator=generator)
    return best_rho


def select_dancer_momentum(
    sampler: ToyVolumeElementRiemannianHMC,
    z_start: torch.Tensor,
    generator: torch.Generator,
    n_steps: int,
    eps_lf: float,
    trials: int = 256,
) -> torch.Tensor:
    best_rho: torch.Tensor | None = None
    best_score = -float("inf")
    z0_np = z_start.detach().cpu().numpy()
    anchors_np = sampler.manifold_anchors().detach().cpu().numpy()
    local_steps = min(int(n_steps), 90)

    for _ in range(trials):
        rho = sampler.sample_momentum(z_start, generator=generator)
        traj = simulate_trajectory(
            sampler=sampler,
            z_start=z_start,
            n_steps=local_steps,
            eps_lf=eps_lf,
            generator=generator,
            rho_init=rho,
        )
        pos = traj["positions"]
        # Distance to nearest anchor at each step.
        dist_matrix = np.linalg.norm(pos[:, None, :] - anchors_np[None, :, :], axis=2)
        dist_to_manifold = np.min(dist_matrix, axis=1)
        mean_speed = float(np.mean(traj["speeds"]))
        speed_var = float(np.std(traj["speeds"]))
        spread = float(np.mean(np.linalg.norm(pos - z0_np.reshape(1, 2), axis=1)))
        mean_density = float(np.mean(traj["density"]))
        end_dist = float(dist_to_manifold[-1])
        drift_penalty = max(0.0, float(np.max(dist_to_manifold)) - 1.8)
        score = (
            mean_speed
            + 0.15 * speed_var
            + 0.15 * spread
            + 0.9 * mean_density
            - 1.6 * drift_penalty
            - 0.7 * end_dist
        )
        if score > best_score:
            best_score = score
            best_rho = rho

    if best_rho is None:
        return sampler.sample_momentum(z_start, generator=generator)
    return best_rho


def build_potential_grid(
    sampler: ToyVolumeElementRiemannianHMC,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = torch.linspace(xlim[0], xlim[1], n_grid, device=sampler.device)
    y = torch.linspace(ylim[0], ylim[1], n_grid, device=sampler.device)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    with torch.no_grad():
        u = sampler.potential(points).reshape(n_grid, n_grid).cpu().numpy()
    x_np = xx.cpu().numpy()
    y_np = yy.cpu().numpy()
    return x_np, y_np, u, points.cpu().numpy()


def make_figure(
    sampler: ToyVolumeElementRiemannianHMC,
    traj_a: dict[str, np.ndarray],
    traj_b: dict[str, np.ndarray],
    out_path: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    n_grid: int,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    xx, yy, u_grid, _ = build_potential_grid(sampler, xlim=xlim, ylim=ylim, n_grid=n_grid)

    # Positive log-display value; inverted so low-U manifold appears dark with viridis_r.
    u_shift = u_grid - float(np.min(u_grid)) + 1e-3
    u_plot = float(np.max(u_shift)) - u_shift + 1e-3
    levels = np.geomspace(float(np.min(u_plot)), float(np.max(u_plot)), 40)

    fig, ax = plt.subplots(figsize=(12, 10))
    cf = ax.contourf(
        xx,
        yy,
        u_plot,
        levels=levels,
        cmap="viridis_r",
        norm=LogNorm(vmin=float(np.min(u_plot)), vmax=float(np.max(u_plot))),
        alpha=0.9,
    )
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("Log-scaled potential field (contrast-inverted)")

    anchors = sampler.manifold_anchors().detach().cpu().numpy()
    ax.scatter(
        anchors[:, 0],
        anchors[:, 1],
        marker="o",
        s=22,
        c="white",
        edgecolors="black",
        linewidths=0.6,
        alpha=0.85,
        zorder=7,
        label="Ancres manifold",
    )

    pa = traj_a["positions"]
    pb = traj_b["positions"]

    # Trajectory A: parachutiste (void -> manifold)
    ax.plot(pa[:, 0], pa[:, 1], color="#d62728", linewidth=2.4, label="Trajectoire A: Parachutiste", zorder=8)
    ax.scatter(pa[0, 0], pa[0, 1], c="#d62728", marker="X", s=190, edgecolors="black", linewidths=1.0, zorder=9)
    mark_a = np.arange(0, pa.shape[0], 4)
    ax.scatter(pa[mark_a, 0], pa[mark_a, 1], c=np.linspace(0.2, 1.0, len(mark_a)), cmap="Reds", s=28, zorder=9)
    qa = np.arange(0, pa.shape[0] - 1, 6)
    ax.quiver(
        pa[qa, 0],
        pa[qa, 1],
        pa[qa + 1, 0] - pa[qa, 0],
        pa[qa + 1, 1] - pa[qa, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#d62728",
        width=0.0032,
        alpha=0.55,
        zorder=8,
    )

    # Trajectory B: danseur (on manifold)
    ax.plot(pb[:, 0], pb[:, 1], color="#00bcd4", linewidth=2.2, label="Trajectoire B: Danseur", zorder=8)
    ax.scatter(pb[0, 0], pb[0, 1], c="#00bcd4", marker="X", s=190, edgecolors="black", linewidths=1.0, zorder=9)
    mark_b = np.arange(0, pb.shape[0], 4)
    ax.scatter(pb[mark_b, 0], pb[mark_b, 1], c=np.linspace(0.2, 1.0, len(mark_b)), cmap="winter", s=28, zorder=9)
    qb = np.arange(0, pb.shape[0] - 1, 6)
    ax.quiver(
        pb[qb, 0],
        pb[qb, 1],
        pb[qb + 1, 0] - pb[qb, 0],
        pb[qb + 1, 1] - pb[qb, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#00bcd4",
        width=0.0032,
        alpha=0.65,
        zorder=8,
    )

    ax.annotate(
        "VOID\n(Masse elevee, pente raide)",
        xy=(-9.2, 8.2),
        xytext=(-10.6, 10.0),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.4},
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "black", "alpha": 0.9},
    )
    ax.annotate(
        "MANIFOLD EN CROISSANT\n(Masse faible, vallee)",
        xy=(-2.4, 2.8),
        xytext=(1.6, 4.5),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.4},
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "black", "alpha": 0.9},
    )

    ax.set_title("Toy VolumeElementRiemannianHMC Dynamics on a Synthetic 2D Manifold")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", frameon=True, framealpha=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize VolumeElementRiemannianHMC behavior on a synthetic 2D metric."
    )
    parser.add_argument("--manifold", type=str, default="crescent", choices=["crescent", "two_blobs"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--eps_floor", type=float, default=1e-4)
    parser.add_argument("--volume_power", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--eps_a", type=float, default=1.4, help="Leapfrog step for trajectory A.")
    parser.add_argument("--eps_b", type=float, default=0.08, help="Leapfrog step for trajectory B.")
    parser.add_argument("--search_trials_a", type=int, default=800)
    parser.add_argument("--search_trials_b", type=int, default=256)
    parser.add_argument("--start_void_x", type=float, default=-11.0)
    parser.add_argument("--start_void_y", type=float, default=11.0)
    parser.add_argument("--grid_n", type=int, default=350)
    parser.add_argument("--output", type=str, default="results/toy_volume_riemannian/toy_volume_riemannian.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device("cpu")
    cfg = ToyMetricConfig(
        manifold=str(args.manifold),
        sigma=float(args.sigma),
        eps_floor=float(args.eps_floor),
        volume_power=float(args.volume_power),
    )
    sampler = ToyVolumeElementRiemannianHMC(cfg, device=device)

    gen_a = torch.Generator(device=device)
    gen_b = torch.Generator(device=device)
    gen_a.manual_seed(int(args.seed) + 11)
    gen_b.manual_seed(int(args.seed) + 29)

    z_start_a = torch.tensor([float(args.start_void_x), float(args.start_void_y)], dtype=torch.float32, device=device)
    anchors = sampler.manifold_anchors()
    z_start_b = anchors[len(anchors) // 3].clone().detach()

    rho_a = select_parachutist_momentum(
        sampler=sampler,
        z_start=z_start_a,
        generator=gen_a,
        n_steps=int(args.steps),
        eps_lf=float(args.eps_a),
        trials=int(args.search_trials_a),
    )
    rho_b = select_dancer_momentum(
        sampler=sampler,
        z_start=z_start_b,
        generator=gen_b,
        n_steps=int(args.steps),
        eps_lf=float(args.eps_b),
        trials=int(args.search_trials_b),
    )

    traj_a = simulate_trajectory(
        sampler=sampler,
        z_start=z_start_a,
        n_steps=int(args.steps),
        eps_lf=float(args.eps_a),
        generator=gen_a,
        rho_init=rho_a,
    )
    traj_b = simulate_trajectory(
        sampler=sampler,
        z_start=z_start_b,
        n_steps=int(args.steps),
        eps_lf=float(args.eps_b),
        generator=gen_b,
        rho_init=rho_b,
    )

    out_path = Path(args.output)
    make_figure(
        sampler=sampler,
        traj_a=traj_a,
        traj_b=traj_b,
        out_path=out_path,
        xlim=(-12.5, 12.5),
        ylim=(-12.5, 12.5),
        n_grid=int(args.grid_n),
    )

    def _summary(name: str, tr: dict[str, np.ndarray]) -> str:
        h0 = tr["hamiltonian"][0]
        h1 = tr["hamiltonian"][-1]
        d0 = tr["density"][0]
        d1 = tr["density"][-1]
        v0 = tr["speeds"][0]
        v1 = tr["speeds"][-1]
        zf = tr["positions"][-1]
        return (
            f"{name}: D(start)={d0:.3e}, D(end)={d1:.3e}, "
            f"speed(start)={v0:.3e}, speed(end)={v1:.3e}, "
            f"H drift={h1 - h0:+.3e}, end=({zf[0]:.3f}, {zf[1]:.3f})"
        )

    print(f"Saved figure to: {out_path}")
    print(_summary("Trajectory A", traj_a))
    print(_summary("Trajectory B", traj_b))


if __name__ == "__main__":
    main()
