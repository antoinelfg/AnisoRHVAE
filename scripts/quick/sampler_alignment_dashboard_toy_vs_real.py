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

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from src.models.rhvae_geometry import GeometryRHVAE, GeometryRHVAEConfig
from src.models.samplers.hmc_sampler import VolumeElementRiemannianHMCSampler
from src.utils.metric_helpers import load_metric_bundle
from src.utils.rescue_metrics import model_r0


class ToyCrescentMetricModel(torch.nn.Module):
    """Toy 2D metric model exposing the RHVAE metric API: G_inv / G / centroids."""

    def __init__(
        self,
        sigma: float = 1.2,
        eps_floor: float = 1e-4,
        crescent_radius: float = 4.0,
        crescent_y_scale: float = 0.8,
        theta_min_deg: float = 35.0,
        theta_max_deg: float = 325.0,
        n_centroids: int = 28,
    ) -> None:
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(1))
        self.latent_dim = 2
        self.sigma = float(sigma)
        self.sigma2 = float(sigma) ** 2
        self.eps_floor = float(eps_floor)

        t0 = np.deg2rad(float(theta_min_deg))
        t1 = np.deg2rad(float(theta_max_deg))
        theta = torch.linspace(float(t0), float(t1), max(8, int(n_centroids)))
        x = float(crescent_radius) * torch.cos(theta)
        y = float(crescent_y_scale) * float(crescent_radius) * torch.sin(theta)
        self.register_buffer("centroids_tens", torch.stack([x, y], dim=1))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _density(self, z: torch.Tensor) -> torch.Tensor:
        # D(z) = sum_k exp(-||z-c_k||^2/sigma^2) + eps
        diff = z.unsqueeze(1) - self.centroids_tens.unsqueeze(0)
        sq = torch.sum(diff * diff, dim=-1)
        return torch.sum(torch.exp(-sq / self.sigma2), dim=1) + self.eps_floor

    def G_inv(self, z: torch.Tensor) -> torch.Tensor:
        d = torch.clamp(self._density(z), min=self.eps_floor)
        eye = torch.eye(2, device=z.device, dtype=z.dtype).unsqueeze(0)
        return d.view(-1, 1, 1) * eye

    def G(self, z: torch.Tensor) -> torch.Tensor:
        d = torch.clamp(self._density(z), min=self.eps_floor)
        eye = torch.eye(2, device=z.device, dtype=z.dtype).unsqueeze(0)
        return (1.0 / d).view(-1, 1, 1) * eye


class DirectionalBiasMetricWrapper(torch.nn.Module):
    """Optional SPD metric wrapper adding a radial precision boost in void.

    G_inv_biased(z) = G_inv_base(z) + boost * w(dist(z,M)) * s(z) * u(z)u(z)^T
    where u(z) points to nearest centroid and s(z) is local trace scale.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        boost: float,
        r0: float,
        sharpness: float = 6.0,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.boost = float(boost)
        self.r0 = float(r0)
        self.sharpness = float(sharpness)
        self.jitter = float(jitter)

    @property
    def latent_dim(self) -> int:
        return int(self.base_model.latent_dim)

    @property
    def centroids_tens(self) -> torch.Tensor:
        return self.base_model.centroids_tens

    @property
    def device(self) -> torch.device:
        if hasattr(self.base_model, "device"):
            return self.base_model.device
        return next(self.base_model.parameters()).device

    def G_inv(self, z: torch.Tensor) -> torch.Tensor:
        g0 = self.base_model.G_inv(z)
        if self.boost <= 0.0:
            return g0

        c = self.centroids_tens.to(z.device)
        d = torch.cdist(z, c)
        idx = torch.argmin(d, dim=1)
        near = c[idx]
        vec = near - z
        dist = torch.linalg.norm(vec, dim=1)
        u = vec / torch.clamp(dist.unsqueeze(-1), min=1e-8)
        uu_t = torch.einsum("bi,bj->bij", u, u)

        # Weight rises outside manifold shell.
        w = torch.sigmoid((dist - float(self.r0)) * float(self.sharpness))
        local_scale = torch.einsum("bii->b", g0) / float(g0.shape[-1])
        add = (float(self.boost) * w * local_scale).view(-1, 1, 1) * uu_t
        g = g0 + add

        g = 0.5 * (g + g.transpose(-1, -2))
        if self.jitter > 0.0:
            eye = torch.eye(g.shape[-1], device=g.device, dtype=g.dtype).unsqueeze(0)
            g = g + float(self.jitter) * eye
        return g

    def G(self, z: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(self.G_inv(z))


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


def choose_dims_from_variance(centroids: torch.Tensor) -> tuple[int, int]:
    if centroids.shape[1] <= 2:
        return 0, 1
    var = torch.var(centroids, dim=0)
    idx = torch.topk(var, k=2).indices.tolist()
    d0, d1 = int(idx[0]), int(idx[1])
    if d0 == d1:
        d1 = (d0 + 1) % int(centroids.shape[1])
    return d0, d1


def generate_void_starts(
    centroids: torch.Tensor,
    dims: tuple[int, int],
    n_starts: int,
    mult_min: float,
    mult_max: float,
) -> torch.Tensor:
    d0, d1 = dims
    c2 = centroids[:, [d0, d1]]
    center2 = c2.mean(dim=0)
    shifted = c2 - center2
    cov = (shifted.T @ shifted) / max(1, shifted.shape[0] - 1)
    _, evecs = torch.linalg.eigh(cov)
    e1 = evecs[:, -1]
    e2 = evecs[:, -2] if evecs.shape[1] > 1 else torch.tensor([0.0, 1.0], device=c2.device)
    r_max = torch.linalg.norm(shifted, dim=1).max()

    z_center = centroids.mean(dim=0)
    out = []
    angles = torch.linspace(0.0, 2.0 * torch.pi, int(n_starts) + 1, device=centroids.device)[:-1]
    for a in angles:
        direction2 = torch.cos(a) * e1 + torch.sin(a) * e2
        direction2 = direction2 / (torch.linalg.norm(direction2) + 1e-9)
        phase = 0.5 * (1.0 + torch.sin(2.5 * a))
        mul = float(mult_min) + (float(mult_max) - float(mult_min)) * float(phase)
        z = z_center.clone()
        z[d0] = center2[0] + mul * r_max * direction2[0]
        z[d1] = center2[1] + mul * r_max * direction2[1]
        out.append(z)
    return torch.stack(out, dim=0)


def infer_r0(model: Any, centroids: torch.Tensor) -> float:
    if hasattr(model, "_compute_r0"):
        return float(model_r0(model, device=centroids.device))
    # Toy fallback: radial shell where gaussian anchor weight starts to fade.
    sigma = float(getattr(model, "sigma", 1.2))
    tau = 0.20
    return float(sigma * np.sqrt(-np.log(tau)))


def potential_logdet_inv(model: Any, z: torch.Tensor, volume_power: float) -> torch.Tensor:
    eff = float(volume_power) + 0.5
    g_inv = model.G_inv(z)
    logdet_inv = torch.linalg.slogdet(g_inv).logabsdet
    return -(eff * logdet_inv)


def evaluate_potential_grid(
    model: Any,
    centroids: torch.Tensor,
    dims: tuple[int, int],
    volume_power: float,
    grid_n: int,
    bounds_scale: float,
    min_bounds: float,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    d0, d1 = dims
    c2 = centroids[:, [d0, d1]].detach().cpu().numpy()
    center = c2.mean(axis=0)
    radii = np.linalg.norm(c2 - center[None, :], axis=1)
    b = max(float(np.percentile(radii, 95) * bounds_scale), float(min_bounds))
    x0, x1 = float(center[0] - b), float(center[0] + b)
    y0, y1 = float(center[1] - b), float(center[1] + b)

    x = np.linspace(x0, x1, int(grid_n))
    y = np.linspace(y0, y1, int(grid_n))
    xx, yy = np.meshgrid(x, y, indexing="xy")
    flat = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)

    z = centroids.mean(dim=0, keepdim=True).repeat(flat.shape[0], 1)
    z[:, d0] = torch.from_numpy(flat[:, 0]).to(z.device)
    z[:, d1] = torch.from_numpy(flat[:, 1]).to(z.device)

    vals = []
    with torch.no_grad():
        for i in range(0, z.shape[0], int(batch_size)):
            chunk = z[i : i + int(batch_size)]
            u = potential_logdet_inv(model, chunk, volume_power=volume_power)
            vals.append(u.detach().cpu().numpy())
    u_grid = np.concatenate(vals, axis=0).reshape(int(grid_n), int(grid_n))
    return xx, yy, u_grid, (x0, x1, y0, y1)


def run_rhmc_ensemble(
    model: Any,
    dims: tuple[int, int],
    starts: torch.Tensor,
    volume_power: float,
    steps: int,
    n_lf_inner: int,
    eps: float,
    fp_steps: int,
    fp_damping: float,
    r0: float,
    max_radius: float,
    seed: int,
    momentum_persist: float,
    use_dual_metric: bool,
) -> dict[str, Any]:
    sampler = VolumeElementRiemannianHMCSampler(
        model,
        mcmc_steps_nbr=1,
        n_lf=1,
        eps_lf=float(eps),
        exact=True,
        fp_steps=int(fp_steps),
        fp_damping=float(fp_damping),
        volume_power=float(volume_power),
        use_dual_metric=bool(use_dual_metric),
    )
    sampler.momentum_persist = float(momentum_persist)

    d0, d1 = dims
    bsz = int(starts.shape[0])
    z = starts.clone().detach()
    g = torch.Generator(device=z.device)
    g.manual_seed(int(seed))

    path = np.zeros((int(steps) + 1, bsz, 2), dtype=np.float64)
    path[0, :, 0] = z[:, d0].detach().cpu().numpy()
    path[0, :, 1] = z[:, d1].detach().cpu().numpy()

    accept_hist = np.zeros((int(steps), bsz), dtype=np.float64)
    min_dist_hist = np.zeros((int(steps) + 1, bsz), dtype=np.float64)
    cos_prop_hist = np.full((int(steps), bsz), np.nan, dtype=np.float64)
    cos_acc_hist = np.full((int(steps), bsz), np.nan, dtype=np.float64)
    moved_hist = np.zeros((int(steps), bsz), dtype=np.float64)
    outside_hist = np.zeros((int(steps), bsz), dtype=np.float64)

    with torch.no_grad():
        d0_init = torch.cdist(z, model.centroids_tens).min(dim=1).values
    min_dist_hist[0] = d0_init.detach().cpu().numpy()

    escaped = np.zeros(bsz, dtype=bool)
    hit_steps = np.full(bsz, -1, dtype=int)
    first_prop_delta = np.full((bsz, 2), np.nan, dtype=np.float64)
    first_principal_dir = np.full((bsz, 2), np.nan, dtype=np.float64)
    first_cos_principal_abs = np.full(bsz, np.nan, dtype=np.float64)
    first_cos_to_manifold = np.full(bsz, np.nan, dtype=np.float64)

    for t in range(int(steps)):
        z_curr = z.detach().clone().requires_grad_(True)
        rho = sampler._initialize_momentum(z_curr).detach()

        with torch.no_grad():
            h0 = sampler._compute_hamiltonian(z_curr, rho)

        z_prop, rho_prop = z_curr, rho
        for _ in range(int(n_lf_inner)):
            z_prop, rho_prop = sampler._generalized_leapfrog_step(z_prop, rho_prop, float(eps))
            z_prop = z_prop.detach().requires_grad_(True)
            rho_prop = rho_prop.detach()

        with torch.no_grad():
            h1 = sampler._compute_hamiltonian(z_prop, rho_prop)
            log_alpha = h0 - h1
            alpha = torch.exp(torch.clamp(log_alpha, max=0.0))
            u = torch.rand(alpha.shape, generator=g, device=z.device)
            accept = (u < alpha).float().view(-1, 1)

            # Hard safety in far runaway.
            rad = torch.linalg.norm(z_prop, dim=1)
            reject_far = (rad > float(max_radius)).view(-1, 1)
            accept = torch.where(reject_far, torch.zeros_like(accept), accept)

            z_next = (accept * z_prop.detach() + (1.0 - accept) * z_curr.detach()).detach()

            d = torch.cdist(z_curr.detach(), model.centroids_tens)
            near_idx = torch.argmin(d, dim=1)
            near = model.centroids_tens[near_idx]
            to_mani = near - z_curr.detach()
            to_norm = torch.linalg.norm(to_mani, dim=1)
            delta_prop = z_prop.detach() - z_curr.detach()
            delta_acc = z_next - z_curr.detach()
            prop_norm = torch.linalg.norm(delta_prop, dim=1)
            acc_norm = torch.linalg.norm(delta_acc, dim=1)

            denom_prop = torch.clamp(prop_norm * to_norm, min=1e-12)
            denom_acc = torch.clamp(acc_norm * to_norm, min=1e-12)
            cos_prop = torch.sum(delta_prop * to_mani, dim=1) / denom_prop
            cos_acc = torch.sum(delta_acc * to_mani, dim=1) / denom_acc

            min_dist = d[torch.arange(bsz, device=z.device), near_idx]

            if t == 0:
                idx2 = torch.tensor([d0, d1], device=z.device)
                g2 = model.G_inv(z_curr.detach()).index_select(1, idx2).index_select(2, idx2)
                _, evecs = torch.linalg.eigh(g2)
                principal = evecs[:, :, 1]
                dprop2 = delta_prop[:, [d0, d1]]
                mani2 = to_mani[:, [d0, d1]]
                dprop_norm2 = torch.linalg.norm(dprop2, dim=1)
                mani_norm2 = torch.linalg.norm(mani2, dim=1)
                prin_norm2 = torch.linalg.norm(principal, dim=1)
                denom_pm = torch.clamp(dprop_norm2 * prin_norm2, min=1e-12)
                denom_mm = torch.clamp(dprop_norm2 * mani_norm2, min=1e-12)
                cos_pm = torch.sum(dprop2 * principal, dim=1) / denom_pm
                cos_mm = torch.sum(dprop2 * mani2, dim=1) / denom_mm

                first_prop_delta = dprop2.cpu().numpy()
                first_principal_dir = principal.cpu().numpy()
                first_cos_principal_abs = torch.abs(cos_pm).cpu().numpy()
                first_cos_to_manifold = cos_mm.cpu().numpy()

        z = z_next
        path[t + 1, :, 0] = z[:, d0].detach().cpu().numpy()
        path[t + 1, :, 1] = z[:, d1].detach().cpu().numpy()
        accept_hist[t] = accept.view(-1).cpu().numpy()
        min_dist_hist[t + 1] = min_dist.detach().cpu().numpy()

        outside = (min_dist > float(r0)).cpu().numpy()
        outside_hist[t] = outside.astype(np.float64)

        cp = cos_prop.detach().cpu().numpy()
        ca = cos_acc.detach().cpu().numpy()
        cp[~outside] = np.nan
        ca[~outside] = np.nan
        moved = (acc_norm > 1e-10).cpu().numpy()
        ca[~moved] = np.nan
        cos_prop_hist[t] = cp
        cos_acc_hist[t] = ca
        moved_hist[t] = moved.astype(np.float64)

        escaped = escaped | (np.linalg.norm(path[t + 1], axis=1) > float(max_radius))
        new_hits = (hit_steps < 0) & (min_dist_hist[t + 1] <= float(r0))
        hit_steps[new_hits] = t + 1

    acceptance_rate = float(np.mean(accept_hist))
    rescue_rate = float(np.mean(hit_steps >= 0))
    median_hit = float(np.median(hit_steps[hit_steps >= 0])) if np.any(hit_steps >= 0) else float("nan")
    proposal_align = float(np.nanmean(cos_prop_hist))
    accepted_align = float(np.nanmean(cos_acc_hist))
    prop_opp_rate = float(np.nanmean(cos_prop_hist < 0.0))
    acc_opp_rate = float(np.nanmean(cos_acc_hist < 0.0))
    first_align_principal_abs = float(np.nanmean(first_cos_principal_abs))
    first_align_to_manifold = float(np.nanmean(first_cos_to_manifold))

    with torch.no_grad():
        z_start = starts
        z_end = z
        ld_start = torch.linalg.slogdet(model.G_inv(z_start)).logabsdet.cpu().numpy()
        ld_end = torch.linalg.slogdet(model.G_inv(z_end)).logabsdet.cpu().numpy()
    delta_ld = ld_end - ld_start

    return {
        "path": path,
        "accept_hist": accept_hist,
        "min_dist_hist": min_dist_hist,
        "cos_prop_hist": cos_prop_hist,
        "cos_acc_hist": cos_acc_hist,
        "outside_hist": outside_hist,
        "moved_hist": moved_hist,
        "hit_steps": hit_steps,
        "summary": {
            "acceptance_rate": acceptance_rate,
            "rescue_rate": rescue_rate,
            "median_hit_step": median_hit,
            "proposal_alignment_outside_mean": proposal_align,
            "accepted_alignment_outside_mean": accepted_align,
            "proposal_opposition_rate_outside": prop_opp_rate,
            "accepted_opposition_rate_outside": acc_opp_rate,
            "first_step_abs_alignment_principal_mean": first_align_principal_abs,
            "first_step_alignment_to_manifold_mean": first_align_to_manifold,
            "delta_logdet_inv_mean": float(np.mean(delta_ld)),
            "delta_logdet_inv_median": float(np.median(delta_ld)),
            "delta_logdet_inv_min": float(np.min(delta_ld)),
            "delta_logdet_inv_max": float(np.max(delta_ld)),
            "escaped_count": int(np.sum(escaped)),
            "n_chains": int(bsz),
            "steps": int(steps),
            "n_lf_inner": int(n_lf_inner),
            "eps": float(eps),
            "r0": float(r0),
        },
        "first_prop_delta": first_prop_delta,
        "first_principal_dir": first_principal_dir,
        "first_cos_principal_abs": first_cos_principal_abs,
        "first_cos_to_manifold": first_cos_to_manifold,
    }


def _safe(v: float) -> str:
    return "nan" if not np.isfinite(v) else f"{v:.4g}"


def plot_column(
    ax_map: Any,
    ax_hist: Any,
    ax_ts: Any,
    title: str,
    xx: np.ndarray,
    yy: np.ndarray,
    u: np.ndarray,
    centroids_2d: np.ndarray,
    result: dict[str, Any],
    r0: float,
) -> None:
    path = result["path"]  # [T+1,B,2]
    summary = result["summary"]
    n_steps, bsz = path.shape[0] - 1, path.shape[1]

    # Trajectory map
    hm = ax_map.contourf(xx, yy, u, levels=38, cmap="viridis")
    plt.colorbar(hm, ax=ax_map, fraction=0.045, pad=0.02, label="U(z)")
    ax_map.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c="white",
        edgecolors="black",
        linewidths=0.45,
        s=15,
        alpha=0.9,
        zorder=7,
    )
    cmap = plt.get_cmap("turbo")
    for i in range(bsz):
        p = path[:, i, :]
        col = cmap(i / max(1, bsz - 1))
        ax_map.plot(p[:, 0], p[:, 1], color=col, alpha=0.90, linewidth=1.5, zorder=8)
        ax_map.scatter(p[0, 0], p[0, 1], color=col, marker="X", s=45, edgecolors="black", linewidths=0.5, zorder=9)

    # Hypothesis overlay:
    # - orange arrow = first proposal step direction
    # - white line = local principal mobility axis (signless)
    starts = path[0]
    dprop = np.asarray(result.get("first_prop_delta"), dtype=float)
    pdir = np.asarray(result.get("first_principal_dir"), dtype=float)
    if dprop.shape == starts.shape and pdir.shape == starts.shape:
        step_norm = np.linalg.norm(dprop, axis=1, keepdims=True)
        step_dir = dprop / np.clip(step_norm, 1e-12, None)
        vlen = max(float(np.nanmedian(step_norm)) * 5.0, 0.9)
        ax_map.quiver(
            starts[:, 0],
            starts[:, 1],
            step_dir[:, 0] * vlen,
            step_dir[:, 1] * vlen,
            color="#ff7f0e",
            alpha=0.92,
            width=0.0032,
            scale=1.0,
            scale_units="xy",
            zorder=10,
        )
        pnorm = np.linalg.norm(pdir, axis=1, keepdims=True)
        pdir_n = pdir / np.clip(pnorm, 1e-12, None)
        half = 0.6 * vlen
        for i in range(starts.shape[0]):
            x0 = starts[i, 0] - half * pdir_n[i, 0]
            y0 = starts[i, 1] - half * pdir_n[i, 1]
            x1 = starts[i, 0] + half * pdir_n[i, 0]
            y1 = starts[i, 1] + half * pdir_n[i, 1]
            ax_map.plot([x0, x1], [y0, y1], color="white", linewidth=1.35, alpha=0.85, zorder=9)
    ax_map.set_title(f"{title} - RHMC Trajectories")
    ax_map.set_xlabel("z[a]")
    ax_map.set_ylabel("z[b]")
    ax_map.set_aspect("equal")

    txt = (
        f"acc={summary['acceptance_rate']:.3f}\n"
        f"rescue={summary['rescue_rate']:.3f}\n"
        f"align(prop)={_safe(summary['proposal_alignment_outside_mean'])}\n"
        f"align(acc)={_safe(summary['accepted_alignment_outside_mean'])}\n"
        f"first|cos(step,prin)|={_safe(summary['first_step_abs_alignment_principal_mean'])}\n"
        f"first cos(step,mani)={_safe(summary['first_step_alignment_to_manifold_mean'])}\n"
        f"esc={summary['escaped_count']}/{summary['n_chains']}"
    )
    ax_map.text(
        0.02,
        0.98,
        txt,
        transform=ax_map.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "black", "alpha": 0.85},
    )

    # Alignment histogram
    cp = result["cos_prop_hist"].reshape(-1)
    ca = result["cos_acc_hist"].reshape(-1)
    cp = cp[np.isfinite(cp)]
    ca = ca[np.isfinite(ca)]
    bins = np.linspace(-1.0, 1.0, 36)
    ax_hist.hist(cp, bins=bins, alpha=0.55, color="#e15759", label="proposal (outside)")
    ax_hist.hist(ca, bins=bins, alpha=0.55, color="#4e79a7", label="accepted move (outside)")
    ax_hist.axvline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax_hist.set_title(f"{title} - Alignment Distribution")
    ax_hist.set_xlabel("cosine to nearest-manifold direction")
    ax_hist.set_ylabel("count")
    ax_hist.legend(frameon=True, fontsize=9)

    # Time-series
    min_d = result["min_dist_hist"]  # [T+1,B]
    md_mean = np.mean(min_d, axis=1)
    md_p25 = np.percentile(min_d, 25, axis=1)
    md_p75 = np.percentile(min_d, 75, axis=1)
    acc_t = np.mean(result["accept_hist"], axis=1) if n_steps > 0 else np.array([])
    t = np.arange(min_d.shape[0])

    ax_ts.plot(t, md_mean, color="#1f77b4", linewidth=2.1, label="mean min-dist")
    ax_ts.fill_between(t, md_p25, md_p75, color="#1f77b4", alpha=0.20, linewidth=0.0, label="IQR")
    ax_ts.axhline(float(r0), color="black", linestyle="--", linewidth=1.1, label="r0")
    ax_ts.set_title(f"{title} - Distance-to-Manifold Dynamics")
    ax_ts.set_xlabel("step")
    ax_ts.set_ylabel("min distance")

    ax2 = ax_ts.twinx()
    if acc_t.size > 0:
        ax2.plot(np.arange(1, n_steps + 1), acc_t, color="#ff7f0e", linewidth=1.8, label="accept(t)")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel("acceptance")

    h1, l1 = ax_ts.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_ts.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9, frameon=True)


def save_summary(out_dir: Path, toy_summary: dict[str, Any], real_summary: dict[str, Any], args: argparse.Namespace) -> None:
    payload = {
        "timestamp": out_dir.name,
        "args": vars(args),
        "toy": toy_summary,
        "real_4k": real_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Sampler Alignment Dashboard Summary")
    lines.append("")
    lines.append("Comparison of `volume_riemannian` RHMC on Toy metric vs Real 4K metric.")
    lines.append("")
    lines.append("| Metric | Toy | Real 4K |")
    lines.append("|---|---:|---:|")
    keys = [
        "acceptance_rate",
        "rescue_rate",
        "median_hit_step",
        "proposal_alignment_outside_mean",
        "accepted_alignment_outside_mean",
        "proposal_opposition_rate_outside",
        "accepted_opposition_rate_outside",
        "first_step_abs_alignment_principal_mean",
        "first_step_alignment_to_manifold_mean",
        "delta_logdet_inv_mean",
        "delta_logdet_inv_median",
        "escaped_count",
    ]
    for k in keys:
        v_t = toy_summary.get(k, float("nan"))
        v_r = real_summary.get(k, float("nan"))
        st = _safe(float(v_t)) if isinstance(v_t, (float, int, np.floating, np.integer)) else str(v_t)
        sr = _safe(float(v_r)) if isinstance(v_r, (float, int, np.floating, np.integer)) else str(v_r)
        lines.append(f"| `{k}` | {st} | {sr} |")

    lines.append("")
    lines.append("Interpretation:")
    lines.append("- `proposal_alignment_outside_mean`: geometry + integrator effect before Metropolis.")
    lines.append("- `accepted_alignment_outside_mean`: actual chain displacement alignment.")
    lines.append("- `proposal/accepted_opposition_rate_outside`: fraction of outside steps pointing opposite the nearest-manifold direction.")
    lines.append("- `first_step_abs_alignment_principal_mean`: first-step alignment with principal mobility axis (signless).")
    lines.append("- `first_step_alignment_to_manifold_mean`: first-step alignment toward nearest-manifold direction.")
    lines.append("- `rescue_rate`: fraction of chains that reached shell `min_dist <= r0`.")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare sampler alignment: toy metric vs real 4K metric.")
    p.add_argument("--real_model_path", type=str, default="outputs/reference_models/4K")
    p.add_argument("--output_dir", type=str, default="results/sampler_alignment_dashboard")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--n_starts", type=int, default=16)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--n_lf_inner", type=int, default=6)
    p.add_argument("--eps_toy", type=float, default=0.18)
    p.add_argument("--eps_real", type=float, default=0.02)
    p.add_argument("--fp_steps", type=int, default=3)
    p.add_argument("--fp_damping", type=float, default=0.72)
    p.add_argument("--volume_power", type=float, default=0.8)
    p.add_argument("--grid_n", type=int, default=180)
    p.add_argument("--bounds_scale", type=float, default=2.25)
    p.add_argument("--min_bounds", type=float, default=6.0)
    p.add_argument("--start_mult_toy_min", type=float, default=2.2)
    p.add_argument("--start_mult_toy_max", type=float, default=3.0)
    p.add_argument("--start_mult_real_min", type=float, default=1.6)
    p.add_argument("--start_mult_real_max", type=float, default=2.4)
    p.add_argument("--toy_sigma", type=float, default=2.8)
    p.add_argument("--toy_eps_floor", type=float, default=1e-4)
    p.add_argument("--max_radius", type=float, default=45.0)
    p.add_argument("--momentum_persist", type=float, default=0.85)
    p.add_argument("--real_directional_boost", type=float, default=0.0)
    p.add_argument("--real_directional_sharpness", type=float, default=6.0)
    p.add_argument("--real_directional_r0_scale", type=float, default=1.0)
    p.add_argument("--use_dual_metric", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device("cpu")
    real_model = load_geometry_model(Path(args.real_model_path), device=device)
    toy_model = ToyCrescentMetricModel(
        sigma=float(args.toy_sigma),
        eps_floor=float(args.toy_eps_floor),
    ).to(device)
    toy_model.eval()

    toy_dims = choose_dims_from_variance(toy_model.centroids_tens)
    real_dims = choose_dims_from_variance(real_model.centroids_tens)

    toy_r0 = infer_r0(toy_model, toy_model.centroids_tens)
    real_r0 = infer_r0(real_model, real_model.centroids_tens)
    real_model_effective: Any = real_model
    if float(args.real_directional_boost) > 0.0:
        real_model_effective = DirectionalBiasMetricWrapper(
            base_model=real_model,
            boost=float(args.real_directional_boost),
            r0=float(real_r0) * float(args.real_directional_r0_scale),
            sharpness=float(args.real_directional_sharpness),
            jitter=1e-6,
        ).to(device)

    toy_starts = generate_void_starts(
        centroids=toy_model.centroids_tens.detach(),
        dims=toy_dims,
        n_starts=int(args.n_starts),
        mult_min=float(args.start_mult_toy_min),
        mult_max=float(args.start_mult_toy_max),
    )
    real_starts = generate_void_starts(
        centroids=real_model_effective.centroids_tens.detach(),
        dims=real_dims,
        n_starts=int(args.n_starts),
        mult_min=float(args.start_mult_real_min),
        mult_max=float(args.start_mult_real_max),
    )

    toy_result = run_rhmc_ensemble(
        model=toy_model,
        dims=toy_dims,
        starts=toy_starts,
        volume_power=float(args.volume_power),
        steps=int(args.steps),
        n_lf_inner=int(args.n_lf_inner),
        eps=float(args.eps_toy),
        fp_steps=int(args.fp_steps),
        fp_damping=float(args.fp_damping),
        r0=float(toy_r0),
        max_radius=float(args.max_radius),
        seed=int(args.seed + 31),
        momentum_persist=float(args.momentum_persist),
        use_dual_metric=bool(args.use_dual_metric),
    )
    real_result = run_rhmc_ensemble(
        model=real_model_effective,
        dims=real_dims,
        starts=real_starts,
        volume_power=float(args.volume_power),
        steps=int(args.steps),
        n_lf_inner=int(args.n_lf_inner),
        eps=float(args.eps_real),
        fp_steps=int(args.fp_steps),
        fp_damping=float(args.fp_damping),
        r0=float(real_r0),
        max_radius=float(args.max_radius),
        seed=int(args.seed + 53),
        momentum_persist=float(args.momentum_persist),
        use_dual_metric=bool(args.use_dual_metric),
    )

    toy_xx, toy_yy, toy_u, _ = evaluate_potential_grid(
        model=toy_model,
        centroids=toy_model.centroids_tens.detach(),
        dims=toy_dims,
        volume_power=float(args.volume_power),
        grid_n=int(args.grid_n),
        bounds_scale=float(args.bounds_scale),
        min_bounds=float(args.min_bounds),
    )
    real_xx, real_yy, real_u, _ = evaluate_potential_grid(
        model=real_model_effective,
        centroids=real_model_effective.centroids_tens.detach(),
        dims=real_dims,
        volume_power=float(args.volume_power),
        grid_n=int(args.grid_n),
        bounds_scale=float(args.bounds_scale),
        min_bounds=float(args.min_bounds),
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(3, 2, figsize=(22, 17))
    plot_column(
        ax_map=axes[0, 0],
        ax_hist=axes[1, 0],
        ax_ts=axes[2, 0],
        title="Toy metric",
        xx=toy_xx,
        yy=toy_yy,
        u=toy_u,
        centroids_2d=toy_model.centroids_tens[:, [toy_dims[0], toy_dims[1]]].detach().cpu().numpy(),
        result=toy_result,
        r0=float(toy_r0),
    )
    plot_column(
        ax_map=axes[0, 1],
        ax_hist=axes[1, 1],
        ax_ts=axes[2, 1],
        title=f"Real 4K metric dims={real_dims}",
        xx=real_xx,
        yy=real_yy,
        u=real_u,
        centroids_2d=real_model_effective.centroids_tens[:, [real_dims[0], real_dims[1]]].detach().cpu().numpy(),
        result=real_result,
        r0=float(real_r0),
    )
    fig.suptitle(
        "Sampler Alignment Dashboard - VolumeElementRiemannianRHMC (Toy vs Real 4K)",
        fontsize=21,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sampler_alignment_toy_vs_real.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    save_summary(
        out_dir=out_dir,
        toy_summary=toy_result["summary"],
        real_summary=real_result["summary"],
        args=args,
    )

    print(f"Saved dashboard: {out_path}")
    print(f"Saved summary: {out_dir / 'summary.json'}")
    print(
        "Toy summary: "
        f"acc={toy_result['summary']['acceptance_rate']:.3f}, "
        f"align_prop={toy_result['summary']['proposal_alignment_outside_mean']:.3f}, "
        f"align_acc={toy_result['summary']['accepted_alignment_outside_mean']:.3f}, "
        f"rescue={toy_result['summary']['rescue_rate']:.3f}"
    )
    print(
        "Real summary: "
        f"acc={real_result['summary']['acceptance_rate']:.3f}, "
        f"align_prop={real_result['summary']['proposal_alignment_outside_mean']:.3f}, "
        f"align_acc={real_result['summary']['accepted_alignment_outside_mean']:.3f}, "
        f"rescue={real_result['summary']['rescue_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
