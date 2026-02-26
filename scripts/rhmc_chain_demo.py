#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

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


SAMPLERS: dict[str, Any] = {
    "riemannian": RiemannianHMCSampler,
    "geodesic": GeodesicHMCSampler,
    "volume": RHVAEVolumeElementHMCSampler,
    "volume_riemannian": VolumeElementRiemannianHMCSampler,
    "volume_det": RHVAELogDetHMCSampler,
    "volume_riemannian_det": GeodesicHMCSampler,
    "dual_riemannian": DualRiemannianHMCSampler,
    "hybrid_volume_mix": None,
}


class HybridVolumeMixSampler:
    """Alternate `volume` (explore) and `volume_riemannian` (rescue) kernels."""

    is_hybrid_mix = True

    def __init__(
        self,
        model: GeometryRHVAE,
        mcmc_steps_nbr: int = 100,
        n_lf: int = 15,
        eps_lf: float = 0.03,
        beta_zero: float = 1.0,
        exact: bool = True,
        volume_power: float = 0.5,
        fp_steps: int = 15,
        fp_damping: float = 0.7,
        radial_prior_weight: float = 0.0,
        radial_prior_center: list[float] | None = None,
        hybrid_explore_steps: int = 3,
        hybrid_rescue_steps: int = 1,
        hybrid_rescue_warmup_steps: int = 0,
        hybrid_rescue_use_dual_metric: bool = False,
    ) -> None:
        self.model = model
        self.device = model.device if hasattr(model, "device") else next(model.parameters()).device
        self.mcmc_steps_nbr = int(mcmc_steps_nbr)
        self.n_lf = int(n_lf)
        self.eps_lf = float(eps_lf)
        self.volume_power = float(volume_power)
        self.hybrid_explore_steps = max(1, int(hybrid_explore_steps))
        self.hybrid_rescue_steps = max(1, int(hybrid_rescue_steps))
        self.hybrid_rescue_warmup_steps = max(0, int(hybrid_rescue_warmup_steps))
        self.last_acceptance_rate = float("nan")

        self.explore_sampler = RHVAEVolumeElementHMCSampler(
            model,
            mcmc_steps_nbr=mcmc_steps_nbr,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
            exact=exact,
            volume_power=volume_power,
        )
        self.rescue_sampler = VolumeElementRiemannianHMCSampler(
            model,
            mcmc_steps_nbr=mcmc_steps_nbr,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
            exact=exact,
            fp_steps=fp_steps,
            fp_damping=fp_damping,
            volume_power=volume_power,
            radial_prior_weight=radial_prior_weight,
            radial_prior_center=radial_prior_center,
            use_dual_metric=bool(hybrid_rescue_use_dual_metric),
        )

        self._exact = bool(exact)
        self._momentum_persist = 0.0
        self._fp_steps = int(fp_steps)
        self._fp_damping = float(fp_damping)
        self.exact = bool(exact)
        self.momentum_persist = 0.0
        self.fp_steps = int(fp_steps)
        self.fp_damping = float(fp_damping)

    @property
    def exact(self) -> bool:
        return self._exact

    @exact.setter
    def exact(self, value: bool) -> None:
        self._exact = bool(value)
        self.explore_sampler.exact = bool(value)
        self.rescue_sampler.exact = bool(value)

    @property
    def momentum_persist(self) -> float:
        return self._momentum_persist

    @momentum_persist.setter
    def momentum_persist(self, value: float) -> None:
        self._momentum_persist = float(value)
        self.explore_sampler.momentum_persist = float(value)
        self.rescue_sampler.momentum_persist = float(value)

    @property
    def fp_steps(self) -> int:
        return self._fp_steps

    @fp_steps.setter
    def fp_steps(self, value: int) -> None:
        self._fp_steps = int(value)
        self.rescue_sampler.fp_steps = int(value)

    @property
    def fp_damping(self) -> float:
        return self._fp_damping

    @fp_damping.setter
    def fp_damping(self, value: float) -> None:
        self._fp_damping = float(value)
        self.rescue_sampler.fp_damping = float(value)

    @property
    def cycle_length(self) -> int:
        return int(self.hybrid_explore_steps + self.hybrid_rescue_steps)

    def kernel_key_for_step(self, step_idx: int) -> str:
        if int(step_idx) < int(self.hybrid_rescue_warmup_steps):
            return "rescue"
        pos = int(step_idx) % self.cycle_length
        return "explore" if pos < self.hybrid_explore_steps else "rescue"

    def sampler_for_step(self, step_idx: int) -> Any:
        key = self.kernel_key_for_step(step_idx)
        return self.explore_sampler if key == "explore" else self.rescue_sampler

    def _initialize_momentum(self, z: torch.Tensor) -> torch.Tensor:
        return self.explore_sampler._initialize_momentum(z)

    def _compute_hamiltonian(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        return self.explore_sampler._compute_hamiltonian(z, rho)

    def sample(self, n_samples: int = 100, eps_jitter: float = 0.0, n_lf_jitter: int = 0) -> torch.Tensor:
        device = self.device
        K = self.model.centroids_tens.shape[0]
        idx = torch.randint(K, (n_samples,), device=device)
        z = self.model.centroids_tens[idx].detach().clone().to(device)
        rho_prev_map: dict[str, torch.Tensor | None] = {"explore": None, "rescue": None}
        accept_count = 0.0

        for step in range(self.mcmc_steps_nbr):
            kernel_key = self.kernel_key_for_step(step)
            active_sampler = self.sampler_for_step(step)
            trans = _single_hmc_transition(
                active_sampler=active_sampler,
                z_curr=z,
                n_lf_local=self.n_lf,
                eps_lf_local=self.eps_lf,
                rho_prev=rho_prev_map[kernel_key],
                eps_jitter_local=eps_jitter,
                n_lf_jitter_local=n_lf_jitter,
            )
            z = trans["z_next"]
            rho_prev_map[kernel_key] = trans["rho_prev_next"]
            accept_count += float(trans["accept_rate"]) * float(n_samples)

        self.last_acceptance_rate = float(accept_count / max(1, self.mcmc_steps_nbr * n_samples))
        print(f"✅ Hybrid(Volume+Riemannian) Acceptance Rate: {self.last_acceptance_rate:.3f}")
        return z.detach()

    def sample_prior(self, num_samples: int, method: str = "hybrid") -> torch.Tensor:
        return self.sample(num_samples)

    def sample_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor, method: str = "hybrid") -> torch.Tensor:
        eps = torch.randn_like(mu)
        return (mu + eps * torch.exp(0.5 * log_var)).detach()


def _build_model(metric_path: Path, device: torch.device) -> GeometryRHVAE:
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


def _build_sampler(
    name: str,
    model: GeometryRHVAE,
    mcmc_steps: int,
    n_lf: int,
    eps_lf: float,
    beta_zero: float,
    volume_power: float = 0.5,
    radial_prior_weight: float = 0.0,
    radial_prior_center: list[float] | None = None,
    hybrid_explore_steps: int = 3,
    hybrid_rescue_steps: int = 1,
    hybrid_rescue_warmup_steps: int = 0,
    hybrid_rescue_use_dual_metric: bool = False,
    use_dual_metric: bool = False,
):
    if name == "hybrid_volume_mix":
        return HybridVolumeMixSampler(
            model,
            mcmc_steps_nbr=mcmc_steps,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
            exact=True,
            volume_power=volume_power,
            fp_steps=15,
            fp_damping=0.7,
            radial_prior_weight=radial_prior_weight,
            radial_prior_center=radial_prior_center,
            hybrid_explore_steps=hybrid_explore_steps,
            hybrid_rescue_steps=hybrid_rescue_steps,
            hybrid_rescue_warmup_steps=hybrid_rescue_warmup_steps,
            hybrid_rescue_use_dual_metric=hybrid_rescue_use_dual_metric,
        )
    sampler_cls = SAMPLERS[name]
    if name == "geodesic":
        return sampler_cls(
            model,
            mcmc_steps_nbr=mcmc_steps,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
            include_metropolis=True,
        )
    if name == "riemannian":
        return sampler_cls(
            model,
            mcmc_steps_nbr=mcmc_steps,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
            include_volume_grad=True,
        )
    if name in {"volume", "volume_riemannian"}:
        extra_kwargs = {}
        if name == "volume_riemannian":
            extra_kwargs = {
                "radial_prior_weight": radial_prior_weight,
                "radial_prior_center": radial_prior_center,
                "use_dual_metric": bool(use_dual_metric),
            }
        return sampler_cls(
            model,
            mcmc_steps_nbr=mcmc_steps,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
            volume_power=volume_power,
            **extra_kwargs,
        )
    return sampler_cls(
        model,
        mcmc_steps_nbr=mcmc_steps,
        n_lf=n_lf,
        eps_lf=eps_lf,
        beta_zero=beta_zero,
    )


def _compute_hamiltonian(sampler: Any, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    if bool(getattr(sampler, "is_hybrid_mix", False)):
        sampler = sampler.explore_sampler
    if hasattr(sampler, "_compute_hamiltonian"):
        return sampler._compute_hamiltonian(z, rho)
    if hasattr(sampler, "_hamiltonian"):
        return sampler._hamiltonian(z, rho)
    return torch.zeros(z.shape[0], device=z.device)


def _compute_kinetic(sampler: Any, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    if bool(getattr(sampler, "is_hybrid_mix", False)):
        sampler = sampler.explore_sampler
    if isinstance(sampler, (RiemannianHMCSampler, GeodesicHMCSampler, VolumeElementRiemannianHMCSampler)):
        G_inv = sampler.model.G_inv(z)
        return 0.5 * torch.einsum("bi,bij,bj->b", rho, G_inv, rho)
    if isinstance(sampler, DualRiemannianHMCSampler):
        G = sampler.model.G(z)
        return 0.5 * torch.einsum("bi,bij,bj->b", rho, G, rho)
    return 0.5 * torch.sum(rho * rho, dim=1)


def _energy_terms(
    sampler: Any, z: torch.Tensor, rho: torch.Tensor
) -> tuple[float, float, float]:
    with torch.no_grad():
        H = _compute_hamiltonian(sampler, z, rho)
        K = _compute_kinetic(sampler, z, rho)
        U = H - K
    return float(U.mean().item()), float(K.mean().item()), float(H.mean().item())


def run_conservation_trace(
    start_z: torch.Tensor,
    sampler: Any,
    n_lf: int,
    eps_lf: float,
    apply_tempering: bool = False,
) -> np.ndarray:
    z = start_z.clone().detach().requires_grad_(True)
    rho = sampler._initialize_momentum(z)
    hs = []
    with torch.no_grad():
        hs.append(_compute_hamiltonian(sampler, z, rho).mean().item())

    use_tempering = (
        bool(apply_tempering)
        and (not bool(getattr(sampler, "exact", False)))
        and hasattr(sampler, "_tempering")
        and hasattr(sampler, "beta_zero_sqrt")
    )
    beta_sqrt_old = sampler.beta_zero_sqrt if use_tempering else None

    for k in range(n_lf):
        if hasattr(sampler, "_generalized_leapfrog_step"):
            z, rho = sampler._generalized_leapfrog_step(z, rho, eps_lf)
        elif hasattr(sampler, "_leapfrog"):
            z, rho = sampler._leapfrog(z, rho, eps_lf)

        if use_tempering and beta_sqrt_old is not None:
            beta_sqrt = sampler._tempering(k + 1, n_lf, sampler.beta_zero_sqrt)
            if torch.is_tensor(beta_sqrt):
                beta_sqrt = beta_sqrt.to(z.device)
            else:
                beta_sqrt = torch.tensor(beta_sqrt, device=z.device)
            rho = (beta_sqrt_old / beta_sqrt) * rho
            beta_sqrt_old = beta_sqrt

        with torch.no_grad():
            hs.append(_compute_hamiltonian(sampler, z, rho).mean().item())

    return np.array(hs)


def _single_hmc_transition(
    active_sampler: Any,
    z_curr: torch.Tensor,
    n_lf_local: int,
    eps_lf_local: float,
    rho_prev: torch.Tensor | None,
    eps_jitter_local: float = 0.0,
    n_lf_jitter_local: int = 0,
    no_metropolis_local: bool = False,
) -> dict[str, Any]:
    z_curr = z_curr.detach().requires_grad_(True)
    if hasattr(active_sampler, "_refresh_momentum"):
        rho = active_sampler._refresh_momentum(z_curr, rho_prev)
    else:
        rho = active_sampler._initialize_momentum(z_curr)
    rho0 = rho
    with torch.no_grad():
        H0 = _compute_hamiltonian(active_sampler, z_curr, rho)

    z_prop = z_curr.clone()
    rho_prop = rho.clone()
    use_tempering = (
        (not bool(getattr(active_sampler, "exact", False)))
        and hasattr(active_sampler, "_tempering")
        and hasattr(active_sampler, "beta_zero_sqrt")
    )
    beta_sqrt_old = active_sampler.beta_zero_sqrt if use_tempering else None

    if n_lf_jitter_local > 0:
        jitter = torch.randint(-n_lf_jitter_local, n_lf_jitter_local + 1, (1,), device=z_curr.device).item()
        local_n_lf = max(1, int(n_lf_local) + int(jitter))
    else:
        local_n_lf = int(n_lf_local)

    for k in range(local_n_lf):
        if eps_jitter_local > 0.0:
            u = (torch.rand(1, device=z_curr.device).item() - 0.5) * 2.0 * float(eps_jitter_local)
            eps_eff = float(eps_lf_local) * (1.0 + u)
        else:
            eps_eff = float(eps_lf_local)
        if hasattr(active_sampler, "_generalized_leapfrog_step"):
            z_prop, rho_prop = active_sampler._generalized_leapfrog_step(z_prop, rho_prop, eps_eff)
        elif hasattr(active_sampler, "_leapfrog"):
            z_prop, rho_prop = active_sampler._leapfrog(z_prop, rho_prop, eps_eff)

        if use_tempering and beta_sqrt_old is not None:
            beta_sqrt = active_sampler._tempering(k + 1, local_n_lf, active_sampler.beta_zero_sqrt)
            if torch.is_tensor(beta_sqrt):
                beta_sqrt = beta_sqrt.to(z_curr.device)
            else:
                beta_sqrt = torch.tensor(beta_sqrt, device=z_curr.device)
            rho_prop = (beta_sqrt_old / beta_sqrt) * rho_prop
            beta_sqrt_old = beta_sqrt

    with torch.no_grad():
        H1 = _compute_hamiltonian(active_sampler, z_prop, rho_prop)
        dH = H1 - H0
        if bool(no_metropolis_local):
            moves = torch.ones_like(H0).view(-1, 1)
            z_next = z_prop.detach()
            rho_mix = rho_prop.detach()
        else:
            alpha = torch.exp(-dH).clamp(max=1.0)
            u = torch.rand_like(alpha)
            moves = (u < alpha).float().view(-1, 1)
            z_next = (moves * z_prop + (1.0 - moves) * z_curr).detach()
            rho_mix = (moves * rho_prop + (1.0 - moves) * rho).detach()

        if getattr(active_sampler, "momentum_persist", 0.0) > 0.0:
            if bool(no_metropolis_local):
                rho_prev_next = rho_prop.detach()
            else:
                rho_prev_next = (moves * rho_prop + (1.0 - moves) * (-rho0)).detach()
        else:
            rho_prev_next = None

    U_val, K_val, H_val = _energy_terms(active_sampler, z_next, rho_mix)
    return {
        "z_next": z_next,
        "rho_prev_next": rho_prev_next,
        "proposal_dh": float(dH.mean().item()),
        "accept_rate": float(moves.mean().item()),
        "U": float(U_val),
        "K": float(K_val),
        "H": float(H_val),
    }


def run_hmc_chain(
    start_z: torch.Tensor,
    sampler: Any,
    chain_length: int,
    n_lf: int,
    eps_lf: float,
    eps_jitter: float = 0.0,
    n_lf_jitter: int = 0,
    no_metropolis: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z = start_z.clone().detach().requires_grad_(True)
    chain = [z.detach().cpu().squeeze().numpy()]
    potentials: list[float] = []
    kinetics: list[float] = []
    hamiltonians: list[float] = []
    accept_flags: list[float] = []
    proposal_dh: list[float] = []

    if bool(getattr(sampler, "is_hybrid_mix", False)):
        init_sampler = sampler.sampler_for_step(0)
    else:
        init_sampler = sampler

    rho_init = init_sampler._initialize_momentum(z)
    U0, K0, H0 = _energy_terms(init_sampler, z, rho_init)
    potentials.append(U0)
    kinetics.append(K0)
    hamiltonians.append(H0)

    rho_prev_single: torch.Tensor | None = None
    rho_prev_map: dict[str, torch.Tensor | None] = {"explore": None, "rescue": None}

    for step_idx in range(chain_length):
        if bool(getattr(sampler, "is_hybrid_mix", False)):
            kernel_key = sampler.kernel_key_for_step(step_idx)
            active_sampler = sampler.sampler_for_step(step_idx)
            trans = _single_hmc_transition(
                active_sampler=active_sampler,
                z_curr=z,
                n_lf_local=n_lf,
                eps_lf_local=eps_lf,
                rho_prev=rho_prev_map[kernel_key],
                eps_jitter_local=eps_jitter,
                n_lf_jitter_local=n_lf_jitter,
                no_metropolis_local=bool(no_metropolis),
            )
            rho_prev_map[kernel_key] = trans["rho_prev_next"]
        else:
            active_sampler = sampler
            trans = _single_hmc_transition(
                active_sampler=active_sampler,
                z_curr=z,
                n_lf_local=n_lf,
                eps_lf_local=eps_lf,
                rho_prev=rho_prev_single,
                eps_jitter_local=eps_jitter,
                n_lf_jitter_local=n_lf_jitter,
                no_metropolis_local=bool(no_metropolis),
            )
            rho_prev_single = trans["rho_prev_next"]

        z = trans["z_next"].requires_grad_(True)
        potentials.append(float(trans["U"]))
        kinetics.append(float(trans["K"]))
        hamiltonians.append(float(trans["H"]))
        accept_flags.append(float(trans["accept_rate"]))
        proposal_dh.append(float(trans["proposal_dh"]))
        chain.append(z.detach().cpu().squeeze().numpy())

    energies = {
        "potential": np.array(potentials),
        "kinetic": np.array(kinetics),
        "hamiltonian": np.array(hamiltonians),
        "accept": np.array(accept_flags),
        "proposal_dh": np.array(proposal_dh),
    }
    return np.array(chain), energies


def _lag1_autocorr(x: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:] - x[1:].mean()
    denom = np.sqrt((x0 * x0).sum() * (x1 * x1).sum())
    if denom <= 1e-12:
        return float("nan")
    return float((x0 * x1).sum() / denom)


def summarize_chain(label: str, chain: np.ndarray, energies: dict[str, np.ndarray]) -> dict[str, float]:
    h = energies.get("hamiltonian", np.array([]))
    dh = energies.get("proposal_dh", np.array([]))
    accept = energies.get("accept", np.array([]))
    step_len = np.linalg.norm(np.diff(chain, axis=0), axis=1) if chain.shape[0] > 1 else np.array([])

    print(f"\n[{label}] Diagnostics")
    diag = {
        "accept_rate": float(accept.mean()) if accept.size > 0 else float("nan"),
        "proposal_dh_mean": float(dh.mean()) if dh.size > 0 else float("nan"),
        "proposal_dh_std": float(dh.std()) if dh.size > 0 else float("nan"),
        "proposal_dh_max": float(np.max(np.abs(dh))) if dh.size > 0 else float("nan"),
        "h_lag1": float(_lag1_autocorr(h)) if h.size > 1 else float("nan"),
        "step_len_mean": float(step_len.mean()) if step_len.size > 0 else float("nan"),
        "step_len_median": float(np.median(step_len)) if step_len.size > 0 else float("nan"),
    }

    if accept.size > 0:
        print(f"  accept rate: {diag['accept_rate']:.3f}")
    if dh.size > 0:
        print(
            f"  proposal ΔH mean: {diag['proposal_dh_mean']:.4f}, std: {diag['proposal_dh_std']:.4f}, max |ΔH|: {diag['proposal_dh_max']:.4f}"
        )
    if h.size > 1:
        print(f"  H lag-1 autocorr: {diag['h_lag1']:.3f}")
    if step_len.size > 0:
        print(f"  step length mean: {diag['step_len_mean']:.3f}, median: {diag['step_len_median']:.3f}")

    return diag


def compute_score(centroid: dict[str, float], void: dict[str, float], target_accept: float) -> float:
    """Heuristic score: favor larger steps, lower ΔH variance, and acceptance near target."""
    step_mean = np.nanmean([centroid.get("step_len_mean"), void.get("step_len_mean")])
    dh_std = np.nanmean([centroid.get("proposal_dh_std"), void.get("proposal_dh_std")])
    acc = np.nanmean([centroid.get("accept_rate"), void.get("accept_rate")])
    if np.isnan(step_mean) or np.isnan(dh_std) or np.isnan(acc):
        return float("nan")
    return float(step_mean - 0.5 * dh_std - abs(acc - target_accept))


def _grid_logdet(
    model: GeometryRHVAE,
    bounds: float,
    grid_size: int,
    mode: str = "logdet_inv",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = torch.linspace(-bounds, bounds, grid_size)
    ys = torch.linspace(-bounds, bounds, grid_size)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
    pts = pts.to(next(model.parameters()).device)
    pts = torch.cat([pts, torch.zeros(pts.shape[0], model.latent_dim - 2, device=pts.device)], dim=1)
    with torch.no_grad():
        if mode == "logdet":
            G = model.G(pts)
            Z = torch.linalg.slogdet(G).logabsdet
        else:
            G_inv = model.G_inv(pts)
            Z = torch.linalg.slogdet(G_inv).logabsdet
    Z = Z.view(grid_size, grid_size).cpu().numpy()
    return X.cpu().numpy(), Y.cpu().numpy(), Z


def _grid_energy_maps(
    model: GeometryRHVAE,
    sampler: Any,
    bounds: float,
    grid_size: int,
    rho_seed: int = 0,
    kinetic_mode: str = "fixed_rho",
    kinetic_mc_samples: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = torch.linspace(-bounds, bounds, grid_size)
    ys = torch.linspace(-bounds, bounds, grid_size)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
    pts = pts.to(next(model.parameters()).device)
    if model.latent_dim > 2:
        pts = torch.cat(
            [pts, torch.zeros(pts.shape[0], model.latent_dim - 2, device=pts.device)], dim=1
        )
    with torch.no_grad():
        # Potential: evaluate H(z, 0) so K = 0
        zero_rho = torch.zeros_like(pts)
        U = _compute_hamiltonian(sampler, pts, zero_rho)

        if kinetic_mode == "fixed_rho":
            # Kinetic: evaluate with a fixed unit rho direction
            gen = torch.Generator(device=pts.device)
            gen.manual_seed(int(rho_seed))
            rho_vec = torch.randn(model.latent_dim, generator=gen, device=pts.device)
            rho_vec = rho_vec / (rho_vec.norm() + 1e-8)
            rho = rho_vec.unsqueeze(0).expand(pts.shape[0], -1)
            K = _compute_kinetic(sampler, pts, rho)
        elif kinetic_mode == "trace_ginv":
            G_inv = model.G_inv(pts)
            K = 0.5 * torch.einsum("bii->b", G_inv)
        elif kinetic_mode == "trace_g":
            G = model.G(pts)
            K = 0.5 * torch.einsum("bii->b", G)
        elif kinetic_mode == "lambda_max_ginv":
            G_inv = model.G_inv(pts)
            evals = torch.linalg.eigvalsh(G_inv)
            K = 0.5 * evals[:, -1]
        elif kinetic_mode == "lambda_min_ginv":
            G_inv = model.G_inv(pts)
            evals = torch.linalg.eigvalsh(G_inv)
            K = 0.5 * evals[:, 0]
        elif kinetic_mode == "expected_mc":
            # Monte Carlo estimate of E[K] with rho ~ N(0, G(z))
            # Note: expectation should be ~0.5 * D everywhere.
            D = model.latent_dim
            K_accum = torch.zeros(pts.shape[0], device=pts.device)
            # chunk to limit memory
            chunk = 1024
            for start in range(0, pts.shape[0], chunk):
                end = min(start + chunk, pts.shape[0])
                pts_chunk = pts[start:end]
                G = model.G(pts_chunk)
                try:
                    L = torch.linalg.cholesky(G)
                except torch.linalg.LinAlgError:
                    evals, evecs = torch.linalg.eigh(G)
                    evals = torch.clamp(evals, min=1e-6)
                    L = evecs @ torch.diag_embed(torch.sqrt(evals))
                # sample rho: [S, B, D]
                gamma = torch.randn(kinetic_mc_samples, pts_chunk.shape[0], D, device=pts.device)
                rho = torch.einsum("bij,sbj->sbi", L, gamma)
                G_inv = model.G_inv(pts_chunk)
                # K for each sample
                K_s = 0.5 * torch.einsum("sbi,bij,sbj->sb", rho, G_inv, rho)
                K_accum[start:end] = K_s.mean(dim=0)
            K = K_accum
        else:
            raise ValueError(f"Unknown kinetic_mode: {kinetic_mode}")

    U = U.view(grid_size, grid_size).cpu().numpy()
    K = K.view(grid_size, grid_size).cpu().numpy()
    return X.cpu().numpy(), Y.cpu().numpy(), U, K


def _grid_potential_map(
    model: GeometryRHVAE,
    sampler: Any,
    bounds: float,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = torch.linspace(-bounds, bounds, grid_size)
    ys = torch.linspace(-bounds, bounds, grid_size)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
    pts = pts.to(next(model.parameters()).device)
    if model.latent_dim > 2:
        pts = torch.cat(
            [pts, torch.zeros(pts.shape[0], model.latent_dim - 2, device=pts.device)], dim=1
        )
    with torch.no_grad():
        zero_rho = torch.zeros_like(pts)
        U = _compute_hamiltonian(sampler, pts, zero_rho)
    U = U.view(grid_size, grid_size).cpu().numpy()
    return X.cpu().numpy(), Y.cpu().numpy(), U


def _anisotropy_quiver(
    model: GeometryRHVAE,
    bounds: float,
    grid_size: int,
    mode: str = "G",
    stride: int = 6,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = torch.linspace(-bounds, bounds, grid_size)
    ys = torch.linspace(-bounds, bounds, grid_size)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    X = X[::stride, ::stride]
    Y = Y[::stride, ::stride]
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1).to(next(model.parameters()).device)
    if model.latent_dim > 2:
        pts = torch.cat(
            [pts, torch.zeros(pts.shape[0], model.latent_dim - 2, device=pts.device)], dim=1
        )
    with torch.no_grad():
        mats = model.G(pts) if mode == "G" else model.G_inv(pts)
        evals, evecs = torch.linalg.eigh(mats)
        # Use principal eigenvector (largest eigenvalue)
        v = evecs[:, :, -1]
        # Anisotropy strength based on eigenvalue ratio
        ratio = (evals[:, -1] / evals[:, 0].clamp_min(1e-12))
        v2 = v[:, :2]
        # Normalize direction and scale by anisotropy strength
        v2 = v2 / (v2.norm(dim=1, keepdim=True) + 1e-8)
        ratio = ratio / ratio.max().clamp_min(1e-8)
        v2 = v2 * ratio.unsqueeze(1) * float(scale)
    U = v2[:, 0].reshape(X.shape).cpu().numpy()
    V = v2[:, 1].reshape(Y.shape).cpu().numpy()
    return X.cpu().numpy(), Y.cpu().numpy(), U, V


def _grid_anisotropy_map(
    model: GeometryRHVAE,
    bounds: float,
    grid_size: int,
    mode: str = "G",
    metric: str = "logcond",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = torch.linspace(-bounds, bounds, grid_size)
    ys = torch.linspace(-bounds, bounds, grid_size)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
    pts = pts.to(next(model.parameters()).device)
    if model.latent_dim > 2:
        pts = torch.cat(
            [pts, torch.zeros(pts.shape[0], model.latent_dim - 2, device=pts.device)], dim=1
        )
    with torch.no_grad():
        mats = model.G(pts) if mode == "G" else model.G_inv(pts)
        evals = torch.linalg.eigvalsh(mats)
        lam_min = evals[:, 0].clamp_min(1e-12)
        lam_max = evals[:, -1].clamp_min(1e-12)
        if metric == "lambda_max":
            A = lam_max
        elif metric == "lambda_min":
            A = lam_min
        elif metric == "cond":
            A = lam_max / lam_min
        else:  # logcond
            A = torch.log10(lam_max / lam_min)
    A = A.view(grid_size, grid_size).cpu().numpy()
    return X.cpu().numpy(), Y.cpu().numpy(), A


def main() -> None:
    parser = argparse.ArgumentParser(description="RHMC chain demo (centroid vs void).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model dir or rhvae_metric.pt")
    parser.add_argument("--sampler", type=str, default="volume", choices=sorted(SAMPLERS.keys()))
    parser.add_argument("--chain_length", type=int, default=60)
    parser.add_argument("--n_lf", type=int, default=20)
    parser.add_argument("--eps_lf", type=float, default=0.02)
    parser.add_argument("--beta_zero", type=float, default=1.0)
    parser.add_argument("--fp_steps", type=int, default=None, help="Fixed-point steps for implicit RHMC.")
    parser.add_argument("--fp_damping", type=float, default=None, help="Fixed-point damping for implicit RHMC.")
    parser.add_argument(
        "--momentum_persist",
        type=float,
        default=0.0,
        help="Momentum persistence alpha in [0,1). 0=full refresh (standard HMC).",
    )
    parser.add_argument(
        "--volume_power",
        type=float,
        default=0.5,
        help="Exponent for volume-element target: pi(z) ∝ det(G_inv)^volume_power.",
    )
    parser.add_argument(
        "--radial_prior_weight",
        type=float,
        default=0.0,
        help="Optional radial prior weight (lambda) for volume_riemannian sampler.",
    )
    parser.add_argument(
        "--radial_prior_center",
        type=float,
        nargs="+",
        default=None,
        help="Center for radial prior (defaults to origin).",
    )
    parser.add_argument(
        "--hybrid_explore_steps",
        type=int,
        default=3,
        help="For hybrid sampler: number of `volume` transitions per cycle.",
    )
    parser.add_argument(
        "--hybrid_rescue_steps",
        type=int,
        default=1,
        help="For hybrid sampler: number of `volume_riemannian` transitions per cycle.",
    )
    parser.add_argument(
        "--hybrid_rescue_warmup_steps",
        type=int,
        default=0,
        help="For hybrid sampler: initial number of transitions forced to rescue kernel.",
    )
    parser.add_argument(
        "--hybrid_rescue_use_dual_metric",
        action="store_true",
        help="For hybrid sampler: run rescue kernel with dual RHMC metric convention (M=G^-1).",
    )
    exact_group = parser.add_mutually_exclusive_group()
    exact_group.add_argument("--exact", dest="exact", action="store_true", help="Use exact implicit RHMC.")
    exact_group.add_argument("--approx", dest="exact", action="store_false", help="Use legacy approximate updates.")
    parser.set_defaults(exact=True)
    parser.add_argument("--grid_size", type=int, default=120)
    parser.add_argument("--bounds", type=float, default=6.0)
    parser.add_argument("--void_scale", type=float, default=2.0)
    parser.add_argument(
        "--start_pos",
        type=float,
        nargs="+",
        default=None,
        help="Override void start position (provide 2 or latent_dim floats).",
    )
    parser.add_argument(
        "--contour",
        type=str,
        default="logdet_inv",
        choices=["logdet_inv", "logdet"],
        help="Contour field (logdet_inv or logdet).",
    )
    parser.add_argument(
        "--show_anisotropy_quiver",
        action="store_true",
        help="Overlay principal anisotropy directions on a grid.",
    )
    parser.add_argument(
        "--anisotropy_mode",
        type=str,
        default="G",
        choices=["G", "G_inv"],
        help="Which metric to use for anisotropy directions (G or G_inv).",
    )
    parser.add_argument(
        "--anisotropy_stride",
        type=int,
        default=6,
        help="Stride for anisotropy quiver grid (larger = fewer arrows).",
    )
    parser.add_argument(
        "--anisotropy_scale",
        type=float,
        default=1.0,
        help="Scale factor for anisotropy arrow lengths.",
    )
    parser.add_argument(
        "--metric_panel",
        action="store_true",
        help="Save an extra metric panel (anisotropy heatmap + quiver).",
    )
    parser.add_argument(
        "--anisotropy_heatmap",
        type=str,
        default="logcond",
        choices=["logcond", "cond", "lambda_max", "lambda_min"],
        help="Heatmap for anisotropy panel (logcond/cond/lambda_max/lambda_min).",
    )
    parser.add_argument(
        "--skip_energy_maps",
        action="store_true",
        help="Skip potential/kinetic heatmaps (faster).",
    )
    parser.add_argument(
        "--potential_only",
        action="store_true",
        help="Plot only potential U(z) (skip kinetic/H traces and kinetic heatmap).",
    )
    parser.add_argument(
        "--kinetic_mode",
        type=str,
        default="fixed_rho",
        choices=[
            "fixed_rho",
            "trace_ginv",
            "trace_g",
            "lambda_max_ginv",
            "lambda_min_ginv",
            "expected_mc",
        ],
        help="How to visualize the kinetic field.",
    )
    parser.add_argument(
        "--kinetic_mc_samples",
        type=int,
        default=16,
        help="MC samples for kinetic_mode=expected_mc.",
    )
    parser.add_argument(
        "--diagnostics_only",
        action="store_true",
        help="Skip all plotting and only print diagnostics (fast).",
    )
    parser.add_argument(
        "--sweep_eps",
        type=str,
        default=None,
        help="Comma-separated eps values to sweep in one run.",
    )
    parser.add_argument(
        "--sweep_volume_power",
        type=str,
        default=None,
        help="Comma-separated volume_power values to sweep in one run.",
    )
    parser.add_argument(
        "--skip_conservation",
        action="store_true",
        help="Skip single-trajectory Hamiltonian conservation plot.",
    )
    parser.add_argument(
        "--conservation_tempering",
        action="store_true",
        help="Apply tempering during conservation trace (default: off).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None, help="Output path for PNG")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project (optional).")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (optional).")
    parser.add_argument("--wandb_group", type=str, default=None, help="W&B group (optional).")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name (optional).")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated W&B tags.")
    parser.add_argument(
        "--score_target_accept",
        type=float,
        default=0.8,
        help="Target acceptance for the sweep score heuristic.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    metric_path = model_path / "rhvae_metric.pt" if model_path.is_dir() else model_path
    device = torch.device(args.device)

    model = _build_model(metric_path, device)
    centroids = model.centroids_tens.to(device)

    wandb_run = None
    if wandb is not None:
        should_init = bool(
            args.wandb_project
            or os.environ.get("WANDB_PROJECT")
            or os.environ.get("WANDB_SWEEP_ID")
            or os.environ.get("WANDB_RUN_ID")
        )
        if should_init:
            tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None
            wandb_kwargs = {"config": vars(args)}
            if args.wandb_project:
                wandb_kwargs["project"] = args.wandb_project
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            if args.wandb_group:
                wandb_kwargs["group"] = args.wandb_group
            if args.wandb_name:
                wandb_kwargs["name"] = args.wandb_name
            if tags:
                wandb_kwargs["tags"] = tags
            wandb_run = wandb.init(**wandb_kwargs)
    elif args.wandb_project:
        print("wandb not installed; skipping logging.")

    if centroids.shape[1] < 2:
        raise ValueError("Latent dim must be >= 2 for plotting.")

    # Centroid start
    centroid_start = centroids[0:1]

    # Void start: move away from centroid mean along farthest direction
    c2 = centroids[:, :2]
    mean = c2.mean(dim=0)
    dists = torch.norm(c2 - mean, dim=1)
    far_idx = torch.argmax(dists)
    direction = c2[far_idx] - mean
    norm = torch.norm(direction)
    if norm < 1e-6:
        direction = torch.tensor([1.0, 0.0], device=device)
        norm = torch.norm(direction)
    radius = dists.max() * args.void_scale
    void_2d = mean + (direction / norm) * radius
    void_start = torch.zeros(1, model.latent_dim, device=device)
    void_start[0, :2] = void_2d
    if args.start_pos is not None:
        vals = [float(v) for v in args.start_pos]
        if len(vals) < 2:
            raise ValueError("--start_pos expects at least 2 values.")
        if len(vals) > model.latent_dim:
            raise ValueError(
                f"--start_pos has {len(vals)} values but latent_dim is {model.latent_dim}."
            )
        void_start = torch.zeros(1, model.latent_dim, device=device)
        void_start[0, : len(vals)] = torch.tensor(vals, device=device)
        print(f"Using custom void start: {void_start[0].detach().cpu().numpy()}")

    def _parse_float_list(raw: str | None, default: float) -> list[float]:
        if raw is None:
            return [float(default)]
        items = [r.strip() for r in raw.split(",") if r.strip()]
        return [float(v) for v in items] if items else [float(default)]

    eps_list = _parse_float_list(args.sweep_eps, args.eps_lf)
    vol_list = _parse_float_list(args.sweep_volume_power, args.volume_power)

    def _out_path(base: str | None, eps_val: float, vol_val: float) -> str | None:
        if base is None:
            return None
        path = Path(base)
        if len(eps_list) * len(vol_list) == 1:
            return str(path)
        stem = path.stem
        suffix = path.suffix or ".png"
        return str(path.with_name(f"{stem}_eps{eps_val:g}_vp{vol_val:g}{suffix}"))

    # Fast diagnostics sweep (no plotting)
    if args.diagnostics_only:
        for eps_val in eps_list:
            for vol_val in vol_list:
                sampler = _build_sampler(
                    args.sampler,
                    model,
                    args.chain_length,
                    args.n_lf,
                    eps_val,
                    args.beta_zero,
                    volume_power=vol_val,
                    radial_prior_weight=args.radial_prior_weight,
                    radial_prior_center=args.radial_prior_center,
                    hybrid_explore_steps=args.hybrid_explore_steps,
                    hybrid_rescue_steps=args.hybrid_rescue_steps,
                    hybrid_rescue_warmup_steps=args.hybrid_rescue_warmup_steps,
                    hybrid_rescue_use_dual_metric=args.hybrid_rescue_use_dual_metric,
                )
                sampler.exact = bool(args.exact)
                if args.fp_steps is not None:
                    sampler.fp_steps = int(args.fp_steps)
                if args.fp_damping is not None:
                    sampler.fp_damping = float(args.fp_damping)
                sampler.momentum_persist = float(args.momentum_persist)

                centroid_chain, centroid_energy = run_hmc_chain(
                    centroid_start, sampler, args.chain_length, args.n_lf, eps_val
                )
                void_chain, void_energy = run_hmc_chain(
                    void_start, sampler, args.chain_length, args.n_lf, eps_val
                )
                label = f"Centroid eps={eps_val:g} vp={vol_val:g}"
                centroid_diag = summarize_chain(label, centroid_chain, centroid_energy)
                label = f"Void eps={eps_val:g} vp={vol_val:g}"
                void_diag = summarize_chain(label, void_chain, void_energy)
                score = compute_score(centroid_diag, void_diag, args.score_target_accept)

                if wandb_run is not None:
                    metrics = {
                        "eps_lf": eps_val,
                        "volume_power": vol_val,
                        "centroid/accept": centroid_diag["accept_rate"],
                        "centroid/proposal_dh_std": centroid_diag["proposal_dh_std"],
                        "centroid/proposal_dh_max": centroid_diag["proposal_dh_max"],
                        "centroid/step_len_mean": centroid_diag["step_len_mean"],
                        "centroid/h_lag1": centroid_diag["h_lag1"],
                        "void/accept": void_diag["accept_rate"],
                        "void/proposal_dh_std": void_diag["proposal_dh_std"],
                        "void/proposal_dh_max": void_diag["proposal_dh_max"],
                        "void/step_len_mean": void_diag["step_len_mean"],
                        "void/h_lag1": void_diag["h_lag1"],
                        "diag/score": score,
                    }
                    wandb_run.log(metrics)
        return

    # Single-configuration plotting (or multi-config with output suffixes)
    for eps_val in eps_list:
        for vol_val in vol_list:
            sampler = _build_sampler(
                args.sampler,
                model,
                args.chain_length,
                args.n_lf,
                eps_val,
                args.beta_zero,
                volume_power=vol_val,
                radial_prior_weight=args.radial_prior_weight,
                radial_prior_center=args.radial_prior_center,
                hybrid_explore_steps=args.hybrid_explore_steps,
                hybrid_rescue_steps=args.hybrid_rescue_steps,
                hybrid_rescue_warmup_steps=args.hybrid_rescue_warmup_steps,
                hybrid_rescue_use_dual_metric=args.hybrid_rescue_use_dual_metric,
            )
            # Configure implicit solver / exactness / momentum persistence if supported
            sampler.exact = bool(args.exact)
            if args.fp_steps is not None:
                sampler.fp_steps = int(args.fp_steps)
            if args.fp_damping is not None:
                sampler.fp_damping = float(args.fp_damping)
            sampler.momentum_persist = float(args.momentum_persist)

            centroid_chain, centroid_energy = run_hmc_chain(
                centroid_start, sampler, args.chain_length, args.n_lf, eps_val
            )
            void_chain, void_energy = run_hmc_chain(
                void_start, sampler, args.chain_length, args.n_lf, eps_val
            )
            centroid_diag = summarize_chain("Centroid", centroid_chain, centroid_energy)
            void_diag = summarize_chain("Void", void_chain, void_energy)
            score = compute_score(centroid_diag, void_diag, args.score_target_accept)

            if wandb_run is not None:
                metrics = {
                    "eps_lf": eps_val,
                    "volume_power": vol_val,
                    "centroid/accept": centroid_diag["accept_rate"],
                    "centroid/proposal_dh_std": centroid_diag["proposal_dh_std"],
                    "centroid/proposal_dh_max": centroid_diag["proposal_dh_max"],
                    "centroid/step_len_mean": centroid_diag["step_len_mean"],
                    "centroid/h_lag1": centroid_diag["h_lag1"],
                    "void/accept": void_diag["accept_rate"],
                    "void/proposal_dh_std": void_diag["proposal_dh_std"],
                    "void/proposal_dh_max": void_diag["proposal_dh_max"],
                    "void/step_len_mean": void_diag["step_len_mean"],
                    "void/h_lag1": void_diag["h_lag1"],
                    "diag/score": score,
                }
                wandb_run.log(metrics)

            # Use per-config output path if multiple configs
            out_path = _out_path(args.out, eps_val, vol_val)

            centroid_cons = None
            void_cons = None
            if not args.skip_conservation:
                centroid_cons = run_conservation_trace(
                    centroid_start, sampler, args.n_lf, eps_val, apply_tempering=args.conservation_tempering
                )
                void_cons = run_conservation_trace(
                    void_start, sampler, args.n_lf, eps_val, apply_tempering=args.conservation_tempering
                )

            X, Y, Z = _grid_logdet(model, args.bounds, args.grid_size, mode=args.contour)
            quiver_data = None
            if args.show_anisotropy_quiver or args.metric_panel:
                quiver_data = _anisotropy_quiver(
                    model,
                    args.bounds,
                    args.grid_size,
                    mode=args.anisotropy_mode,
                    stride=args.anisotropy_stride,
                    scale=args.anisotropy_scale,
                )
            metric_map = None
            if args.metric_panel:
                metric_map = _grid_anisotropy_map(
                    model,
                    args.bounds,
                    args.grid_size,
                    mode=args.anisotropy_mode,
                    metric=args.anisotropy_heatmap,
                )

            include_cons = centroid_cons is not None and void_cons is not None
            if args.skip_energy_maps:
                rows = 3 if include_cons else 2
                fig, axes = plt.subplots(
                    rows, 2, figsize=(12, 4 + 2 * rows), gridspec_kw={"height_ratios": [2.0, 1.0] + ([1.0] if include_cons else [])}
                )
                chain_axes = [axes[0, 0], axes[0, 1]]
                energy_axes = [axes[1, 0], axes[1, 1]]
                map_axes = None
                cons_axes = [axes[2, 0], axes[2, 1]] if include_cons else None
            else:
                rows = 3 if include_cons else 2
                fig, axes = plt.subplots(
                    rows,
                    3,
                    figsize=(16, 4 + 2 * rows),
                    gridspec_kw={"height_ratios": [2.0, 1.0] + ([1.0] if include_cons else [])},
                )
                chain_axes = [axes[0, 0], axes[0, 1]]
                energy_axes = [axes[1, 0], axes[1, 1]]
                map_axes = [axes[0, 2], axes[1, 2]]
                cons_axes = [axes[2, 0], axes[2, 1]] if include_cons else None

            for ax, chain, title in [
                (chain_axes[0], centroid_chain, "Chain from Centroid"),
                (chain_axes[1], void_chain, "Chain from Void"),
            ]:
                ax.contour(X, Y, Z, levels=12, linewidths=0.8, alpha=0.8)
                if quiver_data is not None:
                    qx, qy, qu, qv = quiver_data
                    ax.quiver(
                        qx,
                        qy,
                        qu,
                        qv,
                        color="black",
                        alpha=0.25,
                        linewidth=0.3,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                    )
                ax.plot(chain[:, 0], chain[:, 1], "-o", markersize=3, linewidth=1.2)
                ax.scatter(chain[0, 0], chain[0, 1], marker="*", s=120, edgecolor="black", zorder=5)
                ax.set_xlabel("z1")
                ax.set_ylabel("z2")
                ax.set_aspect("equal")
                ax.set_title(title)

            # Energy traces
            for ax, energy, title in [
                (energy_axes[0], centroid_energy, "Energies (Centroid Chain)"),
                (energy_axes[1], void_energy, "Energies (Void Chain)"),
            ]:
                steps = np.arange(len(energy["potential"]))
                ax.plot(steps, energy["potential"], label="U(z)", linewidth=1.3)
                if not args.potential_only:
                    ax.plot(steps, energy["kinetic"], label="K(ρ)", linewidth=1.3)
                    ax.plot(steps, energy["hamiltonian"], label="H", linewidth=1.5)
                ax.set_xlabel("Step")
                ax.set_ylabel("Energy")
                ax.set_title(title)
                ax.grid(alpha=0.25)

                # Overlay accept/reject as a secondary axis (0/1)
                accept = energy.get("accept")
                if accept is not None:
                    accept = np.asarray(accept)
                    if accept.shape[0] < steps.shape[0]:
                        accept_steps = steps[1:]
                    else:
                        accept_steps = steps[: accept.shape[0]]
                    ax2 = ax.twinx()
                    # 0/1 accept markers
                    ax2.plot(
                        accept_steps,
                        accept,
                        color="gray",
                        alpha=0.5,
                        linewidth=0.0,
                        marker="o",
                        markersize=2.0,
                        label="accept (0/1)",
                    )
                    # running acceptance rate
                    run_mean = np.cumsum(accept) / np.arange(1, accept.shape[0] + 1)
                    ax2.plot(
                        accept_steps,
                        run_mean,
                        color="black",
                        alpha=0.7,
                        linewidth=1.0,
                        label="accept rate",
                    )
                    ax2.set_ylim(-0.05, 1.05)
                    ax2.set_ylabel("Accept / Rate")
                    ax2.grid(False)
                    ax2.legend(fontsize=8, loc="upper right", frameon=False)

                ax.legend(fontsize=8, loc="best", frameon=False)

            if cons_axes is not None:
                for ax, cons, title in [
                    (cons_axes[0], centroid_cons, "Single-Trajectory H (Centroid)"),
                    (cons_axes[1], void_cons, "Single-Trajectory H (Void)"),
                ]:
                    steps = np.arange(len(cons))
                    ax.plot(steps, cons, linewidth=1.4, color="tab:green")
                    ax.set_xlabel("Leapfrog Step")
                    ax.set_ylabel("H")
                    ax.set_title(title)
                    ax.grid(alpha=0.25)

            if map_axes is not None:
                pot_ax, kin_ax = map_axes
                if args.potential_only:
                    Xm, Ym, Umap = _grid_potential_map(model, sampler, args.bounds, args.grid_size)
                else:
                    Xm, Ym, Umap, Kmap = _grid_energy_maps(
                        model,
                        sampler,
                        args.bounds,
                        args.grid_size,
                        kinetic_mode=args.kinetic_mode,
                        kinetic_mc_samples=args.kinetic_mc_samples,
                    )
                pot = pot_ax.contourf(Xm, Ym, Umap, levels=18, cmap="viridis")
                pot_ax.scatter(centroids[:, 0].cpu(), centroids[:, 1].cpu(), c="gray", s=10, alpha=0.4)
                pot_ax.set_title("Potential U(z) Heatmap")
                pot_ax.set_xlabel("z1")
                pot_ax.set_ylabel("z2")
                pot_ax.set_aspect("equal")
                fig.colorbar(pot, ax=pot_ax, fraction=0.046, pad=0.04)

                if args.potential_only:
                    kin_ax.axis("off")
                else:
                    kin = kin_ax.contourf(Xm, Ym, Kmap, levels=18, cmap="magma")
                    kin_ax.scatter(centroids[:, 0].cpu(), centroids[:, 1].cpu(), c="gray", s=10, alpha=0.4)
                    kinetic_labels = {
                        "fixed_rho": "K(z; fixed ρ)",
                        "trace_ginv": "0.5 tr(G^{-1})",
                        "trace_g": "0.5 tr(G)",
                        "lambda_max_ginv": "0.5 λ_max(G^{-1})",
                        "lambda_min_ginv": "0.5 λ_min(G^{-1})",
                        "expected_mc": "E[K] (MC)",
                    }
                    kin_label = kinetic_labels.get(args.kinetic_mode, "K(z)")
                    kin_ax.set_title(f"Kinetic {kin_label} Heatmap")
                    kin_ax.set_xlabel("z1")
                    kin_ax.set_ylabel("z2")
                    kin_ax.set_aspect("equal")
                    fig.colorbar(kin, ax=kin_ax, fraction=0.046, pad=0.04)
                if cons_axes is not None:
                    extra_ax = axes[2, 2]
                    extra_ax.axis("off")

            metric_fig = None
            if metric_map is not None:
                Xm, Ym, Amap = metric_map
                metric_fig, metric_ax = plt.subplots(1, 1, figsize=(6, 5))
                hm = metric_ax.contourf(Xm, Ym, Amap, levels=18, cmap="plasma")
                metric_ax.scatter(centroids[:, 0].cpu(), centroids[:, 1].cpu(), c="gray", s=10, alpha=0.4)
                if quiver_data is not None:
                    qx, qy, qu, qv = quiver_data
                    metric_ax.quiver(
                        qx,
                        qy,
                        qu,
                        qv,
                        color="black",
                        alpha=0.35,
                        linewidth=0.3,
                        angles="xy",
                        scale_units="xy",
                        scale=1.0,
                    )
                metric_labels = {
                    "logcond": "log10 κ",
                    "cond": "κ",
                    "lambda_max": "λ_max",
                    "lambda_min": "λ_min",
                }
                metric_label = metric_labels.get(args.anisotropy_heatmap, "anisotropy")
                metric_ax.set_title(f"Metric Anisotropy ({metric_label}, {args.anisotropy_mode})")
                metric_ax.set_xlabel("z1")
                metric_ax.set_ylabel("z2")
                metric_ax.set_aspect("equal")
                metric_fig.colorbar(hm, ax=metric_ax, fraction=0.046, pad=0.04)

            fig.tight_layout()
            if out_path:
                out_file = Path(out_path)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_file, dpi=150, bbox_inches="tight")
                print(f"Saved plot to {out_file}")
                if wandb_run is not None:
                    wandb_run.log({"plot/main": wandb.Image(str(out_file))})
                plt.close(fig)
                if metric_fig is not None:
                    metric_fig.tight_layout()
                    metric_out = out_file.with_name(f"{out_file.stem}_metric{out_file.suffix or '.png'}")
                    metric_fig.savefig(metric_out, dpi=150, bbox_inches="tight")
                    print(f"Saved metric panel to {metric_out}")
                    if wandb_run is not None:
                        wandb_run.log({"plot/metric": wandb.Image(str(metric_out))})
                    plt.close(metric_fig)
            else:
                if metric_fig is not None:
                    metric_fig.tight_layout()
                plt.show()


if __name__ == "__main__":
    main()
