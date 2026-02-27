#!/usr/bin/env python3
"""
Sampling Diagnostics for GeometryRHVAE
======================================

Comprehensive sampling, generation, and interpolation diagnostics.
Can be run standalone or called at the end of training runs.

Features:
- FID score computation with multiple samplers
- Multi-starting-point sampling analysis
- Interpolation comparison (Linear, SLERP, Geodesic, Metric-weighted)
- RHMC chain diagnostics
- Sample quality metrics (diversity, coverage, precision/recall)
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Optional, Callable

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

try:
    import wandb
except ImportError:
    wandb = None

try:
    from cleanfid import fid as cleanfid
    HAS_CLEANFID = True
except ImportError:
    HAS_CLEANFID = False

try:
    from scipy.spatial.distance import cdist as scipy_cdist
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path as scipy_shortest_path
except ImportError:
    scipy_cdist = None
    csr_matrix = None
    scipy_shortest_path = None

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from src.models.rhvae_geometry import GeometryRHVAE, GeometryRHVAEConfig
from src.models.samplers.hmc_sampler import (
    RiemannianHMCSampler,
    VolumeElementRiemannianHMCSampler,
)
from src.utils.metric_helpers import load_metric_bundle


# =============================================================================
# Utility Functions
# =============================================================================

def build_output_dir(base: Path | str) -> Path:
    """Create timestamped output directory."""
    base_path = Path(base)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base_path / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_plot(
    fig: plt.Figure,
    path: Path,
    tag: str,
    wandb_run: Optional[Any] = None,
    dpi: int = 150,
) -> None:
    """Save figure and optionally log to WandB."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if wandb_run is not None and wandb is not None:
        wandb_run.log({tag: wandb.Image(fig)})
    plt.close(fig)


def pad_latent(points: torch.Tensor, latent_dim: int) -> torch.Tensor:
    """Pad 2D points to full latent dimension."""
    if latent_dim == points.shape[1]:
        return points
    padded = torch.zeros(points.shape[0], latent_dim, device=points.device, dtype=points.dtype)
    padded[:, : points.shape[1]] = points
    return padded


def run_hmc_chain(
    start_z: torch.Tensor,
    sampler: Any,
    chain_length: int,
    n_lf: int,
    eps_lf: float,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, int]]:
    """Run a single Metropolis-corrected chain from a starting point with error tracking."""
    z = start_z.clone().detach().requires_grad_(True)
    chain = [z.detach().cpu().squeeze().numpy()]
    energies: list[float] = []
    accept_count = 0
    errors = {"linalg_errors": 0, "divergences": 0}

    for _ in range(chain_length):
        try:
            rho = sampler._initialize_momentum(z)
            with torch.no_grad():
                if hasattr(sampler, "_compute_hamiltonian"):
                    H0 = sampler._compute_hamiltonian(z, rho)
                elif hasattr(sampler, "_hamiltonian"):
                    H0 = sampler._hamiltonian(z, rho)
                else:
                    H0 = torch.zeros(1, device=z.device)

            z_prop = z.clone()
            rho_prop = rho.clone()
            use_tempering = (
                (not bool(getattr(sampler, "exact", False)))
                and hasattr(sampler, "_tempering")
                and hasattr(sampler, "beta_zero_sqrt")
            )
            beta_sqrt_old = sampler.beta_zero_sqrt if use_tempering else None

            for k in range(n_lf):
                if hasattr(sampler, "_generalized_leapfrog_step"):
                    z_prop, rho_prop = sampler._generalized_leapfrog_step(z_prop, rho_prop, eps_lf)
                elif hasattr(sampler, "_leapfrog"):
                    z_prop, rho_prop = sampler._leapfrog(z_prop, rho_prop, eps_lf)

                if use_tempering and beta_sqrt_old is not None:
                    beta_sqrt = sampler._tempering(k + 1, n_lf, sampler.beta_zero_sqrt)
                    if torch.is_tensor(beta_sqrt):
                        beta_sqrt = beta_sqrt.to(z.device)
                    else:
                        beta_sqrt = torch.tensor(beta_sqrt, device=z.device)
                    rho_prop = (beta_sqrt_old / beta_sqrt) * rho_prop
                    beta_sqrt_old = beta_sqrt

            with torch.no_grad():
                if hasattr(sampler, "_compute_hamiltonian"):
                    H1 = sampler._compute_hamiltonian(z_prop, rho_prop)
                elif hasattr(sampler, "_hamiltonian"):
                    H1 = sampler._hamiltonian(z_prop, rho_prop)
                else:
                    H1 = torch.zeros(1, device=z.device)

                if torch.isnan(H1).any() or torch.isinf(H1).any():
                    errors["divergences"] += 1
                    # reject step automatically
                    H1 = torch.tensor([float("inf")], device=z.device)
                    
                alpha = torch.exp(-(H1 - H0)).clamp(max=1.0)
                u = torch.rand_like(alpha)
                if (u < alpha).all():
                    z = z_prop.detach().requires_grad_(True)
                    accept_count += 1

                energies.append(H1.mean().item())
                
        except torch.linalg.LinAlgError:
            errors["linalg_errors"] += 1
            # Step rejected natively by exception; retain old z.
            energies.append(float("inf"))

        chain.append(z.detach().cpu().squeeze().numpy())

    acceptance = accept_count / max(chain_length, 1)
    return np.array(chain), np.array(energies), acceptance, errors


# =============================================================================
# Model Loading
# =============================================================================

def load_model_and_centroids(
    model_path: Path | str,
    device: torch.device,
    aggressive_void: bool = False,  # Use aggressive void parameters for sampling
) -> tuple[GeometryRHVAE, torch.Tensor]:
    """Load trained GeometryRHVAE model and centroids."""
    folder = Path(model_path)
    if folder.suffix == ".pt":
        folder = folder.parent
    if not folder.exists():
        raise FileNotFoundError(f"Model path {folder} does not exist")

    centroids, atoms, temperature, regularization, cfg_dict = load_metric_bundle(
        folder / "rhvae_metric.pt"
    )
    cfg_kwargs = dict(cfg_dict) if cfg_dict is not None else {}
    cfg_kwargs.setdefault("input_dim", (centroids.shape[1],))
    cfg_kwargs.setdefault("latent_dim", centroids.shape[1])
    cfg_kwargs.setdefault("temperature", float(temperature))
    cfg_kwargs.setdefault("regularization", float(regularization))
    if not isinstance(cfg_kwargs["input_dim"], tuple):
        cfg_kwargs["input_dim"] = tuple(cfg_kwargs["input_dim"])
    
    # Override with aggressive void parameters for better sampling
    if aggressive_void:
        print("  Using AGGRESSIVE void parameters for sampling:")
        cfg_kwargs["void_decay_scale"] = 1.5
        cfg_kwargs["void_decay_power"] = 3.0
        cfg_kwargs["radial_stretch"] = 15.0
        cfg_kwargs["temperature"] = 0.5
        temperature = 0.5  # Also update for model
        print(f"    void_decay_scale=1.5, void_decay_power=3.0, radial_stretch=15.0, temperature=0.5")

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
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys in state_dict")
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected keys in state_dict")
    else:
        raise FileNotFoundError(
            f"Model weights not found: {model_pt}\n"
            "The decoder will produce noise without trained weights.\n"
            "Please ensure the training run completed and saved the model."
        )

    model.eval()
    return model, centroids.to(device)


# =============================================================================
# Sampler Factory
# =============================================================================

SAMPLER_REGISTRY = {
    "riemannian": RiemannianHMCSampler,
    "volume_riemannian": VolumeElementRiemannianHMCSampler,
}


def create_sampler(
    model: GeometryRHVAE,
    sampler_name: str,
    mcmc_steps: int = 100,
    n_lf: int = 15,
    eps_lf: float = 0.03,
    beta_zero: float = 1.0,
    use_dual_metric: bool = False,
    adaptive_dual_step: bool = False,
    adaptive_max_dual_displacement: float = 0.07,
    adaptive_min_step_scale: float = 0.05,
    volume_power: float = 2.0,
    radial_prior_weight: float = 0.1,
    momentum_persist: float = 0.0,
    fp_steps: int = 15,
    fp_damping: float = 0.72,
    use_physics_sampler_defaults: bool = False,
):
    """Create a sampler by name. If use_physics_sampler_defaults is True, overrides 
    hyper-parameters with theoretically rigorously derived bounds from the geometry base."""
    # Deduce physics sampler equivalents if requested
    if use_physics_sampler_defaults and hasattr(model, 'temperature'):
        import math
        T = float(model.temperature.detach().cpu().item())
        d = float(model.latent_dim)
        sigma = T / math.sqrt(2.0)
        
        eps_lf = T / 10.0
        n_lf = 10
        momentum_persist = sigma
        volume_power = d
        adaptive_min_step_scale = eps_lf
        adaptive_max_dual_displacement = eps_lf
        
        print(f"  [Physics Sampler Config Applied: eps_lf={eps_lf:.4f}, "
              f"n_lf={n_lf}, momentum_persist={momentum_persist:.4f}, volume_power={volume_power}]")

    if sampler_name == "gaussian":
        return None  # Will use standard Gaussian sampling
    
    sampler_cls = SAMPLER_REGISTRY.get(sampler_name)
    if sampler_cls is None:
        raise ValueError(f"Unknown sampler: {sampler_name}. Available: {list(SAMPLER_REGISTRY.keys())}")
    
    if sampler_name == "riemannian":
        return sampler_cls(
            model,
            mcmc_steps_nbr=mcmc_steps,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
            include_volume_grad=True,
        )
    elif sampler_name == "volume_riemannian":
        sampler = sampler_cls(
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
    else:
        return sampler_cls(
            model,
            mcmc_steps_nbr=mcmc_steps,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
        )


def sample_latents(
    model: GeometryRHVAE,
    sampler_name: str,
    n_samples: int,
    device: torch.device,
    mcmc_steps: int = 100,
    n_lf: int = 15,
    eps_lf: float = 0.03,
    beta_zero: float = 1.0,
    init_std: float = 1.0,
    use_dual_metric: bool = False,
    adaptive_dual_step: bool = False,
    adaptive_max_dual_displacement: float = 0.07,
    adaptive_min_step_scale: float = 0.05,
    volume_power: float = 2.0,
    radial_prior_weight: float = 0.1,
    momentum_persist: float = 0.0,
    fp_steps: int = 15,
    fp_damping: float = 0.72,
    use_physics_sampler_defaults: bool = False,
) -> tuple[torch.Tensor, float]:
    """
    Sample latent codes using specified sampler.
    
    Returns:
        samples: [n_samples, latent_dim]
        acceptance_rate: float (0.0 for gaussian)
    """
    # Ensure model is on correct device
    model = model.to(device)
    
    if sampler_name == "gaussian":
        # Standard Gaussian prior (no metric)
        samples = torch.randn(n_samples, model.latent_dim, device=device) * init_std
        return samples, 0.0
    
    sampler = create_sampler(
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
        use_physics_sampler_defaults=use_physics_sampler_defaults,
    )
    
    # Ensure sampler's device attribute is correct
    sampler.device = device
    
    if hasattr(sampler, "sample"):
        if sampler_name == "riemannian":
            samples = sampler.sample(n_samples, init_std=init_std)
        else:
            samples = sampler.sample(n_samples)
    else:
        samples = sampler.sample_prior(n_samples)
    
    acceptance_rate = getattr(sampler, "last_acceptance_rate", 0.0)
    return samples.to(device), float(acceptance_rate)


def decode_latents(
    model: GeometryRHVAE,
    z: torch.Tensor,
    batch_size: int = 64,
) -> torch.Tensor:
    """Decode latent codes to images."""
    model.eval()
    all_recons = []
    
    with torch.no_grad():
        for i in range(0, z.shape[0], batch_size):
            z_batch = z[i:i + batch_size]
            recon = model.decoder(z_batch)["reconstruction"]
            all_recons.append(recon)
    
    return torch.cat(all_recons, dim=0)


# =============================================================================
# FID Computation
# =============================================================================

def compute_fid_score(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """
    Compute FID score between real and generated images.
    
    Uses clean-fid if available, otherwise falls back to simple feature-based comparison.
    
    Args:
        real_images: [N, C, H, W] or [N, D] flattened
        generated_images: [M, C, H, W] or [M, D] flattened
    
    Returns:
        FID score (lower is better)
    """
    # Ensure all tensors are on the same device
    real_images = real_images.to(device)
    generated_images = generated_images.to(device)
    
    # Ensure images are in [N, C, H, W] format
    if real_images.dim() == 2:
        # Assume 64x64 grayscale images
        side = int(np.sqrt(real_images.shape[1]))
        real_images = real_images.view(-1, 1, side, side)
    if generated_images.dim() == 2:
        side = int(np.sqrt(generated_images.shape[1]))
        generated_images = generated_images.view(-1, 1, side, side)
    
    # Convert to 3-channel for FID computation
    if real_images.shape[1] == 1:
        real_images = real_images.repeat(1, 3, 1, 1)
    if generated_images.shape[1] == 1:
        generated_images = generated_images.repeat(1, 3, 1, 1)
    
    # Resize to 299x299 for Inception (required by clean-fid)
    real_resized = F.interpolate(real_images, size=(299, 299), mode="bilinear", align_corners=False)
    gen_resized = F.interpolate(generated_images, size=(299, 299), mode="bilinear", align_corners=False)
    
    if HAS_CLEANFID:
        # Use clean-fid for accurate FID computation
        import tempfile
        import os
        from PIL import Image
        
        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = os.path.join(tmpdir, "real")
            gen_dir = os.path.join(tmpdir, "gen")
            os.makedirs(real_dir)
            os.makedirs(gen_dir)
            
            # Save images
            for i, img in enumerate(real_resized[:min(2048, len(real_resized))]):
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(img_np).save(os.path.join(real_dir, f"{i:05d}.png"))
            
            for i, img in enumerate(gen_resized[:min(2048, len(gen_resized))]):
                img_np = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(img_np).save(os.path.join(gen_dir, f"{i:05d}.png"))
            
            score = cleanfid.compute_fid(real_dir, gen_dir, device=device)
            return float(score)
    else:
        # Fallback: compute simple feature-based distance
        # Use downsampled images to avoid OOM with covariance computation
        # Downsample to 32x32 for memory efficiency
        real_small = F.interpolate(real_images, size=(32, 32), mode="bilinear", align_corners=False)
        gen_small = F.interpolate(generated_images, size=(32, 32), mode="bilinear", align_corners=False)
        
        real_flat = real_small.view(real_small.shape[0], -1).float()
        gen_flat = gen_small.view(gen_small.shape[0], -1).float()
        
        mu_real = real_flat.mean(dim=0)
        mu_gen = gen_flat.mean(dim=0)
        
        # Mean squared difference as primary metric
        diff = mu_real - mu_gen
        mean_diff = torch.dot(diff, diff).item()
        
        # Also compute variance difference for a rough FID approximation
        var_real = real_flat.var(dim=0)
        var_gen = gen_flat.var(dim=0)
        var_diff = ((var_real.sqrt() - var_gen.sqrt()) ** 2).sum().item()
        
        fid_approx = mean_diff + var_diff
        
        print("[WARNING] clean-fid not installed. Using approximate FID metric (mean+var diff).")
        return fid_approx


def run_fid_evaluation(
    model: GeometryRHVAE,
    real_data: torch.Tensor,
    device: torch.device,
    samplers: list[str] = ["gaussian", "riemannian", "geodesic", "volume"],
    n_samples: int = 2000,
    mcmc_steps: int = 100,
    n_lf: int = 15,
    eps_lf: float = 0.03,
    use_dual_metric: bool = False,
    adaptive_dual_step: bool = False,
    adaptive_max_dual_displacement: float = 0.07,
    adaptive_min_step_scale: float = 0.05,
    volume_power: float = 2.0,
    radial_prior_weight: float = 0.1,
    momentum_persist: float = 0.0,
    fp_steps: int = 15,
    fp_damping: float = 0.72,
    use_physics_sampler_defaults: bool = False,
    out_dir: Optional[Path] = None,
    wandb_run: Optional[Any] = None,
) -> dict[str, dict[str, float]]:
    """
    Run FID evaluation for multiple samplers.
    
    Returns:
        Dictionary mapping sampler names to metrics (fid, acceptance_rate)
    """
    results = {}
    
    for sampler_name in samplers:
        print(f"\n=== FID Evaluation: {sampler_name} ===")
        
        try:
            # Sample latents
            z_samples, acc_rate = sample_latents(
                model, sampler_name, n_samples, device,
                mcmc_steps=mcmc_steps, n_lf=n_lf, eps_lf=eps_lf,
                use_dual_metric=use_dual_metric,
                adaptive_dual_step=adaptive_dual_step,
                adaptive_max_dual_displacement=adaptive_max_dual_displacement,
                adaptive_min_step_scale=adaptive_min_step_scale,
                volume_power=volume_power,
                radial_prior_weight=radial_prior_weight,
                momentum_persist=momentum_persist,
                fp_steps=fp_steps,
                fp_damping=fp_damping,
                use_physics_sampler_defaults=use_physics_sampler_defaults,
            )
            
            # Decode to images
            generated = decode_latents(model, z_samples)
            
            # Compute FID
            fid_score = compute_fid_score(real_data, generated, device)
            
            results[sampler_name] = {
                "fid": fid_score,
                "acceptance_rate": acc_rate,
            }
            
            print(f"  FID: {fid_score:.2f}, Acceptance: {acc_rate:.3f}")
            
            # Save sample grid
            if out_dir is not None:
                save_sample_grid(
                    generated[:64],
                    out_dir / f"generated_samples_{sampler_name}.png",
                    f"sampling/samples_{sampler_name}",
                    wandb_run,
                    title=f"Generated Samples ({sampler_name})",
                )
        
        except Exception as e:
            print(f"  Error: {e}")
            results[sampler_name] = {"fid": float("inf"), "acceptance_rate": 0.0, "error": str(e)}
    
    # Plot FID comparison
    if out_dir is not None:
        plot_fid_comparison(results, out_dir, wandb_run)
    
    return results


def save_sample_grid(
    samples: torch.Tensor,
    path: Path,
    tag: str,
    wandb_run: Optional[Any] = None,
    title: str = "Generated Samples",
    nrow: int = 8,
) -> None:
    """Save a grid of sample images."""
    n_samples = samples.shape[0]
    ncol = min(nrow, n_samples)
    nrow_actual = (n_samples + ncol - 1) // ncol
    
    # Reshape if flattened
    if samples.dim() == 2:
        side = int(np.sqrt(samples.shape[1]))
        samples = samples.view(-1, 1, side, side)
    
    fig, axes = plt.subplots(nrow_actual, ncol, figsize=(ncol * 1.5, nrow_actual * 1.5))
    if nrow_actual == 1:
        axes = axes.reshape(1, -1)
    if ncol == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(nrow_actual):
        for j in range(ncol):
            idx = i * ncol + j
            ax = axes[i, j]
            ax.axis("off")
            if idx < n_samples:
                img = samples[idx].squeeze().cpu().numpy()
                ax.imshow(np.clip(img, 0, 1), cmap="gray")
    
    fig.suptitle(title)
    plt.tight_layout()
    save_plot(fig, path, tag, wandb_run)


def plot_fid_comparison(
    results: dict[str, dict[str, float]],
    out_dir: Path,
    wandb_run: Optional[Any] = None,
) -> None:
    """Plot FID comparison bar chart."""
    samplers = list(results.keys())
    fids = [results[s].get("fid", float("inf")) for s in samplers]
    
    # Filter out infinite values for plotting
    valid_mask = [f < float("inf") for f in fids]
    valid_samplers = [s for s, v in zip(samplers, valid_mask) if v]
    valid_fids = [f for f, v in zip(fids, valid_mask) if v]
    
    if not valid_fids:
        print("No valid FID scores to plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(valid_samplers, valid_fids, color="steelblue", edgecolor="black")
    
    # Add value labels
    for bar, fid in zip(bars, valid_fids):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{fid:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    ax.set_xlabel("Sampler")
    ax.set_ylabel("FID Score (lower is better)")
    ax.set_title("FID Comparison Across Samplers")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, out_dir / "fid_comparison.png", "sampling/fid_comparison", wandb_run)


# =============================================================================
# Multi-Starting-Point Sampling
# =============================================================================

def multi_start_sampling(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    device: torch.device,
    n_starts: int = 16,
    n_samples_per_start: int = 50,
    sampler_name: str = "volume",
    mcmc_steps: int = 50,
    n_lf: int = 10,
    eps_lf: float = 0.03,
    use_dual_metric: bool = False,
    adaptive_dual_step: bool = False,
    adaptive_max_dual_displacement: float = 0.07,
    adaptive_min_step_scale: float = 0.05,
    volume_power: float = 2.0,
    radial_prior_weight: float = 0.1,
    momentum_persist: float = 0.0,
    fp_steps: int = 15,
    fp_damping: float = 0.72,
    use_physics_sampler_defaults: bool = False,
    out_dir: Optional[Path] = None,
    wandb_run: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Sample from multiple starting points and analyze coverage/acceptance.
    
    Starts from:
    - Near centroids (on-manifold)
    - Random Gaussian (off-manifold)
    - Grid across latent space
    """
    results = {
        "centroid_starts": [],
        "gaussian_starts": [],
        "grid_starts": [],
    }
    
    # 1. Near-centroid starts
    print("\n=== Multi-Start Sampling: Near Centroids ===")
    n_centroid_starts = min(n_starts // 2, centroids.shape[0])
    centroid_indices = torch.randperm(centroids.shape[0])[:n_centroid_starts]
    
    for idx in centroid_indices:
        start_point = centroids[idx:idx+1] + 0.1 * torch.randn(1, model.latent_dim, device=device)
        sampler = create_sampler(
            model,
            sampler_name,
            mcmc_steps,
            n_lf,
            eps_lf,
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
        
        if sampler is not None:
            chain, energies, acceptance, errors = run_hmc_chain(
                start_point, sampler, n_samples_per_start, n_lf, eps_lf
            )
            samples_tensor = torch.as_tensor(chain)
            results["centroid_starts"].append({
                "start": start_point.cpu(),
                "samples": samples_tensor,
                "acceptance": acceptance,
                "errors": errors,
            })
    
    # 2. Random Gaussian starts
    print("=== Multi-Start Sampling: Gaussian Off-Manifold ===")
    n_gaussian_starts = n_starts // 4
    
    for i in range(n_gaussian_starts):
        start_point = torch.randn(1, model.latent_dim, device=device) * 2.0
        sampler = create_sampler(
            model,
            sampler_name,
            mcmc_steps,
            n_lf,
            eps_lf,
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
        
        if sampler is not None:
            chain, energies, acceptance, errors = run_hmc_chain(
                start_point, sampler, n_samples_per_start, n_lf, eps_lf
            )
            samples_tensor = torch.as_tensor(chain)
            results["gaussian_starts"].append({
                "start": start_point.cpu(),
                "samples": samples_tensor,
                "acceptance": acceptance,
                "errors": errors,
            })
    
    # 3. Grid starts
    print("=== Multi-Start Sampling: Grid ===")
    c_min = centroids[:, :2].min(dim=0).values.cpu()
    c_max = centroids[:, :2].max(dim=0).values.cpu()
    margin = 0.5 * (c_max - c_min)
    
    grid_size = int(np.sqrt(n_starts // 4)) + 1
    x_range = torch.linspace(c_min[0] - margin[0], c_max[0] + margin[0], grid_size)
    y_range = torch.linspace(c_min[1] - margin[1], c_max[1] + margin[1], grid_size)
    
    for x in x_range:
        for y in y_range:
            start_2d = torch.tensor([[x, y]], device=device)
            start_point = pad_latent(start_2d, model.latent_dim)
            sampler = create_sampler(
                model,
                sampler_name,
                mcmc_steps // 2,
                max(1, n_lf // 2),
                eps_lf,
                use_dual_metric=use_dual_metric,
                adaptive_dual_step=adaptive_dual_step,
                adaptive_max_dual_displacement=adaptive_max_dual_displacement,
                adaptive_min_step_scale=adaptive_min_step_scale,
                volume_power=volume_power,
                radial_prior_weight=radial_prior_weight,
                momentum_persist=momentum_persist,
                fp_steps=fp_steps,
                fp_damping=fp_damping,
                use_physics_sampler_defaults=use_physics_sampler_defaults,
            )
            
            if sampler is not None:
                chain, energies, acceptance, errors = run_hmc_chain(
                    start_point, sampler, n_samples_per_start // 2, max(1, n_lf // 2), eps_lf
                )
                samples_tensor = torch.as_tensor(chain)
                results["grid_starts"].append({
                    "start": start_point.cpu(),
                    "samples": samples_tensor,
                    "acceptance": acceptance,
                    "errors": errors,
                })
    
    # Visualize
    if out_dir is not None:
        plot_multi_start_results(results, centroids, out_dir, wandb_run)
    
    return results


def plot_multi_start_results(
    results: dict[str, list],
    centroids: torch.Tensor,
    out_dir: Path,
    wandb_run: Optional[Any] = None,
) -> None:
    """Visualize multi-start sampling results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    c_np = centroids[:, :2].cpu().numpy()
    
    # Plot 1: Centroid starts
    ax = axes[0]
    ax.scatter(c_np[:, 0], c_np[:, 1], c="gray", alpha=0.3, s=20, label="Centroids")
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results["centroid_starts"])))
    for i, data in enumerate(results["centroid_starts"]):
        samples = data["samples"][:, :2].numpy()
        ax.plot(samples[:, 0], samples[:, 1], "-", color=colors[i], alpha=0.7, linewidth=0.8)
        ax.scatter(data["start"][0, 0], data["start"][0, 1], marker="*", s=100, color=colors[i], edgecolor="black")
    
    ax.set_title("Chains from Centroids")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_aspect("equal")
    
    # Plot 2: Gaussian starts
    ax = axes[1]
    ax.scatter(c_np[:, 0], c_np[:, 1], c="gray", alpha=0.3, s=20, label="Centroids")
    
    colors = plt.cm.Set1(np.linspace(0, 1, max(1, len(results["gaussian_starts"]))))
    for i, data in enumerate(results["gaussian_starts"]):
        samples = data["samples"][:, :2].numpy()
        ax.plot(samples[:, 0], samples[:, 1], "-", color=colors[i], alpha=0.7, linewidth=0.8)
        ax.scatter(data["start"][0, 0], data["start"][0, 1], marker="*", s=100, color=colors[i], edgecolor="black")
    
    ax.set_title("Chains from Off-Manifold")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_aspect("equal")
    
    # Plot 3: Grid coverage
    ax = axes[2]
    ax.scatter(c_np[:, 0], c_np[:, 1], c="gray", alpha=0.3, s=20, label="Centroids")
    
    # Collect all grid samples
    all_grid_samples = []
    for data in results["grid_starts"]:
        all_grid_samples.append(data["samples"][:, :2].numpy())
    
    if all_grid_samples:
        all_grid_samples = np.concatenate(all_grid_samples, axis=0)
        ax.scatter(all_grid_samples[:, 0], all_grid_samples[:, 1], c="blue", alpha=0.1, s=5)
    
    ax.set_title("Grid Start Coverage")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_aspect("equal")
    
    plt.tight_layout()
    save_plot(fig, out_dir / "multi_start_sampling.png", "sampling/multi_start", wandb_run)


# =============================================================================
# Interpolation Methods
# =============================================================================

def linear_interpolation(z1: torch.Tensor, z2: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
    """Standard linear interpolation."""
    t = torch.linspace(0, 1, n_steps, device=z1.device).unsqueeze(1)
    return z1 + t * (z2 - z1)


def slerp_interpolation(z1: torch.Tensor, z2: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
    """Spherical linear interpolation (SLERP)."""
    # Normalize to unit sphere
    z1_norm = z1 / (z1.norm() + 1e-8)
    z2_norm = z2 / (z2.norm() + 1e-8)
    
    # Compute angle
    dot = torch.clamp(torch.sum(z1_norm * z2_norm), -1.0, 1.0)
    theta = torch.acos(dot)
    
    t = torch.linspace(0, 1, n_steps, device=z1.device)
    
    if theta.abs() < 1e-6:
        # Nearly parallel - fall back to linear
        return linear_interpolation(z1, z2, n_steps)
    
    # SLERP formula
    sin_theta = torch.sin(theta)
    results = []
    for ti in t:
        w1 = torch.sin((1 - ti) * theta) / sin_theta
        w2 = torch.sin(ti * theta) / sin_theta
        interp = w1 * z1_norm + w2 * z2_norm
        # Scale back to original magnitudes
        scale = (1 - ti) * z1.norm() + ti * z2.norm()
        results.append(interp * scale)
    
    return torch.stack(results)


def get_graph_initialization(
    z1: torch.Tensor,
    z2: torch.Tensor,
    centroids: torch.Tensor,
    n_steps: int,
    k: int = 10,
) -> torch.Tensor:
    """Find approximate shortest path over a centroid k-NN graph."""
    if scipy_cdist is None or scipy_shortest_path is None or csr_matrix is None:
        t = torch.linspace(0, 1, n_steps, device=z1.device).unsqueeze(1)
        return z1 + t * (z2 - z1)

    if centroids is None or centroids.numel() == 0:
        t = torch.linspace(0, 1, n_steps, device=z1.device).unsqueeze(1)
        return z1 + t * (z2 - z1)

    c_np = centroids.detach().cpu().numpy()
    z1_np = z1.detach().cpu().numpy().reshape(1, -1)
    z2_np = z2.detach().cpu().numpy().reshape(1, -1)

    dists_start = scipy_cdist(z1_np, c_np)[0]
    dists_end = scipy_cdist(z2_np, c_np)[0]
    start_idx = int(np.argmin(dists_start))
    end_idx = int(np.argmin(dists_end))

    if start_idx == end_idx:
        t = torch.linspace(0, 1, n_steps, device=z1.device).unsqueeze(1)
        return z1 + t * (z2 - z1)

    dists = scipy_cdist(c_np, c_np)
    n_points = dists.shape[0]
    if n_points <= 1:
        t = torch.linspace(0, 1, n_steps, device=z1.device).unsqueeze(1)
        return z1 + t * (z2 - z1)

    k = min(k, n_points - 1)
    adj = np.zeros((n_points, n_points), dtype=float)
    for i in range(n_points):
        nn_idx = np.argsort(dists[i])[1 : k + 1]
        adj[i, nn_idx] = dists[i, nn_idx]
        adj[nn_idx, i] = dists[i, nn_idx]

    graph = csr_matrix(adj)
    _, predecessors = scipy_shortest_path(
        graph,
        directed=False,
        return_predecessors=True,
        indices=start_idx,
    )

    path_indices = [end_idx]
    curr = end_idx
    while curr != start_idx:
        curr = predecessors[curr]
        if curr == -9999 or curr < 0:
            t = torch.linspace(0, 1, n_steps, device=z1.device).unsqueeze(1)
            return z1 + t * (z2 - z1)
        path_indices.append(curr)
    path_indices = path_indices[::-1]

    raw_path = c_np[path_indices]
    full_raw = np.concatenate([z1_np, raw_path, z2_np], axis=0)

    seg_dists = np.linalg.norm(full_raw[1:] - full_raw[:-1], axis=1)
    cum_dist = np.insert(np.cumsum(seg_dists), 0, 0.0)
    if cum_dist[-1] == 0:
        return z1.unsqueeze(0).repeat(n_steps, 1)

    target_dists = np.linspace(0, cum_dist[-1], n_steps)
    resampled = np.zeros((n_steps, full_raw.shape[1]))
    for d in range(full_raw.shape[1]):
        resampled[:, d] = np.interp(target_dists, cum_dist, full_raw[:, d])

    return torch.tensor(resampled, device=z1.device, dtype=z1.dtype)


def geodesic_interpolation(
    model: GeometryRHVAE,
    z1: torch.Tensor,
    z2: torch.Tensor,
    centroids: Optional[torch.Tensor] = None,
    n_steps: int = 64,
    iterations: int = 2000,
    lr: float = 0.01,
    wall_strength: float = 0,
) -> torch.Tensor:
    """
    Energy-minimizing geodesic interpolation using the learned metric.
    Minimizes integral of z'^T G(z) z' along the path by default.

    If wall_strength > 0, the objective switches to a log-det potential
    U(z) = 0.5 * logdet(G) (equivalently -0.5 * logdet(G_inv)),
    with a smoothness term.
    If centroids are provided, a graph-based initialization is used.
    """
    device = z1.device
    z1 = z1.reshape(-1)
    z2 = z2.reshape(-1)
    
    # Initialize with graph-based path when using attraction.
    if centroids is not None and wall_strength > 0:
        init_path = get_graph_initialization(z1, z2, centroids, n_steps)
    else:
        t = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)
        init_path = z1 + t * (z2 - z1)
    
    # Optimize inner points
    inner = init_path[1:-1].clone().detach().requires_grad_(True)
    
    # Use Adam with scheduling for better convergence
    optimizer = torch.optim.Adam([inner], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=lr * 0.01)
    
    best_energy = float('inf')
    best_inner = inner.clone().detach()
    
    for i in range(iterations):
        # Reparameterize to keep points equidistant and avoid path collapse.
        if i % 20 == 0 and i > 0:
            with torch.no_grad():
                full_np = torch.cat([z1.unsqueeze(0), inner, z2.unsqueeze(0)], dim=0).cpu().numpy()
                seg_dists = np.linalg.norm(full_np[1:] - full_np[:-1], axis=1)
                cum_dist = np.concatenate(([0.0], np.cumsum(seg_dists)))
                total_len = cum_dist[-1]
                if total_len > 0:
                    target_dists = np.linspace(0, total_len, n_steps)
                    new_full = np.zeros_like(full_np)
                    for d in range(full_np.shape[1]):
                        new_full[:, d] = np.interp(target_dists, cum_dist, full_np[:, d])
                    new_inner = torch.tensor(new_full[1:-1], device=device, dtype=inner.dtype)
                    inner.data.copy_(new_inner)

        optimizer.zero_grad()
        full = torch.cat([z1.unsqueeze(0), inner, z2.unsqueeze(0)], dim=0)
        deltas = full[1:] - full[:-1]
        mids = 0.5 * (full[1:] + full[:-1])

        # Use G for geodesic energy: G is small on manifold, large in void.
        G_mid = model.G(mids)

        if wall_strength > 0.0:
            # Encourage high det(G_inv) (manifold) via logdet(G).
            # logdet(G) = -logdet(G_inv) so minimizing +0.5 logdet(G) does this.
            _, logdet_G = torch.linalg.slogdet(G_mid)
            potential_energy = 0.5 * logdet_G.mean()
            smoothness_energy = (deltas.norm(dim=1) ** 2).mean()
            energy = 100.0 * smoothness_energy + wall_strength * potential_energy
        else:
            # Standard Riemannian energy: sum of v^T G v.
            energy = torch.einsum("ni,nij,nj->n", deltas, G_mid, deltas).sum()

        # Curvature regularization (straightness).
        if inner.shape[0] > 1:
            second_deriv = inner[2:] - 2 * inner[1:-1] + inner[:-2]
            reg = 0.1 * (second_deriv ** 2).mean()
            total_loss = energy + reg
        else:
            total_loss = energy
        
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([inner], max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Track best solution
        with torch.no_grad():
            if energy.item() < best_energy:
                best_energy = energy.item()
                best_inner = inner.clone().detach()
    
    with torch.no_grad():
        full = torch.cat([z1.unsqueeze(0), best_inner, z2.unsqueeze(0)], dim=0)
    
    return full


def metric_weighted_interpolation(
    model: GeometryRHVAE,
    z1: torch.Tensor,
    z2: torch.Tensor,
    n_steps: int = 10,
) -> torch.Tensor:
    """
    Linear interpolation with metric-based step weighting.
    Takes smaller steps in high-curvature regions.
    """
    device = z1.device
    z1 = z1.reshape(-1)
    z2 = z2.reshape(-1)
    
    # First pass: compute metric along linear path to estimate curvature
    t_uniform = torch.linspace(0, 1, n_steps * 4, device=device).unsqueeze(1)
    linear_path = z1 + t_uniform * (z2 - z1)
    
    with torch.no_grad():
        # Use G_inv instead of G: curvature ∝ sqrt(det(G_inv))
        G_inv = model.G_inv(linear_path)
        # Use log-determinant as curvature proxy
        _, logdet_inv = torch.linalg.slogdet(G_inv)
        curvature = torch.exp(logdet_inv / 2)  # sqrt(det(G_inv))
    
    # Compute cumulative arc length weighted by curvature
    curvature = curvature.cpu().numpy()
    cumsum = np.cumsum(curvature)
    cumsum = cumsum / cumsum[-1]  # Normalize to [0, 1]
    
    # Find t values for uniform spacing in weighted space
    t_weighted = np.interp(np.linspace(0, 1, n_steps), cumsum, np.linspace(0, 1, len(cumsum)))
    t_weighted = torch.tensor(t_weighted, device=device, dtype=z1.dtype).unsqueeze(1)
    
    return z1 + t_weighted * (z2 - z1)


def run_interpolation_comparison(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    device: torch.device,
    n_pairs: int = 4,
    n_steps: int = 10,
    out_dir: Optional[Path] = None,
    wandb_run: Optional[Any] = None,
) -> dict[str, list[dict]]:
    """
    Compare different interpolation methods.
    """
    results = {
        "linear": [],
        "slerp": [],
        "geodesic": [],
        "metric_weighted": [],
    }
    
    # Select random pairs of centroids
    n_centroids = centroids.shape[0]
    pairs = []
    for _ in range(n_pairs):
        i, j = torch.randperm(n_centroids)[:2].tolist()
        pairs.append((centroids[i], centroids[j]))
    
    print("\n=== Interpolation Comparison ===")

    n_high_res = 100
    
    for pair_idx, (z1, z2) in enumerate(pairs):
        print(f"  Pair {pair_idx + 1}/{n_pairs}")
        
        # Linear
        linear_path = linear_interpolation(z1, z2, n_high_res)
        results["linear"].append({"path": linear_path.cpu(), "pair_idx": pair_idx})
        
        # SLERP
        slerp_path = slerp_interpolation(z1, z2, n_high_res)
        results["slerp"].append({"path": slerp_path.cpu(), "pair_idx": pair_idx})
        
        # Geodesic
        geodesic_path = geodesic_interpolation(
            model,
            z1,
            z2,
            centroids=centroids,
            n_steps=n_high_res,
            iterations=1000,
            wall_strength=10.0,
        )
        results["geodesic"].append({"path": geodesic_path.cpu(), "pair_idx": pair_idx})
        
        # Metric-weighted
        metric_path = metric_weighted_interpolation(model, z1, z2, n_high_res)
        results["metric_weighted"].append({"path": metric_path.cpu(), "pair_idx": pair_idx})
    
    # Decode and visualize
    if out_dir is not None:
        plot_interpolation_comparison(model, results, centroids, out_dir, wandb_run)
    
    return results


def plot_interpolation_comparison(
    model: GeometryRHVAE,
    results: dict[str, list[dict]],
    centroids: torch.Tensor,
    out_dir: Path,
    wandb_run: Optional[Any] = None,
) -> None:
    """Visualize interpolation comparison."""
    methods = list(results.keys())
    n_pairs = len(results[methods[0]])
    
    # Plot 1: Paths in latent space
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]
    
    c_np = centroids[:, :2].cpu().numpy()
    colors = plt.cm.tab10(np.linspace(0, 1, n_pairs))
    
    for ax, method in zip(axes, methods):
        ax.scatter(c_np[:, 0], c_np[:, 1], c="gray", alpha=0.3, s=20)
        
        for i, data in enumerate(results[method]):
            path = data["path"][:, :2].numpy()
            ax.plot(path[:, 0], path[:, 1], "-o", color=colors[i], markersize=3, linewidth=1.5)
        
        ax.set_title(method.replace("_", " ").title())
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_aspect("equal")
    
    plt.tight_layout()
    save_plot(fig, out_dir / "interpolation_paths.png", "sampling/interpolation_paths", wandb_run)
    
    # Plot 2: Decoded image sequences
    device = next(model.parameters()).device
    
    for method in methods:
        fig, axes = plt.subplots(n_pairs, 10, figsize=(15, n_pairs * 1.5))
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        
        for pair_idx, data in enumerate(results[method]):
            path = data["path"].to(device)
            
            # Subsample to 10 frames
            indices = torch.linspace(0, path.shape[0] - 1, 10).long()
            path_sub = path[indices]
            
            # Decode
            with torch.no_grad():
                decoded = model.decoder(path_sub)["reconstruction"]
            
            for frame_idx in range(10):
                ax = axes[pair_idx, frame_idx]
                img = decoded[frame_idx].cpu()
                if img.dim() == 1:
                    side = int(np.sqrt(img.shape[0]))
                    img = img.view(side, side)
                else:
                    img = img.squeeze()
                ax.imshow(np.clip(img.numpy(), 0, 1), cmap="gray")
                ax.axis("off")
        
        fig.suptitle(f"Interpolation: {method.replace('_', ' ').title()}")
        plt.tight_layout()
        save_plot(
            fig,
            out_dir / f"interpolation_{method}.png",
            f"sampling/interpolation_{method}",
            wandb_run,
        )


# =============================================================================
# Quality Metrics
# =============================================================================

def compute_diversity_score(samples: torch.Tensor) -> float:
    """
    Compute diversity as average pairwise distance in latent space.
    """
    n = samples.shape[0]
    if n < 2:
        return 0.0
    
    # Compute pairwise distances
    dists = torch.cdist(samples, samples)
    
    # Average of upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(n, n, device=samples.device), diagonal=1).bool()
    avg_dist = dists[mask].mean().item()
    
    return avg_dist


def compute_coverage_score(
    samples: torch.Tensor,
    centroids: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Compute coverage: fraction of centroids that have at least one sample nearby.
    """
    # For each centroid, find distance to nearest sample
    dists = torch.cdist(centroids, samples)
    min_dists = dists.min(dim=1).values
    
    # Use k-th percentile of centroid spread as threshold
    centroid_dists = torch.cdist(centroids, centroids)
    threshold = torch.kthvalue(centroid_dists.view(-1), min(k * centroids.shape[0], centroid_dists.numel())).values
    
    # Fraction of centroids covered
    covered = (min_dists < threshold).float().mean().item()
    
    return covered


def compute_precision_recall(
    real_samples: torch.Tensor,
    generated_samples: torch.Tensor,
    k: int = 5,
) -> tuple[float, float]:
    """
    Compute precision and recall using k-NN manifolds.
    
    Precision: fraction of generated samples that fall within the real manifold
    Recall: fraction of real samples that are covered by generated samples
    """
    # Build k-NN radius for real samples
    real_dists = torch.cdist(real_samples, real_samples)
    # k-th nearest neighbor distance for each real sample
    real_radii = torch.topk(real_dists, k + 1, dim=1, largest=False).values[:, -1]
    
    # Build k-NN radius for generated samples
    gen_dists = torch.cdist(generated_samples, generated_samples)
    gen_radii = torch.topk(gen_dists, k + 1, dim=1, largest=False).values[:, -1]
    
    # Precision: generated samples within real manifold
    cross_dists_gr = torch.cdist(generated_samples, real_samples)
    min_dists_gr = cross_dists_gr.min(dim=1).values
    precision = (min_dists_gr < real_radii.mean()).float().mean().item()
    
    # Recall: real samples covered by generated manifold
    cross_dists_rg = torch.cdist(real_samples, generated_samples)
    min_dists_rg = cross_dists_rg.min(dim=1).values
    recall = (min_dists_rg < gen_radii.mean()).float().mean().item()
    
    return precision, recall


def run_quality_metrics(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    real_data: torch.Tensor,
    device: torch.device,
    samplers: list[str] = ["gaussian", "riemannian", "geodesic"],
    n_samples: int = 1000,
    mcmc_steps: int = 100,
    n_lf: int = 15,
    eps_lf: float = 0.03,
    use_dual_metric: bool = False,
    adaptive_dual_step: bool = False,
    adaptive_max_dual_displacement: float = 0.07,
    adaptive_min_step_scale: float = 0.05,
    volume_power: float = 2.0,
    radial_prior_weight: float = 0.1,
    momentum_persist: float = 0.0,
    fp_steps: int = 15,
    fp_damping: float = 0.72,
    use_physics_sampler_defaults: bool = False,
    out_dir: Optional[Path] = None,
    wandb_run: Optional[Any] = None,
) -> dict[str, dict[str, float]]:
    """
    Compute quality metrics for different samplers.
    """
    results = {}
    
    print("\n=== Quality Metrics ===")
    
    # Encode real data to latent space
    with torch.no_grad():
        real_flat = real_data.view(real_data.shape[0], -1).to(device)
        enc_out = model.encoder(real_flat)
        real_latents = enc_out.embedding
    
    for sampler_name in samplers:
        print(f"\n  {sampler_name}:")
        
        try:
            # Generate samples
            z_samples, acc_rate = sample_latents(
                model, sampler_name, n_samples, device,
                mcmc_steps=mcmc_steps,
                n_lf=n_lf,
                eps_lf=eps_lf,
                use_dual_metric=use_dual_metric,
                adaptive_dual_step=adaptive_dual_step,
                adaptive_max_dual_displacement=adaptive_max_dual_displacement,
                adaptive_min_step_scale=adaptive_min_step_scale,
                volume_power=volume_power,
                radial_prior_weight=radial_prior_weight,
                momentum_persist=momentum_persist,
                fp_steps=fp_steps,
                fp_damping=fp_damping,
                use_physics_sampler_defaults=use_physics_sampler_defaults,
            )
            
            # Diversity
            diversity = compute_diversity_score(z_samples)
            
            # Coverage
            coverage = compute_coverage_score(z_samples, centroids)
            
            # Precision/Recall
            precision, recall = compute_precision_recall(
                real_latents[:min(1000, real_latents.shape[0])],
                z_samples[:min(1000, z_samples.shape[0])],
            )
            
            results[sampler_name] = {
                "diversity": diversity,
                "coverage": coverage,
                "precision": precision,
                "recall": recall,
                "acceptance_rate": acc_rate,
            }
            
            print(f"    Diversity: {diversity:.3f}")
            print(f"    Coverage: {coverage:.3f}")
            print(f"    Precision: {precision:.3f}")
            print(f"    Recall: {recall:.3f}")
        
        except Exception as e:
            print(f"    Error: {e}")
            results[sampler_name] = {"error": str(e)}
    
    # Plot comparison
    if out_dir is not None:
        plot_quality_metrics(results, out_dir, wandb_run)
    
    return results


def plot_quality_metrics(
    results: dict[str, dict[str, float]],
    out_dir: Path,
    wandb_run: Optional[Any] = None,
) -> None:
    """Plot quality metrics comparison."""
    metrics = ["diversity", "coverage", "precision", "recall"]
    samplers = [s for s in results.keys() if "error" not in results[s]]
    
    if not samplers:
        return
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    
    x = np.arange(len(samplers))
    
    for ax, metric in zip(axes, metrics):
        values = [results[s].get(metric, 0) for s in samplers]
        bars = ax.bar(x, values, color="steelblue", edgecolor="black")
        
        ax.set_xticks(x)
        ax.set_xticklabels(samplers, rotation=45, ha="right")
        ax.set_ylabel(metric.title())
        ax.set_title(metric.title())
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    
    plt.tight_layout()
    save_plot(fig, out_dir / "quality_metrics.png", "sampling/quality_metrics", wandb_run)


# =============================================================================
# RHMC Chain Diagnostics
# =============================================================================

def run_rhmc_diagnostics(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    device: torch.device,
    n_chains: int = 4,
    chain_length: int = 200,
    sampler_name: str = "volume",
    n_lf: int = 15,
    eps_lf: float = 0.03,
    use_dual_metric: bool = False,
    adaptive_dual_step: bool = False,
    adaptive_max_dual_displacement: float = 0.07,
    adaptive_min_step_scale: float = 0.05,
    volume_power: float = 2.0,
    radial_prior_weight: float = 0.1,
    momentum_persist: float = 0.0,
    fp_steps: int = 15,
    fp_damping: float = 0.72,
    use_physics_sampler_defaults: bool = False,
    out_dir: Optional[Path] = None,
    wandb_run: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Run RHMC chain diagnostics with multiple chains.
    
    Runs Metropolis-corrected chains using the selected sampler.
    """
    print("\n=== RHMC Chain Diagnostics ===")
    
    chains = []
    energies = []
    acceptance_rates = []
    
    for chain_idx in range(n_chains):
        print(f"  Chain {chain_idx + 1}/{n_chains}")
        
        # Start from random centroid
        start_idx = torch.randint(0, centroids.shape[0], (1,)).item()
        z0 = centroids[start_idx:start_idx + 1].clone().to(device)
        
        sampler = create_sampler(
            model, sampler_name, chain_length, n_lf, eps_lf, 
            use_dual_metric=use_dual_metric,
            adaptive_dual_step=adaptive_dual_step,
            adaptive_max_dual_displacement=adaptive_max_dual_displacement,
            adaptive_min_step_scale=adaptive_min_step_scale,
            volume_power=volume_power,
            radial_prior_weight=radial_prior_weight,
            momentum_persist=momentum_persist,
            fp_steps=fp_steps,
            fp_damping=fp_damping,
            use_physics_sampler_defaults=use_physics_sampler_defaults,
        )
        
        if sampler is None:
            continue
        
        chain, chain_energies, acceptance, errors = run_hmc_chain(
            z0, sampler, chain_length, n_lf, eps_lf
        )
        chains.append(chain)
        energies.append(chain_energies)
        acceptance_rates.append(acceptance)
        # We could technically aggregate errors here, but for now just returning them
        print(f"    Errors: {errors}")
    
    results = {
        "chains": chains,
        "energies": energies,
        "acceptance_rates": acceptance_rates,
        "mean_acceptance": np.mean(acceptance_rates),
    }
    
    print(f"  Mean acceptance rate: {results['mean_acceptance']:.3f}")
    
    # Visualize
    if out_dir is not None:
        plot_rhmc_diagnostics(results, centroids, out_dir, wandb_run)
    
    return results


def plot_rhmc_diagnostics(
    results: dict[str, Any],
    centroids: torch.Tensor,
    out_dir: Path,
    wandb_run: Optional[Any] = None,
) -> None:
    """Visualize RHMC chain diagnostics."""
    chains = results["chains"]
    energies = results["energies"]
    
    if not chains:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    c_np = centroids[:, :2].cpu().numpy()
    colors = plt.cm.Set1(np.linspace(0, 1, len(chains)))
    
    # Plot 1: Chains in latent space
    ax = axes[0, 0]
    ax.scatter(c_np[:, 0], c_np[:, 1], c="gray", alpha=0.3, s=20, label="Centroids")
    
    for i, chain in enumerate(chains):
        if chain.shape[1] >= 2:
            ax.plot(chain[:, 0], chain[:, 1], "-", color=colors[i], alpha=0.7, linewidth=1)
            ax.scatter(chain[0, 0], chain[0, 1], marker="o", s=50, color=colors[i], edgecolor="black")
    
    ax.set_title("RHMC Chains in Latent Space")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_aspect("equal")
    
    # Plot 2: Energy traces
    ax = axes[0, 1]
    for i, energy in enumerate(energies):
        ax.plot(energy, color=colors[i], alpha=0.7, label=f"Chain {i+1}")
    ax.set_title("Energy Traces")
    ax.set_xlabel("Step")
    ax.set_ylabel("Hamiltonian")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Marginal distributions
    ax = axes[1, 0]
    for i, chain in enumerate(chains):
        if chain.shape[1] >= 1:
            ax.hist(chain[:, 0], bins=30, alpha=0.5, color=colors[i], label=f"Chain {i+1}")
    ax.set_title("Marginal Distribution (z1)")
    ax.set_xlabel("z1")
    ax.set_ylabel("Count")
    ax.legend()
    
    # Plot 4: Acceptance rates
    ax = axes[1, 1]
    ax.bar(range(len(results["acceptance_rates"])), results["acceptance_rates"], color="steelblue")
    ax.axhline(y=results["mean_acceptance"], color="red", linestyle="--", label=f"Mean: {results['mean_acceptance']:.3f}")
    ax.set_title("Acceptance Rates per Chain")
    ax.set_xlabel("Chain")
    ax.set_ylabel("Acceptance Rate")
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    save_plot(fig, out_dir / "rhmc_diagnostics.png", "sampling/rhmc_diagnostics", wandb_run)


# =============================================================================
# Main Entry Points
# =============================================================================

def run_sampling_diagnostics(
    model_path: Path | str,
    real_data: Optional[torch.Tensor] = None,
    output_dir: Optional[Path | str] = None,
    device: Optional[torch.device] = None,
    wandb_run: Optional[Any] = None,
    # Optional: pass model directly (skip loading from disk)
    model: Optional[GeometryRHVAE] = None,
    # Metric override
    aggressive_void: bool = False,
    # FID parameters
    run_fid: bool = True,
    fid_samples: int = 2000,
    # Default FID samplers
    fid_samplers: list[str] = ["gaussian", "volume"],
    # Multi-start parameters
    run_multi_start: bool = True,
    n_starts: int = 16,
    multi_start_sampler: str = "volume",
    # Interpolation parameters
    run_interpolation: bool = True,
    n_interp_pairs: int = 4,
    # Quality metrics parameters
    run_quality: bool = True,
    quality_samples: int = 1000,
    quality_samplers: list[str] = ["gaussian", "volume"],
    # RHMC diagnostics parameters
    do_rhmc_diagnostics: bool = True,
    n_chains: int = 4,
    chain_length: int = 200,
    rhmc_sampler: str = "volume",
    # MCMC parameters
    mcmc_steps: int = 100,
    n_lf: int = 15,
    eps_lf: float = 0.03,
    use_dual_metric: bool = False,
    adaptive_dual_step: bool = False,
    adaptive_max_dual_displacement: float = 0.07,
    adaptive_min_step_scale: float = 0.05,
    volume_power: float = 2.0,
    radial_prior_weight: float = 0.1,
    momentum_persist: float = 0.0,
    fp_steps: int = 15,
    fp_damping: float = 0.72,
    use_physics_sampler_defaults: bool = False,
) -> dict[str, Any]:
    """
    Run comprehensive sampling diagnostics.
    
    Args:
        model_path: Path to trained model directory (used for output and loading if model not provided)
        real_data: Real data for FID computation [N, C, H, W] or [N, D]
        output_dir: Output directory for plots and metrics
        device: PyTorch device
        wandb_run: WandB run for logging
        model: Optional pre-loaded model (if provided, skips loading from disk)
        ... (see parameter docs above)
    
    Returns:
        Dictionary containing all computed metrics
    """
    model_path = Path(model_path)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if output_dir is None:
        output_dir = model_path / "sampling_diagnostics"
    out_dir = build_output_dir(output_dir)
    
    print(f"\n{'='*60}")
    print("Sampling Diagnostics for GeometryRHVAE")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Output dir: {out_dir}")
    print(f"Device: {device}")
    
    # Load model (or use provided model)
    if model is not None:
        # Use provided model - get centroids from it
        model = model.to(device)
        model.eval()
        # Handle centroids - prefer centroids_tens (tensor) over centroids (deque)
        if hasattr(model, 'centroids_tens') and model.centroids_tens is not None and model.centroids_tens.numel() > 0:
            centroids = model.centroids_tens.to(device)
        elif hasattr(model, 'centroids') and model.centroids is not None:
            raw_centroids = model.centroids
            if hasattr(raw_centroids, 'to'):
                centroids = raw_centroids.to(device)
            elif hasattr(raw_centroids, '__iter__') and len(raw_centroids) > 0:
                centroids = torch.stack([c for c in raw_centroids]).to(device)
            else:
                raise ValueError("Model centroids deque is empty")
        else:
            raise ValueError("Model has no centroids set")
        print(f"Using provided model with {centroids.shape[0]} centroids, latent_dim={model.latent_dim}")
    else:
        # Load from disk
        model, centroids = load_model_and_centroids(model_path, device, aggressive_void=aggressive_void)
        print(f"Loaded model with {centroids.shape[0]} centroids, latent_dim={model.latent_dim}")
    
    all_results = {}
    
    # 1. FID Evaluation
    if run_fid and real_data is not None:
        fid_results = run_fid_evaluation(
            model, real_data, device,
            samplers=fid_samplers,
            n_samples=fid_samples,
            mcmc_steps=mcmc_steps,
            n_lf=n_lf,
            eps_lf=eps_lf,
            use_dual_metric=use_dual_metric,
            adaptive_dual_step=adaptive_dual_step,
            adaptive_max_dual_displacement=adaptive_max_dual_displacement,
            adaptive_min_step_scale=adaptive_min_step_scale,
            volume_power=volume_power,
            radial_prior_weight=radial_prior_weight,
            momentum_persist=momentum_persist,
            fp_steps=fp_steps,
            fp_damping=fp_damping,
            use_physics_sampler_defaults=use_physics_sampler_defaults,
            out_dir=out_dir,
            wandb_run=wandb_run,
        )
        all_results["fid"] = fid_results
    
    # 2. Multi-Start Sampling
    if run_multi_start:
        multi_start_results = multi_start_sampling(
            model, centroids, device,
            n_starts=n_starts,
            sampler_name=multi_start_sampler,
            mcmc_steps=mcmc_steps // 2,
            n_lf=n_lf // 2,
            eps_lf=eps_lf,
            use_dual_metric=use_dual_metric,
            adaptive_dual_step=adaptive_dual_step,
            adaptive_max_dual_displacement=adaptive_max_dual_displacement,
            adaptive_min_step_scale=adaptive_min_step_scale,
            volume_power=volume_power,
            radial_prior_weight=radial_prior_weight,
            momentum_persist=momentum_persist,
            fp_steps=fp_steps,
            fp_damping=fp_damping,
            use_physics_sampler_defaults=use_physics_sampler_defaults,
            out_dir=out_dir,
            wandb_run=wandb_run,
        )
        all_results["multi_start"] = {
            "n_centroid_chains": len(multi_start_results["centroid_starts"]),
            "n_gaussian_chains": len(multi_start_results["gaussian_starts"]),
            "n_grid_chains": len(multi_start_results["grid_starts"]),
        }
    
    # 3. Interpolation Comparison
    if run_interpolation:
        interp_results = run_interpolation_comparison(
            model, centroids, device,
            n_pairs=n_interp_pairs,
            out_dir=out_dir,
            wandb_run=wandb_run,
        )
        all_results["interpolation"] = {
            "methods": list(interp_results.keys()),
            "n_pairs": n_interp_pairs,
        }
    
    # 4. Quality Metrics
    if run_quality and real_data is not None:
        quality_results = run_quality_metrics(
            model, centroids, real_data, device,
            samplers=quality_samplers,
            n_samples=quality_samples,
            mcmc_steps=mcmc_steps,
            n_lf=n_lf,
            eps_lf=eps_lf,
            use_dual_metric=use_dual_metric,
            adaptive_dual_step=adaptive_dual_step,
            adaptive_max_dual_displacement=adaptive_max_dual_displacement,
            adaptive_min_step_scale=adaptive_min_step_scale,
            volume_power=volume_power,
            radial_prior_weight=radial_prior_weight,
            momentum_persist=momentum_persist,
            fp_steps=fp_steps,
            fp_damping=fp_damping,
            use_physics_sampler_defaults=use_physics_sampler_defaults,
            out_dir=out_dir,
            wandb_run=wandb_run,
        )
        all_results["quality"] = quality_results
    
    # 5. RHMC Chain Diagnostics
    if do_rhmc_diagnostics:
        rhmc_results = run_rhmc_diagnostics(
            model, centroids, device,
            n_chains=n_chains,
            chain_length=chain_length,
            sampler_name=rhmc_sampler,
            n_lf=n_lf,
            eps_lf=eps_lf,
            use_dual_metric=use_dual_metric,
            adaptive_dual_step=adaptive_dual_step,
            adaptive_max_dual_displacement=adaptive_max_dual_displacement,
            adaptive_min_step_scale=adaptive_min_step_scale,
            volume_power=volume_power,
            radial_prior_weight=radial_prior_weight,
            momentum_persist=momentum_persist,
            fp_steps=fp_steps,
            fp_damping=fp_damping,
            use_physics_sampler_defaults=use_physics_sampler_defaults,
            out_dir=out_dir,
            wandb_run=wandb_run,
        )
        all_results["rhmc"] = {
            "mean_acceptance": rhmc_results["mean_acceptance"],
            "acceptance_rates": rhmc_results["acceptance_rates"],
        }
    
    # Save summary
    summary_path = out_dir / "metrics_summary.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    with open(summary_path, "w") as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*60}")
    
    # Log summary to WandB
    if wandb_run is not None and wandb is not None:
        flat_results = {}
        for category, metrics in all_results.items():
            if isinstance(metrics, dict):
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        flat_results[f"sampling/{category}/{name}"] = value
                    elif isinstance(value, dict):
                        for subname, subvalue in value.items():
                            if isinstance(subvalue, (int, float)):
                                flat_results[f"sampling/{category}/{name}/{subname}"] = subvalue
        wandb_run.log(flat_results)
    
    return all_results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sampling diagnostics for GeometryRHVAE models."
    )
    
    # Required
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory (containing rhvae_model.pt and rhvae_metric.pt)",
    )
    
    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to real data (.pt file) for FID computation",
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: model_path/sampling_diagnostics)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device",
    )
    
    # Metric override
    parser.add_argument("--aggressive_void", action="store_true", 
                        help="Use aggressive void parameters (stronger RHMC manifold adherence)")
    
    # What to run
    parser.add_argument("--skip_fid", action="store_true", help="Skip FID evaluation")
    parser.add_argument("--skip_multi_start", action="store_true", help="Skip multi-start sampling")
    parser.add_argument("--skip_interpolation", action="store_true", help="Skip interpolation comparison")
    parser.add_argument("--skip_quality", action="store_true", help="Skip quality metrics")
    parser.add_argument("--skip_rhmc", action="store_true", help="Skip RHMC diagnostics")
    
    # FID parameters
    parser.add_argument("--fid_samples", type=int, default=2000, help="Number of samples for FID")
    parser.add_argument(
        "--fid_samplers",
        type=str,
        nargs="+",
        default=["gaussian", "volume"],
        help="Samplers to evaluate for FID",
    )
    
    # Multi-start parameters
    parser.add_argument("--n_starts", type=int, default=16, help="Number of starting points")
    parser.add_argument(
        "--multi_start_sampler",
        type=str,
        default="volume",
        choices=list(SAMPLER_REGISTRY.keys()),
        help="Sampler used for multi-start chains",
    )
    
    # Interpolation parameters
    parser.add_argument("--n_interp_pairs", type=int, default=4, help="Number of interpolation pairs")
    
    # Quality parameters
    parser.add_argument("--quality_samples", type=int, default=1000, help="Samples for quality metrics")
    parser.add_argument(
        "--quality_samplers",
        type=str,
        nargs="+",
        default=["gaussian", "volume"],
        help="Samplers to evaluate for quality metrics",
    )
    
    # RHMC parameters
    parser.add_argument("--n_chains", type=int, default=4, help="Number of RHMC chains")
    parser.add_argument("--chain_length", type=int, default=200, help="Length of each chain")
    parser.add_argument(
        "--rhmc_sampler",
        type=str,
        default="volume",
        choices=list(SAMPLER_REGISTRY.keys()),
        help="Sampler used for RHMC diagnostics",
    )
    
    # MCMC parameters
    parser.add_argument("--mcmc_steps", type=int, default=100, help="MCMC steps for samplers")
    parser.add_argument("--n_lf", type=int, default=15, help="Leapfrog steps")
    parser.add_argument("--eps_lf", type=float, default=0.03, help="Leapfrog step size")
    parser.add_argument("--use_dual_metric", action="store_true", help="Use dual metric convention (M=G^{-1}).")
    parser.add_argument("--adaptive_dual_step", action="store_true", help="Enable adaptive step scaling in RHMC.")
    parser.add_argument("--adaptive_max_dual_displacement", type=float, default=0.07)
    parser.add_argument("--adaptive_min_step_scale", type=float, default=0.05)
    parser.add_argument("--volume_power", type=float, default=2.0, help="Volume-element exponent for volume_riemannian.")
    parser.add_argument("--radial_prior_weight", type=float, default=0.1, help="Optional radial prior stabilizer.")
    parser.add_argument("--momentum_persist", type=float, default=0.0)
    parser.add_argument("--fp_steps", type=int, default=15, help="Fixed-point iterations for implicit RHMC.")
    parser.add_argument("--fp_damping", type=float, default=0.72, help="Fixed-point damping factor.")
    parser.add_argument("--physics_sampler", action="store_true", help="Automatically deduce sampler parameters from temperature and latent dimension.")
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    
    args = parser.parse_args()
    
    # Load real data
    real_data = None
    data_path = args.data_path
    
    # If no data path provided, try to find train_data.pt in model directory
    if data_path is None:
        model_dir = Path(args.model_path)
        if model_dir.suffix == ".pt":
            model_dir = model_dir.parent
        auto_data_path = model_dir / "train_data.pt"
        if auto_data_path.exists():
            data_path = str(auto_data_path)
            print(f"Auto-detected training data: {data_path}")
    
    if data_path:
        print(f"Loading real data from: {data_path}")
        real_data = torch.load(data_path)
        if isinstance(real_data, dict):
            real_data = real_data.get("data", real_data.get("images", None))
        if real_data is not None:
            print(f"  Shape: {real_data.shape}")
        else:
            print("  Warning: Could not extract data from file")
    
    # Initialize WandB
    wandb_run = None
    if args.wandb_project and wandb is not None:
        run_name = args.wandb_run_name or f"sampling_diag_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )
    
    # Run diagnostics
    results = run_sampling_diagnostics(
        model_path=args.model_path,
        real_data=real_data,
        output_dir=args.output_dir,
        device=torch.device(args.device),
        wandb_run=wandb_run,
        aggressive_void=args.aggressive_void,
        run_fid=not args.skip_fid and real_data is not None,
        fid_samples=args.fid_samples,
        fid_samplers=args.fid_samplers,
        run_multi_start=not args.skip_multi_start,
        n_starts=args.n_starts,
        multi_start_sampler=args.multi_start_sampler,
        run_interpolation=not args.skip_interpolation,
        n_interp_pairs=args.n_interp_pairs,
        run_quality=not args.skip_quality and real_data is not None,
        quality_samples=args.quality_samples,
        quality_samplers=args.quality_samplers,
        do_rhmc_diagnostics=not args.skip_rhmc,
        n_chains=args.n_chains,
        chain_length=args.chain_length,
        rhmc_sampler=args.rhmc_sampler,
        mcmc_steps=args.mcmc_steps,
        n_lf=args.n_lf,
        eps_lf=args.eps_lf,
        use_dual_metric=args.use_dual_metric,
        adaptive_dual_step=args.adaptive_dual_step,
        adaptive_max_dual_displacement=args.adaptive_max_dual_displacement,
        adaptive_min_step_scale=args.adaptive_min_step_scale,
        volume_power=args.volume_power,
        radial_prior_weight=args.radial_prior_weight,
        momentum_persist=args.momentum_persist,
        fp_steps=args.fp_steps,
        fp_damping=args.fp_damping,
        use_physics_sampler_defaults=args.physics_sampler,
    )
    
    # Finish WandB
    if wandb_run is not None:
        wandb_run.finish()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
