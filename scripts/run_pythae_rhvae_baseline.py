#!/usr/bin/env python
"""
Run original pythae RHVAE as a baseline to compare metric learning.
Uses t=0 frames by default; can optionally flatten all frames.
Full visualization parity with Stage C experiments.
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def _maybe_add_sys_path(path: Path) -> bool:
    if path.exists():
        sys.path.insert(0, str(path))
        return True
    return False

# Prefer vendored pythae if present; otherwise fall back to RlVAE's copy.
_maybe_add_sys_path(PROJECT_ROOT / "src" / "lib" / "src")

_rlvae_root = os.environ.get("RLVAE_ROOT", None)
if _rlvae_root is None:
    _rlvae_root = PROJECT_ROOT.parent / "RlVAE"
else:
    _rlvae_root = Path(_rlvae_root).expanduser()

if isinstance(_rlvae_root, Path) and _rlvae_root.exists():
    _maybe_add_sys_path(_rlvae_root / "src" / "lib" / "src")
    _maybe_add_sys_path(_rlvae_root / "src")

import torch  # pyright: ignore[reportMissingImports]  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]  # pyright: ignore[reportMissingImports]
import matplotlib.colors as mcolors  # pyright: ignore[reportMissingImports]
import matplotlib.patches as patches  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
from datetime import datetime

# Import pythae RHVAE
from pythae.models.rhvae import RHVAE, RHVAEConfig  # pyright: ignore[reportMissingImports]
from src.models.rhvae_geometry import GeometryRHVAE, GeometryRHVAEConfig
from scripts.analyze_metric_full import run_analysis
from scripts.sampling_diagnostics import run_sampling_diagnostics
from torch.utils.data import DataLoader, TensorDataset  # pyright: ignore[reportMissingImports]
from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]

# Import our data module
from src.data.ellipse_datamodule import EllipseSequenceDataModule


def create_frames_dataset(data_module, split='train', frame_mode='t0', max_frames=None, seed=42):
    """Extract frames from sequence data (t0 or all)."""
    data_module.setup('fit')
    
    if split == 'train':
        loader = data_module.train_dataloader()
    else:
        loader = data_module.val_dataloader()
    
    all_frames = []
    for batch in loader:
        # Handle different batch formats
        if isinstance(batch, dict):
            x = batch.get('data', batch.get('x', batch.get('images', None)))
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        
        if x is None:
            print(f"[DEBUG] Batch keys: {batch.keys() if isinstance(batch, dict) else type(batch)}")
            continue
        
        # x shape: [B, T, C, H, W]
        if x.dim() == 5:
            if frame_mode == 'all':
                frames = x.reshape(-1, *x.shape[2:])
            else:
                frames = x[:, 0]
        elif x.dim() == 4:
            frames = x  # Already [B, C, H, W]
        else:
            print(f"[DEBUG] Unexpected x shape: {x.shape}")
            continue
        
        all_frames.append(frames)
    
    if len(all_frames) == 0:
        raise RuntimeError("No frames extracted from dataset!")
    
    all_frames = torch.cat(all_frames, dim=0)
    if max_frames is not None and all_frames.shape[0] > max_frames:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(all_frames.shape[0], generator=g)[:max_frames]
        all_frames = all_frames[idx]

    print(f"[RHVAE BASELINE] Extracted {all_frames.shape[0]} frames, shape: {all_frames.shape[1:]}")
    return all_frames


def _get_centroids_tensor(model):
    """Return centroids as a [N, D] tensor if available."""
    c_tensor = None
    if hasattr(model, 'centroids_tens') and isinstance(model.centroids_tens, torch.Tensor):
        if model.centroids_tens.numel() > 0:
            c_tensor = model.centroids_tens
    if c_tensor is None and hasattr(model, 'centroids'):
        c_raw = model.centroids
        if hasattr(c_raw, '__iter__') and not isinstance(c_raw, torch.Tensor):
            if len(c_raw) == 0:
                return None
            c_tensor = torch.stack(list(c_raw))
        else:
            c_tensor = c_raw
    if not isinstance(c_tensor, torch.Tensor) or c_tensor.numel() == 0:
        return None
    if c_tensor.ndim == 3:
        c_tensor = c_tensor.reshape(-1, c_tensor.shape[-1])
    return c_tensor


def _subsample_centroids(model, max_centroids, seed):
    """Subsample centroids and metric matrices to a fixed budget."""
    if max_centroids is None:
        return
    centroids = _get_centroids_tensor(model)
    if not isinstance(centroids, torch.Tensor):
        return
    n_total = centroids.shape[0]
    if n_total <= max_centroids:
        return
    M_tens = getattr(model, 'M_tens', None)
    if not isinstance(M_tens, torch.Tensor) or M_tens.shape[0] != n_total:
        print("[RHVAE BASELINE] ⚠️ Cannot subsample centroids: missing or mismatched M_tens")
        return
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n_total, generator=g)[:max_centroids]
    idx = idx.to(centroids.device)
    model.centroids_tens = centroids.index_select(0, idx)
    model.M_tens = M_tens.index_select(0, idx)
    # Also subsample attractor precisions if they exist and match the old count
    P_tens = getattr(model, 'P_tens', None)
    if isinstance(P_tens, torch.Tensor) and P_tens.shape[0] == n_total:
        model.P_tens = P_tens.index_select(0, idx)
    elif isinstance(P_tens, torch.Tensor) and P_tens.shape[0] != max_centroids:
        # Recompute P_tens from the subsampled M_tens
        if hasattr(model, '_update_attractor_precisions'):
            model._update_attractor_precisions(model.M_tens)
    print(f"[RHVAE BASELINE] Subsampled centroids: {n_total} → {max_centroids}")


def _suggest_temperature(centroids, sample_cap=2000, stat="median_nn"):
    """Centroid-distance heuristic for RHVAE temperature.
    
    The temperature T controls the Gaussian kernel width via exp(-d²/T²).
    We compute a candidate from centroid distances, then enforce a floor
    based on the latent spread to prevent sub-diffraction kernel widths
    that cause extreme metric gradients and NaN in the RHMC sampler.
    """
    if not isinstance(centroids, torch.Tensor) or centroids.shape[0] < 2:
        return None
    c = centroids.detach().cpu()
    if c.shape[0] > sample_cap:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(c.shape[0], generator=g)[:sample_cap]
        c = c.index_select(0, idx)

    # Compute a principled temperature floor: Silverman-like lower bound.
    # For N centroids in D dimensions, the minimum sensible bandwidth is
    # sigma * (4/(D+2))^(1/(D+4)) * N^(-1/(D+4)).
    # This prevents the kernel from becoming a delta function.
    N_pts = float(c.shape[0])
    D_dim = float(c.shape[1])
    latent_std = float(c.std(dim=0).mean().item())
    silverman_floor = latent_std * ((4.0 / (D_dim + 2.0)) ** (1.0 / (D_dim + 4.0))) * (N_pts ** (-1.0 / (D_dim + 4.0)))

    if stat == "silverman":
        return silverman_floor

    dists = torch.cdist(c, c)
    if stat == "median_pairwise":
        triu_idx = torch.triu_indices(dists.shape[0], dists.shape[1], offset=1)
        pairwise_dists = dists[triu_idx[0], triu_idx[1]]
        if pairwise_dists.numel() == 0:
            return None
        candidate = float(pairwise_dists.median().item())
        return max(candidate, silverman_floor)

    dists.fill_diagonal_(float("inf"))
    if stat == "mean_pairwise":
        finite = dists[torch.isfinite(dists)]
        if finite.numel() == 0:
            return None
        candidate = float(finite.mean().item())
        return max(candidate, silverman_floor)
        
    if stat.startswith("mean_knn_"):
        k = int(stat.split("_")[-1])
        # Ensure k is within bounds (if there are fewer than k points, use max possible k)
        max_k = dists.shape[1] - 1
        safe_k = min(k, max_k)
        if safe_k <= 0:
            return None
        # sort distances: the 0th element is the 1st neighbor (since diagonal is inf)
        knn_dists = dists.sort(dim=1).values[:, safe_k - 1]
        candidate = float(knn_dists.mean().item())
        if candidate < silverman_floor:
            print(
                f"[RHVAE BASELINE] T floor activated: knn_candidate={candidate:.4f} "
                f"< silverman_floor={silverman_floor:.4f}, using floor"
            )
        return max(candidate, silverman_floor)

    nn = dists.min(dim=1).values
    if nn.numel() == 0:
        return None
    if stat == "mean_nn":
        candidate = float(nn.mean().item())
    else:
        candidate = float(nn.median().item())
    return max(candidate, silverman_floor)


def visualize_reconstructions(model, data, epoch, output_dir, wandb, device, n_samples=8):
    """Visualize original vs reconstructed images."""
    model.eval()
    
    # Get a batch of samples
    idx = torch.randperm(len(data))[:n_samples]
    samples = data[idx].to(device).requires_grad_(True)
    
    # RHVAE needs gradients for RHMC even during inference
    with torch.enable_grad():
        model_input = {"data": samples}
        model_output = model(model_input)
        recon = model_output.recon_x.detach()
    
    # Reshape to images (assuming flattened square images)
    input_dim = samples.shape[1]
    import math
    img_size = int(math.sqrt(input_dim))
    samples_img = samples.detach().view(n_samples, 1, img_size, img_size).cpu()
    recon_img = recon.view(n_samples, 1, img_size, img_size).cpu()
    
    # Create figure: 2 rows x n_samples columns
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
    fig.suptitle(f'Epoch {epoch}: Originals (top) vs Reconstructions (bottom)', fontsize=12)
    
    for i in range(n_samples):
        axes[0, i].imshow(samples_img[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[1, i].imshow(recon_img[i, 0].clamp(0, 1), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Recon')
    
    plt.tight_layout()
    
    save_path = output_dir / f'reconstructions_epoch_{epoch:03d}.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    
    if wandb is not None and wandb.run is not None:
        wandb.log({f"reconstructions": wandb.Image(fig)}, step=epoch)
    
    plt.close(fig)
    model.train()


def visualize_latent_space(model, data, epoch, output_dir, wandb, device, n_samples=500):
    """Visualize latent space encodings."""
    model.eval()
    
    # Get samples
    idx = torch.randperm(len(data))[:n_samples]
    samples = data[idx].to(device).requires_grad_(True)
    
    # RHVAE needs gradients for RHMC
    with torch.enable_grad():
        model_input = {"data": samples}
        model_output = model(model_input)
        z = model_output.z.detach().cpu().numpy()
    
    # Get centroids if available
    centroids = None
    c_tensor = _get_centroids_tensor(model)
    if isinstance(c_tensor, torch.Tensor):
        c_np = c_tensor.detach().cpu().numpy()
        if c_np.ndim == 2 and c_np.shape[1] >= 2:
            centroids = c_np[:, :2]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(z[:, 0], z[:, 1], alpha=0.5, s=20, c='blue', label='Encodings')
    
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, marker='x', label='Centroids')
    
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_title(f'Epoch {epoch}: Latent Space')
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    save_path = output_dir / f'latent_space_epoch_{epoch:03d}.png'
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    
    if wandb is not None and wandb.run is not None:
        wandb.log({f"latent_space": wandb.Image(fig)}, step=epoch)
    
    plt.close(fig)
    model.train()


def visualize_metric_field(model, epoch, output_dir, wandb, device, grid_res=40):
    """Visualize the learned metric field."""
    model.eval()
    
    # Get centroids for bounds
    centroids = None
    c_tensor = _get_centroids_tensor(model)
    if isinstance(c_tensor, torch.Tensor):
        c_np = c_tensor.detach().cpu().numpy()
        if c_np.ndim == 2 and c_np.shape[1] >= 2:
            centroids = c_np[:, :2]
    
    # Set grid bounds
    if centroids is not None:
        c_min = centroids.min(axis=0)
        c_max = centroids.max(axis=0)
        x_min_base, x_max_base = float(c_min[0]), float(c_max[0])
        y_min_base, y_max_base = float(c_min[1]), float(c_max[1])
    else:
        x_min_base, x_max_base = -5.0, 5.0
        y_min_base, y_max_base = -5.0, 5.0
    
    # Add margin
    x_margin = (x_max_base - x_min_base) * 0.3
    y_margin = (y_max_base - y_min_base) * 0.3
    x_min, x_max = x_min_base - x_margin, x_max_base + x_margin
    y_min, y_max = y_min_base - y_margin, y_max_base + y_margin
    
    x_range = np.linspace(x_min, x_max, grid_res)
    y_range = np.linspace(y_min, y_max, grid_res)
    X, Y = np.meshgrid(x_range, y_range)
    grid_pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    z_grid = torch.tensor(grid_pts, dtype=torch.float32, device=device)
    if model.latent_dim > 2:
        pad = torch.zeros(z_grid.shape[0], model.latent_dim - 2, device=device)
        z_grid = torch.cat([z_grid, pad], dim=1)
    elif model.latent_dim < 2:
        z_grid = z_grid[:, :model.latent_dim]
            
    # Compute metric
    with torch.no_grad():
        G_grid = model.G(z_grid)
    
    G_np = G_grid.cpu().numpy()
    
    if G_np.ndim == 4:
        G_np = G_np.mean(axis=1)
    if G_np.shape[-1] > 2:
        G_np = G_np[:, :2, :2]
    
    actual_n = G_np.shape[0]
    actual_grid_res = int(np.sqrt(actual_n))
    if actual_grid_res ** 2 != actual_n:
        model.train()
        return
    
    if actual_grid_res != grid_res:
        grid_res = actual_grid_res
        x_range = np.linspace(x_min, x_max, grid_res)
        y_range = np.linspace(y_min, y_max, grid_res)
    
    # Compute eigenvalues
    eigvals = np.linalg.eigvalsh(G_np)
    cond = eigvals[:, -1] / (eigvals[:, 0] + 1e-8)
    logdet = np.log(np.abs(np.linalg.det(G_np)) + 1e-12)
    
    # Reshape
    cond_grid = cond.reshape(grid_res, grid_res)
    logdet_grid = logdet.reshape(grid_res, grid_res)
    eig_min = eigvals[:, 0].reshape(grid_res, grid_res)
    eig_max = eigvals[:, -1].reshape(grid_res, grid_res)
    
    extent = [x_min, x_max, y_min, y_max]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Epoch {epoch}: RHVAE Metric Field', fontsize=14)
    
    # Condition number
    im0 = axes[0, 0].imshow(cond_grid, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    if centroids is not None:
        axes[0, 0].scatter(centroids[:, 0], centroids[:, 1], c='red', s=15, alpha=0.7)
    axes[0, 0].set_title(f'Condition Number\nrange: [{cond.min():.2f}, {cond.max():.2f}]')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Log determinant
    im1 = axes[0, 1].imshow(logdet_grid, extent=extent, origin='lower', cmap='RdBu_r', aspect='auto')
    if centroids is not None:
        axes[0, 1].scatter(centroids[:, 0], centroids[:, 1], c='black', s=15, alpha=0.7)
    axes[0, 1].set_title(f'log|det(G)|\nrange: [{logdet.min():.2f}, {logdet.max():.2f}]')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Min eigenvalue
    im2 = axes[1, 0].imshow(eig_min, extent=extent, origin='lower', cmap='plasma', aspect='auto')
    if centroids is not None:
        axes[1, 0].scatter(centroids[:, 0], centroids[:, 1], c='white', s=15, alpha=0.7)
    axes[1, 0].set_title(f'Min Eigenvalue\nrange: [{eigvals[:, 0].min():.3f}, {eigvals[:, 0].max():.3f}]')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Max eigenvalue
    im3 = axes[1, 1].imshow(eig_max, extent=extent, origin='lower', cmap='plasma', aspect='auto')
    if centroids is not None:
        axes[1, 1].scatter(centroids[:, 0], centroids[:, 1], c='white', s=15, alpha=0.7)
    axes[1, 1].set_title(f'Max Eigenvalue\nrange: [{eigvals[:, -1].min():.3f}, {eigvals[:, -1].max():.3f}]')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    
    save_path = output_dir / f'metric_field_epoch_{epoch:03d}.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if wandb is not None and wandb.run is not None:
        wandb.log({f"metric_field": wandb.Image(fig)}, step=epoch)
        wandb.log({f"metric_field_image": wandb.Image(str(save_path))}, step=epoch)
        # Log scalar metrics
        wandb.log({
            "metric/condition_mean": float(cond.mean()),
            "metric/condition_max": float(cond.max()),
            "metric/logdet_mean": float(logdet.mean()),
            "metric/eig_min_mean": float(eigvals[:, 0].mean()),
            "metric/eig_max_mean": float(eigvals[:, -1].mean()),
        }, step=epoch)
    
    plt.close(fig)
    model.train()


def _pad_grid_to_latent(grid_pts: np.ndarray, latent_dim: int, device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(grid_pts, dtype=torch.float32, device=device)
    if latent_dim > tensor.shape[1]:
        pad = torch.zeros(tensor.shape[0], latent_dim - tensor.shape[1], device=device)
        tensor = torch.cat([tensor, pad], dim=1)
    elif latent_dim < tensor.shape[1]:
        tensor = tensor[:, :latent_dim]
    return tensor


def visualize_metric_tissot(model, epoch, output_dir, wandb, device, grid_res=16):
    """Render Tissot ellipses colored by condition number and log to WandB."""
    model.eval()
    if model.latent_dim < 2:
        return

    centroids = None
    c_tensor = _get_centroids_tensor(model)
    if isinstance(c_tensor, torch.Tensor):
        c_np = c_tensor.detach().cpu().numpy()
        if c_np.ndim == 2 and c_np.shape[1] >= 2:
            centroids = c_np[:, :2]

    if centroids is not None:
        x_min, y_min = centroids.min(axis=0)
        x_max, y_max = centroids.max(axis=0)
        x_range = np.linspace(x_min - 0.5, x_max + 0.5, grid_res)
        y_range = np.linspace(y_min - 0.5, y_max + 0.5, grid_res)
    else:
        axis = 5.0
        x_range = np.linspace(-axis, axis, grid_res)
        y_range = np.linspace(-axis, axis, grid_res)

    X, Y = np.meshgrid(x_range, y_range)
    grid_pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    with torch.no_grad():
        zs = _pad_grid_to_latent(grid_pts, model.latent_dim, device)
        G_inv = model.G_inv(zs)

    covs = G_inv[:, :2, :2].cpu().numpy()
    ratios = []
    ellipses = []
    for idx, cov in enumerate(covs):
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-8, None)
        ratio = float(eigvals[-1] / eigvals[0])
        ratios.append(ratio)
        width = 2 * np.sqrt(eigvals[-1])
        height = 2 * np.sqrt(eigvals[0])
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        ellipses.append((idx, width, height, angle))

    ratios_arr = np.array(ratios)
    norm = mcolors.Normalize(vmin=np.min(ratios_arr), vmax=np.max(ratios_arr) + 1e-8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array(ratios_arr)

    fig, ax = plt.subplots(figsize=(8, 8))
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c='black', alpha=0.4, s=18, label='Centroids')

    for idx, width, height, angle in ellipses:
        ellipse = patches.Ellipse(
            (grid_pts[idx, 0], grid_pts[idx, 1]),
            width,
            height,
            angle=angle,
            edgecolor=sm.to_rgba(ratios[idx]),
            facecolor='none',
            linewidth=1.0,
        )
        ax.add_patch(ellipse)

    ax.set_xlim(x_range[0], x_range[-1])
    ax.set_ylim(y_range[0], y_range[-1])
    ax.set_aspect('equal')
    ax.set_title(f'Epoch {epoch}: Metric Tissot Indicatrices')
    fig.colorbar(sm, ax=ax, label='Condition Number')
    plt.tight_layout()

    save_path = output_dir / f'metric_tissot_epoch_{epoch:03d}.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if wandb is not None and wandb.run is not None:
        wandb.log({
            'metric_tissot': wandb.Image(fig),
            'metric_tissot_condition_mean': float(ratios_arr.mean()),
            'metric_tissot_condition_max': float(ratios_arr.max()),
        }, step=epoch)
    plt.close(fig)
    model.train()


def train_rhvae_with_logging(
    model,
    train_data,
    val_data,
    epochs,
    batch_size,
    lr,
    device,
    output_dir,
    wandb,
    vis_every=10,
    max_centroids=None,
    target_n_centroids=None,
    centroid_seed=42,
    auto_temperature=False,
    auto_temperature_stat="median_nn",
    auto_temperature_every=0,
    temperature_scale=1.0,
):
    """Training loop with full logging and visualization."""
    model = model.to(device)
    model.train()

    train_dataset = TensorDataset(train_data)
    n_train = int(len(train_dataset))
    if n_train <= 0:
        raise RuntimeError("Empty training dataset after frame extraction; cannot train RHVAE.")
    effective_batch_size = int(min(max(1, int(batch_size)), n_train))
    if effective_batch_size != int(batch_size):
        print(
            "[RHVAE BASELINE] Adjusting batch_size for low-data regime: "
            f"requested={int(batch_size)} -> effective={effective_batch_size} (train_samples={n_train})"
        )
    # Keep the last (possibly small) batch so low-data regimes still produce updates.
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, drop_last=False)

    val_dataset = TensorDataset(val_data) if val_data is not None else None
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False) if val_dataset else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    history = {
        'train_loss': [], 'train_recon': [], 'train_kl': [],
        'train_prior': [], 'train_kinetic': [], 'train_logq': [],
        'val_loss': [], 'val_recon': [], 'val_kl': [],
        'val_prior': [], 'val_kinetic': [], 'val_logq': [],
    }
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_prior = 0.0
        epoch_kinetic = 0.0
        epoch_logq = 0.0
        n_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for (batch_data,) in pbar:
            batch_data = batch_data.to(device)
            
            optimizer.zero_grad()
            
            model_input = {"data": batch_data}
            model_output = model(model_input)
            
            loss = model_output.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            batch_size_actual = batch_data.shape[0]
            epoch_loss += loss.item() * batch_size_actual

            # RHVAE decomposition: recon_nll and remaining KL-like term
            with torch.no_grad():
                recon_x = model_output.recon_x
                z = model_output.z
                rho = model_output.rho
                G_inv = model_output.G_inv
                G_log_det = model_output.G_log_det
                log_var = model_output.log_var
                eps0 = model_output.eps0

                log_p_x_given_z = model._log_p_x_given_z(recon_x, batch_data)
                recon_nll = -log_p_x_given_z.mean()

                log_p_z = model._log_z(z)
                prior_nll = -log_p_z.mean()

                quad = torch.einsum("bi,bij,bj->b", rho, G_inv, rho)
                log_rho = -0.5 * quad - 0.5 * G_log_det
                kinetic_nll = -log_rho.mean()

                normal = torch.distributions.MultivariateNormal(
                    loc=torch.zeros(model.latent_dim, device=batch_data.device),
                    covariance_matrix=torch.eye(model.latent_dim, device=batch_data.device),
                )
                log_q = normal.log_prob(eps0) - 0.5 * log_var.sum(dim=1)
                log_q_mean = log_q.mean()

                kl_like = loss.detach() - recon_nll

            epoch_recon += recon_nll.item() * batch_size_actual
            epoch_kl += kl_like.item() * batch_size_actual
            epoch_prior += prior_nll.item() * batch_size_actual
            epoch_kinetic += kinetic_nll.item() * batch_size_actual
            epoch_logq += log_q_mean.item() * batch_size_actual
            n_samples += batch_size_actual
            
            pbar.set_postfix({'loss': f'{loss.item():.2f}'})
        
        avg_train_loss = epoch_loss / max(n_samples, 1)
        avg_train_recon = epoch_recon / max(n_samples, 1)
        avg_train_kl = epoch_kl / max(n_samples, 1)
        avg_train_prior = epoch_prior / max(n_samples, 1)
        avg_train_kinetic = epoch_kinetic / max(n_samples, 1)
        avg_train_logq = epoch_logq / max(n_samples, 1)
        
        history['train_loss'].append(avg_train_loss)
        history['train_recon'].append(avg_train_recon)
        history['train_kl'].append(avg_train_kl)
        history['train_prior'].append(avg_train_prior)
        history['train_kinetic'].append(avg_train_kinetic)
        history['train_logq'].append(avg_train_logq)
        
        # Update RHVAE metric once per epoch (matches Pythae trainer behavior)
        if hasattr(model, 'update'):
            with torch.no_grad():
                model.update()
        # If max_centroids is not set but auto_temperature is active,
        # cap with the requested n_centroids budget (fallback: model config).
        effective_max_centroids = max_centroids
        if effective_max_centroids is None and auto_temperature:
            if target_n_centroids is not None and int(target_n_centroids) > 0:
                effective_max_centroids = int(target_n_centroids)
            else:
                effective_max_centroids = getattr(model.model_config, 'n_centroids', None)
        _subsample_centroids(model, effective_max_centroids, centroid_seed)
        if auto_temperature:
            if int(auto_temperature_every) <= 0:
                should_update_temp = not getattr(model, "_auto_temperature_set", False)
            else:
                should_update_temp = (epoch % max(1, int(auto_temperature_every))) == 0
        else:
            should_update_temp = False
        if should_update_temp:
            centroids_now = _get_centroids_tensor(model)
            suggested = _suggest_temperature(centroids_now, stat=auto_temperature_stat)
            if suggested is not None:
                new_T = max(1e-6, float(temperature_scale) * suggested)
                with torch.no_grad():
                    model.temperature.fill_(new_T)
                    if hasattr(model, "update_physics_parameters"):
                        model.update_physics_parameters()
                model._auto_temperature_set = True
                print(
                    "[RHVAE BASELINE] Auto temperature set: "
                    f"T={new_T:.4f} (stat={auto_temperature_stat}, base={suggested:.4f})"
                )

        # Validation - RHVAE needs gradients for RHMC, so we keep requires_grad
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_recon = 0.0
            val_kl = 0.0
            val_prior = 0.0
            val_kinetic = 0.0
            val_logq = 0.0
            val_samples = 0
            
            # RHVAE uses RHMC which needs gradients - use enable_grad
            with torch.enable_grad():
                for (batch_data,) in val_loader:
                    batch_data = batch_data.to(device).requires_grad_(True)
                    model_input = {"data": batch_data}
                    model_output = model(model_input)
                    
                    batch_size_actual = batch_data.shape[0]
                    val_loss += model_output.loss.item() * batch_size_actual

                    with torch.no_grad():
                        recon_x = model_output.recon_x
                        z = model_output.z
                        rho = model_output.rho
                        G_inv = model_output.G_inv
                        G_log_det = model_output.G_log_det
                        log_var = model_output.log_var
                        eps0 = model_output.eps0

                        log_p_x_given_z = model._log_p_x_given_z(recon_x, batch_data)
                        recon_nll = -log_p_x_given_z.mean()

                        log_p_z = model._log_z(z)
                        prior_nll = -log_p_z.mean()

                        quad = torch.einsum("bi,bij,bj->b", rho, G_inv, rho)
                        log_rho = -0.5 * quad - 0.5 * G_log_det
                        kinetic_nll = -log_rho.mean()

                        normal = torch.distributions.MultivariateNormal(
                            loc=torch.zeros(model.latent_dim, device=batch_data.device),
                            covariance_matrix=torch.eye(model.latent_dim, device=batch_data.device),
                        )
                        log_q = normal.log_prob(eps0) - 0.5 * log_var.sum(dim=1)
                        log_q_mean = log_q.mean()

                        kl_like = model_output.loss.detach() - recon_nll

                    val_recon += recon_nll.item() * batch_size_actual
                    val_kl += kl_like.item() * batch_size_actual
                    val_prior += prior_nll.item() * batch_size_actual
                    val_kinetic += kinetic_nll.item() * batch_size_actual
                    val_logq += log_q_mean.item() * batch_size_actual
                    val_samples += batch_size_actual
            
            avg_val_loss = val_loss / max(val_samples, 1)
            avg_val_recon = val_recon / max(val_samples, 1)
            avg_val_kl = val_kl / max(val_samples, 1)
            avg_val_prior = val_prior / max(val_samples, 1)
            avg_val_kinetic = val_kinetic / max(val_samples, 1)
            avg_val_logq = val_logq / max(val_samples, 1)
            
            history['val_loss'].append(avg_val_loss)
            history['val_recon'].append(avg_val_recon)
            history['val_kl'].append(avg_val_kl)
            history['val_prior'].append(avg_val_prior)
            history['val_kinetic'].append(avg_val_kinetic)
            history['val_logq'].append(avg_val_logq)
            
            scheduler.step(avg_val_loss)
            model.train()
        else:
            scheduler.step(avg_train_loss)
        
        # Log to wandb
        if wandb is not None and wandb.run is not None:
            log_dict = {
                "train/loss": avg_train_loss,
                "train/recon_nll": avg_train_recon,
                "train/kl_like": avg_train_kl,
                "train/prior_nll": avg_train_prior,
                "train/kinetic_nll": avg_train_kinetic,
                "train/log_q": avg_train_logq,
                "train/epoch": epoch,
                "lr": optimizer.param_groups[0]['lr'],
            }
            if val_loader is not None:
                log_dict.update({
                    "val/loss": avg_val_loss,
                    "val/recon_nll": avg_val_recon,
                    "val/kl_like": avg_val_kl,
                    "val/prior_nll": avg_val_prior,
                    "val/kinetic_nll": avg_val_kinetic,
                    "val/log_q": avg_val_logq,
                })
            wandb.log(log_dict, step=epoch)
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            val_str = f", val_loss={avg_val_loss:.4f}" if val_loader else ""
            print(
                f"Epoch {epoch}: loss={avg_train_loss:.4f}, recon_nll={avg_train_recon:.4f}, "
                f"kl_like={avg_train_kl:.4f}{val_str}"
            )
        
        # Visualizations
        if epoch == 1 or epoch % vis_every == 0 or epoch == epochs:
            visualize_reconstructions(model, train_data, epoch, output_dir, wandb, device)
            visualize_latent_space(model, train_data, epoch, output_dir, wandb, device)
            visualize_metric_field(model, epoch, output_dir, wandb, device)
            visualize_metric_tissot(model, epoch, output_dir, wandb, device)
    
    return model, history


def plot_training_curves(history, output_dir, wandb):
    """Plot training curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    if history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Recon loss
    axes[1].plot(epochs, history['train_recon'], 'b-', label='Train')
    if history['val_recon']:
        axes[1].plot(epochs, history['val_recon'], 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Recon Loss (MSE)')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL loss
    axes[2].plot(epochs, history['train_kl'], 'b-', label='Train')
    if history['val_kl']:
        axes[2].plot(epochs, history['val_kl'], 'r-', label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Loss')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / 'training_curves.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if wandb is not None and wandb.run is not None:
        wandb.log({"training_curves": wandb.Image(fig)})
    
    plt.close(fig)
    print(f"[RHVAE BASELINE] Saved training curves to: {save_path}")


def main():
    import argparse
    aniso_2d_only_keys = {
        "atom_power",
        "void_threshold",
        "void_decay_scale",
        "void_decay_power",
        "radial_stretch",
        "transition_steepness",
        "attractor_gamma",
    }
    geometry_cases = {
        "baseline": {
            "kernel_type": "isotropic",
            "use_attractor": False,
            "rhvae_variant": "standard",
        },
        "highway": {
            "kernel_type": "mahalanobis",
            "use_attractor": False,
            "rhvae_variant": "geometry",
        },
        "hard_funnel": {
            "kernel_type": "mahalanobis",
            "use_attractor": True,
            "attractor_smoothness": "hard",
            "attractor_metric": "euclidean",
            "attractor_use_det": False,
            "rhvae_variant": "geometry",
        },
        "smooth_funnel": {
            "kernel_type": "mahalanobis",
            "use_attractor": True,
            "attractor_smoothness": "soft",
            "attractor_metric": "euclidean",
            "attractor_use_det": False,
            "attractor_k_nearest": 10,
            "atom_norm": "trace",
            "rhvae_variant": "geometry",
        },
        "gravity_well": {
            "kernel_type": "mahalanobis",
            "use_attractor": True,
            "attractor_smoothness": "soft",
            "attractor_metric": "euclidean",
            "attractor_use_det": True,
            "attractor_k_nearest": 10,
            "atom_norm": "trace",
            "rhvae_variant": "geometry",
        },
        "aniso": {
            "kernel_type": "mahalanobis",
            "atom_norm": "trace",
            "atom_power": 1.0959864113997024,
            "kernel_power": 1.0,
            "precision_jitter": 0.01,
            "void_threshold": 1.2,
            "void_weight_threshold": -1.0,
            "void_decay_type": "invquad",
            "void_decay_scale": 8.87131893173153,
            "void_decay_power": 1.7447710342008058,
            "void_decay_softplus_k": 5.0,
            "radial_stretch": 9.252860449508804,
            "transition_steepness": 7.276725930229776,
            "use_attractor": True,
            "attractor_smoothness": "soft",
            "attractor_metric": "mahalanobis",
            "attractor_use_det": True,
            "attractor_gamma": 7.624554217806352,
            "attractor_k_nearest": 1,
            "attractor_bias_energy": 18.0,
            "regularization": 0.05,
            "rhmc_volume_power": 1.0,
            "rhvae_variant": "geometry",
        },
        "physics": {
            "kernel_type": "mahalanobis",
            "atom_norm": "trace",
            "atom_power": 1.0959864113997024,
            "kernel_power": 1.0,
            "precision_jitter": 0.01,
            "void_weight_threshold": -1.0,
            "void_decay_type": "invquad",
            "void_decay_softplus_k": 5.0,
            "use_attractor": True,
            "attractor_smoothness": "soft",
            "attractor_metric": "mahalanobis",
            "attractor_use_det": True,
            "attractor_k_nearest": 1,
            "attractor_bias_energy": 18.0,
            "rhvae_variant": "geometry",
            "use_physics_init": True,
        },
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--n_centroids', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--regularization', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='outputs/pythae_rhvae_baseline')
    parser.add_argument('--num_sequences', type=int, default=800)
    parser.add_argument('--frame_mode', type=str, default='t0', choices=['t0', 'all'])
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--max_centroids', type=int, default=None)
    parser.add_argument('--auto_temperature', action='store_true')
    parser.add_argument(
        '--auto_temperature_stat',
        type=str,
        default='median_nn',
        choices=['median_nn', 'mean_nn', 'mean_pairwise', 'median_pairwise', 'silverman', 'mean_knn_5'],
    )
    parser.add_argument(
        '--auto_temperature_every',
        type=int,
        default=0,
        help='0 sets temperature once after metric update; >0 updates every N epochs.',
    )
    parser.add_argument('--temperature_scale', type=float, default=1.0)
    parser.add_argument('--vis_every', type=int, default=10, help='Visualize every N epochs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rhvae_variant', type=str, default='standard', choices=['standard', 'attractor', 'geometry'])
    parser.add_argument(
        '--geometry_case',
        type=str,
        default=None,
        choices=sorted(geometry_cases.keys()),
        help='Preset geometry case (overrides relevant attractor/kernel args)',
    )
    parser.add_argument('--use_attractor', action='store_true', help='Enable attractor metric (attractor variant)')
    parser.add_argument('--use_attractor_value', type=int, default=None, help='Explicit 0/1 for sweeps')
    parser.add_argument('--attractor_smoothness', type=str, default='hard', choices=['hard', 'soft'])
    parser.add_argument('--attractor_metric', type=str, default='euclidean', choices=['euclidean', 'mahalanobis'])
    parser.add_argument('--attractor_gamma', type=float, default=5.0)
    parser.add_argument('--attractor_k_nearest', type=int, default=5)
    parser.add_argument('--attractor_use_det', action='store_true')
    parser.add_argument('--attractor_use_det_value', type=int, default=None, help='Explicit 0/1 for sweeps')
    parser.add_argument('--attractor_bias_energy', type=float, default=15.0)
    parser.add_argument('--void_threshold', type=float, default=1.5)
    parser.add_argument('--void_weight_threshold', type=float, default=-1.0)
    parser.add_argument(
        '--void_decay_type',
        type=str,
        default='invquad',
        choices=['none', 'invquad'],
    )
    parser.add_argument('--void_decay_scale', type=float, default=1.0)
    parser.add_argument('--void_decay_power', type=float, default=2.0)
    parser.add_argument('--void_decay_softplus_k', type=float, default=5.0)
    parser.add_argument(
        '--void_eigshape_mode',
        type=str,
        default='none',
        choices=['none', 'det_preserving_spectral'],
        help='Optional determinant-preserving spectral reshaping for void branch.',
    )
    # Add dataset argument
    parser.add_argument(
        '--dataset',
        type=str,
        default='ellipses',
        help='Which dataset to load (ellipses, rotmnist).',
    )
    parser.add_argument(
        '--void_eigshape_alpha_min',
        type=float,
        default=1.0,
        help='Apply void eigenshape only where alpha >= threshold.',
    )
    parser.add_argument('--use_physics_init', action='store_true', help='Use from_physics to derive hyperparameters.')
    parser.add_argument(
        '--void_eigshape_power',
        type=float,
        default=-1.0,
        help="Eigenshape exponent s in lambda' = g * (lambda/g)^s.",
    )
    parser.add_argument(
        '--void_eigshape_eig_floor',
        type=float,
        default=1e-8,
        help='Eigenvalue floor used during eigenshape reconstruction.',
    )
    parser.add_argument('--radial_stretch', type=float, default=10.0)
    parser.add_argument('--transition_steepness', type=float, default=5.0)
    parser.add_argument('--kernel_type', type=str, default='isotropic', choices=['isotropic', 'mahalanobis'])
    parser.add_argument('--precision_jitter', type=float, default=1e-6)
    parser.add_argument('--atom_power', type=float, default=1.0)
    parser.add_argument('--kernel_power', type=float, default=1.0)
    parser.add_argument('--atom_norm', type=str, default='trace', choices=['none', 'trace', 'det'])
    parser.add_argument(
        '--rhmc_integrator',
        type=str,
        default='explicit',
        choices=['explicit', 'implicit'],
        help='RHMC integrator for training (geometry variant).',
    )
    parser.add_argument('--rhmc_fp_steps', type=int, default=6)
    parser.add_argument('--rhmc_fp_damping', type=float, default=0.5)
    parser.add_argument('--atom_scale', type=float, default=1.0)
    parser.add_argument(
        '--use_dual_metric',
        nargs='?',
        const='True',
        default='False',
        type=lambda x: (str(x).lower() == 'true') if str(x).lower() in ['true', 'false'] else x,
    )
    parser.add_argument('--rhmc_adaptive_dual_step', action='store_true', default=False)
    parser.add_argument('--rhmc_adaptive_max_dual_displacement', type=float, default=0.5)
    parser.add_argument('--rhmc_adaptive_min_step_scale', type=float, default=0.05)
    parser.add_argument('--rhmc_volume_power', type=float, default=2.0)
    parser.add_argument('--rhmc_radial_prior_weight', type=float, default=0.1)
    parser.add_argument('--rhmc_momentum_persist', type=float, default=0.0)
    parser.add_argument('--sampling_mcmc_steps', type=int, default=50)
    parser.add_argument('--sampling_n_lf', type=int, default=10)
    parser.add_argument('--sampling_eps_lf', type=float, default=0.03)
    parser.add_argument('--wandb_project', type=str, default='RHVAE-baseline')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_tags', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_name_mode', type=str, default='timestamp', choices=['timestamp', 'auto', 'manual'])
    parser.add_argument('--analysis_skip_distortion', action='store_true', help='Skip distortion analysis plots.')
    parser.add_argument('--analysis_far_pairs', type=int, default=0, help='Number of far centroid pairs for analysis geodesics.')
    parser.add_argument('--analysis_random_pairs', type=int, default=8, help='Number of random centroid pairs for analysis geodesics.')
    parser.add_argument('--skip_analysis', action='store_true', help='Skip metric analysis after training.')
    parser.add_argument(
        '--analysis_rhmc_sampler',
        type=str,
        default=None,
        choices=['riemannian', 'geodesic', 'volume', 'volume_riemannian'],
        help='RHMC sampler for analysis (riemannian, geodesic-uniform, volume-element, or volume_riemannian).',
    )
    # Sampling diagnostics arguments
    parser.add_argument('--skip_sampling_diagnostics', action='store_true', help='Skip sampling diagnostics after training.')
    parser.add_argument('--sampling_fid_samples', type=int, default=1000, help='Number of samples for FID computation.')
    parser.add_argument(
        '--sampling_fid_samplers',
        type=str,
        nargs='+',
        default=None,
        help='Samplers to evaluate for FID.',
    )
    parser.add_argument('--sampling_n_starts', type=int, default=12, help='Number of starting points for multi-start sampling.')
    parser.add_argument('--sampling_n_interp_pairs', type=int, default=4, help='Number of interpolation pairs.')
    parser.add_argument('--sampling_quality_samples', type=int, default=500, help='Samples for quality metrics.')
    parser.add_argument('--sampling_n_chains', type=int, default=4, help='Number of RHMC chains for diagnostics.')
    parser.add_argument('--sampling_chain_length', type=int, default=100, help='Length of RHMC chains.')
    args = parser.parse_args()

    if args.geometry_case is not None:
        preset = dict(geometry_cases[args.geometry_case])
        if args.geometry_case == "aniso":
            physics_mode = bool(args.use_physics_init) or int(args.latent_dim) > 2
            if physics_mode:
                for key in aniso_2d_only_keys:
                    preset.pop(key, None)
                if not args.use_physics_init:
                    args.use_physics_init = True
                    print(
                        "[RHVAE BASELINE] geometry_case=aniso with latent_dim>2: "
                        "enabling --use_physics_init to avoid 2D-only metric presets."
                    )
                print(
                    "[RHVAE BASELINE] geometry_case=aniso in physics mode: "
                    "keeping shared anisotropic defaults (including regularization=0.05 and rhmc_volume_power=1.0) "
                    "and deriving dimension-sensitive geometry from physics."
                )
        for key, value in preset.items():
            setattr(args, key, value)
        args.use_attractor_value = None
        args.attractor_use_det_value = None
        print(f"[RHVAE BASELINE] Using geometry_case preset: {args.geometry_case}")

    if args.attractor_use_det_value is not None:
        args.attractor_use_det = bool(args.attractor_use_det_value)
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[RHVAE BASELINE] Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    use_attractor_flag = args.use_attractor
    if args.use_attractor_value is not None:
        use_attractor_flag = bool(args.use_attractor_value)
    use_attractor = use_attractor_flag or args.rhvae_variant == "attractor"
    if args.analysis_rhmc_sampler is None:
        args.analysis_rhmc_sampler = "volume"

    if args.sampling_fid_samplers is None:
        args.sampling_fid_samplers = ["gaussian", "volume"]

    print(f"[RHVAE BASELINE] Analysis RHMC sampler: {args.analysis_rhmc_sampler}")
    print(f"[RHVAE BASELINE] FID samplers: {args.sampling_fid_samplers}")

    # Initialize wandb FIRST (before training)
    wandb = None
    try:
        import wandb as wandb_module
        wandb = wandb_module
        run_name = args.wandb_run_name
        if args.wandb_name_mode == "auto":
            run_name = (
                f"geom_{args.kernel_type}"
                f"_ap{args.atom_power}_kp{args.kernel_power}"
                f"_attr{int(use_attractor)}"
                f"_ns{args.num_sequences}"
                f"_mf{args.max_frames if args.max_frames is not None else 'all'}"
                f"_seed{args.seed}"
            )
        elif args.wandb_name_mode == "timestamp" or run_name is None:
            run_name = f"rhvae_baseline_{timestamp}"

        tags = [tag.strip() for tag in args.wandb_tags.split(",")] if args.wandb_tags else None
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            tags=tags,
            name=run_name,
            config={
                "latent_dim": args.latent_dim,
                "n_centroids": args.n_centroids,
                "temperature": args.temperature,
                "regularization": args.regularization,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "num_sequences": args.num_sequences,
                "frame_mode": args.frame_mode,
                "max_frames": args.max_frames,
                "seed": args.seed,
                "auto_temperature": args.auto_temperature,
                "auto_temperature_stat": args.auto_temperature_stat,
                "auto_temperature_every": args.auto_temperature_every,
                "temperature_scale": args.temperature_scale,
                "rhvae_variant": args.rhvae_variant,
                "use_attractor": use_attractor,
                "void_threshold": args.void_threshold,
                "void_weight_threshold": args.void_weight_threshold,
                "void_decay_type": args.void_decay_type,
                "void_decay_scale": args.void_decay_scale,
                "void_decay_power": args.void_decay_power,
                "void_decay_softplus_k": args.void_decay_softplus_k,
                "void_eigshape_mode": args.void_eigshape_mode,
                "void_eigshape_alpha_min": args.void_eigshape_alpha_min,
                "void_eigshape_power": args.void_eigshape_power,
                "void_eigshape_eig_floor": args.void_eigshape_eig_floor,
                "radial_stretch": args.radial_stretch,
                "transition_steepness": args.transition_steepness,
                "kernel_type": args.kernel_type,
                "precision_jitter": args.precision_jitter,
                "atom_power": args.atom_power,
                "kernel_power": args.kernel_power,
                "atom_norm": args.atom_norm,
                "attractor_smoothness": args.attractor_smoothness,
                "attractor_metric": args.attractor_metric,
                "attractor_gamma": args.attractor_gamma,
                "attractor_k_nearest": args.attractor_k_nearest,
                "attractor_use_det": args.attractor_use_det,
                "attractor_bias_energy": args.attractor_bias_energy,
                "geometry_case": args.geometry_case,
                "skip_analysis": args.skip_analysis,
                "analysis_rhmc_sampler": args.analysis_rhmc_sampler,
                "use_dual_metric": args.use_dual_metric,
                "rhmc_adaptive_dual_step": args.rhmc_adaptive_dual_step,
                "rhmc_adaptive_max_dual_displacement": args.rhmc_adaptive_max_dual_displacement,
                "rhmc_adaptive_min_step_scale": args.rhmc_adaptive_min_step_scale,
                "rhmc_volume_power": args.rhmc_volume_power,
                "rhmc_radial_prior_weight": args.rhmc_radial_prior_weight,
                "rhmc_momentum_persist": args.rhmc_momentum_persist,
                "sampling_mcmc_steps": args.sampling_mcmc_steps,
                "sampling_n_lf": args.sampling_n_lf,
                "sampling_eps_lf": args.sampling_eps_lf,
            }
        )
        print("[RHVAE BASELINE] WandB initialized")
    except Exception as e:
        print(f"[RHVAE BASELINE] WandB init failed: {e}")
        wandb = None
    
    # Load data
    frame_mode = args.frame_mode
    if getattr(args, "dataset", "ellipses") == "rotmnist":
        from src.utils.low_data_io import load_split_tensor, load_subset_indices, subset_from_indices
        print(f"[RHVAE BASELINE] Loading RotMNIST low-data subset: N={args.max_frames if args.max_frames else args.num_sequences}")
        # Need to dynamically load the prepared rotmnist split
        # the subset_n is carried over via max_frames in our Hydra translation
        rotmnist_dir = PROJECT_ROOT / "data" / "processed" / "rotmnist" / "v1"
        if not rotmnist_dir.exists():
            import subprocess
            print("[RHVAE BASELINE] RotMNIST not found. Running preparation script...")
            subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "prepare_rotmnist_lowdata.py")], check=True)
            
        x_train, _ = load_split_tensor(rotmnist_dir, "train")
        x_val, _ = load_split_tensor(rotmnist_dir, "val")
        
        subset_n = args.max_frames if args.max_frames is not None else 50
        try:
            indices_dict = load_subset_indices(rotmnist_dir, "train")
            idx_train = indices_dict[subset_n][str(args.seed)]
            train_data = subset_from_indices(x_train, idx_train)
        except Exception as e:
            print(f"[RHVAE BASELINE] Fallback exact extraction. Error: {e}")
            g = torch.Generator().manual_seed(args.seed)
            idx_train = torch.randperm(x_train.size(0), generator=g)[:subset_n]
            train_data = x_train[idx_train]
            
        g_val = torch.Generator().manual_seed(args.seed + 1)
        # Scale validation set down proportionally
        val_n = max(1, int(subset_n * 0.2)) 
        idx_val = torch.randperm(x_val.size(0), generator=g_val)[:val_n]
        val_data = x_val[idx_val]
        
        # RotMNIST is 1x28x28
        train_data = train_data.view(-1, 1, 28, 28)
        val_data = val_data.view(-1, 1, 28, 28)
    else:
        print("[RHVAE BASELINE] Loading ellipse data...")
        from omegaconf import OmegaConf  # pyright: ignore[reportMissingImports]
        data_config = OmegaConf.create({
            'num_sequences': args.num_sequences,
            'seq_len': 8,
            'sequence_length': 8,
            'image_size': [64, 64],
            'batch_size': args.batch_size,
            'num_workers': 0,
            'min_radius': 8,
            'max_radius': 20,
            'min_eccentricity': 0.0,
            'max_eccentricity': 0.9,
            'fix_center': True,
            'fix_theta': True,
            'fix_intensity': True,
            'keep_major_axis_constant': True,
            'keep_area_constant': False,
            'outline_only': False,
            'outline_width': 2,
            'antialias': True,
            'supersample_factor': 4,
            'seed': args.seed,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
        })
        
        data_module = EllipseSequenceDataModule(data_config)
        train_data = create_frames_dataset(
            data_module,
            'train',
            frame_mode=frame_mode,
            max_frames=args.max_frames,
            seed=args.seed,
        )
        val_data = create_frames_dataset(
            data_module,
            'val',
            frame_mode=frame_mode,
            max_frames=args.max_frames,
            seed=args.seed + 1,
        )
    
    # Flatten images for RHVAE (it expects [B, input_dim])
    input_dim = train_data.shape[1] * train_data.shape[2] * train_data.shape[3]
    train_flat = train_data.view(train_data.shape[0], -1)
    val_flat = val_data.view(val_data.shape[0], -1) if val_data is not None else None
    
    print(f"[RHVAE BASELINE] Training data shape: {train_flat.shape}")
    if val_flat is not None:
        print(f"[RHVAE BASELINE] Validation data shape: {val_flat.shape}")
    
    # User Request: Adjust the number of centroids to the number of points natively
    n_points = train_flat.shape[0]
    capped_centroids = min(n_points, 500)
    args.n_centroids = capped_centroids
    args.max_centroids = capped_centroids
    
    # Create RHVAE config
    use_geometry = args.kernel_type != "isotropic" or use_attractor or args.rhvae_variant == "geometry"
    use_physics_init = bool(getattr(args, "use_physics_init", False)) and use_geometry
    config_cls = GeometryRHVAEConfig if use_geometry else RHVAEConfig
    config_kwargs = dict(
        input_dim=(input_dim,),
        latent_dim=args.latent_dim,
        n_lf=3,  # Leapfrog steps
        eps_lf=0.001,  # Leapfrog step size
        beta_zero=0.3,  # Initial temperature
        temperature=args.temperature,
        regularization=args.regularization,
    )
    # Pythae RHVAEConfig does not accept n_centroid_candidates; keep config minimal.
    if use_geometry:
        geometry_kwargs = dict(
            use_attractor=use_attractor,
            void_decay_type=args.void_decay_type,
            void_decay_softplus_k=args.void_decay_softplus_k,
            void_eigshape_mode=args.void_eigshape_mode,
            void_eigshape_alpha_min=args.void_eigshape_alpha_min,
            void_eigshape_power=args.void_eigshape_power,
            void_eigshape_eig_floor=args.void_eigshape_eig_floor,
            kernel_type=args.kernel_type,
            precision_jitter=args.precision_jitter,
            atom_power=args.atom_power,
            kernel_power=args.kernel_power,
            atom_norm=args.atom_norm,
            attractor_smoothness=args.attractor_smoothness,
            attractor_metric=args.attractor_metric,
            attractor_k_nearest=args.attractor_k_nearest,
            attractor_use_det=args.attractor_use_det,
            attractor_bias_energy=args.attractor_bias_energy,
            rhmc_integrator=args.rhmc_integrator,
            rhmc_fp_steps=args.rhmc_fp_steps,
            rhmc_fp_damping=args.rhmc_fp_damping,
            rhmc_adaptive_dual_step=args.rhmc_adaptive_dual_step,
            rhmc_adaptive_max_dual_displacement=args.rhmc_adaptive_max_dual_displacement,
            rhmc_adaptive_min_step_scale=args.rhmc_adaptive_min_step_scale,
            atom_scale=args.atom_scale,
        )
        if use_physics_init:
            print(
                "[RHVAE BASELINE] --use_physics_init active: deriving "
                "void_threshold/attractor_gamma/transition_steepness/void_decay_power/"
                "void_decay_scale/radial_stretch from (temperature, latent_dim)."
            )
        else:
            geometry_kwargs.update(
                void_threshold=args.void_threshold,
                void_weight_threshold=args.void_weight_threshold,
                void_decay_scale=args.void_decay_scale,
                void_decay_power=args.void_decay_power,
                radial_stretch=args.radial_stretch,
                transition_steepness=args.transition_steepness,
                attractor_gamma=args.attractor_gamma,
            )
        config_kwargs.update(geometry_kwargs)
    if use_physics_init:
        rhvae_config = config_cls.from_physics(**config_kwargs)
    else:
        rhvae_config = config_cls(**config_kwargs)

    print(
        "[RHVAE BASELINE] RHVAE config: "
        f"latent_dim={args.latent_dim}, n_centroids={args.n_centroids}, "
        f"T={args.temperature}, variant={args.rhvae_variant}, "
        f"kernel={args.kernel_type}, attractor={use_attractor}, "
    )

    # Create model
    model_cls = GeometryRHVAE if use_geometry else RHVAE
    
    if getattr(args, "dataset", "ellipses") == "rotmnist":
        from pythae.models.nn import BaseEncoder, BaseDecoder
        from pythae.models.base.base_utils import ModelOutput
        import torch.nn as nn
        
        class CNNEncoderRotMNIST(BaseEncoder):
            def __init__(self, config):
                super().__init__()
                self.input_dim = config.input_dim
                self.latent_dim = config.latent_dim
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
                    nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(16, 64), nn.SiLU(),
                    nn.Conv2d(64, 128, 3, 2, 0), nn.GroupNorm(32, 128), nn.SiLU(),
                    nn.Flatten()
                )
                self.embedding = nn.Linear(128 * 3 * 3, self.latent_dim)
                self.log_var = nn.Linear(128 * 3 * 3, self.latent_dim)
                
            def forward(self, x):
                out = self.conv(x.view(-1, 1, 28, 28))
                return ModelOutput(
                    embedding=self.embedding(out),
                    log_covariance=self.log_var(out)
                )
                
        class CNNDecoderRotMNIST(BaseDecoder):
            def __init__(self, config):
                super().__init__()
                self.input_dim = config.input_dim
                self.latent_dim = config.latent_dim
                self.fc = nn.Sequential(nn.Linear(self.latent_dim, 128 * 3 * 3), nn.SiLU())
                self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 3, 2, 0), nn.GroupNorm(16, 64), nn.SiLU(),
                    nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
                    nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
                )

            def forward(self, z):
                out = self.fc(z).view(-1, 128, 3, 3)
                out = self.deconv(out)
                return ModelOutput(reconstruction=out.view((z.shape[0],) + self.input_dim))

        print("[RHVAE BASELINE] Using 3-layer CNN architecture for RotMNIST")
        encoder = CNNEncoderRotMNIST(rhvae_config)
        decoder = CNNDecoderRotMNIST(rhvae_config)
        model = model_cls(rhvae_config, encoder=encoder, decoder=decoder)
    else:
        model = model_cls(rhvae_config)
        
    model = model.to(device)
    
    # Train with full logging!
    print(f"[RHVAE BASELINE] Starting training for {args.epochs} epochs...")
    model, history = train_rhvae_with_logging(
        model, train_flat, val_flat, args.epochs, args.batch_size, args.lr, 
        device, output_dir, wandb, vis_every=args.vis_every,
        max_centroids=args.max_centroids,
        target_n_centroids=args.n_centroids,
        centroid_seed=args.seed,
        auto_temperature=args.auto_temperature,
        auto_temperature_stat=args.auto_temperature_stat,
        auto_temperature_every=args.auto_temperature_every,
        temperature_scale=args.temperature_scale,
    )
    
    print("[RHVAE BASELINE] Training complete!")
    
    # Plot final training curves
    plot_training_curves(history, output_dir, wandb)
    
    # Get centroids from trained model
    centroids = _get_centroids_tensor(model)
    if isinstance(centroids, torch.Tensor):
        print(f"[RHVAE BASELINE] Learned {centroids.shape[0]} centroids")
        
        # Final metric visualization
        visualize_metric_field(model, args.epochs, output_dir, wandb, device)
        
        # Get metric matrices
        M_tens = None
        if hasattr(model, 'M_tens') and isinstance(model.M_tens, torch.Tensor) and model.M_tens.numel() > 0:
            M_tens = model.M_tens
        elif hasattr(model, 'M'):
            M_raw = model.M
            if hasattr(M_raw, '__iter__') and not isinstance(M_raw, torch.Tensor):
                if len(M_raw) > 0:
                    M_tens = torch.stack(list(M_raw))
            else:
                M_tens = M_raw
        
        # Save metric data
        config_dict = rhvae_config.dict() if hasattr(rhvae_config, "dict") else dict(rhvae_config.__dict__)
        metric_data = {
            'centroids': centroids.cpu(),
            'metric_matrices': M_tens.cpu() if M_tens is not None else None,
            'temperature': float(model.temperature.item()) if hasattr(model, "temperature") else float(args.temperature),
            'regularization': float(model.lbd.item()) if hasattr(model, "lbd") else float(args.regularization),
            'config': config_dict,
        }
        metric_path = output_dir / 'rhvae_metric.pt'
        torch.save(metric_data, metric_path)
        print(f"[RHVAE BASELINE] Saved metric to: {metric_path}")
        
        # Save training data for standalone sampling diagnostics
        train_data_path = output_dir / 'train_data.pt'
        torch.save({'data': train_flat.cpu(), 'shape': train_data.shape}, train_data_path)
        print(f"[RHVAE BASELINE] Saved training data to: {train_data_path}")
        
        if wandb is not None and wandb.run is not None:
            wandb.save(str(metric_path))
        if args.skip_analysis:
            print("[RHVAE BASELINE] Skipping metric analysis (--skip_analysis).")
        else:
            analysis_dir = output_dir / "analysis_results"
            support_images = train_data[: min(512, train_data.shape[0])].clone()
            run_analysis(
                output_dir,
                analysis_dir,
                torch.device(device),
                wandb_run=wandb.run if wandb is not None and wandb.run is not None else None,
                skip_distortion=args.analysis_skip_distortion,
                far_pairs=args.analysis_far_pairs,
                random_pairs=args.analysis_random_pairs,
                rhmc_sampler=args.analysis_rhmc_sampler,
                support_images=support_images,
                support_max_samples=512,
                use_dual_metric=args.use_dual_metric,
                adaptive_dual_step=args.rhmc_adaptive_dual_step,
                adaptive_max_dual_displacement=args.rhmc_adaptive_max_dual_displacement,
                adaptive_min_step_scale=args.rhmc_adaptive_min_step_scale,
                rhmc_mcmc_steps=args.sampling_mcmc_steps,
                rhmc_n_lf=args.sampling_n_lf,
                rhmc_eps_lf=args.sampling_eps_lf,
                rhmc_beta_zero=1.0,
                rhmc_volume_power=args.rhmc_volume_power,
                rhmc_radial_prior_weight=args.rhmc_radial_prior_weight,
                rhmc_momentum_persist=args.rhmc_momentum_persist,
                rhmc_fp_steps=args.rhmc_fp_steps,
                rhmc_fp_damping=args.rhmc_fp_damping,
            )
        
    # Save model (before diagnostics so standalone runs can also load it)
    model_save_path = output_dir / 'rhvae_model.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f"[RHVAE BASELINE] Saved model to: {model_save_path}")
    
    # Run sampling diagnostics (FID, interpolation, multi-start, quality metrics)
    if not args.skip_sampling_diagnostics:
        print("[RHVAE BASELINE] Running sampling diagnostics...")
        sampling_results = run_sampling_diagnostics(
            model_path=output_dir,
            real_data=train_flat,
            output_dir=output_dir / "sampling_diagnostics",
            device=torch.device(device),
            wandb_run=wandb.run if wandb is not None and wandb.run is not None else None,
            model=model,  # Pass model directly to avoid reloading
            run_fid=True,
            fid_samples=args.sampling_fid_samples,
            fid_samplers=args.sampling_fid_samplers,
            run_multi_start=True,
            n_starts=args.sampling_n_starts,
            multi_start_sampler=args.analysis_rhmc_sampler,
            run_interpolation=True,
            n_interp_pairs=args.sampling_n_interp_pairs,
            run_quality=True,
            quality_samples=args.sampling_quality_samples,
            quality_samplers=args.sampling_fid_samplers,
            do_rhmc_diagnostics=True,
            n_chains=args.sampling_n_chains,
            chain_length=args.sampling_chain_length,
            rhmc_sampler=args.analysis_rhmc_sampler,
            mcmc_steps=args.sampling_mcmc_steps,
            n_lf=args.sampling_n_lf,
            eps_lf=args.sampling_eps_lf,
            use_dual_metric=args.use_dual_metric,
            adaptive_dual_step=args.rhmc_adaptive_dual_step,
            adaptive_max_dual_displacement=args.rhmc_adaptive_max_dual_displacement,
            adaptive_min_step_scale=args.rhmc_adaptive_min_step_scale,
            volume_power=args.rhmc_volume_power,
            radial_prior_weight=args.rhmc_radial_prior_weight,
            momentum_persist=args.rhmc_momentum_persist,
            fp_steps=args.rhmc_fp_steps,
            fp_damping=args.rhmc_fp_damping,
        )
        print("[RHVAE BASELINE] Sampling diagnostics complete.")
    
    # Save training history
    history_path = output_dir / 'training_history.pt'
    torch.save(history, history_path)
    print(f"[RHVAE BASELINE] Saved history to: {history_path}")
    
    print(f"\n[RHVAE BASELINE] All outputs saved to: {output_dir}")
    
    # Finish wandb
    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
