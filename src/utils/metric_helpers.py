from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import torch


def _pick_first(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _resolve_metric_path(model_path: Path | str) -> Path:
    path = Path(model_path)
    if path.is_dir():
        metric_path = path / "rhvae_metric.pt"
    else:
        metric_path = path
    if not metric_path.exists():
        raise FileNotFoundError(metric_path)
    return metric_path


def load_metric_bundle(
    metric_path: Path | str,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, Optional[dict[str, Any]]]:
    """Load centroids, atoms, temperature, regularization, and config dict from rhvae_metric.pt."""
    path = Path(metric_path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    centroids = _pick_first(state.get("centroids"), state.get("centroids_tens"))
    atoms = _pick_first(state.get("M_matrices"), state.get("metric_matrices"), state.get("M_tens"), state.get("M"))
    if centroids is None or atoms is None:
        raise ValueError(f"Metric file missing centroids or atoms: {path}")
    temperature = float(state.get("temperature", 0.5))
    regularization = float(state.get("regularization", 1e-3))
    config = state.get("config")
    if not isinstance(atoms, torch.Tensor):
        atoms = torch.as_tensor(atoms)
    if not isinstance(centroids, torch.Tensor):
        centroids = torch.as_tensor(centroids)
    return centroids, atoms, temperature, regularization, config


def load_learned_atoms(model_path: Path | str) -> torch.Tensor:
    """Load learned atoms from rhvae_metric.pt (or given metric file)."""
    metric_path = _resolve_metric_path(model_path)
    _, atoms, _, _, _ = load_metric_bundle(metric_path)
    return atoms


def _extract_batch_frames(batch, frame_mode: str) -> Optional[torch.Tensor]:
    if isinstance(batch, dict):
        x = batch.get("data", batch.get("x", batch.get("images")))
    elif isinstance(batch, (list, tuple)):
        x = batch[0]
    else:
        x = batch
    if x is None:
        return None
    if x.dim() == 5:
        if frame_mode == "all":
            return x.reshape(-1, *x.shape[2:])
        return x[:, 0]
    if x.dim() == 4:
        return x
    return None


def get_empirical_atoms(model, dataloader=None, cache_dir=None, force_compute=False):
    """
    Récupère les atomes empiriques (covariances).
    Cherche d'abord dans le cache, sinon calcule depuis le dataloader.
    """
    cache_path = None
    if cache_dir:
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        if cache_dir.suffix == ".pt":
            cache_dir = cache_dir.parent
        cache_path = cache_dir / "empirical_atoms.pt"

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    if cache_path and cache_path.exists() and not force_compute:
        print(f"Loading cached empirical atoms from {cache_path}")
        return torch.load(cache_path, map_location=device)

    if dataloader is None:
        print("⚠️ Warning: No cache found and no dataloader provided. Cannot compute empirical atoms.")
        return None

    if not hasattr(model, "centroids_tens") or not hasattr(model, "M_tens"):
        raise ValueError("Model must expose both centroids_tens and M_tens to compute empirical atoms.")

    print("--- Computing Empirical Atoms from Data (The 'Yellow Snake') ---")
    model.eval()

    centroids = model.centroids_tens.to(device)
    K, D = centroids.shape
    epsilon = 1e-4 * torch.eye(D, device=device)

    encoded = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch.get("data", batch.get("x", batch.get("images")))
            elif isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            if x is None:
                continue
            x = x.to(device)
            if x.dim() > 2:
                x = x.view(x.shape[0], -1)
            z = model.encoder(x).embedding
            encoded.append(z)

    if not encoded:
        raise RuntimeError("No frames were encoded from the dataloader.")

    z_data = torch.cat(encoded, dim=0)
    print(f"Encoded {len(z_data)} points.")

    dists = torch.cdist(z_data, centroids)
    labels = torch.argmin(dists, dim=1)

    covs = []
    for k in range(K):
        cluster = z_data[labels == k]
        if cluster.shape[0] < D + 2:
            cov = model.M_tens[k].to(device)
        else:
            centered = cluster - cluster.mean(dim=0, keepdim=True)
            cov = centered.T @ centered
            cov = cov / max(cluster.shape[0] - 1, 1)
            cov = cov + epsilon
        covs.append(cov)

    empirical_atoms = torch.stack(covs)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(empirical_atoms, cache_path)
        print(f"Saved empirical atoms to {cache_path}")

    return empirical_atoms
