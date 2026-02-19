#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "lib" / "src"))

from pythae.models.rhvae import RHVAE, RHVAEConfig
from pythae.models.rhvae.rhvae_utils import create_inverse_metric, create_metric
from scripts.sampling_diagnostics import geodesic_interpolation
from src.models.model_adapter import ModelAdapter
from src.models.rhvae_geometry import GeometryRHVAE, GeometryRHVAEConfig
from src.models.samplers.hmc_sampler import RHVAEVolumeElementHMCSampler
from src.utils.low_data_io import load_split_tensor, load_subset_indices, subset_from_indices
from src.utils.low_data_metrics import (
    augmentation_eval,
    compute_d_rmse,
    compute_fid,
    compute_geo_euc_ratio,
    compute_interp_smoothness,
    compute_prd,
)
from src.utils.wandb_logging import (
    init_wandb_run,
    log_dir_artifact,
    make_wandb_image,
    safe_wandb_finish,
    safe_wandb_log,
)


class _VanillaVAE(torch.nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 512):
        super().__init__()
        in_dim = 28 * 28
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.fc_mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_dim, latent_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, in_dim),
            torch.nn.Sigmoid(),
        )

    def encode_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class _Encoder(torch.nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(torch.nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 28 * 28),
            torch.nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _EBM(torch.nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class VanillaVAEAdapter:
    def __init__(self, model_dir: Path, device: torch.device):
        payload = torch.load(model_dir / "model.pt", map_location=device)
        latent_dim = int(payload.get("latent_dim", 16))
        hidden_dim = int(payload.get("hidden_dim", 512))
        model = _VanillaVAE(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()

        self.model = model
        self.model_id = "vanilla_vae"
        self.latent_dim = latent_dim
        self.device = device

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xx = x.to(self.device).reshape(x.shape[0], -1)
        with torch.no_grad():
            mu, _ = self.model.encode_stats(xx)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        zz = z.to(self.device)
        with torch.no_grad():
            recon = self.model.decode(zz)
        return recon.reshape(zz.shape[0], 1, 28, 28)

    def sample_latents(self, n_samples: int, device: torch.device) -> torch.Tensor:
        return torch.randn(int(n_samples), self.latent_dim, device=device)

    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, n_steps: int = 32) -> torch.Tensor:
        t = torch.linspace(0, 1, n_steps, device=z1.device).unsqueeze(1)
        return z1.unsqueeze(0) + t * (z2.unsqueeze(0) - z1.unsqueeze(0))

    def metric_tensor_or_proxy(self, z: torch.Tensor) -> torch.Tensor:
        zz = z if z.dim() == 2 else z.unsqueeze(0)
        eye = torch.eye(self.latent_dim, device=zz.device, dtype=zz.dtype)
        return eye.unsqueeze(0).expand(zz.shape[0], -1, -1)


class RHVAEAdapter:
    def __init__(self, model_dir: Path, device: torch.device, model_id: str):
        metric_payload = torch.load(model_dir / "rhvae_metric.pt", map_location="cpu")
        cfg_dict = dict(metric_payload.get("config", {}))

        centroids = metric_payload.get("centroids")
        atoms = metric_payload.get("metric_matrices")
        if not isinstance(centroids, torch.Tensor) or not isinstance(atoms, torch.Tensor):
            raise RuntimeError(f"Invalid RHVAE metric payload in {model_dir}")

        latent_dim = int(cfg_dict.get("latent_dim", centroids.shape[-1]))

        is_geometry = bool(cfg_dict.get("use_attractor", False) or cfg_dict.get("kernel_type", "isotropic") != "isotropic" or model_id == "aniso")

        if is_geometry:
            allowed = {k for k, v in GeometryRHVAEConfig.__dataclass_fields__.items() if v.init}
            init_kwargs = {k: v for k, v in cfg_dict.items() if k in allowed}
            init_kwargs.setdefault("input_dim", (28 * 28,))
            init_kwargs.setdefault("latent_dim", latent_dim)
            init_kwargs.setdefault("temperature", float(metric_payload.get("temperature", 0.5)))
            init_kwargs.setdefault("regularization", float(metric_payload.get("regularization", 0.01)))
            config = GeometryRHVAEConfig(**init_kwargs)
            model = GeometryRHVAE(config).to(device)
            model.set_centroids(centroids.to(device))
            model.set_atoms(atoms.to(device))
            model.temperature.data = torch.tensor(float(metric_payload.get("temperature", 0.5)), device=device)
            model.lbd.data = torch.tensor(float(metric_payload.get("regularization", 0.01)), device=device)
            model._refresh_metric_hooks()
        else:
            allowed = {k for k, v in RHVAEConfig.__dataclass_fields__.items() if v.init}
            init_kwargs = {k: v for k, v in cfg_dict.items() if k in allowed}
            init_kwargs.setdefault("input_dim", (28 * 28,))
            init_kwargs.setdefault("latent_dim", latent_dim)
            init_kwargs.setdefault("temperature", float(metric_payload.get("temperature", 0.5)))
            init_kwargs.setdefault("regularization", float(metric_payload.get("regularization", 0.01)))
            config = RHVAEConfig(**init_kwargs)
            model = RHVAE(config).to(device)
            model.M_tens = atoms.to(device)
            model.centroids_tens = centroids.to(device)
            model.temperature.data = torch.tensor(float(metric_payload.get("temperature", 0.5)), device=device)
            model.lbd.data = torch.tensor(float(metric_payload.get("regularization", 0.01)), device=device)
            model.G = create_metric(model)
            model.G_inv = create_inverse_metric(model)

        state = torch.load(model_dir / "rhvae_model.pt", map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()

        self.model = model
        self.model_id = model_id
        self.latent_dim = int(latent_dim)
        self.device = device
        self.centroids = centroids.to(device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xx = x.to(self.device).reshape(x.shape[0], -1)
        with torch.no_grad():
            out = self.model.encoder(xx)
            return out.embedding

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        zz = z.to(self.device)
        with torch.no_grad():
            recon = self.model.decoder(zz)["reconstruction"]
        return recon.reshape(zz.shape[0], 1, 28, 28)

    def sample_latents(self, n_samples: int, device: torch.device) -> torch.Tensor:
        try:
            sampler = RHVAEVolumeElementHMCSampler(
                self.model,
                mcmc_steps_nbr=60,
                n_lf=20,
                eps_lf=0.02,
                beta_zero=1.0,
            )
            z = sampler.sample(int(n_samples))
            return z.to(device)
        except Exception:
            return torch.randn(int(n_samples), self.latent_dim, device=device)

    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, n_steps: int = 32) -> torch.Tensor:
        z1d = z1.to(self.device).reshape(-1)
        z2d = z2.to(self.device).reshape(-1)
        try:
            path = geodesic_interpolation(
                self.model,
                z1d,
                z2d,
                centroids=self.centroids,
                n_steps=n_steps,
                iterations=80,
                lr=0.01,
                wall_strength=8.0 if self.model_id == "aniso" else 0.0,
            )
            return path
        except Exception:
            t = torch.linspace(0, 1, n_steps, device=self.device).unsqueeze(1)
            return z1d.unsqueeze(0) + t * (z2d.unsqueeze(0) - z1d.unsqueeze(0))

    def metric_tensor_or_proxy(self, z: torch.Tensor) -> torch.Tensor:
        zz = z.to(self.device)
        if zz.dim() == 1:
            zz = zz.unsqueeze(0)
        with torch.no_grad():
            return self.model.G_inv(zz)


class EBMConformalAdapter:
    def __init__(self, model_dir: Path, device: torch.device):
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        latent_dim = int(cfg.get("latent_dim", 16))
        enc_h = int(cfg.get("encoder_hidden_dim", 512))
        ebm_h = int(cfg.get("ebm_hidden_dim", 256))
        beta = float(cfg.get("beta_conformal", 1.0))

        enc_payload = torch.load(model_dir / "encoder.pt", map_location=device)
        dec_payload = torch.load(model_dir / "decoder.pt", map_location=device)
        ebm_payload = torch.load(model_dir / "ebm.pt", map_location=device)

        enc = _Encoder(latent_dim=latent_dim, hidden_dim=enc_h).to(device)
        dec = _Decoder(latent_dim=latent_dim, hidden_dim=enc_h).to(device)
        ebm = _EBM(latent_dim=latent_dim, hidden_dim=ebm_h).to(device)

        enc.load_state_dict(enc_payload["state_dict"])
        dec.load_state_dict(dec_payload["state_dict"])
        ebm.load_state_dict(ebm_payload["state_dict"])
        enc.eval()
        dec.eval()
        ebm.eval()

        self.model_id = "ebm_conformal"
        self.latent_dim = latent_dim
        self.device = device
        self.encoder = enc
        self.decoder = dec
        self.ebm = ebm
        self.beta_conformal = beta

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xx = x.to(self.device).reshape(x.shape[0], -1)
        with torch.no_grad():
            return self.encoder(xx)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        zz = z.to(self.device)
        with torch.no_grad():
            recon = self.decoder(zz)
        return recon.reshape(zz.shape[0], 1, 28, 28)

    def sample_latents(self, n_samples: int, device: torch.device) -> torch.Tensor:
        return torch.randn(int(n_samples), self.latent_dim, device=device)

    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, n_steps: int = 32) -> torch.Tensor:
        t = torch.linspace(0, 1, n_steps, device=self.device).unsqueeze(1)
        z1d = z1.to(self.device).reshape(-1)
        z2d = z2.to(self.device).reshape(-1)
        return z1d.unsqueeze(0) + t * (z2d.unsqueeze(0) - z1d.unsqueeze(0))

    def metric_tensor_or_proxy(self, z: torch.Tensor) -> torch.Tensor:
        zz = z.to(self.device)
        if zz.dim() == 1:
            zz = zz.unsqueeze(0)
        with torch.no_grad():
            energy = self.ebm(zz)
            scalar = torch.exp(-self.beta_conformal * energy).reshape(-1, 1, 1)
            eye = torch.eye(self.latent_dim, device=zz.device, dtype=zz.dtype).unsqueeze(0)
            return scalar * eye


def _find_latest_run(parent: Path, required_files: list[str]) -> Path | None:
    if not parent.exists() or not parent.is_dir():
        return None
    runs = [p for p in parent.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        if all((run / f).exists() for f in required_files):
            return run
    return None


def _resolve_model_dir(
    model_id: str,
    n: int,
    seed: int,
    model_root: Path,
    alias_root: Path,
    required_files: list[str],
) -> Path | None:
    candidate_parent = model_root / model_id / f"N{int(n):03d}" / f"seed{int(seed):03d}"
    latest = _find_latest_run(candidate_parent, required_files)
    if latest is not None:
        return latest

    alias = alias_root / model_id
    if alias.exists() and all((alias / f).exists() for f in required_files):
        return alias

    return None


def _load_registry_required_files(model_registry: Path) -> dict[str, list[str]]:
    import yaml

    payload = yaml.safe_load(model_registry.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for row in payload.get("models", []):
        out[str(row["model_id"])] = [str(x) for x in row.get("required_files", [])]
    return out


def _make_adapter(model_id: str, model_dir: Path, device: torch.device) -> ModelAdapter:
    if model_id == "vanilla_vae":
        return VanillaVAEAdapter(model_dir, device)
    if model_id in {"rhvae_standard", "aniso"}:
        return RHVAEAdapter(model_dir, device, model_id=model_id)
    if model_id == "ebm_conformal":
        return EBMConformalAdapter(model_dir, device)
    raise ValueError(f"Unsupported model_id: {model_id}")


def _safe_mean(vals: list[float]) -> float:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _safe_std(vals: list[float]) -> float:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.std())


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    if n == 0:
        return (float("nan"), float("nan"))
    samples = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        samples.append(float(np.mean(values[idx])))
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi


def _holm_adjust(pvals: list[tuple[str, float]]) -> dict[str, float]:
    if not pvals:
        return {}
    sorted_items = sorted(pvals, key=lambda x: x[1])
    m = len(sorted_items)
    adjusted: dict[str, float] = {}
    prev = 0.0
    for i, (key, p) in enumerate(sorted_items):
        adj = min(1.0, (m - i) * p)
        adj = max(adj, prev)
        prev = adj
        adjusted[key] = adj
    return adjusted


def _evaluate_model(
    adapter: ModelAdapter,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    n_gen_samples: int,
    n_interp_pairs: int,
    interp_steps: int,
    aug_synth_samples: int,
    do_augmentation: bool,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    z_gen = adapter.sample_latents(n_gen_samples, device=device)
    x_gen = adapter.decode(z_gen).detach().cpu()

    fid = compute_fid(test_x, x_gen, device=device)

    real_lat = adapter.encode(train_x.to(device)).detach().cpu()
    prd = compute_prd(train_x, x_gen, k=5)

    pair_metrics_ratio: list[float] = []
    pair_metrics_drmse: list[float] = []
    pair_metrics_smooth: list[float] = []

    if real_lat.shape[0] >= 2:
        n_candidates = min(500, int(real_lat.shape[0]))
        cand_idx = rng.choice(real_lat.shape[0], size=n_candidates, replace=False)
        zc = real_lat[cand_idx]
        dist = torch.cdist(zc, zc).numpy()
        np.fill_diagonal(dist, -np.inf)

        flat = []
        for i in range(dist.shape[0]):
            j = int(np.argmax(dist[i]))
            if dist[i, j] > 0:
                flat.append((float(dist[i, j]), int(cand_idx[i]), int(cand_idx[j])))
        flat.sort(key=lambda x: x[0], reverse=True)
        selected = flat[: max(1, int(n_interp_pairs))]

        for _, i, j in selected:
            z1 = real_lat[i].to(device)
            z2 = real_lat[j].to(device)
            path = adapter.interpolate(z1, z2, n_steps=interp_steps).detach().cpu()
            pair_metrics_ratio.append(compute_geo_euc_ratio(path))
            decoded_path = adapter.decode(path.to(device)).detach().cpu()
            pair_metrics_drmse.append(compute_d_rmse(decoded_path, train_x))
            pair_metrics_smooth.append(compute_interp_smoothness(decoded_path))

    aug_bal = float("nan")
    aug_f1 = float("nan")
    if do_augmentation:
        z_syn = adapter.sample_latents(aug_synth_samples, device=device)
        x_syn = adapter.decode(z_syn).detach().cpu()

        # Pseudo-label synthetic samples via nearest class centroid in latent space.
        class_proto: dict[int, torch.Tensor] = {}
        for cls in torch.unique(train_y).tolist():
            cls = int(cls)
            mask = train_y == cls
            if int(mask.sum().item()) > 0:
                class_proto[cls] = real_lat[mask].mean(dim=0)

        if class_proto:
            proto_cls = sorted(class_proto.keys())
            proto_lat = torch.stack([class_proto[c] for c in proto_cls], dim=0)
            d = torch.cdist(z_syn.detach().cpu(), proto_lat)
            nn = torch.argmin(d, dim=1)
            y_syn = torch.tensor([proto_cls[int(i)] for i in nn], dtype=torch.long)
        else:
            y_syn = torch.zeros((x_syn.shape[0],), dtype=torch.long)

        aug = augmentation_eval(
            real_train_x=train_x,
            real_train_y=train_y,
            synthetic_x=x_syn,
            synthetic_y=y_syn,
            test_x=test_x,
            test_y=test_y,
            device=device,
            epochs=5,
            batch_size=64,
            lr=1e-3,
        )
        aug_bal = float(aug["balanced_accuracy"])
        aug_f1 = float(aug["macro_f1"])

    return {
        "fid": float(fid),
        "prd_precision": float(prd.get("precision", float("nan"))),
        "prd_recall": float(prd.get("recall", float("nan"))),
        "geo_euc_ratio": _safe_mean(pair_metrics_ratio),
        "d_rmse": _safe_mean(pair_metrics_drmse),
        "interp_smoothness": _safe_mean(pair_metrics_smooth),
        "aug_balanced_accuracy": aug_bal,
        "aug_macro_f1": aug_f1,
    }


def _plot_bridge_over_void(
    out_path: Path,
    adapters: dict[str, ModelAdapter],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    device: torch.device,
) -> None:
    if "aniso" not in adapters or "rhvae_standard" not in adapters:
        return

    aniso = adapters["aniso"]
    z = aniso.encode(train_x.to(device)).detach().cpu()

    class_means = {}
    for cls in torch.unique(train_y).tolist():
        cls = int(cls)
        m = train_y == cls
        if int(m.sum().item()) > 0:
            class_means[cls] = z[m].mean(dim=0)
    if len(class_means) < 2:
        return

    keys = sorted(class_means.keys())
    best_pair = None
    best_dist = -1.0
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            d = torch.linalg.norm(class_means[a] - class_means[b]).item()
            if d > best_dist:
                best_dist = d
                best_pair = (a, b)
    if best_pair is None:
        return

    a, b = best_pair
    idx_a = int(torch.where(train_y == a)[0][0].item())
    idx_b = int(torch.where(train_y == b)[0][0].item())
    xa = train_x[idx_a : idx_a + 1].to(device)
    xb = train_x[idx_b : idx_b + 1].to(device)

    model_order = ["rhvae_standard", "aniso"]
    n_steps = 10

    fig, axes = plt.subplots(len(model_order), n_steps, figsize=(1.6 * n_steps, 2.4 * len(model_order)))
    if len(model_order) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, model_id in enumerate(model_order):
        ad = adapters[model_id]
        z1 = ad.encode(xa)[0]
        z2 = ad.encode(xb)[0]
        path = ad.interpolate(z1, z2, n_steps=n_steps)
        imgs = ad.decode(path).detach().cpu()

        for c in range(n_steps):
            ax = axes[r, c]
            ax.imshow(imgs[c, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(model_id, fontsize=10)

    fig.suptitle(f"Bridge over the Void: class {a} -> class {b}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_logdet_heatmap(out_path: Path, adapter: ModelAdapter, images: torch.Tensor, device: torch.device) -> None:
    z = adapter.encode(images.to(device)).detach().cpu()
    if z.shape[0] == 0:
        return
    z_np = z.numpy()
    if z_np.shape[1] > 2:
        pca = PCA(n_components=2, random_state=0)
        zz = pca.fit_transform(z_np)
    else:
        zz = z_np[:, :2]

    z_eval = z.to(device)
    with torch.no_grad():
        ginv = adapter.metric_tensor_or_proxy(z_eval)
        logdet = torch.linalg.slogdet(ginv).logabsdet.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(zz[:, 0], zz[:, 1], c=logdet, s=8, alpha=0.8, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="log det(G^-1)")
    ax.set_title(f"Latent Heatmap ({adapter.model_id})")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_ellipses(out_path: Path, adapter: ModelAdapter, latent_samples: torch.Tensor, device: torch.device) -> None:
    z = latent_samples.detach().cpu()
    if z.shape[0] == 0:
        return
    z2 = z[:, :2]
    lo = z2.min(dim=0).values
    hi = z2.max(dim=0).values

    xs = torch.linspace(float(lo[0]), float(hi[0]), 10)
    ys = torch.linspace(float(lo[1]), float(hi[1]), 10)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z2[:, 0], z2[:, 1], s=4, alpha=0.15, color="gray")

    scale = 0.2 * float(torch.mean(hi - lo).item() + 1e-6)

    for x in xs:
        for y in ys:
            zz = torch.zeros((1, adapter.latent_dim), dtype=torch.float32, device=device)
            zz[0, 0] = float(x)
            zz[0, 1] = float(y)
            with torch.no_grad():
                ginv = adapter.metric_tensor_or_proxy(zz)[0, :2, :2].detach().cpu().numpy()
            ginv = 0.5 * (ginv + ginv.T)
            vals, vecs = np.linalg.eigh(ginv)
            vals = np.clip(vals, 1e-8, None)
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]

            width = scale / math.sqrt(vals[0])
            height = scale / math.sqrt(vals[1])
            angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))

            e = Ellipse(
                xy=(float(x), float(y)),
                width=width,
                height=height,
                angle=angle,
                edgecolor="tab:blue",
                facecolor="none",
                lw=0.8,
                alpha=0.8,
            )
            ax.add_patch(e)

    ax.set_title(f"Metric Tensor Ellipses ({adapter.model_id})")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_history_metrics(model_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    history = payload.get("history", [])
    if not isinstance(history, list) or not history:
        return None
    train = []
    val = []
    for row in history:
        if not isinstance(row, dict):
            continue
        tr = row.get("train_loss")
        va = row.get("val_loss")
        if tr is None or va is None:
            continue
        try:
            tr_f = float(tr)
            va_f = float(va)
        except Exception:
            continue
        if not np.isfinite(tr_f) or not np.isfinite(va_f):
            continue
        train.append(tr_f)
        val.append(va_f)
    if not train or not val:
        return None
    return np.asarray(train, dtype=float), np.asarray(val, dtype=float)


def _plot_rhvae_training_metrics(
    out_path: Path,
    rows: list[dict[str, Any]],
    subset_ns: list[int],
) -> None:
    model_labels = {
        "rhvae_standard": "RHVAE (Standard)",
        "aniso": "AnisoRHVAE",
    }
    colors = {
        "rhvae_standard": "tab:blue",
        "aniso": "tab:orange",
    }
    model_order = ["rhvae_standard", "aniso"]

    n_rows = len(subset_ns)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.4 * n_rows), squeeze=False)
    plotted_any = False

    for ridx, n in enumerate(subset_ns):
        for mid in model_order:
            runs = [Path(str(r["model_dir"])) for r in rows if r.get("status") == "ok" and int(r.get("n", -1)) == int(n) and r.get("model_id") == mid]
            histories: list[tuple[np.ndarray, np.ndarray]] = []
            for run_dir in runs:
                h = _load_history_metrics(run_dir)
                if h is not None:
                    histories.append(h)
            if not histories:
                continue

            min_len = min(int(h[0].shape[0]) for h in histories)
            if min_len <= 0:
                continue

            train_mat = np.stack([h[0][:min_len] for h in histories], axis=0)
            val_mat = np.stack([h[1][:min_len] for h in histories], axis=0)
            epochs = np.arange(1, min_len + 1, dtype=int)

            train_mean = train_mat.mean(axis=0)
            train_std = train_mat.std(axis=0)
            val_mean = val_mat.mean(axis=0)
            val_std = val_mat.std(axis=0)

            ax_train = axes[ridx, 0]
            ax_val = axes[ridx, 1]
            color = colors[mid]
            label = model_labels[mid]

            ax_train.plot(epochs, train_mean, color=color, lw=2.0, label=label)
            ax_train.fill_between(epochs, train_mean - train_std, train_mean + train_std, color=color, alpha=0.2)

            ax_val.plot(epochs, val_mean, color=color, lw=2.0, label=label)
            ax_val.fill_between(epochs, val_mean - val_std, val_mean + val_std, color=color, alpha=0.2)
            plotted_any = True

        axes[ridx, 0].set_title(f"N={int(n)} Train Loss")
        axes[ridx, 1].set_title(f"N={int(n)} Val Loss")
        axes[ridx, 0].set_xlabel("epoch")
        axes[ridx, 1].set_xlabel("epoch")
        axes[ridx, 0].set_ylabel("loss")
        axes[ridx, 1].set_ylabel("loss")
        axes[ridx, 0].grid(alpha=0.25)
        axes[ridx, 1].grid(alpha=0.25)
        axes[ridx, 0].legend()
        axes[ridx, 1].legend()

    if not plotted_any:
        plt.close(fig)
        return

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run low-data benchmark across model families.")
    parser.add_argument("--processed_dir", type=str, default="data/processed/rotmnist/v1")
    parser.add_argument("--model_root", type=str, default="outputs/low_data_models/rotmnist_v1")
    parser.add_argument("--model_alias_root", type=str, default="outputs/reference_models")
    parser.add_argument("--model_registry", type=str, default="configs/assets/model_registry.yaml")
    parser.add_argument("--models", nargs="+", default=["vanilla_vae", "rhvae_standard", "aniso", "ebm_conformal"])
    parser.add_argument("--subset_ns", nargs="+", type=int, default=[50, 100, 500])
    parser.add_argument("--subset_seeds", nargs="+", type=int, default=[13, 29, 47, 71, 89])
    parser.add_argument("--n_gen_samples", type=int, default=2000)
    parser.add_argument("--n_interp_pairs", type=int, default=64)
    parser.add_argument("--interp_steps", type=int, default=32)
    parser.add_argument("--aug_synth_samples", type=int, default=500)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results/low_data_benchmark")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_name_mode", type=str, default="auto", choices=["timestamp", "auto", "manual"])
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_max_plot_images", type=int, default=24)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    if args.quick:
        args.subset_ns = [50]
        args.subset_seeds = [13]
        args.n_gen_samples = min(args.n_gen_samples, 128)
        args.n_interp_pairs = min(args.n_interp_pairs, 2)
        os.environ["LOW_DATA_DISABLE_CLEANFID"] = "1"

    wandb_run = init_wandb_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        tags=args.wandb_tags,
        run_name=args.wandb_run_name,
        name_mode=args.wandb_name_mode,
        mode=args.wandb_mode,
        config=vars(args),
        name_prefix="low_data_benchmark",
    )

    try:
        processed_dir = (ROOT / args.processed_dir).resolve()
        model_root = (ROOT / args.model_root).resolve()
        alias_root = (ROOT / args.model_alias_root).resolve()
        registry_path = (ROOT / args.model_registry).resolve()

        run_dir = (ROOT / args.output_dir).resolve() / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fig_dir = run_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        required = _load_registry_required_files(registry_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        full_train_x, full_train_y = load_split_tensor(processed_dir, "train")
        test_x, test_y = load_split_tensor(processed_dir, "test")

        rows: list[dict[str, Any]] = []
        summary: dict[str, Any] = {}

        canonical_adapters: dict[str, ModelAdapter] = {}
        canonical_train_x = None
        canonical_train_y = None
        run_step = 0

        for n in args.subset_ns:
            summary[str(n)] = {}
            for seed in args.subset_seeds:
                subset_idx = load_subset_indices(processed_dir, n, seed)
                train_x, train_y = subset_from_indices(full_train_x, full_train_y, subset_idx)

                adapters_for_seed: dict[str, ModelAdapter] = {}
                for model_id in args.models:
                    req = required.get(model_id, [])
                    model_dir = _resolve_model_dir(
                        model_id=model_id,
                        n=n,
                        seed=seed,
                        model_root=model_root,
                        alias_root=alias_root,
                        required_files=req,
                    )
                    if model_dir is None:
                        row = {
                            "n": int(n),
                            "seed": int(seed),
                            "model_id": model_id,
                            "status": "missing_model",
                        }
                        rows.append(row)
                        run_step += 1
                        safe_wandb_log(
                            wandb_run,
                            {
                                "progress/n": int(n),
                                "progress/seed": int(seed),
                                "progress/model_id": str(model_id),
                                "progress/status": "missing_model",
                            },
                            step=run_step,
                        )
                        continue

                    try:
                        adapter = _make_adapter(model_id, model_dir, device=device)
                    except Exception as exc:
                        row = {
                            "n": int(n),
                            "seed": int(seed),
                            "model_id": model_id,
                            "status": f"load_error:{exc}",
                        }
                        rows.append(row)
                        run_step += 1
                        safe_wandb_log(
                            wandb_run,
                            {
                                "progress/n": int(n),
                                "progress/seed": int(seed),
                                "progress/model_id": str(model_id),
                                "progress/status": "load_error",
                            },
                            step=run_step,
                        )
                        continue

                    adapters_for_seed[model_id] = adapter

                    metrics = _evaluate_model(
                        adapter=adapter,
                        train_x=train_x,
                        train_y=train_y,
                        test_x=test_x,
                        test_y=test_y,
                        device=device,
                        n_gen_samples=args.n_gen_samples,
                        n_interp_pairs=args.n_interp_pairs,
                        interp_steps=args.interp_steps,
                        aug_synth_samples=args.aug_synth_samples,
                        do_augmentation=(int(n) == 50),
                        seed=seed,
                    )

                    row = {
                        "n": int(n),
                        "seed": int(seed),
                        "model_id": model_id,
                        "status": "ok",
                        "model_dir": str(model_dir),
                    }
                    row.update(metrics)
                    rows.append(row)
                    run_step += 1
                    safe_wandb_log(
                        wandb_run,
                        {
                            "seed_metrics/fid": float(row.get("fid", float("nan"))),
                            "seed_metrics/prd_precision": float(row.get("prd_precision", float("nan"))),
                            "seed_metrics/prd_recall": float(row.get("prd_recall", float("nan"))),
                            "seed_metrics/geo_euc_ratio": float(row.get("geo_euc_ratio", float("nan"))),
                            "seed_metrics/d_rmse": float(row.get("d_rmse", float("nan"))),
                            "seed_metrics/interp_smoothness": float(row.get("interp_smoothness", float("nan"))),
                            "seed_metrics/aug_balanced_accuracy": float(row.get("aug_balanced_accuracy", float("nan"))),
                            "seed_metrics/aug_macro_f1": float(row.get("aug_macro_f1", float("nan"))),
                            "progress/n": int(n),
                            "progress/seed": int(seed),
                            "progress/model_id": str(model_id),
                        },
                        step=run_step,
                    )

                if int(n) == 50 and int(seed) == 13 and adapters_for_seed:
                    canonical_adapters = adapters_for_seed
                    canonical_train_x = train_x
                    canonical_train_y = train_y

            for model_id in args.models:
                mrows = [r for r in rows if r.get("status") == "ok" and r.get("n") == int(n) and r.get("model_id") == model_id]
                metric_names = [
                    "fid",
                    "prd_precision",
                    "prd_recall",
                    "geo_euc_ratio",
                    "d_rmse",
                    "interp_smoothness",
                    "aug_balanced_accuracy",
                    "aug_macro_f1",
                ]
                stats = {}
                for name in metric_names:
                    vals = [float(r.get(name, float("nan"))) for r in mrows]
                    stats[name] = {
                        "mean": _safe_mean(vals),
                        "std": _safe_std(vals),
                        "n": int(np.sum(np.isfinite(np.asarray(vals, dtype=float)))),
                    }
                summary[str(n)][model_id] = stats

        if canonical_adapters and canonical_train_x is not None and canonical_train_y is not None:
            _plot_bridge_over_void(fig_dir / "bridge_over_void.png", canonical_adapters, canonical_train_x, canonical_train_y, device)
            for model_id, adapter in canonical_adapters.items():
                _plot_logdet_heatmap(fig_dir / f"latent_logdet_heatmap_{model_id}_N050_seed013.png", adapter, canonical_train_x, device)
                z = adapter.encode(canonical_train_x.to(device)).detach().cpu()
                _plot_metric_ellipses(fig_dir / f"metric_ellipses_{model_id}_N050_seed013.png", adapter, z, device)
        _plot_rhvae_training_metrics(
            fig_dir / "training_metrics_rhvae_vs_aniso.png",
            rows=rows,
            subset_ns=[int(x) for x in args.subset_ns],
        )

        metrics_table_path = run_dir / "metrics_table.csv"
        _write_csv(metrics_table_path, rows)

        summary_payload = {
            "created_at": datetime.datetime.now().isoformat(),
            "models": args.models,
            "subset_ns": args.subset_ns,
            "subset_seeds": args.subset_seeds,
            "summary": summary,
        }
        (run_dir / "summary_by_model_n.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        # Statistical tests: aniso vs each baseline per metric and N.
        metric_names = [
            "fid",
            "prd_precision",
            "prd_recall",
            "geo_euc_ratio",
            "d_rmse",
            "interp_smoothness",
            "aug_balanced_accuracy",
            "aug_macro_f1",
        ]
        higher_better = {"prd_precision", "prd_recall", "geo_euc_ratio", "aug_balanced_accuracy", "aug_macro_f1"}

        stat_tests: dict[str, Any] = {}
        for n in args.subset_ns:
            key_n = str(n)
            stat_tests[key_n] = {}
            for metric in metric_names:
                comparisons = []
                raw_pvals: list[tuple[str, float]] = []
                for baseline in [m for m in args.models if m != "aniso"]:
                    paired = []
                    for seed in args.subset_seeds:
                        r_a = next((r for r in rows if r.get("status") == "ok" and r.get("n") == int(n) and r.get("seed") == int(seed) and r.get("model_id") == "aniso"), None)
                        r_b = next((r for r in rows if r.get("status") == "ok" and r.get("n") == int(n) and r.get("seed") == int(seed) and r.get("model_id") == baseline), None)
                        if r_a is None or r_b is None:
                            continue
                        va = float(r_a.get(metric, float("nan")))
                        vb = float(r_b.get(metric, float("nan")))
                        if not np.isfinite(va) or not np.isfinite(vb):
                            continue
                        paired.append((int(seed), va, vb))

                    if not paired:
                        comparisons.append(
                            {
                                "baseline": baseline,
                                "n_pairs": 0,
                                "p_value": float("nan"),
                                "p_holm": float("nan"),
                                "aniso_minus_baseline_mean": float("nan"),
                                "ci95_low": float("nan"),
                                "ci95_high": float("nan"),
                            }
                        )
                        continue

                    arr_a = np.asarray([x[1] for x in paired], dtype=float)
                    arr_b = np.asarray([x[2] for x in paired], dtype=float)
                    diff = arr_a - arr_b

                    if metric not in higher_better:
                        diff = -diff

                    mean_diff = float(np.mean(diff))
                    ci_low, ci_high = _bootstrap_ci(diff, n_boot=args.bootstrap_samples, alpha=0.05, seed=0)

                    alternative = "greater" if metric in higher_better else "less"
                    try:
                        p = float(wilcoxon(arr_a, arr_b, alternative=alternative).pvalue)
                    except Exception:
                        p = float("nan")

                    comparisons.append(
                        {
                            "baseline": baseline,
                            "n_pairs": int(len(paired)),
                            "p_value": p,
                            "p_holm": float("nan"),
                            "aniso_minus_baseline_mean": mean_diff,
                            "ci95_low": ci_low,
                            "ci95_high": ci_high,
                        }
                    )
                    if np.isfinite(p):
                        raw_pvals.append((baseline, p))

                holm = _holm_adjust(raw_pvals)
                for comp in comparisons:
                    b = comp["baseline"]
                    if b in holm:
                        comp["p_holm"] = float(holm[b])
                stat_tests[key_n][metric] = comparisons

        (run_dir / "stat_tests.json").write_text(json.dumps(stat_tests, indent=2), encoding="utf-8")

        for n_key, model_payload in summary.items():
            for model_id, metric_payload in model_payload.items():
                for metric_name, stat_row in metric_payload.items():
                    safe_wandb_log(
                        wandb_run,
                        {
                            f"summary/N{n_key}/{model_id}/{metric_name}_mean": float(stat_row.get("mean", float("nan"))),
                            f"summary/N{n_key}/{model_id}/{metric_name}_std": float(stat_row.get("std", float("nan"))),
                        },
                    )

        pngs = sorted(fig_dir.glob("*.png"))
        limit = len(pngs) if args.wandb_max_plot_images < 0 else min(len(pngs), int(args.wandb_max_plot_images))
        for p in pngs[:limit]:
            img = make_wandb_image(p)
            if img is not None:
                safe_wandb_log(wandb_run, {f"plots/{p.stem}": img})

        if wandb_run is not None:
            wandb_run.summary["output_dir"] = str(run_dir)
            wandb_run.summary["rows_total"] = int(len(rows))
            wandb_run.summary["rows_ok"] = int(sum(1 for r in rows if r.get("status") == "ok"))

        log_dir_artifact(
            wandb_run,
            run_dir,
            artifact_name=f"low_data_benchmark_{run_dir.name}",
            artifact_type="low_data_benchmark",
        )

        print(f"[run_low_data_benchmark] saved to: {run_dir}")
        print("- summary_by_model_n.json")
        print("- metrics_table.csv")
        print("- stat_tests.json")
        print("- figures/")
    finally:
        safe_wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
