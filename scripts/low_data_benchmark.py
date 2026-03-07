#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "lib" / "src"))

from pythae.models.rhvae import RHVAE, RHVAEConfig
from pythae.models.rhvae.rhvae_utils import create_inverse_metric, create_metric
from scripts.rhmc_chain_demo import _build_sampler as _build_demo_sampler
from scripts.rhmc_chain_demo import run_hmc_chain as _run_demo_chain
from scripts.sampling_diagnostics import geodesic_interpolation
from src.models.rhvae_geometry import GeometryRHVAE, GeometryRHVAEConfig
from src.utils.low_data_io import load_split_tensor, load_subset_indices, model_output_dir, subset_from_indices
from src.utils.low_data_metrics import augmentation_eval
from src.utils.rescue_metrics import compute_hit_steps, rescue_rate_at_horizon


@dataclass(frozen=True)
class BenchmarkConfig:
    processed_dir: Path
    output_dir: Path
    model_root: Path
    subset_ns: list[int]
    subset_seeds: list[int]
    epochs: int
    augment_multiplier: int


@dataclass(frozen=True)
class VolumeSamplerConfig:
    volume_power: float
    beta_zero: float
    mcmc_steps: int
    n_lf: int
    eps_lf: float
    grad_clip_norm: float
    logdet_clip: float
    eps_backoff: float
    max_backoff_trials: int
    warmup_steps: int


@dataclass(frozen=True)
class FailureDiagConfig:
    stretch_factor: float
    chains: int
    horizon: int
    n_lf: int
    eps_volume: float
    eps_riem: float


class BenchmarkAdapter(Protocol):
    model_id: str
    latent_dim: int

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        ...

    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, n_steps: int = 32) -> torch.Tensor:
        ...

    def logdet_ginv(self, z: torch.Tensor, with_grad: bool) -> torch.Tensor:
        ...


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


class VanillaAdapter:
    def __init__(self, model_dir: Path, device: torch.device, model_id: str = "vanilla_vae"):
        payload = torch.load(model_dir / "model.pt", map_location=device)
        latent_dim = int(payload.get("latent_dim", 16))
        hidden_dim = int(payload.get("hidden_dim", 512))
        model = _VanillaVAE(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()

        self.model = model
        self.model_id = model_id
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

    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, n_steps: int = 32) -> torch.Tensor:
        t = torch.linspace(0, 1, n_steps, device=z1.device).unsqueeze(1)
        return z1.unsqueeze(0) + t * (z2.unsqueeze(0) - z1.unsqueeze(0))

    def logdet_ginv(self, z: torch.Tensor, with_grad: bool) -> torch.Tensor:
        del with_grad
        return torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)


class RHVAEAdapter:
    def __init__(self, model_dir: Path, device: torch.device, model_id: str):
        metric_payload = torch.load(model_dir / "rhvae_metric.pt", map_location="cpu")
        cfg_dict = dict(metric_payload.get("config", {}))

        centroids = metric_payload.get("centroids")
        atoms = metric_payload.get("metric_matrices")
        if not isinstance(centroids, torch.Tensor) or not isinstance(atoms, torch.Tensor):
            raise RuntimeError(f"Invalid RHVAE metric payload in {model_dir}")

        latent_dim = int(cfg_dict.get("latent_dim", centroids.shape[-1]))

        is_geometry = bool(
            cfg_dict.get("use_attractor", False)
            or cfg_dict.get("kernel_type", "isotropic") != "isotropic"
            or str(model_id).startswith("aniso")
        )

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

    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, n_steps: int = 32) -> torch.Tensor:
        z1d = z1.to(self.device).reshape(-1)
        z2d = z2.to(self.device).reshape(-1)
        try:
            return geodesic_interpolation(
                self.model,
                z1d,
                z2d,
                centroids=self.centroids,
                n_steps=n_steps,
                iterations=400,
                lr=0.01,
                wall_strength=8.0 if str(self.model_id).startswith("aniso") else 0.0,
            )
        except Exception:
            t = torch.linspace(0, 1, n_steps, device=self.device).unsqueeze(1)
            return z1d.unsqueeze(0) + t * (z2d.unsqueeze(0) - z1d.unsqueeze(0))

    def logdet_ginv(self, z: torch.Tensor, with_grad: bool) -> torch.Tensor:
        zz = z if with_grad else z.detach()
        ginv = self.model.G_inv(zz)
        return torch.linalg.slogdet(ginv).logabsdet


class EBMConformalAdapter:
    def __init__(self, model_dir: Path, device: torch.device, model_id: str = "ebm_conformal"):
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

        self.model_id = model_id
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

    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, n_steps: int = 32) -> torch.Tensor:
        t = torch.linspace(0, 1, n_steps, device=self.device).unsqueeze(1)
        z1d = z1.to(self.device).reshape(-1)
        z2d = z2.to(self.device).reshape(-1)
        return z1d.unsqueeze(0) + t * (z2d.unsqueeze(0) - z1d.unsqueeze(0))

    def logdet_ginv(self, z: torch.Tensor, with_grad: bool) -> torch.Tensor:
        zz = z if with_grad else z.detach()
        energy = self.ebm(zz)
        return ebm_logdet_from_energy(energy, latent_dim=self.latent_dim, beta=self.beta_conformal)


def ebm_logdet_from_energy(energy: torch.Tensor, latent_dim: int, beta: float) -> torch.Tensor:
    return float(latent_dim) * (-float(beta) * energy)


class VolumePotentialSampler:
    def __init__(self, adapter: BenchmarkAdapter, cfg: VolumeSamplerConfig, device: torch.device):
        self.adapter = adapter
        self.cfg = cfg
        self.device = device
        self.last_acceptance_rate = float("nan")

    def _volume_scale(self, step_idx: int) -> float:
        if self.cfg.warmup_steps <= 0:
            return self.cfg.volume_power
        frac = min(1.0, float(step_idx + 1) / float(self.cfg.warmup_steps))
        return self.cfg.volume_power * frac

    def _potential(self, z: torch.Tensor, step_idx: int) -> torch.Tensor:
        logdet = self.adapter.logdet_ginv(z, with_grad=z.requires_grad)
        c = float(self.cfg.logdet_clip)
        logdet = torch.clamp(logdet, min=-c, max=c)
        return -self._volume_scale(step_idx) * logdet

    def _grad_u(self, z: torch.Tensor, step_idx: int) -> tuple[torch.Tensor, bool]:
        z_req = z.detach().clone().requires_grad_(True)
        u = self._potential(z_req, step_idx).sum()
        grad = torch.autograd.grad(u, z_req, create_graph=False, allow_unused=False)[0]
        if not torch.isfinite(grad).all():
            return torch.zeros_like(z), False
        norms = torch.linalg.norm(grad, dim=1, keepdim=True)
        clip = float(self.cfg.grad_clip_norm)
        scale = torch.clamp(clip / torch.clamp(norms, min=1e-12), max=1.0)
        grad = grad * scale
        return grad.detach(), True

    def _hamiltonian(self, z: torch.Tensor, rho: torch.Tensor, step_idx: int) -> torch.Tensor:
        with torch.no_grad():
            u = self._potential(z, step_idx)
            k = 0.5 * torch.sum(rho * rho, dim=1)
            return u + k

    def _leapfrog(self, z: torch.Tensor, rho: torch.Tensor, eps: float, step_idx: int) -> tuple[torch.Tensor, torch.Tensor, bool]:
        grad_u, ok = self._grad_u(z, step_idx)
        if not ok:
            return z, rho, False
        rho_half = rho - 0.5 * eps * grad_u
        z_new = z + eps * rho_half
        if not torch.isfinite(z_new).all():
            return z, rho, False
        grad_u_new, ok2 = self._grad_u(z_new, step_idx)
        if not ok2:
            return z, rho, False
        rho_new = rho_half - 0.5 * eps * grad_u_new
        if not torch.isfinite(rho_new).all():
            return z, rho, False
        return z_new, rho_new, True

    def _one_step(self, z: torch.Tensor, step_idx: int) -> tuple[torch.Tensor, dict[str, float]]:
        rho0 = torch.randn_like(z) / math.sqrt(max(self.cfg.beta_zero, 1e-8))
        h0 = self._hamiltonian(z, rho0, step_idx)

        accepted = torch.zeros(z.shape[0], device=z.device, dtype=torch.bool)
        z_out = z.clone()
        nonfinite_rejects = 0.0
        eps_used = float(self.cfg.eps_lf)

        for _trial in range(max(1, self.cfg.max_backoff_trials)):
            z_prop = z.clone()
            rho_prop = rho0.clone()
            ok = True
            for _ in range(self.cfg.n_lf):
                z_prop, rho_prop, ok = self._leapfrog(z_prop, rho_prop, eps_used, step_idx)
                if not ok:
                    break
            if ok:
                h1 = self._hamiltonian(z_prop, rho_prop, step_idx)
                delta = h1 - h0
                delta = torch.clamp(delta, min=-80.0, max=80.0)
                alpha = torch.exp(-delta).clamp(max=1.0)
                u = torch.rand_like(alpha)
                accepted = (u < alpha)
                z_out = torch.where(accepted.view(-1, 1), z_prop, z)
                break
            nonfinite_rejects += 1.0
            eps_used = eps_used * float(self.cfg.eps_backoff)

        return z_out.detach(), {
            "acceptance_rate": float(accepted.float().mean().item()),
            "nonfinite_rejects": float(nonfinite_rejects),
            "eps_used": float(eps_used),
        }

    def run_chain(self, z0: torch.Tensor, horizon: int) -> tuple[torch.Tensor, dict[str, float]]:
        z = z0.to(self.device).detach().clone()
        traj = [z.detach().cpu()]
        accepts: list[float] = []
        rejects: list[float] = []
        for step in range(int(horizon)):
            z, diag = self._one_step(z, step_idx=step)
            traj.append(z.detach().cpu())
            accepts.append(diag["acceptance_rate"])
            rejects.append(diag["nonfinite_rejects"])
        self.last_acceptance_rate = float(np.mean(accepts)) if accepts else float("nan")
        return torch.stack(traj, dim=1).squeeze(0) if z0.shape[0] == 1 else torch.stack(traj, dim=1), {
            "acceptance_rate_mean": self.last_acceptance_rate,
            "nonfinite_rejects_total": float(np.sum(rejects)),
        }

    def sample(self, n_samples: int) -> tuple[torch.Tensor, dict[str, float]]:
        z = torch.randn(int(n_samples), self.adapter.latent_dim, device=self.device)
        accepts: list[float] = []
        rejects: list[float] = []
        for step in range(self.cfg.mcmc_steps):
            z, diag = self._one_step(z, step_idx=step)
            accepts.append(diag["acceptance_rate"])
            rejects.append(diag["nonfinite_rejects"])
        self.last_acceptance_rate = float(np.mean(accepts)) if accepts else float("nan")
        return z.detach(), {
            "acceptance_rate_mean": self.last_acceptance_rate,
            "nonfinite_rejects_total": float(np.sum(rejects)),
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _required_files(model_id: str) -> list[str]:
    if model_id == "vanilla_vae":
        return ["model.pt", "config.json", "train_data.pt", "metrics.json"]
    if model_id in {"rhvae_standard", "aniso_rs1", "aniso_rs6"}:
        return ["rhvae_model.pt", "rhvae_metric.pt"]
    if model_id == "ebm_conformal":
        return ["encoder.pt", "decoder.pt", "ebm.pt", "config.json", "train_data.pt", "metrics.json"]
    raise ValueError(f"Unsupported model_id: {model_id}")


def _find_latest_run(parent: Path, required_files: list[str]) -> Path | None:
    if not parent.exists() or not parent.is_dir():
        return None
    runs = sorted([p for p in parent.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        if all((run / f).exists() for f in required_files):
            return run
    return None


def _resolve_model_dir(model_root: Path, model_id: str, n: int, seed: int) -> Path | None:
    parent = model_root / model_id / f"N{int(n):03d}" / f"seed{int(seed):03d}"
    return _find_latest_run(parent, _required_files(model_id))


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def _ensure_data_ready(processed_dir: Path) -> None:
    required = [processed_dir / "train.pt", processed_dir / "val.pt", processed_dir / "test.pt"]
    if all(p.exists() for p in required):
        return
    _run([sys.executable, "scripts/prepare_rotmnist_lowdata.py", "--processed_dir", str(processed_dir)])


def _ensure_model_ready(
    model_root: Path,
    processed_dir: Path,
    model_id: str,
    n: int,
    seed: int,
    epochs: int,
    isotropic_stretch: float,
    anisotropic_stretch: float,
) -> Path:
    existing = _resolve_model_dir(model_root, model_id, n, seed)
    if existing is not None:
        return existing

    if model_id == "vanilla_vae":
        cmd = [
            sys.executable,
            "scripts/train_vanilla_vae.py",
            "--processed_dir",
            str(processed_dir),
            "--n",
            str(n),
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--output_root",
            str(model_root),
            "--model_id",
            model_id,
        ]
    elif model_id == "rhvae_standard":
        cmd = [
            sys.executable,
            "scripts/train_rhvae_tensor.py",
            "--mode",
            "standard",
            "--processed_dir",
            str(processed_dir),
            "--n",
            str(n),
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--output_root",
            str(model_root),
            "--model_id",
            model_id,
        ]
    elif model_id in {"aniso_rs1", "aniso_rs6"}:
        stretch = isotropic_stretch if model_id == "aniso_rs1" else anisotropic_stretch
        cmd = [
            sys.executable,
            "scripts/train_rhvae_tensor.py",
            "--mode",
            "aniso",
            "--processed_dir",
            str(processed_dir),
            "--n",
            str(n),
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--radial_stretch",
            str(stretch),
            "--output_root",
            str(model_root),
            "--model_id",
            model_id,
        ]
    elif model_id == "ebm_conformal":
        half_epochs = max(2, int(epochs // 2))
        cmd = [
            sys.executable,
            "scripts/train_ebm_conformal.py",
            "--processed_dir",
            str(processed_dir),
            "--n",
            str(n),
            "--seed",
            str(seed),
            "--ae_epochs",
            str(half_epochs),
            "--ebm_epochs",
            str(half_epochs),
            "--output_root",
            str(model_root),
            "--model_id",
            model_id,
        ]
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")

    _run(cmd)
    resolved = _resolve_model_dir(model_root, model_id, n, seed)
    if resolved is None:
        raise RuntimeError(f"Model {model_id} N={n} seed={seed} was not created")
    return resolved


def _make_adapter(model_id: str, model_dir: Path, device: torch.device) -> BenchmarkAdapter:
    if model_id == "vanilla_vae":
        return VanillaAdapter(model_dir, device=device, model_id=model_id)
    if model_id in {"rhvae_standard", "aniso_rs1", "aniso_rs6"}:
        return RHVAEAdapter(model_dir, device=device, model_id=model_id)
    if model_id == "ebm_conformal":
        return EBMConformalAdapter(model_dir, device=device, model_id=model_id)
    raise ValueError(f"Unsupported model_id: {model_id}")


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _safe_std(values: list[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.std())


def _bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, arr.size, size=arr.size)
        boots.append(float(np.mean(arr[idx])))
    return float(np.quantile(boots, alpha / 2.0)), float(np.quantile(boots, 1.0 - alpha / 2.0))


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


def _nearest_neighbor_threshold(real_lat: torch.Tensor, quantile: float = 0.9) -> float:
    if real_lat.shape[0] < 2:
        return 1e-6
    d = torch.cdist(real_lat, real_lat)
    eye = torch.eye(d.shape[0], dtype=torch.bool, device=d.device)
    d = d.masked_fill(eye, float("inf"))
    nn = d.min(dim=1).values.detach().cpu().numpy()
    q = float(np.quantile(nn, quantile))
    return max(1e-6, q)


def _sample_void_starts(real_lat: torch.Tensor, n_ring: int, n_gauss: int, seed: int) -> torch.Tensor:
    device = real_lat.device
    latent_dim = real_lat.shape[1]
    center = real_lat.mean(dim=0)
    radius = float(2.5 * torch.linalg.norm(real_lat - center, dim=1).max().item())

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    starts = []
    if n_ring > 0:
        phase = float(torch.rand(1, generator=gen, device=device).item()) * (2.0 * math.pi)
        theta = torch.linspace(0.0, 2.0 * math.pi, n_ring + 1, device=device)[:-1] + phase
        ring = center.unsqueeze(0).repeat(n_ring, 1)
        ring[:, 0] = center[0] + radius * torch.cos(theta)
        if latent_dim > 1:
            ring[:, 1] = center[1] + radius * torch.sin(theta)
        starts.append(ring)

    if n_gauss > 0:
        std = max(1.0, radius / 2.0)
        gauss = center.unsqueeze(0) + torch.randn(n_gauss, latent_dim, device=device, generator=gen) * std
        starts.append(gauss)

    if not starts:
        return center.unsqueeze(0)
    return torch.cat(starts, dim=0)


def _rescue_directionality(path: torch.Tensor, real_lat: torch.Tensor, tau: float) -> float:
    # path shape [T+1, D]
    if path.shape[0] < 2:
        return float("nan")
    with torch.no_grad():
        d = torch.cdist(path, real_lat)
        nn_idx = torch.argmin(d, dim=1)
        nearest = real_lat[nn_idx]
        min_d = d[torch.arange(path.shape[0], device=path.device), nn_idx]

        step = path[1:] - path[:-1]
        to_manifold = nearest[:-1] - path[:-1]
        mask = min_d[:-1] > float(tau)
        if not torch.any(mask):
            return float("nan")

        step_sel = step[mask]
        to_sel = to_manifold[mask]
        denom = torch.linalg.norm(step_sel, dim=1) * torch.linalg.norm(to_sel, dim=1)
        denom = torch.clamp(denom, min=1e-12)
        cos = torch.sum(step_sel * to_sel, dim=1) / denom
        cos = torch.clamp(cos, -1.0, 1.0)
        return float(torch.mean(cos).item())


def compute_manifold_rescue_metrics(
    adapter: BenchmarkAdapter,
    sampler_cfg: VolumeSamplerConfig,
    real_lat: torch.Tensor,
    n_ring: int,
    n_gauss: int,
    horizon: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    tau = _nearest_neighbor_threshold(real_lat, quantile=0.9)
    starts = _sample_void_starts(real_lat, n_ring=n_ring, n_gauss=n_gauss, seed=seed)
    sampler = VolumePotentialSampler(adapter=adapter, cfg=sampler_cfg, device=device)

    all_min_d: list[np.ndarray] = []
    hit_steps_all: list[int] = []
    dir_scores: list[float] = []

    for i in range(starts.shape[0]):
        chain, diag = sampler.run_chain(starts[i : i + 1], horizon=horizon)
        del diag
        chain = chain.to(device)
        with torch.no_grad():
            min_d = torch.cdist(chain, real_lat).min(dim=1).values
        all_min_d.append(min_d.detach().cpu().numpy())
        hs = compute_hit_steps(min_d.detach().cpu().numpy().reshape(1, -1), threshold=float(tau))[0]
        hit_steps_all.append(int(hs))
        dir_scores.append(_rescue_directionality(chain, real_lat, tau=float(tau)))

    dists = np.stack(all_min_d, axis=0)
    hit_steps = np.asarray(hit_steps_all, dtype=int)
    rescue_rate = rescue_rate_at_horizon(hit_steps, horizon=int(horizon))

    valid_steps = hit_steps[hit_steps >= 0]
    if valid_steps.size == 0:
        median_steps = float(horizon + 1)
    else:
        median_steps = float(np.median(valid_steps))

    rescue_curve = []
    for t in range(int(horizon) + 1):
        rescue_curve.append(float(np.mean((hit_steps >= 0) & (hit_steps <= t))))

    return {
        "tau": float(tau),
        "rescue_rate": float(rescue_rate),
        "median_steps_to_rescue": float(median_steps),
        "rescue_curve": rescue_curve,
        "rescue_directionality": _safe_mean([float(x) for x in dir_scores]),
        "sampler_acceptance_mean": float(sampler.last_acceptance_rate),
        "hit_steps": hit_steps.tolist(),
    }


def _select_farthest_interclass_pairs(latents: torch.Tensor, labels: torch.Tensor, n_pairs: int) -> list[tuple[int, int]]:
    if latents.shape[0] < 2:
        return []
    d = torch.cdist(latents, latents).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    cand = []
    for i in range(latents.shape[0]):
        for j in range(i + 1, latents.shape[0]):
            if int(y[i]) == int(y[j]):
                continue
            cand.append((float(d[i, j]), i, j))
    cand.sort(key=lambda x: x[0], reverse=True)
    out: list[tuple[int, int]] = []
    used = set()
    for _, i, j in cand:
        if len(out) >= int(n_pairs):
            break
        key = tuple(sorted((int(i), int(j))))
        if key in used:
            continue
        used.add(key)
        out.append((int(i), int(j)))
    return out


def latent_d_rmse(path_latents: torch.Tensor, real_latents: torch.Tensor) -> float:
    if path_latents.numel() == 0 or real_latents.numel() == 0:
        return float("nan")
    d = torch.cdist(path_latents, real_latents)
    min_d = d.min(dim=1).values
    return float(torch.sqrt(torch.mean(min_d * min_d)).item())


def compute_latent_path_metrics(
    adapter: BenchmarkAdapter,
    train_lat: torch.Tensor,
    train_y: torch.Tensor,
    tau: float,
    n_pairs: int,
    n_steps: int,
) -> dict[str, float]:
    pairs = _select_farthest_interclass_pairs(train_lat, train_y, n_pairs=n_pairs)
    if not pairs:
        return {
            "latent_d_rmse": float("nan"),
            "path_hugging_fraction": float("nan"),
            "n_pairs": 0.0,
        }

    drmse_vals: list[float] = []
    hugging: list[float] = []
    adapter_device = getattr(adapter, "device", train_lat.device)

    for i, j in pairs:
        z1 = train_lat[i].to(adapter_device)
        z2 = train_lat[j].to(z1.device)
        path = adapter.interpolate(z1, z2, n_steps=n_steps).detach().cpu()
        drmse_vals.append(latent_d_rmse(path, train_lat.cpu()))
        with torch.no_grad():
            min_d = torch.cdist(path, train_lat.cpu()).min(dim=1).values
            hugging.append(float(torch.mean((min_d <= float(tau)).float()).item()))

    return {
        "latent_d_rmse": _safe_mean(drmse_vals),
        "path_hugging_fraction": _safe_mean(hugging),
        "n_pairs": float(len(pairs)),
    }


def _pseudo_label_from_latent(
    z_syn: torch.Tensor,
    z_real: torch.Tensor,
    y_real: torch.Tensor,
) -> torch.Tensor:
    class_proto: dict[int, torch.Tensor] = {}
    for cls in torch.unique(y_real).tolist():
        cls_i = int(cls)
        mask = y_real == cls_i
        if int(mask.sum().item()) > 0:
            class_proto[cls_i] = z_real[mask].mean(dim=0)

    if not class_proto:
        return torch.zeros((z_syn.shape[0],), dtype=torch.long)

    proto_cls = sorted(class_proto.keys())
    proto_lat = torch.stack([class_proto[c] for c in proto_cls], dim=0)
    d = torch.cdist(z_syn.cpu(), proto_lat.cpu())
    nn = torch.argmin(d, dim=1)
    return torch.tensor([proto_cls[int(i)] for i in nn], dtype=torch.long)


def compute_balanced_accuracy_gain(
    adapter: BenchmarkAdapter,
    sampler_cfg: VolumeSamplerConfig,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    n_syn: int,
    device: torch.device,
) -> dict[str, float]:
    z_real = adapter.encode(train_x.to(device)).detach().cpu()

    sampler = VolumePotentialSampler(adapter=adapter, cfg=sampler_cfg, device=device)
    z_syn, sampler_diag = sampler.sample(n_samples=int(n_syn))
    x_syn = adapter.decode(z_syn).detach().cpu()
    y_syn = _pseudo_label_from_latent(z_syn.detach().cpu(), z_real, train_y.cpu())

    empty_x = torch.zeros((0, 1, train_x.shape[-2], train_x.shape[-1]), dtype=train_x.dtype)
    empty_y = torch.zeros((0,), dtype=torch.long)

    real_only = augmentation_eval(
        real_train_x=train_x.cpu(),
        real_train_y=train_y.cpu(),
        synthetic_x=empty_x,
        synthetic_y=empty_y,
        test_x=test_x.cpu(),
        test_y=test_y.cpu(),
        device=device,
        epochs=5,
        batch_size=64,
        lr=1e-3,
    )

    augmented = augmentation_eval(
        real_train_x=train_x.cpu(),
        real_train_y=train_y.cpu(),
        synthetic_x=x_syn,
        synthetic_y=y_syn,
        test_x=test_x.cpu(),
        test_y=test_y.cpu(),
        device=device,
        epochs=5,
        batch_size=64,
        lr=1e-3,
    )

    return {
        "balanced_accuracy_real_only": float(real_only["balanced_accuracy"]),
        "balanced_accuracy_augmented": float(augmented["balanced_accuracy"]),
        "balanced_accuracy_gain": float(augmented["balanced_accuracy"] - real_only["balanced_accuracy"]),
        "sampler_acceptance_gen": float(sampler_diag.get("acceptance_rate_mean", float("nan"))),
    }


def _stats_by_model_n(rows: list[dict[str, Any]], subset_ns: list[int], models: list[str], bootstrap_samples: int) -> dict[str, Any]:
    metrics = [
        "rescue_rate",
        "median_steps_to_rescue",
        "latent_d_rmse",
        "path_hugging_fraction",
        "balanced_accuracy_real_only",
        "balanced_accuracy_augmented",
        "balanced_accuracy_gain",
        "rescue_directionality",
        "sampler_acceptance_mean",
    ]
    out: dict[str, Any] = {}
    for n in subset_ns:
        out[str(n)] = {}
        for model_id in models:
            sub = [r for r in rows if r.get("status") == "ok" and int(r.get("n", -1)) == int(n) and r.get("model_id") == model_id]
            out[str(n)][model_id] = {}
            for m in metrics:
                vals = [float(r.get(m, float("nan"))) for r in sub]
                lo, hi = _bootstrap_ci(vals, n_boot=bootstrap_samples, alpha=0.05, seed=0)
                out[str(n)][model_id][m] = {
                    "mean": _safe_mean(vals),
                    "std": _safe_std(vals),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n": int(np.sum(np.isfinite(np.asarray(vals, dtype=float)))),
                }
    return out


def _paired_stat_tests(rows: list[dict[str, Any]], subset_ns: list[int], subset_seeds: list[int]) -> dict[str, Any]:
    metrics = {
        "rescue_rate": "greater",
        "latent_d_rmse": "less",
        "balanced_accuracy_gain": "greater",
    }
    baselines = ["vanilla_vae", "rhvae_standard", "ebm_conformal"]

    out: dict[str, Any] = {}
    for n in subset_ns:
        out[str(n)] = {}
        for metric, alternative in metrics.items():
            rows_metric = []
            pvals: list[tuple[str, float]] = []
            for baseline in baselines:
                pairs = []
                for seed in subset_seeds:
                    ra = next(
                        (r for r in rows if r.get("status") == "ok" and int(r.get("n", -1)) == int(n) and int(r.get("seed", -1)) == int(seed) and r.get("model_id") == "aniso_rs6"),
                        None,
                    )
                    rb = next(
                        (r for r in rows if r.get("status") == "ok" and int(r.get("n", -1)) == int(n) and int(r.get("seed", -1)) == int(seed) and r.get("model_id") == baseline),
                        None,
                    )
                    if ra is None or rb is None:
                        continue
                    va = float(ra.get(metric, float("nan")))
                    vb = float(rb.get(metric, float("nan")))
                    if np.isfinite(va) and np.isfinite(vb):
                        pairs.append((int(seed), va, vb))

                if not pairs:
                    rows_metric.append(
                        {
                            "baseline": baseline,
                            "n_pairs": 0,
                            "p_value": float("nan"),
                            "p_holm": float("nan"),
                            "aniso_rs6_minus_baseline_mean": float("nan"),
                        }
                    )
                    continue

                arr_a = np.asarray([p[1] for p in pairs], dtype=float)
                arr_b = np.asarray([p[2] for p in pairs], dtype=float)
                diff = arr_a - arr_b
                try:
                    p = float(wilcoxon(arr_a, arr_b, alternative=alternative).pvalue)
                except Exception:
                    p = float("nan")

                if np.isfinite(p):
                    pvals.append((baseline, p))

                rows_metric.append(
                    {
                        "baseline": baseline,
                        "n_pairs": int(len(pairs)),
                        "p_value": p,
                        "p_holm": float("nan"),
                        "aniso_rs6_minus_baseline_mean": float(np.mean(diff)),
                    }
                )

            holm = _holm_adjust(pvals)
            for row in rows_metric:
                b = row["baseline"]
                if b in holm:
                    row["p_holm"] = float(holm[b])
            out[str(n)][metric] = rows_metric

    return out


def _anisotropy_ablation(rows: list[dict[str, Any]], subset_ns: list[int], subset_seeds: list[int]) -> dict[str, Any]:
    metrics = {
        "rescue_rate": "greater",
        "latent_d_rmse": "less",
        "balanced_accuracy_gain": "greater",
    }

    out: dict[str, Any] = {}
    for n in subset_ns:
        out[str(n)] = {}
        for metric, alternative in metrics.items():
            pairs = []
            for seed in subset_seeds:
                r6 = next(
                    (r for r in rows if r.get("status") == "ok" and int(r.get("n", -1)) == int(n) and int(r.get("seed", -1)) == int(seed) and r.get("model_id") == "aniso_rs6"),
                    None,
                )
                r1 = next(
                    (r for r in rows if r.get("status") == "ok" and int(r.get("n", -1)) == int(n) and int(r.get("seed", -1)) == int(seed) and r.get("model_id") == "aniso_rs1"),
                    None,
                )
                if r6 is None or r1 is None:
                    continue
                v6 = float(r6.get(metric, float("nan")))
                v1 = float(r1.get(metric, float("nan")))
                if np.isfinite(v6) and np.isfinite(v1):
                    pairs.append((int(seed), v6, v1))

            if not pairs:
                out[str(n)][metric] = {
                    "n_pairs": 0,
                    "p_value": float("nan"),
                    "mean_delta_aniso_rs6_minus_rs1": float("nan"),
                    "supports_h3": False,
                }
                continue

            a6 = np.asarray([p[1] for p in pairs], dtype=float)
            a1 = np.asarray([p[2] for p in pairs], dtype=float)
            delta = a6 - a1
            try:
                p = float(wilcoxon(a6, a1, alternative=alternative).pvalue)
            except Exception:
                p = float("nan")

            mean_delta = float(np.mean(delta))
            if metric == "latent_d_rmse":
                supports = np.isfinite(p) and p < 0.05 and mean_delta < 0.0
            else:
                supports = np.isfinite(p) and p < 0.05 and mean_delta > 0.0

            out[str(n)][metric] = {
                "n_pairs": int(len(pairs)),
                "p_value": p,
                "mean_delta_aniso_rs6_minus_rs1": mean_delta,
                "supports_h3": bool(supports),
            }

    return out


def _aggregate_failure_metrics(energies: list[dict[str, np.ndarray]]) -> dict[str, float]:
    acc = []
    k_all = []
    u_all = []
    h_all = []
    for e in energies:
        a = np.asarray(e.get("accept", np.array([])), dtype=float)
        if a.size:
            acc.append(float(np.mean(a)))
        k = np.asarray(e.get("kinetic", np.array([])), dtype=float)
        u = np.asarray(e.get("potential", np.array([])), dtype=float)
        h = np.asarray(e.get("hamiltonian", np.array([])), dtype=float)
        if k.size:
            k_all.append(k)
        if u.size:
            u_all.append(u)
        if h.size:
            h_all.append(h)

    k_cat = np.concatenate(k_all) if k_all else np.array([], dtype=float)
    u_cat = np.concatenate(u_all) if u_all else np.array([], dtype=float)
    h_cat = np.concatenate(h_all) if h_all else np.array([], dtype=float)

    return {
        "acceptance_mean": float(np.mean(acc)) if acc else float("nan"),
        "kinetic_p95": float(np.percentile(k_cat, 95)) if k_cat.size else float("nan"),
        "kinetic_max": float(np.max(k_cat)) if k_cat.size else float("nan"),
        "potential_p95": float(np.percentile(u_cat, 95)) if u_cat.size else float("nan"),
        "hamiltonian_std": float(np.std(h_cat)) if h_cat.size else float("nan"),
        "divergence_fraction": float(np.mean(~np.isfinite(h_cat))) if h_cat.size else float("nan"),
    }


def _plot_failure_traces(
    out_dir: Path,
    energies_volume: list[dict[str, np.ndarray]],
    energies_riem: list[dict[str, np.ndarray]],
) -> None:
    def _mean_trace(energies: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
        arrs = [np.asarray(e.get(key, np.array([])), dtype=float) for e in energies]
        arrs = [a for a in arrs if a.size > 0]
        if not arrs:
            return np.array([])
        m = min(len(a) for a in arrs)
        if m <= 0:
            return np.array([])
        stack = np.stack([a[:m] for a in arrs], axis=0)
        return stack.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    for col, (title, energies) in enumerate(
        [
            ("volume (Euclidean)", energies_volume),
            ("volume_riemannian", energies_riem),
        ]
    ):
        ax = axes[0, col]
        k = _mean_trace(energies, "kinetic")
        u = _mean_trace(energies, "potential")
        h = _mean_trace(energies, "hamiltonian")
        x = np.arange(len(k)) if len(k) else np.arange(len(u))
        if len(k):
            ax.plot(x[: len(k)], k, label="K")
        if len(u):
            ax.plot(x[: len(u)], u, label="U")
        if len(h):
            ax.plot(x[: len(h)], h, label="H")
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("energy")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figures" / "failure_energy_traces_volume_vs_riemannian.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    for col, (title, energies) in enumerate(
        [
            ("volume (Euclidean)", energies_volume),
            ("volume_riemannian", energies_riem),
        ]
    ):
        ax = axes[0, col]
        for e in energies:
            k = np.asarray(e.get("kinetic", np.array([])), dtype=float)
            u = np.asarray(e.get("potential", np.array([])), dtype=float)
            if k.size and u.size:
                m = min(k.size, u.size)
                ax.scatter(u[:m], k[:m], s=8, alpha=0.25)
        ax.set_title(f"K vs U: {title}")
        ax.set_xlabel("U")
        ax.set_ylabel("K")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "failure_k_vs_u_volume_vs_riemannian.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_failure_analysis(
    model_dir: Path,
    train_lat: torch.Tensor,
    cfg: FailureDiagConfig,
    volume_power: float,
    beta_zero: float,
    device: torch.device,
    out_dir: Path,
    seed: int,
) -> dict[str, Any]:
    adapter = RHVAEAdapter(model_dir=model_dir, device=device, model_id="aniso_rs6")
    model = adapter.model

    starts = _sample_void_starts(train_lat.to(device), n_ring=max(1, cfg.chains // 2), n_gauss=max(1, cfg.chains - cfg.chains // 2), seed=seed)
    starts = starts[: cfg.chains]

    energies_volume: list[dict[str, np.ndarray]] = []
    energies_riem: list[dict[str, np.ndarray]] = []

    base_stretch = float(getattr(model, "radial_stretch", 1.0))

    for i in range(starts.shape[0]):
        start = starts[i : i + 1]

        model.radial_stretch = base_stretch
        sampler_v = _build_demo_sampler(
            "volume",
            model,
            mcmc_steps=cfg.horizon,
            n_lf=cfg.n_lf,
            eps_lf=cfg.eps_volume,
            beta_zero=beta_zero,
            volume_power=volume_power,
        )
        _, e_v = _run_demo_chain(start, sampler_v, cfg.horizon, cfg.n_lf, cfg.eps_volume)
        energies_volume.append(e_v)

        model.radial_stretch = base_stretch * float(cfg.stretch_factor)
        sampler_r = _build_demo_sampler(
            "volume_riemannian",
            model,
            mcmc_steps=cfg.horizon,
            n_lf=cfg.n_lf,
            eps_lf=cfg.eps_riem,
            beta_zero=beta_zero,
            volume_power=volume_power,
        )
        _, e_r = _run_demo_chain(start, sampler_r, cfg.horizon, cfg.n_lf, cfg.eps_riem)
        energies_riem.append(e_r)

    model.radial_stretch = base_stretch

    _plot_failure_traces(out_dir=out_dir, energies_volume=energies_volume, energies_riem=energies_riem)

    m_v = _aggregate_failure_metrics(energies_volume)
    m_r = _aggregate_failure_metrics(energies_riem)
    spike_ratio = float(m_r["kinetic_p95"] / max(m_v["kinetic_p95"], 1e-12)) if np.isfinite(m_r["kinetic_p95"]) else float("nan")

    return {
        "model_dir": str(model_dir),
        "n_chains": int(starts.shape[0]),
        "stretch_base": float(base_stretch),
        "stretch_riemannian": float(base_stretch * float(cfg.stretch_factor)),
        "volume": m_v,
        "volume_riemannian": m_r,
        "kinetic_p95_spike_ratio_riem_over_volume": spike_ratio,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_handoff_prompt(path: Path) -> None:
    text = """Role: Senior ML Research Engineer. Project: AnisoRHVAE low-data paper protocol.

Goal:
Implement and run scripts/low_data_benchmark.py for RotMNIST HDLSS (N=50,100), comparing vanilla_vae, rhvae_standard, ebm_conformal, aniso_rs1 (radial_stretch=1.0), aniso_rs6 (radial_stretch=6.0). Use volume (Euclidean momentum) as the primary sampler for all quantitative metrics.

Core scientific framing:
We are prioritizing Riemannian potential / gravity-well effects over Riemannian kinetic trajectories. Show that geometric potential structure drives rescue/connectivity, while volume_riemannian can destabilize under strong anisotropy.

Required outputs:
1) Primary metrics per model/N/seed:
- rescue_rate@horizon, median_steps_to_rescue
- latent_d_rmse (path-to-manifold proximity)
- balanced_accuracy_gain with M=10N synthetic samples
2) Anisotropy ablation:
- paired comparison aniso_rs1 vs aniso_rs6 with Wilcoxon + Holm
3) EBM comparison:
- explicit conformal metric proxy G_inv(z)=exp(-beta*E(z))I
- paired comparison aniso_rs6 vs ebm_conformal on primary metrics
4) Failure analysis:
- volume_riemannian trajectories from void starts
- plots of K(t), U(t), H(t), and K-vs-U showing kinetic spikes/instability
5) Artifacts:
- raw_metrics.csv, summary_by_model_n.json, stat_tests.json, anisotropy_ablation.json, failure_volume_riemannian.json, figures/*
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Low-data gravity-well benchmark protocol for RotMNIST.")
    p.add_argument("--processed_dir", type=str, default="data/processed/rotmnist/v1")
    p.add_argument("--model_root", type=str, default="outputs/low_data_models/rotmnist_v1")
    p.add_argument("--subset_ns", nargs="+", type=int, default=[50, 100])
    p.add_argument("--subset_seeds", nargs="+", type=int, default=[13, 29, 47, 71, 89])
    p.add_argument("--output_dir", type=str, default="results/low_data_benchmark_protocol")
    p.add_argument("--epochs", type=int, default=20)

    p.add_argument("--isotropic_stretch", type=float, default=1.0)
    p.add_argument("--anisotropic_stretch", type=float, default=6.0)

    p.add_argument("--volume_power", type=float, default=1.0)
    p.add_argument("--beta_zero", type=float, default=8.0)
    p.add_argument("--mcmc_steps", type=int, default=60)
    p.add_argument("--n_lf", type=int, default=20)
    p.add_argument("--eps_lf", type=float, default=0.02)
    p.add_argument("--grad_clip_norm", type=float, default=25.0)
    p.add_argument("--logdet_clip", type=float, default=80.0)
    p.add_argument("--eps_backoff", type=float, default=0.5)
    p.add_argument("--max_backoff_trials", type=int, default=4)
    p.add_argument("--warmup_steps", type=int, default=8)

    p.add_argument("--rescue_ring_starts", type=int, default=16)
    p.add_argument("--rescue_gaussian_starts", type=int, default=16)
    p.add_argument("--rescue_horizon", type=int, default=40)

    p.add_argument("--drmse_pairs", type=int, default=64)
    p.add_argument("--drmse_steps", type=int, default=32)

    p.add_argument("--augment_multiplier", type=int, default=10)
    p.add_argument("--bootstrap_samples", type=int, default=1000)

    p.add_argument("--failure_stretch_factor", type=float, default=2.0)
    p.add_argument("--failure_chains", type=int, default=16)
    p.add_argument("--failure_horizon", type=int, default=60)
    p.add_argument("--failure_n_lf", type=int, default=20)
    p.add_argument("--failure_eps_volume", type=float, default=0.02)
    p.add_argument("--failure_eps_riem", type=float, default=0.05)

    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _seed_everything(0)

    if args.quick:
        args.subset_ns = [50]
        args.subset_seeds = [13]
        args.epochs = min(args.epochs, 1)
        args.augment_multiplier = min(args.augment_multiplier, 1)
        args.bootstrap_samples = min(args.bootstrap_samples, 200)
        args.mcmc_steps = min(args.mcmc_steps, 8)
        args.n_lf = min(args.n_lf, 4)
        args.rescue_ring_starts = min(args.rescue_ring_starts, 2)
        args.rescue_gaussian_starts = min(args.rescue_gaussian_starts, 2)
        args.rescue_horizon = min(args.rescue_horizon, 12)
        args.drmse_pairs = min(args.drmse_pairs, 4)
        args.drmse_steps = min(args.drmse_steps, 8)
        args.failure_chains = min(args.failure_chains, 2)
        args.failure_horizon = min(args.failure_horizon, 8)
        args.failure_n_lf = min(args.failure_n_lf, 4)

    bench_cfg = BenchmarkConfig(
        processed_dir=(ROOT / args.processed_dir).resolve(),
        output_dir=(ROOT / args.output_dir).resolve(),
        model_root=(ROOT / args.model_root).resolve(),
        subset_ns=[int(x) for x in args.subset_ns],
        subset_seeds=[int(x) for x in args.subset_seeds],
        epochs=int(args.epochs),
        augment_multiplier=int(args.augment_multiplier),
    )
    sampler_cfg = VolumeSamplerConfig(
        volume_power=float(args.volume_power),
        beta_zero=float(args.beta_zero),
        mcmc_steps=int(args.mcmc_steps),
        n_lf=int(args.n_lf),
        eps_lf=float(args.eps_lf),
        grad_clip_norm=float(args.grad_clip_norm),
        logdet_clip=float(args.logdet_clip),
        eps_backoff=float(args.eps_backoff),
        max_backoff_trials=int(args.max_backoff_trials),
        warmup_steps=int(args.warmup_steps),
    )
    failure_cfg = FailureDiagConfig(
        stretch_factor=float(args.failure_stretch_factor),
        chains=int(args.failure_chains),
        horizon=int(args.failure_horizon),
        n_lf=int(args.failure_n_lf),
        eps_volume=float(args.failure_eps_volume),
        eps_riem=float(args.failure_eps_riem),
    )

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = bench_cfg.output_dir / ts
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "prompts").mkdir(parents=True, exist_ok=True)

    _ensure_data_ready(bench_cfg.processed_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_train_x, full_train_y = load_split_tensor(bench_cfg.processed_dir, "train")
    test_x, test_y = load_split_tensor(bench_cfg.processed_dir, "test")

    models = ["vanilla_vae", "rhvae_standard", "ebm_conformal", "aniso_rs1", "aniso_rs6"]
    rows: list[dict[str, Any]] = []

    canonical_failure_model: Path | None = None
    canonical_failure_lat: torch.Tensor | None = None

    for n in bench_cfg.subset_ns:
        for seed in bench_cfg.subset_seeds:
            subset_idx = load_subset_indices(bench_cfg.processed_dir, n, seed)
            train_x, train_y = subset_from_indices(full_train_x, full_train_y, subset_idx)

            for model_id in models:
                try:
                    model_dir = _ensure_model_ready(
                        model_root=bench_cfg.model_root,
                        processed_dir=bench_cfg.processed_dir,
                        model_id=model_id,
                        n=n,
                        seed=seed,
                        epochs=bench_cfg.epochs,
                        isotropic_stretch=float(args.isotropic_stretch),
                        anisotropic_stretch=float(args.anisotropic_stretch),
                    )
                    adapter = _make_adapter(model_id, model_dir, device=device)

                    real_lat = adapter.encode(train_x.to(device)).detach().cpu()

                    rescue = compute_manifold_rescue_metrics(
                        adapter=adapter,
                        sampler_cfg=sampler_cfg,
                        real_lat=real_lat.to(device),
                        n_ring=int(args.rescue_ring_starts),
                        n_gauss=int(args.rescue_gaussian_starts),
                        horizon=int(args.rescue_horizon),
                        seed=seed,
                        device=device,
                    )

                    path_metrics = compute_latent_path_metrics(
                        adapter=adapter,
                        train_lat=real_lat,
                        train_y=train_y,
                        tau=float(rescue["tau"]),
                        n_pairs=int(args.drmse_pairs),
                        n_steps=int(args.drmse_steps),
                    )

                    n_syn = int(max(1, bench_cfg.augment_multiplier * int(n)))
                    downstream = compute_balanced_accuracy_gain(
                        adapter=adapter,
                        sampler_cfg=sampler_cfg,
                        train_x=train_x,
                        train_y=train_y,
                        test_x=test_x,
                        test_y=test_y,
                        n_syn=n_syn,
                        device=device,
                    )

                    row = {
                        "status": "ok",
                        "n": int(n),
                        "seed": int(seed),
                        "model_id": str(model_id),
                        "model_dir": str(model_dir),
                        "tau": float(rescue["tau"]),
                        "rescue_rate": float(rescue["rescue_rate"]),
                        "median_steps_to_rescue": float(rescue["median_steps_to_rescue"]),
                        "rescue_directionality": float(rescue["rescue_directionality"]),
                        "sampler_acceptance_mean": float(rescue["sampler_acceptance_mean"]),
                        "latent_d_rmse": float(path_metrics["latent_d_rmse"]),
                        "path_hugging_fraction": float(path_metrics["path_hugging_fraction"]),
                        "balanced_accuracy_real_only": float(downstream["balanced_accuracy_real_only"]),
                        "balanced_accuracy_augmented": float(downstream["balanced_accuracy_augmented"]),
                        "balanced_accuracy_gain": float(downstream["balanced_accuracy_gain"]),
                        "sampler_acceptance_gen": float(downstream["sampler_acceptance_gen"]),
                    }
                    rows.append(row)

                    if canonical_failure_model is None and model_id == "aniso_rs6" and int(n) == bench_cfg.subset_ns[0] and int(seed) == bench_cfg.subset_seeds[0]:
                        canonical_failure_model = model_dir
                        canonical_failure_lat = real_lat.clone()

                except Exception as exc:
                    rows.append(
                        {
                            "status": f"error:{exc}",
                            "n": int(n),
                            "seed": int(seed),
                            "model_id": str(model_id),
                        }
                    )

    summary = _stats_by_model_n(rows, bench_cfg.subset_ns, models, bootstrap_samples=int(args.bootstrap_samples))
    stat_tests = _paired_stat_tests(rows, bench_cfg.subset_ns, bench_cfg.subset_seeds)
    ablation = _anisotropy_ablation(rows, bench_cfg.subset_ns, bench_cfg.subset_seeds)

    failure = {
        "status": "skipped",
    }
    if canonical_failure_model is not None and canonical_failure_lat is not None:
        failure = run_failure_analysis(
            model_dir=canonical_failure_model,
            train_lat=canonical_failure_lat,
            cfg=failure_cfg,
            volume_power=sampler_cfg.volume_power,
            beta_zero=sampler_cfg.beta_zero,
            device=device,
            out_dir=run_dir,
            seed=int(bench_cfg.subset_seeds[0]),
        )

        discussion = (
            "The strong anisotropy required to bridge low-density voids induces stiffness in "
            "Riemannian kinetic updates, which amplifies kinetic spikes and causes unstable proposals. "
            "The Euclidean-momentum volume sampler remains navigable because geometry enters through "
            "the potential field rather than velocity amplification."
        )
        (run_dir / "failure_discussion_template.md").write_text(discussion + "\n", encoding="utf-8")

    _write_csv(run_dir / "raw_metrics.csv", rows)
    (run_dir / "summary_by_model_n.json").write_text(
        json.dumps(
            {
                "created_at": datetime.datetime.now().isoformat(),
                "subset_ns": bench_cfg.subset_ns,
                "subset_seeds": bench_cfg.subset_seeds,
                "models": models,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "stat_tests.json").write_text(json.dumps(stat_tests, indent=2), encoding="utf-8")
    (run_dir / "anisotropy_ablation.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    (run_dir / "failure_volume_riemannian.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")

    _write_handoff_prompt(run_dir / "prompts" / "handoff_prompt.txt")

    print(f"[low_data_benchmark] saved to: {run_dir}")
    print("- raw_metrics.csv")
    print("- summary_by_model_n.json")
    print("- stat_tests.json")
    print("- anisotropy_ablation.json")
    print("- failure_volume_riemannian.json")
    print("- prompts/handoff_prompt.txt")


if __name__ == "__main__":
    main()
