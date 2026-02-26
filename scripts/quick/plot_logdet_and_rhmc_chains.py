#!/usr/bin/env python3
"""Plot log det(G) landscapes and RHMC chains for low-data benchmark models."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_low_data_benchmark import (  # noqa: E402
    _load_registry_required_files,
    _make_adapter,
    _resolve_model_dir,
)
from src.models.model_adapter import ModelAdapter  # noqa: E402
from src.utils.low_data_io import (  # noqa: E402
    load_split_tensor,
    load_subset_indices,
    subset_from_indices,
)


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _compute_bounds(z2: np.ndarray, quantile: float, padding: float) -> float:
    if z2.size == 0:
        return 4.0
    q = min(max(float(quantile), 0.75), 0.9995)
    lo = np.quantile(z2, 1.0 - q, axis=0)
    hi = np.quantile(z2, q, axis=0)
    bound = float(np.max(np.abs(np.concatenate([lo, hi], axis=0))))
    if not np.isfinite(bound):
        bound = 4.0
    return max(1.0, bound * float(padding))


def _build_grid(latent_dim: int, bound: float, resolution: int) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    axis = np.linspace(-bound, bound, int(resolution), dtype=np.float32)
    X, Y = np.meshgrid(axis, axis)
    pts2 = np.stack([X.ravel(), Y.ravel()], axis=1)
    pts = np.zeros((pts2.shape[0], int(latent_dim)), dtype=np.float32)
    pts[:, :2] = pts2
    return axis, X, torch.from_numpy(pts)


def _metric_ginv(adapter: ModelAdapter, z: torch.Tensor) -> torch.Tensor:
    # Prefer differentiable metric access for RHMC gradients.
    model = getattr(adapter, "model", None)
    if model is not None and hasattr(model, "G_inv"):
        return model.G_inv(z)

    if getattr(adapter, "model_id", "") == "ebm_conformal" and hasattr(adapter, "ebm"):
        energy = adapter.ebm(z)
        scalar = torch.exp(-float(getattr(adapter, "beta_conformal", 1.0)) * energy).reshape(-1, 1, 1)
        eye = torch.eye(int(getattr(adapter, "latent_dim", z.shape[1])), device=z.device, dtype=z.dtype).unsqueeze(0)
        return scalar * eye

    return adapter.metric_tensor_or_proxy(z)


def _logdet_g(adapter: ModelAdapter, z: torch.Tensor) -> torch.Tensor:
    g_inv = _metric_ginv(adapter, z)
    logdet_ginv = torch.linalg.slogdet(g_inv).logabsdet
    return -logdet_ginv


def _eval_logdet_grid(
    adapter: ModelAdapter,
    grid_latents: torch.Tensor,
    resolution: int,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    vals: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, grid_latents.shape[0], int(batch_size)):
            z = grid_latents[i : i + int(batch_size)].to(device)
            vals.append(_logdet_g(adapter, z).detach().cpu())
    flat = torch.cat(vals, dim=0).numpy()
    return flat.reshape(int(resolution), int(resolution))


def _potential_energy(adapter: ModelAdapter, z: torch.Tensor, volume_power: float) -> torch.Tensor:
    # U(z) = -volume_power * logdet(G_inv(z)) = -0.5*logdet(G_inv) for volume_power=0.5
    g_inv = _metric_ginv(adapter, z)
    logdet_ginv = torch.linalg.slogdet(g_inv).logabsdet
    return -float(volume_power) * logdet_ginv


def _potential_grad(adapter: ModelAdapter, z: torch.Tensor, volume_power: float) -> torch.Tensor:
    z_req = z.clone().detach().requires_grad_(True)
    U = _potential_energy(adapter, z_req, volume_power).sum()
    grad = torch.autograd.grad(U, z_req, create_graph=False)[0]
    return grad.detach()


def _run_chain(
    adapter: ModelAdapter,
    z_init: torch.Tensor,
    steps: int,
    n_lf: int,
    eps_lf: float,
    volume_power: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    z = z_init.clone().detach().reshape(1, -1)
    path = [z[0, :2].detach().cpu().numpy()]
    trace_logdet_g = []
    accepted = 0

    for _ in range(int(steps)):
        z0 = z.clone().detach()
        rho = torch.randn_like(z0)
        with torch.no_grad():
            H0 = _potential_energy(adapter, z0, volume_power) + 0.5 * torch.sum(rho * rho, dim=1)

        z_prop = z0
        rho_prop = rho
        valid = True
        for _ in range(int(n_lf)):
            grad = _potential_grad(adapter, z_prop, volume_power)
            if not torch.isfinite(grad).all():
                valid = False
                break
            rho_half = rho_prop - 0.5 * float(eps_lf) * grad
            z_prop = z_prop + float(eps_lf) * rho_half
            grad_new = _potential_grad(adapter, z_prop, volume_power)
            if not torch.isfinite(grad_new).all():
                valid = False
                break
            rho_prop = rho_half - 0.5 * float(eps_lf) * grad_new

        if valid:
            with torch.no_grad():
                H1 = _potential_energy(adapter, z_prop, volume_power) + 0.5 * torch.sum(rho_prop * rho_prop, dim=1)
                alpha = torch.exp(-(H1 - H0)).clamp(max=1.0)
                move = bool((torch.rand_like(alpha) < alpha).item())
            if move:
                z = z_prop.detach()
                accepted += 1
            else:
                z = z0
        else:
            z = z0

        with torch.no_grad():
            ld = _logdet_g(adapter, z).item()
        trace_logdet_g.append(float(ld))
        path.append(z[0, :2].detach().cpu().numpy())

    acc = float(accepted / max(1, int(steps)))
    return np.stack(path, axis=0), np.asarray(trace_logdet_g, dtype=np.float32), acc


def _median_nn_distance(z: torch.Tensor) -> float:
    if z.shape[0] <= 1:
        return 1.0
    with torch.no_grad():
        d = torch.cdist(z, z)
        eye = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
        d = d.masked_fill(eye, float("inf"))
        nn = d.amin(dim=1)
    med = float(torch.median(nn).item())
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return med


def _sample_starts(
    z_train: torch.Tensor,
    manifold_starts: int,
    near_starts: int,
    far_starts: int,
    near_sigma_factor: float,
    far_scale: float,
    latent_dim: int,
    device: torch.device,
) -> list[tuple[str, torch.Tensor]]:
    starts: list[tuple[str, torch.Tensor]] = []
    n = int(z_train.shape[0])
    if n <= 0:
        total = max(1, int(manifold_starts) + int(near_starts) + int(far_starts))
        for i in range(total):
            starts.append((f"fallback_{i+1}", torch.randn((1, int(latent_dim)), device=device)))
        return starts

    mean = z_train.mean(dim=0, keepdim=True)
    nn_med = _median_nn_distance(z_train)
    sigma = float(near_sigma_factor) * float(nn_med)
    radii = torch.linalg.norm(z_train - mean, dim=1)
    r_ref = float(torch.quantile(radii, 0.9).item()) if radii.numel() else 1.0
    if not np.isfinite(r_ref) or r_ref <= 0:
        r_ref = max(1.0, nn_med)

    if int(manifold_starts) > 0:
        idx = np.random.choice(n, size=min(int(manifold_starts), n), replace=False)
        for i, j in enumerate(idx.tolist()):
            starts.append((f"manifold_{i+1}", z_train[j : j + 1].to(device)))

    if int(near_starts) > 0:
        idx = np.random.choice(n, size=min(int(near_starts), n), replace=False)
        for i, j in enumerate(idx.tolist()):
            z0 = z_train[j : j + 1].to(device) + sigma * torch.randn((1, int(latent_dim)), device=device)
            starts.append((f"near_{i+1}", z0))

    if int(far_starts) > 0:
        dirs = torch.randn((int(far_starts), int(latent_dim)), device=device)
        dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
        z_far = mean.to(device) + float(far_scale) * float(r_ref) * dirs
        for i in range(int(far_starts)):
            starts.append((f"far_{i+1}", z_far[i : i + 1]))

    if not starts:
        starts.append(("manifold_1", z_train[0:1].to(device)))
    return starts


def _plot_model_overlay(
    out_path: Path,
    model_id: str,
    bound: float,
    logdet_grid: np.ndarray,
    z2_train: np.ndarray,
    chain_paths: list[np.ndarray],
    chain_labels: list[str],
    accept_rates: list[float],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(
        logdet_grid,
        origin="lower",
        extent=[-bound, bound, -bound, bound],
        cmap="magma",
        aspect="auto",
    )
    if z2_train.size > 0:
        ax.scatter(z2_train[:, 0], z2_train[:, 1], s=8, c="deepskyblue", alpha=0.3, edgecolors="none")
    zone_colors = {"manifold": "#4c78a8", "near": "#f58518", "far": "#e45756", "fallback": "#54a24b"}
    for idx, path in enumerate(chain_paths):
        label = chain_labels[idx] if idx < len(chain_labels) else f"chain_{idx+1}"
        zone = label.split("_")[0]
        color = zone_colors.get(zone, "#999999")
        ax.plot(path[:, 0], path[:, 1], lw=1.3, alpha=0.9, color=color, label=label)

    mean_acc = float(np.mean(accept_rates)) if accept_rates else float("nan")
    ax.set_title(f"{model_id}: log det(G) + chains on/near/far (acc={mean_acc:.3f})")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.colorbar(im, ax=ax, label="log det(G)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_trace(out_path: Path, model_id: str, traces: list[np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, tr in enumerate(traces):
        x = np.arange(1, tr.shape[0] + 1)
        ax.plot(x, tr, lw=1.2, alpha=0.85, label=f"chain {i+1}")
    ax.set_title(f"{model_id}: RHMC chain trace (log det(G))")
    ax.set_xlabel("step")
    ax.set_ylabel("log det(G)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_panel(
    out_path: Path,
    model_order: list[str],
    per_model: dict[str, dict[str, Any]],
) -> None:
    cols = len(model_order)
    fig, axes = plt.subplots(1, cols, figsize=(6.6 * cols, 5.2), squeeze=False)
    for c, model_id in enumerate(model_order):
        ax = axes[0, c]
        item = per_model[model_id]
        bound = float(item["bound"])
        grid = item["logdet_grid"]
        z2 = item["z2_train"]
        paths = item["paths"]
        acc = item["accept_rates"]
        im = ax.imshow(
            grid,
            origin="lower",
            extent=[-bound, bound, -bound, bound],
            cmap="magma",
            aspect="auto",
        )
        if z2.size > 0:
            ax.scatter(z2[:, 0], z2[:, 1], s=6, c="deepskyblue", alpha=0.28, edgecolors="none")
        for path in paths:
            ax.plot(path[:, 0], path[:, 1], lw=1.1, alpha=0.85, color="white")
        ax.set_title(f"{model_id}\nacc={np.mean(acc):.3f}")
        ax.set_xlabel("z1")
        if c == 0:
            ax.set_ylabel("z2")
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log det(G)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot log det(G) landscapes and RHMC chains.")
    p.add_argument("--processed_dir", type=str, default="data/processed/rotmnist/v1")
    p.add_argument("--model_root", type=str, default="outputs/low_data_models/rotmnist_v1")
    p.add_argument("--model_alias_root", type=str, default="outputs/reference_models")
    p.add_argument("--model_registry", type=str, default="configs/assets/model_registry.yaml")
    p.add_argument("--models", nargs="+", default=["rhvae_standard", "aniso", "ebm_conformal"])
    p.add_argument("--subset_n", type=int, default=50)
    p.add_argument("--subset_seed", type=int, default=13)
    p.add_argument("--grid_res", type=int, default=140)
    p.add_argument("--grid_quantile", type=float, default=0.99)
    p.add_argument("--grid_padding", type=float, default=1.35)
    p.add_argument("--manifold_starts", type=int, default=3)
    p.add_argument("--near_starts", type=int, default=3)
    p.add_argument("--far_starts", type=int, default=3)
    p.add_argument("--near_sigma_factor", type=float, default=1.2)
    p.add_argument("--far_scale", type=float, default=2.4)
    p.add_argument("--chain_steps", type=int, default=80)
    p.add_argument("--chain_n_lf", type=int, default=20)
    p.add_argument("--chain_eps_lf", type=float, default=0.02)
    p.add_argument("--volume_power", type=float, default=0.5)
    p.add_argument("--output_dir", type=str, default="results/metric_logdet_rhmc")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    processed_dir = (ROOT / args.processed_dir).resolve()
    model_root = (ROOT / args.model_root).resolve()
    alias_root = (ROOT / args.model_alias_root).resolve()
    registry_path = (ROOT / args.model_registry).resolve()
    required = _load_registry_required_files(registry_path)

    full_train_x, full_train_y = load_split_tensor(processed_dir, "train")
    subset_idx = load_subset_indices(processed_dir, int(args.subset_n), int(args.subset_seed))
    train_x, train_y = subset_from_indices(full_train_x, full_train_y, subset_idx)
    _ = train_y

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = (ROOT / args.output_dir).resolve() / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    model_order: list[str] = []
    per_model: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {
        "created_at": ts,
        "subset_n": int(args.subset_n),
        "subset_seed": int(args.subset_seed),
        "device": str(device),
        "chain_config": {
            "manifold_starts": int(args.manifold_starts),
            "near_starts": int(args.near_starts),
            "far_starts": int(args.far_starts),
            "near_sigma_factor": float(args.near_sigma_factor),
            "far_scale": float(args.far_scale),
            "chain_steps": int(args.chain_steps),
            "chain_n_lf": int(args.chain_n_lf),
            "chain_eps_lf": float(args.chain_eps_lf),
            "volume_power": float(args.volume_power),
        },
        "models": {},
    }

    for model_id in args.models:
        req = required.get(model_id, [])
        model_dir = _resolve_model_dir(
            model_id=model_id,
            n=int(args.subset_n),
            seed=int(args.subset_seed),
            model_root=model_root,
            alias_root=alias_root,
            required_files=req,
        )
        if model_dir is None:
            print(f"[warn] missing model assets for {model_id}; skipping")
            continue

        adapter = _make_adapter(model_id, model_dir, device)
        with torch.no_grad():
            z_train = adapter.encode(train_x.to(device)).detach().cpu()
        z2 = z_train[:, :2].numpy()
        bound = _compute_bounds(z2, quantile=float(args.grid_quantile), padding=float(args.grid_padding))
        axis, _, grid_latents = _build_grid(adapter.latent_dim, bound, int(args.grid_res))
        logdet_grid = _eval_logdet_grid(adapter, grid_latents, int(args.grid_res), device=device)

        starts_labeled = _sample_starts(
            z_train=z_train.to(device),
            manifold_starts=int(args.manifold_starts),
            near_starts=int(args.near_starts),
            far_starts=int(args.far_starts),
            near_sigma_factor=float(args.near_sigma_factor),
            far_scale=float(args.far_scale),
            latent_dim=int(adapter.latent_dim),
            device=device,
        )

        chain_paths: list[np.ndarray] = []
        chain_traces: list[np.ndarray] = []
        chain_labels: list[str] = []
        accept_rates: list[float] = []
        for label, z_start in starts_labeled:
            path, trace, acc = _run_chain(
                adapter=adapter,
                z_init=z_start,
                steps=int(args.chain_steps),
                n_lf=int(args.chain_n_lf),
                eps_lf=float(args.chain_eps_lf),
                volume_power=float(args.volume_power),
            )
            chain_paths.append(path)
            chain_traces.append(trace)
            chain_labels.append(str(label))
            accept_rates.append(acc)

        model_dir_out = run_dir / model_id
        _plot_model_overlay(
            out_path=model_dir_out / f"logdetG_rhmc_overlay_{model_id}.png",
            model_id=model_id,
            bound=bound,
            logdet_grid=logdet_grid,
            z2_train=z2,
            chain_paths=chain_paths,
            chain_labels=chain_labels,
            accept_rates=accept_rates,
        )
        _plot_trace(
            out_path=model_dir_out / f"rhmc_logdet_trace_{model_id}.png",
            model_id=model_id,
            traces=chain_traces,
        )

        per_model[model_id] = {
            "bound": bound,
            "axis": axis,
            "z2_train": z2,
            "logdet_grid": logdet_grid,
            "paths": chain_paths,
            "labels": chain_labels,
            "traces": chain_traces,
            "accept_rates": accept_rates,
        }
        model_order.append(model_id)

        summary["models"][model_id] = {
            "model_dir": str(model_dir),
            "mean_acceptance": _as_float(np.mean(accept_rates) if accept_rates else float("nan")),
            "std_acceptance": _as_float(np.std(accept_rates) if accept_rates else float("nan")),
            "mean_trace_logdetG": _as_float(np.mean(np.concatenate(chain_traces)) if chain_traces else float("nan")),
            "std_trace_logdetG": _as_float(np.std(np.concatenate(chain_traces)) if chain_traces else float("nan")),
            "bound": float(bound),
            "grid_res": int(args.grid_res),
        }

        print(
            "[ok] "
            f"{model_id}: acc_mean={summary['models'][model_id]['mean_acceptance']:.3f} "
            f"logdetG_mean={summary['models'][model_id]['mean_trace_logdetG']:.3f}"
        )

    if model_order:
        _plot_combined_panel(
            out_path=run_dir / "combined_logdetG_rhmc_overlay.png",
            model_order=model_order,
            per_model=per_model,
        )

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()
