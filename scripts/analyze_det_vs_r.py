#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rhvae_geometry import GeometryRHVAE, GeometryRHVAEConfig
from src.utils.metric_helpers import load_metric_bundle


def _build_model(model_path: Path, device: torch.device) -> tuple[GeometryRHVAE, torch.Tensor]:
    metric_file = model_path / "rhvae_metric.pt"
    centroids, atoms, temperature, regularization, cfg_dict = load_metric_bundle(metric_file)
    cfg_kwargs = dict(cfg_dict) if cfg_dict is not None else {}
    cfg_kwargs.setdefault("input_dim", (centroids.shape[1],))
    cfg_kwargs.setdefault("latent_dim", centroids.shape[1])
    cfg_kwargs.setdefault("temperature", float(temperature))
    cfg_kwargs.setdefault("regularization", float(regularization))
    if not isinstance(cfg_kwargs["input_dim"], tuple):
        cfg_kwargs["input_dim"] = tuple(cfg_kwargs["input_dim"])

    allowed_fields = {
        name for name, field in GeometryRHVAEConfig.__dataclass_fields__.items() if field.init
    }
    filtered_kwargs = {k: v for k, v in cfg_kwargs.items() if k in allowed_fields}
    config = GeometryRHVAEConfig(**filtered_kwargs)
    model = GeometryRHVAE(config).to(device)
    model.set_centroids(centroids.to(device))
    model.set_atoms(torch.as_tensor(atoms, dtype=torch.float32).to(device))
    model.temperature.data = torch.tensor(float(temperature), device=device)
    model.lbd.data = torch.tensor(float(regularization), device=device)
    model._refresh_metric_hooks()
    model.eval()
    return model, centroids.to(device)


def _compute_radial_inv(
    model: GeometryRHVAE,
    z: torch.Tensor,
    radial_stretch: float | None = None,
) -> torch.Tensor:
    centroids = model.centroids_tens.to(z.device)
    precisions = None
    if model.attractor_metric == "mahalanobis" or model.attractor_use_det:
        precisions = model._get_attractor_precisions(None, z.device)

    beta = float(radial_stretch) if radial_stretch is not None else float(model.radial_stretch)

    if model.attractor_smoothness == "hard":
        diff = centroids.unsqueeze(0) - z.unsqueeze(1)
        if model.attractor_metric == "mahalanobis":
            tmp = torch.einsum("bkd,kde->bke", diff, precisions)
            attractor_dists_sq = (tmp * diff).sum(dim=-1)
        else:
            attractor_dists_sq = (diff ** 2).sum(dim=-1)
        _, nearest_idx = attractor_dists_sq.min(dim=1)
        targets = centroids[nearest_idx]
        direction = targets - z
        u_hat = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-8)
        radial = _radial_from_direction(u_hat, beta, model)
        return radial

    weights = model._compute_soft_attractor_weights(z, centroids, precisions)
    diff = centroids.unsqueeze(0) - z.unsqueeze(1)
    u_hat = diff / (torch.norm(diff, dim=-1, keepdim=True) + 1e-8)
    u_hat_flat = u_hat.reshape(-1, u_hat.shape[-1])
    radial = _radial_from_direction(u_hat_flat, beta, model)
    radial = radial.view(z.shape[0], centroids.shape[0], z.shape[1], z.shape[1])
    radial = torch.einsum("bk,bkij->bij", weights, radial)
    return radial


def _radial_from_direction(
    u_hat: torch.Tensor, beta: float, model: GeometryRHVAE
) -> torch.Tensor:
    uu_t = torch.einsum("bi,bj->bij", u_hat, u_hat)
    d = u_hat.shape[1]
    eye = torch.eye(d, device=u_hat.device, dtype=u_hat.dtype).unsqueeze(0)
    return beta * uu_t + model.lbd.to(u_hat.device) * eye

def _logdet_ginv(g_inv: torch.Tensor) -> torch.Tensor:
    _, logabsdet = torch.linalg.slogdet(g_inv)
    return logabsdet


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace det(G) along a descent path from a centroid.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--centroid_idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--step_size", type=float, default=0.02)
    parser.add_argument("--normalize_grad", action="store_true")
    parser.add_argument("--no_normalize_grad", action="store_true")
    parser.add_argument("--grid_res", type=int, default=80)
    parser.add_argument("--grid_bounds", type=float, default=6.0)
    parser.add_argument(
        "--direction_mode",
        type=str,
        choices=["grad", "random_line"],
        default="grad",
    )
    parser.add_argument("--direction_seed", type=int, default=None)
    parser.add_argument(
        "--det_target",
        type=str,
        choices=["g", "g_inv"],
        default="g",
    )
    parser.add_argument("--log_det", action="store_true")
    parser.add_argument("--alpha_override", action="store_true")
    parser.add_argument(
        "--alpha_tau",
        type=float,
        default=-1.0,
        help="If > 0, set r0 = T * sqrt(-ln(tau)) for the alpha override.",
    )
    parser.add_argument(
        "--alpha_steepness",
        type=float,
        default=None,
        help="Override transition steepness for the alpha override.",
    )
    parser.add_argument(
        "--override_radial_stretch",
        type=float,
        default=None,
        help="Override radial_stretch when computing void metric.",
    )
    parser.add_argument("--start_offset", type=float, default=0.0)
    parser.add_argument("--min_grad_norm", type=float, default=1e-6)
    parser.add_argument("--output_dir", type=str, default="outputs/det_vs_r")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    device = torch.device(args.device)
    model, centroids = _build_model(model_path, device)
    if args.centroid_idx < 0 or args.centroid_idx >= centroids.shape[0]:
        raise ValueError(f"centroid_idx must be in [0, {centroids.shape[0] - 1}]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    c0 = centroids[args.centroid_idx].detach()
    fallback = None
    line_dir = None
    if args.direction_mode == "random_line":
        gen = None
        if args.direction_seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(args.direction_seed)
        line_dir = torch.randn(c0.shape, device=device, generator=gen)
        if torch.norm(line_dir) < 1e-12:
            line_dir = torch.randn_like(c0)
        line_dir = line_dir / torch.norm(line_dir).clamp_min(1e-12)
    else:
        fallback_dir = centroids - c0.unsqueeze(0)
        fallback_norms = torch.norm(fallback_dir, dim=1)
        farthest_idx = torch.argmax(fallback_norms).item() if fallback_norms.numel() else 0
        fallback = fallback_dir[farthest_idx]
        if torch.norm(fallback) < 1e-12:
            fallback = torch.randn_like(c0)
        fallback = fallback / torch.norm(fallback).clamp_min(1e-12)

    z = c0.clone().detach()
    if args.start_offset > 0:
        if args.direction_mode == "random_line":
            z = z + args.start_offset * line_dir
        else:
            z = z + args.start_offset * fallback

    r_vals = []
    det_base_vals = []
    det_void_vals = []
    det_mix_vals = []
    alpha_vals = []

    z_path = []
    for _ in range(args.steps):
        z = z.detach().requires_grad_(True).unsqueeze(0)

        base = model._compute_base_inverse_metric(z)
        base = model._stabilize_metric(base)
        logdet_ginv_base = _logdet_ginv(base)
        logdet_base = -logdet_ginv_base if args.det_target == "g" else logdet_ginv_base
        grad = torch.autograd.grad(logdet_base.sum(), z)[0]

        z_det = z.detach()
        with torch.no_grad():
            diff_eucl = centroids.unsqueeze(0) - z_det.unsqueeze(1)
            min_dists_eucl = torch.sqrt(
                (diff_eucl ** 2).sum(dim=-1).min(dim=1).values + 1e-10
            )
            radial = _compute_radial_inv(
                model,
                z_det,
                radial_stretch=args.override_radial_stretch,
            )
            decay = model._compute_void_decay(min_dists_eucl).view(-1, 1, 1)
            radial = decay * radial
            logdet_ginv_void = _logdet_ginv(radial)
            if args.alpha_override:
                if args.alpha_tau > 0:
                    r0 = model.temperature.to(z_det.device) * torch.sqrt(
                        torch.tensor(-np.log(args.alpha_tau), device=z_det.device)
                    )
                else:
                    r0 = model.temperature.to(z_det.device) * model.void_threshold
                steepness = (
                    float(args.alpha_steepness)
                    if args.alpha_steepness is not None
                    else float(model.transition_steepness)
                )
                alpha = torch.sigmoid((min_dists_eucl - r0) * steepness)
            else:
                alpha = model._compute_alpha(min_dists_eucl)

            logdet_void = -logdet_ginv_void if args.det_target == "g" else logdet_ginv_void

            det_base = torch.exp(logdet_base)
            det_void = torch.exp(logdet_void)
            det_mix = (1 - alpha) * det_base + alpha * det_void

            r = torch.norm(z_det.squeeze(0) - c0).item()

        r_vals.append(r)
        if args.log_det:
            det_base_vals.append(logdet_base.item())
            det_void_vals.append(logdet_void.item())
            det_mix_vals.append(torch.log(det_mix.clamp_min(1e-12)).item())
        else:
            det_base_vals.append(det_base.item())
            det_void_vals.append(det_void.item())
            det_mix_vals.append(det_mix.item())
        alpha_vals.append(alpha.item())
        z_path.append(z_det.squeeze(0).detach().cpu().numpy())

        with torch.no_grad():
            if args.direction_mode == "random_line":
                z = (z_det.squeeze(0) + args.step_size * line_dir).detach()
            else:
                direction = grad.squeeze(0)
                normalize_grad = True
                if args.no_normalize_grad:
                    normalize_grad = False
                elif args.normalize_grad:
                    normalize_grad = True
                norm = torch.norm(direction)
                if normalize_grad:
                    if norm < args.min_grad_norm:
                        direction = fallback
                    else:
                        direction = direction / norm
                z = (z_det.squeeze(0) - args.step_size * direction).detach()

    data = np.column_stack([r_vals, det_base_vals, det_void_vals, det_mix_vals, alpha_vals])
    np.savetxt(
        output_dir / "det_vs_r.csv",
        data,
        delimiter=",",
        header="r,metric_base,metric_void,metric_mix,alpha",
        comments="",
    )

    z_path = np.asarray(z_path)
    z_path_2d = z_path[:, :2]

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])

    ax0 = fig.add_subplot(gs[0, :])
    grid = torch.linspace(-args.grid_bounds, args.grid_bounds, args.grid_res, device=device)
    xx, yy = torch.meshgrid(grid, grid, indexing="xy")
    grid_flat = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    if model.latent_dim > 2:
        padding = torch.zeros(grid_flat.shape[0], model.latent_dim - 2, device=device)
        grid_z = torch.cat([grid_flat, padding], dim=1)
    else:
        grid_z = grid_flat
    with torch.no_grad():
        g_inv = model.G_inv(grid_z)
        logdet_ginv = _logdet_ginv(g_inv)
        if args.det_target == "g":
            logdet = (-logdet_ginv).reshape(args.grid_res, args.grid_res).cpu().numpy()
        else:
            logdet = logdet_ginv.reshape(args.grid_res, args.grid_res).cpu().numpy()
    levels = 40
    cf = ax0.contourf(
        xx.cpu().numpy(),
        yy.cpu().numpy(),
        logdet,
        levels=levels,
        cmap="viridis",
        alpha=0.9,
    )
    label_target = "G" if args.det_target == "g" else "G_inv"
    fig.colorbar(cf, ax=ax0, label=f"log det {label_target}")
    centroids_np = centroids[:, :2].detach().cpu().numpy()
    ax0.scatter(centroids_np[:, 0], centroids_np[:, 1], s=15, c="white", alpha=0.5)
    ax0.plot(z_path_2d[:, 0], z_path_2d[:, 1], color="red", lw=1.5)
    ax0.scatter(z_path_2d[:, 0], z_path_2d[:, 1], s=12, color="red", alpha=0.8)
    ax0.set_title(f"Metric Landscape (log det {label_target}) with path + samples")
    ax0.set_xlabel("Latent Dim 1")
    ax0.set_ylabel("Latent Dim 2")
    ax0.set_aspect("equal", adjustable="box")

    ax1 = fig.add_subplot(gs[1, 0])
    metric_name = "log det" if args.log_det else "det"
    ax1.scatter(r_vals, det_base_vals, label=f"{metric_name} {label_target} (base)", s=10, alpha=0.7)
    ax1.scatter(r_vals, det_void_vals, label=f"{metric_name} {label_target} (void)", s=10, alpha=0.7)
    ax1.scatter(r_vals, det_mix_vals, label=f"{metric_name} mix", s=10, alpha=0.7)
    ax1.set_xlabel("r(z) from fixed centroid")
    ax1.set_ylabel(f"{metric_name} {label_target}")
    ax1.set_title(f"{metric_name} {label_target} vs r(z)")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.scatter(r_vals, alpha_vals, label="alpha(z)", s=10, alpha=0.7)
    ax2.set_xlabel("r(z) from fixed centroid")
    ax2.set_ylabel("alpha")
    ax2.set_title("Alpha vs r(z)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "det_vs_r_summary.png", dpi=150)

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
