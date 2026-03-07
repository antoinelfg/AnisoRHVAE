#!/usr/bin/env python3
"""Metric dashboard: force fields + geometry diagnostics for RHVAE metrics.

This script builds a single visual dashboard and a scorecard to assess whether
the learned metric exhibits the intended behavior (manifold attraction,
stable void behavior, anisotropy structure, SPD validity).
"""

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
from mpl_toolkits.mplot3d import Axes3D  # pyright: ignore[reportMissingImports,reportUnusedImport]

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from src.models.rhvae_geometry import GeometryRHVAE, GeometryRHVAEConfig
from src.utils.metric_helpers import load_metric_bundle
from src.utils.rescue_metrics import (
    compute_border_tunnel_metrics,
    compute_plateau_alignment_metrics,
    compute_tangent_alignment_metric,
    model_r0,
)


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


def parse_dims(dims: str, centroids: torch.Tensor) -> tuple[int, int]:
    latent_dim = int(centroids.shape[1])
    if latent_dim < 2:
        raise ValueError("Need latent_dim >= 2 for dashboard plots.")

    if dims.lower() == "auto":
        var = torch.var(centroids, dim=0)
        top2 = torch.topk(var, k=2).indices.tolist()
        d0, d1 = int(top2[0]), int(top2[1])
        if d0 == d1:
            d1 = (d0 + 1) % latent_dim
        return d0, d1

    parts = [p.strip() for p in dims.split(",")]
    if len(parts) != 2:
        raise ValueError("--dims must be 'auto' or 'i,j'.")
    d0, d1 = int(parts[0]), int(parts[1])
    if d0 < 0 or d1 < 0 or d0 >= latent_dim or d1 >= latent_dim or d0 == d1:
        raise ValueError(f"Invalid dims for latent_dim={latent_dim}: ({d0}, {d1})")
    return d0, d1


def build_grid(
    centroids: torch.Tensor,
    dims: tuple[int, int],
    grid_n: int,
    bounds_scale: float,
    min_bounds: float,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, np.ndarray]:
    d0, d1 = dims
    c = centroids.detach().cpu().numpy()
    c2 = c[:, [d0, d1]]
    c2_center = c2.mean(axis=0)
    radii = np.linalg.norm(c2 - c2_center[None, :], axis=1)
    bound = max(float(np.percentile(radii, 95) * bounds_scale), float(min_bounds))

    x = np.linspace(c2_center[0] - bound, c2_center[0] + bound, int(grid_n))
    y = np.linspace(c2_center[1] - bound, c2_center[1] + bound, int(grid_n))
    xx, yy = np.meshgrid(x, y, indexing="xy")
    flat = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)

    latent_dim = int(centroids.shape[1])
    center_full = centroids.mean(dim=0, keepdim=True).repeat(flat.shape[0], 1)
    center_full[:, d0] = torch.from_numpy(flat[:, 0]).to(center_full.device)
    center_full[:, d1] = torch.from_numpy(flat[:, 1]).to(center_full.device)
    return xx, yy, center_full, c2


def _chunked(iter_size: int, chunk: int):
    for i in range(0, iter_size, chunk):
        yield i, min(iter_size, i + chunk)


def evaluate_grid_fields(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    points_full: torch.Tensor,
    dims: tuple[int, int],
    volume_power: float,
    batch_size: int,
) -> dict[str, np.ndarray]:
    d0, d1 = dims
    idx2 = torch.tensor([d0, d1], device=points_full.device)
    n = points_full.shape[0]

    logdet_inv_all = np.empty(n, dtype=np.float64)
    cond2d_all = np.empty(n, dtype=np.float64)
    mob2d_all = np.empty(n, dtype=np.float64)
    pdir_x_all = np.empty(n, dtype=np.float64)
    pdir_y_all = np.empty(n, dtype=np.float64)
    lmax2d_all = np.empty(n, dtype=np.float64)
    min_dist_all = np.empty(n, dtype=np.float64)
    nearest_x_all = np.empty(n, dtype=np.float64)
    nearest_y_all = np.empty(n, dtype=np.float64)

    with torch.no_grad():
        for i0, i1 in _chunked(n, int(batch_size)):
            z = points_full[i0:i1]
            g_inv = model.G_inv(z)
            logdet_inv = torch.linalg.slogdet(g_inv).logabsdet
            g2 = g_inv.index_select(1, idx2).index_select(2, idx2)
            evals, evecs = torch.linalg.eigh(g2)
            evals = torch.clamp(evals, min=1e-10)
            cond2d = evals[:, 1] / evals[:, 0]
            mob2d = torch.sqrt(torch.clamp(evals.sum(dim=1), min=1e-12))
            pdir = evecs[:, :, 1]
            lmax = evals[:, 1]

            d = torch.cdist(z, centroids)
            near_idx = torch.argmin(d, dim=1)
            near = centroids[near_idx]
            min_dist = d[torch.arange(z.shape[0], device=z.device), near_idx]

            logdet_inv_all[i0:i1] = logdet_inv.cpu().numpy()
            cond2d_all[i0:i1] = cond2d.cpu().numpy()
            mob2d_all[i0:i1] = mob2d.cpu().numpy()
            pdir_x_all[i0:i1] = pdir[:, 0].cpu().numpy()
            pdir_y_all[i0:i1] = pdir[:, 1].cpu().numpy()
            lmax2d_all[i0:i1] = lmax.cpu().numpy()
            min_dist_all[i0:i1] = min_dist.cpu().numpy()
            nearest_x_all[i0:i1] = near[:, d0].cpu().numpy()
            nearest_y_all[i0:i1] = near[:, d1].cpu().numpy()

    eff = float(volume_power) + 0.5
    potential = -eff * logdet_inv_all

    return {
        "logdet_inv": logdet_inv_all,
        "potential": potential,
        "cond2d": cond2d_all,
        "mobility2d": mob2d_all,
        "principal_x": pdir_x_all,
        "principal_y": pdir_y_all,
        "lambda_max2d": lmax2d_all,
        "min_dist": min_dist_all,
        "nearest_x": nearest_x_all,
        "nearest_y": nearest_y_all,
    }


def evaluate_volume_force(
    model: GeometryRHVAE,
    points_full: torch.Tensor,
    dims: tuple[int, int],
    volume_power: float,
    batch_size: int,
) -> np.ndarray:
    d0, d1 = dims
    n = points_full.shape[0]
    force = np.empty((n, 2), dtype=np.float64)
    eff = float(volume_power) + 0.5
    for i0, i1 in _chunked(n, int(batch_size)):
        z = points_full[i0:i1].clone().detach().requires_grad_(True)
        g_inv = model.G_inv(z)
        logdet_inv = torch.linalg.slogdet(g_inv).logabsdet
        grad = torch.autograd.grad(logdet_inv.sum(), z, create_graph=False)[0]
        f = eff * grad
        force[i0:i1, 0] = f[:, d0].detach().cpu().numpy()
        force[i0:i1, 1] = f[:, d1].detach().cpu().numpy()
    return force


def evaluate_rescue_step_quiver(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    points_full: torch.Tensor,
    dims: tuple[int, int],
) -> np.ndarray:
    d0, d1 = dims
    out = np.empty((points_full.shape[0], 2), dtype=np.float64)
    with torch.no_grad():
        g_inv = model.G_inv(points_full)
        d = torch.cdist(points_full, centroids)
        near_idx = torch.argmin(d, dim=1)
        near = centroids[near_idx]
        direction = near - points_full
        direction = direction / torch.clamp(torch.linalg.norm(direction, dim=1, keepdim=True), min=1e-8)
        step = torch.einsum("bij,bj->bi", g_inv, direction)
        out[:, 0] = step[:, d0].cpu().numpy()
        out[:, 1] = step[:, d1].cpu().numpy()
    return out


def sample_diag_points(
    centroids: torch.Tensor,
    dims: tuple[int, int],
    n_points: int,
    bounds_xy: tuple[float, float, float, float],
    seed: int,
) -> torch.Tensor:
    d0, d1 = dims
    x0, x1, y0, y1 = bounds_xy
    device = centroids.device
    latent_dim = int(centroids.shape[1])
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    r = torch.rand(int(n_points), 2, generator=gen, device=device)
    xy = torch.zeros(int(n_points), 2, device=device)
    xy[:, 0] = float(x0) + (float(x1) - float(x0)) * r[:, 0]
    xy[:, 1] = float(y0) + (float(y1) - float(y0)) * r[:, 1]

    z = centroids.mean(dim=0, keepdim=True).repeat(int(n_points), 1)
    z[:, d0] = xy[:, 0]
    z[:, d1] = xy[:, 1]
    if latent_dim > 2:
        z = z + 0.01 * torch.randn_like(z, generator=gen)
    return z


def _region_masks(min_dist: np.ndarray, r0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(min_dist, dtype=float)
    manifold = d <= (0.9 * float(r0))
    transition = (d > 0.9 * float(r0)) & (d <= 1.8 * float(r0))
    far_void = d >= (2.6 * float(r0))
    return manifold, transition, far_void


def _safe_median(x: np.ndarray, mask: np.ndarray) -> float:
    v = np.asarray(x, dtype=float)[np.asarray(mask, dtype=bool)]
    if v.size == 0:
        return float("nan")
    return float(np.median(v))


def _safe_mean(x: np.ndarray, mask: np.ndarray) -> float:
    v = np.asarray(x, dtype=float)[np.asarray(mask, dtype=bool)]
    if v.size == 0:
        return float("nan")
    return float(np.mean(v))


def compute_diagnostics(
    model: GeometryRHVAE,
    centroids: torch.Tensor,
    grid_points_full: torch.Tensor,
    dims: tuple[int, int],
    fields: dict[str, np.ndarray],
    q_points_full: torch.Tensor,
    q_force_xy: np.ndarray,
    q_step_xy: np.ndarray,
    r0: float,
    spd_points: int,
    seed: int,
) -> dict[str, Any]:
    min_dist = fields["min_dist"]
    logdet_inv = fields["logdet_inv"]
    cond2d = fields["cond2d"]
    mobility = fields["mobility2d"]

    manifold_mask, transition_mask, far_void_mask = _region_masks(min_dist, r0)

    med_logdet_m = _safe_median(logdet_inv, manifold_mask)
    med_logdet_t = _safe_median(logdet_inv, transition_mask)
    med_logdet_v = _safe_median(logdet_inv, far_void_mask)
    logdet_contrast = float(med_logdet_m - med_logdet_v) if np.isfinite(med_logdet_m) and np.isfinite(med_logdet_v) else float("nan")

    cond_p95 = float(np.nanpercentile(cond2d, 95))
    cond_void_p95 = float(np.nanpercentile(cond2d[far_void_mask], 95)) if np.any(far_void_mask) else float("nan")
    cond_manifold_med = _safe_median(cond2d, manifold_mask)
    cond_void_med = _safe_median(cond2d, far_void_mask)

    mobility_manifold = _safe_mean(mobility, manifold_mask)
    mobility_void = _safe_mean(mobility, far_void_mask)
    mobility_ratio = float(mobility_manifold / max(mobility_void, 1e-12)) if np.isfinite(mobility_manifold) and np.isfinite(mobility_void) else float("nan")

    d0, d1 = dims
    with torch.no_grad():
        q_d = torch.cdist(q_points_full, centroids)
        q_near_idx = torch.argmin(q_d, dim=1)
        q_near = centroids[q_near_idx]
        dir_xy = torch.stack(
            [q_near[:, d0] - q_points_full[:, d0], q_near[:, d1] - q_points_full[:, d1]],
            dim=1,
        ).cpu().numpy()
        q_min_dist = q_d[torch.arange(q_points_full.shape[0], device=q_points_full.device), q_near_idx].cpu().numpy()

    f = np.asarray(q_force_xy, dtype=float)
    s = np.asarray(q_step_xy, dtype=float)
    dxy = np.asarray(dir_xy, dtype=float)
    f_norm = np.linalg.norm(f, axis=1)
    s_norm = np.linalg.norm(s, axis=1)
    d_norm = np.linalg.norm(dxy, axis=1)
    cos_force = np.sum(f * dxy, axis=1) / np.clip(f_norm * d_norm, 1e-12, None)
    cos_step = np.sum(s * dxy, axis=1) / np.clip(s_norm * d_norm, 1e-12, None)
    outside_q = q_min_dist > float(r0)
    inside_q = ~outside_q

    force_align_outside = float(np.mean(cos_force[outside_q])) if np.any(outside_q) else float("nan")
    force_opp_rate_outside = float(np.mean(cos_force[outside_q] < 0.0)) if np.any(outside_q) else float("nan")
    step_align_outside = float(np.mean(cos_step[outside_q])) if np.any(outside_q) else float("nan")
    thr_plateau = 0.1 * float(np.median(f_norm[inside_q])) if np.any(inside_q) else 1e-3
    plateau_fraction_outside = float(np.mean(f_norm[outside_q] < thr_plateau)) if np.any(outside_q) else float("nan")

    # Extended diagnostics from existing utility metrics.
    d0, d1 = dims
    diag_points = sample_diag_points(
        centroids=centroids,
        dims=dims,
        n_points=1800,
        bounds_xy=(
            float(grid_points_full[:, d0].min().item()),
            float(grid_points_full[:, d0].max().item()),
            float(grid_points_full[:, d1].min().item()),
            float(grid_points_full[:, d1].max().item()),
        ),
        seed=int(seed + 73),
    )
    plateau_align = compute_plateau_alignment_metrics(
        model=model,
        points=diag_points,
        centroids=centroids,
        r0=float(r0),
    )
    border = compute_border_tunnel_metrics(
        model=model,
        points=diag_points,
        centroids=centroids,
        r0=float(r0),
    )
    tangent = compute_tangent_alignment_metric(
        model=model,
        centroids=centroids,
        r0=float(r0),
        n_samples=1024,
        k_neighbors=8,
        band_min=0.6,
        band_max=1.2,
        seed=int(seed + 97),
    )

    # SPD check on random points.
    x0 = float(grid_points_full[:, d0].min().item())
    x1 = float(grid_points_full[:, d0].max().item())
    y0 = float(grid_points_full[:, d1].min().item())
    y1 = float(grid_points_full[:, d1].max().item())
    spd_z = sample_diag_points(
        centroids=centroids,
        dims=dims,
        n_points=int(spd_points),
        bounds_xy=(x0, x1, y0, y1),
        seed=int(seed + 123),
    )
    with torch.no_grad():
        evals = torch.linalg.eigvalsh(model.G_inv(spd_z))
        min_eval = evals.min(dim=1).values.cpu().numpy()
    spd_violation_rate = float(np.mean(min_eval <= 0.0))
    min_eval_p01 = float(np.percentile(min_eval, 1))

    metrics = {
        "spd_violation_rate": spd_violation_rate,
        "min_eigenvalue_p01": min_eval_p01,
        "logdet_manifold_median": med_logdet_m,
        "logdet_transition_median": med_logdet_t,
        "logdet_far_void_median": med_logdet_v,
        "logdet_contrast_manifold_minus_void": logdet_contrast,
        "logdet_monotonic_manifold_gt_transition_gt_void": bool(
            np.isfinite(med_logdet_m)
            and np.isfinite(med_logdet_t)
            and np.isfinite(med_logdet_v)
            and (med_logdet_m > med_logdet_t > med_logdet_v)
        ),
        "cond2d_p95": cond_p95,
        "cond2d_void_p95": cond_void_p95,
        "cond2d_manifold_median": cond_manifold_med,
        "cond2d_void_median": cond_void_med,
        "mobility2d_manifold_mean": mobility_manifold,
        "mobility2d_void_mean": mobility_void,
        "mobility2d_ratio_manifold_over_void": mobility_ratio,
        "force_alignment_outside_mean": force_align_outside,
        "force_opposition_rate_outside": force_opp_rate_outside,
        "step_alignment_outside_mean": step_align_outside,
        "plateau_fraction_outside_quiver": plateau_fraction_outside,
        "plateau_fraction_outside_rescue_metric": float(plateau_align["plateau_fraction"]),
        "rescue_alignment_mean": float(plateau_align["alignment_mean"]),
        "border_overshoot_index": float(border["border_overshoot_index"]),
        "tunnel_neff_transition_median": float(border["tunnel_neff_transition_median"]),
        "tunnel_neff_far_void_median": float(border["tunnel_neff_far_void_median"]),
        "tangent_alignment_mean": float(tangent["tangent_alignment_mean"]),
        "r0": float(r0),
        "n_grid_points": int(grid_points_full.shape[0]),
        "n_quiver_points": int(q_points_full.shape[0]),
    }
    return metrics


def build_scorecard(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []

    def add(name: str, key: str, objective: str, good_if: str, fn) -> None:
        value = metrics.get(key, float("nan"))
        rows.append(
            {
                "name": name,
                "key": key,
                "objective": objective,
                "good_if": good_if,
                "value": value,
                "pass": bool(fn(value)) if np.isfinite(value) or isinstance(value, bool) else False,
            }
        )

    add(
        "SPD validity",
        "spd_violation_rate",
        "Metric must stay SPD in sampled space",
        "rate <= 1e-4",
        lambda v: float(v) <= 1e-4,
    )
    add(
        "Volume contrast",
        "logdet_contrast_manifold_minus_void",
        "Manifold should have higher log det(G_inv) than far void",
        "contrast > 0",
        lambda v: float(v) > 0.0,
    )
    add(
        "Monotonic shells",
        "logdet_monotonic_manifold_gt_transition_gt_void",
        "Determinant should decrease from manifold to void",
        "True",
        lambda v: bool(v),
    )
    add(
        "Void force alignment",
        "force_alignment_outside_mean",
        "Volume force should point toward manifold on average outside",
        "mean cosine > 0",
        lambda v: float(v) > 0.0,
    )
    add(
        "Void opposition rate",
        "force_opposition_rate_outside",
        "Too many opposite arrows indicate unstable rescue dynamics",
        "rate < 0.35",
        lambda v: float(v) < 0.35,
    )
    add(
        "Plateau outside",
        "plateau_fraction_outside_rescue_metric",
        "Avoid flat gradients in void",
        "fraction < 0.35",
        lambda v: float(v) < 0.35,
    )
    add(
        "Border overshoot",
        "border_overshoot_index",
        "Transition shell should not overshoot manifold and void medians",
        "index <= 0",
        lambda v: float(v) <= 0.0,
    )
    add(
        "Tangent coherence",
        "tangent_alignment_mean",
        "Principal metric direction should match local tangent near manifold",
        "alignment > 0.55",
        lambda v: float(v) > 0.55,
    )
    return rows


def _to_float(v: Any) -> float:
    if isinstance(v, bool):
        return float(v)
    try:
        return float(v)
    except Exception:
        return float("nan")


def save_scorecard(out_dir: Path, metrics: dict[str, Any], rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    payload = {
        "metadata": metadata,
        "metrics": metrics,
        "scorecard": rows,
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Metric Dashboard Scorecard")
    lines.append("")
    lines.append("## Diagnostic Definitions")
    lines.append("")
    lines.append("- `SPD validity`: fraction of sampled points where `G_inv` is not positive definite.")
    lines.append("- `Volume contrast`: median(log det G_inv) on manifold minus far-void median.")
    lines.append("- `Monotonic shells`: manifold > transition > far-void for log det(G_inv).")
    lines.append("- `Void force alignment`: cosine between volume force and vector to nearest centroid outside manifold.")
    lines.append("- `Void opposition rate`: fraction of outside points where force points away from nearest centroid.")
    lines.append("- `Plateau outside`: outside fraction with near-flat volume force norm.")
    lines.append("- `Border overshoot`: transition-shell determinant overshoot indicator (should stay <= 0).")
    lines.append("- `Tangent coherence`: alignment between principal metric direction and local manifold tangent.")
    lines.append("")
    lines.append("## Scorecard")
    lines.append("")
    lines.append("| Check | Value | Objective | Good If | Pass |")
    lines.append("|---|---:|---|---|:---:|")
    for row in rows:
        value = row["value"]
        if isinstance(value, bool):
            value_str = "true" if value else "false"
        elif np.isfinite(_to_float(value)):
            value_str = f"{_to_float(value):.4g}"
        else:
            value_str = "nan"
        lines.append(
            f"| {row['name']} | {value_str} | {row['objective']} | {row['good_if']} | {'PASS' if row['pass'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    for k, v in metadata.items():
        lines.append(f"- `{k}`: `{v}`")
    (out_dir / "diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def _sample_for_plot(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def _norm01(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    lo = np.nanpercentile(arr, 5)
    hi = np.nanpercentile(arr, 95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _binned_profile(x: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bins = np.linspace(np.nanmin(x), np.nanmax(x), int(n_bins) + 1)
    mids = 0.5 * (bins[:-1] + bins[1:])
    med = np.full(int(n_bins), np.nan, dtype=float)
    for i in range(int(n_bins)):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if np.any(mask):
            med[i] = np.nanmedian(y[mask])
    return mids, med


def plot_dashboard(
    out_path: Path,
    xx: np.ndarray,
    yy: np.ndarray,
    c2: np.ndarray,
    fields: dict[str, np.ndarray],
    qx: np.ndarray,
    qy: np.ndarray,
    q_force: np.ndarray,
    q_step: np.ndarray,
    q_pdir: np.ndarray,
    q_lmax: np.ndarray,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    dims: tuple[int, int],
    volume_power: float,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    n_grid = xx.shape[0]

    u = fields["potential"].reshape(n_grid, n_grid)
    ld = fields["logdet_inv"].reshape(n_grid, n_grid)
    cond = fields["cond2d"].reshape(n_grid, n_grid)
    mob = fields["mobility2d"].reshape(n_grid, n_grid)
    min_dist = fields["min_dist"]

    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.26, wspace=0.24)

    # Panel A: potential + volume force quiver
    ax = fig.add_subplot(gs[0, 0])
    hm = ax.contourf(xx, yy, u, levels=40, cmap="viridis")
    fig.colorbar(hm, ax=ax, fraction=0.045, pad=0.02, label="U(z)")
    force_norm = np.linalg.norm(q_force, axis=1, keepdims=True)
    f_dir = q_force / np.clip(force_norm, 1e-12, None)
    ax.quiver(
        qx,
        qy,
        f_dir[:, 0],
        f_dir[:, 1],
        color="white",
        alpha=0.75,
        width=0.0030,
        scale=28,
    )
    ax.scatter(c2[:, 0], c2[:, 1], c="black", s=14, alpha=0.75)
    ax.set_title("Potential Landscape + Volume Force Direction")
    ax.set_xlabel(f"z[{dims[0]}]")
    ax.set_ylabel(f"z[{dims[1]}]")
    ax.set_aspect("equal")

    # Panel B: logdet map
    ax = fig.add_subplot(gs[0, 1])
    hm = ax.contourf(xx, yy, ld, levels=40, cmap="magma")
    cs = ax.contour(xx, yy, ld, levels=12, colors="white", linewidths=0.4, alpha=0.35)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    fig.colorbar(hm, ax=ax, fraction=0.045, pad=0.02, label="log det(G_inv)")
    ax.scatter(c2[:, 0], c2[:, 1], c="#8de8ff", s=15, edgecolors="black", linewidths=0.2)
    ax.set_title("Volume Term Structure")
    ax.set_xlabel(f"z[{dims[0]}]")
    ax.set_ylabel(f"z[{dims[1]}]")
    ax.set_aspect("equal")

    # Panel C: 3D surface inspired by analyze_metric_full.py
    ax3d = fig.add_subplot(gs[0, 2], projection="3d")
    surf = ax3d.plot_surface(xx, yy, u, cmap="cividis", linewidth=0, antialiased=True, alpha=0.94)
    z_min = float(np.min(u))
    z_max = float(np.max(u))
    z_off = z_min - 0.20 * (z_max - z_min + 1e-9)
    ax3d.contour(xx, yy, u, zdir="z", offset=z_off, levels=12, cmap="cividis")
    c_u = []
    for i in range(c2.shape[0]):
        # nearest lookup on grid for display-only scatter elevation
        ix = int(np.argmin(np.abs(xx[0] - c2[i, 0])))
        iy = int(np.argmin(np.abs(yy[:, 0] - c2[i, 1])))
        c_u.append(u[iy, ix])
    c_u = np.asarray(c_u, dtype=float)
    ax3d.scatter(c2[:, 0], c2[:, 1], c_u, c="black", s=12, depthshade=True)
    ax3d.set_title("3D Potential Surface")
    ax3d.set_xlabel(f"z[{dims[0]}]")
    ax3d.set_ylabel(f"z[{dims[1]}]")
    ax3d.set_zlabel("U(z)")
    ax3d.view_init(elev=28, azim=-63)
    fig.colorbar(surf, ax=ax3d, fraction=0.045, pad=0.10, label="U(z)")

    # Panel D: anisotropy + principal directions
    ax = fig.add_subplot(gs[1, 0])
    cond_log = np.log10(np.clip(cond, 1.0, None))
    hm = ax.contourf(xx, yy, cond_log, levels=38, cmap="plasma")
    fig.colorbar(hm, ax=ax, fraction=0.045, pad=0.02, label="log10 cond(G_inv[2D])")
    p_scale = np.sqrt(np.clip(q_lmax, 1e-12, None))
    p_norm = np.linalg.norm(q_pdir, axis=1, keepdims=True)
    p_dir = q_pdir / np.clip(p_norm, 1e-12, None)
    ax.quiver(
        qx,
        qy,
        p_dir[:, 0] * p_scale,
        p_dir[:, 1] * p_scale,
        color="white",
        width=0.0028,
        alpha=0.72,
        scale=40,
    )
    ax.scatter(c2[:, 0], c2[:, 1], c="white", s=10, alpha=0.85)
    ax.set_title("Anisotropy + Principal Mobility Directions")
    ax.set_xlabel(f"z[{dims[0]}]")
    ax.set_ylabel(f"z[{dims[1]}]")
    ax.set_aspect("equal")

    # Panel E: mobility + rescue-step response
    ax = fig.add_subplot(gs[1, 1])
    hm = ax.contourf(xx, yy, mob, levels=40, cmap="YlGnBu")
    fig.colorbar(hm, ax=ax, fraction=0.045, pad=0.02, label="sqrt(trace(G_inv[2D]))")
    s_norm = np.linalg.norm(q_step, axis=1, keepdims=True)
    s_dir = q_step / np.clip(s_norm, 1e-12, None)
    ax.quiver(
        qx,
        qy,
        s_dir[:, 0],
        s_dir[:, 1],
        color="black",
        width=0.0030,
        alpha=0.65,
        scale=32,
    )
    ax.scatter(c2[:, 0], c2[:, 1], c="#f2f2f2", edgecolors="black", linewidths=0.3, s=13)
    ax.set_title("Mobility + Metric-Filtered Rescue Step")
    ax.set_xlabel(f"z[{dims[0]}]")
    ax.set_ylabel(f"z[{dims[1]}]")
    ax.set_aspect("equal")

    # Panel F: force and step alignment histograms outside manifold
    ax = fig.add_subplot(gs[1, 2])
    qd = np.sqrt((qx - np.mean(c2[:, 0])) ** 2 + (qy - np.mean(c2[:, 1])) ** 2)
    outside = qd > float(metrics["r0"])
    f_dir = q_force / np.clip(np.linalg.norm(q_force, axis=1, keepdims=True), 1e-12, None)
    s_dir = q_step / np.clip(np.linalg.norm(q_step, axis=1, keepdims=True), 1e-12, None)
    # approximate projected nearest-centroid vector with radial proxy for display histogram
    radial = np.stack([np.mean(c2[:, 0]) - qx, np.mean(c2[:, 1]) - qy], axis=1)
    radial = radial / np.clip(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12, None)
    c_force = np.sum(f_dir * radial, axis=1)
    c_step = np.sum(s_dir * radial, axis=1)
    bins = np.linspace(-1.0, 1.0, 34)
    ax.hist(c_force[outside], bins=bins, alpha=0.55, color="#e15759", label="volume force")
    ax.hist(c_step[outside], bins=bins, alpha=0.55, color="#4e79a7", label="metric step")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax.set_title("Outside Alignment (toward manifold proxy)")
    ax.set_xlabel("cosine")
    ax.set_ylabel("count")
    ax.legend(frameon=True, fontsize=10)

    # Panel G: distance vs logdet scatter
    ax = fig.add_subplot(gs[2, 0])
    idx = _sample_for_plot(min_dist.shape[0], max_n=9000, seed=19)
    x = min_dist[idx]
    y = fields["logdet_inv"][idx]
    ax.scatter(x, y, c=np.clip(fields["cond2d"][idx], 0, np.nanpercentile(fields["cond2d"], 98)), s=8, alpha=0.33, cmap="inferno")
    ax.axvline(float(metrics["r0"]), color="black", linestyle="--", linewidth=1.1, label="r0")
    ax.axvline(2.6 * float(metrics["r0"]), color="black", linestyle=":", linewidth=1.1, label="far-void")
    ax.set_title("Distance to Manifold vs log det(G_inv)")
    ax.set_xlabel("min distance to centroids")
    ax.set_ylabel("log det(G_inv)")
    ax.legend(frameon=True, fontsize=9)

    # Panel H: radial normalized profiles
    ax = fig.add_subplot(gs[2, 1])
    mids, m_logdet = _binned_profile(min_dist, fields["logdet_inv"], n_bins=24)
    _, m_cond = _binned_profile(min_dist, np.log10(np.clip(fields["cond2d"], 1.0, None)), n_bins=24)
    _, m_mob = _binned_profile(min_dist, fields["mobility2d"], n_bins=24)
    ax.plot(mids, _norm01(m_logdet), color="#e15759", linewidth=2.2, label="logdet_inv (norm)")
    ax.plot(mids, _norm01(m_mob), color="#76b7b2", linewidth=2.2, label="mobility (norm)")
    ax.plot(mids, _norm01(m_cond), color="#f28e2b", linewidth=2.2, label="log10 cond (norm)")
    ax.axvline(float(metrics["r0"]), color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Radial Profiles (normalized)")
    ax.set_xlabel("min distance to centroids")
    ax.set_ylabel("normalized median")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=True, fontsize=9)

    # Panel I: scorecard text
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    n_pass = int(sum(1 for r in rows if r["pass"]))
    n_tot = int(len(rows))
    lines = [
        f"Metric Dashboard Score: {n_pass}/{n_tot} PASS",
        f"dims = ({dims[0]}, {dims[1]})",
        f"volume_power = {volume_power:.3f}",
        "",
    ]
    for row in rows:
        value = row["value"]
        if isinstance(value, bool):
            vstr = "true" if value else "false"
        elif np.isfinite(_to_float(value)):
            vstr = f"{_to_float(value):.4g}"
        else:
            vstr = "nan"
        flag = "PASS" if row["pass"] else "FAIL"
        lines.append(f"[{flag}] {row['name']}: {vstr}")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11.5,
        family="monospace",
    )

    fig.suptitle(
        "RHVAE Metric Dashboard: Forces, Anisotropy, Volume Geometry, and Stability Checks",
        fontsize=20,
        y=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a comprehensive metric force dashboard.")
    p.add_argument("--model_path", type=str, default="outputs/reference_models/4K")
    p.add_argument("--output_dir", type=str, default="results/metric_force_dashboard")
    p.add_argument("--dims", type=str, default="auto", help="auto or 'i,j'")
    p.add_argument("--grid_n", type=int, default=160)
    p.add_argument("--bounds_scale", type=float, default=2.25)
    p.add_argument("--min_bounds", type=float, default=6.0)
    p.add_argument("--quiver_n", type=int, default=22)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--force_batch_size", type=int, default=512)
    p.add_argument("--spd_points", type=int, default=2500)
    p.add_argument("--volume_power", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    model_path = Path(args.model_path)
    device = torch.device("cpu")
    model = load_geometry_model(model_path=model_path, device=device)
    centroids = model.centroids_tens.detach()
    dims = parse_dims(str(args.dims), centroids)
    d0, d1 = dims

    xx, yy, grid_points_full, c2 = build_grid(
        centroids=centroids,
        dims=dims,
        grid_n=int(args.grid_n),
        bounds_scale=float(args.bounds_scale),
        min_bounds=float(args.min_bounds),
    )
    fields = evaluate_grid_fields(
        model=model,
        centroids=centroids,
        points_full=grid_points_full,
        dims=dims,
        volume_power=float(args.volume_power),
        batch_size=int(args.batch_size),
    )

    # Quiver points (coarse grid).
    xq = np.linspace(float(xx.min()), float(xx.max()), int(args.quiver_n))
    yq = np.linspace(float(yy.min()), float(yy.max()), int(args.quiver_n))
    qx_m, qy_m = np.meshgrid(xq, yq, indexing="xy")
    q_flat = np.stack([qx_m.reshape(-1), qy_m.reshape(-1)], axis=1).astype(np.float32)
    q_points_full = centroids.mean(dim=0, keepdim=True).repeat(q_flat.shape[0], 1)
    q_points_full[:, d0] = torch.from_numpy(q_flat[:, 0]).to(device)
    q_points_full[:, d1] = torch.from_numpy(q_flat[:, 1]).to(device)

    q_force = evaluate_volume_force(
        model=model,
        points_full=q_points_full,
        dims=dims,
        volume_power=float(args.volume_power),
        batch_size=int(args.force_batch_size),
    )
    q_step = evaluate_rescue_step_quiver(
        model=model,
        centroids=centroids,
        points_full=q_points_full,
        dims=dims,
    )
    with torch.no_grad():
        idx2 = torch.tensor([d0, d1], device=device)
        q_g2 = model.G_inv(q_points_full).index_select(1, idx2).index_select(2, idx2)
        q_evals, q_evecs = torch.linalg.eigh(q_g2)
    q_pdir = q_evecs[:, :, 1].cpu().numpy()
    q_lmax = q_evals[:, 1].cpu().numpy()

    r0 = model_r0(model, device=device)
    metrics = compute_diagnostics(
        model=model,
        centroids=centroids,
        grid_points_full=grid_points_full,
        dims=dims,
        fields=fields,
        q_points_full=q_points_full,
        q_force_xy=q_force,
        q_step_xy=q_step,
        r0=float(r0),
        spd_points=int(args.spd_points),
        seed=int(args.seed),
    )
    rows = build_scorecard(metrics)

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metric_dashboard.png"

    plot_dashboard(
        out_path=out_path,
        xx=xx,
        yy=yy,
        c2=c2,
        fields=fields,
        qx=qx_m.reshape(-1),
        qy=qy_m.reshape(-1),
        q_force=q_force,
        q_step=q_step,
        q_pdir=q_pdir,
        q_lmax=q_lmax,
        metrics=metrics,
        rows=rows,
        dims=dims,
        volume_power=float(args.volume_power),
    )
    save_scorecard(
        out_dir=out_dir,
        metrics=metrics,
        rows=rows,
        metadata={
            "model_path": str(model_path),
            "dims": [int(d0), int(d1)],
            "grid_n": int(args.grid_n),
            "quiver_n": int(args.quiver_n),
            "volume_power": float(args.volume_power),
            "r0": float(r0),
            "timestamp": stamp,
        },
    )

    n_pass = int(sum(1 for r in rows if r["pass"]))
    print(f"Saved dashboard: {out_path}")
    print(f"Saved diagnostics: {out_dir / 'diagnostics.json'}")
    print(f"Saved scorecard: {out_dir / 'diagnostics.md'}")
    print(f"Score: {n_pass}/{len(rows)} PASS")
    print(
        "Key metrics: "
        f"logdet_contrast={metrics['logdet_contrast_manifold_minus_void']:.3e}, "
        f"force_align_outside={metrics['force_alignment_outside_mean']:.3f}, "
        f"plateau_outside={metrics['plateau_fraction_outside_rescue_metric']:.3f}, "
        f"spd_violation_rate={metrics['spd_violation_rate']:.3e}"
    )


if __name__ == "__main__":
    main()
