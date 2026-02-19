#!/usr/bin/env python3
"""Benchmark sampling/interpolation/rescue for two RHVAE metrics (baseline vs ours).

Runs a common set of diagnostics for each model and optionally produces
side-by-side rescue (void-drop) figures.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
ANALYZE_SCRIPT = ROOT_DIR / "scripts" / "analyze_metric_full.py"
SAMPLING_SCRIPT = ROOT_DIR / "scripts" / "sampling_diagnostics.py"
DEMO_SCRIPT = ROOT_DIR / "scripts" / "rhmc_chain_demo.py"

ANALYSIS_PLOTS = [
    ("Metric Field (Tissot)", "metric_field_tissot.png"),
    ("Log-det Landscape", "logdet_landscape.png"),
    ("Volume Rescue Quiver", "volume_rescue_quiver.png"),
    ("Metric Surface 3D", "metric_surface_3d.png"),
    ("Geodesic Batch", "geodesic_batch.png"),
    ("Geodesic Energy", "geodesic_energy_profiles.png"),
    ("RHMC Chain", "rhmc_chain.png"),
]

SAMPLING_PLOTS = [
    ("Interpolation Paths", "interpolation_paths.png"),
    ("Multi-Start Sampling", "multi_start_sampling.png"),
    ("RHMC Diagnostics", "rhmc_diagnostics.png"),
    ("FID Comparison", "fid_comparison.png"),
    ("Quality Metrics", "quality_metrics.png"),
]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _maybe_extend(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def _run_analyze(model_path: str, out_dir: Path, device: str, bounds: float) -> None:
    cmd = [
        sys.executable,
        str(ANALYZE_SCRIPT),
        "--model_path",
        model_path,
        "--output_dir",
        str(out_dir),
        "--analysis_bounds",
        str(bounds),
        "--device",
        device,
        "--skip_distortion",
    ]
    _run(cmd)


def _run_sampling(
    model_path: str,
    out_dir: Path,
    device: str,
    skip_fid: bool,
    skip_quality: bool,
    skip_multi_start: bool,
    skip_interpolation: bool,
    skip_rhmc: bool,
    aggressive_void: bool,
) -> None:
    cmd = [
        sys.executable,
        str(SAMPLING_SCRIPT),
        "--model_path",
        model_path,
        "--output_dir",
        str(out_dir),
        "--device",
        device,
    ]
    _maybe_extend(cmd, "--skip_fid", skip_fid)
    _maybe_extend(cmd, "--skip_quality", skip_quality)
    _maybe_extend(cmd, "--skip_multi_start", skip_multi_start)
    _maybe_extend(cmd, "--skip_interpolation", skip_interpolation)
    _maybe_extend(cmd, "--skip_rhmc", skip_rhmc)
    _maybe_extend(cmd, "--aggressive_void", aggressive_void)
    _run(cmd)


def _run_rescue(
    model_path: str,
    out_path: Path,
    device: str,
    sampler: str,
    start_pos: Iterable[float],
    chain_length: int,
    n_lf: int,
    eps_lf: float,
    volume_power: float | None,
    exact: bool,
    bounds: float,
    grid_size: int,
) -> None:
    cmd = [
        sys.executable,
        str(DEMO_SCRIPT),
        "--model_path",
        model_path,
        "--sampler",
        sampler,
        "--chain_length",
        str(chain_length),
        "--n_lf",
        str(n_lf),
        "--eps_lf",
        str(eps_lf),
        "--start_pos",
        *[str(v) for v in start_pos],
        "--bounds",
        str(bounds),
        "--grid_size",
        str(grid_size),
        "--out",
        str(out_path),
        "--skip_energy_maps",
        "--skip_conservation",
        "--potential_only",
    ]
    if exact:
        cmd.append("--exact")
    else:
        cmd.append("--approx")
    if volume_power is not None:
        cmd.extend(["--volume_power", str(volume_power)])
    if device:
        cmd.extend(["--device", device])
    _run(cmd)


def _combine_images(left_path: Path, right_path: Path, out_path: Path) -> None:
    left = plt.imread(str(left_path))
    right = plt.imread(str(right_path))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(left)
    axes[0].set_title("Baseline")
    axes[0].axis("off")

    axes[1].imshow(right)
    axes[1].set_title("AnisoRHVAE")
    axes[1].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _latest_subdir(path: Path) -> Path | None:
    if not path.exists():
        return None
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _collect_plot_pairs(
    baseline_root: Path,
    ours_root: Path,
    plot_specs: list[tuple[str, str]],
) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    for title, filename in plot_specs:
        b_path = baseline_root / filename
        o_path = ours_root / filename
        if b_path.exists() and o_path.exists():
            pairs.append((title, b_path, o_path))
    return pairs


def _build_panel(
    pairs: list[tuple[str, Path, Path]],
    out_path: Path,
    left_title: str = "Baseline",
    right_title: str = "AnisoRHVAE",
) -> None:
    if not pairs:
        return
    n_rows = len(pairs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    for idx, (row_title, left_path, right_path) in enumerate(pairs):
        left_img = plt.imread(str(left_path))
        right_img = plt.imread(str(right_path))
        ax_left, ax_right = axes[idx]
        ax_left.imshow(left_img)
        ax_right.imshow(right_img)
        ax_left.axis("off")
        ax_right.axis("off")
        if idx == 0:
            ax_left.set_title(left_title)
            ax_right.set_title(right_title)
        ax_left.set_ylabel(row_title, rotation=90, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sampling/interpolation/rescue for baseline vs AnisoRHVAE."
    )
    parser.add_argument("--baseline_model_path", type=str, required=True)
    parser.add_argument("--ours_model_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/benchmark_compare")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--analysis_bounds", type=float, default=6.0)
    parser.add_argument("--grid_size", type=int, default=60)

    parser.add_argument("--skip_analyze", action="store_true")
    parser.add_argument("--skip_sampling", action="store_true")
    parser.add_argument("--skip_rescue", action="store_true")
    parser.add_argument("--skip_compare_panels", action="store_true")

    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--skip_quality", action="store_true")
    parser.add_argument("--skip_multi_start", action="store_true")
    parser.add_argument("--skip_interpolation", action="store_true")
    parser.add_argument("--skip_rhmc", action="store_true")
    parser.add_argument("--aggressive_void", action="store_true")

    parser.add_argument("--start_pos", type=float, nargs="+", default=[6.0, 8.0])
    parser.add_argument("--chain_length", type=int, default=80)
    parser.add_argument("--n_lf", type=int, default=80)
    parser.add_argument("--eps_lf", type=float, default=0.005)
    parser.add_argument("--volume_power", type=float, default=1.0)
    parser.add_argument("--baseline_sampler", type=str, default="riemannian")
    parser.add_argument("--ours_sampler", type=str, default="volume_riemannian")
    parser.add_argument("--approx", action="store_true")

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = _ensure_dir(Path(args.out_dir) / timestamp)

    baseline_dir = _ensure_dir(out_root / "baseline")
    ours_dir = _ensure_dir(out_root / "ours")

    if not args.skip_analyze:
        _run_analyze(args.baseline_model_path, baseline_dir / "analysis_results", args.device, args.analysis_bounds)
        _run_analyze(args.ours_model_path, ours_dir / "analysis_results", args.device, args.analysis_bounds)

    if not args.skip_sampling:
        _run_sampling(
            args.baseline_model_path,
            baseline_dir / "sampling_diagnostics",
            args.device,
            skip_fid=args.skip_fid,
            skip_quality=args.skip_quality,
            skip_multi_start=args.skip_multi_start,
            skip_interpolation=args.skip_interpolation,
            skip_rhmc=args.skip_rhmc,
            aggressive_void=args.aggressive_void,
        )
        _run_sampling(
            args.ours_model_path,
            ours_dir / "sampling_diagnostics",
            args.device,
            skip_fid=args.skip_fid,
            skip_quality=args.skip_quality,
            skip_multi_start=args.skip_multi_start,
            skip_interpolation=args.skip_interpolation,
            skip_rhmc=args.skip_rhmc,
            aggressive_void=args.aggressive_void,
        )

    if not args.skip_rescue:
        baseline_void = baseline_dir / "void_drop.png"
        ours_void = ours_dir / "void_drop.png"
        _run_rescue(
            args.baseline_model_path,
            baseline_void,
            args.device,
            sampler=args.baseline_sampler,
            start_pos=args.start_pos,
            chain_length=args.chain_length,
            n_lf=args.n_lf,
            eps_lf=args.eps_lf,
            volume_power=None,
            exact=not args.approx,
            bounds=args.analysis_bounds,
            grid_size=args.grid_size,
        )
        _run_rescue(
            args.ours_model_path,
            ours_void,
            args.device,
            sampler=args.ours_sampler,
            start_pos=args.start_pos,
            chain_length=args.chain_length,
            n_lf=args.n_lf,
            eps_lf=args.eps_lf,
            volume_power=args.volume_power,
            exact=not args.approx,
            bounds=args.analysis_bounds,
            grid_size=args.grid_size,
        )
        _combine_images(baseline_void, ours_void, out_root / "void_drop_comparison.png")

    if not args.skip_compare_panels:
        baseline_analysis_base = baseline_dir / "analysis_results"
        ours_analysis_base = ours_dir / "analysis_results"
        baseline_analysis = _latest_subdir(baseline_analysis_base) or baseline_analysis_base
        ours_analysis = _latest_subdir(ours_analysis_base) or ours_analysis_base
        analysis_pairs = _collect_plot_pairs(baseline_analysis, ours_analysis, ANALYSIS_PLOTS)
        _build_panel(
            analysis_pairs,
            out_root / "analysis_comparison_panel.png",
        )

        baseline_sampling_base = baseline_dir / "sampling_diagnostics"
        ours_sampling_base = ours_dir / "sampling_diagnostics"
        baseline_sampling = _latest_subdir(baseline_sampling_base) or baseline_sampling_base
        ours_sampling = _latest_subdir(ours_sampling_base) or ours_sampling_base
        sampling_pairs = _collect_plot_pairs(baseline_sampling, ours_sampling, SAMPLING_PLOTS)
        _build_panel(
            sampling_pairs,
            out_root / "sampling_comparison_panel.png",
        )

    print(f"Benchmark outputs saved to: {out_root}")


if __name__ == "__main__":
    main()
