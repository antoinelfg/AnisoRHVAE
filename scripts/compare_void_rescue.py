#!/usr/bin/env python3
"""Run a side-by-side void-drop comparison for baseline vs. AnisoRHVAE.

This script runs `scripts/rhmc_chain_demo.py` for two models and
creates a combined comparison figure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
DEMO_SCRIPT = ROOT_DIR / "scripts" / "rhmc_chain_demo.py"


def _run_demo(
    model_path: str,
    sampler: str,
    out_path: Path,
    start_pos: list[float],
    chain_length: int,
    n_lf: int,
    eps_lf: float,
    volume_power: float | None,
    exact: bool,
    device: str | None,
    skip_energy_maps: bool,
    skip_conservation: bool,
    potential_only: bool,
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
        "--out",
        str(out_path),
        "--start_pos",
        *[str(v) for v in start_pos],
    ]
    if exact:
        cmd.append("--exact")
    else:
        cmd.append("--approx")
    if device:
        cmd.extend(["--device", device])
    if volume_power is not None:
        cmd.extend(["--volume_power", str(volume_power)])
    if skip_energy_maps:
        cmd.append("--skip_energy_maps")
    if skip_conservation:
        cmd.append("--skip_conservation")
    if potential_only:
        cmd.append("--potential_only")

    subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Void-drop comparison for baseline vs. AnisoRHVAE.")
    parser.add_argument("--baseline_model_path", type=str, required=True)
    parser.add_argument("--ours_model_path", type=str, required=True)
    parser.add_argument("--start_pos", type=float, nargs="+", default=[3.0, 3.0])
    parser.add_argument("--chain_length", type=int, default=60)
    parser.add_argument("--n_lf", type=int, default=60)
    parser.add_argument("--eps_lf", type=float, default=0.005)
    parser.add_argument("--volume_power", type=float, default=1.0)
    parser.add_argument("--baseline_sampler", type=str, default="riemannian")
    parser.add_argument("--ours_sampler", type=str, default="volume_riemannian")
    parser.add_argument("--approx", action="store_true", help="Use approximate integrator (default: exact).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default="results/void_rescue_comparison.png")
    parser.add_argument("--baseline_out", type=str, default=None)
    parser.add_argument("--ours_out", type=str, default=None)
    parser.add_argument(
        "--full_plots",
        action="store_true",
        help="Include energy maps/conservation and full energy traces.",
    )

    args = parser.parse_args()

    def _resolve(path: Path) -> Path:
        return path if path.is_absolute() else (ROOT_DIR / path)

    out_path = _resolve(Path(args.out))
    baseline_out = (
        _resolve(Path(args.baseline_out))
        if args.baseline_out
        else out_path.with_name(f"{out_path.stem}_baseline.png")
    )
    ours_out = (
        _resolve(Path(args.ours_out))
        if args.ours_out
        else out_path.with_name(f"{out_path.stem}_ours.png")
    )

    exact = not args.approx
    skip_energy_maps = not args.full_plots
    skip_conservation = not args.full_plots
    potential_only = not args.full_plots

    print("Running baseline void drop...")
    _run_demo(
        model_path=args.baseline_model_path,
        sampler=args.baseline_sampler,
        out_path=baseline_out,
        start_pos=args.start_pos,
        chain_length=args.chain_length,
        n_lf=args.n_lf,
        eps_lf=args.eps_lf,
        volume_power=None,
        exact=exact,
        device=args.device,
        skip_energy_maps=skip_energy_maps,
        skip_conservation=skip_conservation,
        potential_only=potential_only,
    )

    print("Running AnisoRHVAE void drop...")
    _run_demo(
        model_path=args.ours_model_path,
        sampler=args.ours_sampler,
        out_path=ours_out,
        start_pos=args.start_pos,
        chain_length=args.chain_length,
        n_lf=args.n_lf,
        eps_lf=args.eps_lf,
        volume_power=args.volume_power,
        exact=exact,
        device=args.device,
        skip_energy_maps=skip_energy_maps,
        skip_conservation=skip_conservation,
        potential_only=potential_only,
    )

    _combine_images(baseline_out, ours_out, out_path)
    print(f"Saved comparison to {out_path}")
    print(f"Saved baseline plot to {baseline_out}")
    print(f"Saved ours plot to {ours_out}")


if __name__ == "__main__":
    main()
