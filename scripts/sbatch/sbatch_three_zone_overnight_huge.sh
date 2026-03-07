#!/bin/bash
#SBATCH --job-name=threezone_huge
#SBATCH --output=logs/threezone_huge_%j.out
#SBATCH --error=logs/threezone_huge_%j.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Under Slurm, BASH_SOURCE may point to a spool copy; prefer submit directory.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${DEFAULT_REPO_ROOT}}}"
cd "${REPO_ROOT}"

if [[ ! -w "${REPO_ROOT}" ]]; then
  echo "[threezone_huge] ERROR: repo root is not writable: ${REPO_ROOT}" >&2
  echo "[threezone_huge] Set REPO_ROOT to a writable checkout path." >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/logs"

module purge
source /scratch/alaforgu/miniconda3/etc/profile.d/conda.sh

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-outputs/metric_core4_sweep/2026-02-17_11-07-08}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/three_zone_overnight_huge}"
RUN_STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="${OUTPUT_ROOT}/${RUN_STAMP}"

DEVICE="${DEVICE:-cuda}"
SAMPLER_NAME="${SAMPLER_NAME:-volume_riemannian}"
USE_DUAL_METRIC="${USE_DUAL_METRIC:-1}"
VOLUME_POWER="${VOLUME_POWER:-0.8}"
STEPS="${STEPS:-70}"
N_LF_INNER="${N_LF_INNER:-10}"
EPS="${EPS:-0.03}"
FP_STEPS="${FP_STEPS:-3}"
FP_DAMPING="${FP_DAMPING:-0.72}"
MOMENTUM_PERSIST="${MOMENTUM_PERSIST:-0.0}"

SEEDS="${SEEDS:-13,29,47,71,89}"
ALPHA_GRID="${ALPHA_GRID:-0.75,0.80,0.85,0.90}"
POWER_GRID="${POWER_GRID:--1.0,-1.1,-1.2,-1.3}"
RADIAL_GRID="${RADIAL_GRID:-0.05,0.07,0.09}"
EIG_FLOOR="${EIG_FLOOR:-1e-8}"
TOPK_PLOTS="${TOPK_PLOTS:-8}"
PLOT_SEEDS="${PLOT_SEEDS:-13,29,47}"

NEAR_MULTIPLIER="${NEAR_MULTIPLIER:-1.2}"
FAR_MULTIPLIER="${FAR_MULTIPLIER:-2.4}"
MANIFOLD_POINTS="${MANIFOLD_POINTS:-3}"

# Make sweep config available to the Python heredoc process.
export MODEL_PATH OUTPUT_ROOT RUN_STAMP RUN_DIR
export DEVICE SAMPLER_NAME USE_DUAL_METRIC VOLUME_POWER
export STEPS N_LF_INNER EPS FP_STEPS FP_DAMPING MOMENTUM_PERSIST
export SEEDS ALPHA_GRID POWER_GRID RADIAL_GRID EIG_FLOOR TOPK_PLOTS PLOT_SEEDS
export NEAR_MULTIPLIER FAR_MULTIPLIER MANIFOLD_POINTS

echo "=========================================="
echo "Three-Zone Overnight Huge Sweep"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-local}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "PWD:  ${REPO_ROOT}"
echo "Model: ${MODEL_PATH}"
echo "Run dir: ${RUN_DIR}"
echo "Sampler: ${SAMPLER_NAME} (dual=${USE_DUAL_METRIC})"
echo "Seeds: ${SEEDS}"
echo "Alpha grid: ${ALPHA_GRID}"
echo "Power grid: ${POWER_GRID}"
echo "Radial grid: ${RADIAL_GRID}"
echo "=========================================="

mkdir -p "${RUN_DIR}"

python - <<'PY'
import csv
import itertools
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from scripts.quick.visualize_rhmc_three_zones_real_metric import (
    choose_manifold_starts,
    make_outside_starts,
)
from scripts.rhmc_chain_demo import _build_model, _build_sampler, run_hmc_chain


def parse_list(raw: str, cast):
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def metric_file(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_dir():
        return p / "rhvae_metric.pt"
    return p


def model_arg(path_like: str) -> str:
    p = Path(path_like)
    return str(p)


def cfg_name(alpha: float, power: float, radial: float) -> str:
    return f"a{alpha:.2f}_p{power:.2f}_rw{radial:.2f}".replace("-", "m")


run_dir = Path(os.environ["RUN_DIR"])
run_dir.mkdir(parents=True, exist_ok=True)

device_req = os.environ.get("DEVICE", "cuda")
if device_req == "cuda" and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_path_env = os.environ["MODEL_PATH"]
model = _build_model(metric_file(model_path_env), device)

sampler_name = os.environ["SAMPLER_NAME"]
use_dual = bool(int(os.environ.get("USE_DUAL_METRIC", "1")))
volume_power = float(os.environ["VOLUME_POWER"])
steps = int(os.environ["STEPS"])
n_lf = int(os.environ["N_LF_INNER"])
eps = float(os.environ["EPS"])
fp_steps = int(os.environ["FP_STEPS"])
fp_damping = float(os.environ["FP_DAMPING"])
momentum_persist = float(os.environ["MOMENTUM_PERSIST"])

seeds = parse_list(os.environ["SEEDS"], int)
alpha_grid = parse_list(os.environ["ALPHA_GRID"], float)
power_grid = parse_list(os.environ["POWER_GRID"], float)
radial_grid = parse_list(os.environ["RADIAL_GRID"], float)
eig_floor = float(os.environ["EIG_FLOOR"])
topk_plots = int(os.environ["TOPK_PLOTS"])
plot_seeds = parse_list(os.environ["PLOT_SEEDS"], int)

near_multiplier = float(os.environ["NEAR_MULTIPLIER"])
far_multiplier = float(os.environ["FAR_MULTIPLIER"])
manifold_points = int(os.environ["MANIFOLD_POINTS"])

centroids = model.centroids_tens.detach()
starts_m = choose_manifold_starts(centroids, (0, 1), manifold_points)
z_near, z_far = make_outside_starts(centroids, (0, 1), near_multiplier, far_multiplier)
starts = [
    ("manifold_1", starts_m[0]),
    ("manifold_2", starts_m[1]),
    ("manifold_3", starts_m[2]),
    ("near_outside", z_near),
    ("far_outside", z_far),
]
C = model.centroids_tens

rows = []
total_cfg = len(alpha_grid) * len(power_grid) * len(radial_grid)
cfg_idx = 0

for alpha, power, radial in itertools.product(alpha_grid, power_grid, radial_grid):
    cfg_idx += 1
    name = cfg_name(alpha, power, radial)
    print(f"[{cfg_idx}/{total_cfg}] cfg={name}", flush=True)

    per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model.void_eigshape_mode = "det_preserving_spectral"
        model.void_eigshape_alpha_min = float(alpha)
        model.void_eigshape_power = float(power)
        model.void_eigshape_eig_floor = eig_floor

        sampler = _build_sampler(
            name=sampler_name,
            model=model,
            mcmc_steps=steps,
            n_lf=n_lf,
            eps_lf=eps,
            beta_zero=1.0,
            volume_power=volume_power,
            radial_prior_weight=float(radial),
            radial_prior_center=None,
            use_dual_metric=bool(use_dual),
        )
        sampler.exact = True
        sampler.fp_steps = fp_steps
        sampler.fp_damping = fp_damping
        sampler.momentum_persist = momentum_persist

        end = {}
        start = {}
        for label, z0 in starts:
            z0b = z0.unsqueeze(0).to(device)
            chain, _ = run_hmc_chain(
                start_z=z0b,
                sampler=sampler,
                chain_length=steps,
                n_lf=n_lf,
                eps_lf=eps,
                eps_jitter=0.0,
                n_lf_jitter=0,
            )
            z_end = torch.tensor(chain[-1], dtype=C.dtype, device=C.device)
            start[label] = float(torch.min(torch.norm(C - z0.to(C.device), dim=1)).item())
            end[label] = float(torch.min(torch.norm(C - z_end, dim=1)).item())

        manifold_mean = (end["manifold_1"] + end["manifold_2"] + end["manifold_3"]) / 3.0
        near_end = end["near_outside"]
        far_end = end["far_outside"]
        near_delta = near_end - start["near_outside"]
        far_delta = far_end - start["far_outside"]
        score = far_end + 0.7 * near_end + 0.5 * manifold_mean
        per_seed.append(
            {
                "seed": int(seed),
                "manifold_mean_end": manifold_mean,
                "near_end": near_end,
                "far_end": far_end,
                "near_delta": near_delta,
                "far_delta": far_delta,
                "score": score,
            }
        )

    agg = {
        "cfg_name": name,
        "alpha_min": float(alpha),
        "power": float(power),
        "radial_weight": float(radial),
        "n_seeds": len(per_seed),
        "manifold_mean_end_mean": float(np.mean([r["manifold_mean_end"] for r in per_seed])),
        "manifold_mean_end_std": float(np.std([r["manifold_mean_end"] for r in per_seed])),
        "near_end_mean": float(np.mean([r["near_end"] for r in per_seed])),
        "near_end_std": float(np.std([r["near_end"] for r in per_seed])),
        "far_end_mean": float(np.mean([r["far_end"] for r in per_seed])),
        "far_end_std": float(np.std([r["far_end"] for r in per_seed])),
        "near_delta_mean": float(np.mean([r["near_delta"] for r in per_seed])),
        "far_delta_mean": float(np.mean([r["far_delta"] for r in per_seed])),
        "score_mean": float(np.mean([r["score"] for r in per_seed])),
        "score_std": float(np.std([r["score"] for r in per_seed])),
    }
    rows.append(agg)

rows.sort(key=lambda r: (r["score_mean"], r["far_end_mean"]))

summary_csv = run_dir / "three_zone_sweep_summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

best = rows[0]
(run_dir / "best_config.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
print(f"Best config: {json.dumps(best, indent=2)}", flush=True)
print(f"Saved sweep CSV: {summary_csv}", flush=True)

topk = rows[: max(1, topk_plots)]
(run_dir / "topk_configs.json").write_text(json.dumps(topk, indent=2), encoding="utf-8")

# Render three-zone plots for top-k configs and selected seeds.
for rank, cfg in enumerate(topk, start=1):
    cfg_dir = run_dir / "topk_plots" / f"rank{rank:02d}_{cfg['cfg_name']}"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for seed in plot_seeds:
        cmd = [
            "python",
            "scripts/quick/visualize_rhmc_three_zones_real_metric.py",
            "--model_path",
            model_arg(model_path_env),
            "--output_dir",
            str(cfg_dir),
            "--seed",
            str(seed),
            "--sampler_name",
            sampler_name,
            "--volume_power",
            str(volume_power),
            "--steps",
            str(steps),
            "--n_lf_inner",
            str(n_lf),
            "--eps",
            str(eps),
            "--fp_steps",
            str(fp_steps),
            "--fp_damping",
            str(fp_damping),
            "--momentum_persist",
            str(momentum_persist),
            "--eps_jitter",
            "0.0",
            "--n_lf_jitter",
            "0",
            "--radial_prior_weight",
            str(cfg["radial_weight"]),
            "--void_eigshape_mode",
            "det_preserving_spectral",
            "--void_eigshape_alpha_min",
            str(cfg["alpha_min"]),
            "--void_eigshape_power",
            str(cfg["power"]),
            "--void_eigshape_eig_floor",
            str(eig_floor),
            "--manifold_points",
            str(manifold_points),
            "--near_multiplier",
            str(near_multiplier),
            "--far_multiplier",
            str(far_multiplier),
        ]
        if use_dual:
            cmd.append("--use_dual_metric")

        print(f"[plot] rank={rank} seed={seed} cmd={' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

print("Overnight huge sweep complete.", flush=True)
PY

echo "Done. run_dir=${RUN_DIR}"
