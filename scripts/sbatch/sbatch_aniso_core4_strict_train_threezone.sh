#!/usr/bin/env bash
#SBATCH --job-name=aniso_core4_strict
#SBATCH --output=logs/aniso_core4_strict_%j.out
#SBATCH --error=logs/aniso_core4_strict_%j.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${REPO_ROOT}"

mkdir -p logs

module purge || true
if [[ -f /scratch/alaforgu/miniconda3/etc/profile.d/conda.sh ]]; then
  source /scratch/alaforgu/miniconda3/etc/profile.d/conda.sh
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
BENCH_CONFIG="${BENCH_CONFIG:-configs/benchmark_aniso_core4_strict_train_threezone.yaml}"
SUBSET_NS="${SUBSET_NS:-50 100 1000}"
SEED="${SEED:-13}"
REUSE_POLICY="${REUSE_POLICY:-reuse_latest}"
WANDB_MODE="${WANDB_MODE:-online}"

read -r -a NS_ARR <<< "${SUBSET_NS}"

CMD=(
  "${PYTHON_BIN}"
  scripts/run_aniso_core4_strict_train_threezone_benchmark.py
  --config "${BENCH_CONFIG}"
  --seed "${SEED}"
  --reuse_policy "${REUSE_POLICY}"
  --python_bin "${PYTHON_BIN}"
)

if [[ ${#NS_ARR[@]} -gt 0 ]]; then
  CMD+=(--subset_ns "${NS_ARR[@]}")
fi
if [[ -n "${WANDB_MODE}" ]]; then
  CMD+=(--wandb_mode "${WANDB_MODE}")
fi

echo "=========================================="
echo "Aniso Core4 Strict Train+ThreeZone"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-local}"
echo "repo_root=${REPO_ROOT}"
echo "python=${PYTHON_BIN}"
echo "config=${BENCH_CONFIG}"
echo "subset_ns=${SUBSET_NS}"
echo "seed=${SEED}"
echo "reuse_policy=${REUSE_POLICY}"
echo "wandb_mode=${WANDB_MODE}"
echo "=========================================="
echo "[sbatch_aniso_core4_strict] cmd: ${CMD[*]}"

"${CMD[@]}"

echo "[sbatch_aniso_core4_strict] complete"
