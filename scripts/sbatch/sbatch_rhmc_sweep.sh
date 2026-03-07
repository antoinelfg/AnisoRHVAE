#!/bin/bash
#SBATCH --job-name=rhmc_sweep
#SBATCH --output=logs/rhmc_sweep_%j.out
#SBATCH --error=logs/rhmc_sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=gpu

set -euo pipefail

mkdir -p logs

module purge

source /scratch/alaforgu/miniconda3/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-rhmc-geometry}"
export WANDB_ENTITY="${WANDB_ENTITY:-YOUR_ENTITY}"

SWEEP_ID="${SWEEP_ID:-${1:-}}"
MAX_RUNS="${MAX_RUNS:-${2:-0}}"

if [[ -z "${SWEEP_ID}" ]]; then
  echo "Missing SWEEP_ID."
  echo "Usage: sbatch scripts/sbatch_rhmc_sweep.sh SWEEP_ID [MAX_RUNS]"
  echo "Example (full path): sbatch scripts/sbatch_rhmc_sweep.sh entity/project/sweep_id"
  exit 1
fi

if [[ "${SWEEP_ID}" == */* ]]; then
  SWEEP_PATH="${SWEEP_ID}"
else
  if [[ "${WANDB_ENTITY}" == "YOUR_ENTITY" || -z "${WANDB_ENTITY}" || -z "${WANDB_PROJECT}" ]]; then
    echo "Set WANDB_ENTITY and WANDB_PROJECT, or pass full sweep path entity/project/sweep_id."
    exit 1
  fi
  SWEEP_PATH="${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
fi

echo "=========================================="
echo "🚀 RHMC Sweep Agent"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "Sweep: ${SWEEP_PATH}"
echo "Max runs: ${MAX_RUNS}"
echo "=========================================="

AGENT_CMD="wandb agent ${SWEEP_PATH}"
if [[ "${MAX_RUNS}" -gt 0 ]]; then
  AGENT_CMD="${AGENT_CMD} --count ${MAX_RUNS}"
fi

echo "Running: ${AGENT_CMD}"
eval "${AGENT_CMD}"
