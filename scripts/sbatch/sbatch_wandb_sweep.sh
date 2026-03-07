#!/bin/bash
#SBATCH --job-name=ghosttrack_sweep
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --time=47:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu

module purge

source /scratch/alaforgu/miniconda3/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_MODE=online
export WANDB_PROJECT="rhvae-geometry-benchmark"
export WANDB_ENTITY="antoine-laforgue-mines-paris-alumni"

SWEEP_ID="xjxx6cw1"
MAX_RUNS=${1:-0}

echo "=========================================="
echo "🚀 RHVAE Sweep Agent"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "Sweep: ${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
echo "Max runs: ${MAX_RUNS}"
echo "=========================================="

AGENT_CMD="wandb agent ${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
if [ "$MAX_RUNS" -gt 0 ]; then
  AGENT_CMD="${AGENT_CMD} --count ${MAX_RUNS}"
fi

echo "Running: ${AGENT_CMD}"
eval "${AGENT_CMD}"
