#!/bin/bash
#SBATCH --job-name=metric_assess
#SBATCH --output=logs/metric_assess_%j.out
#SBATCH --error=logs/metric_assess_%j.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu

set -euo pipefail

mkdir -p logs

module purge
source /scratch/alaforgu/miniconda3/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-rhvae-geometry-benchmark}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"

BASELINE_MODEL_PATH="${BASELINE_MODEL_PATH:-outputs/pythae_rhvae_baseline/2026-02-09_15-04-06}"
OURS_MODEL_PATH="${OURS_MODEL_PATH:-outputs/pythae_rhvae_baseline/2026-02-09_18-16-00}"
PROTOCOL_PROFILE="${PROTOCOL_PROFILE:-all}"
BENCH_PROFILE="${BENCH_PROFILE:-short}"
RUN_FID="${RUN_FID:-0}"
RUN_QUALITY="${RUN_QUALITY:-1}"
SAVE_PLOTS="${SAVE_PLOTS:-1}"
COLLECT_REFERENCE_VISUALS="${COLLECT_REFERENCE_VISUALS:-1}"
SAMPLING_SEEDS="${SAMPLING_SEEDS:-13 29 47 71 89}"
WAND_GROUP="${WAND_GROUP:-metric_assessment_benchmark}"

echo "=========================================="
echo "Metric Assessment Benchmark"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-local}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Time: $(date)"
echo "Baseline: ${BASELINE_MODEL_PATH}"
echo "Aniso: ${OURS_MODEL_PATH}"
echo "Benchmark profile: ${BENCH_PROFILE}"
echo "Protocol profile: ${PROTOCOL_PROFILE}"
echo "Seeds: ${SAMPLING_SEEDS}"
echo "W&B project/entity: ${WANDB_PROJECT}/${WANDB_ENTITY:-<default>}"
echo "=========================================="

CMD=(python scripts/metric_assessment_suite.py
  --baseline_model_path "${BASELINE_MODEL_PATH}"
  --ours_model_path "${OURS_MODEL_PATH}"
  --profile "${BENCH_PROFILE}"
  --protocol_profile "${PROTOCOL_PROFILE}"
  --sampling_seeds ${SAMPLING_SEEDS}
  --wandb_project "${WANDB_PROJECT}"
  --wandb_group "${WAND_GROUP}"
  --wandb_name_mode auto
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb_entity "${WANDB_ENTITY}")
fi

if [[ "${RUN_QUALITY}" -eq 1 ]]; then
  CMD+=(--run_quality)
fi
if [[ "${RUN_FID}" -eq 1 ]]; then
  CMD+=(--run_fid)
else
  CMD+=(--skip_fid)
fi
if [[ "${SAVE_PLOTS}" -eq 1 ]]; then
  CMD+=(--save_plots)
fi
if [[ "${COLLECT_REFERENCE_VISUALS}" -eq 1 ]]; then
  CMD+=(--collect_reference_visuals)
fi

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
