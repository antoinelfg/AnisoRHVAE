#!/usr/bin/env bash
#SBATCH --job-name=rhvae_aniso_heatmaps
#SBATCH --output=logs/rhvae_aniso_heatmaps_%j.out
#SBATCH --error=logs/rhvae_aniso_heatmaps_%j.err
#SBATCH --time=24:00:00
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

if [[ ! -w "${REPO_ROOT}" ]]; then
  echo "[sbatch_train_rhvae_aniso_heatmaps] ERROR: repo root is not writable: ${REPO_ROOT}" >&2
  exit 2
fi

mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-python}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed/rotmnist/v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/low_data_models/rotmnist_v1}"
SUBSET_NS="${SUBSET_NS:-50 100 500}"
SUBSET_SEEDS="${SUBSET_SEEDS:-13 29 47 71 89}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RHVAE_DROP_LAST="${RHVAE_DROP_LAST:-auto}"
RHVAE_VIS_EVERY="${RHVAE_VIS_EVERY:-5}"
RHVAE_METRIC_GRID_RES="${RHVAE_METRIC_GRID_RES:-40}"
RHVAE_METRIC_TISSOT_GRID_RES="${RHVAE_METRIC_TISSOT_GRID_RES:-16}"
RHVAE_AUTO_TEMPERATURE="${RHVAE_AUTO_TEMPERATURE:-0}"
RHVAE_AUTO_TEMPERATURE_STAT="${RHVAE_AUTO_TEMPERATURE_STAT:-mean_nn}"
RHVAE_AUTO_TEMPERATURE_EVERY="${RHVAE_AUTO_TEMPERATURE_EVERY:-0}"
RHVAE_TEMPERATURE_SCALE="${RHVAE_TEMPERATURE_SCALE:-1.0}"
QUICK="${QUICK:-0}"

WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-low_data_rhvae_aniso_heatmaps}"
WANDB_TAGS="${WANDB_TAGS:-low-data,rotmnist,heatmaps}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_NAME_MODE="${WANDB_NAME_MODE:-auto}"

read -r -a NS_ARR <<< "${SUBSET_NS}"
read -r -a SEED_ARR <<< "${SUBSET_SEEDS}"

CMD=(
  "${PYTHON_BIN}" scripts/train_missing_low_data_models.py
  --models rhvae_standard aniso
  --processed_dir "${PROCESSED_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --subset_ns "${NS_ARR[@]}"
  --subset_seeds "${SEED_ARR[@]}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --materialize_aliases
  --rhvae_drop_last "${RHVAE_DROP_LAST}"
  --rhvae_plot_heatmaps_during_training
  --rhvae_vis_every "${RHVAE_VIS_EVERY}"
  --rhvae_metric_grid_res "${RHVAE_METRIC_GRID_RES}"
  --rhvae_metric_tissot_grid_res "${RHVAE_METRIC_TISSOT_GRID_RES}"
  --wandb_name_mode "${WANDB_NAME_MODE}"
)

if [[ "${RHVAE_AUTO_TEMPERATURE}" == "1" ]]; then
  CMD+=(
    --rhvae_auto_temperature
    --rhvae_auto_temperature_stat "${RHVAE_AUTO_TEMPERATURE_STAT}"
    --rhvae_auto_temperature_every "${RHVAE_AUTO_TEMPERATURE_EVERY}"
    --rhvae_temperature_scale "${RHVAE_TEMPERATURE_SCALE}"
  )
fi

if [[ "${QUICK}" == "1" ]]; then
  CMD+=(--quick)
fi

if [[ -n "${WANDB_PROJECT}" ]]; then
  CMD+=(--wandb_project "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb_entity "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_GROUP}" ]]; then
  CMD+=(--wandb_group "${WANDB_GROUP}")
fi
if [[ -n "${WANDB_TAGS}" ]]; then
  CMD+=(--wandb_tags "${WANDB_TAGS}")
fi
if [[ -n "${WANDB_MODE}" ]]; then
  CMD+=(--wandb_mode "${WANDB_MODE}")
fi

echo "[sbatch_train_rhvae_aniso_heatmaps] repo=${REPO_ROOT}"
echo "[sbatch_train_rhvae_aniso_heatmaps] cmd: ${CMD[*]}"
"${CMD[@]}"
echo "[sbatch_train_rhvae_aniso_heatmaps] complete"
