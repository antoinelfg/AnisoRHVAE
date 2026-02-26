#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Reproducible retraining launcher for ellipse basic setup (latent_dim=2),
# using the existing pythae baseline training pipeline that already emits:
# - reconstructions_epoch_*.png (original vs recon)
# - analysis_results/*/metric_surface_3d*.png (3D metric landscape)
# - rhvae_model.pt / rhvae_metric.pt checkpoints

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/retrain_basic_ellipses_latent2}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.0005}"
NUM_SEQUENCES="${NUM_SEQUENCES:-200}"
MAX_FRAMES="${MAX_FRAMES:-3000}"
VIS_EVERY="${VIS_EVERY:-5}"
SEED_BASELINE="${SEED_BASELINE:-42}"
SEED_ANISO="${SEED_ANISO:-42}"

# wandb handling:
# - WANDB_MODE=offline by default for non-interactive robustness
# - set WANDB_MODE=online + WANDB_API_KEY externally to upload runs
export WANDB_MODE="${WANDB_MODE:-offline}"
WAND_PROJECT="${WAND_PROJECT:-RHVAE-baseline}"
WAND_ENTITY="${WAND_ENTITY:-}"
WAND_GROUP="${WAND_GROUP:-retrain_basic_ellipses_latent2}"
WAND_TAGS="${WAND_TAGS:-ellipse,basic,latent2,retrain}"

mkdir -p "${OUTPUT_ROOT}"

run_model () {
  local model_name="$1"
  local geometry_case="$2"
  local seed="$3"

  local cmd=(
    python scripts/run_pythae_rhvae_baseline.py
    --latent_dim 2
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --num_sequences "${NUM_SEQUENCES}"
    --max_frames "${MAX_FRAMES}"
    --frame_mode t0
    --n_centroids 100
    --vis_every "${VIS_EVERY}"
    --seed "${seed}"
    --geometry_case "${geometry_case}"
    --output_dir "${OUTPUT_ROOT}/${model_name}"
    --analysis_skip_distortion
    --skip_sampling_diagnostics
    --wandb_project "${WAND_PROJECT}"
    --wandb_group "${WAND_GROUP}"
    --wandb_tags "${WAND_TAGS}"
    --wandb_name_mode auto
  )

  if [[ -n "${WAND_ENTITY}" ]]; then
    cmd+=(--wandb_entity "${WAND_ENTITY}")
  fi

  echo "[retrain_basic_ellipses_latent2] running ${model_name} (${geometry_case})"
  echo "[retrain_basic_ellipses_latent2] cmd: ${cmd[*]}"
  "${cmd[@]}"
}

run_model "baseline" "baseline" "${SEED_BASELINE}"
run_model "aniso" "gravity_well" "${SEED_ANISO}"

echo "[retrain_basic_ellipses_latent2] complete"
echo "[retrain_basic_ellipses_latent2] outputs in: ${OUTPUT_ROOT}"

