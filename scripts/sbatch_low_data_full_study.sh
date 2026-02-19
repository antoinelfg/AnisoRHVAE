#!/usr/bin/env bash
#SBATCH --job-name=low_data_full_study
#SBATCH --output=logs/low_data_full_study_%j.out
#SBATCH --error=logs/low_data_full_study_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

# NOTE:
# Under SLURM, "$0" can point to a spool copy of the script (often non-writable).
# Prefer the submit directory, with an explicit override if needed.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

if [[ ! -w "${REPO_ROOT}" ]]; then
  echo "[sbatch_low_data_full_study] ERROR: repo root is not writable: ${REPO_ROOT}" >&2
  echo "[sbatch_low_data_full_study] Set REPO_ROOT to a writable checkout path." >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs}"
RUN_DIR="${RESULTS_ROOT}/low_data_benchmark/${RUN_ID}"
MARKERS_DIR="${RUN_DIR}/stage_markers"
mkdir -p "${MARKERS_DIR}" "${LOG_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ALLOW_WANDB_ASSETS="${ALLOW_WANDB_ASSETS:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rhvae-low-data-benchmark}"
WANDB_ENTITY="${WANDB_ENTITY:-antoine-laforgue-mines-paris-alumni}"
WANDB_GROUP="${WANDB_GROUP:-low_data_full_study}"
WANDB_TAGS="${WANDB_TAGS:-low-data,rotmnist}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_NAME_MODE="${WANDB_NAME_MODE:-auto}"

function mark_done() {
  local stage="$1"
  touch "${MARKERS_DIR}/${stage}.done"
}

function is_done() {
  local stage="$1"
  [[ -f "${MARKERS_DIR}/${stage}.done" ]]
}

function run_stage() {
  local stage="$1"
  shift
  if is_done "$stage"; then
    echo "[sbatch_low_data_full_study] skip stage '${stage}' (already done)"
    return 0
  fi
  echo "[sbatch_low_data_full_study] running stage '${stage}'"
  "$@"
  mark_done "$stage"
}

SYNC_ARGS=(
  "$PYTHON_BIN" scripts/sync_assets.py
  --skip_datasets
  --materialize_aliases
)
if [[ "$ALLOW_WANDB_ASSETS" == "1" ]]; then
  SYNC_ARGS+=(--allow_wandb)
fi

WARGS=()
if [[ -n "$WANDB_PROJECT" ]]; then
  WARGS+=(--wandb_project "$WANDB_PROJECT")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
  WARGS+=(--wandb_entity "$WANDB_ENTITY")
fi
if [[ -n "$WANDB_GROUP" ]]; then
  WARGS+=(--wandb_group "$WANDB_GROUP")
fi
if [[ -n "$WANDB_TAGS" ]]; then
  WARGS+=(--wandb_tags "$WANDB_TAGS")
fi
if [[ -n "$WANDB_MODE" ]]; then
  WARGS+=(--wandb_mode "$WANDB_MODE")
fi
if [[ -n "$WANDB_NAME_MODE" ]]; then
  WARGS+=(--wandb_name_mode "$WANDB_NAME_MODE")
fi

run_stage sync_assets "${SYNC_ARGS[@]}"

run_stage prepare_data \
  "$PYTHON_BIN" scripts/prepare_rotmnist_lowdata.py

run_stage train_missing \
  "$PYTHON_BIN" scripts/train_missing_low_data_models.py \
    --materialize_aliases \
    "${WARGS[@]}"

run_stage evaluate_all \
  "$PYTHON_BIN" scripts/run_low_data_benchmark.py \
    --output_dir "${RESULTS_ROOT}/low_data_benchmark/${RUN_ID}" \
    "${WARGS[@]}"

AGG_CMD="$PYTHON_BIN scripts/sync_assets.py --materialize_aliases"
if [[ "$ALLOW_WANDB_ASSETS" == "1" ]]; then
  AGG_CMD+=" --allow_wandb"
fi
AGG_CMD+=" && $PYTHON_BIN scripts/verify_assets.py"

run_stage aggregate_report \
  /usr/bin/bash -lc "$AGG_CMD"

echo "[sbatch_low_data_full_study] complete. run_dir=${RUN_DIR}"
