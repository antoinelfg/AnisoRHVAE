#!/usr/bin/env bash
#SBATCH --job-name=ellipses_lowdata_l2
#SBATCH --output=logs/ellipses_lowdata_l2_%j.out
#SBATCH --error=logs/ellipses_lowdata_l2_%j.err
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
source /scratch/alaforgu/miniconda3/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

# ---------------------------------------------------------------------------
# Dataset preparation (ellipses -> low-data format)
# ---------------------------------------------------------------------------
PREPARE_DATA="${PREPARE_DATA:-1}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed/ellipses_lowdata_v1}"
DATASET_ID="${DATASET_ID:-ellipses_lowdata_v1}"
DATA_MANIFEST_OUT="${DATA_MANIFEST_OUT:-assets/manifests/data_manifest_ellipses_lowdata.json}"
DATA_SEED="${DATA_SEED:-13}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1600}"
SEQ_LEN="${SEQ_LEN:-8}"
FRAME_MODE="${FRAME_MODE:-t0}"
MAX_FRAMES="${MAX_FRAMES:-}"

# ---------------------------------------------------------------------------
# Training sweep
# ---------------------------------------------------------------------------
MODELS="${MODELS:-vanilla_vae rhvae_standard aniso}"
SUBSET_NS="${SUBSET_NS:-50 100 500 1000}"
SUBSET_SEEDS="${SUBSET_SEEDS:-13}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/low_data_models/ellipses_lowdata_v1_latent2}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LATENT_DIM="${LATENT_DIM:-2}"
MATERIALIZE_ALIASES="${MATERIALIZE_ALIASES:-0}"
QUICK="${QUICK:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-1}"

# Benchmark (post-training metrics: FID/PRD/interp/augmentation + stats)
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
BENCH_OUTPUT_DIR="${BENCH_OUTPUT_DIR:-results/low_data_benchmark_ellipses_latent2}"
BENCH_N_GEN_SAMPLES="${BENCH_N_GEN_SAMPLES:-2000}"
BENCH_N_INTERP_PAIRS="${BENCH_N_INTERP_PAIRS:-64}"
BENCH_INTERP_STEPS="${BENCH_INTERP_STEPS:-32}"
BENCH_AUG_SYNTH_SAMPLES="${BENCH_AUG_SYNTH_SAMPLES:-500}"
BENCH_BOOTSTRAP_SAMPLES="${BENCH_BOOTSTRAP_SAMPLES:-1000}"
BENCH_RHVAE_SAMPLER_NAME="${BENCH_RHVAE_SAMPLER_NAME:-volume_riemannian}"
BENCH_RHVAE_SAMPLER_MCMC_STEPS="${BENCH_RHVAE_SAMPLER_MCMC_STEPS:-60}"
BENCH_RHVAE_SAMPLER_N_LF="${BENCH_RHVAE_SAMPLER_N_LF:-10}"
BENCH_RHVAE_SAMPLER_EPS_LF="${BENCH_RHVAE_SAMPLER_EPS_LF:-0.03}"
BENCH_RHVAE_SAMPLER_VOLUME_POWER="${BENCH_RHVAE_SAMPLER_VOLUME_POWER:-2.0}"
BENCH_RHVAE_SAMPLER_USE_DUAL_METRIC="${BENCH_RHVAE_SAMPLER_USE_DUAL_METRIC:-1}"
BENCH_RHVAE_SAMPLER_FP_STEPS="${BENCH_RHVAE_SAMPLER_FP_STEPS:-15}"
BENCH_RHVAE_SAMPLER_FP_DAMPING="${BENCH_RHVAE_SAMPLER_FP_DAMPING:-0.72}"
BENCH_RHVAE_SAMPLER_MOMENTUM_PERSIST="${BENCH_RHVAE_SAMPLER_MOMENTUM_PERSIST:-0.0}"
BENCH_RHVAE_SAMPLER_RADIAL_PRIOR_WEIGHT="${BENCH_RHVAE_SAMPLER_RADIAL_PRIOR_WEIGHT:-0.1}"
BENCH_RHVAE_SAMPLER_ADAPTIVE_DUAL_STEP="${BENCH_RHVAE_SAMPLER_ADAPTIVE_DUAL_STEP:-1}"
BENCH_RHVAE_SAMPLER_ADAPTIVE_MAX_DISP="${BENCH_RHVAE_SAMPLER_ADAPTIVE_MAX_DISP:-0.09}"
BENCH_RHVAE_SAMPLER_ADAPTIVE_MIN_SCALE="${BENCH_RHVAE_SAMPLER_ADAPTIVE_MIN_SCALE:-0.05}"

# Post-run visuals
RUN_POST_VISUALS="${RUN_POST_VISUALS:-1}"
THREE_ZONE_N="${THREE_ZONE_N:-1000}"
THREE_ZONE_SEED="${THREE_ZONE_SEED:-13}"
THREE_ZONE_OUTPUT_DIR="${THREE_ZONE_OUTPUT_DIR:-results/rhmc_three_zone_ellipses_latent2}"
THREE_ZONE_RADIAL_PRIOR_WEIGHT="${THREE_ZONE_RADIAL_PRIOR_WEIGHT:-0.1}"
THREE_ZONE_VOLUME_POWER="${THREE_ZONE_VOLUME_POWER:-2.0}"
THREE_ZONE_STEPS="${THREE_ZONE_STEPS:-70}"
THREE_ZONE_N_LF_INNER="${THREE_ZONE_N_LF_INNER:-10}"
THREE_ZONE_EPS="${THREE_ZONE_EPS:-0.03}"
THREE_ZONE_FP_STEPS="${THREE_ZONE_FP_STEPS:-15}"
THREE_ZONE_FP_DAMPING="${THREE_ZONE_FP_DAMPING:-0.72}"
THREE_ZONE_ADAPTIVE_MAX_DISP="${THREE_ZONE_ADAPTIVE_MAX_DISP:-0.09}"
THREE_ZONE_ADAPTIVE_MIN_SCALE="${THREE_ZONE_ADAPTIVE_MIN_SCALE:-0.05}"
CHAIN_PANEL_OUTPUT_DIR="${CHAIN_PANEL_OUTPUT_DIR:-results/latent_chain_panels_ellipses_latent2}"

# RHVAE/Aniso specifics
RHVAE_STANDARD_BACKEND="${RHVAE_STANDARD_BACKEND:-run_with_config}"
RHVAE_STANDARD_NUM_SEQUENCES_MODE="${RHVAE_STANDARD_NUM_SEQUENCES_MODE:-from_n}"
RHVAE_STANDARD_NUM_SEQUENCES="${RHVAE_STANDARD_NUM_SEQUENCES:-200}"
RHVAE_STANDARD_MAX_FRAMES_MODE="${RHVAE_STANDARD_MAX_FRAMES_MODE:-n}"
RHVAE_STANDARD_MAX_FRAMES="${RHVAE_STANDARD_MAX_FRAMES:-3000}"
RHVAE_STANDARD_FRAME_MODE="${RHVAE_STANDARD_FRAME_MODE:-t0}"
RHVAE_STANDARD_LR="${RHVAE_STANDARD_LR:-0.0005}"
RHVAE_STANDARD_N_CENTROIDS="${RHVAE_STANDARD_N_CENTROIDS:-100}"
RHVAE_STANDARD_TEMPERATURE="${RHVAE_STANDARD_TEMPERATURE:-0.5}"
RHVAE_STANDARD_REGULARIZATION="${RHVAE_STANDARD_REGULARIZATION:-0.01}"
RHVAE_STANDARD_ANALYSIS_SAMPLER="${RHVAE_STANDARD_ANALYSIS_SAMPLER:-volume}"
RHVAE_STANDARD_SAMPLING_FID_SAMPLES="${RHVAE_STANDARD_SAMPLING_FID_SAMPLES:-1000}"
RHVAE_STANDARD_SAMPLING_QUALITY_SAMPLES="${RHVAE_STANDARD_SAMPLING_QUALITY_SAMPLES:-500}"
RHVAE_STANDARD_SAMPLING_N_CHAINS="${RHVAE_STANDARD_SAMPLING_N_CHAINS:-4}"
RHVAE_STANDARD_SAMPLING_CHAIN_LENGTH="${RHVAE_STANDARD_SAMPLING_CHAIN_LENGTH:-100}"
RHVAE_STANDARD_SKIP_SAMPLING_DIAGNOSTICS="${RHVAE_STANDARD_SKIP_SAMPLING_DIAGNOSTICS:-0}"
RHVAE_ANISO_PROFILE="${RHVAE_ANISO_PROFILE:-core4_spatial}"
RHVAE_ANISO_BACKEND="${RHVAE_ANISO_BACKEND:-run_with_config}"
RHVAE_ANISO_NUM_SEQUENCES_MODE="${RHVAE_ANISO_NUM_SEQUENCES_MODE:-from_n}"
RHVAE_ANISO_NUM_SEQUENCES="${RHVAE_ANISO_NUM_SEQUENCES:-200}"
RHVAE_ANISO_MAX_FRAMES_MODE="${RHVAE_ANISO_MAX_FRAMES_MODE:-n}"
RHVAE_ANISO_MAX_FRAMES="${RHVAE_ANISO_MAX_FRAMES:-3000}"
RHVAE_ANISO_FRAME_MODE="${RHVAE_ANISO_FRAME_MODE:-t0}"
RHVAE_ANISO_LR="${RHVAE_ANISO_LR:-0.0005}"
RHVAE_ANISO_N_CENTROIDS="${RHVAE_ANISO_N_CENTROIDS:-100}"
RHVAE_ANISO_TEMPERATURE="${RHVAE_ANISO_TEMPERATURE:-0.7960961952783896}"
RHVAE_ANISO_REGULARIZATION="${RHVAE_ANISO_REGULARIZATION:-0.6}"
RHVAE_ANISO_ANALYSIS_SAMPLER="${RHVAE_ANISO_ANALYSIS_SAMPLER:-volume}"
RHVAE_ANISO_SAMPLING_FID_SAMPLES="${RHVAE_ANISO_SAMPLING_FID_SAMPLES:-1000}"
RHVAE_ANISO_SAMPLING_QUALITY_SAMPLES="${RHVAE_ANISO_SAMPLING_QUALITY_SAMPLES:-500}"
RHVAE_ANISO_SAMPLING_N_CHAINS="${RHVAE_ANISO_SAMPLING_N_CHAINS:-4}"
RHVAE_ANISO_SAMPLING_CHAIN_LENGTH="${RHVAE_ANISO_SAMPLING_CHAIN_LENGTH:-100}"
RHVAE_ANISO_SKIP_SAMPLING_DIAGNOSTICS="${RHVAE_ANISO_SKIP_SAMPLING_DIAGNOSTICS:-0}"
RHVAE_DROP_LAST="${RHVAE_DROP_LAST:-auto}"
RHVAE_AUTO_TEMPERATURE="${RHVAE_AUTO_TEMPERATURE:-1}"
RHVAE_AUTO_TEMPERATURE_STAT="${RHVAE_AUTO_TEMPERATURE_STAT:-mean_nn}"
RHVAE_AUTO_TEMPERATURE_EVERY="${RHVAE_AUTO_TEMPERATURE_EVERY:-1}"
RHVAE_TEMPERATURE_SCALE="${RHVAE_TEMPERATURE_SCALE:-1.0}"
RHVAE_PLOT_HEATMAPS_DURING_TRAINING="${RHVAE_PLOT_HEATMAPS_DURING_TRAINING:-1}"
RHVAE_VIS_EVERY="${RHVAE_VIS_EVERY:-5}"
RHVAE_METRIC_GRID_RES="${RHVAE_METRIC_GRID_RES:-40}"
RHVAE_METRIC_TISSOT_GRID_RES="${RHVAE_METRIC_TISSOT_GRID_RES:-16}"

# W&B
WANDB_PROJECT="${WANDB_PROJECT:-RHVAE-baseline}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-ellipses_lowdata_latent2}"
WANDB_TAGS="${WANDB_TAGS:-ellipses,low-data,latent2,vanilla,rhvae,aniso}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_NAME_MODE="${WANDB_NAME_MODE:-auto}"

read -r -a NS_ARR <<< "${SUBSET_NS}"
read -r -a SEED_ARR <<< "${SUBSET_SEEDS}"
read -r -a MODEL_ARR <<< "${MODELS}"

echo "=========================================="
echo "Ellipses Low-Data Latent=2 Study"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-local}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Time: $(date)"
echo "processed_dir=${PROCESSED_DIR}"
echo "subset_ns=${SUBSET_NS}"
echo "subset_seeds=${SUBSET_SEEDS}"
echo "models=${MODELS}"
echo "epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, latent_dim=${LATENT_DIM}"
echo "aniso_profile=${RHVAE_ANISO_PROFILE}"
echo "aniso_backend=${RHVAE_ANISO_BACKEND}"
echo "wandb=${WANDB_PROJECT}/${WANDB_ENTITY:-<default>} group=${WANDB_GROUP}"
echo "=========================================="

if [[ "${PREPARE_DATA}" == "1" ]]; then
  PREP_CMD=(
    "${PYTHON_BIN}" scripts/prepare_ellipses_lowdata.py
    --processed_dir "${PROCESSED_DIR}"
    --dataset_id "${DATASET_ID}"
    --num_sequences "${NUM_SEQUENCES}"
    --seq_len "${SEQ_LEN}"
    --frame_mode "${FRAME_MODE}"
    --seed "${DATA_SEED}"
    --subset_ns "${NS_ARR[@]}"
    --subset_seeds "${SEED_ARR[@]}"
    --data_manifest_out "${DATA_MANIFEST_OUT}"
  )
  if [[ -n "${MAX_FRAMES}" ]]; then
    PREP_CMD+=(--max_frames "${MAX_FRAMES}")
  fi
  echo "[ellipses_lowdata_l2] prep cmd: ${PREP_CMD[*]}"
  "${PREP_CMD[@]}"
fi

TRAIN_CMD=(
  "${PYTHON_BIN}" scripts/train_missing_low_data_models.py
  --models "${MODEL_ARR[@]}"
  --processed_dir "${PROCESSED_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --subset_ns "${NS_ARR[@]}"
  --subset_seeds "${SEED_ARR[@]}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --latent_dim "${LATENT_DIM}"
  --rhvae_standard_backend "${RHVAE_STANDARD_BACKEND}"
  --rhvae_standard_num_sequences_mode "${RHVAE_STANDARD_NUM_SEQUENCES_MODE}"
  --rhvae_standard_num_sequences "${RHVAE_STANDARD_NUM_SEQUENCES}"
  --rhvae_standard_max_frames_mode "${RHVAE_STANDARD_MAX_FRAMES_MODE}"
  --rhvae_standard_max_frames "${RHVAE_STANDARD_MAX_FRAMES}"
  --rhvae_standard_frame_mode "${RHVAE_STANDARD_FRAME_MODE}"
  --rhvae_standard_lr "${RHVAE_STANDARD_LR}"
  --rhvae_standard_n_centroids "${RHVAE_STANDARD_N_CENTROIDS}"
  --rhvae_standard_temperature "${RHVAE_STANDARD_TEMPERATURE}"
  --rhvae_standard_regularization "${RHVAE_STANDARD_REGULARIZATION}"
  --rhvae_standard_analysis_sampler "${RHVAE_STANDARD_ANALYSIS_SAMPLER}"
  --rhvae_standard_sampling_fid_samples "${RHVAE_STANDARD_SAMPLING_FID_SAMPLES}"
  --rhvae_standard_sampling_quality_samples "${RHVAE_STANDARD_SAMPLING_QUALITY_SAMPLES}"
  --rhvae_standard_sampling_n_chains "${RHVAE_STANDARD_SAMPLING_N_CHAINS}"
  --rhvae_standard_sampling_chain_length "${RHVAE_STANDARD_SAMPLING_CHAIN_LENGTH}"
  --rhvae_aniso_profile "${RHVAE_ANISO_PROFILE}"
  --rhvae_aniso_backend "${RHVAE_ANISO_BACKEND}"
  --rhvae_aniso_num_sequences_mode "${RHVAE_ANISO_NUM_SEQUENCES_MODE}"
  --rhvae_aniso_num_sequences "${RHVAE_ANISO_NUM_SEQUENCES}"
  --rhvae_aniso_max_frames_mode "${RHVAE_ANISO_MAX_FRAMES_MODE}"
  --rhvae_aniso_max_frames "${RHVAE_ANISO_MAX_FRAMES}"
  --rhvae_aniso_frame_mode "${RHVAE_ANISO_FRAME_MODE}"
  --rhvae_aniso_lr "${RHVAE_ANISO_LR}"
  --rhvae_aniso_n_centroids "${RHVAE_ANISO_N_CENTROIDS}"
  --rhvae_aniso_temperature "${RHVAE_ANISO_TEMPERATURE}"
  --rhvae_aniso_regularization "${RHVAE_ANISO_REGULARIZATION}"
  --rhvae_aniso_analysis_sampler "${RHVAE_ANISO_ANALYSIS_SAMPLER}"
  --rhvae_aniso_sampling_fid_samples "${RHVAE_ANISO_SAMPLING_FID_SAMPLES}"
  --rhvae_aniso_sampling_quality_samples "${RHVAE_ANISO_SAMPLING_QUALITY_SAMPLES}"
  --rhvae_aniso_sampling_n_chains "${RHVAE_ANISO_SAMPLING_N_CHAINS}"
  --rhvae_aniso_sampling_chain_length "${RHVAE_ANISO_SAMPLING_CHAIN_LENGTH}"
  --rhvae_drop_last "${RHVAE_DROP_LAST}"
  --rhvae_vis_every "${RHVAE_VIS_EVERY}"
  --rhvae_metric_grid_res "${RHVAE_METRIC_GRID_RES}"
  --rhvae_metric_tissot_grid_res "${RHVAE_METRIC_TISSOT_GRID_RES}"
  --wandb_name_mode "${WANDB_NAME_MODE}"
)

if [[ "${MATERIALIZE_ALIASES}" == "1" ]]; then
  TRAIN_CMD+=(--materialize_aliases)
fi
if [[ "${RHVAE_AUTO_TEMPERATURE}" == "1" ]]; then
  TRAIN_CMD+=(
    --rhvae_auto_temperature
    --rhvae_auto_temperature_stat "${RHVAE_AUTO_TEMPERATURE_STAT}"
    --rhvae_auto_temperature_every "${RHVAE_AUTO_TEMPERATURE_EVERY}"
    --rhvae_temperature_scale "${RHVAE_TEMPERATURE_SCALE}"
  )
fi
if [[ "${RHVAE_PLOT_HEATMAPS_DURING_TRAINING}" == "1" ]]; then
  TRAIN_CMD+=(--rhvae_plot_heatmaps_during_training)
fi
if [[ "${RHVAE_STANDARD_SKIP_SAMPLING_DIAGNOSTICS}" == "1" ]]; then
  TRAIN_CMD+=(--rhvae_standard_skip_sampling_diagnostics)
fi
if [[ "${RHVAE_ANISO_SKIP_SAMPLING_DIAGNOSTICS}" == "1" ]]; then
  TRAIN_CMD+=(--rhvae_aniso_skip_sampling_diagnostics)
fi
if [[ "${QUICK}" == "1" ]]; then
  TRAIN_CMD+=(--quick)
fi
if [[ "${FORCE_RETRAIN}" == "1" ]]; then
  TRAIN_CMD+=(--force_retrain)
fi

if [[ -n "${WANDB_PROJECT}" ]]; then
  TRAIN_CMD+=(--wandb_project "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  TRAIN_CMD+=(--wandb_entity "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_GROUP}" ]]; then
  TRAIN_CMD+=(--wandb_group "${WANDB_GROUP}")
fi
if [[ -n "${WANDB_TAGS}" ]]; then
  TRAIN_CMD+=(--wandb_tags "${WANDB_TAGS}")
fi
if [[ -n "${WANDB_MODE}" ]]; then
  export WANDB_MODE
  TRAIN_CMD+=(--wandb_mode "${WANDB_MODE}")
fi

echo "[ellipses_lowdata_l2] train cmd: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

if [[ "${RUN_BENCHMARK}" == "1" ]]; then
  BENCH_CMD=(
    "${PYTHON_BIN}" scripts/run_low_data_benchmark.py
    --processed_dir "${PROCESSED_DIR}"
    --model_root "${OUTPUT_ROOT}"
    --models "${MODEL_ARR[@]}"
    --subset_ns "${NS_ARR[@]}"
    --subset_seeds "${SEED_ARR[@]}"
    --n_gen_samples "${BENCH_N_GEN_SAMPLES}"
    --n_interp_pairs "${BENCH_N_INTERP_PAIRS}"
    --interp_steps "${BENCH_INTERP_STEPS}"
    --aug_synth_samples "${BENCH_AUG_SYNTH_SAMPLES}"
    --bootstrap_samples "${BENCH_BOOTSTRAP_SAMPLES}"
    --rhvae_sampler_name "${BENCH_RHVAE_SAMPLER_NAME}"
    --rhvae_sampler_mcmc_steps "${BENCH_RHVAE_SAMPLER_MCMC_STEPS}"
    --rhvae_sampler_n_lf "${BENCH_RHVAE_SAMPLER_N_LF}"
    --rhvae_sampler_eps_lf "${BENCH_RHVAE_SAMPLER_EPS_LF}"
    --rhvae_sampler_volume_power "${BENCH_RHVAE_SAMPLER_VOLUME_POWER}"
    --rhvae_sampler_fp_steps "${BENCH_RHVAE_SAMPLER_FP_STEPS}"
    --rhvae_sampler_fp_damping "${BENCH_RHVAE_SAMPLER_FP_DAMPING}"
    --rhvae_sampler_momentum_persist "${BENCH_RHVAE_SAMPLER_MOMENTUM_PERSIST}"
    --rhvae_sampler_radial_prior_weight "${BENCH_RHVAE_SAMPLER_RADIAL_PRIOR_WEIGHT}"
    --rhvae_sampler_adaptive_max_dual_displacement "${BENCH_RHVAE_SAMPLER_ADAPTIVE_MAX_DISP}"
    --rhvae_sampler_adaptive_min_step_scale "${BENCH_RHVAE_SAMPLER_ADAPTIVE_MIN_SCALE}"
    --output_dir "${BENCH_OUTPUT_DIR}"
    --wandb_name_mode "${WANDB_NAME_MODE}"
  )
  if [[ "${BENCH_RHVAE_SAMPLER_USE_DUAL_METRIC}" == "1" ]]; then
    BENCH_CMD+=(--rhvae_sampler_use_dual_metric)
  fi
  if [[ "${BENCH_RHVAE_SAMPLER_ADAPTIVE_DUAL_STEP}" == "1" ]]; then
    BENCH_CMD+=(--rhvae_sampler_adaptive_dual_step)
  fi
  if [[ "${QUICK}" == "1" ]]; then
    BENCH_CMD+=(--quick)
  fi
  if [[ -n "${WANDB_PROJECT}" ]]; then
    BENCH_CMD+=(--wandb_project "${WANDB_PROJECT}")
  fi
  if [[ -n "${WANDB_ENTITY}" ]]; then
    BENCH_CMD+=(--wandb_entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_GROUP}" ]]; then
    BENCH_CMD+=(--wandb_group "${WANDB_GROUP}_benchmark")
  fi
  if [[ -n "${WANDB_TAGS}" ]]; then
    BENCH_CMD+=(--wandb_tags "${WANDB_TAGS},benchmark")
  fi
  if [[ -n "${WANDB_MODE}" ]]; then
    BENCH_CMD+=(--wandb_mode "${WANDB_MODE}")
  fi
  echo "[ellipses_lowdata_l2] benchmark cmd: ${BENCH_CMD[*]}"
  "${BENCH_CMD[@]}"
fi

if [[ "${RUN_POST_VISUALS}" == "1" ]]; then
  N_PAD="$(printf 'N%03d' "${THREE_ZONE_N}")"
  SEED_PAD="$(printf 'seed%03d' "${THREE_ZONE_SEED}")"

  for MID in rhvae_standard aniso; do
    BASE_DIR="${OUTPUT_ROOT}/${MID}/${N_PAD}/${SEED_PAD}"
    RUN_DIR=""
    if [[ -d "${BASE_DIR}" ]]; then
      RUN_DIR="$(ls -1dt "${BASE_DIR}"/* 2>/dev/null | head -n1 || true)"
    fi
    if [[ -z "${RUN_DIR}" ]]; then
      echo "[ellipses_lowdata_l2] WARN: no run dir found for ${MID} at ${BASE_DIR}, skipping three-zones."
      continue
    fi
    TZ_CMD=(
      "${PYTHON_BIN}" scripts/quick/visualize_rhmc_three_zones_real_metric.py
      --model_path "${RUN_DIR}"
      --output_dir "${THREE_ZONE_OUTPUT_DIR}/${MID}"
      --seed "${THREE_ZONE_SEED}"
      --sampler_name volume_riemannian
      --volume_power "${THREE_ZONE_VOLUME_POWER}"
      --steps "${THREE_ZONE_STEPS}"
      --n_lf_inner "${THREE_ZONE_N_LF_INNER}"
      --eps "${THREE_ZONE_EPS}"
      --fp_steps "${THREE_ZONE_FP_STEPS}"
      --fp_damping "${THREE_ZONE_FP_DAMPING}"
      --momentum_persist 0.0
      --radial_prior_weight "${THREE_ZONE_RADIAL_PRIOR_WEIGHT}"
      --manifold_points 3
      --near_points 3
      --far_points 3
      --near_multiplier 1.2
      --far_multiplier 2.4
      --max_radius 45.0
      --grid_n 180
      --void_eigshape_mode det_preserving_spectral
      --void_eigshape_alpha_min 0.8
      --void_eigshape_power -1.2
      --void_eigshape_eig_floor 1e-8
      --use_dual_metric
      --adaptive_dual_step
      --adaptive_max_dual_displacement "${THREE_ZONE_ADAPTIVE_MAX_DISP}"
      --adaptive_min_step_scale "${THREE_ZONE_ADAPTIVE_MIN_SCALE}"
    )
    echo "[ellipses_lowdata_l2] three-zones cmd (${MID}): ${TZ_CMD[*]}"
    "${TZ_CMD[@]}"
  done

  # Generic latent chain panel including vanilla (not Riemannian three-zone, but comparable latent-chain diagnostics).
  PANEL_CMD=(
    "${PYTHON_BIN}" scripts/quick/plot_logdet_and_rhmc_chains.py
    --processed_dir "${PROCESSED_DIR}"
    --model_root "${OUTPUT_ROOT}"
    --models vanilla_vae rhvae_standard aniso
    --subset_n "${THREE_ZONE_N}"
    --subset_seed "${THREE_ZONE_SEED}"
    --manifold_starts 3
    --near_starts 3
    --far_starts 3
    --near_sigma_factor 1.2
    --far_scale 2.4
    --chain_steps 80
    --chain_n_lf 20
    --chain_eps_lf 0.02
    --output_dir "${CHAIN_PANEL_OUTPUT_DIR}"
  )
  echo "[ellipses_lowdata_l2] chain-panel cmd: ${PANEL_CMD[*]}"
  "${PANEL_CMD[@]}"
fi
echo "[ellipses_lowdata_l2] complete"
