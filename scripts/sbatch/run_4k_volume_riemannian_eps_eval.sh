#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <prefilter|confirm> <eps1> [eps2 ...]"
  echo "Example: $0 prefilter 0.0185 0.0200 0.0220"
  exit 1
fi

MODE="$1"
shift

BASELINE_MODEL_PATH="${BASELINE_MODEL_PATH:-outputs/pythae_rhvae_baseline/2026-02-09_15-04-06}"
OURS_MODEL_PATH="${OURS_MODEL_PATH:-outputs/reference_models/4K}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/metric_assessment_eps_4k_volume_riemannian}"
DEVICE="${DEVICE:-cuda}"
WGROUP_BASE="${WGROUP_BASE:-metric_assessment_eps_4k_volume_riemannian}"

COMMON_ARGS=(
  --baseline_model_path "${BASELINE_MODEL_PATH}"
  --ours_model_path "${OURS_MODEL_PATH}"
  --profile smoke
  --protocols matched
  --aniso_only
  --skip_matched_pilot
  --sampler_name volume_riemannian
  --strict_volume_power 0.8
  --strict_n_lf 30
  --strict_fp_steps 8
  --strict_fp_damping 0.7129
  --device "${DEVICE}"
  --skip_fid
)

case "${MODE}" in
  prefilter)
    MODE_ARGS=(
      --sampling_seeds 13 29
      --allow_two_seed_prefilter
      --chain_length 20
      --sampling_mcmc_steps 20
      --coverage_samples 40
      --rescue_ring_starts 1
      --rescue_gaussian_starts 1
      --rescue_horizon 12
    )
    ;;
  confirm)
    MODE_ARGS=(
      --sampling_seeds 13 29 47
      --chain_length 40
      --sampling_mcmc_steps 40
      --coverage_samples 120
      --rescue_ring_starts 4
      --rescue_gaussian_starts 4
      --rescue_horizon 25
    )
    ;;
  *)
    echo "Invalid mode: ${MODE}. Expected prefilter or confirm."
    exit 1
    ;;
esac

mkdir -p "${OUTPUT_ROOT}/${MODE}"

for EPS in "$@"; do
  EPS_TAG="$(echo "${EPS}" | sed 's/-/m/g; s/\./p/g')"
  OUT_DIR="${OUTPUT_ROOT}/${MODE}/eps_${EPS_TAG}"
  mkdir -p "${OUT_DIR}"

  echo "============================================================"
  echo "Mode: ${MODE} | eps=${EPS}"
  echo "Output root: ${OUT_DIR}"
  echo "============================================================"

  python scripts/metric_assessment_suite.py \
    "${COMMON_ARGS[@]}" \
    "${MODE_ARGS[@]}" \
    --matched_eps_grid "${EPS}" \
    --output_dir "${OUT_DIR}" \
    --wandb_group "${WGROUP_BASE}_${MODE}" \
    --wandb_name_mode auto \
    ${EXTRA_ARGS:-}

done

echo "Done. Summarize with:"
echo "  python scripts/summarize_eps_metric_assessment.py --root ${OUTPUT_ROOT}/${MODE}"
