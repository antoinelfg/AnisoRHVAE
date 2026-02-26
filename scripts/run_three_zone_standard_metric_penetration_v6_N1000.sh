#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" scripts/quick/visualize_rhmc_three_zones_real_metric.py \
  --model_path outputs/low_data_models/ellipses_auto_scaling_v6/aniso/N1000/seed013/2026-02-20_18-06-11 \
  --sampler_name volume_riemannian \
  --use_dual_metric False \
  --atom_scale 100.0 \
  --volume_power 2.0 \
  --radial_prior_weight 0.1 \
  --steps 500 \
  --n_lf_inner 10 \
  --eps 0.05 \
  --adaptive_dual_step \
  --adaptive_max_dual_displacement 0.05 \
  --fp_steps 15 \
  --void_eigshape_mode none \
  --output_dir results/rhmc_standard_metric_N1000_penetration \
  "$@"
