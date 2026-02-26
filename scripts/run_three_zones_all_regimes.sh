#!/bin/bash
set -euo pipefail

cd /home/alaforgu/scratch/longitudinal_experiments/AnisoRHVAE

echo "Running enhanced RHMC Three-Zone diagnostics on all v6 models..."

for N in 50 100 1000; do
  NPAD=$(printf '%03d' "$N")
  MODEL_DIR=$(ls -td outputs/low_data_models/ellipses_auto_scaling_v6/aniso/N${NPAD}/seed013/* 2>/dev/null | head -1)

  if [ -z "$MODEL_DIR" ]; then
    echo "⚠️  N=$N: no model found, skipping."
    continue
  fi

  echo "━━━ N=$N ━━━  model=$MODEL_DIR"

  python scripts/quick/visualize_rhmc_three_zones_real_metric.py \
    --model_path "$MODEL_DIR" \
    --output_dir "results/rhmc_three_zone_auto_scaling_v6/N${NPAD}" \
    --sampler_name volume_riemannian \
    --steps 200 \
    --n_lf_inner 6 \
    --eps 0.02 \
    --volume_power 2.0 \
    --radial_prior_weight 0.1 \
    --manifold_points 3 \
    --near_points 1 \
    --far_points 1 \
    --near_multiplier 1.20 \
    --far_multiplier 2.40 \
    --grid_n 180 \
    --seed 13 \
    --void_eigshape_mode det_preserving_spectral \
    --void_eigshape_alpha_min 0.8 \
    --void_eigshape_power -1.2 \
    --void_eigshape_eig_floor 1e-8 \
    --use_dual_metric \
    --adaptive_dual_step \
    --adaptive_max_dual_displacement 0.09 \
    --adaptive_min_step_scale 0.05

  echo "✅ N=$N complete."
  echo ""
done

echo "All regimes processed."
