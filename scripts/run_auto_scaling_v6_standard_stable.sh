#!/bin/bash
set -euo pipefail

export WANDB_MODE=online
export WANDB_PROJECT="RHVAE-benchmark"

cd /home/alaforgu/scratch/longitudinal_experiments/AnisoRHVAE

echo "Launching v6-style Benchmark with Standard Metric Stability Fixes..."

for N in 50 100 1000; do
  if [ "$N" -eq 50 ]; then
    MAX_FRAMES=50
    NUM_SEQ=63
  elif [ "$N" -eq 100 ]; then
    MAX_FRAMES=100
    NUM_SEQ=125
  elif [ "$N" -eq 1000 ]; then
    MAX_FRAMES=1000
    NUM_SEQ=1250
  fi
  
  CENTROIDS=100
  NPAD=$(printf '%03d' "$N")
  
  echo "━━━ Training N=$N with n_centroids=$CENTROIDS (v6 baseline) ━━━"

  srun -p gpu --gpus=1 python scripts/run_pythae_rhvae_baseline.py \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --seed 13 \
    --output_dir "outputs/pythae_baselines/standard_metric/v6_style/N${NPAD}" \
    --num_sequences "$NUM_SEQ" \
    --frame_mode t0 \
    --latent_dim 2 \
    --n_centroids "$CENTROIDS" \
    --max_centroids "$CENTROIDS" \
    --regularization 0.01 \
    --vis_every 10 \
    --auto_temperature \
    --auto_temperature_stat mean_knn_5 \
    --auto_temperature_every 10 \
    --temperature_scale 1.0 \
    --rhvae_variant geometry \
    --geometry_case aniso \
    --analysis_rhmc_sampler volume_riemannian \
    --sampling_fid_samplers gaussian volume_riemannian \
    --use_dual_metric False \
    --atom_scale 100.0 \
    --rhmc_integrator implicit \
    --rhmc_fp_steps 15 \
    --rhmc_adaptive_dual_step \
    --rhmc_adaptive_max_dual_displacement 0.05 \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "v6_standard_stable_N${NPAD}_seed013"

  echo "✅ N=$N complete."
  echo ""
done

echo "Benchmark completed."
