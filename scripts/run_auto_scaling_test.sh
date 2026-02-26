#!/bin/bash
set -euo pipefail

export WANDB_MODE=online
export WANDB_PROJECT="AnisoRHVAE-AutoScaling"

cd /home/alaforgu/scratch/longitudinal_experiments/AnisoRHVAE

echo "Launching Multi-Regime Training (N=50, 100, 1000) with Option C: The k-NN Trick, requesting GPU via srun..."

for N in 50 100 1000; do
  if [ "$N" -eq 50 ]; then
    CENTROIDS=12
  elif [ "$N" -eq 100 ]; then
    CENTROIDS=15
  elif [ "$N" -eq 1000 ]; then
    CENTROIDS=100
  fi

  echo "Training N=$N with n_centroids=$CENTROIDS..."
  srun -p gpu --gpus=1 python scripts/train_missing_low_data_models.py \
    --models aniso \
    --processed_dir data/processed/ellipses_lowdata_v1 \
    --output_root outputs/low_data_models/ellipses_auto_scaling_v5 \
    --subset_ns "$N" \
    --subset_seeds 13 \
    --epochs 100 \
    --batch_size 64 \
    --latent_dim 2 \
    --rhvae_aniso_backend run_with_config \
    --rhvae_aniso_profile core4_spatial \
    --rhvae_aniso_num_sequences_mode from_n \
    --rhvae_aniso_max_frames_mode n \
    --rhvae_aniso_frame_mode t0 \
    --rhvae_aniso_lr 0.001 \
    --rhvae_aniso_regularization 0.01 \
    --rhvae_aniso_n_centroids "$CENTROIDS" \
    --rhvae_aniso_analysis_sampler volume \
    --rhvae_auto_temperature \
    --rhvae_auto_temperature_stat mean_knn_5 \
    --rhvae_auto_temperature_every 10 \
    --wandb_project $WANDB_PROJECT \
    --force_retrain
done

echo "Training completed."
