#!/bin/bash
set -euo pipefail

export WANDB_MODE=online
export WANDB_PROJECT="AnisoRHVAE-AutoScaling"

cd /home/alaforgu/scratch/longitudinal_experiments/AnisoRHVAE

echo "Launching v7 Training: fewer centroids (10/20/100 for N=50/100/1000)..."

for N in 50 100 1000; do
  if [ "$N" -eq 50 ]; then
    CENTROIDS=10
    MAX_FRAMES=50
    NUM_SEQ=63
  elif [ "$N" -eq 100 ]; then
    CENTROIDS=20
    MAX_FRAMES=100
    NUM_SEQ=125
  elif [ "$N" -eq 1000 ]; then
    CENTROIDS=100
    MAX_FRAMES=1000
    NUM_SEQ=1250
  fi

  NPAD=$(printf '%03d' "$N")
  echo "━━━ Training N=$N with n_centroids=$CENTROIDS ━━━"

  srun -p gpu --gpus=1 python scripts/run_pythae_rhvae_baseline.py \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --seed 13 \
    --output_dir "outputs/low_data_models/ellipses_auto_scaling_v7/aniso/N${NPAD}/seed013" \
    --num_sequences "$NUM_SEQ" \
    --frame_mode t0 \
    --latent_dim 2 \
    --n_centroids "$CENTROIDS" \
    --max_centroids "$CENTROIDS" \
    --regularization 0.01 \
    --vis_every 10 \
    --analysis_rhmc_sampler volume \
    --sampling_fid_samples 1000 \
    --sampling_quality_samples 500 \
    --sampling_n_chains 4 \
    --sampling_chain_length 100 \
    --max_frames "$MAX_FRAMES" \
    --auto_temperature \
    --auto_temperature_stat mean_knn_5 \
    --auto_temperature_every 10 \
    --temperature_scale 1.0 \
    --rhvae_variant geometry \
    --use_attractor \
    --kernel_type mahalanobis \
    --atom_norm trace \
    --atom_power 1.0959864113997024 \
    --kernel_power 1.0 \
    --precision_jitter 0.01 \
    --attractor_smoothness soft \
    --attractor_metric mahalanobis \
    --attractor_use_det \
    --attractor_gamma 7.624554217806352 \
    --attractor_k_nearest 1 \
    --attractor_bias_energy 18.0 \
    --void_threshold 1.2 \
    --void_weight_threshold -1.0 \
    --void_decay_type invquad \
    --void_decay_scale 8.87131893173153 \
    --void_decay_power 1.7447710342008058 \
    --void_decay_softplus_k 5.0 \
    --radial_stretch 9.252860449508804 \
    --transition_steepness 7.276725930229776 \
    --void_eigshape_mode det_preserving_spectral \
    --void_eigshape_alpha_min 0.8 \
    --void_eigshape_power -1.2 \
    --void_eigshape_eig_floor 1e-8 \
    --wandb_name_mode auto \
    --wandb_run_name "low_data_aniso_v7_N${NPAD}_seed013"

  echo "✅ N=$N complete."
  echo ""
done

echo "All v7 models trained."
