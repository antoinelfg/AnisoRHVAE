#!/bin/bash
set -euo pipefail

cd /home/alaforgu/scratch/longitudinal_experiments/AnisoRHVAE

# The target N=1000 model trained with Option C in v5 output dir
MODEL_DIR=$(ls -td outputs/low_data_models/ellipses_auto_scaling_v5/aniso/N1000/seed013/* | head -1)

if [ -z "$MODEL_DIR" ]; then
    echo "N=1000 model not found. Sampler analysis cannot run yet."
    exit 1
fi

echo "Running Sampler Robustness Check on model: $MODEL_DIR"

# Launching Pythae visualize analysis script
srun -p gpu --gpus=1 python scripts/visualize_rhmc_three_zones_real_metric.py \
    --model_dir "$MODEL_DIR" \
    --data_dir data/processed/ellipses_lowdata_v1 \
    --num_points 100 \
    --burn_in 50 \
    --save_name "robustness_check_N1000.png" \
    --debug_output \
    --steps 200

echo "Sampler robustness check complete. Outputs dumped to $MODEL_DIR"
