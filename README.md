# AnisoRHVAE

## Overview
Anisotropic Riemannian Metric VAE for robust optimization and generation in data-scarce regimes. This repository introduces a geometry learning framework that balances on-manifold precision with off-manifold attraction (the "3-zone" metric).

- **Core Contribution**: Anisotropic Riemannian Metric (`VolumeElementRiemannianHMCSampler`).
- **Framework**: Extends Pythae with custom metric learning, Riemannian HMC, and extensive diagnostic suites.
- **Key Files**:
  - `src/models/rhvae_geometry.py`: Core metric and geometry definitions.
  - `src/models/samplers/hmc_sampler.py`: Riemannian HMC implementations.

## Reproducing Low-Data Benchmarks

To reproduce the low-data benchmark experiments (e.g., on RotMNIST):

1. **Prepare Data and Assets**:
   ```bash
   python scripts/sync_assets.py --materialize_aliases
   python scripts/prepare_rotmnist_lowdata.py
   ```

2. **Train Models**:
   ```bash
   python scripts/train_missing_low_data_models.py --materialize_aliases
   ```

3. **Run Evaluation**:
   ```bash
   python scripts/run_low_data_benchmark.py
   python scripts/verify_assets.py
   ```

*(Alternatively, use the SLURM one-shot launcher: `sbatch scripts/sbatch_low_data_full_study.sh`)*

## Reproducing 3-Zone Diagnostics

To visualize the 3-zone metric behavior and run diagnostics on toy datasets:

1. **Prepare 2D Dataset**:
   ```bash
   python scripts/prepare_toy2d_lowdata.py --dataset moons \
     --processed_dir data/processed/toy2d/moons_v1 --standardize \
     --subset_ns 50 100 500
   ```

2. **Train AnisoRHVAE with 3-Zone Metric**:
   ```bash
   python scripts/train_rhvae_tensor.py \
     --mode aniso --processed_dir data/processed/toy2d/moons_v1 \
     --n 50 --seed 13 --latent_dim 2 --epochs 60 \
     --auto_temperature --auto_temperature_stat mean_nn --auto_temperature_every 1 \
     --plot_heatmaps_during_training --vis_every 5
   ```

3. **Run Full Metric Analysis**:
   ```bash
   python scripts/analyze_metric_full.py \
     --model_path outputs/rhvae_tensor/<RUN_FOLDER> \
     --output_dir outputs/analysis_results
   ```
