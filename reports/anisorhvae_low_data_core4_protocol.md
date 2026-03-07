# AnisoRHVAE Low-Data Core4 Protocol

## 1) Purpose

This document defines the full experimental protocol for the low-data AnisoRHVAE study using the `core4_spatial` geometry profile and strict sampler controls.

Primary goal:
- Produce a reviewer-grade comparison between `vanilla_vae`, `rhvae_standard`, and `aniso`.
- Isolate improvements due to:
1. Model and metric geometry.
2. Sampler dynamics.
3. Optional engineered bias via `radial_prior_weight`.

This protocol is both:
- A scientific protocol (what claims can be made).
- An execution runbook (what to run, where outputs are written, how to validate).


## 2) Scope And Claims

### In scope
- RotMNIST low-data study.
- Family comparison across multiple training set sizes.
- Geometry vs sampler ablation with strict sampler parity.
- Bias ablation (`radial_prior_weight = 0.0` vs `0.08`).
- Three-zone RHMC trajectory diagnostics with near/far starts.
- Optional ellipse-data sanity track with 2D latent geometry for direct visual inspection.

### Out of scope
- New model architecture design.
- Non-RotMNIST datasets.
- Hyperparameter sweeps outside fixed protocol defaults.

### Core claims this protocol is designed to test
1. Aniso geometry is more robust in low-data regimes than vanilla latent geometry.
2. Gains are not only from sampler tuning.
3. An engineered radial prior can help, but should not be the only reason for gains.


## 3) Canonical References (Source Of Truth)

Core4 geometry reference:
- `outputs/metric_core4_sweep/2026-02-17_11-07-08/rhvae_metric.pt`
- `configs/run_geometry_metric_core4_4k_seed13.yaml`

Main orchestration and evaluation scripts:
- `scripts/prepare_rotmnist_lowdata.py`
- `scripts/train_missing_low_data_models.py`
- `scripts/train_rhvae_tensor.py`
- `scripts/run_low_data_benchmark.py`
- `scripts/metric_assessment_suite.py`
- `scripts/quick/visualize_rhmc_three_zones_real_metric.py`
- `scripts/compare_sampler_assessment_runs.py`


## 4) Fixed Technical Configuration

## 4.1 Core4 spatial geometry for Aniso

The `core4_spatial` profile in `scripts/train_rhvae_tensor.py` is fixed to:

- `kernel_type = mahalanobis`
- `atom_norm = trace`
- `atom_power = 1.0959864113997024`
- `kernel_power = 1.0`
- `precision_jitter = 0.01`
- `void_threshold = 1.2`
- `void_weight_threshold = -1.0`
- `void_decay_type = invquad`
- `void_decay_scale = 8.87131893173153`
- `void_decay_power = 1.7447710342008058`
- `void_decay_softplus_k = 5.0`
- `radial_stretch = 9.252860449508804` (unless overridden)
- `transition_steepness = 7.276725930229776`
- `use_attractor = true`
- `attractor_smoothness = soft`
- `attractor_metric = mahalanobis`
- `attractor_use_det = true`
- `attractor_gamma = 7.624554217806352`
- `attractor_k_nearest = 1`
- `attractor_bias_energy = 18.0`

Note:
- `void_eigshape_*` in this profile is left at neutral training defaults (`mode=none`) to preserve parity with the selected metric artifact.
- Eigenshape is introduced explicitly during sampling diagnostics and strict RHMC analysis.

## 4.2 Eigenshape settings for RHMC diagnostics

- `void_eigshape_mode = det_preserving_spectral`
- `void_eigshape_alpha_min = 0.8`
- `void_eigshape_power = -1.2`
- `void_eigshape_eig_floor = 1e-8`

## 4.3 Strict sampler settings

For geometry-isolation comparisons:
- `sampler_name = volume_riemannian`
- `mass_mode = dual`
- `volume_power = 2.0`
- `n_lf = 10`
- `eps_lf = 0.05`
- `fp_steps = 15`
- `fp_damping = 0.72`
- `momentum_persist = 0.0`
- `adaptive_dual_step = false`
- `dynamic_jitter_scale = 0.0`

Bias ablation:
- `radial_prior_weight in {0.0, 0.1}`

Operational note:
- The strict settings above are the paper-facing exact parity settings for scientific isolation.
- Any run with `adaptive_dual_step = true` or `dynamic_jitter_scale > 0` belongs to the appendix/engineering track (`dual_safe`), not the main exact tables.

## 4.4 Ellipse operational command (validated)

The following command is an appendix-only `dual_safe` operating point for the ellipse baseline metric:

```bash
python scripts/quick/visualize_rhmc_three_zones_real_metric.py \
  --model_path outputs/metric_core4_sweep/2026-02-17_11-07-08 \
  --output_dir results/rhmc_three_zone_best_eigshape_plus \
  --seed 13 \
  --sampler_name volume_riemannian \
  --volume_power 2.0 \
  --steps 70 \
  --n_lf_inner 10 \
  --eps 0.05 \
  --eps_jitter 0.0 \
  --n_lf_jitter 0 \
  --fp_steps 15 \
  --fp_damping 0.72 \
  --momentum_persist 0.0 \
  --radial_prior_weight 0.1 \
  --only_outside_starts \
  --near_points 3 \
  --far_points 3 \
  --near_multiplier 1.2 \
  --far_multiplier 2.4 \
  --max_radius 45.0 \
  --grid_n 180 \
  --void_eigshape_mode det_preserving_spectral \
  --void_eigshape_alpha_min 0.8 \
  --void_eigshape_power -1.2 \
  --void_eigshape_eig_floor 1e-8 \
  --mass_mode dual \
  --adaptive_dual_step \
  --adaptive_max_dual_displacement 0.09 \
  --adaptive_min_step_scale 0.05 \
  --dynamic_jitter_scale 1e-5
```

Interpretation:
- This is a tuned operating point for this specific metric and data manifold.
- It is not an exact RHMC result because the adaptive dual step is active and dynamic jitter is non-zero.
- It should be reported separately from the main exact ablations.


## 5) Code-Level Implementation Changes

These changes are already implemented.

## 5.1 `scripts/train_rhvae_tensor.py`
- Added `--aniso_profile core4_spatial`.
- Added exact parameter mapping in `_build_model(...)`.

## 5.2 `scripts/train_missing_low_data_models.py`
- Added `--rhvae_aniso_profile`.
- Forwarded it to Aniso training calls (`train_rhvae_tensor.py --aniso_profile ...`).

## 5.3 `scripts/metric_assessment_suite.py`

Extended `SamplerConfig` with:
- `mass_mode`
- `adaptive_dual_step`
- `adaptive_max_dual_displacement`
- `adaptive_min_step_scale`
- `dynamic_jitter_scale`

Added strict CLI flags:
- `--strict_mass_mode`
- `--strict_adaptive_dual_step`
- `--strict_no_adaptive_dual_step`
- `--strict_adaptive_max_dual_displacement`
- `--strict_adaptive_min_step_scale`
- `--strict_dynamic_jitter_scale`

Propagated these fields through:
- `sampler_from_config(...)`
- matched pilot selection
- tuned pilot selection
- cloned strict/matched/tuned configs

Audit persistence:
- New fields are written into `protocol_configs.json` because config serialization uses `asdict(cfg)`.


## 6) End-To-End Execution Protocol

## 6.1 Step 1: Prepare low-data subsets

```bash
python scripts/prepare_rotmnist_lowdata.py \
  --subset_ns 50 100 500 5000 \
  --subset_seeds 13 29 47 71 89
```

Output location:
- `data/processed/rotmnist/v1`
- subset JSON files under `data/processed/rotmnist/v1/subsets`

## 6.2 Step 2: Train all model families

```bash
python scripts/train_missing_low_data_models.py \
  --models vanilla_vae rhvae_standard aniso ebm_conformal \
  --subset_ns 50 100 500 5000 \
  --subset_seeds 13 29 47 71 89 \
  --epochs 40 \
  --rhvae_aniso_profile core4_spatial \
  --rhvae_auto_temperature \
  --rhvae_auto_temperature_stat mean_nn \
  --rhvae_auto_temperature_every 1 \
  --materialize_aliases
```

Output root:
- `outputs/low_data_models/rotmnist_v1`

Expected model folder structure:
- `outputs/low_data_models/rotmnist_v1/<model_id>/N<nnn>/seed<sss>/<timestamp>/...`

## 6.3 Step 3: Family benchmark

```bash
python scripts/run_low_data_benchmark.py \
  --models vanilla_vae rhvae_standard aniso ebm_conformal \
  --subset_ns 50 100 500 5000 \
  --subset_seeds 13 29 47 71 89 \
  --n_gen_samples 2000 \
  --n_interp_pairs 64 \
  --interp_steps 32 \
  --aug_synth_samples 500 \
  --bootstrap_samples 1000 \
  --output_dir results/low_data_benchmark_aniso4c
```

Expected artifacts:
- `summary_by_model_n.json`
- `metrics_table.csv`
- `stat_tests.json`
- `figures/*`

## 6.4 Step 4: Geometry isolation with strict sampler parity

For each `N in {50,100,500,5000}`, use `seed013` baseline and aniso runs:

```bash
python scripts/metric_assessment_suite.py \
  --baseline_model_path <PATH_RHVAE_STANDARD_N_seed013> \
  --ours_model_path <PATH_ANISO_4C_N_seed013> \
  --protocol_profile strict \
  --profile short \
  --sampling_seeds 13 29 47 \
  --sampler_name volume_riemannian \
  --strict_mass_mode dual \
  --strict_volume_power 2.0 \
  --strict_n_lf 10 \
  --strict_eps_lf 0.05 \
  --strict_fp_steps 15 \
  --strict_fp_damping 0.72 \
  --strict_momentum_persist 0.0 \
  --strict_no_adaptive_dual_step \
  --strict_dynamic_jitter_scale 0.0 \
  --strict_radial_prior_weight 0.1 \
  --run_quality --skip_fid --save_plots \
  --output_dir results/metric_assessment_aniso4c/N<NNN>/w010
```

Expected artifacts:
- `summary_by_protocol.json`
- `raw_metrics.csv`
- `scorecard.csv`
- `ranking_summary.json`
- `protocol_configs.json`

## 6.5 Step 5: Bias ablation

Rerun Step 4 with:
- `--strict_radial_prior_weight 0.0`
- output to `.../w0`

Compare runs:

```bash
python scripts/compare_sampler_assessment_runs.py \
  --run_a results/metric_assessment_aniso4c/N<NNN>/w0/<timestamp> \
  --run_b results/metric_assessment_aniso4c/N<NNN>/w008/<timestamp> \
  --label_a unbiased \
  --label_b engineered \
  --protocol strict \
  --model_key aniso
```

## 6.6 Step 6: Three-zone trajectory diagnostics (outside only, 3+3 starts)

```bash
python scripts/quick/visualize_rhmc_three_zones_real_metric.py \
  --model_path <PATH_ANISO_4C_RUN> \
  --output_dir results/rhmc_three_zone_aniso4c \
  --seed 13 \
  --sampler_name volume_riemannian \
  --volume_power 2.0 \
  --steps 70 \
  --n_lf_inner 10 \
  --eps 0.05 \
  --fp_steps 15 \
  --fp_damping 0.72 \
  --momentum_persist 0.0 \
  --mass_mode dual \
  --dynamic_jitter_scale 0.0 \
  --radial_prior_weight 0.1 \
  --void_eigshape_mode det_preserving_spectral \
  --void_eigshape_alpha_min 0.8 \
  --void_eigshape_power -1.2 \
  --void_eigshape_eig_floor 1e-8 \
  --only_outside_starts \
  --near_points 3 \
  --far_points 3
```

Repeat with:
- `--mass_mode standard`

## 6.7 Optional branch: Ellipse 2D sanity-check track

Purpose:
- Run the same geometry/sampler stack on the ellipse data generator used by the pythae baseline script.
- Keep latent DoF at 2 (`latent_dim=2`) for direct geometric visualization.
- Use this as a diagnostic/interpretability track, not as a substitute for RotMNIST low-data conclusions.

### 6.7.1 Train RHVAE standard (ellipse data, latent 2D)

```bash
python scripts/run_with_config.py \
  --config configs/run_rhvae_baseline_100ep.yaml \
  output_dir=outputs/ellipse_2d_check/rhvae_standard \
  epochs=40 \
  seed=13 \
  lr=0.0005 \
  batch_size=64 \
  num_sequences=200 \
  max_frames=3000 \
  frame_mode=t0 \
  latent_dim=2 \
  n_centroids=100
```

### 6.7.2 Train Aniso core4 (ellipse data, latent 2D)

```bash
python scripts/run_with_config.py \
  --config configs/run_geometry_metric_core4_4k_seed13.yaml \
  output_dir=outputs/ellipse_2d_check/aniso_core4 \
  epochs=40 \
  seed=13 \
  latent_dim=2 \
  n_centroids=100 \
  num_sequences=200 \
  max_frames=3000 \
  frame_mode=t0
```

### 6.7.3 Resolve latest run folders

```bash
BASE_MODEL_PATH="$(ls -dt outputs/ellipse_2d_check/rhvae_standard/* | head -1)"
ANISO_MODEL_PATH="$(ls -dt outputs/ellipse_2d_check/aniso_core4/* | head -1)"
echo "BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "ANISO_MODEL_PATH=${ANISO_MODEL_PATH}"
```

### 6.7.4 Run strict parity assessment on ellipse runs

```bash
python scripts/metric_assessment_suite.py \
  --baseline_model_path "${BASE_MODEL_PATH}" \
  --ours_model_path "${ANISO_MODEL_PATH}" \
  --protocol_profile strict \
  --profile short \
  --sampling_seeds 13 29 47 \
  --sampler_name volume_riemannian \
  --strict_mass_mode dual \
  --strict_volume_power 2.0 \
  --strict_n_lf 10 \
  --strict_eps_lf 0.05 \
  --strict_fp_steps 15 \
  --strict_fp_damping 0.72 \
  --strict_momentum_persist 0.0 \
  --strict_no_adaptive_dual_step \
  --strict_dynamic_jitter_scale 0.0 \
  --strict_radial_prior_weight 0.1 \
  --run_quality --skip_fid --save_plots \
  --output_dir results/ellipse_2d_metric_assessment_core4
```

### 6.7.5 Run three-zone visualization on ellipse Aniso model

```bash
python scripts/quick/visualize_rhmc_three_zones_real_metric.py \
  --model_path "${ANISO_MODEL_PATH}" \
  --output_dir results/ellipse_2d_three_zone_core4 \
  --seed 13 \
  --sampler_name volume_riemannian \
  --volume_power 2.0 \
  --steps 70 \
  --n_lf_inner 10 \
  --eps 0.05 \
  --eps_jitter 0.0 \
  --n_lf_jitter 0 \
  --fp_steps 15 \
  --fp_damping 0.72 \
  --momentum_persist 0.0 \
  --radial_prior_weight 0.1 \
  --only_outside_starts \
  --near_points 3 \
  --far_points 3 \
  --near_multiplier 1.2 \
  --far_multiplier 2.4 \
  --max_radius 45.0 \
  --grid_n 180 \
  --void_eigshape_mode det_preserving_spectral \
  --void_eigshape_alpha_min 0.8 \
  --void_eigshape_power -1.2 \
  --void_eigshape_eig_floor 1e-8 \
  --mass_mode dual \
  --dynamic_jitter_scale 0.0
```

### 6.7.6 Interpretation boundary for ellipse track

- This branch is primarily for geometric debugging and visual interpretability in 2D latent space.
- It does not replace RotMNIST low-data statistical benchmarking.
- Report it as a sanity/diagnostic appendix unless replicated under target benchmark data.


## 7) Data And Artifact Map (Where Everything Goes)

## 7.1 Data
- Processed dataset: `data/processed/rotmnist/v1`
- Train/val/test tensors: `train.pt`, `val.pt`, `test.pt`
- Subset definitions: `data/processed/rotmnist/v1/subsets/Nxxx_seedyyy.json`

## 7.2 Model artifacts
- `outputs/low_data_models/rotmnist_v1/...`
- Aniso and RHVAE runs contain:
  - `rhvae_model.pt`
  - `rhvae_metric.pt`
  - `train_data.pt`
  - diagnostics plots and history

## 7.3 Benchmark outputs
- Family benchmark:
  - `results/low_data_benchmark_aniso4c/<timestamp>/...`
- Strict geometry assessment:
  - `results/metric_assessment_aniso4c/N<NNN>/w0/<timestamp>/...`
  - `results/metric_assessment_aniso4c/N<NNN>/w008/<timestamp>/...`
- Three-zone diagnostics:
  - `results/rhmc_three_zone_aniso4c/<timestamp>/...`


## 8) KPI Definitions And Interpretation

Main KPIs for the paper narrative:
- `fid` (lower is better): generation realism/divergence.
- `prd_precision` (higher is better): sample quality fidelity.
- `prd_recall` (higher is better): mode coverage.
- `d_rmse` (lower is better): latent path distance behavior.
- `geo_euc_ratio` (higher is typically better): geodesic usefulness vs Euclidean baseline.
- `aug_balanced_accuracy` (higher is better): downstream utility of generated data.

Interpretation guidance:
- A higher `prd_recall` with similar precision supports the "better mode coverage" claim.
- Low-data robustness is demonstrated by flatter metric-vs-N degradation curves.
- In strict protocol, gains indicate geometry contribution, not sampler mismatch.


## 9) Acceptance Criteria (Decision Gates)

1. Low-data robustness:
- Aniso curve is flatter than vanilla on at least 3 low-data regimes.
- Aniso is comparable or better than RHVAE standard in the same regimes.

2. Coverage:
- `prd_recall(aniso) > prd_recall(rhvae_standard)`.
- Statistical support preferred via paired tests with Holm correction.

3. Geometry isolation:
- Under strict same-sampler settings, Aniso remains better on rescue and coverage metrics.

4. Bias audit:
- `radial_prior_weight=0.0` remains competitive.
- `0.08` can improve rescue speed but should not be the single source of gains.

5. Numerical stability:
- No chronic FP saturation.
- `eps_scale_mean` tends to be lower in far-void than on manifold.
- Cholesky fallback remains near zero.


## 10) Scientific Implications

## 10.1 If protocol succeeds
- Supports the claim that geometry design in latent voids matters in sparse-data settings.
- Supports that AnisoRHVAE improves connectivity between modes.
- Shows that dual RHMC with controlled adaptation can be stable and useful.

## 10.2 If only biased setting wins
- Claim weakens to "engineered prior helps," not "geometry alone helps."
- Must clearly state bias dependence in conclusions.

## 10.3 If strict parity removes gains
- Improvement likely sampler-driven or due to mismatched operating point.
- Requires revisiting geometry-only claim.


## 11) Engineering Implications

1. Dual metric dynamics is sensitive to local anisotropy.
2. Adaptive dual step control is practically required for stable trajectories in void regions.
3. `fp_steps` and `fp_damping` must be tuned with transition steepness and eigenshape.
4. `--quick` mode in `metric_assessment_suite.py` intentionally clamps some strict settings:
- For example, `strict_fp_steps` may be capped to `3`.
- Use full mode for real conclusions.


## 12) Known Risks And Failure Modes

1. FP saturation:
- If high, implicit updates are not converging reliably.
- Mitigations: increase `fp_steps`, adjust `fp_damping`, soften transition.

2. Under-adaptation in void:
- If `eps_scale_mean` stays too close to `1.0` in far-void, displacement control is too weak.
- Mitigations: reduce `adaptive_max_dual_displacement`, lower base `eps`, or increase LF steps.

3. Over-biasing:
- High `radial_prior_weight` can over-shape trajectories and compromise "unbiased geometry" narrative.

4. Quick profile misuse:
- Quick smoke runs are for plumbing only, not final scientific conclusions.


## 13) Reproducibility Checklist

Before running full study:
1. Verify dataset subsets exist for all `N` and seeds.
2. Verify Aniso runs use `core4_spatial`.
3. Verify strict protocol config persistence in `protocol_configs.json`.
4. Verify all runs use the same seed lists specified by protocol.
5. Archive exact command lines and timestamps in lab notes.

After running:
1. Confirm all required output files exist per run.
2. Confirm no silent fallback changed intended sampler mode.
3. Confirm reported metrics are from non-quick runs for final tables.


## 14) Smoke Test Recipe (Mandatory Before Full Grid)

1. Train one Aniso smoke run:
- `N=50`, `seed=13`, `epochs=1`, `aniso_profile=core4_spatial`.
- Inspect `rhvae_metric.pt -> config`.

2. Run one strict dual smoke assessment:
- `aniso_only`, strict dual exact flags enabled.
- Inspect `protocol_configs.json` for:
  - `mass_mode`
  - `adaptive_dual_step`
  - `dynamic_jitter_scale`

3. Run one three-zone outside-only diagnostic:
- `near_points=3`, `far_points=3`.
- Check that distance-to-manifold traces are produced and grouped by start type.


## 15) Recommended Reporting Structure (Paper)

1. Family benchmark table:
- `vanilla_vae`, `rhvae_standard`, `aniso` across N and metrics.

2. Low-data degradation plot:
- X-axis: N
- Y-axis: FID and PRD recall
- Show confidence intervals.

3. Strict parity ablation:
- Same sampler settings for baseline and aniso.
- Report rescue and coverage metrics.

4. Bias ablation:
- `w=0.0` vs `w=0.08`
- Include effect size and qualitative trajectory differences.

5. Three-zone qualitative panel:
- Outside starts only, 3 near + 3 far.
- Distance trajectories plus acceptance/stability traces.


## 16) Practical Notes

- Keep run outputs in dedicated roots per phase to avoid mixing artifacts.
- For expensive grids, run `N=50,100` first to validate behavior and only then expand to `500,5000`.
- Keep `--no_metropolis` reserved for dynamics debugging; final quantitative metrics should include Metropolis correction.
- Treat metric and RHMC knobs as dataset-specific: changing the data distribution implies changing the learned metric and re-tuning RHMC operating values.
- Do not assume transfer of `radial_prior_weight`, eigenshape knobs, or adaptive displacement caps across datasets.
- Ellipse 2D runs are ideal for qualitative debugging of forces/flows but should be marked as a separate evidence layer from RotMNIST low-data results.


## 16.1 Dataset transfer policy (mandatory)

When moving away from the ellipse baseline data:
1. Retrain the metric on the new dataset (do not reuse old metric artifacts).
2. Re-run RHMC diagnostics to re-estimate stable sampler ranges.
3. Re-run strict parity and bias ablations with the new metric.
4. Re-validate acceptance, FP saturation, and outside-to-manifold rescue behavior.


## 17) Current Status Of Repository Support

As of current implementation:
- `core4_spatial` profile is available in training.
- Missing-model trainer can target that profile.
- Metric assessment supports strict dual/adaptive controls and persists them to protocol config outputs.
- Smoke validation has already confirmed:
  - profile values in serialized Aniso metric config.
  - strict dual/adaptive fields in `protocol_configs.json`.
