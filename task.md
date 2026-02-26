You are a senior ML experiment auditor and code execution agent.

Repository:
- `/home/alaforgu/scratch/longitudinal_experiments/AnisoRHVAE`

Mission:
- Explain deeply what this project is doing in the current benchmarking setup.
- Verify that everything is correctly wired end-to-end.
- Verify that RHVAE standard and Aniso training both use the pythae baseline pipeline.
- Verify that benchmark sampling uses the intended strict `volume_riemannian` sampler for RHVAE/Aniso.
- Verify that Vanilla VAE is trained separately and sampled separately (Gaussian latent baseline).
- Run checks with the canonical parameters that already worked for Aniso 4c and RHMC dual adaptive sampling.

Non-negotiable constraints:
1. Do not silently change hyperparameters.
2. Do not silently fall back to alternate training backends.
3. Do not silently switch sampler for RHVAE/Aniso.
4. Produce evidence for every claim (command output, file, config dump).
5. If something is wrong, patch minimally and re-run validation.
6. Keep all WandB logging active if configured.

Canonical settings to enforce

A) Dataset/check context
- Main check dataset: `data/processed/ellipses_lowdata_v1`
- Latent dimension: `2`
- Low-data sizes: `N in {50, 100, 500, 1000}`
- Seed set for full run: `{13}` for smoke; `{13,29,47,71,89}` optional extended

B) Aniso profile (core4_spatial exact values)
- `kernel_type=mahalanobis`
- `atom_norm=trace`
- `atom_power=1.0959864113997024`
- `kernel_power=1.0`
- `precision_jitter=0.01`
- `void_threshold=1.2`
- `void_weight_threshold=-1.0`
- `void_decay_type=invquad`
- `void_decay_scale=8.87131893173153`
- `void_decay_power=1.7447710342008058`
- `void_decay_softplus_k=5.0`
- `radial_stretch=9.252860449508804`
- `transition_steepness=7.276725930229776`
- `use_attractor=true`
- `attractor_smoothness=soft`
- `attractor_metric=mahalanobis`
- `attractor_use_det=true`
- `attractor_gamma=7.624554217806352`
- `attractor_k_nearest=1`
- `attractor_bias_energy=18.0`
- `void_eigshape_mode=det_preserving_spectral`
- `void_eigshape_alpha_min=0.8`
- `void_eigshape_power=-1.2`
- `void_eigshape_eig_floor=1e-8`

C) RHMC sampler settings that worked
- `sampler_name=volume_riemannian`
- `use_dual_metric=true`
- `volume_power=2.0`
- `n_lf=10`
- `eps_lf=0.03`
- `fp_steps=15`
- `fp_damping=0.72`
- `momentum_persist=0.0`
- `adaptive_dual_step=true`
- `adaptive_max_dual_displacement=0.09`
- `adaptive_min_step_scale=0.05`
- `radial_prior_weight=0.1`
- `mcmc_steps=60` for benchmark generation

D) Auto-temperature policy
- Enable auto-temperature for RHVAE and Aniso:
- `--rhvae_auto_temperature`
- `--rhvae_auto_temperature_stat mean_nn`
- `--rhvae_auto_temperature_every 1`
- `--rhvae_temperature_scale 1.0`

Required audit steps

1) Static linkage audit (code-level)
Run and inspect:
```bash
cd /home/alaforgu/scratch/longitudinal_experiments/AnisoRHVAE
rg -n "rhvae_standard_backend|rhvae_aniso_backend|run_pythae_rhvae_baseline|rhvae_sampler_name|volume_riemannian|rhvae_sampler_" \
  scripts/train_missing_low_data_models.py scripts/run_low_data_benchmark.py scripts/sbatch/sbatch_train_low_data_ellipses_latent2.sh
Confirm:

train_missing_low_data_models.py defaults:
rhvae_standard_backend=run_with_config
rhvae_aniso_backend=run_with_config
rhvae_standard branch calls run_pythae_rhvae_baseline.py --rhvae_variant standard
aniso branch calls run_pythae_rhvae_baseline.py with core4 geometry args
run_low_data_benchmark.py exposes strict RHVAE sampler CLI flags and passes them into RHVAEAdapter -> VolumeElementRiemannianHMCSampler
VanillaVAEAdapter.sample_latents remains Gaussian baseline
Smoke training run (fast sanity)
Run exactly:
bash

python scripts/train_missing_low_data_models.py \
  --models vanilla_vae rhvae_standard aniso \
  --processed_dir data/processed/ellipses_lowdata_v1 \
  --output_root outputs/low_data_models/ellipses_lowdata_v1_latent2_check \
  --subset_ns 50 \
  --subset_seeds 13 \
  --epochs 5 \
  --batch_size 64 \
  --latent_dim 2 \
  --rhvae_standard_backend run_with_config \
  --rhvae_standard_num_sequences_mode from_n \
  --rhvae_standard_max_frames_mode n \
  --rhvae_standard_frame_mode t0 \
  --rhvae_standard_lr 0.0005 \
  --rhvae_standard_regularization 0.01 \
  --rhvae_standard_analysis_sampler volume \
  --rhvae_aniso_backend run_with_config \
  --rhvae_aniso_profile core4_spatial \
  --rhvae_aniso_num_sequences_mode from_n \
  --rhvae_aniso_max_frames_mode n \
  --rhvae_aniso_frame_mode t0 \
  --rhvae_aniso_lr 0.0005 \
  --rhvae_aniso_regularization 0.6 \
  --rhvae_aniso_analysis_sampler volume \
  --rhvae_auto_temperature \
  --rhvae_auto_temperature_stat mean_nn \
  --rhvae_auto_temperature_every 1 \
  --rhvae_temperature_scale 1.0
Artifact integrity checks
For each model run folder (latest timestamp under N050/seed013):
vanilla must contain model.pt, training_history.json
rhvae/aniso must contain rhvae_model.pt, rhvae_metric.pt, training_history.pt
no missing critical files
Run:

bash

python - <<'PY'
from pathlib import Path
import torch, json

root = Path("outputs/low_data_models/ellipses_lowdata_v1_latent2_check")
for mid in ["vanilla_vae","rhvae_standard","aniso"]:
    p = root / mid / "N050" / "seed013"
    runs = sorted([d for d in p.glob("*") if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
    if not runs:
        print(mid, "MISSING_RUN")
        continue
    r = runs[0]
    print("\nMODEL", mid, "RUN", r)
    if mid == "vanilla_vae":
        print("model.pt", (r/"model.pt").exists())
        print("training_history.json", (r/"training_history.json").exists())
    else:
        print("rhvae_model.pt", (r/"rhvae_model.pt").exists())
        print("rhvae_metric.pt", (r/"rhvae_metric.pt").exists())
        payload = torch.load(r/"rhvae_metric.pt", map_location="cpu")
        cfg = payload.get("config", {})
        print("temperature", payload.get("temperature", None))
        print("regularization", payload.get("regularization", None))
        print("kernel_type", cfg.get("kernel_type", None))
        if mid == "aniso":
            keys = ["kernel_type","atom_norm","atom_power","kernel_power","precision_jitter","void_threshold",
                    "void_weight_threshold","void_decay_type","void_decay_scale","void_decay_power",
                    "void_decay_softplus_k","radial_stretch","transition_steepness","use_attractor",
                    "attractor_smoothness","attractor_metric","attractor_use_det","attractor_gamma",
                    "attractor_k_nearest","attractor_bias_energy","void_eigshape_mode",
                    "void_eigshape_alpha_min","void_eigshape_power","void_eigshape_eig_floor"]
            for k in keys:
                print(k, cfg.get(k, None))
PY
Benchmark alignment check (strict sampler path)
Run:
bash

python scripts/run_low_data_benchmark.py \
  --processed_dir data/processed/ellipses_lowdata_v1 \
  --model_root outputs/low_data_models/ellipses_lowdata_v1_latent2_check \
  --models vanilla_vae rhvae_standard aniso \
  --subset_ns 50 \
  --subset_seeds 13 \
  --n_gen_samples 512 \
  --n_interp_pairs 8 \
  --interp_steps 32 \
  --aug_synth_samples 128 \
  --bootstrap_samples 200 \
  --rhvae_sampler_name volume_riemannian \
  --rhvae_sampler_mcmc_steps 60 \
  --rhvae_sampler_n_lf 10 \
  --rhvae_sampler_eps_lf 0.03 \
  --rhvae_sampler_volume_power 2.0 \
  --rhvae_sampler_use_dual_metric \
  --rhvae_sampler_fp_steps 15 \
  --rhvae_sampler_fp_damping 0.72 \
  --rhvae_sampler_momentum_persist 0.0 \
  --rhvae_sampler_radial_prior_weight 0.1 \
  --rhvae_sampler_adaptive_dual_step \
  --rhvae_sampler_adaptive_max_dual_displacement 0.09 \
  --rhvae_sampler_adaptive_min_step_scale 0.05 \
  --output_dir results/low_data_benchmark_ellipses_latent2_check
Confirm outputs exist:

summary_by_model_n.json
metrics_table.csv
stat_tests.json (if produced)
figures directory
RHMC three-zone diagnostic check (near/far only, 3+3)
Use latest Aniso run path from step 3:
bash

python scripts/quick/visualize_rhmc_three_zones_real_metric.py \
  --model_path <ANISO_RUN_PATH> \
  --output_dir results/rhmc_three_zone_ellipses_latent2_check \
  --seed 13 \
  --sampler_name volume_riemannian \
  --volume_power 2.0 \
  --steps 70 \
  --n_lf_inner 10 \
  --eps 0.03 \
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
  --use_dual_metric \
  --adaptive_dual_step \
  --adaptive_max_dual_displacement 0.09 \
  --adaptive_min_step_scale 0.05
Confirm in console/summary:

dual-step adaptation enabled
fixed-point saturation low
eps_scale_mean lower for far than manifold (or at least visibly reduced in far zone)
no Cholesky fallback explosion
Optional full overnight launch via sbatch
If smoke passes, run full:
N={50,100,500,1000}, epochs=200, same sampler settings
use sbatch_train_low_data_ellipses_latent2.sh with exported vars enforcing strict sampler and pythae backends
Expected final report format

Executive summary (pass/fail overall)
Wiring audit table:
component
expected behavior
observed behavior
evidence file/line or command output
status (PASS/FAIL)
Runtime checks table:
smoke train
artifact integrity
benchmark strict sampler
three-zone diagnostics
Exact resolved run paths used
Any mismatch and minimal patch applied
Final “GO / NO-GO” for full benchmark
Important interpretation requirement

Explain clearly why this setup is scientifically aligned:
Vanilla: separate simple latent baseline
RHVAE standard and Aniso: same training backend (pythae baseline)
Same strict RHMC sampling family for RHVAE/Aniso in benchmark
Aniso advantage then reflects geometry/métrique effects, not pipeline mismatch