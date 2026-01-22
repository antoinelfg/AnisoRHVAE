# AnisoRHVAE

## Project summary
- Riemannian VAE geometry variants (baseline, highway, hard funnel, smooth funnel, gravity well).
- Inverse metric: base + attractor/void with alpha gating.
- Attractor uses soft/hard, euclidean/mahalanobis, optional det weighting.
- Void decay uses invquad + softplus smoothing.

## Files to know
- `src/models/rhvae_geometry.py` is the core geometry.
- `scripts/run_pythae_rhvae_baseline.py` is the training entrypoint.
- `scripts/analyze_metric_full.py` logs full 3D metric landscapes and diagnostics.
- `scripts/analyze_det_vs_r.py` analyzes log det vs r.

## Example run command
```bash
python scripts/run_pythae_rhvae_baseline.py \
  --geometry_case gravity_well \
  --temperature 0.5 \
  --regularization 0.01 \
  --precision_jitter 0.001 \
  --void_threshold 1.5 \
  --void_decay_type invquad \
  --void_decay_scale 1.0 \
  --void_decay_power 2.0 \
  --void_decay_softplus_k 5.0 \
  --transition_steepness 5.0 \
  --radial_stretch 5.0 \
  --attractor_gamma 5.0 \
  --attractor_k_nearest 10 \
  --num_sequences 200 \
  --max_frames 3000 \
  --epochs 100 \
  --batch_size 64 \
  --seed 42 \
  --wandb_name_mode auto \
  --wandb_project rhvae-geometry-benchmark
```

## Analysis command
```bash
python scripts/analyze_det_vs_r.py \
  --model_path outputs/pythae_rhvae_baseline/<RUN_FOLDER> \
  --centroid_idx 100 \
  --steps 400 \
  --step_size 0.005 \
  --direction_mode random_line \
  --direction_seed 7 \
  --det_target g \
  --log_det \
  --grid_bounds 6 \
  --output_dir outputs/det_vs_r_logdet_g
```
