# W&B Dashboard Template: Metric Assessment Benchmark

Use this layout for the baseline RHVAE vs AnisoRHVAE benchmark logs produced by
`scripts/metric_assessment_suite.py`.

## 1) Run the benchmark

```bash
python scripts/metric_assessment_suite.py \
  --baseline_model_path outputs/pythae_rhvae_baseline/2026-02-09_15-04-06 \
  --ours_model_path outputs/pythae_rhvae_baseline/2026-02-09_18-16-00 \
  --profile short \
  --protocol_profile all \
  --sampling_seeds 13 29 47 71 89 \
  --run_quality --skip_fid --save_plots --collect_reference_visuals \
  --wandb_max_plot_images -1 \
  --wandb_project rhvae-geometry-benchmark \
  --wandb_entity <your_entity> \
  --wandb_group metric_assessment_benchmark \
  --wandb_name_mode auto
```

SLURM option:

```bash
sbatch scripts/sbatch_metric_assessment_benchmark.sh
```

## 2) Run via W&B Sweep (optional)

```bash
wandb sweep scripts/wandb_metric_assessment_sweep.yaml
sbatch scripts/sbatch_metric_assessment_sweep.sh <entity/project/sweep_id>
```

## 3) Workspace filters

- Group filter: `metric_assessment_benchmark`
- Optional tag filter: set via `--wandb_tags` and filter by your tag.

## 4) Recommended dashboard panels

- Scalar panels:
  - `scorecard/claim_validated`
  - `scorecard/strict/pass`
  - `scorecard/matched/pass`
- Summary metric panels:
  - `summary/strict/aniso/ess_norm_min`
  - `summary/strict/aniso/coverage_local`
  - `summary/strict/aniso/rescue_rate`
  - `summary/strict/aniso/median_steps_to_manifold`
  - baseline equivalents under `summary/strict/baseline/*`
- Bar charts logged by script:
  - `dashboard/acceptance_mean_bar`
  - `dashboard/ess_norm_min_bar`
  - `dashboard/coverage_local_bar`
  - `dashboard/rescue_rate_bar`
  - `dashboard/median_steps_to_manifold_bar`
  - `dashboard/plateau_fraction_bar`
- Tables:
  - `tables/raw_metrics`
  - `tables/summary`
  - `tables/scorecard`
- Media panels:
  - Keys under `plots/*` (energy/accept traces, ACF, ESS bars, rescue curves, coverage maps)
  - Keys under `reference/*` (reconstruction, training curves, interpolation, RHMC, rescue, metric 3D, geodesic panels)

## 5) Artifact to inspect/download

Each run logs one artifact of type `metric_assessment` containing:
- `raw_metrics.csv`
- `summary_by_protocol.json`
- `scorecard.csv`
- `scorecard.md`
- `protocol_configs.json`
- `model_run_bilan.csv` (reconstruction + generation quick bilan from existing run artifacts)
- `plots/`
