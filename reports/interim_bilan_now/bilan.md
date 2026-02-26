# Interim Bilan Report

## Executive summary
1. This interim bilan compares metric/sampler stability and model quality across [50, 100, 500] using existing artifacts only.
2. Sampler comparability gate is PASS for strict/matched before any causal metric-vs-sampler claim.
3. Current recommendation is: **Parallel workstream** (decision hierarchy A/B/C with conflict-to-parallel fallback).
4. Rule A active=True, Rule B active=True, Rule C active=True.
5. Low-data missingness is explicit: augmentation KPIs are NaN at N=100/500 when n=0 and excluded from aggregate denominator.
6. Confidence level: **medium-low** (small n in metric_assessment and low_data, plus current coverage state).
7. Three-zone auto-detection selected run: none yet (completed-run criteria applied).
8. All CIs in this report use t-intervals (small-sample), with source CI kept in JSON audit fields.

## KPI findings

### Table 1 — Sampler/Geometry Stability (baseline vs aniso, strict + matched)
| Protocol | KPI | Baseline | Aniso | Absolute diff | Delta % vs baseline |
| --- | --- | --- | --- | --- | --- |
| strict | acceptance_mean | 1.0000 | 0.9969 | -0.0031 | -0.31 |
| strict | dh_p95_abs | 0.0183 | 0.0199 | 0.0015 | 8.46 |
| strict | h_drift_slope_abs_mean | 0.0135 | 0.0144 | 0.0009 | 6.38 |
| strict | coverage_local | 0.0273 | 0.0273 | 0.0000 | 0.00 |
| strict | rescue_rate | 0.0000 | 0.3750 | 0.3750 | NaN |
| strict | median_steps_to_manifold | 21.0000 | 10.2500 | -10.7500 | -51.19 |
| strict | ess_norm_min | 25.4990 | 32.8711 | 7.3720 | 28.91 |
| strict | iact_median | 36.8532 | 26.0729 | -10.7803 | -29.25 |
| matched | acceptance_mean | 1.0000 | 0.9969 | -0.0031 | -0.31 |
| matched | dh_p95_abs | 0.0183 | 0.0199 | 0.0015 | 8.46 |
| matched | h_drift_slope_abs_mean | 0.0135 | 0.0144 | 0.0009 | 6.38 |
| matched | coverage_local | 0.0273 | 0.0273 | 0.0000 | 0.00 |
| matched | rescue_rate | 0.0000 | 0.3750 | 0.3750 | NaN |
| matched | median_steps_to_manifold | 21.0000 | 10.2500 | -10.7500 | -51.19 |
| matched | ess_norm_min | 25.4990 | 32.8711 | 7.3720 | 28.91 |
| matched | iact_median | 36.8532 | 26.0729 | -10.7803 | -29.25 |

### Table 2 — Model Quality Across Data Regimes (N=50/100/500)
| N | KPI | vanilla_vae | rhvae_standard | aniso | ebm_conformal | Best |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | fid | 260.8957 | 358.2052 | 311.3589 | 320.5055 | vanilla_vae |
| 50 | d_rmse | 0.0080 | 0.2179 | 0.5701 | 0.4347 | vanilla_vae |
| 50 | geo_euc_ratio_error | 0.0000 | 0.6113 | 0.6344 | 0.0000 | vanilla_vae |
| 50 | interp_smoothness | 0.0000 | 0.0000 | 0.0000 | 0.0000 | vanilla_vae |
| 50 | aug_balanced_accuracy | 0.1709 | 0.1141 | 0.1209 | 0.1094 | vanilla_vae |
| 50 | aug_macro_f1 | 0.1265 | 0.0388 | 0.0537 | 0.0274 | vanilla_vae |
| 100 | fid | 243.5156 | 329.4425 | 274.2339 | 270.4632 | vanilla_vae |
| 100 | d_rmse | 0.0049 | 0.1680 | 0.6134 | 0.3536 | vanilla_vae |
| 100 | geo_euc_ratio_error | 0.0000 | 0.8555 | 1.6686 | 0.0000 | vanilla_vae |
| 100 | interp_smoothness | 0.0000 | 0.0000 | 0.0000 | 0.0000 | vanilla_vae |
| 100 | aug_balanced_accuracy | NaN | NaN | NaN | NaN | NaN |
| 100 | aug_macro_f1 | NaN | NaN | NaN | NaN | NaN |
| 500 | fid | 297.5776 | 350.9792 | 284.5313 | 275.4465 | ebm_conformal |
| 500 | d_rmse | 0.0024 | 0.1349 | 0.6263 | 0.2326 | vanilla_vae |
| 500 | geo_euc_ratio_error | 0.0000 | 0.5438 | 0.8958 | 0.0000 | ebm_conformal |
| 500 | interp_smoothness | 0.0000 | 0.0000 | 0.0000 | 0.0000 | vanilla_vae |
| 500 | aug_balanced_accuracy | NaN | NaN | NaN | NaN | NaN |
| 500 | aug_macro_f1 | NaN | NaN | NaN | NaN | NaN |

Interpretation: interp_smoothness is log-transformed for aggregate scoring due scale compression; geo_euc_ratio is converted to distance-to-1 error before ranking/scoring.

## Metric vs sampler diagnosis
Verdict: **Parallel workstream**

- Rule A (True): integration accepted but wrong direction / weak pull-to-manifold
- Evidence A: baseline acceptance_mean (strict) = 1.0000
- Evidence A: baseline rescue_rate (strict) = 0.0000
- Evidence A: far_outside_rescue_success (three-zone) = NaN
- Rule B (True): metric shaping helps recovery without major integrator instability
- Evidence B: rescue_rate gain (aniso-baseline, strict) = 0.3750
- Evidence B: median_steps ratio aniso/baseline (strict) = 0.4881
- Evidence B: relative changes: dh_p95=0.0846, h_drift=0.0638
- Rule C (True): trade-off between geometric structure and reconstruction fidelity
- Evidence C: vanilla dominates fid+d_rmse in 2/3 N-regimes
- Evidence C: aniso rescue_rate vs baseline (strict) = 0.3750 vs 0.0000
- Evidence C: asymmetry note: rescue geometry KPIs are not available for all low-data models
- Asymmetry note: rescue geometry KPIs are not uniformly available across all low-data model families; this is handled as coverage limitation.

## Current risks and confidence level
- Confidence level: **medium-low**
- Comparability differences count: 0
- Missing/NaN KPI entries tracked: 32
- Seed check records: metric=4, low_data=72

## Limitations
- Small sample sizes (metric_assessment n=2 seeds; low_data typically n=5 seeds).
- Augmentation KPIs are unavailable (NaN, n=0) for N=100 and N=500 in current low-data artifacts.
- Three-zone diagnostics depend on latest completed overnight run; in-progress runs are excluded by completion criteria.
- Any quick-mode or profile constraints from source runs are inherited; no retraining/re-evaluation is performed here.

## Next 5 prioritized actions
1. Complete one three-zone overnight run to unlock far-rescue and escape-rate confidence with full seed support.
2. Fill low-data augmentation metrics at N=100/500 (currently n=0) to stabilize cross-regime aggregate comparisons.
3. Run a targeted sampler-only sweep around current aniso config to test if Rule A can be mitigated without geometry change.
4. Add bootstrap sensitivity on aggregate ranking (winsorization + log transform choices) to confirm recommendation stability.
5. Promote this interim bilan script into CI/regression checks so new runs auto-refresh decision evidence.

## Sanity check
| Check | Actual | Approx target | Abs diff | Within tol |
| --- | --- | --- | --- | --- |
| metric_strict_baseline_acceptance | 1.0000 | 1.0000 | 0.0000 | True |
| metric_strict_baseline_rescue | 0.0000 | 0.0000 | 0.0000 | True |
| metric_strict_baseline_steps | 21.0000 | 21.0000 | 0.0000 | True |
| metric_strict_aniso_acceptance | 0.9969 | 0.9970 | 0.0001 | True |
| metric_strict_aniso_rescue | 0.3750 | 0.3750 | 0.0000 | True |
| metric_strict_aniso_steps | 10.2500 | 10.2500 | 0.0000 | True |
| lowdata_n50_vanilla_fid | 260.8957 | 260.9000 | 0.0043 | True |
| lowdata_n50_rhvae_standard_fid | 358.2052 | 358.2000 | 0.0052 | True |
| lowdata_n50_aniso_fid | 311.3589 | 311.4000 | 0.0411 | True |
| lowdata_n50_ebm_fid | 320.5055 | 320.5000 | 0.0055 | True |

## Missing/NaN metrics
| Source | Protocol | N | Model | KPI | Reason |
| --- | --- | --- | --- | --- | --- |
| low_data_benchmark |  | 100 | aniso | aug_balanced_accuracy | n_nonpositive |
| low_data_benchmark |  | 100 | aniso | aug_balanced_accuracy | nan_mean |
| low_data_benchmark |  | 100 | aniso | aug_macro_f1 | n_nonpositive |
| low_data_benchmark |  | 100 | aniso | aug_macro_f1 | nan_mean |
| low_data_benchmark |  | 100 | ebm_conformal | aug_balanced_accuracy | n_nonpositive |
| low_data_benchmark |  | 100 | ebm_conformal | aug_balanced_accuracy | nan_mean |
| low_data_benchmark |  | 100 | ebm_conformal | aug_macro_f1 | n_nonpositive |
| low_data_benchmark |  | 100 | ebm_conformal | aug_macro_f1 | nan_mean |
| low_data_benchmark |  | 100 | rhvae_standard | aug_balanced_accuracy | n_nonpositive |
| low_data_benchmark |  | 100 | rhvae_standard | aug_balanced_accuracy | nan_mean |
| low_data_benchmark |  | 100 | rhvae_standard | aug_macro_f1 | n_nonpositive |
| low_data_benchmark |  | 100 | rhvae_standard | aug_macro_f1 | nan_mean |
| low_data_benchmark |  | 100 | vanilla_vae | aug_balanced_accuracy | n_nonpositive |
| low_data_benchmark |  | 100 | vanilla_vae | aug_balanced_accuracy | nan_mean |
| low_data_benchmark |  | 100 | vanilla_vae | aug_macro_f1 | n_nonpositive |
| low_data_benchmark |  | 100 | vanilla_vae | aug_macro_f1 | nan_mean |
| low_data_benchmark |  | 500 | aniso | aug_balanced_accuracy | n_nonpositive |
| low_data_benchmark |  | 500 | aniso | aug_balanced_accuracy | nan_mean |
| low_data_benchmark |  | 500 | aniso | aug_macro_f1 | n_nonpositive |
| low_data_benchmark |  | 500 | aniso | aug_macro_f1 | nan_mean |

_Only first 20 rows shown out of 32._

Final recommendation: **Parallel workstream**
- strict sampler geometry: rescue_rate 0.0000 -> 0.3750 and median_steps 21.00 -> 10.25 (baseline -> aniso).
- Quality aggregate winners by regime: N=50:vanilla_vae, N=100:vanilla_vae, N=500:vanilla_vae; robust model across N: vanilla_vae.
- Coverage status: sampler configs baseline/aniso are comparable on strict+matched; three-zone completed run not yet available; missing KPI entries logged: 32.
