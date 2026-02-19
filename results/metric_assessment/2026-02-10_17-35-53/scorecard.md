# Metric Assessment Scorecard

- `claim_validated`: **False**

| protocol | metric | baseline | aniso | baseline_status | aniso_status | relative_pass | target |
|---|---|---:|---:|---|---|---|---|
| strict | acceptance_mean | 1.0 | 0.996875 | red | red |  |  |
| strict | dh_p95_abs | 0.01831994801759719 | 0.01986940503120419 | green | green |  |  |
| strict | h_drift_slope_abs_mean | 0.013496743889682026 | 0.014357827191914987 | yellow | yellow |  |  |
| strict | ess_norm_min | 25.499035055400526 | 32.871072738308115 | red | red |  |  |
| strict | iact_median | 36.853217440267116 | 26.07292699303357 | yellow | yellow |  |  |
| strict | coverage_local | 0.02734375 | 0.02734375 | red | red |  |  |
| strict | rescue_rate | 0.0 | 0.375 | red | red |  |  |
| strict | median_steps_to_manifold | 21.0 | 10.25 | green | green |  |  |
| strict | plateau_fraction | 0.625 | 0.0 | red | green |  |  |
| strict | relative_ess_norm_min | 25.499035055400526 | 32.871072738308115 |  |  | True | >= +20% |
| strict | relative_coverage_local | 0.02734375 | 0.02734375 |  |  | False | >= +0.10 |
| strict | relative_rescue_rate | 0.0 | 0.375 |  |  | True | >= +0.15 |
| strict | relative_median_steps_to_manifold | 21.0 | 10.25 |  |  | True | <= -20% |
| matched | acceptance_mean | 1.0 | 0.996875 | red | red |  |  |
| matched | dh_p95_abs | 0.01831994801759719 | 0.01986940503120419 | green | green |  |  |
| matched | h_drift_slope_abs_mean | 0.013496743889682026 | 0.014357827191914987 | yellow | yellow |  |  |
| matched | ess_norm_min | 25.499035055400526 | 32.871072738308115 | red | red |  |  |
| matched | iact_median | 36.853217440267116 | 26.07292699303357 | yellow | yellow |  |  |
| matched | coverage_local | 0.02734375 | 0.02734375 | red | red |  |  |
| matched | rescue_rate | 0.0 | 0.375 | red | red |  |  |
| matched | median_steps_to_manifold | 21.0 | 10.25 | green | green |  |  |
| matched | plateau_fraction | 0.625 | 0.0 | red | green |  |  |
| matched | relative_ess_norm_min | 25.499035055400526 | 32.871072738308115 |  |  | True | >= +20% |
| matched | relative_coverage_local | 0.02734375 | 0.02734375 |  |  | False | >= +0.10 |
| matched | relative_rescue_rate | 0.0 | 0.375 |  |  | True | >= +0.15 |
| matched | relative_median_steps_to_manifold | 21.0 | 10.25 |  |  | True | <= -20% |
| overall | claim_validated |  |  |  |  | False | Aniso no red stability + >=3/4 gains in strict or matched |
