# Metric Assessment Scorecard

| protocol | metric | baseline | aniso | baseline_status | aniso_status | relative_pass | target |
|---|---|---:|---:|---|---|---|---|
| strict | acceptance_mean | 1.0 | 1.0 | red | red |  |  |
| strict | dh_p95_abs | 0.016853898763656616 | 0.01982452869415283 | green | green |  |  |
| strict | h_drift_slope_abs_mean | 0.030334151709871256 | 0.01366346668529431 | red | yellow |  |  |
| strict | ess_norm_min | 73.79858424511612 | 83.23903041157392 | red | yellow |  |  |
| strict | iact_median | 11.305017702895356 | 11.254545995904067 | green | green |  |  |
| strict | coverage_local | 0.03125 | 0.0390625 | red | red |  |  |
| strict | rescue_rate | 0.0 | 0.5 | red | red |  |  |
| strict | median_steps_to_manifold | 21.0 | 2.5 | green | green |  |  |
| strict | plateau_fraction | 0.75 | 0.0 | red | green |  |  |
| strict | relative_ess_norm_min | 73.79858424511612 | 83.23903041157392 |  |  | False | >= +20% |
| strict | relative_coverage_local | 0.03125 | 0.0390625 |  |  | False | >= +0.10 |
| strict | relative_rescue_rate | 0.0 | 0.5 |  |  | True | >= +0.15 |
| strict | relative_median_steps_to_manifold | 21.0 | 2.5 |  |  | True | <= -20% |
