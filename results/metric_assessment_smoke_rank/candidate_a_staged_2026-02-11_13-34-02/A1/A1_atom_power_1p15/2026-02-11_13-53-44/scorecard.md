# Metric Assessment Scorecard

| protocol | metric | baseline | aniso | baseline_status | aniso_status | relative_pass | target |
|---|---|---:|---:|---|---|---|---|
| strict | acceptance_mean | 1.0 | 1.0 | red | red |  |  |
| strict | dh_p95_abs | 0.000744807720184326 | 0.016439506411552427 | green | green |  |  |
| strict | h_drift_slope_abs_mean | 0.010253509361766902 | 0.010041017121433763 | yellow | yellow |  |  |
| strict | ess_norm_min | 109.15811696084752 | 71.1398232687465 | yellow | red |  |  |
| strict | iact_median | 6.136925345295865 | 11.755714758783329 | green | green |  |  |
| strict | coverage_local | 0.08854166666666667 | 0.09895833333333333 | red | red |  |  |
| strict | rescue_rate | 0.5 | 0.25 | red | red |  |  |
| strict | median_steps_to_manifold | 11.5 | 12.833333333333334 | green | green |  |  |
| strict | plateau_fraction | 0.8843507178278532 | 0.0 | red | green |  |  |
| strict | tangent_alignment_mean | 0.6963202754656473 | 0.6083609859148661 | red | red |  |  |
| strict | rescue_directionality_mean | 0.32773830288267863 | 0.9618752928950789 | red | green |  |  |
| strict | border_overshoot_index | -4.244349201520284 | -2.3546628952026367 | green | green |  |  |
| strict | relative_ess_norm_min | 109.15811696084752 | 71.1398232687465 |  |  | False | >= +20% |
| strict | relative_coverage_local | 0.08854166666666667 | 0.09895833333333333 |  |  | False | >= +0.10 |
| strict | relative_rescue_rate | 0.5 | 0.25 |  |  | False | >= +0.15 |
| strict | relative_median_steps_to_manifold | 11.5 | 12.833333333333334 |  |  | False | <= -20% |
