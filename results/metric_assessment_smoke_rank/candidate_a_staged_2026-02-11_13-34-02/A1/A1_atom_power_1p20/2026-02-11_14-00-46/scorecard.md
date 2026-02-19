# Metric Assessment Scorecard

| protocol | metric | baseline | aniso | baseline_status | aniso_status | relative_pass | target |
|---|---|---:|---:|---|---|---|---|
| strict | acceptance_mean | 1.0 | 1.0 | red | red |  |  |
| strict | dh_p95_abs | 0.000744807720184326 | 0.0008577585220336907 | green | green |  |  |
| strict | h_drift_slope_abs_mean | 0.010253509361766902 | 0.005882584940988758 | yellow | green |  |  |
| strict | ess_norm_min | 109.15811696084752 | 74.36108995495194 | yellow | red |  |  |
| strict | iact_median | 6.136925345295865 | 11.020503993667903 | green | green |  |  |
| strict | coverage_local | 0.08854166666666667 | 0.09895833333333333 | red | red |  |  |
| strict | rescue_rate | 0.5 | 0.25 | red | red |  |  |
| strict | median_steps_to_manifold | 11.5 | 12.333333333333334 | green | green |  |  |
| strict | plateau_fraction | 0.8843507178278532 | 0.0 | red | green |  |  |
| strict | tangent_alignment_mean | 0.6963202754656473 | 0.5330816010634104 | red | red |  |  |
| strict | rescue_directionality_mean | 0.32773830288267863 | 0.9603454089236653 | red | green |  |  |
| strict | border_overshoot_index | -4.244349201520284 | -2.835017204284668 | green | green |  |  |
| strict | relative_ess_norm_min | 109.15811696084752 | 74.36108995495194 |  |  | False | >= +20% |
| strict | relative_coverage_local | 0.08854166666666667 | 0.09895833333333333 |  |  | False | >= +0.10 |
| strict | relative_rescue_rate | 0.5 | 0.25 |  |  | False | >= +0.15 |
| strict | relative_median_steps_to_manifold | 11.5 | 12.333333333333334 |  |  | False | <= -20% |
