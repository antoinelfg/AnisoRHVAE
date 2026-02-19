# Metric Assessment Scorecard

| protocol | metric | baseline | aniso | baseline_status | aniso_status | relative_pass | target |
|---|---|---:|---:|---|---|---|---|
| strict | acceptance_mean | 1.0 | 1.0 | red | red |  |  |
| strict | dh_p95_abs | 0.000744807720184326 | 0.016053225596745803 | green | green |  |  |
| strict | h_drift_slope_abs_mean | 0.010253509361766902 | 0.012591289978578708 | yellow | yellow |  |  |
| strict | ess_norm_min | 109.15811696084752 | 67.36243188824058 | yellow | red |  |  |
| strict | iact_median | 6.136925345295865 | 12.577262782872966 | green | green |  |  |
| strict | coverage_local | 0.08854166666666667 | 0.09375 | red | red |  |  |
| strict | rescue_rate | 0.5 | 0.25 | red | red |  |  |
| strict | median_steps_to_manifold | 11.5 | 13.0 | green | green |  |  |
| strict | plateau_fraction | 0.8843507178278532 | 0.0 | red | green |  |  |
| strict | tangent_alignment_mean | 0.6963202754656473 | 0.6596746842066447 | red | red |  |  |
| strict | rescue_directionality_mean | 0.32773830288267863 | 0.9917967128566176 | red | green |  |  |
| strict | border_overshoot_index | -4.244349201520284 | -2.4897165298461914 | green | green |  |  |
| strict | relative_ess_norm_min | 109.15811696084752 | 67.36243188824058 |  |  | False | >= +20% |
| strict | relative_coverage_local | 0.08854166666666667 | 0.09375 |  |  | False | >= +0.10 |
| strict | relative_rescue_rate | 0.5 | 0.25 |  |  | False | >= +0.15 |
| strict | relative_median_steps_to_manifold | 11.5 | 13.0 |  |  | False | <= -20% |
