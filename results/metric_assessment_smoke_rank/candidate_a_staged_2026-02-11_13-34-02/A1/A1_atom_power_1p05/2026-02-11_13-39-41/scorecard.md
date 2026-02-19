# Metric Assessment Scorecard

| protocol | metric | baseline | aniso | baseline_status | aniso_status | relative_pass | target |
|---|---|---:|---:|---|---|---|---|
| strict | acceptance_mean | 1.0 | 1.0 | red | red |  |  |
| strict | dh_p95_abs | 0.000744807720184326 | 0.012447370092074065 | green | green |  |  |
| strict | h_drift_slope_abs_mean | 0.010253509361766902 | 0.012104500010856431 | yellow | yellow |  |  |
| strict | ess_norm_min | 109.15811696084752 | 63.51237009628026 | yellow | red |  |  |
| strict | iact_median | 6.136925345295865 | 12.361574753193148 | green | green |  |  |
| strict | coverage_local | 0.08854166666666667 | 0.08333333333333333 | red | red |  |  |
| strict | rescue_rate | 0.5 | 0.25 | red | red |  |  |
| strict | median_steps_to_manifold | 11.5 | 14.666666666666666 | green | green |  |  |
| strict | plateau_fraction | 0.8843507178278532 | 0.0 | red | green |  |  |
| strict | tangent_alignment_mean | 0.6963202754656473 | 0.6207570632298788 | red | red |  |  |
| strict | rescue_directionality_mean | 0.32773830288267863 | 0.9296336448312047 | red | green |  |  |
| strict | border_overshoot_index | -4.244349201520284 | -2.1520901322364807 | green | green |  |  |
| strict | relative_ess_norm_min | 109.15811696084752 | 63.51237009628026 |  |  | False | >= +20% |
| strict | relative_coverage_local | 0.08854166666666667 | 0.08333333333333333 |  |  | False | >= +0.10 |
| strict | relative_rescue_rate | 0.5 | 0.25 |  |  | False | >= +0.15 |
| strict | relative_median_steps_to_manifold | 11.5 | 14.666666666666666 |  |  | False | <= -20% |
