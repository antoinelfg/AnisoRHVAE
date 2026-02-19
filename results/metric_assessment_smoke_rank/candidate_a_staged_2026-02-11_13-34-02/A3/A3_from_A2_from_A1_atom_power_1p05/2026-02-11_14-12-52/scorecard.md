# Metric Assessment Scorecard

| protocol | metric | baseline | aniso | baseline_status | aniso_status | relative_pass | target |
|---|---|---:|---:|---|---|---|---|
| strict | acceptance_mean | 1.0 | 1.0 | red | red |  |  |
| strict | dh_p95_abs | 0.000744807720184326 | 0.0018824696540832506 | green | green |  |  |
| strict | h_drift_slope_abs_mean | 0.010253509361766902 | 0.006351427783703314 | yellow | green |  |  |
| strict | ess_norm_min | 109.15811696084752 | 79.85291164507248 | yellow | red |  |  |
| strict | iact_median | 6.136925345295865 | 10.092042077625266 | green | green |  |  |
| strict | coverage_local | 0.08854166666666667 | 0.08072916666666667 | red | red |  |  |
| strict | rescue_rate | 0.5 | 0.25 | red | red |  |  |
| strict | median_steps_to_manifold | 11.5 | 12.833333333333334 | green | green |  |  |
| strict | plateau_fraction | 0.8843507178278532 | 0.003968253968253968 | red | green |  |  |
| strict | tangent_alignment_mean | 0.6963202754656473 | 0.5286442041397095 | red | red |  |  |
| strict | rescue_directionality_mean | 0.32773830288267863 | 0.9872185890396725 | red | green |  |  |
| strict | border_overshoot_index | -4.244349201520284 | -1.131864070892334 | green | green |  |  |
| strict | relative_ess_norm_min | 109.15811696084752 | 79.85291164507248 |  |  | False | >= +20% |
| strict | relative_coverage_local | 0.08854166666666667 | 0.08072916666666667 |  |  | False | >= +0.10 |
| strict | relative_rescue_rate | 0.5 | 0.25 |  |  | False | >= +0.15 |
| strict | relative_median_steps_to_manifold | 11.5 | 12.833333333333334 |  |  | False | <= -20% |
