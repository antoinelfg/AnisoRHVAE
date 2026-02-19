# Metric Assessment Scorecard

| protocol | metric | baseline | aniso | baseline_status | aniso_status | relative_pass | target |
|---|---|---:|---:|---|---|---|---|
| strict | acceptance_mean | 1.0 | 0.99375 | red | red |  |  |
| strict | dh_p95_abs | 0.016760587692260742 | 0.01932035684585571 | green | green |  |  |
| strict | h_drift_slope_abs_mean | 0.01870237616746407 | 0.011117289463679045 | yellow | yellow |  |  |
| strict | ess_norm_min | 73.88010589535094 | 81.0341829386467 | red | yellow |  |  |
| strict | iact_median | 10.171297809329173 | 11.458869243231312 | green | green |  |  |
| strict | coverage_local | 0.02734375 | 0.02734375 | red | red |  |  |
| strict | rescue_rate | 0.0 | 0.375 | red | red |  |  |
| strict | median_steps_to_manifold | 21.0 | 10.25 | green | green |  |  |
| strict | plateau_fraction | 0.625 | 0.0 | red | green |  |  |
| strict | tangent_alignment_mean | 0.7000663578510284 | 0.6297697722911835 | red | red |  |  |
| strict | rescue_directionality_mean | 0.49883004783411455 | 0.9483397622654397 | yellow | green |  |  |
| strict | border_overshoot_index | 3.510589599609375 | -1.8031127452850342 | red | green |  |  |
| strict | relative_ess_norm_min | 73.88010589535094 | 81.0341829386467 |  |  | False | >= +20% |
| strict | relative_coverage_local | 0.02734375 | 0.02734375 |  |  | False | >= +0.10 |
| strict | relative_rescue_rate | 0.0 | 0.375 |  |  | True | >= +0.15 |
| strict | relative_median_steps_to_manifold | 21.0 | 10.25 |  |  | True | <= -20% |
