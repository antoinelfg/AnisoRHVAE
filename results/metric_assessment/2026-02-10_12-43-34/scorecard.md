# Metric Assessment Scorecard

- `claim_validated`: **False**

| protocol | metric | baseline | aniso | baseline_status | aniso_status | relative_pass | target |
|---|---|---:|---:|---|---|---|---|
| strict | acceptance_mean | 1.0 | 1.0 | red | red |  |  |
| strict | dh_p95_abs | 0.014367538690567009 | 0.02132513523101805 | green | green |  |  |
| strict | h_drift_slope_abs_mean | 0.3232213299382816 | 0.10202349749478455 | red | red |  |  |
| strict | ess_norm_min | 618.1716781235694 | 616.8671038959865 | green | green |  |  |
| strict | iact_median | 1.4418127913168668 | 1.4442846579012762 | green | green |  |  |
| strict | coverage_local | 0.0 | 0.0078125 | red | red |  |  |
| strict | rescue_rate | 0.0 | 0.0 | red | red |  |  |
| strict | median_steps_to_manifold | 6.0 | 6.0 | green | green |  |  |
| strict | plateau_fraction | 1.0 | 0.0 | red | green |  |  |
| strict | relative_ess_norm_min | 618.1716781235694 | 616.8671038959865 |  |  | False | >= +20% |
| strict | relative_coverage_local | 0.0 | 0.0078125 |  |  | False | >= +0.10 |
| strict | relative_rescue_rate | 0.0 | 0.0 |  |  | False | >= +0.15 |
| strict | relative_median_steps_to_manifold | 6.0 | 6.0 |  |  | False | <= -20% |
| matched | acceptance_mean | 1.0 | 1.0 | red | red |  |  |
| matched | dh_p95_abs | 0.014367538690567009 | 0.02132513523101805 | green | green |  |  |
| matched | h_drift_slope_abs_mean | 0.3232213299382816 | 0.10202349749478455 | red | red |  |  |
| matched | ess_norm_min | 618.1716781235694 | 616.8671038959865 | green | green |  |  |
| matched | iact_median | 1.4418127913168668 | 1.4442846579012762 | green | green |  |  |
| matched | coverage_local | 0.0 | 0.0078125 | red | red |  |  |
| matched | rescue_rate | 0.0 | 0.0 | red | red |  |  |
| matched | median_steps_to_manifold | 6.0 | 6.0 | green | green |  |  |
| matched | plateau_fraction | 1.0 | 0.0 | red | green |  |  |
| matched | relative_ess_norm_min | 618.1716781235694 | 616.8671038959865 |  |  | False | >= +20% |
| matched | relative_coverage_local | 0.0 | 0.0078125 |  |  | False | >= +0.10 |
| matched | relative_rescue_rate | 0.0 | 0.0 |  |  | False | >= +0.15 |
| matched | relative_median_steps_to_manifold | 6.0 | 6.0 |  |  | False | <= -20% |
| tuned | acceptance_mean | 1.0 | 1.0 | red | red |  |  |
| tuned | dh_p95_abs | 0.014367538690567009 | 0.02132513523101805 | green | green |  |  |
| tuned | h_drift_slope_abs_mean | 0.3232213299382816 | 0.10202349749478455 | red | red |  |  |
| tuned | ess_norm_min | 618.1716781235694 | 616.8671038959865 | green | green |  |  |
| tuned | iact_median | 1.4418127913168668 | 1.4442846579012762 | green | green |  |  |
| tuned | coverage_local | 0.0 | 0.0078125 | red | red |  |  |
| tuned | rescue_rate | 0.0 | 0.0 | red | red |  |  |
| tuned | median_steps_to_manifold | 6.0 | 6.0 | green | green |  |  |
| tuned | plateau_fraction | 1.0 | 0.0 | red | green |  |  |
| tuned | relative_ess_norm_min | 618.1716781235694 | 616.8671038959865 |  |  | False | >= +20% |
| tuned | relative_coverage_local | 0.0 | 0.0078125 |  |  | False | >= +0.10 |
| tuned | relative_rescue_rate | 0.0 | 0.0 |  |  | False | >= +0.15 |
| tuned | relative_median_steps_to_manifold | 6.0 | 6.0 |  |  | False | <= -20% |
| overall | claim_validated |  |  |  |  | False | Aniso no red stability + >=3/4 gains in strict or matched |
