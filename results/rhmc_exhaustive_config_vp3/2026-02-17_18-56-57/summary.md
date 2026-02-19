# RHMC Exhaustive Sweep (vp=3.0, n_lf_inner=12, eps=0.02)

- Model: `outputs/metric_core4_sweep/2026-02-17_13-53-54`
- Seeds: 2 (13..14)

## dual_metropolis_off
| start | closer_rate | ever_close_r1 | mean_d_start | mean_d_end | mean_d_min | turnback_rate | mean_turnbacks | mean_accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| far_outside | 1.000 | 0.000 | 7.239 | 3.377 | 3.361 | 1.000 | 2.00 | 1.000 |
| manifold_1 | 0.000 | 1.000 | 0.000 | 0.075 | 0.000 | 1.000 | 7.00 | 1.000 |
| manifold_2 | 0.000 | 1.000 | 0.000 | 0.199 | 0.000 | 1.000 | 13.50 | 1.000 |
| manifold_3 | 0.000 | 1.000 | 0.000 | 0.066 | 0.000 | 1.000 | 24.50 | 1.000 |
| near_outside | 1.000 | 1.000 | 1.034 | 0.633 | 0.038 | 1.000 | 20.00 | 1.000 |

## dual_metropolis_on
| start | closer_rate | ever_close_r1 | mean_d_start | mean_d_end | mean_d_min | turnback_rate | mean_turnbacks | mean_accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| far_outside | 0.500 | 0.000 | 7.239 | 6.379 | 4.940 | 1.000 | 29.00 | 0.992 |
| manifold_1 | 0.000 | 1.000 | 0.000 | 0.183 | 0.000 | 1.000 | 33.50 | 1.000 |
| manifold_2 | 0.000 | 1.000 | 0.000 | 0.070 | 0.000 | 1.000 | 34.00 | 0.992 |
| manifold_3 | 0.000 | 1.000 | 0.000 | 0.311 | 0.000 | 1.000 | 32.00 | 1.000 |
| near_outside | 1.000 | 1.000 | 1.034 | 0.308 | 0.033 | 1.000 | 32.00 | 1.000 |

## std_metropolis_off
| start | closer_rate | ever_close_r1 | mean_d_start | mean_d_end | mean_d_min | turnback_rate | mean_turnbacks | mean_accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| far_outside | 0.000 | 1.000 | 7.239 | 43.481 | 0.552 | 1.000 | 1.00 | 1.000 |
| manifold_1 | 0.000 | 1.000 | 0.000 | 0.192 | 0.000 | 1.000 | 35.50 | 1.000 |
| manifold_2 | 0.000 | 1.000 | 0.000 | 0.089 | 0.000 | 1.000 | 34.00 | 1.000 |
| manifold_3 | 0.000 | 1.000 | 0.000 | 0.695 | 0.000 | 1.000 | 27.50 | 1.000 |
| near_outside | 0.000 | 0.500 | 1.034 | 42.472 | 0.576 | 1.000 | 6.50 | 1.000 |

## std_metropolis_on
| start | closer_rate | ever_close_r1 | mean_d_start | mean_d_end | mean_d_min | turnback_rate | mean_turnbacks | mean_accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| far_outside | 0.500 | 0.000 | 7.239 | 7.635 | 6.799 | 1.000 | 30.50 | 0.983 |
| manifold_1 | 0.000 | 1.000 | 0.000 | 0.231 | 0.000 | 1.000 | 40.00 | 0.967 |
| manifold_2 | 0.000 | 1.000 | 0.000 | 0.142 | 0.000 | 1.000 | 34.50 | 0.967 |
| manifold_3 | 0.000 | 1.000 | 0.000 | 0.297 | 0.000 | 1.000 | 38.50 | 0.983 |
| near_outside | 1.000 | 1.000 | 1.034 | 0.483 | 0.038 | 1.000 | 39.00 | 0.975 |
