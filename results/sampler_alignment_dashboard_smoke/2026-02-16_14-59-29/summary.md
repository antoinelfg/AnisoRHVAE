# Sampler Alignment Dashboard Summary

Comparison of `volume_riemannian` RHMC on Toy metric vs Real 4K metric.

| Metric | Toy | Real 4K |
|---|---:|---:|
| `acceptance_rate` | 1 | 1 |
| `rescue_rate` | 0 | 0 |
| `median_hit_step` | nan | nan |
| `proposal_alignment_outside_mean` | 0.002678 | -0.03471 |
| `accepted_alignment_outside_mean` | 0.002678 | -0.03471 |
| `delta_logdet_inv_mean` | 0 | -0.003656 |
| `delta_logdet_inv_median` | 0 | -0.02096 |
| `escaped_count` | 0 | 0 |

Interpretation:
- `proposal_alignment_outside_mean`: geometry + integrator effect before Metropolis.
- `accepted_alignment_outside_mean`: actual chain displacement alignment.
- `rescue_rate`: fraction of chains that reached shell `min_dist <= r0`.