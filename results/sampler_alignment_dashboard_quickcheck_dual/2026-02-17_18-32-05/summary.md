# Sampler Alignment Dashboard Summary

Comparison of `volume_riemannian` RHMC on Toy metric vs Real 4K metric.

| Metric | Toy | Real 4K |
|---|---:|---:|
| `acceptance_rate` | 0.3604 | 0.9875 |
| `rescue_rate` | 0.375 | 0 |
| `median_hit_step` | 3 | nan |
| `proposal_alignment_outside_mean` | 0.9861 | -0.03432 |
| `accepted_alignment_outside_mean` | 0.6434 | -0.03243 |
| `proposal_opposition_rate_outside` | 0.002083 | 0.5167 |
| `accepted_opposition_rate_outside` | 0.002083 | 0.5062 |
| `first_step_abs_alignment_principal_mean` | 0.6405 | 0.3448 |
| `first_step_alignment_to_manifold_mean` | 0.9946 | -0.1465 |
| `delta_logdet_inv_mean` | 6.037 | -0.01939 |
| `delta_logdet_inv_median` | 0 | -0.02076 |
| `escaped_count` | 0 | 0 |

Interpretation:
- `proposal_alignment_outside_mean`: geometry + integrator effect before Metropolis.
- `accepted_alignment_outside_mean`: actual chain displacement alignment.
- `proposal/accepted_opposition_rate_outside`: fraction of outside steps pointing opposite the nearest-manifold direction.
- `first_step_abs_alignment_principal_mean`: first-step alignment with principal mobility axis (signless).
- `first_step_alignment_to_manifold_mean`: first-step alignment toward nearest-manifold direction.
- `rescue_rate`: fraction of chains that reached shell `min_dist <= r0`.