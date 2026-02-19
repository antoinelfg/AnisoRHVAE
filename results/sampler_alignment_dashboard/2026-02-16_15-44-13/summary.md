# Sampler Alignment Dashboard Summary

Comparison of `volume_riemannian` RHMC on Toy metric vs Real 4K metric.

| Metric | Toy | Real 4K |
|---|---:|---:|
| `acceptance_rate` | 1 | 0.9922 |
| `rescue_rate` | 0 | 0 |
| `median_hit_step` | nan | nan |
| `proposal_alignment_outside_mean` | 0 | 0.01401 |
| `accepted_alignment_outside_mean` | nan | 0.01385 |
| `proposal_opposition_rate_outside` | 0 | 0.2604 |
| `accepted_opposition_rate_outside` | 0 | 0.2589 |
| `first_step_abs_alignment_principal_mean` | 0 | 0.03358 |
| `first_step_alignment_to_manifold_mean` | 0 | 0.05216 |
| `delta_logdet_inv_mean` | 0 | -0.005055 |
| `delta_logdet_inv_median` | 0 | -0.007532 |
| `escaped_count` | 0 | 0 |

Interpretation:
- `proposal_alignment_outside_mean`: geometry + integrator effect before Metropolis.
- `accepted_alignment_outside_mean`: actual chain displacement alignment.
- `proposal/accepted_opposition_rate_outside`: fraction of outside steps pointing opposite the nearest-manifold direction.
- `first_step_abs_alignment_principal_mean`: first-step alignment with principal mobility axis (signless).
- `first_step_alignment_to_manifold_mean`: first-step alignment toward nearest-manifold direction.
- `rescue_rate`: fraction of chains that reached shell `min_dist <= r0`.