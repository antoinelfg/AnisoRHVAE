# Sampler Alignment Dashboard Summary

Comparison of `volume_riemannian` RHMC on Toy metric vs Real 4K metric.

| Metric | Toy | Real 4K |
|---|---:|---:|
| `acceptance_rate` | 1 | 0.9406 |
| `rescue_rate` | 0 | 0.125 |
| `median_hit_step` | nan | 112.5 |
| `proposal_alignment_outside_mean` | 0 | 0.003491 |
| `accepted_alignment_outside_mean` | nan | 0.06005 |
| `proposal_opposition_rate_outside` | 0 | 0.4911 |
| `accepted_opposition_rate_outside` | 0 | 0.4328 |
| `first_step_abs_alignment_principal_mean` | 0 | 0.9116 |
| `first_step_alignment_to_manifold_mean` | 0 | -0.02454 |
| `delta_logdet_inv_mean` | 0 | 0.2258 |
| `delta_logdet_inv_median` | 0 | 0.2655 |
| `escaped_count` | 0 | 0 |

Interpretation:
- `proposal_alignment_outside_mean`: geometry + integrator effect before Metropolis.
- `accepted_alignment_outside_mean`: actual chain displacement alignment.
- `proposal/accepted_opposition_rate_outside`: fraction of outside steps pointing opposite the nearest-manifold direction.
- `first_step_abs_alignment_principal_mean`: first-step alignment with principal mobility axis (signless).
- `first_step_alignment_to_manifold_mean`: first-step alignment toward nearest-manifold direction.
- `rescue_rate`: fraction of chains that reached shell `min_dist <= r0`.