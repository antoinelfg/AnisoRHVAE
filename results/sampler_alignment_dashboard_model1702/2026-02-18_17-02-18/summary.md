# Sampler Alignment Dashboard Summary

Comparison of `volume_riemannian` RHMC on Toy metric vs Real 4K metric.

| Metric | Toy | Real 4K |
|---|---:|---:|
| `acceptance_rate` | 0.9982 | 0.9893 |
| `rescue_rate` | 0.25 | 0.125 |
| `median_hit_step` | 58.5 | 35 |
| `proposal_alignment_outside_mean` | 0.08084 | -0.04378 |
| `accepted_alignment_outside_mean` | 0.08084 | -0.05119 |
| `proposal_opposition_rate_outside` | 0.4232 | 0.5125 |
| `accepted_opposition_rate_outside` | 0.4232 | 0.5125 |
| `first_step_abs_alignment_principal_mean` | 0.5516 | 0.8957 |
| `first_step_alignment_to_manifold_mean` | 0.01818 | 0.7227 |
| `delta_logdet_inv_mean` | 5.279 | -0.1083 |
| `delta_logdet_inv_median` | 2.45 | -0.1191 |
| `escaped_count` | 0 | 0 |

Interpretation:
- `proposal_alignment_outside_mean`: geometry + integrator effect before Metropolis.
- `accepted_alignment_outside_mean`: actual chain displacement alignment.
- `proposal/accepted_opposition_rate_outside`: fraction of outside steps pointing opposite the nearest-manifold direction.
- `first_step_abs_alignment_principal_mean`: first-step alignment with principal mobility axis (signless).
- `first_step_alignment_to_manifold_mean`: first-step alignment toward nearest-manifold direction.
- `rescue_rate`: fraction of chains that reached shell `min_dist <= r0`.