# Sampler Alignment Dashboard Summary

Comparison of `volume_riemannian` RHMC on Toy metric vs Real 4K metric.

| Metric | Toy | Real 4K |
|---|---:|---:|
| `acceptance_rate` | 0.312 | 0.9943 |
| `rescue_rate` | 0.3125 | 0 |
| `median_hit_step` | 2 | nan |
| `proposal_alignment_outside_mean` | 0.9884 | 0.004699 |
| `accepted_alignment_outside_mean` | 0.5321 | 0.005129 |
| `proposal_opposition_rate_outside` | 0.001563 | 0.4849 |
| `accepted_opposition_rate_outside` | 0.001563 | 0.4802 |
| `first_step_abs_alignment_principal_mean` | 0.6526 | 0.2731 |
| `first_step_alignment_to_manifold_mean` | 0.9924 | -0.09275 |
| `delta_logdet_inv_mean` | 4.381 | -0.001711 |
| `delta_logdet_inv_median` | 0 | -0.0007157 |
| `escaped_count` | 0 | 0 |

Interpretation:
- `proposal_alignment_outside_mean`: geometry + integrator effect before Metropolis.
- `accepted_alignment_outside_mean`: actual chain displacement alignment.
- `proposal/accepted_opposition_rate_outside`: fraction of outside steps pointing opposite the nearest-manifold direction.
- `first_step_abs_alignment_principal_mean`: first-step alignment with principal mobility axis (signless).
- `first_step_alignment_to_manifold_mean`: first-step alignment toward nearest-manifold direction.
- `rescue_rate`: fraction of chains that reached shell `min_dist <= r0`.