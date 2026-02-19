# Sampler Alignment Dashboard Summary

Comparison of `volume_riemannian` RHMC on Toy metric vs Real 4K metric.

| Metric | Toy | Real 4K |
|---|---:|---:|
| `acceptance_rate` | 0 | 0.1812 |
| `rescue_rate` | 0 | 1 |
| `median_hit_step` | nan | 7 |
| `proposal_alignment_outside_mean` | 0.9977 | 0.1214 |
| `accepted_alignment_outside_mean` | nan | 0.3509 |
| `proposal_opposition_rate_outside` | 0 | 0.04896 |
| `accepted_opposition_rate_outside` | 0 | 0.02708 |
| `first_step_abs_alignment_principal_mean` | 0.6356 | 0.9988 |
| `first_step_alignment_to_manifold_mean` | 0.9978 | -0.1017 |
| `delta_logdet_inv_mean` | 0 | 3.757 |
| `delta_logdet_inv_median` | 0 | 3.945 |
| `escaped_count` | 0 | 0 |

Interpretation:
- `proposal_alignment_outside_mean`: geometry + integrator effect before Metropolis.
- `accepted_alignment_outside_mean`: actual chain displacement alignment.
- `proposal/accepted_opposition_rate_outside`: fraction of outside steps pointing opposite the nearest-manifold direction.
- `first_step_abs_alignment_principal_mean`: first-step alignment with principal mobility axis (signless).
- `first_step_alignment_to_manifold_mean`: first-step alignment toward nearest-manifold direction.
- `rescue_rate`: fraction of chains that reached shell `min_dist <= r0`.