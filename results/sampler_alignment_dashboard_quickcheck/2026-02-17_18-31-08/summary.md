# Sampler Alignment Dashboard Summary

Comparison of `volume_riemannian` RHMC on Toy metric vs Real 4K metric.

| Metric | Toy | Real 4K |
|---|---:|---:|
| `acceptance_rate` | 0.9981 | 0.9898 |
| `rescue_rate` | 0.3333 | 0.08333 |
| `median_hit_step` | 38 | 35 |
| `proposal_alignment_outside_mean` | 0.09922 | -0.01969 |
| `accepted_alignment_outside_mean` | 0.0982 | -0.02555 |
| `proposal_opposition_rate_outside` | 0.3546 | 0.5074 |
| `accepted_opposition_rate_outside` | 0.3546 | 0.5056 |
| `first_step_abs_alignment_principal_mean` | 0.602 | 0.9614 |
| `first_step_alignment_to_manifold_mean` | 0.0971 | 0.1228 |
| `delta_logdet_inv_mean` | 4.136 | -0.06294 |
| `delta_logdet_inv_median` | 0.8682 | -0.0393 |
| `escaped_count` | 0 | 0 |

Interpretation:
- `proposal_alignment_outside_mean`: geometry + integrator effect before Metropolis.
- `accepted_alignment_outside_mean`: actual chain displacement alignment.
- `proposal/accepted_opposition_rate_outside`: fraction of outside steps pointing opposite the nearest-manifold direction.
- `first_step_abs_alignment_principal_mean`: first-step alignment with principal mobility axis (signless).
- `first_step_alignment_to_manifold_mean`: first-step alignment toward nearest-manifold direction.
- `rescue_rate`: fraction of chains that reached shell `min_dist <= r0`.