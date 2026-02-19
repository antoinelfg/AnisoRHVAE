# Sampler Alignment Dashboard Summary

Comparison of `volume_riemannian` RHMC on Toy metric vs Real 4K metric.

| Metric | Toy | Real 4K |
|---|---:|---:|
| `acceptance_rate` | 1 | 0.9828 |
| `rescue_rate` | 0.0375 | 0.0125 |
| `median_hit_step` | 8 | 8 |
| `proposal_alignment_outside_mean` | 0.118 | 0.01027 |
| `accepted_alignment_outside_mean` | 0.118 | 0.0004841 |
| `proposal_opposition_rate_outside` | 0.4219 | 0.4984 |
| `accepted_opposition_rate_outside` | 0.4219 | 0.4938 |
| `first_step_abs_alignment_principal_mean` | 0.6388 | 0.8619 |
| `first_step_alignment_to_manifold_mean` | 0.1357 | 0.07319 |
| `delta_logdet_inv_mean` | 1.091 | 0.01519 |
| `delta_logdet_inv_median` | 0.1114 | -0.01194 |
| `escaped_count` | 0 | 0 |

Interpretation:
- `proposal_alignment_outside_mean`: geometry + integrator effect before Metropolis.
- `accepted_alignment_outside_mean`: actual chain displacement alignment.
- `proposal/accepted_opposition_rate_outside`: fraction of outside steps pointing opposite the nearest-manifold direction.
- `first_step_abs_alignment_principal_mean`: first-step alignment with principal mobility axis (signless).
- `first_step_alignment_to_manifold_mean`: first-step alignment toward nearest-manifold direction.
- `rescue_rate`: fraction of chains that reached shell `min_dist <= r0`.