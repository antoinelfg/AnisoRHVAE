# Sampler Alignment Dashboard Summary

Comparison of `volume_riemannian` RHMC on Toy metric vs Real 4K metric.

| Metric | Toy | Real 4K |
|---|---:|---:|
| `acceptance_rate` | 0.9974 | 0.9969 |
| `rescue_rate` | 0.375 | 0 |
| `median_hit_step` | 40.5 | nan |
| `proposal_alignment_outside_mean` | 0.03945 | -0.02734 |
| `accepted_alignment_outside_mean` | 0.03945 | -0.03023 |
| `proposal_opposition_rate_outside` | 0.3661 | 0.5109 |
| `accepted_opposition_rate_outside` | 0.3661 | 0.5109 |
| `first_step_abs_alignment_principal_mean` | 0.5322 | 0.907 |
| `first_step_alignment_to_manifold_mean` | -0.1326 | -0.03674 |
| `delta_logdet_inv_mean` | 5.25 | -0.1908 |
| `delta_logdet_inv_median` | 0.5958 | -0.3078 |
| `escaped_count` | 0 | 0 |

Interpretation:
- `proposal_alignment_outside_mean`: geometry + integrator effect before Metropolis.
- `accepted_alignment_outside_mean`: actual chain displacement alignment.
- `proposal/accepted_opposition_rate_outside`: fraction of outside steps pointing opposite the nearest-manifold direction.
- `first_step_abs_alignment_principal_mean`: first-step alignment with principal mobility axis (signless).
- `first_step_alignment_to_manifold_mean`: first-step alignment toward nearest-manifold direction.
- `rescue_rate`: fraction of chains that reached shell `min_dist <= r0`.