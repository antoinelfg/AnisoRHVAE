# Sampler Alignment Dashboard Summary

Comparison of `volume_riemannian` RHMC on Toy metric vs Real 4K metric.

| Metric | Toy | Real 4K |
|---|---:|---:|
| `acceptance_rate` | 0.999 | 0.9979 |
| `rescue_rate` | 1 | 0 |
| `median_hit_step` | 1 | nan |
| `proposal_alignment_outside_mean` | 0.1256 | -0.02744 |
| `accepted_alignment_outside_mean` | 0.1256 | -0.02946 |
| `first_step_abs_alignment_principal_mean` | 0.5034 | 0.9092 |
| `first_step_alignment_to_manifold_mean` | 0.03959 | -0.03698 |
| `delta_logdet_inv_mean` | 2.899 | -0.1212 |
| `delta_logdet_inv_median` | 3.209 | -0.1975 |
| `escaped_count` | 0 | 0 |

Interpretation:
- `proposal_alignment_outside_mean`: geometry + integrator effect before Metropolis.
- `accepted_alignment_outside_mean`: actual chain displacement alignment.
- `first_step_abs_alignment_principal_mean`: first-step alignment with principal mobility axis (signless).
- `first_step_alignment_to_manifold_mean`: first-step alignment toward nearest-manifold direction.
- `rescue_rate`: fraction of chains that reached shell `min_dist <= r0`.