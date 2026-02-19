# Metric Dashboard Scorecard

## Diagnostic Definitions

- `SPD validity`: fraction of sampled points where `G_inv` is not positive definite.
- `Volume contrast`: median(log det G_inv) on manifold minus far-void median.
- `Monotonic shells`: manifold > transition > far-void for log det(G_inv).
- `Void force alignment`: cosine between volume force and vector to nearest centroid outside manifold.
- `Void opposition rate`: fraction of outside points where force points away from nearest centroid.
- `Plateau outside`: outside fraction with near-flat volume force norm.
- `Border overshoot`: transition-shell determinant overshoot indicator (should stay <= 0).
- `Tangent coherence`: alignment between principal metric direction and local manifold tangent.

## Scorecard

| Check | Value | Objective | Good If | Pass |
|---|---:|---|---|:---:|
| SPD validity | 0 | Metric must stay SPD in sampled space | rate <= 1e-4 | PASS |
| Volume contrast | 2.425 | Manifold should have higher log det(G_inv) than far void | contrast > 0 | PASS |
| Monotonic shells | true | Determinant should decrease from manifold to void | True | PASS |
| Void force alignment | 0.9494 | Volume force should point toward manifold on average outside | mean cosine > 0 | PASS |
| Void opposition rate | 0.02434 | Too many opposite arrows indicate unstable rescue dynamics | rate < 0.35 | PASS |
| Plateau outside | 0.001191 | Avoid flat gradients in void | fraction < 0.35 | PASS |
| Border overshoot | -1.958 | Transition shell should not overshoot manifold and void medians | index <= 0 | PASS |
| Tangent coherence | 0.7301 | Principal metric direction should match local tangent near manifold | alignment > 0.55 | PASS |

## Metadata

- `model_path`: `outputs/reference_models/4K`
- `dims`: `[0, 1]`
- `grid_n`: `160`
- `quiver_n`: `22`
- `volume_power`: `0.8`
- `r0`: `0.9553154706954956`
- `timestamp`: `2026-02-16_14-46-14`