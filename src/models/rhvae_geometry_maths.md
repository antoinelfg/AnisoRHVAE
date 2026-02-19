# Inverse metric maths (`_compute_inverse_metric_at_z`)


Notation: $z \in \mathbb{R}^d$ (latent), centroids $\{c_k\}_{k=1}^K$, covariances $\Sigma_k$ from atoms $M_k$, temperature $\tau$, regulariser $\lambda$.

---

## 1. Base inverse metric $G_{\mathrm{base}}^{-1}(z)$

RBF mixture of covariances plus regulariser:

**Kernel distances** (per centroid $k$):

- **Isotropic:**  
  $d_k^2(z) = \|z - c_k\|^2$

- **Mahalanobis:**  
  $d_k^2(z) = (z - c_k)^\top P_k\, (z - c_k)$, with $P_k = \Sigma_k^{-1}$

**Weights:**

$$
w_k(z) = \exp\bigl(-d_k^2(z) / \tau^2\bigr)
$$

**Base inverse metric:**

$$
G_{\mathrm{base}}^{-1}(z)
= \sum_{k=1}^K w_k(z)\, \Sigma_k
+ \lambda I
$$

(Code: `_compute_base_inverse_metric`; then `_stabilize_metric` is applied.)

---

## 2. Void inverse metric $G_{\mathrm{void}}^{-1}(z)$

Radial direction from $z$ toward centroid(s), then stiff along that direction and soft in the transverse directions.

**Direction(s):**

- **Hard:** $u(z) = \frac{c_{k^*}(z) - z}{\|c_{k^*}(z) - z\|}$ with $k^* = \mathrm{argmin}_k\, d_k^2(z)$ (Euclidean or Mahalanobis).
- **Soft (gravity well / smooth attractor):**
  $$
  R(z)=\sum_k \pi_k(z)\,u_k(z)u_k(z)^\top,\qquad
  u_k(z)=\frac{c_k-z}{\|c_k-z\|},
  $$
  where the **weights** are
  $$
  \pi_k(z) = \frac{\exp\!\bigl(-E_k(z)\bigr)}{\sum_j \exp\!\bigl(-E_j(z)\bigr)}.
  $$
  **Soft attractor (no det):**
  $$
  E_k(z)=\gamma\,d_k^2(z).
  $$
  **Gravity well (with det):**
  $$
  E_k(z)=\gamma\,d_k^2(z)+\tfrac{1}{2}\log\det\Sigma_k
  \quad(\text{equivalently } \pi_k\propto \exp(-\gamma d_k^2)/\sqrt{\det\Sigma_k}).
  $$

So the **radial projector** is

$$
R(z) = u(z) u(z)^\top \quad \in \mathbb{R}^{d \times d}.
$$

**Distance used for decay/transition:**  
$r(z) = \min_k \|z - c_k\|$ (Euclidean).

**Void inverse metric:**

$$
G_{\mathrm{void}}^{-1}(z)
= \beta_{\mathrm{long}}\cdot R(z)
+ \lambda\cdot \psi\bigl(r(z)\bigr)\cdot I.
$$

- $\beta_{\mathrm{long}} = \texttt{radial\_stretch}$ (longitudinal eigenvalue).
- $\psi(r) = \texttt{void\_decay}$ (transverse decay, see below); in code, **term_long** uses decay_longitudinal $=1$, **term_trans** uses $\psi(r)$ so that transversely the metric decays with distance.

So:

$$
\boxed{
G_{\mathrm{void}}^{-1}(z)
= \beta\, R(z)
+ \lambda\, \psi(r)\, I.
}
$$

### 2.1 Determinant-preserving spectral reshaping in void (optional)

To preserve the volume term while changing directional anisotropy, we can reshape
the void branch spectrally.

Let
$$
G_{\mathrm{void}}^{-1}(z)=Q(z)\,\mathrm{diag}(\mu_i(z))\,Q(z)^\top,\qquad \mu_i(z)>0.
$$

Define the geometric mean of eigenvalues:
$$
g(z)=\exp\!\left(\frac{1}{d}\sum_{i=1}^d \log \mu_i(z)\right).
$$

For power $s=\texttt{void\_eigshape\_power}$, define reshaped eigenvalues:
$$
\mu_i^{(s)}(z)=g(z)\left(\frac{\mu_i(z)}{g(z)}\right)^s.
$$

Then
$$
G_{\mathrm{void},2}^{-1}(z)=Q(z)\,\mathrm{diag}\!\bigl(\mu_i^{(s)}(z)\bigr)\,Q(z)^\top.
$$

Determinant is preserved pointwise:
$$
\det G_{\mathrm{void},2}^{-1}(z)=\det G_{\mathrm{void}}^{-1}(z),
$$
so $\log\det G^{-1}$ and its gradient contribution are preserved.

Gating by blend coefficient:
- if $\alpha(z)\ge \alpha_{\min}$ (`void_eigshape_alpha_min`), use $G_{\mathrm{void},2}^{-1}(z)$;
- else keep $G_{\mathrm{void}}^{-1}(z)$.

Mode `void_eigshape_mode=none` disables reshaping (default).  
With $s=-1$, anisotropy is fully inverted around the geometric-mean scale.

---

## 3. Void decay $\psi(r)$

**If $\texttt{void\_decay\_type} = \texttt{"none"}$:**  
$\psi(r) = 1$.

**Otherwise:**  
$r_0$ is a scale (from `void_threshold` or from `void_weight_threshold` via $\tau\sqrt{-\log\tau}$), then

$$
\delta = r - r_0,\qquad
\tilde\delta = \mathrm{softplus}(\delta; k),
$$

$$
\psi(r)
= \frac{1}{1 + \bigl(\tilde\delta / s\bigr)^p},
\qquad s = \texttt{void\_decay\_scale},\quad p = \texttt{void\_decay\_power}.
$$

So $\psi \approx 1$ when $r \lesssim r_0$, and $\psi \to 0$ as $r$ grows (transverse stiffness decays).

---

## 4. Blend coefficient $\alpha(z)$

$$
r_0 = \texttt{\_compute\_r0}(r(z)),
\qquad
\alpha(z) = \sigma\bigl( (r(z) - r_0)\, \texttt{transition\_steepness} \bigr),
$$

with $\sigma$ the sigmoid. So:

- $r \ll r_0$: $\alpha \approx 1$ → use void.
- $r \gg r_0$: $\alpha \approx 0$ → use base.

---

## 5. Final inverse metric $G^{-1}(z)$

If $\texttt{use\_attractor}$ is False:

$$
G^{-1}(z) = \mathrm{stabilize}\bigl( G_{\mathrm{base}}^{-1}(z) \bigr).
$$

If True:

$$
\boxed{
G^{-1}(z)
= \mathrm{stabilize}\Bigl(
  \bigl(1 - \alpha(z)\bigr)\, G_{\mathrm{base}}^{-1}(z)
  + \alpha(z)\, G_{\mathrm{void}}^{-1}(z)
\Bigr).
}
$$

Then the metric used in the model is $G(z) = \bigl(G^{-1}(z)\bigr)^{-1}$ (see `_compute_metric_at_z`).

---

## Summary

| Quantity | Formula |
|----------|---------|
| Base | $G_{\mathrm{base}}^{-1} = \sum_k w_k \Sigma_k + \lambda I$, $w_k = \exp(-d_k^2/\tau^2)$ |
| Attractor weights (soft) | $\pi_k \propto \exp\!\bigl(-(\gamma d_k^2 + \tfrac{1}{2}\log\det\Sigma_k)\bigr)$ (det term only if enabled) |
| Radial projector | $R(z)=\sum_k \pi_k\, u_k u_k^\top$, $u_k=(c_k-z)/\|c_k-z\|$ (soft) |
| Void | $G_{\mathrm{void}}^{-1} = \beta\, R(z) + \lambda\, \psi(r)\, I$ |
| Void (spectral-shaped) | $G_{\mathrm{void},2}^{-1}=Q\,\mathrm{diag}(\mu_i^{(s)})Q^\top,\ \mu_i^{(s)}=g(\mu_i/g)^s,\ \det$ preserved |
| Decay | $\psi(r) = 1/\bigl(1 + (\tilde\delta/s)^p\bigr)$ |
| Blend | $\alpha = \sigma\bigl((r - r_0)\,\kappa\bigr)$ |
| Final | $G^{-1} = (1-\alpha)\, G_{\mathrm{base}}^{-1} + \alpha\, \widetilde{G}_{\mathrm{void}}^{-1}$ (then stabilised), with $\widetilde{G}_{\mathrm{void}}^{-1}\in\{G_{\mathrm{void}}^{-1}, G_{\mathrm{void},2}^{-1}\}$ |

So near centroids the latent metric is stiff toward the centroid and soft in the transverse directions; far away it reverts to the standard RBF mixture.

---

# Attractor weights (gamma)

For **soft attractor** the centroid weights use a softmax over an energy:

$$
E_k(z) = \gamma\, d_k^2(z) \;+\; \tfrac{1}{2}\log\det \Sigma_k
\quad\text{(det term only if `attractor-use-det=True`)}
$$

$$
\pi_k(z) = \frac{\exp(-E_k(z))}{\sum_j \exp(-E_j(z))}
$$

If `attractor_k_nearest` is set, only the top‑$k$ weights are kept and renormalized.

**Code:** `GeometryRHVAE._compute_soft_attractor_weights` in `src/models/rhvae_geometry.py`.

---

# Riemannian Sampling & Physics

This section details the sampling objectives and the exact Hamiltonians used by our RHMC samplers.

## 1. General Mathematical Framework

To sample from a target density $\pi(z)$ on a Riemannian manifold with metric $G(z)$, we introduce a momentum variable $\rho$ and simulate Hamiltonian dynamics. The Hamiltonian depends on the **momentum choice**.

### The General Hamiltonian

We start from the joint density $p(z,\rho)$ and define:

#### Option A: Riemannian Momentum (standard)

$$
\rho \sim \mathcal{N}(0,G(z)).
$$

Then
$$
p(z,\rho)\propto \pi(z)\,|G(z)|^{-1/2}\exp\!\Bigl(-\tfrac{1}{2}\rho^\top G^{-1}(z)\rho\Bigr),
$$
so the Hamiltonian is
$$
H(z,\rho)= -\log\pi(z)+\tfrac{1}{2}\rho^\top G^{-1}(z)\rho+\tfrac{1}{2}\log\det G(z).
$$

#### Option B: Euclidean Momentum (stable)

$$
\rho \sim \mathcal{N}(0,I).
$$

Then
$$
p(z,\rho)\propto \pi(z)\exp\!\Bigl(-\tfrac{1}{2}\rho^\top\rho\Bigr),
$$
so the Hamiltonian is
$$
H(z,\rho)= -\log\pi(z)+\tfrac{1}{2}\rho^\top\rho.
$$

---

## 2. Deriving Our Samplers

By selecting a specific target density $\pi(z)$ and momentum choice, we recover each sampler in the codebase.

### Case 1: Geodesic Sampler (uniform manifold sampling)

- **Target:** $\pi(z)\propto \sqrt{\det G(z)}$ (Riemannian volume element).
- **Momentum:** Riemannian (Option A).

**Derivation:** the $\tfrac{1}{2}\log\det G$ term cancels $-\log\pi$.

**Hamiltonian:**
$$
H(z,\rho)=\tfrac{1}{2}\rho^\top G^{-1}(z)\rho.
$$

**Code:** `GeodesicHMCSampler` in `src/models/samplers/hmc_sampler.py`.

---

### Case 2: Standard Riemannian HMC (Gaussian prior)

- **Target:** $\pi(z)\propto \exp(-\tfrac{1}{2}\|z\|^2)$.
- **Momentum:** Riemannian (Option A).

**Hamiltonian:**
$$
H(z,\rho)=\tfrac{1}{2}\|z\|^2+\tfrac{1}{2}\rho^\top G^{-1}(z)\rho+\tfrac{1}{2}\log\det G(z).
$$

**Code:** `RiemannianHMCSampler` in `src/models/samplers/hmc_sampler.py`.

---

### Case 3: "Gravity Well" (volume-element sampling)

- **Target:** $\pi(z)\propto \det(G^{-1}(z))^{\alpha}$ (default $\alpha=\tfrac{1}{2}$).

#### Option B (Euclidean momentum) — **used by default**

$$
H(z,\rho)= -\alpha\log\det G^{-1}(z)+\tfrac{1}{2}\rho^\top\rho.
$$

**Code:** `RHVAEVolumeElementHMCSampler`.

#### Option A (Riemannian momentum)

$$
H(z,\rho)=\tfrac{1}{2}\rho^\top G^{-1}(z)\rho-(\alpha+\tfrac{1}{2})\log\det G^{-1}(z).
$$

**Code:** `VolumeElementRiemannianHMCSampler` (optional).

---

## 3. Summary of Integrators

| Sampler | Kinetic Energy | Integrator Type | Code Reference |
| --- | --- | --- | --- |
| Geodesic | Riemannian | **Implicit** generalized leapfrog (exact) | `GeodesicHMCSampler._generalized_leapfrog_step` |
| Standard RHMC | Riemannian | **Implicit** generalized leapfrog (exact) | `RiemannianHMCSampler._generalized_leapfrog_step` |
| Gravity well (volume) | Euclidean | Standard leapfrog | `RHVAEVolumeElementHMCSampler._leapfrog` |
| Gravity well (volume, Riemannian) | Riemannian | **Implicit** generalized leapfrog (exact) | `VolumeElementRiemannianHMCSampler` |

**Notes:**
- Exact Riemannian integrators require the full $\nabla_z H$ (including kinetic-gradient terms).
- Euclidean integrators only require $\nabla_z\log\det G^{-1}$ (via autograd).
- Tempering (momentum rescaling) is disabled by default for exactness. Set `exact=False` to use legacy tempered dynamics (approximate).

---

## 4. Additional Samplers

### Dual Riemannian HMC
- **Target:** $\pi(z)\propto \exp(-\tfrac{1}{2}\|z\|^2)$.
- **Momentum:** $\rho \sim \mathcal{N}(0, G^{-1}(z))$.
- **Hamiltonian:** $H=\tfrac{1}{2}\|z\|^2+\tfrac{1}{2}\rho^\top G(z)\rho+\tfrac{1}{2}\log\det G^{-1}(z)$.
- **Code:** `DualRiemannianHMCSampler` (implicit generalized leapfrog).

### Manifold Attraction (Exact vs Legacy)
- **Exact (default):** RMHMC with metric $M=G^{-1}$, target $\pi(z)\propto \sqrt{\det G^{-1}(z)}$. The Hamiltonian reduces to kinetic only, $H=\tfrac{1}{2}\rho^\top G(z)\rho$.
- **Legacy (approximate):** Euclidean kinetic with $U(z)=-\tfrac{1}{2}\log\det G^{-1}(z)$ and $z$-updates using $G$.
- **Code:** `ManifoldAttractionHMCSampler` (`exact=True|False`).

---

## 5. Training Integrators (implicit/explicit)

Training uses the **general RHMC Hamiltonian** with $U(z)=-\log p(x|z)$ and the metric correction. We switch between implicit and explicit generalized leapfrog:

- **Implicit:** `_generalized_leapfrog_implicit`  
- **Explicit:** `_leap_step_1/_leap_step_2/_leap_step_3`

**Code:** `src/models/rhvae_geometry.py` (`forward`).

---

# Where these samplers are used

- **Sampling diagnostics (FID, multi-start, chains):**  
  `scripts/sampling_diagnostics.py`  
  CLI flags: `--fid_samplers`, `--multi_start_sampler`, `--rhmc_sampler`, `--quality_samplers`.
- **Metric analysis / RHMC chain plots:**  
  `scripts/analyze_metric_full.py` (`--rhmc_sampler`, default = `volume`).
- **Training run defaults:**  
  `scripts/run_pythae_rhvae_baseline.py` (analysis sampler defaults to `volume`, FID samplers default to `gaussian volume`).
