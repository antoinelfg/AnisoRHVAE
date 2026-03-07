# Riemannian Metric Formulation and Formal Analysis


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

## 2. Asymptotic Radial Extrapolation Metric $G_{\mathrm{asym}}^{-1}(z)$

To ensure stable Hamiltonian dynamics in low-density regions far from the training data, the metric must limit transverse exploration while allowing longitudinal movement towards the data manifold. We formalize this via a piecewise extrapolation.

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

**Asymptotic Extrapolation Metric:**

$$
G_{\mathrm{asym}}^{-1}(z)
= \beta_{\mathrm{long}}\cdot R(z)
+ \lambda\cdot \psi\bigl(r(z)\bigr)\cdot I.
$$

- $\beta_{\mathrm{long}} = \texttt{radial\_stretch}$ (longitudinal eigenvalue).
- $\psi(r) = \texttt{void\_decay}$ (transverse decay, see below); in code, this guarantees that transversely the metric decays with distance.

So:

$$
\boxed{
G_{\mathrm{asym}}^{-1}(z)
= \beta\, R(z)
+ \lambda\, \psi(r)\, I.
}
$$



## 3. Asymptotic Decay $\psi(r)$

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

- $r \ll r_0$: $\alpha \approx 0$ → use base.
- $r \gg r_0$: $\alpha \approx 1$ → use asymptotic extrapolation.

### 4.1 Condition de Lipschitz pour le blending (point fixe)

La transition base/void est gouvernée par
$$
\alpha(z)=\sigma\!\bigl(\kappa(r(z)-r_0)\bigr),\qquad \kappa=\texttt{transition\_steepness}.
$$
Sa dérivée spatiale vérifie
$$
\nabla_z \alpha(z)=\kappa\,\alpha(z)\bigl(1-\alpha(z)\bigr)\,\nabla_z r(z),
\qquad
\|\nabla_z\alpha(z)\|\le \frac{|\kappa|}{4}\,\|\nabla_z r(z)\|.
$$

Comme
$$
G^{-1}(z)=(1-\alpha(z))G^{-1}_{\mathrm{base}}(z)+\alpha(z)G_{\mathrm{asym}}^{-1}(z),
$$
une pente de transition trop raide (grand `transition_steepness`) augmente la variation locale de la métrique et peut faire perdre la contraction des itérations de point fixe de l'intégrateur implicite (dépendant de `fp_damping`, `fp_steps`, `eps_lf`).

En pratique: si l'opérateur implicite n'est plus contractant, les itérations internes saturent (`momentum_fp_saturation_rate`, `position_fp_saturation_rate`), l'énergie dérive et les rejets Metropolis augmentent.

### 4.2 Parameterization and Theoretical Bounds

Rather than treating the hyperparameters ($r_0$, $\kappa$, $\tau$) as arbitrary tuning knobs, they are bounded by statistical and numerical constraints:

1. **The Threshold $r_0$ (Confidence Boundary):** Given the base metric weights $w_k = \exp(-d_k^2 / \tau^2)$, transitioning at an exact confidence boundary $\tau_{\mathrm{weight}}$ implies an algebraic connection:
   $$ \exp\left(-\frac{r_0^2}{\tau^2}\right) = \tau_{\mathrm{weight}} \implies r_0 = \tau \sqrt{-\ln(\tau_{\mathrm{weight}})} $$
   In our empirical benchmark configurations (e.g., standard anisotropic geometry `core4`), we specify this threshold via an empirical grid search ($r_0 = 1.2$), but mathematically this corresponds precisely to choosing a specific statistical confidence boundary.
2. **The Steepness $\kappa$ (Curvature Stability):** As the coefficient $\kappa \to \infty$, the transition becomes discontinuous, sending HMC integration to infinity. In practice, $\kappa$ is bounded globally by the numerical contraction limits of the Generalized Leapfrog integrator (formalized in Theorem 2). We empirically isolate an optimum at $\kappa \approx 7.27$ for peak stability, tightly respecting this theoretical upper bound.
3. **The Temperature Scale $\tau$:** The adaptive temperature is bounded from below by Silverman's Rule of Thumb for Kernel Density Estimation: $\tau \ge \hat{\sigma} \left( \frac{4}{d+2} \right)^{\frac{1}{d+4}} N^{-\frac{1}{d+4}}$. This theoretically guarantees that the local metric atoms overlap sufficiently to maintain a connected Riemannian manifold topology.

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
  + \alpha(z)\, G_{\mathrm{asym}}^{-1}(z)
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
| Asymptotic | $G_{\mathrm{asym}}^{-1} = \beta\, R(z) + \lambda\, \psi(r)\, I$ |
| Decay | $\psi(r) = 1/\bigl(1 + (\tilde\delta/s)^p\bigr)$ |
| Blend | $\alpha = \sigma\bigl((r - r_0)\,\kappa\bigr)$ |
| Final | $G^{-1} = (1-\alpha)\, G_{\mathrm{base}}^{-1} + \alpha\, G_{\mathrm{asym}}^{-1}$ (then stabilised) |

So near centroids the latent metric stays close to the base RBF mixture; far from centroids it smoothly transitions to the anisotropic asymptotic branch.

---

## 6. Theoretical Guarantees (NeurIPS Validation)

### Theorem 1: Global Positive Definiteness
**Statement:** The inverse metric $G^{-1}(z)$ is strictly positive definite (SPD) everywhere in $\mathbb{R}^d$.
**Proof Sketch:** 
1. The base metric $G_{\mathrm{base}}^{-1} = \sum w_k \Sigma_k + \lambda I \succeq \lambda I \succ 0$ because all $\Sigma_k$ are SPD by construction (as covariances).
2. The radial projector $R(z) = u(z)u(z)^\top$ is positive semi-definite (PSD) with eigenvalues 1 and 0.
3. The asymptotic metric $G_{\mathrm{asym}}^{-1}(z) = \beta R(z) + \lambda \psi(r) I$. Since $\psi(r) > 0$ globally, $\lambda \psi(r) I \succ 0$, so $G_{\mathrm{asym}}^{-1} \succ 0$.
4. The final metric is a convex combination weighted by $\alpha(z) \in (0, 1)$. The sum of two SPD matrices is SPD. Finally, `_stabilize_metric` ensures numerical eigenvalue flooring. Thus $G^{-1}(z) \succ 0$ is guaranteed globally.

### Theorem 2: Integrator Stability via Lipschitz Decay
**Statement:** The implicit Generalized Leapfrog algorithm for solving Hamiltonian dynamics diverges if the smooth blending function $\alpha(z)$ violates local Lipschitz bounds.
**Proof Sketch:**
1. The Leapfrog momentum update defines a fixed-point iteration: $\rho_{t+1} = \rho_t - \frac{\epsilon}{2} \nabla_z H(z, \rho_{t+1})$.
2. The contraction mapping theorem guarantees convergence iff the spectral norm of the Hamiltonian Hessian $\|\nabla_{zz}^2 H\|_2 \le \frac{2}{\epsilon}$.
3. Because $H$ depends on the spatial gradient of the metric $\nabla_z G^{-1}$, the curvature incorporates $\nabla_{zz}^2 \alpha(z)$.
4. If $\kappa = \texttt{transition\_steepness}$ is arbitrarily large, $\|\nabla_{zz}^2 \alpha(z)\| \to \infty$, breaking the contraction limit $\frac{2}{\epsilon}$. The code prevents this strictly via damping (`fp_damping`), multi-step iterations (`fp_steps`), and controlling $\kappa$.

### Theorem 3: Bounded Kinetic Energy via Dual RMHMC
**Statement:** In highly anisotropic asymptotic regions, standard Riemannian HMC ($\rho \sim \mathcal{N}(0, G(z))$) draws infinite momentum variance. The Dual formulation ($\rho \sim \mathcal{N}(0, G^{-1}(z))$) guarantees bounded momentum variance and stable trajectories.
**Proof Sketch:**
1. In the asymptotic region, $G_{\mathrm{asym}}^{-1}$ has longitudinal eigenvalue $\beta$ and transverse eigenvalues $\lambda \psi(r) \to 0$ as $r \to \infty$. 
2. Because $G(z) = (G^{-1}(z))^{-1}$, the transverse eigenvalues of the standard Riemannian covariance $G(z)$ diverge to $1 / (\lambda \psi(r)) \to \infty$. Sampling $\rho \sim \mathcal{N}(0, G(z))$ yields infinite momentum, breaking numerical solvers.
3. The Dual formulation uses $M(z) = G^{-1}(z)$ as the mass matrix. By Theorem 1, $G^{-1}(z)$ is globally bounded from above by the RBF mixture and from below by $\lambda \psi(r) > 0$. Therefore, $\rho \sim \mathcal{N}(0, G^{-1}(z))$ always produces finite, well-behaved momentum vectors.
4. Hamilton's equations $\dot{z} = \nabla_\rho H = G(z) \rho$ still evaluate the true metric $G(z)$, but the integration is stabilized because the Euclidean kinetic jumps are bounded by the adaptive dual displacement step limits (`adaptive_max_dual_displacement`).

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

- Let $\nu=\texttt{volume\_power}$ and $\lambda_r=\texttt{radial\_prior\_weight}\ge 0$.
- **Target density (implemented):**
$$
\pi(z)\propto \det(G^{-1}(z))^{\nu}\,
\exp\!\Bigl(-\tfrac{\lambda_r}{2}\|z-c\|^2\Bigr),
$$
with $c=\texttt{radial\_prior\_center}$ (or $0$ if unset).

Hence
$$
-\log \pi(z)= -\nu\log\det G^{-1}(z)+\tfrac{\lambda_r}{2}\|z-c\|^2.
$$

#### Option B (Euclidean momentum) — **used by default**

$$
H(z,\rho)= -\nu\log\det G^{-1}(z)+\tfrac{\lambda_r}{2}\|z-c\|^2+\tfrac{1}{2}\rho^\top\rho.
$$

**Code:** `RHVAEVolumeElementHMCSampler`.

#### Option A (Riemannian momentum)

With RHMC, the exact Hamiltonian is
$$
H(z,\rho)= -\log\pi(z)+\tfrac{1}{2}\rho^\top M^{-1}(z)\rho+\tfrac{1}{2}\log\det M(z),
$$
where $M(z)$ is the mass matrix convention.

**Standard convention** (`mass_mode="standard"`, $M=G$):

$$
H_{\mathrm{std}}(z,\rho)=
\tfrac{1}{2}\rho^\top G^{-1}(z)\rho
-(\nu+\tfrac{1}{2})\log\det G^{-1}(z)
+\tfrac{\lambda_r}{2}\|z-c\|^2.
$$

**Dual convention** (`mass_mode="dual"`, $M=G^{-1}$):

$$
H_{\mathrm{dual}}(z,\rho)=
\tfrac{1}{2}\rho^\top G(z)\rho
-(\nu-\tfrac{1}{2})\log\det G^{-1}(z)
+\tfrac{\lambda_r}{2}\|z-c\|^2.
$$

**Code:** `VolumeElementRiemannianHMCSampler`.

#### Bilan du Potentiel Effectif (Convention Duale)

En mode dual, on peut séparer:
$$
U_{\mathrm{model}}(z)= -\nu\log\det G^{-1}(z)+\tfrac{\lambda_r}{2}\|z-c\|^2,
$$
$$
K_{\mathrm{dual}}(z,\rho)=\tfrac{1}{2}\rho^\top G(z)\rho+\tfrac{1}{2}\log\det G^{-1}(z).
$$
L'Hamiltonien total non séparable est alors
$$
H_{\mathrm{dual}}(z,\rho)=U_{\mathrm{model}}(z)+K_{\mathrm{dual}}(z,\rho),
$$
et la contribution de potentiel effective devient
$$
U_{\mathrm{eff,dual}}(z)=
-(\nu-\tfrac{1}{2})\log\det G^{-1}(z)
+\tfrac{\lambda_r}{2}\|z-c\|^2.
$$

Théorème de l'attracteur (cas $\lambda_r=0$):
- si $\nu>\tfrac{1}{2}$, le terme volumique crée un puits vers les zones de grand $\det G^{-1}$ (centroïdes);
- si $\nu=\tfrac{1}{2}$, le gradient volumique s'annule exactement;
- si $\nu<\tfrac{1}{2}$, la contribution devient anti-attractive.

Le code protège ce cas via `enforce_dual_potential_well=True`: en mode dual, `volume_power<=0.5` sans prior radial positif déclenche une erreur de configuration.

---

## 3. Summary of Integrators

| Sampler | Kinetic Energy | Integrator Type | Code Reference |
| --- | --- | --- | --- |
| Geodesic | Riemannian | **Implicit** generalized leapfrog (exact) | `GeodesicHMCSampler._generalized_leapfrog_step` |
| Standard RHMC | Riemannian | **Implicit** generalized leapfrog (exact) | `RiemannianHMCSampler._generalized_leapfrog_step` |
| Gravity well (volume) | Euclidean | Standard leapfrog | `RHVAEVolumeElementHMCSampler._leapfrog` |
| Gravity well (volume, Riemannian standard/dual) | Riemannian | **Implicit** generalized leapfrog (exact), with optional adaptive dual step | `VolumeElementRiemannianHMCSampler` |

**Notes:**
- Exact Riemannian integrators require the full $\nabla_z H$ (including kinetic-gradient terms).
- Euclidean integrators only require $\nabla_z\log\det G^{-1}$ (via autograd).
- Tempering (momentum rescaling) is disabled by default for exactness. Set `exact=False` to use legacy tempered dynamics (approximate).

### 3.1 Stabilité de l'intégration et vélocité (mode dual)

Les équations de Hamilton donnent
$$
\dot z=\frac{\partial H}{\partial \rho}=M^{-1}(z)\rho.
$$
En mode dual ($M=G^{-1}$), on a
$$
\dot z = G(z)\rho.
$$

Si une valeur propre $\mu_i(z)$ de $G^{-1}(z)$ devient très petite dans une direction locale (zone très anisotrope), la valeur propre correspondante de $G(z)$ vaut $1/\mu_i(z)$ et la norme de vitesse peut exploser.

Critère de stabilité pratique:
$$
\epsilon_{\mathrm{eff}}\|\dot z\|_2 \le \Delta_{\max},
$$
avec $\Delta_{\max}=\texttt{adaptive\_max\_dual\_displacement}$.

Interprétation continue: on veut typiquement $\epsilon_{\mathrm{eff}}\propto 1/\sqrt{\kappa_{\mathrm{loc}}(G)}$ pour limiter le déplacement euclidien quand l'anisotropie locale explose.

Implémentation actuelle (proxy adaptatif):
$$
s=\mathrm{clip}\!\left(\frac{\Delta_{\max}}{\epsilon\,\max_b\|\dot z_b\|_2},\ s_{\min},\ 1\right),
\qquad
\epsilon_{\mathrm{eff}}=s\,\epsilon,
$$
où $s_{\min}=\texttt{adaptive\_min\_step\_scale}$.  
Cela borne le déplacement euclidien local quand la vitesse explose.

### 3.2 Régularisation numérique (Jitter Cholesky)

L'échantillonnage de $\rho$ demande une factorisation de covariance (selon la convention, $G$ ou $G^{-1}$). Dans les zones d'anisotropie extrême, la matrice peut être mal conditionnée.
Dans la branche void, la structure $\beta\,uu^\top+\lambda\psi(r)I$ combine un projecteur radial de rang 1 et des composantes transverses faibles, ce qui accentue ce conditionnement.

Le sampler utilise une régularisation mixte:
$$
j_{\mathrm{dyn}}(z)=\eta_{\mathrm{jit}}\;\mathrm{tr}\bigl(\mathrm{Cov}(z)\bigr),
\qquad
\eta_{\mathrm{jit}}=\texttt{dynamic\_jitter\_scale},
$$
$$
j_{\mathrm{tot}}(z)=
\max\!\bigl(j_{\mathrm{dyn}}(z),\ j_{\mathrm{floor}}\bigr),
\qquad
j_{\mathrm{floor}}=\texttt{cholesky\_jitter}.
$$
Puis
$$
\widetilde{\mathrm{Cov}}(z)=\mathrm{Cov}(z)+j_{\mathrm{tot}}(z)\,I.
$$

En cas d'échec Cholesky, le jitter est augmenté (x10, jusqu'à 3 essais), puis fallback `eigh` avec plancher spectral.  
Cette stratégie garde la matrice strictement SPD en flottants tout en préservant les directions principales (déplacement isotrope des valeurs propres).

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
