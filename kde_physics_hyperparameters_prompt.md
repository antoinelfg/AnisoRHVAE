# AI Investigation Task: Deriving AnisoRHVAE Hyperparameters from KDE Physics

## Objective
The AnisoRHVAE relies on several hyperparameters for its Riemannian metric blending and geometric sampling. We have empirically discovered an extremely stable configuration (the "core4" sweep) for a 2D latent space. 

Your task is to mathematically retro-engineer these empirical values by deriving them strictly from fundamental Kernel Density Estimation (KDE) physics, Gaussian statistics, and Riemannian geometric properties. We do not want arbitrary scaling constants (e.g., $r_0 = 2.4\tau$); we want equations rooted in real statistical thresholds (e.g., $r_0$ equals the $99\%$ confidence interval or $3\sigma$ radius of the KDE).

## Empirical Ground Truth (The Target)
In our highly successful benchmark (`core4` on 2D data), the geometry operates flawlessly with the following fixed values:
- **Spatial dimension ($d$):** 2
- **KDE Bandwidth / Temperature ($\tau$):** 0.5 (Note: Gaussian variance $\sigma^2 = \tau^2 / 2 = 0.125$, standard deviation $\sigma \approx 0.353$)
- **Transition Radius ($r_0$):** 1.2
- **Transition Steepness ($\kappa$):** 7.276
- **Attractor Softmax Stiffness ($\gamma$):** 7.624
- **Asymptotic Transverse Decay Scale ($s$ / `void_decay_scale`):** 8.871
- **Asymptotic Transverse Decay Power ($p$ / `void_decay_power`):** 1.744
- **Asymptotic Longitudinal Stretch ($\beta_{\mathrm{long}}$ / `radial_stretch`):** 9.252
- **Atom Flattening Power ($p_{\mathrm{atom}}$):** 1.096

## Theoretical Framing & Required Derivations

You must formulate rigorous mathematical justifications linking $\tau$ and $d$ to the parameters above. Consider the following physical frameworks:

### 1. The Confidence Boundary ($r_0$)
- **Context:** The base metric is a sum of RBF kernels $w_k = \exp(-d_k^2 / \tau^2)$. 
- **Investigation:** $r_0 = 1.2$. Note that $1.2 / 0.353 \approx 3.4\sigma$. Does $r_0$ perfectly align with a standard statistical cutoff (e.g., the $99\%$ mass of a 2D Gaussian, or a specific Mahalanobis distance threshold)? Derive $r_0(\tau, d)$ using cumulative distribution functions or Fisher Information dropoffs.

### 2. The Attractor Gravity ($\gamma$)
- **Context:** To ensure the metric points to the nearest data cluster when far away, we use a soft weighting: $\pi_k \propto \exp(-\gamma d_k^2)$.
- **Investigation:** We empirically found $\gamma \approx 7.62$. Notice that $2/\tau^2 = 8.0$. Is there a theoretical reason rooted in Bayesian posterior probabilities or KDE overlap integrals that suggests the soft-assignment temperature should be precisely inversely proportional to the variance, perhaps scaled by the dimension $d$? Derive $\gamma(\tau, d)$.

### 3. The Maximum Stable Curvature ($\kappa$)
- **Context:** The sigmoid transition is $\alpha(z) = \sigma(\kappa(r - r_0))$. The Generalized Leapfrog HMC integrator requires the Hamiltonian gradient to satisfy a Lipschitz condition to ensure its fixed-point solver converges.
- **Investigation:** If $\kappa$ is too large, spatial derivatives explode. $\kappa \approx 7.27$. Formulate the maximum allowable Riemannian curvature (or bounded Christoffel symbols) as a function of the leapfrog step size $\epsilon$ and the base KDE bandwidth $\tau$. Show how $\kappa \approx 7.27$ sits exactly on the theoretical edge of numerical contraction.

### 4. Transverse Decay Dynamics ($s$ and $p$)
- **Context:** Transverse variances decay as $1 / (1 + (r/s)^p)$.
- **Investigation:** We found $p \approx 1.75$ and $s \approx 8.87$. In a $d$-dimensional space, the Riemannian volume element $\sqrt{|G|}$ scales with transverse eigenvalues. Derive the asymptotic volume growth. What power $p$ ensures that the volume of the space does not blow up exponentially, maintaining a proper probability measure? Relate $p$ to $d$ (e.g., $p = d - 1/4$). What does $s = 8.87$ represent physically (e.g., $s \approx 10r_0$)?

### 5. Asymptotic Longitudinal Stiffness ($\beta_{\mathrm{long}}$)
- **Context:** $\beta_{\mathrm{long}} = 9.25$. 
- **Investigation:** This controls the restoring force pushing the sampler back to the data. Derive this using the expected kinetic energy of the Hamiltonian system. To ensure the sampler reflects off the void boundary at temperature $T_{HMC}=1$, what must the eigenvalue ratio be between the longitudinal and transverse directions? 

## Instructions for the Agent
1. Do not use generic proportionality constants if a rigorous physical constant (like $\pi$, $\sqrt{2}$, or Gaussian percentiles) exists.
2. Formulate each parameter as a direct function of $\tau$ (the KDE bandwidth) and $d$ (the latent dimension).
3. Conclude with a strictly parameterized geometry where the user *only* inputs $\tau$, and the entire Riemannian geometry self-assembles perfectly according to Gaussian/KDE physics.
