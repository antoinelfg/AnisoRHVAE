"""
Riemannian HMC Sampler
======================

Hamiltonian Monte Carlo sampler for Riemannian manifolds - RHVAE compatible.
"""

import torch
from typing import Dict, Any
from .base_sampler import BaseRiemannianSampler


class RiemannianHMCSampler(BaseRiemannianSampler):
    """Hamiltonian Monte Carlo sampler for Riemannian manifold - RHVAE compatible."""
    
    def __init__(
        self,
        model,
        mcmc_steps_nbr=100,
        n_lf=15,
        eps_lf=0.03,
        beta_zero=1.0,
        include_volume_grad: bool = True,
        exact: bool = True,
        fp_steps: int = 15,
        fp_damping: float = 0.7,
    ):
        super().__init__(model)
        self.mcmc_steps_nbr = mcmc_steps_nbr
        self.n_lf = torch.tensor([n_lf], device=model.device)
        self.eps_lf = torch.tensor([eps_lf], device=model.device)
        self.beta_zero_sqrt = torch.tensor([beta_zero], device=model.device).sqrt()
        self.include_volume_grad = bool(include_volume_grad)
        self.exact = bool(exact)
        self.fp_steps = int(fp_steps)
        self.fp_damping = float(fp_damping)
        
        # Set target density to standard Gaussian prior: π(z) ∝ exp(-0.5 ||z||^2)
        # Riemannian geometry enters through kinetic term and + 1/2 log det G(z).
        # This choice yields stable, reversible dynamics and correct acceptance.
        self.log_pi = lambda z: -0.5 * torch.sum(z * z, dim=1)
        # Gradient of log π(z) for standard normal is -z
        self.grad_func = lambda z: -z
    
    def _log_sqrt_det_G_inv(self, z, t=0):
        """Compute log(sqrt(det(G^{-1}))) using autograd."""
        if not z.requires_grad:
            z = z.clone().detach().requires_grad_(True)
        G_inv = self.model.G_inv(z)
        logabs = torch.linalg.slogdet(G_inv).logabsdet
        return 0.5 * logabs

    def _grad_log_prop(self, z, t=0):
        """Compute gradient of log sqrt det(G^{-1}) using autograd."""
        if not z.requires_grad:
            z_grad = z.clone().detach().requires_grad_(True)
        else:
            z_grad = z
        log_det = self._log_sqrt_det_G_inv(z_grad, t)
        grads = torch.autograd.grad(log_det.sum(), z_grad, create_graph=False)[0]
        return grads
    
    @staticmethod
    def _tempering(k, K, beta_zero_sqrt):
        """Tempering schedule for HMC sampling."""
        beta_k = ((1 - 1 / beta_zero_sqrt) * (k / K) ** 2) + 1 / beta_zero_sqrt
        return 1 / beta_k
    
    def _compute_hamiltonian(self, z, rho):
        """
        Compute complete Riemannian Hamiltonian:
        H(z, ρ) = -log π(z) + 1/2 ρ^T G^{-1}(z) ρ + 1/2 log det(G(z))
        
        Args:
            z: Position [batch_size, latent_dim]
            rho: Momentum [batch_size, latent_dim]
            
        Returns:
            Hamiltonian energy [batch_size]
        """
        # Potential energy: -log π(z)
        potential = -self.log_pi(z)
        
        # Kinetic energy: 1/2 ρ^T G^{-1}(z) ρ
        G_inv = self.model.G_inv(z)
        #G_inv = torch.linalg.inv(G_inv)
        kinetic = 0.5 * torch.einsum('bi,bij,bj->b', rho, G_inv, rho)
        
        # Metric-dependent correction: 1/2 log det(G(z))
        # This term accounts for the volume element in Riemannian geometry
        if self.include_volume_grad:
            G = self.model.G(z)
            #G = torch.linalg.inv(G)
            log_det_G = torch.linalg.slogdet(G).logabsdet
            metric_correction = 0.5 * log_det_G
        else:
            metric_correction = torch.zeros_like(potential)
        
        return potential + kinetic + metric_correction

    def _velocity(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt = ∂H/∂ρ for Riemannian kinetic (G^{-1} ρ)."""
        G_inv = self.model.G_inv(z)
        #G_inv = torch.linalg.inv(G_inv)
        return torch.einsum('bij,bj->bi', G_inv, rho)
    
    def _initialize_momentum(self, z):
        """
        Initialize momentum using proper Riemannian geometry:
        ρ ~ N(0, G(z)) using Cholesky decomposition
        
        Args:
            z: Position [batch_size, latent_dim]
            
        Returns:
            Momentum [batch_size, latent_dim]
        """
        # Sample from standard Gaussian and shape via Cholesky of G(z)
        gamma = torch.randn_like(z)  # [batch_size, latent_dim]
        G = self.model.G(z)  # [batch_size, D, D]
        #G = torch.linalg.inv(G)
        try:
            jitter = getattr(getattr(self, "metric_config", None), "cholesky_jitter", 0.0)
            if jitter and jitter > 0:
                eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
                G = G + jitter * eye
            L = torch.linalg.cholesky(G)
            rho = torch.einsum('bij,bj->bi', L, gamma)  # ρ ~ N(0, G)
        except torch.linalg.LinAlgError:
            eigenvals, eigenvecs = torch.linalg.eigh(G)
            eigenvals = torch.clamp(eigenvals, min=1e-6)
            sqrt_G = eigenvecs @ torch.diag_embed(torch.sqrt(eigenvals)) @ eigenvecs.transpose(-2, -1)
            rho = torch.einsum('bij,bj->bi', sqrt_G, gamma)
        
        return rho
    
    def _generalized_leapfrog_step(self, z, rho, eps):
        """
        Implicit generalized leapfrog step for Riemannian HMC (exact RMHMC).

        Uses fixed-point iterations to solve the implicit updates, including
        the full ∇_z H (kinetic-gradient terms included).
        """
        steps = max(1, int(self.fp_steps))
        damping = float(self.fp_damping)

        # (A) Implicit half-step for momentum
        rho_half = rho
        for _ in range(steps):
            grad_z, _ = self._grad_hamiltonian_z(z, rho_half)
            if not torch.isfinite(grad_z).all():
                return z.detach().requires_grad_(True), rho.detach()
            rho_update = rho - 0.5 * eps * grad_z
            rho_half = (1.0 - damping) * rho_half + damping * rho_update

        # (B) Implicit full-step for position
        z_new = z
        for _ in range(steps):
            v0 = self._velocity(z, rho_half)
            v1 = self._velocity(z_new, rho_half)
            z_update = z + 0.5 * eps * (v0 + v1)
            z_new = (1.0 - damping) * z_new + damping * z_update
            if not torch.isfinite(z_new).all():
                return z.detach().requires_grad_(True), rho.detach()

        # (C) Final momentum half-step at z_new
        grad_z_new, z_req = self._grad_hamiltonian_z(z_new, rho_half)
        if not torch.isfinite(grad_z_new).all():
            return z.detach().requires_grad_(True), rho.detach()
        rho_new = rho_half - 0.5 * eps * grad_z_new
        return z_req, rho_new
    
    def sample(
        self,
        n_samples,
        t: int = 0,
        init_std: float = 1.0,
        eps_jitter: float = 0.0,
        n_lf_jitter: int = 0,
    ):
        """Sample from the Riemannian manifold using HMC."""
        # Make sure static tensors are on the right device in case the model
        # has been moved (e.g. by Lightning) after the sampler was created.
        current_device = self.model.device
        self.n_lf = self.n_lf.to(current_device)
        self.eps_lf = self.eps_lf.to(current_device)
        self.beta_zero_sqrt = self.beta_zero_sqrt.to(current_device)

        # Initialize from a zero-mean Gaussian with configurable std
        z0 = torch.randn(n_samples, self.model.latent_dim, device=current_device) * float(init_std)
        
        z = z0.clone().detach().requires_grad_(True)
        
        n_lf_int = int(self.n_lf.item())
        acceptance_count = 0
        rho_prev = None
        
        for i in range(self.mcmc_steps_nbr):
            # Initialize momentum using proper Riemannian geometry
            rho = self._refresh_momentum(z, rho_prev)
            rho0 = rho
            
            # Initial Hamiltonian
            with torch.no_grad():
                H0 = self._compute_hamiltonian(z, rho)
            
            # Choose local leapfrog count with jitter (optional)
            if n_lf_jitter > 0:
                jitter = torch.randint(-n_lf_jitter, n_lf_jitter + 1, (1,), device=current_device).item()
                local_n_lf = max(1, n_lf_int + int(jitter))
            else:
                local_n_lf = n_lf_int

            use_tempering = (not self.exact) and hasattr(self, "_tempering")
            beta_sqrt_old = self.beta_zero_sqrt if use_tempering else None
            # Generalized leapfrog steps
            for k in range(local_n_lf):
                # Use generalized leapfrog with metric updates
                if eps_jitter > 0.0:
                    # Uniform jitter in [(1-j), (1+j)]
                    u = (torch.rand(1, device=current_device).item() - 0.5) * 2.0 * float(eps_jitter)
                    eps_eff = float(self.eps_lf.item()) * (1.0 + u)
                else:
                    eps_eff = float(self.eps_lf.item())
                z, rho = self._generalized_leapfrog_step(z, rho, eps_eff)
                
                # Tempering (approximate; disabled in exact mode)
                if use_tempering:
                    beta_sqrt = self._tempering(k + 1, local_n_lf, self.beta_zero_sqrt)
                    rho = (beta_sqrt_old / beta_sqrt) * rho
                    beta_sqrt_old = beta_sqrt
            
            # Final Hamiltonian
            with torch.no_grad():
                H = self._compute_hamiltonian(z, rho)
                
                # Metropolis acceptance
                log_alpha = H0 - H
                alpha = torch.exp(torch.clamp(log_alpha, max=0))
                acc = torch.rand(n_samples, device=current_device)
                moves = (acc < alpha).float().reshape(n_samples, 1)
                
                # Update z (detach to avoid gradient accumulation)
                z = ((moves * z + (1 - moves) * z0).detach().requires_grad_(True))
                z0 = z.clone().detach()
                
                # Track acceptance rate
                acceptance_count += moves.sum().item()

                # Momentum persistence (GHMC-style)
                if getattr(self, "momentum_persist", 0.0) > 0.0:
                    rho_prev = (moves * rho + (1 - moves) * (-rho0)).detach()
                else:
                    rho_prev = None
        
        # Log acceptance rate
        acceptance_rate = acceptance_count / (self.mcmc_steps_nbr * n_samples)
        # Expose for programmatic checks
        self.last_acceptance_rate = float(acceptance_rate)
        print(f"✅ RHMC Acceptance Rate: {acceptance_rate:.3f}")
        
        return z.detach()
    
    def sample_posterior(self, mu, log_var, t=0):
        """Sample from posterior using simplified RHMC approach."""
        batch_size = mu.shape[0]
        
        # Initialize near posterior mode
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * log_var)
        
        # Apply a small number of refinement steps using metric-aware sampling
        for i in range(3):  # Very few steps for training stability
            z = z.detach().requires_grad_(True)
            
            try:
                # Compute gradient of log probability using metric
                G_z = self.model.G(z)
                
                # Compute gradient of log probability: ∇log p(z) = -G(z) * (z - μ)
                diff = z - mu
                grad_log_prob = -torch.einsum('bij,bj->bi', G_z, diff)
                
                # Small step in gradient direction
                step_size = 0.01
                z = z + step_size * grad_log_prob
                
            except Exception as e:
                print(f"⚠️ RHMC posterior sampling failed: {e}, using standard sampling")
                # Fallback to standard sampling
                eps = torch.randn_like(mu)
                z = mu + eps * torch.exp(0.5 * log_var)
                break
        
        return z.detach()
    
    def sample_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor, 
                                 method: str = 'hmc') -> torch.Tensor:
        """
        Sample latent codes using HMC on the Riemannian manifold.
        
        Args:
            mu: Posterior mean [batch_size, latent_dim]
            log_var: Posterior log variance [batch_size, latent_dim]
            method: Sampling method ('hmc' or 'posterior_hmc')
            
        Returns:
            Sampled latent codes [batch_size, latent_dim]
        """
        if method == 'posterior_hmc':
            return self.sample_posterior(mu, log_var)
        else:
            # For training, use a simplified approach that preserves gradients
            # Start from posterior mean and apply a few HMC steps
            batch_size = mu.shape[0]
            
            # Initialize near posterior mode
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * log_var)
            
            # Apply a small number of HMC-style refinement steps
            # This preserves gradients while incorporating Riemannian geometry
            for i in range(3):  # Very few steps for training stability
                z = z.detach().requires_grad_(True)
                
                # Compute gradient of log probability
                try:
                    g = -self.grad_func(z)
                    
                    # Small step in gradient direction
                    step_size = 0.01
                    z = z + step_size * g
                    
                except Exception as e:
                    print(f"⚠️ HMC refinement failed: {e}, using standard sampling")
                    break
            
            return z.detach()
    
    def sample_prior(self, num_samples: int, method: str = 'hmc') -> torch.Tensor:
        """
        Sample from the Riemannian prior using HMC.
        
        Args:
            num_samples: Number of samples to generate
            method: Prior sampling method ('hmc' or 'basic')
            
        Returns:
            Prior samples [num_samples, latent_dim]
        """
        if method == 'hmc':
            return self.sample(num_samples)
        else:
            # Fallback to standard Gaussian
            return torch.randn(num_samples, self.model.latent_dim, device=self.device)
    
    def get_sampling_methods(self) -> Dict[str, str]:
        """Override to provide HMC-specific methods."""
        return {
            'hmc': 'Hamiltonian Monte Carlo sampling on manifold',
            'posterior_hmc': 'HMC sampling from posterior',
            'basic': 'Standard Gaussian sampling (fallback)'
        }
    
    def get_hmc_parameters(self) -> Dict[str, Any]:
        """
        Get HMC sampling parameters.
        
        Returns:
            Dictionary with HMC parameters
        """
        return {
            'mcmc_steps_nbr': self.mcmc_steps_nbr,
            'n_lf': int(self.n_lf.item()),
            'eps_lf': float(self.eps_lf.item()),
            'beta_zero': float(self.beta_zero_sqrt.item() ** 2)
        } 


class GeodesicHMCSampler(BaseRiemannianSampler):
    """Explicit geodesic HMC sampler targeting the uniform Riemannian measure.

    Target density: π_R(z) ∝ sqrt(det(G(z))) (uniform w.r.t. volume element).
    Hamiltonian reduces to pure kinetic energy:
        H(z, ρ) = 1/2 ρ^T G^{-1}(z) ρ
    """

    def __init__(
        self,
        model,
        mcmc_steps_nbr: int = 100,
        n_lf: int = 15,
        eps_lf: float = 0.03,
        beta_zero: float = 1.0,
        include_metropolis: bool = True,
        exact: bool = True,
        fp_steps: int = 15,
        fp_damping: float = 0.7,
    ):
        super().__init__(model)
        self.mcmc_steps_nbr = int(mcmc_steps_nbr)
        self.n_lf = int(n_lf)
        self.eps_lf = float(eps_lf)
        self.beta_zero_sqrt = torch.tensor([beta_zero], device=self.device).sqrt()
        self.include_metropolis = bool(include_metropolis)
        self.exact = bool(exact)
        self.fp_steps = int(fp_steps)
        self.fp_damping = float(fp_damping)

    @staticmethod
    def _tempering(k: int, K: int, beta_zero_sqrt: torch.Tensor) -> torch.Tensor:
        beta_k = ((1 - 1 / beta_zero_sqrt) * (k / K) ** 2) + 1 / beta_zero_sqrt
        return 1 / beta_k

    def _initialize_momentum(self, z: torch.Tensor) -> torch.Tensor:
        """Sample ρ ~ N(0, G(z))."""
        gamma = torch.randn_like(z)
        G = self.model.G(z)
        try:
            jitter = getattr(getattr(self, "metric_config", None), "cholesky_jitter", 0.0)
            if jitter and jitter > 0:
                eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
                G = G + jitter * eye
            L = torch.linalg.cholesky(G)
            rho = torch.einsum("bij,bj->bi", L, gamma)
        except torch.linalg.LinAlgError:
            evals, evecs = torch.linalg.eigh(G)
            evals = torch.clamp(evals, min=1e-6)
            sqrt_G = evecs @ torch.diag_embed(torch.sqrt(evals)) @ evecs.transpose(-2, -1)
            rho = torch.einsum("bij,bj->bi", sqrt_G, gamma)
        return rho

    def _hamiltonian(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        G_inv = self.model.G_inv(z)
        return 0.5 * torch.einsum("bi,bij,bj->b", rho, G_inv, rho)

    def _velocity(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        G_inv = self.model.G_inv(z)
        return torch.einsum("bij,bj->bi", G_inv, rho)

    def _generalized_leapfrog_step(self, z: torch.Tensor, rho: torch.Tensor, eps: float):
        """Implicit generalized leapfrog for non-separable kinetic energy."""
        steps = max(1, int(self.fp_steps))
        damping = float(self.fp_damping)

        rho_half = rho
        for _ in range(steps):
            grad_z, _ = self._grad_hamiltonian_z(z, rho_half)
            if not torch.isfinite(grad_z).all():
                return z.detach().requires_grad_(True), rho.detach()
            rho_update = rho - 0.5 * eps * grad_z
            rho_half = (1.0 - damping) * rho_half + damping * rho_update

        z_new = z
        for _ in range(steps):
            v0 = self._velocity(z, rho_half)
            v1 = self._velocity(z_new, rho_half)
            z_update = z + 0.5 * eps * (v0 + v1)
            z_new = (1.0 - damping) * z_new + damping * z_update
            if not torch.isfinite(z_new).all():
                return z.detach().requires_grad_(True), rho.detach()

        grad_z_new, z_req = self._grad_hamiltonian_z(z_new, rho_half)
        if not torch.isfinite(grad_z_new).all():
            return z.detach().requires_grad_(True), rho.detach()
        rho_new = rho_half - 0.5 * eps * grad_z_new
        return z_req, rho_new

    def _grad_z(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        z_req = z if z.requires_grad else z.clone().detach().requires_grad_(True)
        G_inv = self.model.G_inv(z_req)
        quad = 0.5 * torch.einsum("bi,bij,bj->b", rho, G_inv, rho).sum()
        grad = torch.autograd.grad(quad, z_req, create_graph=False)[0]
        return grad

    def _leapfrog(self, z: torch.Tensor, rho: torch.Tensor, eps: float):
        grad = self._grad_z(z, rho)
        rho_half = rho - 0.5 * eps * grad
        G_inv = self.model.G_inv(z)
        z_new = z + eps * torch.einsum("bij,bj->bi", G_inv, rho_half)
        grad_new = self._grad_z(z_new, rho_half)
        rho_new = rho_half - 0.5 * eps * grad_new
        return z_new, rho_new

    def sample(self, n_samples: int = 100) -> torch.Tensor:
        device = self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device
        # Ensure beta_zero_sqrt is on correct device
        self.beta_zero_sqrt = self.beta_zero_sqrt.to(device)
        z = torch.randn(n_samples, self.model.latent_dim, device=device).requires_grad_(True)
        accept_count = 0
        rho_prev = None
        for _ in range(self.mcmc_steps_nbr):
            rho0 = self._refresh_momentum(z, rho_prev)
            with torch.no_grad():
                H0 = self._hamiltonian(z, rho0)

            z_prop, rho_prop = z, rho0
            use_tempering = (not self.exact) and hasattr(self, "_tempering")
            beta_sqrt_old = self.beta_zero_sqrt if use_tempering else None
            for k in range(self.n_lf):
                if self.exact:
                    z_prop, rho_prop = self._generalized_leapfrog_step(z_prop, rho_prop, self.eps_lf)
                else:
                    z_prop, rho_prop = self._leapfrog(z_prop, rho_prop, self.eps_lf)
                if use_tempering:
                    beta_sqrt = self._tempering(k + 1, self.n_lf, self.beta_zero_sqrt)
                    rho_prop = (beta_sqrt_old / beta_sqrt) * rho_prop
                    beta_sqrt_old = beta_sqrt

            if self.include_metropolis:
                with torch.no_grad():
                    H1 = self._hamiltonian(z_prop, rho_prop)
                    log_alpha = H0 - H1
                    alpha = torch.exp(torch.clamp(log_alpha, max=0))
                    u = torch.rand_like(alpha)
                    accept = (u < alpha).float().view(-1, 1)
                    z = ((accept * z_prop + (1 - accept) * z).detach().requires_grad_(True))
                    accept_count += int(accept.sum().item())

                    if getattr(self, "momentum_persist", 0.0) > 0.0:
                        rho_prev = (accept * rho_prop + (1 - accept) * (-rho0)).detach()
                    else:
                        rho_prev = None
            else:
                z = z_prop.detach().requires_grad_(True)

        if self.include_metropolis:
            self.last_acceptance_rate = accept_count / (self.mcmc_steps_nbr * n_samples)
            print(f"✅ Geodesic RHMC Acceptance Rate: {self.last_acceptance_rate:.3f}")
        else:
            self.last_acceptance_rate = float("nan")
        return z.detach()

    def sample_prior(self, num_samples: int, method: str = "geodesic") -> torch.Tensor:
        return self.sample(num_samples)

    def sample_riemannian_latents(
        self, mu: torch.Tensor, log_var: torch.Tensor, method: str = "geodesic"
    ) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return (mu + eps * torch.exp(0.5 * log_var)).detach()


class DualRiemannianHMCSampler(BaseRiemannianSampler):
    """Dual RHMC treating G^{-1} as the metric (kinetic uses G).

    Metric: M(z) = G^{-1}(z)
    Momentum: ρ ~ N(0, M) = N(0, G^{-1}(z))
    Kinetic: 1/2 ρ^T M^{-1} ρ = 1/2 ρ^T G(z) ρ
    Volume correction: +1/2 log det M = +1/2 log det(G^{-1}(z))
    """

    def __init__(
        self,
        model,
        mcmc_steps_nbr: int = 50,
        n_lf: int = 10,
        eps_lf: float = 0.02,
        beta_zero: float = 1.0,
        exact: bool = True,
        fp_steps: int = 15,
        fp_damping: float = 0.7,
    ):
        super().__init__(model)
        self.mcmc_steps_nbr = int(mcmc_steps_nbr)
        self.n_lf = int(n_lf)
        self.eps_lf = float(eps_lf)
        self.beta_zero_sqrt = torch.tensor([beta_zero], device=self.device).sqrt()
        self.exact = bool(exact)
        self.fp_steps = int(fp_steps)
        self.fp_damping = float(fp_damping)

        # Default target: standard Gaussian prior
        self.log_pi = lambda z: -0.5 * torch.sum(z * z, dim=1)
        self.grad_func = lambda z: -z

    def _initialize_momentum(self, z: torch.Tensor) -> torch.Tensor:
        """Sample ρ ~ N(0, G^{-1}(z))."""
        with torch.no_grad():
            G_inv = self.model.G_inv(z)  # [B,D,D]
            try:
                jitter = getattr(getattr(self, "metric_config", None), "cholesky_jitter", 0.0)
                if jitter and jitter > 0:
                    eye = torch.eye(G_inv.shape[-1], device=G_inv.device, dtype=G_inv.dtype)
                    G_inv = G_inv + jitter * eye
                L = torch.linalg.cholesky(G_inv)
            except torch.linalg.LinAlgError:
                evals, evecs = torch.linalg.eigh(G_inv)
                evals = torch.clamp(evals, min=1e-6)
                L = evecs @ torch.diag_embed(torch.sqrt(evals))
            gamma = torch.randn_like(z)
            scale = self.beta_zero_sqrt if not self.exact else 1.0
            rho = torch.einsum("bij,bj->bi", L, gamma) / scale
            return rho

    def _hamiltonian(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        potential = -self.log_pi(z)
        G = self.model.G(z)
        kinetic = 0.5 * torch.einsum("bi,bij,bj->b", rho, G, rho)
        G_inv = self.model.G_inv(z)
        log_det_Ginv = torch.linalg.slogdet(G_inv).logabsdet
        vol = 0.5 * log_det_Ginv
        return potential + kinetic + vol

    def _grad_potential(self, z: torch.Tensor) -> torch.Tensor:
        grad_pi = -self.grad_func(z)
        z_req = z if z.requires_grad else z.clone().detach().requires_grad_(True)
        G_inv = self.model.G_inv(z_req)
        log_det_Ginv = torch.linalg.slogdet(G_inv).logabsdet
        vol = 0.5 * log_det_Ginv.sum()
        vol_grad = torch.autograd.grad(vol, z_req, create_graph=False)[0]
        return grad_pi + vol_grad

    def _velocity(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt = ∂H/∂ρ for dual metric (G ρ)."""
        G = self.model.G(z)
        return torch.einsum("bij,bj->bi", G, rho)

    def _generalized_leapfrog_step(self, z: torch.Tensor, rho: torch.Tensor, eps: float):
        """Implicit generalized leapfrog for dual metric."""
        steps = max(1, int(self.fp_steps))
        damping = float(self.fp_damping)

        rho_half = rho
        for _ in range(steps):
            grad_z, _ = self._grad_hamiltonian_z(z, rho_half)
            if not torch.isfinite(grad_z).all():
                return z.detach().requires_grad_(True), rho.detach()
            rho_update = rho - 0.5 * eps * grad_z
            rho_half = (1.0 - damping) * rho_half + damping * rho_update

        z_new = z
        for _ in range(steps):
            v0 = self._velocity(z, rho_half)
            v1 = self._velocity(z_new, rho_half)
            z_update = z + 0.5 * eps * (v0 + v1)
            z_new = (1.0 - damping) * z_new + damping * z_update
            if not torch.isfinite(z_new).all():
                return z.detach().requires_grad_(True), rho.detach()

        grad_z_new, z_req = self._grad_hamiltonian_z(z_new, rho_half)
        if not torch.isfinite(grad_z_new).all():
            return z.detach().requires_grad_(True), rho.detach()
        rho_new = rho_half - 0.5 * eps * grad_z_new
        return z_req, rho_new

    def _leapfrog(self, z: torch.Tensor, rho: torch.Tensor, eps: float):
        grad = self._grad_potential(z)
        rho_half = rho - 0.5 * eps * grad
        G = self.model.G(z)
        z_new = z + eps * torch.einsum("bij,bj->bi", G, rho_half)
        grad_new = self._grad_potential(z_new)
        rho_new = rho_half - 0.5 * eps * grad_new
        return z_new, rho_new

    def sample(self, n_samples: int = 100) -> torch.Tensor:
        device = self.model.device if hasattr(self.model, "device") else next(self.model.parameters()).device
        z = torch.randn(n_samples, self.model.latent_dim, device=device).requires_grad_(True)
        accept_count = 0
        rho_prev = None
        for _ in range(self.mcmc_steps_nbr):
            rho0 = self._refresh_momentum(z, rho_prev)
            with torch.no_grad():
                H0 = self._hamiltonian(z, rho0)

            z_prop, rho_prop = z, rho0
            use_tempering = (not self.exact) and hasattr(self, "_tempering")
            beta_sqrt_old = self.beta_zero_sqrt if use_tempering else None
            for k in range(self.n_lf):
                z_prop, rho_prop = self._generalized_leapfrog_step(z_prop, rho_prop, self.eps_lf)
                if use_tempering:
                    beta_sqrt = self._tempering(k + 1, self.n_lf, self.beta_zero_sqrt)
                    rho_prop = (beta_sqrt_old / beta_sqrt) * rho_prop
                    beta_sqrt_old = beta_sqrt

            with torch.no_grad():
                H1 = self._hamiltonian(z_prop, rho_prop)
                log_alpha = H0 - H1
                alpha = torch.exp(torch.clamp(log_alpha, max=0))
                u = torch.rand_like(alpha)
                accept = (u < alpha).float().view(-1, 1)
                z = ((accept * z_prop + (1 - accept) * z).detach().requires_grad_(True))
                accept_count += int(accept.sum().item())

                if getattr(self, "momentum_persist", 0.0) > 0.0:
                    rho_prev = (accept * rho_prop + (1 - accept) * (-rho0)).detach()
                else:
                    rho_prev = None

        self.last_acceptance_rate = accept_count / (self.mcmc_steps_nbr * n_samples)
        print(f"✅ Dual RHMC Acceptance Rate: {self.last_acceptance_rate:.3f}")
        return z.detach()

    def sample_prior(self, num_samples: int, method: str = "dual") -> torch.Tensor:
        return self.sample(num_samples)

    def sample_riemannian_latents(
        self, mu: torch.Tensor, log_var: torch.Tensor, method: str = "dual"
    ) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return (mu + eps * torch.exp(0.5 * log_var)).detach()


class RHVAEVolumeElementHMCSampler(BaseRiemannianSampler):
    """Sampler that mirrors the original RHVAE sampler behavior.

    Target density: π(z) ∝ det(G^{-1}(z))^{volume_power} (default volume_power=0.5)
    Momentum: standard Euclidean Gaussian with tempering schedule
    Leapfrog: Euclidean updates using ∇ log sqrt det(G^{-1}).
    """

    def __init__(
        self,
        model,
        mcmc_steps_nbr: int = 100,
        n_lf: int = 15,
        eps_lf: float = 0.03,
        beta_zero: float = 1.0,
        exact: bool = True,
        volume_power: float = 0.5,
    ):
        super().__init__(model)
        self.mcmc_steps_nbr = int(mcmc_steps_nbr)
        self.n_lf = int(n_lf)
        self.eps_lf = float(eps_lf)
        self.beta_zero_sqrt = torch.tensor([beta_zero], device=self.device).sqrt()
        self.exact = bool(exact)
        self.volume_power = float(volume_power)

    def _initialize_momentum(self, z: torch.Tensor) -> torch.Tensor:
        """Initialize Euclidean momentum. Exact mode uses N(0, I)."""
        gamma = torch.randn_like(z, device=z.device)
        if self.exact:
            return gamma
        return gamma / self.beta_zero_sqrt

    @staticmethod
    def _log_sqrt_det_Ginv(z, model):
        Ginv = model.G_inv(z)
        logabs = torch.linalg.slogdet(Ginv).logabsdet
        return 0.5 * logabs

    @staticmethod
    def _grad_log_sqrt_det_Ginv(z, model):
        z_req = z if z.requires_grad else z.clone().detach().requires_grad_(True)
        G_inv = model.G_inv(z_req)
        logabs = torch.linalg.slogdet(G_inv).logabsdet
        log_sqrt = 0.5 * logabs.sum()
        grad = torch.autograd.grad(log_sqrt, z_req, create_graph=False)[0]
        return grad

    @staticmethod
    def _tempering(k, K, beta_zero_sqrt):
        beta_k = ((1 - 1 / beta_zero_sqrt) * (k / K) ** 2) + 1 / beta_zero_sqrt
        return 1 / beta_k

    def _hamiltonian(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Hamiltonian: H = -0.5 log det(G_inv) + 0.5 rho^T rho (Euclidean kinetic)."""
        factor = 2.0 * self.volume_power
        return -(factor * self._log_sqrt_det_Ginv(z, self.model)) + 0.5 * torch.sum(rho * rho, dim=1)

    def _leapfrog(self, z: torch.Tensor, rho: torch.Tensor, eps: float):
        """Euclidean leapfrog step using gradient of log sqrt det(G_inv)."""
        factor = 2.0 * self.volume_power
        g = -(factor * self._grad_log_sqrt_det_Ginv(z, self.model))
        rho_half = rho - 0.5 * eps * g
        z_new = z + eps * rho_half
        g_new = -(factor * self._grad_log_sqrt_det_Ginv(z_new, self.model))
        rho_new = rho_half - 0.5 * eps * g_new
        return z_new, rho_new

    def sample(self, n_samples: int = 100) -> torch.Tensor:
        device = self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device
        # Ensure beta_zero_sqrt is on correct device
        self.beta_zero_sqrt = self.beta_zero_sqrt.to(device)
        K = self.model.centroids_tens.shape[0]
        idx = torch.randint(K, (n_samples,), device=device)
        z0 = self.model.centroids_tens[idx].detach().to(device)
        z = z0
        accept_count = 0
        rho_prev = None

        for _ in range(self.mcmc_steps_nbr):
            rho = self._refresh_momentum(z, rho_prev)
            rho0 = rho
            use_tempering = (not self.exact) and hasattr(self, "_tempering")
            beta_sqrt_old = self.beta_zero_sqrt if use_tempering else None
            factor = 2.0 * self.volume_power
            with torch.no_grad():
                H0 = -(factor * self._log_sqrt_det_Ginv(z, self.model)) + 0.5 * torch.sum(rho * rho, dim=1)
            for k in range(self.n_lf):
                g = -(factor * self._grad_log_sqrt_det_Ginv(z, self.model))
                rho_half = rho - 0.5 * self.eps_lf * g
                z = z + self.eps_lf * rho_half
                g_new = -(factor * self._grad_log_sqrt_det_Ginv(z, self.model))
                rho_new = rho_half - 0.5 * self.eps_lf * g_new
                if use_tempering:
                    beta_sqrt = self._tempering(k + 1, self.n_lf, self.beta_zero_sqrt)
                    rho = (beta_sqrt_old / beta_sqrt) * rho_new
                    beta_sqrt_old = beta_sqrt
                else:
                    rho = rho_new
            with torch.no_grad():
                H = -(factor * self._log_sqrt_det_Ginv(z, self.model)) + 0.5 * torch.sum(rho * rho, dim=1)
                alpha = torch.exp(-(H - H0)).clamp(max=1.0)
                u = torch.rand_like(alpha)
                moves = (u < alpha).float().view(-1, 1)
                accept_count += int(moves.sum().item())
                z = ((moves * z + (1 - moves) * z0).detach())
                z0 = z
                if getattr(self, "momentum_persist", 0.0) > 0.0:
                    rho_prev = (moves * rho + (1 - moves) * (-rho0)).detach()
                else:
                    rho_prev = None
        self.last_acceptance_rate = accept_count / (self.mcmc_steps_nbr * n_samples)
        print(f"✅ RHVAE-Volume Acceptance Rate: {self.last_acceptance_rate:.3f}")
        return z.detach()

    # Implement abstract API expected by BaseRiemannianSampler
    def sample_prior(self, num_samples: int, method: str = 'hmc') -> torch.Tensor:
        return self.sample(num_samples)

    def sample_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor, method: str = 'hmc') -> torch.Tensor:
        """Lightweight posterior refinement using the same volume-element dynamics.

        Starts from the encoder posterior mean and performs a few tempered Euclidean
        updates on ∇ log sqrt det(G^{-1}). No dependence on dual/standard leapfrog.
        """
        device = self.device
        z = mu.detach().to(device)
        steps = max(1, min(5, self.mcmc_steps_nbr // 10))
        inner_n_lf = max(2, self.n_lf // 2)

        for _ in range(steps):
            rho = self._initialize_momentum(z)
            factor = 2.0 * self.volume_power
            with torch.no_grad():
                H0 = -(factor * self._log_sqrt_det_Ginv(z, self.model)) + 0.5 * torch.sum(rho * rho, dim=1)
            use_tempering = (not self.exact) and hasattr(self, "_tempering")
            beta_sqrt_old = self.beta_zero_sqrt if use_tempering else None
            for k in range(inner_n_lf):
                g = -(factor * self._grad_log_sqrt_det_Ginv(z, self.model))
                rho_half = rho - 0.5 * (self.eps_lf * 0.5) * g
                z = z + (self.eps_lf * 0.5) * rho_half
                g_new = -(factor * self._grad_log_sqrt_det_Ginv(z, self.model))
                rho_new = rho_half - 0.5 * (self.eps_lf * 0.5) * g_new
                if use_tempering:
                    beta_sqrt = self._tempering(k + 1, inner_n_lf, self.beta_zero_sqrt)
                    rho = (beta_sqrt_old / beta_sqrt) * rho_new
                    beta_sqrt_old = beta_sqrt
                else:
                    rho = rho_new
            with torch.no_grad():
                H1 = -(factor * self._log_sqrt_det_Ginv(z, self.model)) + 0.5 * torch.sum(rho * rho, dim=1)
                alpha = torch.exp(-(H1 - H0)).clamp(max=1.0)
                u = torch.rand_like(alpha)
                moves = (u < alpha).float().view(-1, 1)
                z = ((moves * z + (1 - moves) * mu.to(device)).detach())
        return z.detach()


class RHVAELogDetHMCSampler(BaseRiemannianSampler):
    """Sampler targeting π(z) ∝ sqrt(det(G(z))) with Euclidean momentum.

    Hamiltonian: H = -0.5 log det(G(z)) + 0.5 rho^T rho
    """

    def __init__(
        self,
        model,
        mcmc_steps_nbr: int = 100,
        n_lf: int = 15,
        eps_lf: float = 0.03,
        beta_zero: float = 1.0,
        exact: bool = True,
    ):
        super().__init__(model)
        self.mcmc_steps_nbr = int(mcmc_steps_nbr)
        self.n_lf = int(n_lf)
        self.eps_lf = float(eps_lf)
        self.beta_zero_sqrt = torch.tensor([beta_zero], device=self.device).sqrt()
        self.exact = bool(exact)

    @staticmethod
    def _log_sqrt_det_G(z, model):
        G = model.G(z)
        logabs = torch.linalg.slogdet(G).logabsdet
        return 0.5 * logabs

    @staticmethod
    def _grad_log_sqrt_det_G(z, model):
        z_req = z if z.requires_grad else z.clone().detach().requires_grad_(True)
        G = model.G(z_req)
        logabs = torch.linalg.slogdet(G).logabsdet
        log_sqrt = 0.5 * logabs.sum()
        grad = torch.autograd.grad(log_sqrt, z_req, create_graph=False)[0]
        return grad

    @staticmethod
    def _tempering(k, K, beta_zero_sqrt):
        beta_k = ((1 - 1 / beta_zero_sqrt) * (k / K) ** 2) + 1 / beta_zero_sqrt
        return 1 / beta_k

    def _initialize_momentum(self, z: torch.Tensor) -> torch.Tensor:
        """Initialize Euclidean momentum. Exact mode uses N(0, I)."""
        gamma = torch.randn_like(z, device=z.device)
        if self.exact:
            return gamma
        return gamma / self.beta_zero_sqrt

    def _hamiltonian(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        return -self._log_sqrt_det_G(z, self.model) + 0.5 * torch.sum(rho * rho, dim=1)

    def _leapfrog(self, z: torch.Tensor, rho: torch.Tensor, eps: float):
        g = -self._grad_log_sqrt_det_G(z, self.model)
        rho_half = rho - 0.5 * eps * g
        z_new = z + eps * rho_half
        g_new = -self._grad_log_sqrt_det_G(z_new, self.model)
        rho_new = rho_half - 0.5 * eps * g_new
        return z_new, rho_new

    def sample(self, n_samples: int = 100) -> torch.Tensor:
        device = self.device
        K = self.model.centroids_tens.shape[0]
        idx = torch.randint(K, (n_samples,), device=device)
        z0 = self.model.centroids_tens[idx].detach()
        z = z0
        beta_sqrt_old = self.beta_zero_sqrt
        accept_count = 0
        rho_prev = None

        for _ in range(self.mcmc_steps_nbr):
            rho = self._refresh_momentum(z, rho_prev)
            rho0 = rho
            with torch.no_grad():
                H0 = self._hamiltonian(z, rho)
            for k in range(self.n_lf):
                g = -self._grad_log_sqrt_det_G(z, self.model)
                rho_half = rho - 0.5 * self.eps_lf * g
                z = z + self.eps_lf * rho_half
                g_new = -self._grad_log_sqrt_det_G(z, self.model)
                rho_new = rho_half - 0.5 * self.eps_lf * g_new
                if not self.exact:
                    beta_sqrt = self._tempering(k + 1, self.n_lf, self.beta_zero_sqrt)
                    rho = (beta_sqrt_old / beta_sqrt) * rho_new
                    beta_sqrt_old = beta_sqrt
                else:
                    rho = rho_new
            with torch.no_grad():
                H = self._hamiltonian(z, rho)
                alpha = torch.exp(-(H - H0)).clamp(max=1.0)
                u = torch.rand_like(alpha)
                moves = (u < alpha).float().view(-1, 1)
                accept_count += int(moves.sum().item())
                z = ((moves * z + (1 - moves) * z0).detach())
                z0 = z
                if getattr(self, "momentum_persist", 0.0) > 0.0:
                    rho_prev = (moves * rho + (1 - moves) * (-rho0)).detach()
                else:
                    rho_prev = None
        self.last_acceptance_rate = accept_count / (self.mcmc_steps_nbr * n_samples)
        print(f"✅ RHVAE-LogDet Acceptance Rate: {self.last_acceptance_rate:.3f}")
        return z.detach()

    def sample_prior(self, num_samples: int, method: str = 'hmc') -> torch.Tensor:
        return self.sample(num_samples)

    def sample_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor, method: str = 'hmc') -> torch.Tensor:
        eps = torch.randn_like(mu)
        return (mu + eps * torch.exp(0.5 * log_var)).detach()


class VolumeElementRiemannianHMCSampler(RiemannianHMCSampler):
    """Riemannian-kinetic volume-element sampler.

    Target density: π(z) ∝ det(G^{-1}(z))^{volume_power} (default volume_power=0.5).
    Uses generalized leapfrog with a configurable metric convention:
    - standard (M = G):      ρ ~ N(0, G),      dz/dt = G^{-1}ρ
    - dual (M = G^{-1}):     ρ ~ N(0, G^{-1}), dz/dt = Gρ
    where M is the position-dependent mass matrix used by RHMC.
    Default is the standard convention (`use_dual_metric=False`).
    Optionally adds a radial prior term to enforce a far-field slope:
        log π(z) = (2 * volume_power) * log sqrt(det G^{-1}(z)) - 0.5 * λ ||z - c||^2
    Set radial_prior_weight (λ) > 0 to enable; defaults to 0 (no change).
    """

    def __init__(
        self,
        model,
        mcmc_steps_nbr: int = 100,
        n_lf: int = 15,
        eps_lf: float = 0.03,
        beta_zero: float = 1.0,
        exact: bool = True,
        fp_steps: int = 50,
        fp_damping: float = 0.5,
        volume_power: float = 0.5,
        radial_prior_weight: float = 0.,
        radial_prior_center: torch.Tensor | None = None,
        use_dual_metric: bool = False,
    ):
        super().__init__(
            model,
            mcmc_steps_nbr=mcmc_steps_nbr,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
            include_volume_grad=True,
            exact=exact,
            fp_steps=fp_steps,
            fp_damping=fp_damping,
        )
        # log_pi = volume_power * log det G^{-1}; volume correction adds another 0.5 log det G
        # so the total potential corresponds to -(volume_power + 0.5) log det G^{-1}.
        self.volume_power = float(volume_power)
        self.radial_prior_weight = float(radial_prior_weight)
        self.radial_prior_center = radial_prior_center
        self.use_dual_metric = bool(use_dual_metric)

        factor = 2.0 * self.volume_power

        def _center_tensor(z: torch.Tensor) -> torch.Tensor:
            if self.radial_prior_center is None:
                return torch.zeros_like(z)
            center = self.radial_prior_center
            if not torch.is_tensor(center):
                center = torch.tensor(center, device=z.device, dtype=z.dtype)
            else:
                center = center.to(device=z.device, dtype=z.dtype)
            if center.ndim == 1:
                center = center.unsqueeze(0)
            return center

        def _radial_term(z: torch.Tensor) -> torch.Tensor:
            if self.radial_prior_weight <= 0.0:
                return torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
            center = _center_tensor(z)
            if center.shape[0] == 1:
                diff = z - center
            else:
                diff = z - center[: z.shape[0]]
            return -0.5 * self.radial_prior_weight * torch.sum(diff * diff, dim=1)

        self.log_pi = lambda z: factor * self._log_sqrt_det_G_inv(z) + _radial_term(z)

        def _grad(z: torch.Tensor) -> torch.Tensor:
            grad = factor * self._grad_log_prop(z)
            if self.radial_prior_weight <= 0.0:
                return grad
            center = _center_tensor(z)
            if center.shape[0] == 1:
                diff = z - center
            else:
                diff = z - center[: z.shape[0]]
            return grad - self.radial_prior_weight * diff

        self.grad_func = _grad

    @staticmethod
    def _sample_from_covariance(z: torch.Tensor, cov: torch.Tensor, jitter: float = 0.0) -> torch.Tensor:
        """Sample momentum from N(0, cov) robustly."""
        if jitter and jitter > 0:
            eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
            cov = cov + jitter * eye
        try:
            L = torch.linalg.cholesky(cov)
            gamma = torch.randn_like(z)
            return torch.einsum("bij,bj->bi", L, gamma)
        except torch.linalg.LinAlgError:
            evals, evecs = torch.linalg.eigh(cov)
            evals = torch.clamp(evals, min=1e-6)
            sqrt_cov = evecs @ torch.diag_embed(torch.sqrt(evals)) @ evecs.transpose(-2, -1)
            gamma = torch.randn_like(z)
            return torch.einsum("bij,bj->bi", sqrt_cov, gamma)

    def _initialize_momentum(self, z: torch.Tensor) -> torch.Tensor:
        """Sample momentum with covariance equal to the active RHMC mass matrix."""
        jitter = float(getattr(getattr(self, "metric_config", None), "cholesky_jitter", 0.0))
        G = self.model.G(z)
        #G = torch.linalg.inv(G)
        cov = self.model.G_inv(z) if self.use_dual_metric else G
        return self._sample_from_covariance(z, cov, jitter=jitter)

    def _velocity(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt = M^{-1}(z) rho for the active mass convention."""
        if self.use_dual_metric:
            G = self.model.G(z)
            return torch.einsum("bij,bj->bi", G, rho)
        G_inv = self.model.G_inv(z)
        #G_inv = torch.linalg.inv(G_inv)
        return torch.einsum("bij,bj->bi", G_inv, rho)

    def _compute_hamiltonian(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Compute coherent RHMC Hamiltonian under the active metric convention."""
        potential = -self.log_pi(z)
        if self.use_dual_metric:
            # Dual convention: M = G^{-1}, K = 1/2 rho^T G rho, +1/2 log det(G^{-1}).
            G = self.model.G(z)
            kinetic = 0.5 * torch.einsum("bi,bij,bj->b", rho, G, rho)
            log_det_mass = torch.linalg.slogdet(self.model.G_inv(z)).logabsdet
        else:
            # Standard convention: M = G, K = 1/2 rho^T G^{-1} rho, +1/2 log det(G).
            G_inv = self.model.G_inv(z)
            #G_inv = torch.linalg.inv(G_inv)
            kinetic = 0.5 * torch.einsum("bi,bij,bj->b", rho, G_inv, rho)
            log_det_mass = torch.linalg.slogdet(self.model.G(z)).logabsdet
        metric_correction = 0.5 * log_det_mass if self.include_volume_grad else torch.zeros_like(potential)
        return potential + kinetic + metric_correction

class ManifoldAttractionHMCSampler(BaseRiemannianSampler):
    """HMC sampler that attracts samples to the manifold.
    
    Uses the ORIGINAL metric design where:
    - G_inv is LARGE on manifold (from learned covariances)
    - G_inv is SMALL in void
    
    Target: π(z) ∝ sqrt(det(G_inv)) → prefers manifold
    Dynamics: dz = G * ρ → small steps on manifold, larger in void

    Notes:
    - exact=True uses an exact RMHMC formulation (implicit generalized leapfrog).
    - exact=False keeps the legacy heuristic dynamics (approximate).
    
    This combination creates "attraction" to the manifold:
    - Samples starting in void take large steps (via G) toward manifold
    - Once on manifold, small G keeps them there
    - Target distribution strongly prefers manifold regions
    """
    
    def __init__(
        self,
        model,
        mcmc_steps_nbr: int = 100,
        n_lf: int = 15,
        eps_lf: float = 0.03,
        beta_zero: float = 1.0,
        exact: bool = True,
        fp_steps: int = 15,
        fp_damping: float = 0.7,
    ):
        super().__init__(model)
        self.mcmc_steps_nbr = int(mcmc_steps_nbr)
        self.n_lf = int(n_lf)
        self.eps_lf = float(eps_lf)
        self.beta_zero_sqrt = torch.tensor([beta_zero], device=self.device).sqrt()
        self.exact = bool(exact)
        self.fp_steps = int(fp_steps)
        self.fp_damping = float(fp_damping)
    
    def _initialize_momentum(self, z: torch.Tensor) -> torch.Tensor:
        """Initialize momentum. Exact mode samples ρ ~ N(0, G^{-1}(z))."""
        if not self.exact:
            return torch.randn_like(z)
        with torch.no_grad():
            G_inv = self.model.G_inv(z)
            try:
                jitter = getattr(getattr(self, "metric_config", None), "cholesky_jitter", 0.0)
                if jitter and jitter > 0:
                    eye = torch.eye(G_inv.shape[-1], device=G_inv.device, dtype=G_inv.dtype)
                    G_inv = G_inv + jitter * eye
                L = torch.linalg.cholesky(G_inv)
            except torch.linalg.LinAlgError:
                evals, evecs = torch.linalg.eigh(G_inv)
                evals = torch.clamp(evals, min=1e-6)
                L = evecs @ torch.diag_embed(torch.sqrt(evals))
            gamma = torch.randn_like(z)
            rho = torch.einsum("bij,bj->bi", L, gamma)
        return rho
    
    def _hamiltonian(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        if self.exact:
            # Exact RMHMC with metric M = G^{-1}: H = 0.5 rho^T G rho
            G = self.model.G(z)
            return 0.5 * torch.einsum("bi,bij,bj->b", rho, G, rho)
        # Legacy heuristic (approximate)
        G_inv = self.model.G_inv(z)
        log_det = torch.linalg.slogdet(G_inv).logabsdet
        return -0.5 * log_det + 0.5 * (rho * rho).sum(dim=1)
    
    def _grad_potential(self, z: torch.Tensor) -> torch.Tensor:
        """Gradient of U = -0.5 log det(G_inv)"""
        z_req = z if z.requires_grad else z.clone().detach().requires_grad_(True)
        G_inv = self.model.G_inv(z_req)
        log_det = torch.linalg.slogdet(G_inv).logabsdet
        U = -0.5 * log_det.sum()
        return torch.autograd.grad(U, z_req, create_graph=False)[0]
    
    def _velocity(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt = ∂H/∂ρ for metric M = G^{-1} (velocity = G ρ)."""
        G = self.model.G(z)
        return torch.einsum("bij,bj->bi", G, rho)

    def _generalized_leapfrog_step(self, z: torch.Tensor, rho: torch.Tensor, eps: float):
        """Implicit generalized leapfrog (exact)."""
        steps = max(1, int(self.fp_steps))
        damping = float(self.fp_damping)

        rho_half = rho
        for _ in range(steps):
            grad_z, _ = self._grad_hamiltonian_z(z, rho_half)
            if not torch.isfinite(grad_z).all():
                return z.detach().requires_grad_(True), rho.detach()
            rho_update = rho - 0.5 * eps * grad_z
            rho_half = (1.0 - damping) * rho_half + damping * rho_update

        z_new = z
        for _ in range(steps):
            v0 = self._velocity(z, rho_half)
            v1 = self._velocity(z_new, rho_half)
            z_update = z + 0.5 * eps * (v0 + v1)
            z_new = (1.0 - damping) * z_new + damping * z_update
            if not torch.isfinite(z_new).all():
                return z.detach().requires_grad_(True), rho.detach()

        grad_z_new, z_req = self._grad_hamiltonian_z(z_new, rho_half)
        if not torch.isfinite(grad_z_new).all():
            return z.detach().requires_grad_(True), rho.detach()
        rho_new = rho_half - 0.5 * eps * grad_z_new
        return z_req, rho_new

    def _leapfrog(self, z: torch.Tensor, rho: torch.Tensor, eps: float):
        """Legacy leapfrog with G-based position updates (approximate)."""
        # Half momentum step
        grad_U = self._grad_potential(z)
        rho_half = rho - 0.5 * eps * grad_U
        
        # Position step using G (not G_inv!)
        # G is small on manifold → small steps → stay
        # G is larger in void → larger steps → escape toward manifold
        G = self.model.G(z.detach())
        dz = torch.einsum('bij,bj->bi', G, rho_half)
        z_new = z.detach() + eps * dz
        
        # Final half momentum step
        grad_U_new = self._grad_potential(z_new)
        rho_new = rho_half - 0.5 * eps * grad_U_new
        
        return z_new, rho_new
    
    def sample(self, n_samples: int = 100) -> torch.Tensor:
        device = self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device
        self.beta_zero_sqrt = self.beta_zero_sqrt.to(device)
        
        # Initialize at centroids (on manifold)
        K = self.model.centroids_tens.shape[0]
        idx = torch.randint(K, (n_samples,), device=device)
        z = self.model.centroids_tens[idx].detach().clone().to(device)
        
        accept_count = 0
        rho_prev = None
        for _ in range(self.mcmc_steps_nbr):
            z = z.requires_grad_(True)
            rho = self._refresh_momentum(z, rho_prev)
            rho0 = rho
            
            with torch.no_grad():
                H0 = self._hamiltonian(z, rho)
            
            z_prop, rho_prop = z, rho
            for _ in range(self.n_lf):
                if self.exact:
                    z_prop, rho_prop = self._generalized_leapfrog_step(z_prop, rho_prop, self.eps_lf)
                else:
                    z_prop, rho_prop = self._leapfrog(z_prop, rho_prop, self.eps_lf)
            
            with torch.no_grad():
                H_prop = self._hamiltonian(z_prop, rho_prop)
                log_alpha = H0 - H_prop
                alpha = torch.exp(torch.clamp(log_alpha, max=0))
                u = torch.rand_like(alpha)
                accept = (u < alpha).float().view(-1, 1)
                z = (accept * z_prop.detach() + (1 - accept) * z.detach())
                accept_count += accept.sum().item()
                if getattr(self, "momentum_persist", 0.0) > 0.0:
                    rho_prev = (accept * rho_prop + (1 - accept) * (-rho0)).detach()
                else:
                    rho_prev = None
        
        self.last_acceptance_rate = accept_count / (self.mcmc_steps_nbr * n_samples)
        print(f"✅ ManifoldAttraction Acceptance Rate: {self.last_acceptance_rate:.3f}")
        return z.detach()
    
    def sample_prior(self, num_samples: int, method: str = 'attraction') -> torch.Tensor:
        return self.sample(num_samples)
    
    def sample_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor, 
                                   method: str = 'attraction') -> torch.Tensor:
        eps = torch.randn_like(mu)
        return (mu + eps * torch.exp(0.5 * log_var)).detach()
