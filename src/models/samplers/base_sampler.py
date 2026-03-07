"""
Base Riemannian Sampler
======================

Abstract base class for all Riemannian sampling strategies.
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseRiemannianSampler(ABC):
    """
    Abstract base class for Riemannian sampling strategies.
    
    All samplers should inherit from this class and implement the required methods.
    """
    
    def __init__(self, model):
        """
        Initialize the sampler with a reference to the model.
        
        Args:
            model: The Riemannian VAE model that provides metric tensor functions
        """
        self.model = model
        self.device = next(model.parameters()).device
        # Defaults for implicit generalized leapfrog (exact RHMC)
        self.fp_steps = 15
        self.fp_damping = 0.7
        # Exact sampling by default unless a sampler opts out
        self.exact = True
        # Momentum persistence (0.0 = full refresh each trajectory)
        self.momentum_persist = 0.0
        
    @abstractmethod
    def sample_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor, 
                                 method: str = 'enhanced') -> torch.Tensor:
        """
        Sample latent codes using Riemannian geometry.
        
        Args:
            mu: Posterior mean [batch_size, latent_dim]
            log_var: Posterior log variance [batch_size, latent_dim]
            method: Sampling method to use
            
        Returns:
            Sampled latent codes [batch_size, latent_dim]
        """
        pass
    
    @abstractmethod
    def sample_prior(self, num_samples: int, method: str = 'geodesic') -> torch.Tensor:
        """
        Sample from the Riemannian prior.
        
        Args:
            num_samples: Number of samples to generate
            method: Prior sampling method to use
            
        Returns:
            Prior samples [num_samples, latent_dim]
        """
        pass
    
    def validate_metric_availability(self) -> bool:
        """
        Check if the model has the required metric tensor components.
        
        Returns:
            True if metric components are available, False otherwise
        """
        required_attrs = ['centroids_tens', 'M_tens', 'G', 'G_inv']
        return all(hasattr(self.model, attr) for attr in required_attrs)
    
    def get_sampling_methods(self) -> Dict[str, str]:
        """
        Get available sampling methods for this sampler.
        
        Returns:
            Dictionary mapping method names to descriptions
        """
        return {
            'enhanced': 'Enhanced Riemannian sampling with centroid influence',
            'geodesic': 'Geodesic-aware sampling along manifold paths',
            'basic': 'Basic metric-aware sampling',
            'standard': 'Standard reparameterization (no Riemannian)'
        }
    
    def get_sampler_info(self) -> Dict[str, Any]:
        """
        Get information about this sampler.
        
        Returns:
            Dictionary with sampler information
        """
        return {
            'sampler_type': self.__class__.__name__,
            'available_methods': list(self.get_sampling_methods().keys()),
            'metric_available': self.validate_metric_availability(),
            'device': str(self.device)
        }

    def _hamiltonian_value(self, z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Return H(z, rho) using whichever Hamiltonian method the sampler defines."""
        if hasattr(self, "_compute_hamiltonian"):
            return self._compute_hamiltonian(z, rho)
        if hasattr(self, "_hamiltonian"):
            return self._hamiltonian(z, rho)
        raise AttributeError(f"{self.__class__.__name__} has no Hamiltonian method")

    def _grad_hamiltonian_z(self, z: torch.Tensor, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ∇_z H(z, rho) via autograd (exact, includes kinetic-gradient terms)."""
        # Detach to avoid backprop-through-graph errors across implicit iterations
        z_req = z.detach().requires_grad_(True)
        rho_req = rho.detach()
        H = self._hamiltonian_value(z_req, rho_req)
        grad = torch.autograd.grad(H.sum(), z_req, create_graph=False)[0]
        return grad, z_req

    def _refresh_momentum(self, z: torch.Tensor, rho_prev: torch.Tensor | None) -> torch.Tensor:
        """Partially refresh momentum while preserving the target momentum distribution."""
        rho_fresh = self._initialize_momentum(z)
        alpha = float(getattr(self, "momentum_persist", 0.0))
        if rho_prev is None or alpha <= 0.0:
            return rho_fresh
        alpha = max(0.0, min(alpha, 0.999))
        scale = (1.0 - alpha ** 2) ** 0.5
        return alpha * rho_prev + scale * rho_fresh
