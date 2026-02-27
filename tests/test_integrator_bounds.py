import pytest
import torch
from src.models.rhvae_geometry import GeometryRHVAEConfig, GeometryRHVAE
from src.models.samplers.hmc_sampler import VolumeElementRiemannianHMCSampler
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP

def test_effective_eps_clipping_behavior():
    config = GeometryRHVAEConfig(
        latent_dim=2,
        input_dim=(2,),
    )
    model = GeometryRHVAE(
        model_config=config,
        encoder=Encoder_VAE_MLP(config),
        decoder=Decoder_AE_MLP(config)
    )
    
    # Needs centroids set to avoid errors in metric evaluation
    mu = torch.randn(5, 2)
    model.set_centroids(mu)
    
    sampler = VolumeElementRiemannianHMCSampler(
        model=model,
        adaptive_dual_step=True,
        adaptive_max_dual_displacement=0.5,
        adaptive_min_step_scale=0.1,
    )
    
    z = torch.randn(10, 2)
    # Give extremely large momentum so speed_max is huge and triggers step scaling
    rho = torch.randn(10, 2) * 1000.0
    eps_base = 0.05
    
    eps_eff, scale = sampler._effective_eps(z, rho, eps_base)
    
    # It should clip the scale
    assert scale < 1.0, "Scale was not clipped despite huge momentum!"
    assert scale >= 0.1, "Scale fell below minimum allowed scale!"
    assert eps_eff == eps_base * scale

    # Now small momentum
    rho_small = torch.randn(10, 2) * 0.001
    eps_eff_small, scale_small = sampler._effective_eps(z, rho_small, eps_base)
    
    assert scale_small == 1.0, "Scale was clipped despite small momentum!"
    assert eps_eff_small == eps_base
