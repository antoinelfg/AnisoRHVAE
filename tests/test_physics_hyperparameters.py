import pytest
import torch
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from src.models.rhvae_geometry import GeometryRHVAEConfig, GeometryRHVAE

def test_physics_hyperparameters_initialization():
    config = GeometryRHVAEConfig.from_physics(
        temperature=0.5,
        latent_dim=2,
        input_dim=(2,), 
        kernel_type="isotropic",
        use_attractor=True,
    )
    
    assert abs(config.void_threshold * config.temperature - 1.25) < 0.1
    assert abs(config.transition_steepness - 7.27) < 0.5
    assert abs(config.attractor_gamma - 8.0) < 0.1
    assert abs(config.void_decay_scale - 9.2) < 0.5
    assert abs(config.radial_stretch - 9.0) < 0.5

def test_metric_evaluation_with_physics_parameters():
    config = GeometryRHVAEConfig.from_physics(
        temperature=0.5,
        latent_dim=2,
        input_dim=(2,), 
        kernel_type="isotropic",
        use_attractor=True,
    )
    model = GeometryRHVAE(
        model_config=config,
        encoder=Encoder_VAE_MLP(config),
        decoder=Decoder_AE_MLP(config)
    )
    z = torch.randn(10, 2)
    mu = torch.randn(5, 2)
    model.set_centroids(mu)
    
    # Should evaluate without errors
    g_inv = model.G_inv(z)
    assert g_inv.shape == (10, 2, 2)
    assert torch.isfinite(g_inv).all()

def test_auto_temperature_dynamic_shifting():
    config = GeometryRHVAEConfig.from_physics(
        temperature=0.5,
        latent_dim=2,
        input_dim=(2,), 
        kernel_type="isotropic",
        use_attractor=True,
    )
    model = GeometryRHVAE(
        model_config=config,
        encoder=Encoder_VAE_MLP(config),
        decoder=Decoder_AE_MLP(config)
    )
    
    with torch.no_grad():
        model.temperature.fill_(0.2)
        model.update_physics_parameters()

    assert abs(model.void_threshold * model.temperature.item() - 0.529) < 0.1
    assert abs(model.transition_steepness - 18.13) < 0.5
    assert abs(model.attractor_gamma - 50.0) < 0.5
