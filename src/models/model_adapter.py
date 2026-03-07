from __future__ import annotations

from typing import Protocol

import torch


class ModelAdapter(Protocol):
    """Common interface used by low-data benchmark evaluation."""

    model_id: str
    latent_dim: int

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images tensor [B, C, H, W] -> latent [B, D]."""

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent [B, D] -> images [B, C, H, W]."""

    def sample_latents(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Sample latent codes from model prior/sampler."""

    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, n_steps: int = 32) -> torch.Tensor:
        """Interpolation path in latent space [n_steps, D]."""

    def metric_tensor_or_proxy(self, z: torch.Tensor) -> torch.Tensor:
        """Return inverse metric/proxy [B, D, D] for geometry diagnostics."""
