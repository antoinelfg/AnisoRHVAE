from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys

import torch
import torch.nn.functional as F


def _load_pythae():
    vendor_root = Path(__file__).resolve().parents[2] / "src" / "lib" / "src"
    pkg_root = vendor_root / "pythae"
    if vendor_root.exists():
        if str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))
        for name in list(sys.modules.keys()):
            if name == "pythae" or name.startswith("pythae."):
                del sys.modules[name]
    try:
        from pythae.models.rhvae import RHVAE, RHVAEConfig
        from pythae.models.rhvae.rhvae_utils import create_inverse_metric, create_metric
        return RHVAE, RHVAEConfig, create_inverse_metric, create_metric
    except ModuleNotFoundError:
        if not pkg_root.exists():
            raise
        import importlib.util
        import importlib

        def _load_module(name: str, path: Path, search: list[str]):
            spec = importlib.util.spec_from_file_location(
                name,
                path,
                submodule_search_locations=search,
            )
            if spec is None or spec.loader is None:
                raise ModuleNotFoundError(f"Unable to load {name} from {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        _load_module("pythae", pkg_root / "__init__.py", [str(pkg_root)])
        models_root = pkg_root / "models"
        _load_module("pythae.models", models_root / "__init__.py", [str(models_root)])
        rhvae_root = models_root / "rhvae"
        _load_module("pythae.models.rhvae", rhvae_root / "__init__.py", [str(rhvae_root)])
        importlib.invalidate_caches()
        from pythae.models.rhvae import RHVAE, RHVAEConfig
        from pythae.models.rhvae.rhvae_utils import create_inverse_metric, create_metric
        return RHVAE, RHVAEConfig, create_inverse_metric, create_metric


RHVAE, RHVAEConfig, create_inverse_metric, create_metric = _load_pythae()


@dataclass
class GeometryRHVAEConfig(RHVAEConfig):
    """Geometry-aware RHVAE configuration."""

    use_attractor: bool = False
    attractor_smoothness: str = "soft"
    attractor_metric: str = "euclidean"
    attractor_gamma: float = 5.0
    attractor_k_nearest: int = 5
    attractor_use_det: bool = False
    attractor_bias_energy: float = 15.0

    void_threshold: float = 1.5
    void_weight_threshold: float = -1.0
    void_decay_type: str = "invquad"
    void_decay_scale: float = 1.0
    void_decay_power: float = 2.0
    void_decay_softplus_k: float = 5.0
    radial_stretch: float = 10.0  # beta
    transition_steepness: float = 5.0
    target_anisotropy: float | None = None
    void_eigshape_mode: str = "none"  # none | det_preserving_spectral
    void_eigshape_alpha_min: float = 1.0
    void_eigshape_power: float = -1.0
    void_eigshape_eig_floor: float = 1e-8

    kernel_type: str = "isotropic"
    precision_jitter: float = 1e-6
    atom_power: float = 1.0
    kernel_power: float = 1.0
    atom_norm: str = "none"
    atom_scale: float = 1.0

    # RHMC integrator control (training)
    rhmc_integrator: str = "explicit"  # explicit | implicit
    rhmc_fp_steps: int = 15
    rhmc_fp_damping: float = 0.7
    rhmc_adaptive_dual_step: bool = False
    rhmc_adaptive_max_dual_displacement: float = 0.05
    rhmc_adaptive_min_step_scale: float = 0.1

    @classmethod
    def from_physics(
        cls,
        *,
        temperature: float,
        regularization: float,
        target_anisotropy: float,
        confidence_threshold: float,
        **kwargs,
    ) -> "GeometryRHVAEConfig":
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if confidence_threshold <= 0 or confidence_threshold >= 1:
            raise ValueError("confidence_threshold must be in (0, 1)")
        if target_anisotropy <= 1:
            raise ValueError("target_anisotropy must be > 1")

        temperature = float(temperature)
        regularization = float(regularization)
        target_anisotropy = float(target_anisotropy)
        confidence_threshold = float(confidence_threshold)

        r0 = temperature * math.sqrt(-math.log(confidence_threshold))
        radial_stretch = regularization * (target_anisotropy - 1.0)

        params = {
            "temperature": temperature,
            "regularization": regularization,
            "radial_stretch": radial_stretch,
            "void_weight_threshold": confidence_threshold,
            "void_threshold": r0 / temperature,
            "target_anisotropy": target_anisotropy,
            "void_decay_scale": 1.0,
            "void_decay_power": 2.0,
            "void_decay_softplus_k": 5.0,
        }
        params.update(kwargs)
        return cls(**params)


class GeometryRHVAE(RHVAE):
    """RHVAE with hot-swappable metric atoms and optional attractor geometry."""

    def __init__(self, model_config: GeometryRHVAEConfig, **kwargs):
        super().__init__(model_config, **kwargs)
        self.rhmc_integrator = str(model_config.rhmc_integrator).lower()
        self.rhmc_fp_steps = int(model_config.rhmc_fp_steps)
        self.rhmc_fp_damping = float(model_config.rhmc_fp_damping)
        self.use_attractor = bool(model_config.use_attractor)
        self.attractor_metric = str(model_config.attractor_metric).lower()
        self.attractor_gamma = float(model_config.attractor_gamma)
        self.attractor_k_nearest = int(model_config.attractor_k_nearest)
        self.attractor_use_det = bool(model_config.attractor_use_det)
        self.attractor_bias_energy = float(model_config.attractor_bias_energy)
        self.void_threshold = float(model_config.void_threshold)
        self.void_weight_threshold = float(model_config.void_weight_threshold)
        self.void_decay_type = str(model_config.void_decay_type).lower()
        self.void_decay_scale = float(model_config.void_decay_scale)
        self.void_decay_power = float(model_config.void_decay_power)
        self.void_decay_softplus_k = float(model_config.void_decay_softplus_k)
        self.radial_stretch = float(model_config.radial_stretch)
        self.transition_steepness = float(model_config.transition_steepness)
        self.void_eigshape_mode = str(
            getattr(model_config, "void_eigshape_mode", "none")
        ).lower()
        self.void_eigshape_alpha_min = float(
            getattr(model_config, "void_eigshape_alpha_min", 1.0)
        )
        self.void_eigshape_power = float(
            getattr(model_config, "void_eigshape_power", -1.0)
        )
        self.void_eigshape_eig_floor = float(
            getattr(model_config, "void_eigshape_eig_floor", 1e-8)
        )
        self.rhmc_adaptive_dual_step = bool(
            getattr(model_config, "rhmc_adaptive_dual_step", False)
        )
        self.rhmc_adaptive_max_dual_displacement = float(
            getattr(model_config, "rhmc_adaptive_max_dual_displacement", 0.05)
        )
        self.rhmc_adaptive_min_step_scale = float(
            getattr(model_config, "rhmc_adaptive_min_step_scale", 0.1)
        )
        self.target_anisotropy = (
            None
            if model_config.target_anisotropy is None
            else float(model_config.target_anisotropy)
        )
        self.kernel_type = str(model_config.kernel_type).lower()
        self.precision_jitter = float(model_config.precision_jitter)
        self.atom_power = float(model_config.atom_power)
        self.kernel_power = float(model_config.kernel_power)
        self.atom_norm = str(model_config.atom_norm).lower()
        self.atom_scale = float(model_config.atom_scale)
        self.attractor_smoothness = str(
            getattr(model_config, "attractor_smoothness", "soft")
        ).lower()
        if self.attractor_smoothness not in {"hard", "soft"}:
            raise ValueError("attractor_smoothness must be 'hard' or 'soft'")
        if self.attractor_metric not in {"euclidean", "mahalanobis"}:
            raise ValueError("attractor_metric must be 'euclidean' or 'mahalanobis'")
        if self.attractor_bias_energy <= 0:
            raise ValueError("attractor_bias_energy must be > 0")
        if self.kernel_type not in {"isotropic", "mahalanobis"}:
            raise ValueError("kernel_type must be 'isotropic' or 'mahalanobis'")
        if self.kernel_type == "mahalanobis" and self.atom_norm != "trace":
            self.atom_norm = "trace"
        if self.atom_norm not in {"none", "trace", "det"}:
            raise ValueError("atom_norm must be 'none', 'trace', or 'det'")
        if self.void_decay_type not in {"none", "invquad"}:
            raise ValueError("void_decay_type must be 'none' or 'invquad'")
        if self.void_eigshape_mode not in {"none", "det_preserving_spectral"}:
            raise ValueError(
                "void_eigshape_mode must be 'none' or 'det_preserving_spectral'"
            )
        if self.void_weight_threshold > 0 and self.void_weight_threshold >= 1:
            raise ValueError("void_weight_threshold must be in (0, 1)")
        if not (0.0 <= self.void_eigshape_alpha_min <= 1.0):
            raise ValueError("void_eigshape_alpha_min must be in [0, 1]")
        if self.void_eigshape_eig_floor <= 0:
            raise ValueError("void_eigshape_eig_floor must be > 0")
        if not math.isfinite(self.void_eigshape_power):
            raise ValueError("void_eigshape_power must be finite")
        if self.void_decay_type != "none":
            if self.void_decay_scale <= 0:
                raise ValueError("void_decay_scale must be > 0")
            if self.void_decay_power <= 0:
                raise ValueError("void_decay_power must be > 0")
            if self.void_decay_softplus_k <= 0:
                raise ValueError("void_decay_softplus_k must be > 0")
        self.P_tens = torch.empty(0, self.latent_dim, self.latent_dim)
        if self.attractor_metric == "mahalanobis" or self.attractor_use_det:
            self._update_attractor_precisions(self.M_tens)
        self._refresh_metric_hooks()

    def _refresh_metric_hooks(self) -> None:
        if self.use_attractor or self.kernel_type != "isotropic":
            self.G_inv = self._compute_inverse_metric_at_z
            self.G = self._compute_metric_at_z
        else:
            self.G = create_metric(self)
            self.G_inv = create_inverse_metric(self)

    @property
    def device(self) -> torch.device:
        """Return the device of the model parameters."""
        if hasattr(self, '_device') and self._device is not None:
            return self._device
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')
    
    @device.setter
    def device(self, value):
        """Allow setting device (for compatibility with base class)."""
        self._device = value

    def set_atoms(self, atoms: torch.Tensor) -> None:
        """Overwrite metric atoms (M_tens) in memory."""
        if atoms.dim() != 3:
            raise ValueError("atoms must be a [K, D, D] tensor")
        self.M_tens = atoms.to(next(self.parameters()).device)
        if self.attractor_metric == "mahalanobis" or self.attractor_use_det:
            self._update_attractor_precisions(self.M_tens)

    def set_centroids(self, centroids: torch.Tensor) -> None:
        """Overwrite centroids in memory."""
        if centroids.dim() != 2:
            raise ValueError("centroids must be a [K, D] tensor")
        self.centroids_tens = centroids.to(next(self.parameters()).device)

    def _update_attractor_precisions(self, atoms: torch.Tensor) -> None:
        cov, _ = self._prepare_atoms(atoms)
        self.P_tens = self._precision_from_cov(cov)

    def _get_attractor_precisions(
        self, atoms: torch.Tensor | None, device: torch.device
    ) -> torch.Tensor:
        if atoms is None and self.P_tens.numel() > 0:
            return self.P_tens.to(device)
        if atoms is None:
            atoms = self.M_tens
        cov, _ = self._prepare_atoms(atoms.to(device))
        return self._precision_from_cov(cov)

    def _compute_base_inverse_metric(self, z: torch.Tensor) -> torch.Tensor:
        centroids = self.centroids_tens.to(z.device)
        atoms = self.M_tens.to(z.device)
        cov, prec = self._prepare_atoms(atoms)
        diff = centroids.unsqueeze(0) - z.unsqueeze(1)
        dists_sq = self._kernel_dists(diff, prec)
        temperature = self.temperature.to(device=z.device, dtype=z.dtype)
        weights = torch.exp(-dists_sq / (temperature**2))
        base = torch.einsum("bk,kij->bij", weights, cov)
        eye = torch.eye(self.latent_dim, device=z.device, dtype=z.dtype).unsqueeze(0)
        base = base + self.lbd.to(device=z.device, dtype=z.dtype) * eye
        return base

    def _compute_soft_attractor_weights(
        self,
        z: torch.Tensor,
        centroids: torch.Tensor,
        precisions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        diff = z.unsqueeze(1) - centroids.unsqueeze(0)
        needs_precisions = self.attractor_metric == "mahalanobis" or self.attractor_use_det
        if needs_precisions and precisions is None:
            precisions = self._get_attractor_precisions(None, z.device)
        if precisions is not None:
            precisions = precisions.to(z.device)

        if self.attractor_metric == "mahalanobis":
            if precisions is None:
                raise ValueError("precision matrix required for mahalanobis attractor")
            tmp = torch.einsum("bkd,kde->bke", diff, precisions)
            dists_sq = torch.einsum("bke,bke->bk", tmp, diff)
        else:
            dists_sq = torch.einsum("bkd,bkd->bk", diff, diff)

        energy = self.attractor_gamma * dists_sq
        if self.attractor_use_det:
            if precisions is None:
                raise ValueError("precision matrix required for determinant weighting")
            _, logabsdet = torch.linalg.slogdet(precisions)
            logdet_cov = -logabsdet
            energy = energy + 0.5 * logdet_cov.unsqueeze(0)

        weights = torch.softmax(-energy, dim=1)
        k_nearest = int(self.attractor_k_nearest)
        if 0 < k_nearest < weights.shape[1]:
            top_vals, top_idx = torch.topk(weights, k_nearest, dim=1)
            pruned = torch.zeros_like(weights)
            pruned.scatter_(1, top_idx, top_vals)
            denom = pruned.sum(dim=1, keepdim=True).clamp_min(1e-12)
            weights = pruned / denom
        return weights

    def _attractor_distances_sq(
        self, diff: torch.Tensor, precisions: torch.Tensor | None
    ) -> torch.Tensor:
        if self.attractor_metric == "mahalanobis":
            if precisions is None:
                raise ValueError("precision matrix required for mahalanobis attractor")
            tmp = torch.einsum("bkd,kde->bke", diff, precisions)
            return torch.einsum("bke,bke->bk", tmp, diff)
        return torch.einsum("bkd,bkd->bk", diff, diff)

    def _min_euclidean_distance(self, diff: torch.Tensor) -> torch.Tensor:
        dists_sq = torch.einsum("bkd,bkd->bk", diff, diff)
        min_dists_sq = dists_sq.min(dim=1).values
        return torch.sqrt(min_dists_sq + 1e-10)

    def _compute_directional_uu(
        self,
        z: torch.Tensor,
        centroids: torch.Tensor,
        precisions: torch.Tensor | None,
    ) -> torch.Tensor:
        diff = centroids.unsqueeze(0) - z.unsqueeze(1)
        if self.attractor_smoothness == "hard":
            dists_sq = self._attractor_distances_sq(diff, precisions)
            nearest_idx = dists_sq.min(dim=1).indices
            direction = centroids[nearest_idx] - z
            denom = torch.linalg.norm(direction, dim=1, keepdim=True).clamp_min(1e-8)
            u_hat = direction / denom
            return torch.einsum("bi,bj->bij", u_hat, u_hat)

        weights = self._compute_soft_attractor_weights(z, centroids, precisions)
        denom = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp_min(1e-8)
        u_hat = diff / denom
        u_hat_flat = u_hat.reshape(-1, u_hat.shape[-1])
        uu_t = torch.einsum("bi,bj->bij", u_hat_flat, u_hat_flat)
        uu_t = uu_t.view(z.shape[0], centroids.shape[0], z.shape[1], z.shape[1])
        return torch.einsum("bk,bkij->bij", weights, uu_t)

    def _compute_void_inverse_metric(
        self,
        z: torch.Tensor,
        centroids: torch.Tensor,
        precisions: torch.Tensor | None,
    ) -> torch.Tensor:
        radial_uu_t = self._compute_directional_uu(z, centroids, precisions)
        beta = torch.tensor(self.radial_stretch, device=z.device, dtype=z.dtype)
        eye = torch.eye(self.latent_dim, device=z.device, dtype=z.dtype).unsqueeze(0)
        return beta * radial_uu_t + self.lbd.to(device=z.device, dtype=z.dtype) * eye

    @staticmethod
    def _det_preserving_eigshape(
        mats: torch.Tensor,
        power: float,
        eig_floor: float,
    ) -> torch.Tensor:
        """Reshape eigenvalues while preserving determinant per matrix.

        λ'_i = g * (λ_i / g)^s with g = exp(mean(log λ)).
        """
        mats = 0.5 * (mats + mats.transpose(-1, -2))
        evals, evecs = torch.linalg.eigh(mats)
        evals = torch.clamp(evals, min=float(eig_floor))
        log_g = torch.mean(torch.log(evals), dim=-1, keepdim=True)
        g = torch.exp(log_g)
        evals_new = g * torch.pow(evals / g, float(power))
        out = evecs @ torch.diag_embed(evals_new) @ evecs.transpose(-1, -2)
        return 0.5 * (out + out.transpose(-1, -2))

    def _apply_void_eigshape(self, void: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Apply optional determinant-preserving spectral reshaping in high-alpha void."""
        if self.void_eigshape_mode == "none":
            return void
        alpha_flat = alpha.reshape(void.shape[0], -1)[:, 0]
        mask = alpha_flat >= float(self.void_eigshape_alpha_min)
        if not bool(mask.any()):
            return void
        out = void.clone()
        out[mask] = self._det_preserving_eigshape(
            out[mask],
            power=float(self.void_eigshape_power),
            eig_floor=float(self.void_eigshape_eig_floor),
        )
        return 0.5 * (out + out.transpose(-1, -2))

    def _compute_inverse_metric_at_z(self, z: torch.Tensor) -> torch.Tensor:
        base = self._compute_base_inverse_metric(z)
        base = self._stabilize_metric(base)
        if not self.use_attractor:
            return base

        centroids = self.centroids_tens.to(z.device)
        diff_eucl = centroids.unsqueeze(0) - z.unsqueeze(1)
        min_dists_eucl = self._min_euclidean_distance(diff_eucl)

        precisions = None
        if self.attractor_metric == "mahalanobis" or self.attractor_use_det:
            precisions = self._get_attractor_precisions(None, z.device)

        radial_uu_t = self._compute_directional_uu(z, centroids, precisions)
        decay_transverse = self._compute_void_decay(min_dists_eucl).view(-1, 1, 1)
        decay_longitudinal = 1.0
        term_long = (self.radial_stretch * decay_longitudinal) * radial_uu_t
        eye = torch.eye(self.latent_dim, device=z.device, dtype=z.dtype).unsqueeze(0)
        term_trans = self.lbd.to(device=z.device, dtype=z.dtype) * decay_transverse * eye
        void = term_long + term_trans

        alpha = self._compute_alpha(min_dists_eucl)
        void = self._apply_void_eigshape(void, alpha)
        alpha = alpha.view(-1, 1, 1)
        blended = (1.0 - alpha) * base + alpha * void
        return self._stabilize_metric(blended)

    def _compute_metric_at_z(self, z: torch.Tensor) -> torch.Tensor:
        try:
            g_inv = self._compute_inverse_metric_at_z(z)
            return torch.linalg.inv(g_inv)
        except (torch.linalg.LinAlgError, ValueError):
            # Return a non-finite tensor to trigger the sampler's fallback
            return torch.full((z.shape[0], self.latent_dim, self.latent_dim), float('nan'), device=z.device, dtype=z.dtype)

    def _kernel_dists(self, diff: torch.Tensor, prec: torch.Tensor | None) -> torch.Tensor:
        if self.kernel_type == "isotropic":
            return torch.einsum("bkd,bkd->bk", diff, diff)
        if prec is None:
            raise ValueError("precision matrix required for mahalanobis kernel")
        tmp = torch.einsum("bkd,kde->bke", diff, prec)
        return torch.einsum("bke,bke->bk", tmp, diff)

    def _kernel_dists_batch(self, diff: torch.Tensor, prec: torch.Tensor | None) -> torch.Tensor:
        if self.kernel_type == "isotropic":
            return torch.einsum("ijd,ijd->ij", diff, diff)
        if prec is None:
            raise ValueError("precision matrix required for mahalanobis kernel")
        tmp = torch.einsum("ijd,jde->ije", diff, prec)
        return torch.einsum("ije,ije->ij", tmp, diff)

    def _prepare_atoms(self, atoms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        cov = 0.5 * (atoms + atoms.transpose(-1, -2))
        needs_shape = (
            abs(self.atom_power - 1.0) > 1e-6
            or abs(self.kernel_power - 1.0) > 1e-6
            or self.atom_norm != "none"
        )
        if not needs_shape and self.kernel_type == "isotropic":
            return cov, None
        if not needs_shape and self.kernel_type == "mahalanobis":
            return cov, self._precision_from_cov(cov)

        work = cov.float() if cov.dtype in (torch.float16, torch.bfloat16) else cov
        evals, evecs = torch.linalg.eigh(work)
        evals = evals.clamp_min(self.precision_jitter)

        evals_cov = evals.pow(self.atom_power)
        mean_orig = evals.mean(dim=-1, keepdim=True).clamp_min(1e-12)
        mean_shaped = evals_cov.mean(dim=-1, keepdim=True).clamp_min(1e-12)
        evals_cov = evals_cov * (mean_orig / mean_shaped)

        if self.atom_norm == "trace":
            scale = evals_cov.mean(dim=-1, keepdim=True).clamp_min(1e-12)
            evals_cov = evals_cov / scale
        elif self.atom_norm == "det":
            logdet = torch.log(evals_cov).sum(dim=-1, keepdim=True)
            scale = torch.exp(logdet / float(self.latent_dim)).clamp_min(1e-12)
            evals_cov = evals_cov / scale

        # Apply global atom scale
        evals_cov = evals_cov * float(self.atom_scale)

        cov_shaped = evecs @ torch.diag_embed(evals_cov) @ evecs.transpose(-1, -2)
        cov_shaped = 0.5 * (cov_shaped + cov_shaped.transpose(-1, -2))
        cov_shaped = cov_shaped.to(cov.dtype)

        if self.kernel_type == "mahalanobis":
            evals_prec = (1.0 / evals_cov).clamp_min(1e-12)
            evals_prec = evals_prec.pow(self.kernel_power)
            prec = evecs @ torch.diag_embed(evals_prec) @ evecs.transpose(-1, -2)
            prec = 0.5 * (prec + prec.transpose(-1, -2))
            prec = prec.to(cov.dtype)
            return cov_shaped, prec

        return cov_shaped, None

    def _precision_from_cov(self, cov: torch.Tensor) -> torch.Tensor:
        d = cov.shape[-1]
        eye = torch.eye(d, device=cov.device, dtype=cov.dtype).unsqueeze(0)
        mats = cov + self.precision_jitter * eye
        chol = torch.linalg.cholesky(mats)
        prec = torch.cholesky_inverse(chol)
        return 0.5 * (prec + prec.transpose(-1, -2))

    def _stabilize_metric(self, g_inv: torch.Tensor) -> torch.Tensor:
        g_inv = 0.5 * (g_inv + g_inv.transpose(-1, -2))
        if self.precision_jitter > 0:
            eye = torch.eye(g_inv.shape[-1], device=g_inv.device, dtype=g_inv.dtype).unsqueeze(0)
            g_inv = g_inv + self.precision_jitter * eye
        return g_inv

    def _compute_training_inverse_metric(
        self, z: torch.Tensor, mu: torch.Tensor, M: torch.Tensor
    ) -> torch.Tensor:
        cov, prec = self._prepare_atoms(M)
        diff = mu.unsqueeze(0) - z.unsqueeze(1)
        dists_sq = self._kernel_dists_batch(diff, prec)
        temperature = self.temperature.to(device=z.device, dtype=z.dtype)
        weights = torch.exp(-dists_sq / (temperature**2))
        base = torch.einsum("bj,jkl->bkl", weights, cov)
        base = base + self.lbd.to(device=z.device, dtype=z.dtype) * torch.eye(
            self.latent_dim, device=z.device, dtype=z.dtype
        )
        base = self._stabilize_metric(base)
        if not self.use_attractor:
            return base

        centroids = mu
        atoms = M
        if len(self.centroids) > 0:
            try:
                centroids = torch.cat(list(self.centroids), dim=0).to(z.device)
                if len(self.M) > 0:
                    atoms = torch.cat(list(self.M), dim=0).to(z.device)
            except RuntimeError:
                centroids = mu
                atoms = M
        if atoms.shape[0] != centroids.shape[0]:
            centroids = mu
            atoms = M

        precisions = None
        if self.attractor_metric == "mahalanobis" or self.attractor_use_det:
            cov_global, _ = self._prepare_atoms(atoms)
            precisions = self._precision_from_cov(cov_global)

        diff_eucl = centroids.unsqueeze(0) - z.unsqueeze(1)
        min_dists_eucl = self._min_euclidean_distance(diff_eucl)

        radial_uu_t = self._compute_directional_uu(z, centroids, precisions)
        decay_transverse = self._compute_void_decay(min_dists_eucl).view(-1, 1, 1)
        decay_longitudinal = 1.0
        term_long = (self.radial_stretch * decay_longitudinal) * radial_uu_t
        eye = torch.eye(self.latent_dim, device=z.device, dtype=z.dtype).unsqueeze(0)
        term_trans = self.lbd.to(device=z.device, dtype=z.dtype) * decay_transverse * eye
        void = term_long + term_trans

        alpha = self._compute_alpha(min_dists_eucl)
        void = self._apply_void_eigshape(void, alpha)
        alpha = alpha.view(-1, 1, 1)
        blended = (1.0 - alpha) * base + alpha * void
        return self._stabilize_metric(blended)

    def _void_from_direction(self, u_hat: torch.Tensor) -> torch.Tensor:
        uu_t = torch.einsum("bi,bj->bij", u_hat, u_hat)
        beta = torch.tensor(self.radial_stretch, device=u_hat.device, dtype=u_hat.dtype)
        eye = torch.eye(u_hat.shape[1], device=u_hat.device, dtype=u_hat.dtype).unsqueeze(0)
        return beta * uu_t + self.lbd.to(device=u_hat.device, dtype=u_hat.dtype) * eye

    def _compute_r0(self, min_dists: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature.to(device=min_dists.device, dtype=min_dists.dtype)
        if self.void_weight_threshold > 0:
            tau = torch.tensor(
                self.void_weight_threshold,
                device=min_dists.device,
                dtype=min_dists.dtype,
            )
            return temperature * torch.sqrt(-torch.log(tau))
        return self.void_threshold * temperature

    def _compute_void_decay(self, min_dists: torch.Tensor) -> torch.Tensor:
        if self.void_decay_type == "none":
            return torch.ones_like(min_dists)
        r0 = self._compute_r0(min_dists)
        delta = min_dists - r0
        k = float(self.void_decay_softplus_k)
        soft_delta = F.softplus(delta, beta=k)
        scale = torch.tensor(
            self.void_decay_scale, device=min_dists.device, dtype=min_dists.dtype
        ).clamp_min(1e-12)
        power = torch.tensor(
            self.void_decay_power, device=min_dists.device, dtype=min_dists.dtype
        ).clamp_min(1e-12)
        return 1.0 / (1.0 + torch.pow(soft_delta / scale, power))

    def _compute_alpha(self, min_dists: torch.Tensor) -> torch.Tensor:
        r0 = self._compute_r0(min_dists)
        return torch.sigmoid((min_dists - r0) * self.transition_steepness)

    def _metric_quantities(
        self,
        z: torch.Tensor,
        mu: torch.Tensor | None = None,
        M: torch.Tensor | None = None,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if training and mu is not None and M is not None:
            G_inv = self._compute_training_inverse_metric(z, mu, M)
        else:
            G_inv = self.G_inv(z)
        logabsdet = torch.linalg.slogdet(G_inv).logabsdet
        G_log_det = -logabsdet
        return G_inv, G_log_det

    def _grad_z_hamiltonian(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        rho: torch.Tensor,
        mu: torch.Tensor | None = None,
        M: torch.Tensor | None = None,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_req = z if z.requires_grad else z.clone().detach().requires_grad_(True)
        if recon_x is None:
            recon_x = self.decoder(z_req)["reconstruction"]
        G_inv, G_log_det = self._metric_quantities(z_req, mu, M, training=training)
        H = self._hamiltonian(recon_x, x, z_req, rho, G_inv, G_log_det)
        grad = torch.autograd.grad(
            H.sum(), z_req, create_graph=self.training, retain_graph=True
        )[0]
        return grad, z_req, G_inv, G_log_det

    def _generalized_leapfrog_implicit(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        rho: torch.Tensor,
        mu: torch.Tensor | None = None,
        M: torch.Tensor | None = None,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = float(self.eps_lf)
        steps = max(1, int(self.rhmc_fp_steps))
        damping = float(self.rhmc_fp_damping)
        adaptive = bool(self.rhmc_adaptive_dual_step)
        max_disp = float(self.rhmc_adaptive_max_dual_displacement)
        min_scale = float(self.rhmc_adaptive_min_step_scale)

        def _explicit_fallback() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            G_inv, G_log_det = self._metric_quantities(z, mu, M, training=training)
            rho_ = self._leap_step_1(recon_x, x, z, rho, G_inv, G_log_det)
            z_new = self._leap_step_2(recon_x, x, z, rho_, G_inv, G_log_det)
            recon_new = self.decoder(z_new)["reconstruction"]
            G_inv_new, G_log_det_new = self._metric_quantities(z_new, mu, M, training=training)
            rho_new = self._leap_step_3(recon_new, x, z_new, rho_, G_inv_new, G_log_det_new)
            return z_new, rho_new, G_inv_new, G_log_det_new

        # (A) Implicit half-step for momentum via fixed-point iterations
        rho_half = rho
        for _ in range(steps):
            grad_z, _, _, _ = self._grad_z_hamiltonian(
                None, x, z, rho_half, mu, M, training=training
            )
            if not torch.isfinite(grad_z).all():
                return _explicit_fallback()
            rho_update = rho - 0.5 * eps * grad_z
            rho_half = (1.0 - damping) * rho_half + damping * rho_update

        # (B) Implicit full-step for position via fixed-point iterations
        z_new = z
        eps_scale = 1.0
        if adaptive:
            # Predict movement at original eps
            G_inv_z, _ = self._metric_quantities(z, mu, M, training=training)
            v_init = torch.einsum("bij,bj->bi", G_inv_z, rho_half)
            # Rough displacement estimate: delta_z approx eps * G_inv * rho
            # For Standard metric, v can be huge.
            disp = torch.norm(v_init, dim=-1) * eps
            # scale eps down if disp > max_disp
            scale = torch.clamp(max_disp / (disp + 1e-9), min=min_scale, max=1.0)
            eps_scale = scale.min().item()
            eps = eps * eps_scale

        for _ in range(steps):
            G_inv_z, _ = self._metric_quantities(z, mu, M, training=training)
            G_inv_new, _ = self._metric_quantities(z_new, mu, M, training=training)
            v0 = torch.einsum("bij,bj->bi", G_inv_z, rho_half)
            v1 = torch.einsum("bij,bj->bi", G_inv_new, rho_half)
            z_update = z + 0.5 * eps * (v0 + v1)
            z_new = (1.0 - damping) * z_new + damping * z_update
            if not torch.isfinite(z_new).all():
                return _explicit_fallback()

        # (C) Final momentum half-step at z_new
        grad_z_new, z_req, G_inv_new, G_log_det_new = self._grad_z_hamiltonian(
            None, x, z_new, rho_half, mu, M, training=training
        )
        if not torch.isfinite(grad_z_new).all():
            return _explicit_fallback()
        rho_new = rho_half - 0.5 * eps * grad_z_new
        return z_req, rho_new, G_inv_new, G_log_det_new

    def forward(self, inputs, **kwargs):
        x = inputs["data"]

        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z0, eps0 = self._sample_gauss(mu, std)

        z = z0
        M: torch.Tensor | None = None

        if self.training:
            L = self.metric(x)["L"]
            M = L @ torch.transpose(L, 1, 2)
            self.M.append(M.detach().clone())
            self.centroids.append(mu.detach().clone())
            G_inv = self._compute_training_inverse_metric(z, mu, M)
        else:
            G = self.G(z)
            G_inv = self.G_inv(z)
            L = torch.linalg.cholesky(G)

        sign, logabsdet = torch.linalg.slogdet(G_inv)
        G_log_det = -logabsdet

        gamma = torch.randn_like(z0, device=x.device)
        rho = gamma / self.beta_zero_sqrt
        beta_sqrt_old = self.beta_zero_sqrt

        rho = (L @ rho.unsqueeze(-1)).squeeze(-1)

        recon_x = self.decoder(z)["reconstruction"]

        use_implicit = self.rhmc_integrator == "implicit"

        for k in range(self.n_lf):
            if use_implicit:
                z, rho, G_inv, G_log_det = self._generalized_leapfrog_implicit(
                    recon_x,
                    x,
                    z,
                    rho,
                    mu=mu,
                    M=M,
                    training=self.training,
                )
                recon_x = self.decoder(z)["reconstruction"]
            else:
                rho_ = self._leap_step_1(recon_x, x, z, rho, G_inv, G_log_det)
                z = self._leap_step_2(recon_x, x, z, rho_, G_inv, G_log_det)
                recon_x = self.decoder(z)["reconstruction"]

                if self.training:
                    G_inv = self._compute_training_inverse_metric(z, mu, M)
                else:
                    G = self.G(z)
                    G_inv = self.G_inv(z)

                sign, logabsdet = torch.linalg.slogdet(G_inv)
                G_log_det = -logabsdet

                rho__ = self._leap_step_3(recon_x, x, z, rho_, G_inv, G_log_det)
                rho = rho__

            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho
            beta_sqrt_old = beta_sqrt

        loss = self.loss_function(
            recon_x, x, z0, z, rho, eps0, gamma, mu, log_var, G_inv, G_log_det
        )

        from pythae.models.base.base_utils import ModelOutput

        output = ModelOutput(
            loss=loss,
            recon_x=recon_x,
            z=z,
            z0=z0,
            rho=rho,
            eps0=eps0,
            gamma=gamma,
            mu=mu,
            log_var=log_var,
            G_inv=G_inv,
            G_log_det=G_log_det,
        )
        return output

    def _update_metric(self):
        super()._update_metric()
        if self.attractor_metric == "mahalanobis" or self.attractor_use_det:
            self._update_attractor_precisions(self.M_tens)
        self._refresh_metric_hooks()

    @classmethod
    def load_from_folder(cls, dir_path):
        model = super().load_from_folder(dir_path)
        if isinstance(model, GeometryRHVAE):
            model._refresh_metric_hooks()
        return model
