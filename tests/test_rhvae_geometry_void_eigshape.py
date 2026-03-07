from __future__ import annotations

from types import SimpleNamespace

import torch

from src.models.rhvae_geometry import GeometryRHVAE


def _make_spd_batch(batch: int, dim: int, seed: int = 7) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    a = torch.randn(batch, dim, dim, generator=g)
    eye = torch.eye(dim).unsqueeze(0)
    return a @ a.transpose(-1, -2) + 0.5 * eye


def test_det_preserving_eigshape_preserves_logdet() -> None:
    mats = _make_spd_batch(batch=8, dim=4)
    out = GeometryRHVAE._det_preserving_eigshape(mats, power=-1.0, eig_floor=1e-8)

    ld_in = torch.linalg.slogdet(mats).logabsdet
    ld_out = torch.linalg.slogdet(out).logabsdet
    assert torch.allclose(ld_in, ld_out, atol=1e-5, rtol=1e-5)


def test_det_preserving_eigshape_outputs_spd_and_symmetric() -> None:
    mats = _make_spd_batch(batch=6, dim=3)
    out = GeometryRHVAE._det_preserving_eigshape(mats, power=-1.0, eig_floor=1e-8)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, out.transpose(-1, -2), atol=1e-7, rtol=1e-7)
    evals = torch.linalg.eigvalsh(out)
    assert torch.all(evals > 0)


def test_apply_void_eigshape_respects_alpha_gating() -> None:
    helper = SimpleNamespace(
        void_eigshape_mode="det_preserving_spectral",
        void_eigshape_alpha_min=0.9,
        void_eigshape_power=-1.0,
        void_eigshape_eig_floor=1e-8,
        _det_preserving_eigshape=GeometryRHVAE._det_preserving_eigshape,
    )
    void = _make_spd_batch(batch=5, dim=2)
    alpha = torch.tensor([0.2, 0.95, 1.0, 0.5, 0.91], dtype=void.dtype)
    out = GeometryRHVAE._apply_void_eigshape(helper, void, alpha)

    changed = torch.linalg.norm((out - void).reshape(void.shape[0], -1), dim=1) > 1e-8
    expected = alpha >= 0.9
    assert torch.equal(changed, expected)


def test_apply_void_eigshape_noop_mode_is_identity() -> None:
    helper = SimpleNamespace(
        void_eigshape_mode="none",
        void_eigshape_alpha_min=0.0,
        void_eigshape_power=-1.0,
        void_eigshape_eig_floor=1e-8,
        _det_preserving_eigshape=GeometryRHVAE._det_preserving_eigshape,
    )
    void = _make_spd_batch(batch=4, dim=3)
    alpha = torch.tensor([0.0, 0.5, 0.9, 1.0], dtype=void.dtype)
    out = GeometryRHVAE._apply_void_eigshape(helper, void, alpha)
    assert torch.allclose(out, void, atol=0.0, rtol=0.0)


def test_anisotropy_inversion_2d_s_minus_one() -> None:
    mat = torch.tensor([[[9.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    out = GeometryRHVAE._det_preserving_eigshape(mat, power=-1.0, eig_floor=1e-8)

    evals_in = torch.linalg.eigvalsh(mat)[0]
    evals_out = torch.linalg.eigvalsh(out)[0]
    ratio_in = evals_in.max() / evals_in.min()
    ratio_out = evals_out.max() / evals_out.min()

    assert torch.allclose(
        torch.linalg.slogdet(mat).logabsdet,
        torch.linalg.slogdet(out).logabsdet,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(ratio_in, ratio_out, atol=1e-6, rtol=1e-6)
    # Largest eigendirection is inverted when s = -1 (diag[9,1] -> diag[1,9]).
    assert torch.allclose(evals_out, torch.tensor([1.0, 9.0]), atol=1e-4, rtol=1e-4)

