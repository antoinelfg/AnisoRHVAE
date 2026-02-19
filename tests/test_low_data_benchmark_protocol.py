from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.low_data_benchmark import (
    VolumePotentialSampler,
    VolumeSamplerConfig,
    compute_manifold_rescue_metrics,
    ebm_logdet_from_energy,
    latent_d_rmse,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _DummyAdapter:
    def __init__(self, latent_dim: int = 4):
        self.model_id = "dummy"
        self.latent_dim = latent_dim
        self.device = torch.device("cpu")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)[:, : self.latent_dim]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros(z.shape[0], 1, 28, 28)

    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, n_steps: int = 32) -> torch.Tensor:
        t = torch.linspace(0, 1, n_steps).unsqueeze(1)
        return z1.unsqueeze(0) + t * (z2.unsqueeze(0) - z1.unsqueeze(0))

    def logdet_ginv(self, z: torch.Tensor, with_grad: bool) -> torch.Tensor:
        zz = z if with_grad else z.detach()
        return -0.5 * torch.sum(zz * zz, dim=1)


def _sampler_cfg() -> VolumeSamplerConfig:
    return VolumeSamplerConfig(
        volume_power=1.0,
        beta_zero=8.0,
        mcmc_steps=6,
        n_lf=3,
        eps_lf=0.03,
        grad_clip_norm=25.0,
        logdet_clip=80.0,
        eps_backoff=0.5,
        max_backoff_trials=3,
        warmup_steps=2,
    )


def test_ebm_conformal_logdet_formula() -> None:
    e = torch.tensor([0.0, 1.5, -2.0], dtype=torch.float32)
    out = ebm_logdet_from_energy(e, latent_dim=16, beta=0.7)
    expected = 16.0 * (-0.7 * e)
    assert torch.allclose(out, expected)


def test_volume_sampler_returns_finite_with_bounded_acceptance() -> None:
    adapter = _DummyAdapter(latent_dim=4)
    sampler = VolumePotentialSampler(adapter=adapter, cfg=_sampler_cfg(), device=torch.device("cpu"))
    z, diag = sampler.sample(n_samples=32)
    assert torch.isfinite(z).all()
    acc = float(diag["acceptance_rate_mean"])
    assert 0.0 <= acc <= 1.0


def test_rescue_detector_expected_hit_steps() -> None:
    adapter = _DummyAdapter(latent_dim=4)
    cfg = _sampler_cfg()
    real_lat = torch.randn(24, 4) * 0.05

    # With no explicit void starts, the helper starts exactly at manifold center.
    out = compute_manifold_rescue_metrics(
        adapter=adapter,
        sampler_cfg=cfg,
        real_lat=real_lat,
        n_ring=0,
        n_gauss=0,
        horizon=8,
        seed=13,
        device=torch.device("cpu"),
    )
    assert out["rescue_rate"] == 1.0
    assert out["median_steps_to_rescue"] == 0.0


def test_latent_d_rmse_zero_on_exact_manifold_points() -> None:
    real = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    path = real.clone()
    assert latent_d_rmse(path, real) == 0.0


@pytest.mark.integration
def test_low_data_benchmark_protocol_quick_smoke(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed" / "rotmnist" / "v1"
    subset_dir = processed_dir / "subsets"
    processed_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)

    n_train = 120
    n_val = 40
    n_test = 40

    train_images = torch.rand(n_train, 1, 28, 28)
    train_labels = torch.tensor([i % 10 for i in range(n_train)], dtype=torch.long)
    val_images = torch.rand(n_val, 1, 28, 28)
    val_labels = torch.tensor([i % 10 for i in range(n_val)], dtype=torch.long)
    test_images = torch.rand(n_test, 1, 28, 28)
    test_labels = torch.tensor([i % 10 for i in range(n_test)], dtype=torch.long)

    torch.save({"images": train_images, "labels": train_labels}, processed_dir / "train.pt")
    torch.save({"images": val_images, "labels": val_labels}, processed_dir / "val.pt")
    torch.save({"images": test_images, "labels": test_labels}, processed_dir / "test.pt")
    (processed_dir / "metadata.json").write_text("{}", encoding="utf-8")

    subset_payload = {
        "dataset_id": "rotmnist_v1",
        "split": "train",
        "n": 50,
        "seed": 13,
        "num_classes": 10,
        "indices": list(range(50)),
        "class_counts": {str(i): 5 for i in range(10)},
    }
    (subset_dir / "N050_seed013.json").write_text(json.dumps(subset_payload), encoding="utf-8")

    model_root = tmp_path / "outputs"
    out_dir = tmp_path / "results"

    subprocess.run(
        [
            sys.executable,
            "scripts/low_data_benchmark.py",
            "--processed_dir",
            str(processed_dir),
            "--model_root",
            str(model_root),
            "--output_dir",
            str(out_dir),
            "--quick",
            "--epochs",
            "1",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert run_dirs
    run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)

    assert (run_dir / "raw_metrics.csv").exists()
    assert (run_dir / "summary_by_model_n.json").exists()
    assert (run_dir / "stat_tests.json").exists()
    assert (run_dir / "anisotropy_ablation.json").exists()
    assert (run_dir / "failure_volume_riemannian.json").exists()
    assert (run_dir / "prompts" / "handoff_prompt.txt").exists()
