from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_low_data_end_to_end_smoke(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed" / "rotmnist" / "v1"
    subset_dir = processed_dir / "subsets"
    processed_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)

    # Build tiny synthetic dataset with 10 classes.
    n_train = 100
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

    subset_idx = list(range(50))
    subset_payload = {
        "dataset_id": "rotmnist_v1",
        "split": "train",
        "n": 50,
        "seed": 13,
        "num_classes": 10,
        "indices": subset_idx,
        "class_counts": {str(i): 5 for i in range(10)},
    }
    (subset_dir / "N050_seed013.json").write_text(json.dumps(subset_payload), encoding="utf-8")

    output_root = tmp_path / "outputs"

    cmds = [
        [
            sys.executable,
            "scripts/train_vanilla_vae.py",
            "--processed_dir",
            str(processed_dir),
            "--n",
            "50",
            "--seed",
            "13",
            "--epochs",
            "1",
            "--output_root",
            str(output_root),
            "--model_id",
            "vanilla_vae",
        ],
        [
            sys.executable,
            "scripts/train_rhvae_tensor.py",
            "--mode",
            "standard",
            "--processed_dir",
            str(processed_dir),
            "--n",
            "50",
            "--seed",
            "13",
            "--epochs",
            "1",
            "--output_root",
            str(output_root),
            "--model_id",
            "rhvae_standard",
        ],
        [
            sys.executable,
            "scripts/train_rhvae_tensor.py",
            "--mode",
            "aniso",
            "--processed_dir",
            str(processed_dir),
            "--n",
            "50",
            "--seed",
            "13",
            "--epochs",
            "1",
            "--output_root",
            str(output_root),
            "--model_id",
            "aniso",
        ],
        [
            sys.executable,
            "scripts/train_ebm_conformal.py",
            "--processed_dir",
            str(processed_dir),
            "--n",
            "50",
            "--seed",
            "13",
            "--ae_epochs",
            "1",
            "--ebm_epochs",
            "1",
            "--output_root",
            str(output_root),
            "--model_id",
            "ebm_conformal",
        ],
    ]

    for cmd in cmds:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    out_dir = tmp_path / "results"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_low_data_benchmark.py",
            "--processed_dir",
            str(processed_dir),
            "--model_root",
            str(output_root),
            "--subset_ns",
            "50",
            "--subset_seeds",
            "13",
            "--output_dir",
            str(out_dir),
            "--n_gen_samples",
            "64",
            "--n_interp_pairs",
            "4",
            "--interp_steps",
            "8",
            "--bootstrap_samples",
            "50",
            "--quick",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert run_dirs, "Benchmark run directory was not created"
    run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)

    assert (run_dir / "summary_by_model_n.json").exists()
    assert (run_dir / "metrics_table.csv").exists()
    assert (run_dir / "stat_tests.json").exists()
    assert (run_dir / "figures").exists()
