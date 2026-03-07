from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_sync_assets_local_candidate_present(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate_model"
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "model.bin").write_bytes(b"ok")

    alias = tmp_path / "aliases" / "m1"
    model_registry = {
        "models": [
            {
                "model_id": "m1",
                "model_family": "toy",
                "alias_path": str(alias),
                "required_files": ["model.bin"],
                "local_candidates": [str(candidate)],
                "wandb_refs": [],
                "train_if_missing": False,
            }
        ]
    }
    data_registry = {"datasets": []}

    reg_m = tmp_path / "model_registry.yaml"
    reg_d = tmp_path / "data_registry.yaml"
    out_m = tmp_path / "model_manifest.json"
    out_d = tmp_path / "data_manifest.json"
    _write_yaml(reg_m, model_registry)
    _write_yaml(reg_d, data_registry)

    cmd = [
        sys.executable,
        "scripts/sync_assets.py",
        "--registry_models",
        str(reg_m),
        "--registry_data",
        str(reg_d),
        "--manifest_out",
        str(out_m),
        "--data_manifest_out",
        str(out_d),
        "--materialize_aliases",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    payload = json.loads(out_m.read_text(encoding="utf-8"))
    assert payload["models"][0]["status"] == "ready"
    assert alias.exists()


def test_sync_assets_missing_non_trainable_exits_nonzero(tmp_path: Path) -> None:
    alias = tmp_path / "aliases" / "m1"
    model_registry = {
        "models": [
            {
                "model_id": "m1",
                "model_family": "toy",
                "alias_path": str(alias),
                "required_files": ["model.bin"],
                "local_candidates": [],
                "wandb_refs": [],
                "train_if_missing": False,
            }
        ]
    }
    data_registry = {"datasets": []}

    reg_m = tmp_path / "model_registry.yaml"
    reg_d = tmp_path / "data_registry.yaml"
    out_m = tmp_path / "model_manifest.json"
    out_d = tmp_path / "data_manifest.json"
    _write_yaml(reg_m, model_registry)
    _write_yaml(reg_d, data_registry)

    cmd = [
        sys.executable,
        "scripts/sync_assets.py",
        "--registry_models",
        str(reg_m),
        "--registry_data",
        str(reg_d),
        "--manifest_out",
        str(out_m),
        "--data_manifest_out",
        str(out_d),
    ]
    proc = subprocess.run(cmd, check=False, cwd=REPO_ROOT)
    assert proc.returncode == 2


def test_sync_assets_missing_trainable_is_allowed(tmp_path: Path) -> None:
    alias = tmp_path / "aliases" / "m1"
    model_registry = {
        "models": [
            {
                "model_id": "m1",
                "model_family": "toy",
                "alias_path": str(alias),
                "required_files": ["model.bin"],
                "local_candidates": [],
                "wandb_refs": [],
                "train_if_missing": True,
            }
        ]
    }
    data_registry = {"datasets": []}

    reg_m = tmp_path / "model_registry.yaml"
    reg_d = tmp_path / "data_registry.yaml"
    out_m = tmp_path / "model_manifest.json"
    out_d = tmp_path / "data_manifest.json"
    _write_yaml(reg_m, model_registry)
    _write_yaml(reg_d, data_registry)

    cmd = [
        sys.executable,
        "scripts/sync_assets.py",
        "--registry_models",
        str(reg_m),
        "--registry_data",
        str(reg_d),
        "--manifest_out",
        str(out_m),
        "--data_manifest_out",
        str(out_d),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    payload = json.loads(out_m.read_text(encoding="utf-8"))
    assert payload["models"][0]["status"] == "missing_train_required"
