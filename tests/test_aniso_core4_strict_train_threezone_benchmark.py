from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml

import scripts.run_aniso_core4_strict_train_threezone_benchmark as benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_cfg(tmp_path: Path) -> Path:
    cfg = {
        "benchmark": {
            "subset_ns": [50, 100, 1000],
            "output_root": str(tmp_path / "benchmark_results"),
        },
        "training": {
            "base_config": str(tmp_path / "base.yaml"),
            "output_root": str(tmp_path / "trained_models"),
            "required_files": ["rhvae_model.pt", "rhvae_metric.pt"],
            "fixed_overrides": {
                "n_centroids": 50,
                "max_centroids": 50,
                "skip_analysis": True,
                "skip_sampling_diagnostics": True,
            },
        },
        "sampling": {
            "script": "scripts/quick/visualize_rhmc_three_zones_real_metric.py",
            "output_root": str(tmp_path / "sampling_results"),
            "common_args": {
                "sampler_name": "volume_riemannian",
                "use_dual_metric": "False",
                "atom_scale": 100.0,
                "volume_power": 2.0,
                "radial_prior_weight": 0.1,
                "steps": 500,
                "n_lf_inner": 10,
                "eps": 0.05,
                "adaptive_dual_step": True,
                "adaptive_max_dual_displacement": 0.05,
                "fp_steps": 15,
                "void_eigshape_mode": "none",
            },
        },
        "wandb": {
            "project": "anisorhvae-core4-strict-train-threezone",
            "group": "aniso_core4_strict_n_benchmark",
            "tags": "base_tag",
            "name_mode": "manual",
            "run_name_template": "aniso_core4_strict_{n_pad}_seed{seed_pad}",
            "run_id_template": "aniso_core4_strict_{n_pad}_seed{seed_pad}",
            "job_type": "three_zone_sampling",
        },
    }
    cfg_path = tmp_path / "benchmark_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "base.yaml").write_text("{}", encoding="utf-8")
    return cfg_path


def _touch_valid_training_run(parent: Path, stamp: str) -> Path:
    run_dir = parent / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rhvae_model.pt").write_text("ok", encoding="utf-8")
    (run_dir / "rhvae_metric.pt").write_text("ok", encoding="utf-8")
    return run_dir


def _touch_valid_sampling_run(parent: Path, stamp: str) -> Path:
    run_dir = parent / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "rhmc_three_zones_real_metric.png").write_text("ok", encoding="utf-8")
    return run_dir


def test_runner_builds_expected_commands_and_n_scaling(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path)
    captured: list[tuple[list[str], dict[str, str]]] = []

    def _fake_run(cmd: list[str], env: dict[str, str]) -> None:
        captured.append((list(cmd), dict(env)))
        cmd_join = " ".join(cmd)
        if "run_with_config.py" in cmd_join:
            output_token = next((x for x in cmd if x.startswith("output_dir=")), None)
            assert output_token is not None
            output_parent = Path(output_token.split("=", 1)[1])
            _touch_valid_training_run(output_parent, "2026-01-01_00-00-00")
            return
        if "visualize_rhmc_three_zones_real_metric.py" in cmd_join:
            out_idx = cmd.index("--output_dir")
            out_parent = Path(cmd[out_idx + 1])
            _touch_valid_sampling_run(out_parent, "2026-01-01_00-01-00")
            return
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(benchmark, "_run_command", _fake_run)

    summary_path = benchmark.run_benchmark(
        config_path=cfg_path,
        subset_ns=[50, 1000],
        seed=13,
        reuse_policy="always_retrain",
        python_bin="python",
        wandb_mode="online",
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert [r["status"] for r in rows] == ["trained", "trained"]

    train_cmds = [cmd for cmd, _ in captured if "run_with_config.py" in " ".join(cmd)]
    assert len(train_cmds) == 2
    assert any("num_sequences=63" in tok for tok in train_cmds[0])
    assert any("max_frames=50" in tok for tok in train_cmds[0])
    assert any("num_sequences=1250" in tok for tok in train_cmds[1])
    assert any("max_frames=1000" in tok for tok in train_cmds[1])
    assert any("n_centroids=50" in tok for tok in train_cmds[0])
    assert any("max_centroids=50" in tok for tok in train_cmds[0])

    sampling_cmds = [cmd for cmd, _ in captured if "visualize_rhmc_three_zones_real_metric.py" in " ".join(cmd)]
    assert len(sampling_cmds) == 2
    for cmd in sampling_cmds:
        cmd_join = " ".join(cmd)
        assert "--sampler_name volume_riemannian" in cmd_join
        assert "--use_dual_metric False" in cmd_join
        assert "--steps 500" in cmd_join
        assert "--n_lf_inner 10" in cmd_join
        assert "--eps 0.05" in cmd_join
        assert "--wandb_job_type three_zone_sampling" in cmd_join

    train_envs = [env for cmd, env in captured if "run_with_config.py" in " ".join(cmd)]
    assert len(train_envs) == 2
    assert all(env.get("WANDB_RESUME") == "allow" for env in train_envs)
    assert all("WANDB_RUN_ID" in env for env in train_envs)


def test_runner_reuse_latest_skips_training(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path)
    train_root = tmp_path / "trained_models"
    existing_parent = train_root / "N050" / "seed013"
    _touch_valid_training_run(existing_parent, "2026-01-01_00-00-00")

    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], env: dict[str, str]) -> None:
        del env
        captured.append(list(cmd))
        cmd_join = " ".join(cmd)
        if "run_with_config.py" in cmd_join:
            raise AssertionError("Training must be skipped in reuse_latest.")
        out_idx = cmd.index("--output_dir")
        out_parent = Path(cmd[out_idx + 1])
        _touch_valid_sampling_run(out_parent, "2026-01-01_00-01-00")

    monkeypatch.setattr(benchmark, "_run_command", _fake_run)

    summary_path = benchmark.run_benchmark(
        config_path=cfg_path,
        subset_ns=[50],
        seed=13,
        reuse_policy="reuse_latest",
        python_bin="python",
        wandb_mode=None,
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["status"] == "reused"
    assert payload["rows"][0]["commands"]["training"] is None
    assert any("visualize_rhmc_three_zones_real_metric.py" in " ".join(cmd) for cmd in captured)


@pytest.mark.integration
def test_cli_reuse_latest_creates_summary_without_training(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path)
    train_root = tmp_path / "trained_models"
    existing_parent = train_root / "N050" / "seed013"
    _touch_valid_training_run(existing_parent, "2026-01-01_00-00-00")

    fake_launcher = tmp_path / "fake_python.sh"
    fake_log = tmp_path / "fake_calls.log"
    fake_launcher.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "echo \"$@\" >> \"$FAKE_CALLS_LOG\"",
                "if [[ \"$1\" == *\"run_with_config.py\" ]]; then",
                "  echo \"unexpected training call\" >&2",
                "  exit 99",
                "fi",
                "if [[ \"$1\" == *\"visualize_rhmc_three_zones_real_metric.py\" ]]; then",
                "  OUT_DIR=\"\"",
                "  ARGS=(\"$@\")",
                "  for ((i=0; i<${#ARGS[@]}; i++)); do",
                "    if [[ \"${ARGS[$i]}\" == \"--output_dir\" ]]; then",
                "      OUT_DIR=\"${ARGS[$((i+1))]}\"",
                "      break",
                "    fi",
                "  done",
                "  RUN_DIR=\"${OUT_DIR}/2026-01-01_00-00-01\"",
                "  mkdir -p \"${RUN_DIR}\"",
                "  echo \"{}\" > \"${RUN_DIR}/summary.json\"",
                "  echo \"ok\" > \"${RUN_DIR}/rhmc_three_zones_real_metric.png\"",
                "fi",
            ]
        ),
        encoding="utf-8",
    )
    fake_launcher.chmod(0o755)

    env = dict(os.environ)
    env["FAKE_CALLS_LOG"] = str(fake_log)

    subprocess.run(
        [
            sys.executable,
            "scripts/run_aniso_core4_strict_train_threezone_benchmark.py",
            "--config",
            str(cfg_path),
            "--subset_ns",
            "50",
            "--seed",
            "13",
            "--reuse_policy",
            "reuse_latest",
            "--python_bin",
            str(fake_launcher),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    bench_root = tmp_path / "benchmark_results"
    run_dirs = [p for p in bench_root.iterdir() if p.is_dir()]
    assert run_dirs, "benchmark output directory was not created"
    latest = max(run_dirs, key=lambda p: p.stat().st_mtime)
    summary_path = latest / "benchmark_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["status"] == "reused"

    call_lines = fake_log.read_text(encoding="utf-8").splitlines()
    assert not any("run_with_config.py" in line for line in call_lines)
    assert any("visualize_rhmc_three_zones_real_metric.py" in line for line in call_lines)
