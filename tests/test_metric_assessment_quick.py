from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.metric_assessment_suite import load_real_data_if_available


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODEL = REPO_ROOT / "outputs/pythae_rhvae_baseline/2026-02-09_15-04-06"
ANISO_MODEL = REPO_ROOT / "outputs/pythae_rhvae_baseline/2026-02-09_18-16-00"


def test_load_real_data_missing_file_returns_none(tmp_path: Path) -> None:
    missing_model_dir = tmp_path / "fake_model"
    missing_model_dir.mkdir(parents=True, exist_ok=True)
    data = load_real_data_if_available(missing_model_dir)
    assert data is None


@pytest.mark.integration
def test_metric_assessment_quick_run_creates_outputs(tmp_path: Path) -> None:
    if not BASELINE_MODEL.exists() or not ANISO_MODEL.exists():
        pytest.skip("Locked model pair not available locally.")

    out_base = tmp_path / "metric_assessment"
    cmd = [
        sys.executable,
        "scripts/metric_assessment_suite.py",
        "--baseline_model_path",
        str(BASELINE_MODEL),
        "--ours_model_path",
        str(ANISO_MODEL),
        "--protocols",
        "strict",
        "--sampling_seeds",
        "13",
        "29",
        "--allow_two_seed_prefilter",
        "--quick",
        "--run_quality",
        "--skip_fid",
        "--n_chains",
        "1",
        "--chain_length",
        "20",
        "--burn_in",
        "5",
        "--sampling_mcmc_steps",
        "2",
        "--coverage_samples",
        "20",
        "--quality_samples",
        "20",
        "--rescue_ring_starts",
        "1",
        "--rescue_gaussian_starts",
        "1",
        "--rescue_horizon",
        "10",
        "--output_dir",
        str(out_base),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    run_dirs = [p for p in out_base.iterdir() if p.is_dir()]
    assert run_dirs, "No timestamped output directory created."
    run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)

    expected = [
        run_dir / "raw_metrics.csv",
        run_dir / "summary_by_protocol.json",
        run_dir / "scorecard.csv",
        run_dir / "scorecard.md",
        run_dir / "ranking_summary.csv",
        run_dir / "ranking_summary.json",
        run_dir / "model_run_bilan.csv",
    ]
    for path in expected:
        assert path.exists(), f"Missing expected output file: {path}"

    with (run_dir / "raw_metrics.csv").open() as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
    required_columns = {
        "model_key",
        "protocol",
        "seed",
        "acceptance_mean",
        "ess_norm_min",
        "coverage_local",
        "rescue_rate",
        "median_steps_to_manifold",
        "plateau_fraction",
    }
    assert required_columns.issubset(fieldnames)
