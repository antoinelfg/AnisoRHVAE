from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from scripts.metric_assessment_suite import (
    DEFAULT_NUMERIC_ARGS,
    apply_profile_overrides,
    build_ranking_wandb_scalars,
    build_model_bilan_rows,
    build_sweep_objectives,
    collect_reference_visuals,
    parse_args,
    summarize_candidate_ranking,
)


def _make_args(**overrides):
    payload = {
        "profile": "full",
        "n_chains": DEFAULT_NUMERIC_ARGS["n_chains"],
        "chain_length": DEFAULT_NUMERIC_ARGS["chain_length"],
        "burn_in": DEFAULT_NUMERIC_ARGS["burn_in"],
        "sampling_mcmc_steps": DEFAULT_NUMERIC_ARGS["sampling_mcmc_steps"],
        "coverage_samples": DEFAULT_NUMERIC_ARGS["coverage_samples"],
        "rescue_ring_starts": DEFAULT_NUMERIC_ARGS["rescue_ring_starts"],
        "rescue_gaussian_starts": DEFAULT_NUMERIC_ARGS["rescue_gaussian_starts"],
        "rescue_horizon": DEFAULT_NUMERIC_ARGS["rescue_horizon"],
        "quality_samples": DEFAULT_NUMERIC_ARGS["quality_samples"],
        "fid_samples": DEFAULT_NUMERIC_ARGS["fid_samples"],
        "pilot_n_chains": DEFAULT_NUMERIC_ARGS["pilot_n_chains"],
        "pilot_chain_length": DEFAULT_NUMERIC_ARGS["pilot_chain_length"],
        "bootstrap_samples": DEFAULT_NUMERIC_ARGS["bootstrap_samples"],
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_apply_profile_overrides_short_changes_defaults_only():
    args = _make_args(profile="short")
    apply_profile_overrides(args)
    assert args.chain_length == 220
    assert args.n_chains == 2
    assert args.rescue_horizon == 55

    custom = _make_args(profile="short", chain_length=333)
    apply_profile_overrides(custom)
    assert custom.chain_length == 333


def test_collect_reference_visuals_uses_latest_epoch(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "training_curves.png").write_bytes(b"curves")
    (model_dir / "reconstructions_epoch_001.png").write_bytes(b"old")
    (model_dir / "reconstructions_epoch_010.png").write_bytes(b"new")
    (model_dir / "latent_space_epoch_010.png").write_bytes(b"latent")
    (model_dir / "metric_field_epoch_010.png").write_bytes(b"field")
    (model_dir / "metric_tissot_epoch_010.png").write_bytes(b"tissot")

    analysis_dir = model_dir / "analysis_results" / "2026-01-01_00-00-00"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "metric_surface_3d.png").write_bytes(b"surface")

    sampling_dir = model_dir / "sampling_diagnostics" / "2026-01-01_00-00-00"
    sampling_dir.mkdir(parents=True, exist_ok=True)
    (sampling_dir / "interpolation_paths.png").write_bytes(b"interp")

    out_dir = tmp_path / "out"
    copied = collect_reference_visuals("baseline", model_dir, out_dir)
    assert copied
    recon_copy = out_dir / "baseline" / "training" / "reconstructions.png"
    assert recon_copy.exists()
    assert recon_copy.read_bytes() == b"new"


def test_build_model_bilan_rows_collects_training_and_sampling(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [10.0, 8.0],
        "train_recon": [5.0, 4.5],
        "val_loss": [11.0, 9.0],
        "val_recon": [6.0, 5.5],
    }
    torch.save(history, model_dir / "training_history.pt")

    sampling_dir = model_dir / "sampling_diagnostics" / "2026-01-01_00-00-00"
    sampling_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rhmc": {"mean_acceptance": 0.75},
        "quality": {"volume": {"precision": 0.8, "recall": 0.7, "coverage": 0.6, "diversity": 1.2}},
        "fid": {"volume": {"fid": 42.0}},
    }
    (sampling_dir / "metrics_summary.json").write_text(json.dumps(payload))

    rows = build_model_bilan_rows({"baseline": model_dir})
    assert len(rows) == 1
    row = rows[0]
    assert row["train_recon_last"] == 4.5
    assert row["val_recon_best"] == 5.5
    assert row["sampling_rhmc_mean_acceptance"] == 0.75
    assert row["sampling_quality_volume_precision"] == 0.8
    assert row["sampling_fid_volume"] == 42.0


def test_summarize_candidate_ranking_uses_median_and_iqr() -> None:
    rows = [
        {
            "protocol": "strict",
            "model_key": "aniso",
            "acceptance_mean": 0.75,
            "dh_p95_abs": 0.8,
            "h_drift_slope_abs_mean": 0.008,
            "coverage_local": 0.60,
            "rescue_rate": 0.82,
            "median_steps_to_manifold": 25.0,
            "plateau_fraction": 0.12,
            "tangent_alignment_mean": 0.86,
            "rescue_directionality_mean": 0.66,
            "border_overshoot_index": -0.05,
        },
        {
            "protocol": "strict",
            "model_key": "aniso",
            "acceptance_mean": 0.78,
            "dh_p95_abs": 0.9,
            "h_drift_slope_abs_mean": 0.009,
            "coverage_local": 0.62,
            "rescue_rate": 0.83,
            "median_steps_to_manifold": 24.0,
            "plateau_fraction": 0.10,
            "tangent_alignment_mean": 0.88,
            "rescue_directionality_mean": 0.67,
            "border_overshoot_index": -0.01,
        },
        {
            "protocol": "strict",
            "model_key": "aniso",
            "acceptance_mean": 0.73,
            "dh_p95_abs": 0.7,
            "h_drift_slope_abs_mean": 0.007,
            "coverage_local": 0.59,
            "rescue_rate": 0.81,
            "median_steps_to_manifold": 26.0,
            "plateau_fraction": 0.11,
            "tangent_alignment_mean": 0.85,
            "rescue_directionality_mean": 0.65,
            "border_overshoot_index": -0.02,
        },
    ]
    rank = summarize_candidate_ranking(rows, protocol="strict", model_key="aniso")
    assert rank["hard_reject"] is False
    assert rank["primary_green_count"] >= 6
    assert rank["n_seeds"] == 3
    assert rank["metrics"]["coverage_local"]["iqr"] > 0.0


def test_summarize_candidate_ranking_rejects_red_stability() -> None:
    rows = [
        {
            "protocol": "strict",
            "model_key": "aniso",
            "acceptance_mean": 0.30,
            "dh_p95_abs": 0.8,
            "h_drift_slope_abs_mean": 0.008,
            "coverage_local": 0.70,
            "rescue_rate": 0.85,
            "median_steps_to_manifold": 20.0,
            "plateau_fraction": 0.10,
            "tangent_alignment_mean": 0.90,
            "rescue_directionality_mean": 0.70,
            "border_overshoot_index": -0.05,
        },
        {
            "protocol": "strict",
            "model_key": "aniso",
            "acceptance_mean": 0.40,
            "dh_p95_abs": 0.9,
            "h_drift_slope_abs_mean": 0.009,
            "coverage_local": 0.65,
            "rescue_rate": 0.82,
            "median_steps_to_manifold": 22.0,
            "plateau_fraction": 0.12,
            "tangent_alignment_mean": 0.88,
            "rescue_directionality_mean": 0.68,
            "border_overshoot_index": -0.03,
        },
    ]
    rank = summarize_candidate_ranking(rows, protocol="strict", model_key="aniso")
    assert rank["hard_reject"] is True
    assert "acceptance_mean" in rank["reject_reasons"]


def test_summarize_candidate_ranking_accepts_exact_like_high_acceptance() -> None:
    rows = [
        {
            "protocol": "strict",
            "model_key": "aniso",
            "acceptance_mean": 1.0,
            "dh_p95_abs": 0.002,
            "h_drift_slope_abs_mean": 0.09,
            "coverage_local": 0.0,
            "rescue_rate": 0.0,
            "median_steps_to_manifold": 5.0,
            "plateau_fraction": 0.0,
            "tangent_alignment_mean": 0.80,
            "rescue_directionality_mean": 0.99,
            "border_overshoot_index": float("nan"),
        },
        {
            "protocol": "strict",
            "model_key": "aniso",
            "acceptance_mean": 1.0,
            "dh_p95_abs": 0.001,
            "h_drift_slope_abs_mean": 0.08,
            "coverage_local": 0.0,
            "rescue_rate": 0.0,
            "median_steps_to_manifold": 4.0,
            "plateau_fraction": 0.0,
            "tangent_alignment_mean": 0.79,
            "rescue_directionality_mean": 1.0,
            "border_overshoot_index": float("nan"),
        },
    ]
    rank = summarize_candidate_ranking(rows, protocol="strict", model_key="aniso")
    assert rank["hard_reject"] is False
    assert "acceptance_mean" not in rank["reject_reasons"]
    assert "h_drift_slope_abs_mean" not in rank["reject_reasons"]


def test_build_sweep_objectives_penalizes_missing_or_nonfinite() -> None:
    payload = {"matched": {"aniso": {"metrics": {}, "statuses": {}}}}
    out = build_sweep_objectives(payload, protocol="matched", model_key="aniso")
    assert out["objective_metric_core4_v1"] == -5.0
    assert out["objective_full_kpi_v1"] <= -5.0


def test_build_sweep_objectives_and_ranking_scalars() -> None:
    rows = [
        {
            "protocol": "matched",
            "model_key": "aniso",
            "acceptance_mean": 0.75,
            "dh_p95_abs": 0.6,
            "h_drift_slope_abs_mean": 0.008,
            "coverage_local": 0.62,
            "rescue_rate": 0.81,
            "median_steps_to_manifold": 24.0,
            "plateau_fraction": 0.12,
            "tangent_alignment_mean": 0.86,
            "rescue_directionality_mean": 0.68,
            "border_overshoot_index": -0.02,
        },
        {
            "protocol": "matched",
            "model_key": "aniso",
            "acceptance_mean": 0.79,
            "dh_p95_abs": 0.7,
            "h_drift_slope_abs_mean": 0.009,
            "coverage_local": 0.60,
            "rescue_rate": 0.80,
            "median_steps_to_manifold": 23.0,
            "plateau_fraction": 0.11,
            "tangent_alignment_mean": 0.85,
            "rescue_directionality_mean": 0.67,
            "border_overshoot_index": -0.01,
        },
        {
            "protocol": "matched",
            "model_key": "aniso",
            "acceptance_mean": 0.77,
            "dh_p95_abs": 0.8,
            "h_drift_slope_abs_mean": 0.01,
            "coverage_local": 0.61,
            "rescue_rate": 0.82,
            "median_steps_to_manifold": 25.0,
            "plateau_fraction": 0.10,
            "tangent_alignment_mean": 0.87,
            "rescue_directionality_mean": 0.69,
            "border_overshoot_index": -0.03,
        },
    ]
    rank = summarize_candidate_ranking(rows, protocol="matched", model_key="aniso")
    ranking_payload = {"matched": {"aniso": rank}}
    objectives = build_sweep_objectives(ranking_payload, protocol="matched", model_key="aniso")
    assert objectives["objective_metric_core4_v1"] > 0.0
    assert objectives["core4_green_count"] >= 3.0

    scalars = build_ranking_wandb_scalars(ranking_payload, sweep_objectives=objectives)
    assert "ranking/matched/aniso/primary_green_count" in scalars
    assert "ranking/matched/aniso/hard_reject" in scalars
    assert "ranking/matched/aniso/iqr_noise_penalty" in scalars
    assert "sweep/objective_metric_core4_v1" in scalars
    assert "sweep/objective_full_kpi_v1" in scalars


def test_parse_args_accepts_strict_mass_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--strict_mass_mode",
            "dual",
            "--sampling_seeds",
            "13",
            "29",
            "47",
        ],
    )
    args = parse_args()
    assert args.strict_mass_mode == "dual"
    assert args.sampler_name == "volume_riemannian"


def test_parse_args_deprecated_strict_dual_alias_maps_to_mass_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--strict_use_dual_metric",
            "True",
            "--sampling_seeds",
            "13",
            "29",
            "47",
        ],
    )
    args = parse_args()
    assert args.strict_mass_mode == "dual"
