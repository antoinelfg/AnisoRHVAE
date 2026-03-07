#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t as student_t

DEFAULT_METRIC_SUMMARY = (
    "results/metric_assessment/2026-02-10_17-35-53/summary_by_protocol.json"
)
DEFAULT_LOWDATA_SUMMARY = (
    "results/low_data_benchmark/2026-02-12_17-56-30/2026-02-12_21-04-28/summary_by_model_n.json"
)
DEFAULT_THREE_ZONE_ROOT = "results/three_zone_overnight_huge"
DEFAULT_REPORT_DIR = "reports/interim_bilan_now"

SAMPLER_KPIS = [
    "acceptance_mean",
    "dh_p95_abs",
    "h_drift_slope_abs_mean",
    "coverage_local",
    "rescue_rate",
    "median_steps_to_manifold",
    "ess_norm_min",
    "iact_median",
]

QUALITY_KPIS_RAW = [
    "fid",
    "d_rmse",
    "geo_euc_ratio",
    "interp_smoothness",
    "aug_balanced_accuracy",
    "aug_macro_f1",
]

QUALITY_KPIS_FINAL = [
    "fid",
    "d_rmse",
    "geo_euc_ratio_error",
    "interp_smoothness",
    "aug_balanced_accuracy",
    "aug_macro_f1",
]

DIRECTION_BY_KPI: dict[str, str] = {
    "acceptance_mean": "higher",
    "dh_p95_abs": "lower",
    "h_drift_slope_abs_mean": "lower",
    "coverage_local": "higher",
    "rescue_rate": "higher",
    "median_steps_to_manifold": "lower",
    "ess_norm_min": "higher",
    "iact_median": "lower",
    "fid": "lower",
    "d_rmse": "lower",
    "geo_euc_ratio_error": "lower",
    "interp_smoothness": "lower",
    "aug_balanced_accuracy": "higher",
    "aug_macro_f1": "higher",
    "on_manifold_drift": "lower",
    "near_outside_convergence": "higher",
    "far_outside_rescue_success": "higher",
    "escape_rate_manifold_start": "lower",
}

COMPARABILITY_KEYS = [
    "sampler_name",
    "exact",
    "volume_power",
    "n_lf",
    "eps_lf",
    "momentum_persist",
    "fp_steps",
    "fp_damping",
]

MODEL_DISPLAY_ORDER = ["vanilla_vae", "rhvae_standard", "aniso", "ebm_conformal"]
PROTOCOL_ORDER = ["strict", "matched"]
N_ORDER = [50, 100, 500]

ZERO_REF_EPS = 1e-12
WINSOR_Q_LOW = 0.05
WINSOR_Q_HIGH = 0.95


@dataclass(frozen=True)
class SanityAnchor:
    label: str
    source: str
    protocol: str | None
    n_value: int | None
    model_key: str
    kpi: str
    approx: float
    tolerance: float


def _fmt_float(v: Any, nd: int = 4) -> str:
    try:
        x = float(v)
    except Exception:
        return "NaN"
    if not math.isfinite(x):
        return "NaN"
    return f"{x:.{nd}f}"


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
        return x
    except Exception:
        return float("nan")


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return -1


def _is_finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def _canonical_family(model_key: str) -> str:
    k = str(model_key).strip().lower()
    if k in {"baseline", "rhvae_standard"}:
        return "rhvae_classic"
    if k in {"aniso", "anisorhvae"}:
        return "aniso"
    if k == "ebm_conformal":
        return "ebm"
    if k == "vanilla_vae":
        return "vanilla"
    return k


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse JSON at {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"Expected top-level object at {path}, got {type(obj).__name__}")
    return obj


def _require_keys(obj: dict[str, Any], required: list[str], context: str) -> None:
    missing = [k for k in required if k not in obj]
    if missing:
        found = sorted(obj.keys())
        raise ValueError(f"{context} missing keys {missing}. Found keys: {found}")


def _extract_stats(metric_item: Any) -> dict[str, float]:
    if isinstance(metric_item, dict):
        mean = _safe_float(metric_item.get("mean"))
        std = _safe_float(metric_item.get("std"))
        n = _safe_float(metric_item.get("n"))
        src_ci_low = _safe_float(metric_item.get("ci95_low"))
        src_ci_high = _safe_float(metric_item.get("ci95_high"))
        return {
            "mean": mean,
            "std": std,
            "n": n,
            "source_ci95_low": src_ci_low,
            "source_ci95_high": src_ci_high,
        }
    value = _safe_float(metric_item)
    return {
        "mean": value,
        "std": float("nan"),
        "n": float("nan"),
        "source_ci95_low": float("nan"),
        "source_ci95_high": float("nan"),
    }


def _t_interval(mean: float, std: float, n: float) -> tuple[float, float]:
    if not (_is_finite(mean) and _is_finite(std) and _is_finite(n)):
        return float("nan"), float("nan")
    n_i = int(round(n))
    if n_i <= 1:
        return float("nan"), float("nan")
    if std < 0:
        return float("nan"), float("nan")
    se = std / math.sqrt(n_i)
    t_quant = float(student_t.ppf(0.975, df=n_i - 1))
    half = t_quant * se
    return mean - half, mean + half


def _normalize_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return False


def parse_metric_assessment(
    summary_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(summary_path)
    _require_keys(payload, ["metadata", "summary"], f"metric_assessment:{summary_path}")
    summary = payload["summary"]
    metadata = payload.get("metadata", {})
    if not isinstance(summary, dict):
        raise ValueError("metric_assessment summary must be an object")

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for protocol in PROTOCOL_ORDER:
        if protocol not in summary:
            raise ValueError(
                f"metric_assessment summary missing protocol '{protocol}'. Available: {list(summary.keys())}"
            )
        pnode = summary[protocol]
        if not isinstance(pnode, dict):
            raise ValueError(f"metric_assessment summary[{protocol}] must be an object")

        for source_model_key in ["baseline", "aniso"]:
            if source_model_key not in pnode:
                raise ValueError(
                    f"metric_assessment summary[{protocol}] missing model '{source_model_key}'. "
                    f"Available: {list(pnode.keys())}"
                )
            mnode = pnode[source_model_key]
            if not isinstance(mnode, dict):
                raise ValueError(
                    f"metric_assessment summary[{protocol}][{source_model_key}] must be an object"
                )
            metrics = mnode.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError(
                    f"metric_assessment summary[{protocol}][{source_model_key}].metrics must be an object"
                )

            for kpi in SAMPLER_KPIS:
                item = metrics.get(kpi)
                if item is None:
                    stats = {
                        "mean": float("nan"),
                        "std": float("nan"),
                        "n": float("nan"),
                        "source_ci95_low": float("nan"),
                        "source_ci95_high": float("nan"),
                    }
                    missing.append(
                        {
                            "source": "metric_assessment",
                            "dataset": "metric_assessment",
                            "protocol": protocol,
                            "n": None,
                            "source_model_key": source_model_key,
                            "kpi": kpi,
                            "reason": "missing_key",
                        }
                    )
                else:
                    stats = _extract_stats(item)

                ci_low, ci_high = _t_interval(stats["mean"], stats["std"], stats["n"])

                row = {
                    "source": "metric_assessment",
                    "dataset": "metric_assessment",
                    "protocol": protocol,
                    "n": np.nan,
                    "source_model_key": source_model_key,
                    "canonical_family": _canonical_family(source_model_key),
                    "kpi": kpi,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "n_seeds": stats["n"],
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "source_ci95_low": stats["source_ci95_low"],
                    "source_ci95_high": stats["source_ci95_high"],
                    "n_rows": _safe_float(mnode.get("n_rows")),
                }
                rows.append(row)

                if not _is_finite(stats["mean"]):
                    missing.append(
                        {
                            "source": "metric_assessment",
                            "dataset": "metric_assessment",
                            "protocol": protocol,
                            "n": None,
                            "source_model_key": source_model_key,
                            "kpi": kpi,
                            "reason": "nan_mean",
                        }
                    )

                if _is_finite(stats["n"]) and int(round(stats["n"])) <= 0:
                    missing.append(
                        {
                            "source": "metric_assessment",
                            "dataset": "metric_assessment",
                            "protocol": protocol,
                            "n": None,
                            "source_model_key": source_model_key,
                            "kpi": kpi,
                            "reason": "n_nonpositive",
                        }
                    )

    return pd.DataFrame(rows), payload, missing


def parse_lowdata_benchmark(
    summary_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(summary_path)
    _require_keys(payload, ["models", "subset_ns", "summary"], f"low_data:{summary_path}")

    summary = payload["summary"]
    models = payload["models"]
    subset_ns = payload["subset_ns"]

    if not isinstance(summary, dict):
        raise ValueError("low_data summary must be an object")
    if not isinstance(models, list):
        raise ValueError("low_data models must be a list")
    if not isinstance(subset_ns, list):
        raise ValueError("low_data subset_ns must be a list")

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for n_val in N_ORDER:
        n_key = str(n_val)
        if n_key not in summary:
            raise ValueError(
                f"low_data summary missing N='{n_key}'. Available N keys: {list(summary.keys())}"
            )
        n_node = summary[n_key]
        if not isinstance(n_node, dict):
            raise ValueError(f"low_data summary[{n_key}] must be an object")

        for source_model_key in MODEL_DISPLAY_ORDER:
            mnode = n_node.get(source_model_key)
            if mnode is None:
                mnode = {}
                missing.append(
                    {
                        "source": "low_data_benchmark",
                        "dataset": "low_data_benchmark",
                        "protocol": None,
                        "n": n_val,
                        "source_model_key": source_model_key,
                        "kpi": "<model_block>",
                        "reason": "missing_model_block",
                    }
                )
            if not isinstance(mnode, dict):
                raise ValueError(
                    f"low_data summary[{n_key}][{source_model_key}] must be an object if present"
                )

            for raw_kpi in QUALITY_KPIS_RAW:
                item = mnode.get(raw_kpi)
                if item is None:
                    stats = {
                        "mean": float("nan"),
                        "std": float("nan"),
                        "n": float("nan"),
                        "source_ci95_low": float("nan"),
                        "source_ci95_high": float("nan"),
                    }
                    missing.append(
                        {
                            "source": "low_data_benchmark",
                            "dataset": "low_data_benchmark",
                            "protocol": None,
                            "n": n_val,
                            "source_model_key": source_model_key,
                            "kpi": raw_kpi,
                            "reason": "missing_key",
                        }
                    )
                else:
                    stats = _extract_stats(item)

                kpi = raw_kpi
                mean_value = stats["mean"]
                if raw_kpi == "geo_euc_ratio":
                    kpi = "geo_euc_ratio_error"
                    mean_value = abs(mean_value - 1.0) if _is_finite(mean_value) else float("nan")

                ci_low, ci_high = _t_interval(mean_value, stats["std"], stats["n"])

                rows.append(
                    {
                        "source": "low_data_benchmark",
                        "dataset": "low_data_benchmark",
                        "protocol": None,
                        "n": float(n_val),
                        "source_model_key": source_model_key,
                        "canonical_family": _canonical_family(source_model_key),
                        "kpi": kpi,
                        "mean": mean_value,
                        "std": stats["std"],
                        "n_seeds": stats["n"],
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "source_ci95_low": stats["source_ci95_low"],
                        "source_ci95_high": stats["source_ci95_high"],
                    }
                )

                if not _is_finite(mean_value):
                    missing.append(
                        {
                            "source": "low_data_benchmark",
                            "dataset": "low_data_benchmark",
                            "protocol": None,
                            "n": n_val,
                            "source_model_key": source_model_key,
                            "kpi": kpi,
                            "reason": "nan_mean",
                        }
                    )

                if _is_finite(stats["n"]) and int(round(stats["n"])) <= 0:
                    missing.append(
                        {
                            "source": "low_data_benchmark",
                            "dataset": "low_data_benchmark",
                            "protocol": None,
                            "n": n_val,
                            "source_model_key": source_model_key,
                            "kpi": kpi,
                            "reason": "n_nonpositive",
                        }
                    )

    return pd.DataFrame(rows), payload, missing


def compute_delta_vs_reference(
    df: pd.DataFrame,
    ref_model_key: str,
    group_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()
    out["delta_pct_vs_classic"] = np.nan
    out["absolute_diff_vs_classic"] = np.nan
    out["delta_note"] = ""

    if out.empty:
        return out

    key_cols = group_cols + ["kpi"]
    ref = (
        out[out["source_model_key"] == ref_model_key][key_cols + ["mean"]]
        .rename(columns={"mean": "reference_mean"})
        .drop_duplicates(subset=key_cols)
    )

    merged = out.merge(ref, on=key_cols, how="left")

    abs_diffs: list[float] = []
    pct_diffs: list[float] = []
    notes: list[str] = []

    for _, row in merged.iterrows():
        mean_val = _safe_float(row.get("mean"))
        ref_val = _safe_float(row.get("reference_mean"))

        if not _is_finite(mean_val) or not _is_finite(ref_val):
            abs_diffs.append(float("nan"))
            pct_diffs.append(float("nan"))
            notes.append("missing_reference_or_value")
            continue

        abs_diff = mean_val - ref_val
        abs_diffs.append(abs_diff)

        if abs(ref_val) <= ZERO_REF_EPS:
            pct_diffs.append(float("nan"))
            notes.append("undefined_zero_reference")
        else:
            pct_diffs.append(100.0 * abs_diff / ref_val)
            notes.append("")

    out["absolute_diff_vs_classic"] = abs_diffs
    out["delta_pct_vs_classic"] = pct_diffs
    out["delta_note"] = notes
    return out


def compute_quality_ranks(df_quality: pd.DataFrame) -> pd.DataFrame:
    out = df_quality.copy()
    out["rank_per_kpi_n"] = np.nan

    for n_val in N_ORDER:
        for kpi in QUALITY_KPIS_FINAL:
            mask = (out["n"] == float(n_val)) & (out["kpi"] == kpi)
            sub = out.loc[mask, ["source_model_key", "mean"]].copy()
            sub = sub[sub["mean"].map(_is_finite)]
            if sub.empty:
                continue

            direction = DIRECTION_BY_KPI[kpi]
            values = sub["mean"].tolist()
            if direction == "higher":
                unique_sorted = sorted(set(values), reverse=True)
            else:
                unique_sorted = sorted(set(values))
            rank_map = {v: i + 1 for i, v in enumerate(unique_sorted)}

            for idx in out.index[mask]:
                val = _safe_float(out.at[idx, "mean"])
                if _is_finite(val):
                    out.at[idx, "rank_per_kpi_n"] = float(rank_map[val])

    return out


def _transform_for_aggregate(kpi: str, value: float) -> float:
    if not _is_finite(value):
        return float("nan")
    if kpi == "interp_smoothness":
        return math.log10(max(value, 0.0) + 1e-15)
    return value


def compute_robust_aggregate_scores(
    quality_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = quality_df.copy()
    work["value_transformed"] = [
        _transform_for_aggregate(kpi, _safe_float(val))
        for kpi, val in zip(work["kpi"], work["mean"], strict=False)
    ]
    work["value_winsor"] = np.nan
    work["kpi_score_01"] = np.nan

    for n_val in N_ORDER:
        for kpi in QUALITY_KPIS_FINAL:
            mask = (work["n"] == float(n_val)) & (work["kpi"] == kpi)
            vals = work.loc[mask, "value_transformed"].astype(float)
            finite_vals = vals[np.isfinite(vals)]
            if finite_vals.empty:
                continue

            if finite_vals.shape[0] == 1:
                q_low = q_high = float(finite_vals.iloc[0])
            else:
                q_low = float(np.nanquantile(finite_vals, WINSOR_Q_LOW))
                q_high = float(np.nanquantile(finite_vals, WINSOR_Q_HIGH))

            clipped = vals.copy()
            clipped = clipped.clip(lower=q_low, upper=q_high)
            work.loc[mask, "value_winsor"] = clipped

            finite_clipped = clipped[np.isfinite(clipped)]
            cmin = float(finite_clipped.min())
            cmax = float(finite_clipped.max())

            if abs(cmax - cmin) <= 1e-12:
                work.loc[mask & work["value_winsor"].map(_is_finite), "kpi_score_01"] = 0.5
                continue

            direction = DIRECTION_BY_KPI[kpi]
            if direction == "higher":
                scores = (clipped - cmin) / (cmax - cmin)
            else:
                scores = (cmax - clipped) / (cmax - cmin)
            work.loc[mask, "kpi_score_01"] = scores

    agg_rows: list[dict[str, Any]] = []
    for n_val in N_ORDER:
        for model_key in MODEL_DISPLAY_ORDER:
            mask = (work["n"] == float(n_val)) & (work["source_model_key"] == model_key)
            scores = work.loc[mask, "kpi_score_01"].astype(float)
            finite_scores = scores[np.isfinite(scores)]
            if finite_scores.empty:
                aggregate_score = float("nan")
                eff_count = 0
            else:
                aggregate_score = float(finite_scores.mean())
                eff_count = int(finite_scores.shape[0])
            agg_rows.append(
                {
                    "n": float(n_val),
                    "source_model_key": model_key,
                    "aggregate_score": aggregate_score,
                    "effective_kpi_count": eff_count,
                }
            )

    agg_df = pd.DataFrame(agg_rows)
    merged = work.merge(agg_df, on=["n", "source_model_key"], how="left")
    return merged, agg_df


def build_winner_summary(agg_df: pd.DataFrame) -> dict[str, Any]:
    best_per_n: dict[str, Any] = {}
    for n_val in N_ORDER:
        sub = agg_df[(agg_df["n"] == float(n_val)) & (agg_df["aggregate_score"].map(_is_finite))]
        if sub.empty:
            best_per_n[str(n_val)] = None
            continue
        sub = sub.sort_values(["aggregate_score", "source_model_key"], ascending=[False, True])
        row = sub.iloc[0]
        best_per_n[str(n_val)] = {
            "source_model_key": str(row["source_model_key"]),
            "aggregate_score": float(row["aggregate_score"]),
        }

    robust = (
        agg_df[agg_df["aggregate_score"].map(_is_finite)]
        .groupby("source_model_key")["aggregate_score"]
        .agg(mean_score="mean", std_score="std", n_regimes="count")
        .reset_index()
    )

    if robust.empty:
        most_robust = None
    else:
        robust = robust.sort_values(["mean_score", "source_model_key"], ascending=[False, True])
        r = robust.iloc[0]
        most_robust = {
            "source_model_key": str(r["source_model_key"]),
            "mean_aggregate_score": float(r["mean_score"]),
            "std_aggregate_score": float(r["std_score"]) if _is_finite(r["std_score"]) else float("nan"),
            "n_regimes": int(r["n_regimes"]),
        }

    return {
        "best_overall_per_n": best_per_n,
        "most_robust_across_n": most_robust,
    }


def detect_latest_completed_three_zone_run(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "root": str(root),
        "selected_run": None,
        "incomplete_runs": [],
        "kpis": {},
        "details": {},
    }

    if not root.exists() or not root.is_dir():
        result["details"] = {"reason": "root_missing_or_not_directory"}
        return result

    run_dirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name, reverse=True)

    for run_dir in run_dirs:
        reasons: list[str] = []
        sweep_csv = run_dir / "three_zone_sweep_summary.csv"
        best_json = run_dir / "best_config.json"

        sweep_df: pd.DataFrame | None = None
        best_cfg: dict[str, Any] | None = None

        if not sweep_csv.exists() or sweep_csv.stat().st_size == 0:
            reasons.append("missing_or_empty_three_zone_sweep_summary.csv")
        else:
            try:
                sweep_df = pd.read_csv(sweep_csv)
                if sweep_df.empty:
                    reasons.append("three_zone_sweep_summary.csv_empty")
            except Exception as exc:
                reasons.append(f"failed_parse_sweep_csv:{exc}")

        if not best_json.exists() or best_json.stat().st_size == 0:
            reasons.append("missing_or_empty_best_config.json")
        else:
            try:
                best_cfg_obj = json.loads(best_json.read_text(encoding="utf-8"))
                if not isinstance(best_cfg_obj, dict):
                    reasons.append("best_config_not_object")
                else:
                    best_cfg = best_cfg_obj
            except Exception as exc:
                reasons.append(f"failed_parse_best_config:{exc}")

        if reasons:
            result["incomplete_runs"].append({"run_dir": str(run_dir), "reasons": reasons})
            continue

        assert sweep_df is not None
        assert best_cfg is not None

        selected_row = None
        cfg_name = str(best_cfg.get("cfg_name", "")).strip()
        if cfg_name and "cfg_name" in sweep_df.columns:
            match = sweep_df[sweep_df["cfg_name"].astype(str) == cfg_name]
            if not match.empty:
                selected_row = match.iloc[0]
        if selected_row is None:
            selected_row = sweep_df.iloc[0]

        # Parse optional top-k summaries for far success / escape rates.
        topk_rank1_dirs = [d for d in run_dir.glob("topk_plots/rank01_*") if d.is_dir()]
        summary_files: list[Path] = []
        for rank_dir in topk_rank1_dirs:
            summary_files.extend(sorted(rank_dir.rglob("summary.json")))

        far_success_flags: list[float] = []
        manifold_escape_flags: list[float] = []

        for sf in summary_files:
            try:
                sobj = json.loads(sf.read_text(encoding="utf-8"))
            except Exception:
                continue
            trajectories = sobj.get("trajectories", [])
            if not isinstance(trajectories, list):
                continue

            for tr in trajectories:
                if not isinstance(tr, dict):
                    continue
                name = str(tr.get("name", ""))
                d_start = _safe_float(tr.get("distance_start"))
                d_end = _safe_float(tr.get("distance_end"))
                escaped = _normalize_bool(tr.get("escaped"))

                if name == "far_outside" and _is_finite(d_start) and _is_finite(d_end):
                    far_success_flags.append(1.0 if d_end < d_start else 0.0)

                if name.startswith("manifold_"):
                    manifold_escape_flags.append(1.0 if escaped else 0.0)

        far_success_rate = (
            float(np.mean(far_success_flags)) if len(far_success_flags) > 0 else float("nan")
        )
        escape_rate_manifold = (
            float(np.mean(manifold_escape_flags)) if len(manifold_escape_flags) > 0 else float("nan")
        )

        on_manifold_drift = _safe_float(selected_row.get("manifold_mean_end_mean"))
        near_delta_mean = _safe_float(selected_row.get("near_delta_mean"))
        near_convergence = -near_delta_mean if _is_finite(near_delta_mean) else float("nan")

        result["available"] = True
        result["selected_run"] = str(run_dir)
        result["kpis"] = {
            "on_manifold_drift": on_manifold_drift,
            "near_outside_convergence": near_convergence,
            "far_outside_rescue_success": far_success_rate,
            "escape_rate_manifold_start": escape_rate_manifold,
        }
        result["details"] = {
            "best_config": best_cfg,
            "selected_sweep_row": {k: _safe_float(v) if isinstance(v, (int, float, np.number)) else v for k, v in selected_row.to_dict().items()},
            "summary_files_used": [str(p) for p in summary_files],
            "n_far_success_samples": len(far_success_flags),
            "n_manifold_escape_samples": len(manifold_escape_flags),
        }
        return result

    if run_dirs:
        result["details"] = {"reason": "no_completed_run_detected"}
    else:
        result["details"] = {"reason": "no_run_dirs_found"}

    return result


def build_three_zone_table(three_zone_info: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not three_zone_info.get("available", False):
        return pd.DataFrame(rows)

    for kpi in [
        "on_manifold_drift",
        "near_outside_convergence",
        "far_outside_rescue_success",
        "escape_rate_manifold_start",
    ]:
        value = _safe_float(three_zone_info.get("kpis", {}).get(kpi))
        rows.append(
            {
                "source": "three_zone_overnight_huge",
                "dataset": "three_zone_overnight_huge",
                "protocol": "three_zone",
                "n": np.nan,
                "source_model_key": "best_config",
                "canonical_family": "aniso",
                "kpi": kpi,
                "mean": value,
                "std": np.nan,
                "n_seeds": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_comparability_status(metric_payload: dict[str, Any]) -> dict[str, Any]:
    summary = metric_payload.get("summary", {})
    metadata = metric_payload.get("metadata", {})
    sampling_seeds = metadata.get("sampling_seeds")
    expected_seed_count = len(sampling_seeds) if isinstance(sampling_seeds, list) else None

    by_protocol: dict[str, Any] = {}
    global_diffs: list[dict[str, Any]] = []

    for protocol in PROTOCOL_ORDER:
        diffs: list[str] = []
        pnode = summary.get(protocol, {})
        if not isinstance(pnode, dict):
            diffs.append("protocol_missing_or_invalid")
            by_protocol[protocol] = {"comparable": False, "differences": diffs}
            global_diffs.extend({"protocol": protocol, "difference": d} for d in diffs)
            continue

        baseline = pnode.get("baseline", {}) if isinstance(pnode.get("baseline"), dict) else {}
        aniso = pnode.get("aniso", {}) if isinstance(pnode.get("aniso"), dict) else {}
        bcfg = baseline.get("sampler_config", {}) if isinstance(baseline.get("sampler_config"), dict) else {}
        acfg = aniso.get("sampler_config", {}) if isinstance(aniso.get("sampler_config"), dict) else {}

        for key in COMPARABILITY_KEYS:
            bval = bcfg.get(key)
            aval = acfg.get(key)
            if bval != aval:
                diffs.append(f"sampler_config_mismatch:{key}:baseline={bval}|aniso={aval}")

        b_rows = _safe_int(baseline.get("n_rows"))
        a_rows = _safe_int(aniso.get("n_rows"))
        if b_rows >= 0 and a_rows >= 0 and b_rows != a_rows:
            diffs.append(f"n_rows_mismatch:baseline={b_rows}|aniso={a_rows}")

        if expected_seed_count is not None:
            if b_rows >= 0 and b_rows != expected_seed_count:
                diffs.append(
                    f"baseline_n_rows_vs_seed_count:baseline_n_rows={b_rows}|seed_count={expected_seed_count}"
                )
            if a_rows >= 0 and a_rows != expected_seed_count:
                diffs.append(
                    f"aniso_n_rows_vs_seed_count:aniso_n_rows={a_rows}|seed_count={expected_seed_count}"
                )

        comparable = len(diffs) == 0
        by_protocol[protocol] = {"comparable": comparable, "differences": diffs}
        global_diffs.extend({"protocol": protocol, "difference": d} for d in diffs)

    comparable_all = all(by_protocol.get(p, {}).get("comparable", False) for p in PROTOCOL_ORDER)
    return {
        "comparable": comparable_all,
        "by_protocol": by_protocol,
        "differences": global_diffs,
        "expected_seed_count": expected_seed_count,
    }


def _get_sampler_mean(
    sampler_df: pd.DataFrame,
    protocol: str,
    source_model_key: str,
    kpi: str,
) -> float:
    sub = sampler_df[
        (sampler_df["protocol"] == protocol)
        & (sampler_df["source_model_key"] == source_model_key)
        & (sampler_df["kpi"] == kpi)
    ]
    if sub.empty:
        return float("nan")
    return _safe_float(sub.iloc[0]["mean"])


def _get_quality_rank(
    quality_df: pd.DataFrame,
    n_val: int,
    model_key: str,
    kpi: str,
) -> float:
    sub = quality_df[
        (quality_df["n"] == float(n_val))
        & (quality_df["source_model_key"] == model_key)
        & (quality_df["kpi"] == kpi)
    ]
    if sub.empty:
        return float("nan")
    return _safe_float(sub.iloc[0]["rank_per_kpi_n"])


def _normalize_value(value: float, values: list[float], direction: str) -> float:
    finite_vals = [v for v in values if _is_finite(v)]
    if not _is_finite(value) or not finite_vals:
        return float("nan")
    vmin = min(finite_vals)
    vmax = max(finite_vals)
    if abs(vmax - vmin) <= 1e-12:
        return 0.5
    if direction == "higher":
        return (value - vmin) / (vmax - vmin)
    return (vmax - value) / (vmax - vmin)


def compute_rule_logic(
    sampler_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    quality_agg_df: pd.DataFrame,
    three_zone_info: dict[str, Any],
    comparability_status: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_protocol = "strict" if "strict" in set(sampler_df["protocol"].dropna()) else "matched"

    baseline_accept = _get_sampler_mean(sampler_df, primary_protocol, "baseline", "acceptance_mean")
    baseline_rescue = _get_sampler_mean(sampler_df, primary_protocol, "baseline", "rescue_rate")

    aniso_rescue = _get_sampler_mean(sampler_df, primary_protocol, "aniso", "rescue_rate")
    baseline_steps = _get_sampler_mean(sampler_df, primary_protocol, "baseline", "median_steps_to_manifold")
    aniso_steps = _get_sampler_mean(sampler_df, primary_protocol, "aniso", "median_steps_to_manifold")
    baseline_dh = _get_sampler_mean(sampler_df, primary_protocol, "baseline", "dh_p95_abs")
    aniso_dh = _get_sampler_mean(sampler_df, primary_protocol, "aniso", "dh_p95_abs")
    baseline_drift = _get_sampler_mean(
        sampler_df, primary_protocol, "baseline", "h_drift_slope_abs_mean"
    )
    aniso_drift = _get_sampler_mean(sampler_df, primary_protocol, "aniso", "h_drift_slope_abs_mean")

    far_rescue_success = _safe_float(
        three_zone_info.get("kpis", {}).get("far_outside_rescue_success")
    )

    rule_a_active = (
        _is_finite(baseline_accept)
        and baseline_accept >= 0.98
        and (
            (_is_finite(baseline_rescue) and baseline_rescue <= 0.10)
            or (_is_finite(far_rescue_success) and far_rescue_success <= 0.50)
        )
    )

    rescue_gain = aniso_rescue - baseline_rescue if (_is_finite(aniso_rescue) and _is_finite(baseline_rescue)) else float("nan")
    steps_ratio = aniso_steps / baseline_steps if (_is_finite(aniso_steps) and _is_finite(baseline_steps) and abs(baseline_steps) > ZERO_REF_EPS) else float("nan")
    dh_rel_change = abs(aniso_dh - baseline_dh) / abs(baseline_dh) if (_is_finite(aniso_dh) and _is_finite(baseline_dh) and abs(baseline_dh) > ZERO_REF_EPS) else float("nan")
    drift_rel_change = abs(aniso_drift - baseline_drift) / abs(baseline_drift) if (_is_finite(aniso_drift) and _is_finite(baseline_drift) and abs(baseline_drift) > ZERO_REF_EPS) else float("nan")

    rule_b_active = (
        _is_finite(rescue_gain)
        and rescue_gain >= 0.15
        and _is_finite(steps_ratio)
        and steps_ratio <= 0.80
        and _is_finite(dh_rel_change)
        and dh_rel_change <= 0.20
        and _is_finite(drift_rel_change)
        and drift_rel_change <= 0.20
    )

    vanilla_dominance_count = 0
    for n_val in N_ORDER:
        rank_fid = _get_quality_rank(quality_df, n_val, "vanilla_vae", "fid")
        rank_rmse = _get_quality_rank(quality_df, n_val, "vanilla_vae", "d_rmse")
        if _is_finite(rank_fid) and _is_finite(rank_rmse) and int(rank_fid) == 1 and int(rank_rmse) == 1:
            vanilla_dominance_count += 1

    manifold_rescue_dom = _is_finite(aniso_rescue) and _is_finite(baseline_rescue) and (aniso_rescue > baseline_rescue)
    rule_c_active = vanilla_dominance_count >= 2 and manifold_rescue_dom

    rules = {
        "rule_a": {
            "active": bool(rule_a_active),
            "message": "integration accepted but wrong direction / weak pull-to-manifold",
            "evidence": [
                f"baseline acceptance_mean ({primary_protocol}) = {_fmt_float(baseline_accept, 4)}",
                f"baseline rescue_rate ({primary_protocol}) = {_fmt_float(baseline_rescue, 4)}",
                f"far_outside_rescue_success (three-zone) = {_fmt_float(far_rescue_success, 4)}",
            ],
        },
        "rule_b": {
            "active": bool(rule_b_active),
            "message": "metric shaping helps recovery without major integrator instability",
            "evidence": [
                f"rescue_rate gain (aniso-baseline, {primary_protocol}) = {_fmt_float(rescue_gain, 4)}",
                f"median_steps ratio aniso/baseline ({primary_protocol}) = {_fmt_float(steps_ratio, 4)}",
                f"relative changes: dh_p95={_fmt_float(dh_rel_change,4)}, h_drift={_fmt_float(drift_rel_change,4)}",
            ],
        },
        "rule_c": {
            "active": bool(rule_c_active),
            "message": "trade-off between geometric structure and reconstruction fidelity",
            "evidence": [
                f"vanilla dominates fid+d_rmse in {vanilla_dominance_count}/3 N-regimes",
                f"aniso rescue_rate vs baseline ({primary_protocol}) = {_fmt_float(aniso_rescue,4)} vs {_fmt_float(baseline_rescue,4)}",
                "asymmetry note: rescue geometry KPIs are not available for all low-data models",
            ],
        },
    }

    # Decision hierarchy
    recommendation = "Parallel workstream"
    rec_reason = []

    if not comparability_status.get("comparable", False):
        recommendation = "Parallel workstream"
        rec_reason.append("protocol comparability gate failed")
    else:
        active_map = {
            "rule_a": "Improve sampler first",
            "rule_b": "Improve metric first",
            "rule_c": "Parallel workstream",
        }
        active_rules = [r for r in ["rule_a", "rule_b", "rule_c"] if rules[r]["active"]]
        active_recs = [active_map[r] for r in active_rules]

        if len(active_recs) == 1:
            recommendation = active_recs[0]
            rec_reason.append(f"single active rule: {active_rules[0]}")
        elif len(active_recs) > 1:
            if len(set(active_recs)) == 1:
                recommendation = active_recs[0]
                rec_reason.append(f"multiple active rules with same direction: {active_rules}")
            else:
                recommendation = "Parallel workstream"
                rec_reason.append(f"conflicting active rules: {active_rules}")
        else:
            # Fallback deficit comparison
            baseline_scores: list[float] = []
            strict_rows = sampler_df[sampler_df["protocol"] == primary_protocol]
            for kpi in SAMPLER_KPIS:
                sb = strict_rows[
                    (strict_rows["kpi"] == kpi)
                    & (strict_rows["source_model_key"] == "baseline")
                ]
                sa = strict_rows[
                    (strict_rows["kpi"] == kpi)
                    & (strict_rows["source_model_key"] == "aniso")
                ]
                if sb.empty or sa.empty:
                    continue
                b_val = _safe_float(sb.iloc[0]["mean"])
                a_val = _safe_float(sa.iloc[0]["mean"])
                score = _normalize_value(b_val, [b_val, a_val], DIRECTION_BY_KPI[kpi])
                if _is_finite(score):
                    baseline_scores.append(score)

            sampler_badness = 1.0 - float(np.mean(baseline_scores)) if baseline_scores else 0.5

            classic_quality = quality_agg_df[
                (quality_agg_df["source_model_key"] == "rhvae_standard")
                & (quality_agg_df["aggregate_score"].map(_is_finite))
            ]["aggregate_score"].astype(float)
            if classic_quality.empty:
                quality_badness = 0.5
            else:
                quality_badness = 1.0 - float(classic_quality.mean())

            if sampler_badness - quality_badness > 0.05:
                recommendation = "Improve sampler first"
            elif quality_badness - sampler_badness > 0.05:
                recommendation = "Improve metric first"
            else:
                recommendation = "Parallel workstream"
            rec_reason.append(
                f"fallback deficits sampler={_fmt_float(sampler_badness,4)} quality={_fmt_float(quality_badness,4)}"
            )

    diagnosis = {
        "comparability_gate": comparability_status,
        "rules": rules,
        "final_recommendation": recommendation,
        "decision_trace": rec_reason,
        "primary_protocol": primary_protocol,
    }

    return rules, diagnosis


def compute_seed_checks(
    metric_payload: dict[str, Any],
    sampler_df: pd.DataFrame,
    lowdata_payload: dict[str, Any],
    quality_df: pd.DataFrame,
) -> dict[str, Any]:
    seed_checks: dict[str, Any] = {}

    metric_seeds = metric_payload.get("metadata", {}).get("sampling_seeds")
    metric_seed_count = len(metric_seeds) if isinstance(metric_seeds, list) else None
    metric_rows_check: list[dict[str, Any]] = []
    for protocol in PROTOCOL_ORDER:
        for model_key in ["baseline", "aniso"]:
            sub = sampler_df[
                (sampler_df["protocol"] == protocol)
                & (sampler_df["source_model_key"] == model_key)
            ]
            n_rows = sub["n_rows"].dropna().unique()
            val = float(n_rows[0]) if len(n_rows) > 0 else float("nan")
            ok = (metric_seed_count is not None) and _is_finite(val) and int(round(val)) == metric_seed_count
            metric_rows_check.append(
                {
                    "protocol": protocol,
                    "source_model_key": model_key,
                    "n_rows": val,
                    "expected_seed_count": metric_seed_count,
                    "ok": bool(ok),
                }
            )

    lowdata_seeds = lowdata_payload.get("subset_seeds")
    lowdata_seed_count = len(lowdata_seeds) if isinstance(lowdata_seeds, list) else None
    low_rows_check: list[dict[str, Any]] = []

    for n_val in N_ORDER:
        for model_key in MODEL_DISPLAY_ORDER:
            sub = quality_df[
                (quality_df["n"] == float(n_val))
                & (quality_df["source_model_key"] == model_key)
            ]
            for kpi in QUALITY_KPIS_FINAL:
                s2 = sub[sub["kpi"] == kpi]
                if s2.empty:
                    continue
                n_metric = _safe_float(s2.iloc[0]["n_seeds"])
                if _is_finite(n_metric) and int(round(n_metric)) == 0:
                    ok = True
                else:
                    ok = (
                        (lowdata_seed_count is not None)
                        and _is_finite(n_metric)
                        and int(round(n_metric)) == lowdata_seed_count
                    )
                low_rows_check.append(
                    {
                        "n": n_val,
                        "source_model_key": model_key,
                        "kpi": kpi,
                        "n_seeds": n_metric,
                        "expected_seed_count": lowdata_seed_count,
                        "ok": bool(ok),
                    }
                )

    seed_checks["metric_assessment"] = {
        "sampling_seeds": metric_seeds,
        "expected_seed_count": metric_seed_count,
        "checks": metric_rows_check,
    }
    seed_checks["low_data_benchmark"] = {
        "subset_seeds": lowdata_seeds,
        "expected_seed_count": lowdata_seed_count,
        "checks": low_rows_check,
    }
    return seed_checks


def build_sanity_checks(
    sampler_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    anchors = [
        # Metric assessment strict
        SanityAnchor(
            label="metric_strict_baseline_acceptance",
            source="metric_assessment",
            protocol="strict",
            n_value=None,
            model_key="baseline",
            kpi="acceptance_mean",
            approx=1.0,
            tolerance=0.01,
        ),
        SanityAnchor(
            label="metric_strict_baseline_rescue",
            source="metric_assessment",
            protocol="strict",
            n_value=None,
            model_key="baseline",
            kpi="rescue_rate",
            approx=0.0,
            tolerance=0.10,
        ),
        SanityAnchor(
            label="metric_strict_baseline_steps",
            source="metric_assessment",
            protocol="strict",
            n_value=None,
            model_key="baseline",
            kpi="median_steps_to_manifold",
            approx=21.0,
            tolerance=5.0,
        ),
        SanityAnchor(
            label="metric_strict_aniso_acceptance",
            source="metric_assessment",
            protocol="strict",
            n_value=None,
            model_key="aniso",
            kpi="acceptance_mean",
            approx=0.997,
            tolerance=0.02,
        ),
        SanityAnchor(
            label="metric_strict_aniso_rescue",
            source="metric_assessment",
            protocol="strict",
            n_value=None,
            model_key="aniso",
            kpi="rescue_rate",
            approx=0.375,
            tolerance=0.15,
        ),
        SanityAnchor(
            label="metric_strict_aniso_steps",
            source="metric_assessment",
            protocol="strict",
            n_value=None,
            model_key="aniso",
            kpi="median_steps_to_manifold",
            approx=10.25,
            tolerance=8.0,
        ),
        # Low-data N=50 FID
        SanityAnchor(
            label="lowdata_n50_vanilla_fid",
            source="low_data_benchmark",
            protocol=None,
            n_value=50,
            model_key="vanilla_vae",
            kpi="fid",
            approx=260.9,
            tolerance=40.0,
        ),
        SanityAnchor(
            label="lowdata_n50_rhvae_standard_fid",
            source="low_data_benchmark",
            protocol=None,
            n_value=50,
            model_key="rhvae_standard",
            kpi="fid",
            approx=358.2,
            tolerance=50.0,
        ),
        SanityAnchor(
            label="lowdata_n50_aniso_fid",
            source="low_data_benchmark",
            protocol=None,
            n_value=50,
            model_key="aniso",
            kpi="fid",
            approx=311.4,
            tolerance=60.0,
        ),
        SanityAnchor(
            label="lowdata_n50_ebm_fid",
            source="low_data_benchmark",
            protocol=None,
            n_value=50,
            model_key="ebm_conformal",
            kpi="fid",
            approx=320.5,
            tolerance=50.0,
        ),
    ]

    rows: list[dict[str, Any]] = []

    for a in anchors:
        actual = float("nan")
        if a.source == "metric_assessment":
            sub = sampler_df[
                (sampler_df["protocol"] == a.protocol)
                & (sampler_df["source_model_key"] == a.model_key)
                & (sampler_df["kpi"] == a.kpi)
            ]
            if not sub.empty:
                actual = _safe_float(sub.iloc[0]["mean"])
        elif a.source == "low_data_benchmark":
            sub = quality_df[
                (quality_df["n"] == float(a.n_value))
                & (quality_df["source_model_key"] == a.model_key)
                & (quality_df["kpi"] == a.kpi)
            ]
            if not sub.empty:
                actual = _safe_float(sub.iloc[0]["mean"])

        abs_diff = abs(actual - a.approx) if _is_finite(actual) else float("nan")
        within = _is_finite(abs_diff) and abs_diff <= a.tolerance

        rows.append(
            {
                "label": a.label,
                "source": a.source,
                "protocol": a.protocol,
                "n": a.n_value,
                "source_model_key": a.model_key,
                "kpi": a.kpi,
                "actual": actual,
                "expected_approx": a.approx,
                "abs_diff": abs_diff,
                "tolerance": a.tolerance,
                "within_tolerance": bool(within),
            }
        )

    return rows


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def build_table1_markdown(sampler_df: pd.DataFrame) -> str:
    lines: list[list[str]] = []
    for protocol in PROTOCOL_ORDER:
        for kpi in SAMPLER_KPIS:
            sb = sampler_df[
                (sampler_df["protocol"] == protocol)
                & (sampler_df["source_model_key"] == "baseline")
                & (sampler_df["kpi"] == kpi)
            ]
            sa = sampler_df[
                (sampler_df["protocol"] == protocol)
                & (sampler_df["source_model_key"] == "aniso")
                & (sampler_df["kpi"] == kpi)
            ]
            b_mean = _safe_float(sb.iloc[0]["mean"]) if not sb.empty else float("nan")
            a_mean = _safe_float(sa.iloc[0]["mean"]) if not sa.empty else float("nan")
            abs_diff = a_mean - b_mean if (_is_finite(a_mean) and _is_finite(b_mean)) else float("nan")
            delta_pct = _safe_float(sa.iloc[0]["delta_pct_vs_classic"]) if not sa.empty else float("nan")
            lines.append(
                [
                    protocol,
                    kpi,
                    _fmt_float(b_mean, 4),
                    _fmt_float(a_mean, 4),
                    _fmt_float(abs_diff, 4),
                    _fmt_float(delta_pct, 2),
                ]
            )
    return _md_table(
        [
            "Protocol",
            "KPI",
            "Baseline",
            "Aniso",
            "Absolute diff",
            "Delta % vs baseline",
        ],
        lines,
    )


def build_table2_markdown(quality_df: pd.DataFrame) -> str:
    lines: list[list[str]] = []
    for n_val in N_ORDER:
        for kpi in QUALITY_KPIS_FINAL:
            model_vals: dict[str, float] = {}
            for m in MODEL_DISPLAY_ORDER:
                sub = quality_df[
                    (quality_df["n"] == float(n_val))
                    & (quality_df["kpi"] == kpi)
                    & (quality_df["source_model_key"] == m)
                ]
                model_vals[m] = _safe_float(sub.iloc[0]["mean"]) if not sub.empty else float("nan")

            finite_items = [(m, v) for m, v in model_vals.items() if _is_finite(v)]
            if not finite_items:
                best = "NaN"
            else:
                if DIRECTION_BY_KPI[kpi] == "higher":
                    best = sorted(finite_items, key=lambda x: (-x[1], x[0]))[0][0]
                else:
                    best = sorted(finite_items, key=lambda x: (x[1], x[0]))[0][0]

            lines.append(
                [
                    str(n_val),
                    kpi,
                    _fmt_float(model_vals["vanilla_vae"], 4),
                    _fmt_float(model_vals["rhvae_standard"], 4),
                    _fmt_float(model_vals["aniso"], 4),
                    _fmt_float(model_vals["ebm_conformal"], 4),
                    best,
                ]
            )

    return _md_table(
        [
            "N",
            "KPI",
            "vanilla_vae",
            "rhvae_standard",
            "aniso",
            "ebm_conformal",
            "Best",
        ],
        lines,
    )


def build_missing_markdown(missing_df: pd.DataFrame, max_rows: int = 20) -> str:
    if missing_df.empty:
        return "No missing/NaN KPI detected."
    sub = missing_df.copy()
    sub = sub.sort_values(
        ["source", "protocol", "n", "source_model_key", "kpi", "reason"],
        ascending=[True, True, True, True, True, True],
    )
    lines: list[list[str]] = []
    for _, row in sub.head(max_rows).iterrows():
        lines.append(
            [
                str(row.get("source", "")),
                str(row.get("protocol", "")) if pd.notna(row.get("protocol")) else "",
                str(int(row.get("n"))) if _is_finite(row.get("n")) else "",
                str(row.get("source_model_key", "")),
                str(row.get("kpi", "")),
                str(row.get("reason", "")),
            ]
        )
    text = _md_table(["Source", "Protocol", "N", "Model", "KPI", "Reason"], lines)
    if len(sub) > max_rows:
        text += f"\n\n_Only first {max_rows} rows shown out of {len(sub)}._"
    return text


def plot_kpi_bars(
    sampler_df: pd.DataFrame,
    quality_agg_df: pd.DataFrame,
    output_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)

    # Left: direction-normalized sampler KPIs for strict baseline/aniso
    norm_rows: list[dict[str, Any]] = []
    strict = sampler_df[sampler_df["protocol"] == "strict"]
    for kpi in SAMPLER_KPIS:
        sb = strict[(strict["kpi"] == kpi) & (strict["source_model_key"] == "baseline")]
        sa = strict[(strict["kpi"] == kpi) & (strict["source_model_key"] == "aniso")]
        if sb.empty or sa.empty:
            continue
        b = _safe_float(sb.iloc[0]["mean"])
        a = _safe_float(sa.iloc[0]["mean"])
        direction = DIRECTION_BY_KPI[kpi]
        b_norm = _normalize_value(b, [b, a], direction)
        a_norm = _normalize_value(a, [b, a], direction)
        norm_rows.append({"kpi": kpi, "model": "baseline", "score": b_norm})
        norm_rows.append({"kpi": kpi, "model": "aniso", "score": a_norm})

    if norm_rows:
        ndf = pd.DataFrame(norm_rows)
        sns.barplot(data=ndf, x="kpi", y="score", hue="model", ax=axes[0])
        axes[0].set_ylim(0, 1.05)
        axes[0].set_xlabel("Sampler/geometry KPI (strict)")
        axes[0].set_ylabel("Direction-normalized score [0,1]")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].set_title("Sampler/Geometry Stability")
    else:
        axes[0].text(0.5, 0.5, "No sampler data", ha="center", va="center")
        axes[0].axis("off")

    # Right: aggregate quality score across N
    q = quality_agg_df.copy()
    q = q[q["aggregate_score"].map(_is_finite)]
    if not q.empty:
        q["N"] = q["n"].astype(int).astype(str)
        sns.barplot(data=q, x="N", y="aggregate_score", hue="source_model_key", ax=axes[1])
        axes[1].set_ylim(0, 1.05)
        axes[1].set_xlabel("N")
        axes[1].set_ylabel("Aggregate score [0,1]")
        axes[1].set_title("Model Quality Across Data Regimes")
    else:
        axes[1].text(0.5, 0.5, "No quality aggregate data", ha="center", va="center")
        axes[1].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_rank_heatmap(quality_df: pd.DataFrame, output_path: Path) -> None:
    heat_rows: list[dict[str, Any]] = []
    for _, row in quality_df.iterrows():
        n_val = int(row["n"]) if _is_finite(row["n"]) else None
        if n_val is None:
            continue
        col = f"N{n_val}:{row['source_model_key']}"
        heat_rows.append(
            {
                "kpi": row["kpi"],
                "col": col,
                "rank": _safe_float(row["rank_per_kpi_n"]),
            }
        )

    hdf = pd.DataFrame(heat_rows)
    if hdf.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No ranking data", ha="center", va="center")
        ax.axis("off")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        return

    ordered_cols: list[str] = []
    for n_val in N_ORDER:
        for model_key in MODEL_DISPLAY_ORDER:
            ordered_cols.append(f"N{n_val}:{model_key}")

    pivot = hdf.pivot_table(index="kpi", columns="col", values="rank", aggfunc="mean")
    existing_cols = [c for c in ordered_cols if c in pivot.columns]
    pivot = pivot.reindex(index=QUALITY_KPIS_FINAL, columns=existing_cols)

    fig, ax = plt.subplots(figsize=(max(12, 1.2 * len(existing_cols)), 6), constrained_layout=True)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu_r",
        cbar_kws={"label": "Rank (1=best)"},
        ax=ax,
        mask=~np.isfinite(pivot.to_numpy()),
    )
    ax.set_title("Per-KPI Rank Heatmap")
    ax.set_xlabel("N / Model")
    ax.set_ylabel("KPI")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_three_zone_summary(three_zone_df: pd.DataFrame, output_path: Path) -> bool:
    if three_zone_df.empty:
        return False

    df = three_zone_df.copy()
    df = df[df["kpi"].isin([
        "on_manifold_drift",
        "near_outside_convergence",
        "far_outside_rescue_success",
        "escape_rate_manifold_start",
    ])]
    if df.empty:
        return False

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.barplot(data=df, x="kpi", y="mean", color="#4c78a8", ax=ax)
    ax.set_xlabel("Three-zone KPI")
    ax.set_ylabel("Value")
    ax.set_title("Three-Zone Dynamics Summary (latest completed run)")
    ax.tick_params(axis="x", rotation=30)

    for i, (_, row) in enumerate(df.iterrows()):
        y = _safe_float(row["mean"])
        txt = _fmt_float(y, 4)
        if _is_finite(y):
            ax.text(i, y, txt, ha="center", va="bottom", fontsize=9)
        else:
            ax.text(i, 0.0, "NaN", ha="center", va="bottom", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return True


def build_bilan_markdown(
    sampler_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    quality_agg_df: pd.DataFrame,
    winner_summary: dict[str, Any],
    rules: dict[str, Any],
    diagnosis: dict[str, Any],
    comparability_status: dict[str, Any],
    seed_checks: dict[str, Any],
    missing_df: pd.DataFrame,
    sanity_rows: list[dict[str, Any]],
    three_zone_info: dict[str, Any],
) -> str:
    recommendation = diagnosis["final_recommendation"]

    # Confidence level heuristic
    missing_ratio = (len(missing_df) / max(1, len(sampler_df) + len(quality_df))) if (len(sampler_df) + len(quality_df)) > 0 else 0.0
    comparable = comparability_status.get("comparable", False)
    if comparable and missing_ratio < 0.15:
        confidence = "medium"
    elif comparable:
        confidence = "medium-low"
    else:
        confidence = "low"

    table1_md = build_table1_markdown(sampler_df)
    table2_md = build_table2_markdown(quality_df)
    missing_md = build_missing_markdown(missing_df)

    sanity_df = pd.DataFrame(sanity_rows)
    sanity_lines: list[list[str]] = []
    if not sanity_df.empty:
        for _, row in sanity_df.iterrows():
            sanity_lines.append(
                [
                    str(row["label"]),
                    _fmt_float(row["actual"], 4),
                    _fmt_float(row["expected_approx"], 4),
                    _fmt_float(row["abs_diff"], 4),
                    str(bool(row["within_tolerance"])),
                ]
            )
    sanity_md = _md_table(["Check", "Actual", "Approx target", "Abs diff", "Within tol"], sanity_lines)

    # Build exactly 3 final evidence bullets
    evidence_bullets: list[str] = []

    # Evidence 1: rescue/steps effect
    rescue_base = _get_sampler_mean(sampler_df, diagnosis["primary_protocol"], "baseline", "rescue_rate")
    rescue_aniso = _get_sampler_mean(sampler_df, diagnosis["primary_protocol"], "aniso", "rescue_rate")
    steps_base = _get_sampler_mean(sampler_df, diagnosis["primary_protocol"], "baseline", "median_steps_to_manifold")
    steps_aniso = _get_sampler_mean(sampler_df, diagnosis["primary_protocol"], "aniso", "median_steps_to_manifold")
    evidence_bullets.append(
        f"{diagnosis['primary_protocol']} sampler geometry: rescue_rate {_fmt_float(rescue_base,4)} -> {_fmt_float(rescue_aniso,4)} and median_steps {_fmt_float(steps_base,2)} -> {_fmt_float(steps_aniso,2)} (baseline -> aniso)."
    )

    # Evidence 2: quality winners
    best_n = winner_summary.get("best_overall_per_n", {})
    best_n_txt = ", ".join(
        [
            f"N={n}:{(best_n.get(str(n)) or {}).get('source_model_key', 'NA')}"
            for n in N_ORDER
        ]
    )
    evidence_bullets.append(
        f"Quality aggregate winners by regime: {best_n_txt}; robust model across N: {(winner_summary.get('most_robust_across_n') or {}).get('source_model_key', 'NA')}."
    )

    # Evidence 3: comparability / coverage
    if comparability_status.get("comparable", False):
        comp_txt = "sampler configs baseline/aniso are comparable on strict+matched"
    else:
        comp_txt = f"comparability failed ({len(comparability_status.get('differences', []))} mismatch(es))"
    three_zone_txt = (
        f"three-zone latest completed run: {three_zone_info.get('selected_run')}"
        if three_zone_info.get("available")
        else "three-zone completed run not yet available"
    )
    evidence_bullets.append(f"Coverage status: {comp_txt}; {three_zone_txt}; missing KPI entries logged: {len(missing_df)}.")

    lines: list[str] = []
    lines.append("# Interim Bilan Report")
    lines.append("")
    lines.append("## Executive summary")
    lines.append(f"1. This interim bilan compares metric/sampler stability and model quality across {N_ORDER} using existing artifacts only.")
    lines.append(
        f"2. Sampler comparability gate is {'PASS' if comparability_status.get('comparable', False) else 'FAIL'} for strict/matched before any causal metric-vs-sampler claim."
    )
    lines.append(
        f"3. Current recommendation is: **{recommendation}** (decision hierarchy A/B/C with conflict-to-parallel fallback)."
    )
    lines.append(
        f"4. Rule A active={rules['rule_a']['active']}, Rule B active={rules['rule_b']['active']}, Rule C active={rules['rule_c']['active']}."
    )
    lines.append(
        f"5. Low-data missingness is explicit: augmentation KPIs are NaN at N=100/500 when n=0 and excluded from aggregate denominator."
    )
    lines.append(
        f"6. Confidence level: **{confidence}** (small n in metric_assessment and low_data, plus current coverage state)."
    )
    lines.append(
        f"7. Three-zone auto-detection selected run: {three_zone_info.get('selected_run') if three_zone_info.get('available') else 'none yet'} (completed-run criteria applied)."
    )
    lines.append(
        "8. All CIs in this report use t-intervals (small-sample), with source CI kept in JSON audit fields."
    )
    lines.append("")

    lines.append("## KPI findings")
    lines.append("")
    lines.append("### Table 1 — Sampler/Geometry Stability (baseline vs aniso, strict + matched)")
    lines.append(table1_md)
    lines.append("")
    lines.append("### Table 2 — Model Quality Across Data Regimes (N=50/100/500)")
    lines.append(table2_md)
    lines.append("")
    lines.append(
        "Interpretation: interp_smoothness is log-transformed for aggregate scoring due scale compression; geo_euc_ratio is converted to distance-to-1 error before ranking/scoring."
    )
    lines.append("")

    lines.append("## Metric vs sampler diagnosis")
    lines.append(f"Verdict: **{recommendation}**")
    lines.append("")
    lines.append(f"- Rule A ({rules['rule_a']['active']}): {rules['rule_a']['message']}")
    for ev in rules["rule_a"]["evidence"]:
        lines.append(f"- Evidence A: {ev}")
    lines.append(f"- Rule B ({rules['rule_b']['active']}): {rules['rule_b']['message']}")
    for ev in rules["rule_b"]["evidence"]:
        lines.append(f"- Evidence B: {ev}")
    lines.append(f"- Rule C ({rules['rule_c']['active']}): {rules['rule_c']['message']}")
    for ev in rules["rule_c"]["evidence"]:
        lines.append(f"- Evidence C: {ev}")
    lines.append("- Asymmetry note: rescue geometry KPIs are not uniformly available across all low-data model families; this is handled as coverage limitation.")
    lines.append("")

    lines.append("## Current risks and confidence level")
    lines.append(f"- Confidence level: **{confidence}**")
    lines.append(f"- Comparability differences count: {len(comparability_status.get('differences', []))}")
    lines.append(f"- Missing/NaN KPI entries tracked: {len(missing_df)}")
    lines.append(f"- Seed check records: metric={len(seed_checks['metric_assessment']['checks'])}, low_data={len(seed_checks['low_data_benchmark']['checks'])}")
    lines.append("")

    lines.append("## Limitations")
    lines.append("- Small sample sizes (metric_assessment n=2 seeds; low_data typically n=5 seeds).")
    lines.append("- Augmentation KPIs are unavailable (NaN, n=0) for N=100 and N=500 in current low-data artifacts.")
    lines.append("- Three-zone diagnostics depend on latest completed overnight run; in-progress runs are excluded by completion criteria.")
    lines.append("- Any quick-mode or profile constraints from source runs are inherited; no retraining/re-evaluation is performed here.")
    lines.append("")

    lines.append("## Next 5 prioritized actions")
    lines.append("1. Complete one three-zone overnight run to unlock far-rescue and escape-rate confidence with full seed support.")
    lines.append("2. Fill low-data augmentation metrics at N=100/500 (currently n=0) to stabilize cross-regime aggregate comparisons.")
    lines.append("3. Run a targeted sampler-only sweep around current aniso config to test if Rule A can be mitigated without geometry change.")
    lines.append("4. Add bootstrap sensitivity on aggregate ranking (winsorization + log transform choices) to confirm recommendation stability.")
    lines.append("5. Promote this interim bilan script into CI/regression checks so new runs auto-refresh decision evidence.")
    lines.append("")

    lines.append("## Sanity check")
    lines.append(sanity_md)
    lines.append("")

    lines.append("## Missing/NaN metrics")
    lines.append(missing_md)
    lines.append("")

    lines.append(f"Final recommendation: **{recommendation}**")
    for ev in evidence_bullets[:3]:
        lines.append(f"- {ev}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build interim bilan report from existing artifacts.")
    parser.add_argument("--metric-summary", type=Path, default=Path(DEFAULT_METRIC_SUMMARY))
    parser.add_argument("--lowdata-summary", type=Path, default=Path(DEFAULT_LOWDATA_SUMMARY))
    parser.add_argument("--three-zone-root", type=Path, default=Path(DEFAULT_THREE_ZONE_ROOT))
    parser.add_argument("--report-dir", type=Path, default=Path(DEFAULT_REPORT_DIR))
    args = parser.parse_args()

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    sampler_df_raw, metric_payload, metric_missing = parse_metric_assessment(args.metric_summary)
    quality_df_raw, lowdata_payload, low_missing = parse_lowdata_benchmark(args.lowdata_summary)

    sampler_df = compute_delta_vs_reference(
        sampler_df_raw,
        ref_model_key="baseline",
        group_cols=["source", "dataset", "protocol"],
    )

    quality_df = compute_delta_vs_reference(
        quality_df_raw,
        ref_model_key="rhvae_standard",
        group_cols=["source", "dataset", "n"],
    )
    quality_df = compute_quality_ranks(quality_df)
    quality_df, quality_agg_df = compute_robust_aggregate_scores(quality_df)

    winner_summary = build_winner_summary(quality_agg_df)

    three_zone_info = detect_latest_completed_three_zone_run(args.three_zone_root)
    three_zone_df = build_three_zone_table(three_zone_info)

    comparability_status = compute_comparability_status(metric_payload)
    seed_checks = compute_seed_checks(metric_payload, sampler_df, lowdata_payload, quality_df)

    rules, diagnosis = compute_rule_logic(
        sampler_df=sampler_df,
        quality_df=quality_df,
        quality_agg_df=quality_agg_df,
        three_zone_info=three_zone_info,
        comparability_status=comparability_status,
    )

    sanity_rows = build_sanity_checks(sampler_df, quality_df)

    missing_df = pd.DataFrame(metric_missing + low_missing).drop_duplicates()

    # Write CSV deliverables
    sampler_csv_path = report_dir / "table_sampler_geometry.csv"
    quality_csv_path = report_dir / "table_quality_by_n.csv"
    sampler_df.sort_values(
        ["source", "protocol", "source_model_key", "kpi"], ascending=[True, True, True, True]
    ).to_csv(sampler_csv_path, index=False)
    quality_df.sort_values(
        ["source", "n", "source_model_key", "kpi"], ascending=[True, True, True, True]
    ).to_csv(quality_csv_path, index=False)

    # Write figures
    fig_kpi_bars = report_dir / "fig_kpi_bars.png"
    fig_rank_heatmap = report_dir / "fig_rank_heatmap.png"
    plot_kpi_bars(sampler_df=sampler_df, quality_agg_df=quality_agg_df, output_path=fig_kpi_bars)
    plot_rank_heatmap(quality_df=quality_df, output_path=fig_rank_heatmap)

    fig_three_zone = report_dir / "fig_three_zone_summary.png"
    has_three_zone_fig = False
    if three_zone_info.get("available", False):
        has_three_zone_fig = plot_three_zone_summary(three_zone_df=three_zone_df, output_path=fig_three_zone)

    # Build markdown
    bilan_md_text = build_bilan_markdown(
        sampler_df=sampler_df,
        quality_df=quality_df,
        quality_agg_df=quality_agg_df,
        winner_summary=winner_summary,
        rules=rules,
        diagnosis=diagnosis,
        comparability_status=comparability_status,
        seed_checks=seed_checks,
        missing_df=missing_df,
        sanity_rows=sanity_rows,
        three_zone_info=three_zone_info,
    )

    bilan_md_path = report_dir / "bilan.md"
    bilan_md_path.write_text(bilan_md_text, encoding="utf-8")

    # Build JSON
    bilan_json_payload: dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "script": "scripts/build_interim_bilan.py",
            "version": "interim_bilan_v2",
        },
        "inputs": {
            "metric_summary": str(args.metric_summary),
            "lowdata_summary": str(args.lowdata_summary),
            "three_zone_root": str(args.three_zone_root),
            "report_dir": str(report_dir),
        },
        "comparability_status": comparability_status,
        "seed_checks": seed_checks,
        "missing_metrics": missing_df.to_dict(orient="records"),
        "sanity_check": sanity_rows,
        "tables": {
            "sampler_geometry_rows": sampler_df.to_dict(orient="records"),
            "quality_by_n_rows": quality_df.to_dict(orient="records"),
        },
        "derived": {
            "winner_summary": winner_summary,
            "rules": rules,
        },
        "three_zone": {
            "detection": three_zone_info,
            "rows": three_zone_df.to_dict(orient="records"),
        },
        "diagnosis": diagnosis,
        "recommendation": diagnosis["final_recommendation"],
        "artifacts": {
            "bilan_md": str(bilan_md_path),
            "bilan_json": str(report_dir / "bilan.json"),
            "table_sampler_geometry_csv": str(sampler_csv_path),
            "table_quality_by_n_csv": str(quality_csv_path),
            "fig_kpi_bars": str(fig_kpi_bars),
            "fig_rank_heatmap": str(fig_rank_heatmap),
            "fig_three_zone_summary": str(fig_three_zone) if has_three_zone_fig else None,
        },
    }

    bilan_json_path = report_dir / "bilan.json"
    bilan_json_path.write_text(json.dumps(bilan_json_payload, indent=2, allow_nan=True), encoding="utf-8")

    # Reproduce script
    reproduce_path = report_dir / "reproduce.sh"
    reproduce_body = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"',
            'cd "${REPO_ROOT}"',
            "",
            "python scripts/build_interim_bilan.py \\",
            f"  --metric-summary {args.metric_summary} \\",
            f"  --lowdata-summary {args.lowdata_summary} \\",
            f"  --three-zone-root {args.three_zone_root} \\",
            f"  --report-dir {report_dir}",
            "",
            "echo 'Generated artifacts:'",
            f"ls -1 {report_dir}",
        ]
    )
    reproduce_path.write_text(reproduce_body + "\n", encoding="utf-8")
    reproduce_path.chmod(0o755)

    print("[build_interim_bilan] Report generated:")
    print(f"- {bilan_md_path}")
    print(f"- {bilan_json_path}")
    print(f"- {sampler_csv_path}")
    print(f"- {quality_csv_path}")
    print(f"- {fig_kpi_bars}")
    print(f"- {fig_rank_heatmap}")
    if has_three_zone_fig:
        print(f"- {fig_three_zone}")
    else:
        print("- fig_three_zone_summary.png not generated (no completed three-zone run with required artifacts)")
    print(f"- {reproduce_path}")


if __name__ == "__main__":
    main()
