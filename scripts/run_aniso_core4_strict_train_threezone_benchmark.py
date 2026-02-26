#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def _sanitize_run_id(raw: str) -> str:
    safe_chars = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_"}:
            safe_chars.append(ch.lower())
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_")


def _latest_valid_run(parent_dir: Path, required_files: list[str]) -> Path | None:
    if not parent_dir.exists() or not parent_dir.is_dir():
        return None
    run_dirs = sorted(
        [p for p in parent_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        if all((run_dir / f).exists() for f in required_files):
            return run_dir
    return None


def _latest_sampling_run(parent_dir: Path) -> Path | None:
    if not parent_dir.exists() or not parent_dir.is_dir():
        return None
    run_dirs = sorted(
        [p for p in parent_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        if (run_dir / "summary.json").exists() and (run_dir / "rhmc_three_zones_real_metric.png").exists():
            return run_dir
    return None


def _fmt_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _as_omegaconf_overrides(overrides: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in overrides.items():
        if value is None:
            continue
        out.append(f"{key}={_fmt_override_value(value)}")
    return out


def _args_dict_to_cli(args_map: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in args_map.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                out.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                continue
            out.append(flag)
            out.extend(str(v) for v in value)
            continue
        out.extend([flag, str(value)])
    return out


def _merge_tags(raw_tags: str | None, extra_tags: list[str]) -> str | None:
    tags: list[str] = []
    if raw_tags:
        tags.extend(t.strip() for t in raw_tags.split(",") if t.strip())
    tags.extend(t.strip() for t in extra_tags if t.strip())
    if not tags:
        return None
    uniq: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        uniq.append(tag)
    return ",".join(uniq)


def _read_meta(run_dir: Path) -> dict[str, Any] | None:
    meta_path = run_dir / "wandb_run_meta.json"
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _write_meta(run_dir: Path, payload: dict[str, Any]) -> None:
    (run_dir / "wandb_run_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _render_template(raw_template: str, *, n: int, n_pad: str, seed: int, seed_pad: str) -> str:
    return str(raw_template).format(n=n, n_pad=n_pad, seed=seed, seed_pad=seed_pad)


def _derive_num_sequences(n: int) -> int:
    return max(1, int(math.ceil(float(n) / 0.8)))


def _run_command(cmd: list[str], env: dict[str, str]) -> None:
    print("[aniso_core4_strict]", shlex.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def _resolve_subset_ns(cfg: dict[str, Any], cli_subset_ns: list[int] | None) -> list[int]:
    if cli_subset_ns:
        return [int(n) for n in cli_subset_ns]
    benchmark_cfg = dict(cfg.get("benchmark", {}) or {})
    values = benchmark_cfg.get("subset_ns", [50, 100, 1000])
    return [int(n) for n in values]


def run_benchmark(
    *,
    config_path: Path,
    subset_ns: list[int] | None,
    seed: int,
    reuse_policy: str,
    python_bin: str,
    wandb_mode: str | None,
) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    benchmark_cfg = dict(cfg.get("benchmark", {}) or {})
    training_cfg = dict(cfg.get("training", {}) or {})
    sampling_cfg = dict(cfg.get("sampling", {}) or {})
    wandb_cfg = dict(cfg.get("wandb", {}) or {})

    subset_values = _resolve_subset_ns(cfg, subset_ns)
    required_files = [str(v) for v in training_cfg.get("required_files", ["rhvae_model.pt", "rhvae_metric.pt"])]

    base_train_config = _repo_path(training_cfg.get("base_config", "configs/run_geometry_aniso_core4_n50_sampler_strict.yaml"))
    train_output_root = _repo_path(
        training_cfg.get("output_root", "outputs/low_data_models/ellipses_one_shot_core4_sampler_strict/aniso")
    )
    sampling_script = _repo_path(
        sampling_cfg.get("script", "scripts/quick/visualize_rhmc_three_zones_real_metric.py")
    )
    sampling_output_root = _repo_path(sampling_cfg.get("output_root", "results/rhmc_standard_metric_strict_core4"))
    benchmark_output_root = _repo_path(
        benchmark_cfg.get("output_root", "results/aniso_core4_strict_train_threezone")
    )

    run_stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    benchmark_dir = benchmark_output_root / run_stamp
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    default_run_name_template = "aniso_core4_strict_{n_pad}_seed{seed_pad}"
    default_run_id_template = "aniso_core4_strict_{n_pad}_seed{seed_pad}"

    rows: list[dict[str, Any]] = []
    for n in subset_values:
        row_started = time.perf_counter()
        n_pad = f"N{int(n):03d}"
        seed_pad = f"{int(seed):03d}"
        run_name_template = str(wandb_cfg.get("run_name_template", default_run_name_template))
        run_id_template = str(wandb_cfg.get("run_id_template", default_run_id_template))
        run_name = _render_template(run_name_template, n=int(n), n_pad=n_pad, seed=int(seed), seed_pad=seed_pad)
        run_id = _sanitize_run_id(
            _render_template(run_id_template, n=int(n), n_pad=n_pad, seed=int(seed), seed_pad=seed_pad)
        )
        tags = _merge_tags(wandb_cfg.get("tags"), [f"n{int(n)}"])

        row: dict[str, Any] = {
            "n": int(n),
            "seed": int(seed),
            "status": "failed",
            "model_path": None,
            "sampling_run_dir": None,
            "wandb_run_id": None,
            "commands": {"training": None, "sampling": None},
            "durations_sec": {"training": 0.0, "sampling": 0.0, "total": 0.0},
            "error": None,
            "fallback_threezone_only": False,
        }

        train_parent = train_output_root / n_pad / f"seed{seed_pad}"
        train_parent.mkdir(parents=True, exist_ok=True)

        sampling_parent = sampling_output_root / n_pad
        sampling_parent.mkdir(parents=True, exist_ok=True)

        try:
            existing_run = _latest_valid_run(train_parent, required_files)
            if existing_run is not None and reuse_policy == "fail_if_exists":
                raise RuntimeError(
                    f"Found existing trained run for N={int(n)} at {existing_run}; reuse_policy=fail_if_exists."
                )

            model_run: Path | None = existing_run
            if existing_run is None or reuse_policy == "always_retrain":
                training_overrides = dict(training_cfg.get("fixed_overrides", {}) or {})
                training_overrides["num_sequences"] = int(_derive_num_sequences(int(n)))
                training_overrides["max_frames"] = int(n)
                training_overrides["output_dir"] = str(train_parent)
                training_overrides["seed"] = int(seed)
                training_overrides["skip_analysis"] = True
                training_overrides["skip_sampling_diagnostics"] = True

                if wandb_cfg.get("project"):
                    training_overrides["wandb_project"] = str(wandb_cfg.get("project"))
                if wandb_cfg.get("entity"):
                    training_overrides["wandb_entity"] = str(wandb_cfg.get("entity"))
                if wandb_cfg.get("group"):
                    training_overrides["wandb_group"] = str(wandb_cfg.get("group"))
                if tags:
                    training_overrides["wandb_tags"] = tags
                training_overrides["wandb_name_mode"] = str(wandb_cfg.get("name_mode", "manual"))
                training_overrides["wandb_run_name"] = run_name

                train_cmd = [
                    str(python_bin),
                    str(_repo_path("scripts/run_with_config.py")),
                    "--config",
                    str(base_train_config),
                    *_as_omegaconf_overrides(training_overrides),
                ]
                row["commands"]["training"] = shlex.join(train_cmd)
                train_env = dict(os.environ)
                train_env["WANDB_RUN_ID"] = run_id
                train_env["WANDB_RESUME"] = "allow"
                if wandb_mode:
                    train_env["WANDB_MODE"] = str(wandb_mode)
                t_train = time.perf_counter()
                _run_command(train_cmd, env=train_env)
                row["durations_sec"]["training"] = float(time.perf_counter() - t_train)

                model_run = _latest_valid_run(train_parent, required_files)
                if model_run is None:
                    raise RuntimeError(f"Training completed but no valid run found in {train_parent}")

                _write_meta(
                    model_run,
                    {
                        "run_id": run_id,
                        "project": wandb_cfg.get("project"),
                        "entity": wandb_cfg.get("entity"),
                        "group": wandb_cfg.get("group"),
                        "run_name": run_name,
                        "n": int(n),
                        "seed": int(seed),
                    },
                )
                row["status"] = "trained"
            else:
                row["status"] = "reused"

            if model_run is None:
                raise RuntimeError(f"No valid model run resolved for N={int(n)}")

            row["model_path"] = str(model_run)

            meta = _read_meta(model_run)
            sampling_env = dict(os.environ)
            sampling_tags = tags
            sampling_run_name = run_name
            wandb_run_id = None

            if meta is not None and meta.get("run_id"):
                wandb_run_id = str(meta["run_id"])
                sampling_env["WANDB_RUN_ID"] = wandb_run_id
                sampling_env["WANDB_RESUME"] = "allow"
            else:
                row["fallback_threezone_only"] = True
                sampling_tags = _merge_tags(tags, ["threezone_only"])
                sampling_run_name = f"{run_name}_threezone_only"
                sampling_env.pop("WANDB_RUN_ID", None)
                sampling_env.pop("WANDB_RESUME", None)

            if wandb_mode:
                sampling_env["WANDB_MODE"] = str(wandb_mode)

            sampling_args = dict(sampling_cfg.get("common_args", {}) or {})
            sampling_args["model_path"] = str(model_run)
            sampling_args["output_dir"] = str(sampling_parent)
            if wandb_cfg.get("project"):
                sampling_args["wandb_project"] = str(wandb_cfg.get("project"))
            if wandb_cfg.get("entity"):
                sampling_args["wandb_entity"] = str(wandb_cfg.get("entity"))
            if wandb_cfg.get("group"):
                sampling_args["wandb_group"] = str(wandb_cfg.get("group"))
            if sampling_tags:
                sampling_args["wandb_tags"] = sampling_tags
            sampling_args["wandb_name_mode"] = str(wandb_cfg.get("name_mode", "manual"))
            sampling_args["wandb_run_name"] = sampling_run_name
            sampling_args["wandb_job_type"] = str(wandb_cfg.get("job_type", "three_zone_sampling"))
            if wandb_mode:
                sampling_args["wandb_mode"] = str(wandb_mode)

            sampling_cmd = [str(python_bin), str(sampling_script), *_args_dict_to_cli(sampling_args)]
            row["commands"]["sampling"] = shlex.join(sampling_cmd)
            t_sampling = time.perf_counter()
            _run_command(sampling_cmd, env=sampling_env)
            row["durations_sec"]["sampling"] = float(time.perf_counter() - t_sampling)

            sampling_run_dir = _latest_sampling_run(sampling_parent)
            if sampling_run_dir is None:
                raise RuntimeError(f"Sampling completed but no valid run found in {sampling_parent}")

            row["sampling_run_dir"] = str(sampling_run_dir)
            row["wandb_run_id"] = wandb_run_id
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
        finally:
            row["durations_sec"]["total"] = float(time.perf_counter() - row_started)
            rows.append(row)
            print(f"[aniso_core4_strict] N={int(n)} status={row['status']}")

    summary = {
        "generated_at": dt.datetime.now().isoformat(),
        "config_path": str(config_path),
        "subset_ns": subset_values,
        "seed": int(seed),
        "reuse_policy": str(reuse_policy),
        "python_bin": str(python_bin),
        "wandb_mode": str(wandb_mode) if wandb_mode else None,
        "rows": rows,
    }
    summary_path = benchmark_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[aniso_core4_strict] benchmark summary: {summary_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark runner: train (run_with_config) + three-zone sampling for Aniso Core4 strict."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark_aniso_core4_strict_train_threezone.yaml",
        help="YAML config for benchmark orchestration.",
    )
    parser.add_argument("--subset_ns", nargs="+", type=int, default=None, help="Optional override for N values.")
    parser.add_argument("--seed", type=int, default=13, help="Shared seed for all regimes.")
    parser.add_argument(
        "--reuse_policy",
        type=str,
        default="reuse_latest",
        choices=["reuse_latest", "always_retrain", "fail_if_exists"],
    )
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    run_benchmark(
        config_path=config_path,
        subset_ns=args.subset_ns,
        seed=int(args.seed),
        reuse_policy=str(args.reuse_policy),
        python_bin=str(args.python_bin),
        wandb_mode=args.wandb_mode,
    )


if __name__ == "__main__":
    main()
