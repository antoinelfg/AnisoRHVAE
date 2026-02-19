from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any

try:
    import wandb as _wandb
except Exception:  # pragma: no cover - optional dependency
    _wandb = None


def parse_wandb_tags(raw_tags: str | None) -> list[str] | None:
    if not raw_tags:
        return None
    tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    return tags or None


def init_wandb_run(
    *,
    project: str | None,
    entity: str | None,
    group: str | None,
    tags: str | None,
    run_name: str | None,
    name_mode: str,
    mode: str | None,
    config: dict[str, Any] | None,
    name_prefix: str,
) -> Any | None:
    if mode:
        os.environ["WANDB_MODE"] = mode

    if _wandb is None:
        if project or os.environ.get("WANDB_PROJECT"):
            print("[warning] wandb not installed; skipping wandb logging.")
        return None

    resolved_project = project or os.environ.get("WANDB_PROJECT")
    resolved_entity = entity or os.environ.get("WANDB_ENTITY")
    if not resolved_project:
        return None

    resolved_name = run_name
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if name_mode == "timestamp":
        resolved_name = run_name or f"{name_prefix}_{stamp}"
    elif name_mode == "auto" and not resolved_name:
        resolved_name = f"{name_prefix}_{stamp}"

    kwargs: dict[str, Any] = {"config": config or {}}
    kwargs["project"] = resolved_project
    if resolved_entity:
        kwargs["entity"] = resolved_entity
    if group:
        kwargs["group"] = group
    parsed_tags = parse_wandb_tags(tags)
    if parsed_tags:
        kwargs["tags"] = parsed_tags
    if resolved_name:
        kwargs["name"] = resolved_name

    try:
        run = _wandb.init(**kwargs)
        print(
            f"[info] wandb enabled: project={resolved_project}, "
            f"entity={resolved_entity or '<default>'}, mode={os.environ.get('WANDB_MODE', 'online')}"
        )
        return run
    except Exception as exc:
        print(f"[warning] failed to initialize wandb: {exc}")
        return None


def safe_wandb_log(run: Any | None, payload: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    try:
        if step is None:
            run.log(payload)
        else:
            run.log(payload, step=int(step))
    except Exception:
        return


def safe_wandb_finish(run: Any | None) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        return


def make_wandb_image(path: Path | str) -> Any | None:
    if _wandb is None:
        return None
    try:
        return _wandb.Image(str(path))
    except Exception:
        return None


def log_dir_artifact(run: Any | None, dir_path: Path | str, *, artifact_name: str, artifact_type: str) -> None:
    if run is None or _wandb is None:
        return
    try:
        artifact = _wandb.Artifact(name=artifact_name, type=artifact_type)
        artifact.add_dir(str(dir_path))
        run.log_artifact(artifact)
    except Exception:
        return
