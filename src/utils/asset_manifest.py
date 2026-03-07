from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in YAML file: {path}")
    return payload


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def resolve_repo_path(root: Path, raw: str | Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def file_entries_with_hashes(root: Path, relative_paths: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel in relative_paths:
        p = (root / rel).resolve()
        out.append(
            {
                "relative_path": rel,
                "absolute_path": str(p),
                "exists": p.exists(),
                "sha256": sha256_file(p) if p.exists() and p.is_file() else None,
                "size_bytes": p.stat().st_size if p.exists() and p.is_file() else None,
            }
        )
    return out
