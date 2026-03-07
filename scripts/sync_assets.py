#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.utils.asset_manifest import dump_json, ensure_dir, file_entries_with_hashes, load_yaml, resolve_repo_path, sha256_file, utc_now_iso


def _clean_refs(refs: list[str] | None) -> list[str]:
    if not refs:
        return []
    return [r.strip() for r in refs if isinstance(r, str) and r.strip()]


def _has_required_files(path: Path, required_files: list[str]) -> bool:
    return all((path / rel).exists() for rel in required_files)


def _symlink_or_copy(alias_path: Path, target_path: Path) -> None:
    ensure_dir(alias_path.parent)
    if alias_path.is_symlink() or alias_path.exists():
        if alias_path.is_dir() and not alias_path.is_symlink():
            # Preserve existing directory by replacing only if it points to the same target.
            # Safer behavior: require manual cleanup.
            if alias_path.resolve() == target_path.resolve():
                return
            raise RuntimeError(f"Alias path exists as a directory: {alias_path}")
        alias_path.unlink()
    os.symlink(target_path, alias_path)


def _download_wandb_artifact(ref: str, out_dir: Path) -> Path | None:
    try:
        import wandb
    except Exception:
        print("[sync_assets] wandb not installed; cannot fetch artifacts.")
        return None

    try:
        api = wandb.Api()
        artifact = api.artifact(ref)
        ensure_dir(out_dir)
        target = Path(artifact.download(root=str(out_dir)))
        return target
    except Exception as exc:
        print(f"[sync_assets] Failed to download artifact '{ref}': {exc}")
        return None


def _resolve_model(
    model_entry: dict[str, Any],
    cache_root: Path,
    materialize_aliases: bool,
    allow_wandb: bool,
    verify_only: bool,
) -> dict[str, Any]:
    model_id = str(model_entry["model_id"])
    alias_path = resolve_repo_path(ROOT, model_entry["alias_path"])
    required_files = [str(x) for x in model_entry.get("required_files", [])]
    local_candidates = [resolve_repo_path(ROOT, p) for p in model_entry.get("local_candidates", [])]
    wandb_refs = _clean_refs(model_entry.get("wandb_refs", []))
    train_if_missing = bool(model_entry.get("train_if_missing", False))

    selected_path: Path | None = None
    source = "missing"

    if alias_path.exists() and _has_required_files(alias_path, required_files):
        selected_path = alias_path
        source = "alias_existing"

    if selected_path is None:
        for cand in local_candidates:
            if cand.exists() and _has_required_files(cand, required_files):
                selected_path = cand
                source = "local_candidate"
                break

    if selected_path is None and allow_wandb:
        for ref in wandb_refs:
            source_id = ref.replace("/", "_").replace(":", "_")
            dl_root = cache_root / model_id / source_id
            downloaded = _download_wandb_artifact(ref, dl_root)
            if downloaded is not None and _has_required_files(downloaded, required_files):
                selected_path = downloaded
                source = f"wandb:{ref}"
                break

    status = "ready" if selected_path is not None else "missing"
    if selected_path is None and train_if_missing:
        status = "missing_train_required"

    if selected_path is not None and materialize_aliases and not verify_only:
        _symlink_or_copy(alias_path, selected_path)

    entry: dict[str, Any] = {
        "model_id": model_id,
        "model_family": model_entry.get("model_family"),
        "status": status,
        "source": source,
        "alias_path": str(alias_path),
        "resolved_path": str(selected_path) if selected_path is not None else None,
        "required_files": required_files,
        "files": [],
        "train_if_missing": train_if_missing,
    }

    if selected_path is not None:
        files = []
        for rel in required_files:
            p = (selected_path / rel).resolve()
            files.append(
                {
                    "relative_path": rel,
                    "absolute_path": str(p),
                    "exists": p.exists(),
                    "sha256": sha256_file(p) if p.exists() else None,
                    "size_bytes": p.stat().st_size if p.exists() else None,
                }
            )
        entry["files"] = files

    return entry


def _resolve_dataset(data_entry: dict[str, Any], verify_only: bool) -> dict[str, Any]:
    dataset_id = str(data_entry["dataset_id"])
    processed_dir = resolve_repo_path(ROOT, data_entry["processed_dir"])
    expected = ["train.pt", "val.pt", "test.pt", "metadata.json"]

    status = "ready" if all((processed_dir / n).exists() for n in expected) else "missing"
    if status == "missing" and not verify_only:
        # Keep sync_assets side-effect free for datasets except manifest; dedicated prep script handles creation.
        pass

    files = []
    for name in expected:
        p = processed_dir / name
        files.append(
            {
                "relative_path": str(Path(data_entry["processed_dir"]) / name),
                "absolute_path": str(p.resolve()),
                "exists": p.exists(),
                "sha256": sha256_file(p) if p.exists() else None,
                "size_bytes": p.stat().st_size if p.exists() else None,
            }
        )

    return {
        "dataset_id": dataset_id,
        "status": status,
        "processed_dir": str(processed_dir),
        "files": files,
        "split_definition": data_entry.get("split_definition", {}),
        "subset_definition": data_entry.get("subset_definition", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve and sync model/dataset assets into stable aliases.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model_ids to resolve.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of dataset_ids to resolve.")
    parser.add_argument("--skip_models", action="store_true", help="Skip model resolution.")
    parser.add_argument("--skip_datasets", action="store_true", help="Skip dataset resolution.")
    parser.add_argument("--registry_models", type=str, default="configs/assets/model_registry.yaml")
    parser.add_argument("--registry_data", type=str, default="configs/assets/data_registry.yaml")
    parser.add_argument("--cache_root", type=str, default="assets/cache/models")
    parser.add_argument("--materialize_aliases", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--allow_wandb", action="store_true")
    parser.add_argument("--manifest_out", type=str, default="assets/manifests/model_manifest.json")
    parser.add_argument("--data_manifest_out", type=str, default="assets/manifests/data_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_registry = load_yaml(resolve_repo_path(ROOT, args.registry_models))
    data_registry = load_yaml(resolve_repo_path(ROOT, args.registry_data))

    model_entries = list(model_registry.get("models", []))
    if args.skip_models:
        model_entries = []
    elif args.models:
        want = set(args.models)
        model_entries = [m for m in model_entries if str(m.get("model_id")) in want]

    data_entries = list(data_registry.get("datasets", []))
    if args.skip_datasets:
        data_entries = []
    elif args.datasets:
        want_d = set(args.datasets)
        data_entries = [d for d in data_entries if str(d.get("dataset_id")) in want_d]

    cache_root = resolve_repo_path(ROOT, args.cache_root)
    ensure_dir(cache_root)

    model_results = [
        _resolve_model(
            model_entry=m,
            cache_root=cache_root,
            materialize_aliases=args.materialize_aliases,
            allow_wandb=args.allow_wandb,
            verify_only=args.verify_only,
        )
        for m in model_entries
    ]

    data_results = [_resolve_dataset(d, verify_only=args.verify_only) for d in data_entries]

    model_manifest = {
        "created_at": utc_now_iso(),
        "verify_only": bool(args.verify_only),
        "allow_wandb": bool(args.allow_wandb),
        "models": model_results,
    }
    data_manifest = {
        "created_at": utc_now_iso(),
        "verify_only": bool(args.verify_only),
        "datasets": data_results,
    }

    dump_json(resolve_repo_path(ROOT, args.manifest_out), model_manifest)
    dump_json(resolve_repo_path(ROOT, args.data_manifest_out), data_manifest)

    missing = [m for m in model_results if m["status"] == "missing"]
    missing_d = [d for d in data_results if d["status"] == "missing"]

    print("[sync_assets] Model statuses:")
    for row in model_results:
        print(f"  - {row['model_id']}: {row['status']} ({row['source']})")

    print("[sync_assets] Dataset statuses:")
    for row in data_results:
        print(f"  - {row['dataset_id']}: {row['status']}")

    train_needed = [m for m in model_results if m["status"] == "missing_train_required"]
    if train_needed:
        print("[sync_assets] Models marked for training (allowed at this stage):")
        for row in train_needed:
            print(f"  - {row['model_id']}")

    if missing or missing_d:
        # Non-zero to help CI detect unresolved assets.
        raise SystemExit(2)


if __name__ == "__main__":
    main()
