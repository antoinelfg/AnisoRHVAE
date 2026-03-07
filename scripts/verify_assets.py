#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.utils.asset_manifest import sha256_file


def _verify_file(path: Path, expected_sha: str | None) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if expected_sha is None:
        return True, "ok"
    actual = sha256_file(path)
    if actual != expected_sha:
        return False, f"sha_mismatch expected={expected_sha} actual={actual}"
    return True, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify assets against model/data manifests.")
    parser.add_argument("--model_manifest", type=str, default="assets/manifests/model_manifest.json")
    parser.add_argument("--data_manifest", type=str, default="assets/manifests/data_manifest.json")
    return parser.parse_args()


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    model_payload = _load(ROOT / args.model_manifest)
    data_payload = _load(ROOT / args.data_manifest)

    ok = True

    for row in model_payload.get("models", []):
        model_id = row.get("model_id")
        status = row.get("status")
        alias_path = Path(str(row.get("alias_path", "")))
        print(f"[verify_assets] model={model_id} status={status}")
        if status != "ready":
            ok = False
            continue
        if not alias_path.exists():
            print(f"  ERROR: alias missing: {alias_path}")
            ok = False
        for f in row.get("files", []):
            p = Path(str(f.get("absolute_path")))
            expected = f.get("sha256")
            valid, msg = _verify_file(p, expected)
            if not valid:
                print(f"  ERROR: {p} -> {msg}")
                ok = False

    for row in data_payload.get("datasets", []):
        ds = row.get("dataset_id")
        status = row.get("status")
        print(f"[verify_assets] dataset={ds} status={status}")
        if status != "ready":
            ok = False
            continue
        for f in row.get("files", []):
            p = Path(str(f.get("absolute_path")))
            expected = f.get("sha256")
            valid, msg = _verify_file(p, expected)
            if not valid:
                print(f"  ERROR: {p} -> {msg}")
                ok = False

    if not ok:
        raise SystemExit(2)
    print("[verify_assets] all checks passed")


if __name__ == "__main__":
    main()
