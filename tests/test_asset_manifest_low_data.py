from __future__ import annotations

from pathlib import Path

from src.utils.asset_manifest import load_yaml, sha256_file


def test_load_yaml_mapping(tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    p.write_text("a: 1\nb: test\n", encoding="utf-8")
    out = load_yaml(p)
    assert out["a"] == 1
    assert out["b"] == "test"


def test_sha256_file_stable(tmp_path: Path) -> None:
    p = tmp_path / "payload.bin"
    p.write_bytes(b"hello-world")
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2
    assert len(h1) == 64
