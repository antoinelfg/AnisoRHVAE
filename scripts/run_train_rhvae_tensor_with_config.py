#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Any

from omegaconf import OmegaConf  # pyright: ignore[reportMissingImports]


def _value_to_cli(key: str, value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bool):
        return [f"--{key}"] if value else []
    if isinstance(value, (list, tuple)):
        args: list[str] = []
        for item in value:
            args.extend(_value_to_cli(key, item))
        return args
    return [f"--{key}", str(value)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train_rhvae_tensor.py from a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Override values as key=value (OmegaConf dotlist).",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    cli_args: list[str] = []
    for key, value in OmegaConf.to_container(cfg, resolve=True).items():
        cli_args.extend(_value_to_cli(key, value))

    sys.argv = ["train_rhvae_tensor.py", *cli_args]
    from train_rhvae_tensor import main as run_main

    run_main()


if __name__ == "__main__":
    main()
