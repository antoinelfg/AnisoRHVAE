#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="${1:-configs/rhmc_three_zone_standard_metric_penetration_v6.yaml}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

python - "${CONFIG_PATH}" <<'PY'
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

script = str(cfg.get("script", "scripts/quick/visualize_rhmc_three_zones_real_metric.py"))
common = dict(cfg.get("common_args", {}) or {})
regimes = cfg.get("regimes", {}) or {}

if not regimes:
    raise SystemExit(f"No regimes found in {cfg_path}")

for regime_name, regime_cfg in regimes.items():
    args = dict(common)
    args.update(dict(regime_cfg or {}))

    cmd = [sys.executable, script]
    for key, value in args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, list):
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
        else:
            cmd.extend([flag, str(value)])

    print(f"\n=== Running {regime_name} ===")
    print(" ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)

print("\nAll three-zone runs completed.")
PY
