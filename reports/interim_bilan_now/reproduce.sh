#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

python scripts/build_interim_bilan.py \
  --metric-summary results/metric_assessment/2026-02-10_17-35-53/summary_by_protocol.json \
  --lowdata-summary results/low_data_benchmark/2026-02-12_17-56-30/2026-02-12_21-04-28/summary_by_model_n.json \
  --three-zone-root results/three_zone_overnight_huge \
  --report-dir reports/interim_bilan_now

echo 'Generated artifacts:'
ls -1 reports/interim_bilan_now
