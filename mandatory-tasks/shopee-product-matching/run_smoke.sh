#!/usr/bin/env bash
# Quick pipeline self-test on SYNTHETIC data with tiny stand-in models (no GPU, no downloads, ~1-2 min).
# It only proves the code runs end-to-end; the numbers are meaningless.
set -euo pipefail
cd "$(dirname "$0")"
python tools/make_synthetic_data.py --out data/synthetic --groups 300
export SHOPEE_SMOKE=1 SHOPEE_DATA_DIR="$PWD/data/synthetic" SHOPEE_CACHE_DIR="$PWD/cache_smoke" MPLBACKEND=Agg
for part in PartA PartB PartC Finale; do
  echo "=== $part ==="; (cd $part && python notebook.py > /dev/null)
done
echo "SMOKE TEST PASSED"
