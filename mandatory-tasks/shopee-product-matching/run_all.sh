#!/usr/bin/env bash
# Execute all four notebooks on the REAL dataset (in order: A, B, C, Finale) and store outputs inside them.
# Usage: bash run_all.sh            (env: SHOPEE_DATA_DIR, SHOPEE_MAX_GROUPS for a faster partial run)
set -euo pipefail
cd "$(dirname "$0")"
python tools/build_notebooks.py
for part in PartA PartB PartC Finale; do
  echo "=== $part ==="
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$part/notebook.ipynb"
done
echo "Done. See */results and Finale/report.pdf"
