#!/bin/bash
# Executes all notebooks in order, saving outputs in place.
set -e
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9
PY=${PY:-.venv/bin/python}
for nb in "$@"; do
  echo "=== $(date '+%H:%M:%S') running $nb"
  $PY -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb"
  echo "=== $(date '+%H:%M:%S') finished $nb"
done
