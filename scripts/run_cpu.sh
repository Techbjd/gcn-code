#!/usr/bin/env bash
# CPU pipeline runner: download -> train -> evaluate -> predict.
# Run from the repo root:  bash scripts/run_cpu.sh
set -euo pipefail

# The machine already has Python with the required packages in the base env,
# so we simply use `python`. Adjust the interpreter if you use a venv/conda.
PYTHON=${PYTHON:-python}
CONFIG=config/cpu.yaml
SMILES=data/raw/example_smiles.csv
OUTPUT=outputs/predictions_cpu.csv

echo "==> [1/4] Downloading ChEMBL MDM2 (CHEMBL5023) bioactivity data (if needed)"
"$PYTHON" -m src.data.download_chembl --config "$CONFIG"

echo "==> [2/4] Training GCN classifier (device: cpu)"
"$PYTHON" -m src.train --config "$CONFIG"

echo "==> [3/4] Evaluating best checkpoint on the held-out test split"
"$PYTHON" -m src.evaluate --config "$CONFIG"

echo "==> [4/4] Predicting activity for example SMILES"
"$PYTHON" -m src.predict --config "$CONFIG" --smiles_file "$SMILES" --output "$OUTPUT"

echo "==> Done. Predictions written to $OUTPUT"