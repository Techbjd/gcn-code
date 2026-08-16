#!/usr/bin/env bash
# GPU pipeline runner: download -> train -> evaluate -> predict.
# Run from the repo root:  bash scripts/run_gpu.sh
# Falls back to CPU automatically if CUDA is unavailable (see src.utils.get_device).
set -euo pipefail

# Use an env var so SLURM/conda users can point to their GPU interpreter, e.g.
#   PYTHON=/path/to/conda/envs/gpu/bin/python bash scripts/run_gpu.sh
PYTHON=${PYTHON:-python}
CONFIG=config/gpu.yaml
SMILES=data/raw/example_smiles.csv
OUTPUT=outputs/predictions_gpu.csv

# Optional: number of DataLoader worker processes. On GPU machines you can
# raise this to 4-8 to keep the GPU fed; leave unset to use the config value.
# NUM_WORKERS=${NUM_WORKERS:-4}

echo "==> [1/4] Downloading ChEMBL MDM2 (CHEMBL5023) bioactivity data (if needed)"
"$PYTHON" -m src.data.download_chembl --config "$CONFIG"

echo "==> [2/4] Training GCN classifier (device: cuda if available)"
"$PYTHON" -m src.train --config "$CONFIG"

echo "==> [3/4] Evaluating best checkpoint on the held-out test split"
"$PYTHON" -m src.evaluate --config "$CONFIG"

echo "==> [4/4] Predicting activity for example SMILES"
"$PYTHON" -m src.predict --config "$CONFIG" --smiles_file "$SMILES" --output "$OUTPUT"

echo "==> Done. Predictions written to $OUTPUT"

# ---------------------------------------------------------------------------
# SLURM example (submit from a login node):
#   sbatch --gres=gpu:1 --cpus-per-task=8 --time=02:00:00 scripts/run_gpu.sh
#
# Conda users: activate the GPU env first, e.g.
#   conda activate gcn-gpu
#   bash scripts/run_gpu.sh
# ---------------------------------------------------------------------------