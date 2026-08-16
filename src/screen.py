"""Screening CLI: score a large SMILES library (e.g., COCONUT) with a trained checkpoint.

Streams the library in chunks, featurizes each chunk across CPU cores in parallel,
scores with the trained GCN, and writes per-chunk CSVs so progress survives disconnects.
"""

import argparse
import glob
import os

import pandas as pd
import torch
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch_geometric.loader import DataLoader

from src.models.gcn import GCNClassifier, load_checkpoint
from src.utils import get_device, load_config, set_seed


def _featurize_batch(pairs):
    """Worker: convert (idx, smiles) pairs to graphs; returns parsed ids and graphs."""
    from src.data.featurize import smiles_to_data

    ids_out, graphs_out = [], []
    for idx, smi in pairs:
        if smi is None or (isinstance(smi, float) and pd.isna(smi)):
            continue
        g = smiles_to_data(str(smi), None, None)
        if g is not None:
            ids_out.append(idx)
            graphs_out.append(g)
    return ids_out, graphs_out


def score_graphs(model, graphs, device, batch_size: int) -> list[float]:
    """Return softmax probability of the active class for each graph."""
    probs: list[float] = []
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            probs.extend(F.softmax(logits, dim=1)[:, 1].tolist())
    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen a large SMILES library with a trained GCN.")
    parser.add_argument("--config", default="config/cpu.yaml", help="Path to YAML config.")
    parser.add_argument("--input", required=True, help="Library CSV (one row per molecule).")
    parser.add_argument("--smiles_col", default="smiles", help="Column holding SMILES.")
    parser.add_argument("--id_col", default="id", help="Column holding the molecule id.")
    parser.add_argument("--output", default="outputs/screen_predictions.csv", help="Combined output CSV.")
    parser.add_argument("--chunk_size", type=int, default=50000, help="Molecules per chunk.")
    parser.add_argument("--workers", type=int, default=0, help="Featurizer processes (0 = auto).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg.get("train", {})
    mcfg = cfg.get("model", {})
    set_seed(tcfg.get("seed", 42))
    device = get_device(cfg)

    model = GCNClassifier(
        num_node_features=42,
        hidden_dim=mcfg.get("hidden_dim", 128),
        num_layers=mcfg.get("num_layers", 3),
        dropout=mcfg.get("dropout", 0.2),
        pooling=mcfg.get("pooling", "sum"),
        model_type=mcfg.get("model_type", "gcn"),
    )
    ckpt_path = tcfg.get("checkpoint", "checkpoints/best_model.pt")
    load_checkpoint(model, ckpt_path, device)
    model.eval()
    print(f"Model on {device} | checkpoint: {ckpt_path}")

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    outdir = os.path.join(os.path.dirname(args.output) or ".", "screen_chunks")
    os.makedirs(outdir, exist_ok=True)

    batch_size = tcfg.get("batch_size", 32)
    cols = pd.read_csv(args.input, nrows=0).columns
    has_id = args.id_col in cols
    usecols = [args.smiles_col] + ([args.id_col] if has_id else [])
    reader = pd.read_csv(args.input, chunksize=args.chunk_size, usecols=usecols)
    chunk_idx, total = 0, 0
    for chunk in reader:
        smis = chunk[args.smiles_col].tolist()
        ids = chunk[args.id_col].tolist() if has_id else list(range(total, total + len(smis)))
        id_name = args.id_col if has_id else "id"
        n = len(ids)
        pairs = list(zip(ids, smis))
        sub = [pairs[j * n // workers:(j + 1) * n // workers] for j in range(workers)]
        sub = [s for s in sub if s]

        graphs, keep_ids = [], []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_featurize_batch, s) for s in sub]
            for f in as_completed(futures):
                i_out, g_out = f.result()
                keep_ids.extend(i_out)
                graphs.extend(g_out)
        if not graphs:
            print(f"chunk {chunk_idx}: no valid molecules; skipping")
            chunk_idx += 1
            continue

        probs = score_graphs(model, graphs, device, batch_size)
        part = pd.DataFrame({id_name: keep_ids, "probability_active": probs})
        part.to_csv(os.path.join(outdir, f"chunk_{chunk_idx}.csv"), index=False)
        total += len(part)
        print(f"chunk {chunk_idx}: {len(part)} scored (running total {total})")
        chunk_idx += 1

    parts = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(outdir, "chunk_*.csv")))]
    out = pd.concat(parts, ignore_index=True).sort_values("probability_active", ascending=False)
    threshold = cfg.get("predict", {}).get("threshold", 0.5)
    out["class"] = (out["probability_active"] >= threshold).astype(int)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out)} predictions -> {args.output}")


if __name__ == "__main__":
    main()