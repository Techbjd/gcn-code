"""Prediction CLI: score new SMILES with a trained checkpoint."""

import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.models.gcn import GCNClassifier, load_checkpoint
from src.utils import get_device, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict MDM2 inhibitor activity for SMILES.")
    parser.add_argument("--config", default="config/cpu.yaml", help="Path to YAML config.")
    parser.add_argument("--smiles_file", required=True, help="CSV with a 'smiles' column (optional 'id').")
    parser.add_argument("--output", default="outputs/predictions.csv", help="Output CSV path.")
    args = parser.parse_args()

    from src.data.featurize import smiles_to_data

    cfg = load_config(args.config)
    tcfg = cfg.get("train", {})
    mcfg = cfg.get("model", {})
    pcfg = cfg.get("predict", {})
    set_seed(tcfg.get("seed", 42))
    device = get_device(cfg)

    df = pd.read_csv(args.smiles_file)
    graphs, ids, ok_smiles = [], [], []
    has_id = "id" in df.columns
    for idx, row in df.iterrows():
        smi = row["smiles"]
        if pd.isna(smi):
            continue
        graph = smiles_to_data(str(smi), None, None)
        if graph is None:
            print(f"WARNING: could not parse SMILES {smi!r}; skipping.")
            continue
        graphs.append(graph)
        ids.append(row["id"] if has_id else idx)
        ok_smiles.append(str(smi))
    if not graphs:
        print("No valid SMILES to predict; nothing written.")
        return
    print(f"Featurized {len(graphs)}/{len(df)} molecules.")

    model = GCNClassifier(
        num_node_features=graphs[0].x.size(1),
        hidden_dim=mcfg.get("hidden_dim", 128),
        num_layers=mcfg.get("num_layers", 3),
        dropout=mcfg.get("dropout", 0.2),
        pooling=mcfg.get("pooling", "sum"),
        model_type=mcfg.get("model_type", "gcn"),
    )
    ckpt_path = tcfg.get("checkpoint", "checkpoints/best_model.pt")
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    loader = DataLoader(graphs, batch_size=tcfg.get("batch_size", 32), shuffle=False)
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            probs.extend(F.softmax(logits, dim=1)[:, 1].tolist())

    threshold = pcfg.get("threshold", 0.5)
    out = pd.DataFrame({"id": ids, "smiles": ok_smiles, "probability_active": probs})
    out["class"] = (out["probability_active"] >= threshold).astype(int)
    out = out.sort_values("probability_active", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Predictions for {len(out)} molecules written to {args.output}")


if __name__ == "__main__":
    main()