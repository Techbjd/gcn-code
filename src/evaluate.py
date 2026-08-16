"""Evaluation CLI: load a checkpoint and evaluate on the test split."""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.loader import DataLoader

from src.models.gcn import GCNClassifier, load_checkpoint
from src.utils import compute_metrics, get_device, load_config, set_seed


def _eval_loader(model, loader, device):
    """Evaluate a loader; returns (avg_loss, y_true, y_pred, y_prob, metrics)."""
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            y_prob.extend(F.softmax(logits, dim=1)[:, 1].tolist())
            y_true.extend(batch.y.tolist())
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob, compute_metrics(y_true, y_pred, y_prob)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained GCN checkpoint on the test split.")
    parser.add_argument("--config", default="config/cpu.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    from src.data.dataset import load_datasets

    tcfg = cfg.get("train", {})
    mcfg = cfg.get("model", {})
    set_seed(tcfg.get("seed", 42))
    device = get_device(cfg)

    data = load_datasets(cfg)
    model = GCNClassifier(
        num_node_features=data["num_node_features"],
        hidden_dim=mcfg.get("hidden_dim", 128),
        num_layers=mcfg.get("num_layers", 3),
        dropout=mcfg.get("dropout", 0.2),
        pooling=mcfg.get("pooling", "sum"),
        model_type=mcfg.get("model_type", "gcn"),
    )
    ckpt_path = tcfg.get("checkpoint", "checkpoints/best_model.pt")
    load_checkpoint(model, ckpt_path, device)
    model.to(device)
    print(f"Loaded checkpoint: {ckpt_path} | Device: {device}")

    test_loader = DataLoader(data["test"], batch_size=tcfg.get("batch_size", 32), shuffle=False)
    y_true, y_pred, y_prob, metrics = _eval_loader(model, test_loader, device)

    print(f"\n=== TEST EVALUATION: {data['metadata']['n_test']} molecules ===")
    print(classification_report(y_true, y_pred, digits=4))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved test metrics -> outputs/test_metrics.json")


if __name__ == "__main__":
    main()