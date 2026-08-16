"""Training CLI: train + validate + final test evaluation."""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch_geometric.loader import DataLoader

from src.models.gcn import GCNClassifier, load_checkpoint
from src.utils import EarlyStopping, compute_metrics, get_device, load_config, set_seed


def _eval_loader(model, loader, device, criterion=None):
    """Evaluate a loader; returns (avg_loss, y_true, y_pred, y_prob, metrics)."""
    model.eval()
    losses, y_true, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            if criterion is not None:
                losses.append(criterion(logits, batch.y).item())
            y_prob.extend(F.softmax(logits, dim=1)[:, 1].tolist())
            y_true.extend(batch.y.tolist())
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, y_true, y_pred, y_prob, compute_metrics(y_true, y_pred, y_prob)


def _build_model(cfg: dict, num_features: int) -> GCNClassifier:
    mcfg = cfg.get("model", {})
    return GCNClassifier(
        num_node_features=num_features,
        hidden_dim=mcfg.get("hidden_dim", 128),
        num_layers=mcfg.get("num_layers", 3),
        dropout=mcfg.get("dropout", 0.2),
        pooling=mcfg.get("pooling", "sum"),
        model_type=mcfg.get("model_type", "gcn"),
    )


def _save_plots(y_true, y_pred, y_prob, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC (AUC={compute_metrics(y_true, y_pred, y_prob)['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "roc_curve.png"))
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(rec, prec, label=f"PR (AP={compute_metrics(y_true, y_pred, y_prob)['pr_auc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pr_curve.png"))
    plt.close(fig)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["Inactive", "Active"])
    ax.set_yticks([0, 1], ["Inactive", "Active"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "confusion_matrix.png"))
    plt.close(fig)


def train(cfg: dict, args) -> dict:
    """Run the full training pipeline; returns test metrics dict."""
    from src.data.dataset import load_datasets

    tcfg = cfg.get("train", {})
    set_seed(tcfg.get("seed", 42))
    device = get_device(cfg)

    data = load_datasets(cfg)
    model = _build_model(cfg, data["num_node_features"]).to(device)
    print(f"Model: {model.model_type} ({model.pooling} pooling), {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Device: {device} | Train={data['metadata']['n_train']} Val={data['metadata']['n_val']} Test={data['metadata']['n_test']}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tcfg.get("lr", 1e-3),
        weight_decay=tcfg.get("weight_decay", 1e-4),
    )
    pos_weight = torch.tensor([data["metadata"]["pos_weight"]], device=device)


    def criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        weight = torch.where(targets == 1, pos_weight, torch.ones_like(pos_weight))
        return (ce * weight).mean()

    bs = tcfg.get("batch_size", 32)
    train_loader = DataLoader(data["train"], batch_size=bs, shuffle=True)
    val_loader = DataLoader(data["val"], batch_size=bs, shuffle=False)
    test_loader = DataLoader(data["test"], batch_size=bs, shuffle=False)

    ckpt_path = tcfg.get("checkpoint", "checkpoints/best_model.pt")
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    es = EarlyStopping(tcfg.get("early_stopping", 10), verbose=True)
    epochs = tcfg.get("epochs", 50)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.y.numel()
            total_correct += (logits.argmax(1) == batch.y).sum().item()
            total_n += batch.y.numel()
        train_loss = total_loss / max(total_n, 1)
        train_acc = total_correct / max(total_n, 1)
        val_loss, _, _, _, val_metrics = _eval_loader(model, val_loader, device, criterion)
        history.append({
            "epoch": epoch,
            "loss": train_loss,
            "acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_metrics["accuracy"],
            "val_auc": val_metrics["roc_auc"],
        })
        print(f"Epoch {epoch:3d}/{epochs}: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} val_auc={val_metrics['roc_auc']:.4f}")
        if es.step(val_loss, model, ckpt_path):
            print(f"Early stopping triggered at epoch {epoch} (best val_loss={es.best:.4f}).")
            break

    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(history).to_csv("outputs/training_history.csv", index=False)
    print(f"Saved training history -> outputs/training_history.csv")

    load_checkpoint(model, ckpt_path, device)
    _, y_true, y_pred, y_prob, metrics = _eval_loader(model, test_loader, device)
    print("\n=== TEST EVALUATION (best checkpoint) ===")
    print(classification_report(y_true, y_pred, digits=4))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    with open("outputs/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved test metrics -> outputs/test_metrics.json")
    _save_plots(y_true, y_pred, y_prob, "outputs/plots")
    print("Saved plots -> outputs/plots/{roc_curve,pr_curve,confusion_matrix}.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GCN classifier on MDM2 activity data.")
    parser.add_argument("--config", default="config/cpu.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, args)


if __name__ == "__main__":
    main()