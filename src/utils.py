"""Shared utilities: config loading, seeding, device selection, metrics, early stopping."""

import random

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.gcn import save_checkpoint


def load_config(path: str) -> dict:
    """Load a YAML config file into a dict (safe load)."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed torch, numpy and random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    """Return torch.device from cfg['train']['device'], falling back to CPU."""
    name = cfg.get("train", {}).get("device", "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; falling back to CPU.")
        name = "cpu"
    return torch.device(name)


def _safe(metric_fn, *args, **kwargs) -> float:
    """Run a sklearn metric, returning 0.0 when the metric is undefined."""
    try:
        value = float(metric_fn(*args, **kwargs))
        return 0.0 if np.isnan(value) else value
    except ValueError:
        return 0.0


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute classification metrics from true/predicted labels and probabilities."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": _safe(roc_auc_score, y_true, y_prob),
        "pr_auc": _safe(average_precision_score, y_true, y_prob),
        "precision": _safe(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe(recall_score, y_true, y_pred, zero_division=0),
        "f1": _safe(f1_score, y_true, y_pred, zero_division=0),
    }


class EarlyStopping:
    """Stop training when val_loss stops improving; saves the best checkpoint."""

    def __init__(self, patience: int, verbose: bool = False):
        self.patience = patience
        self.verbose = verbose
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss: float, model: torch.nn.Module, path: str) -> bool:
        """Track val_loss, save checkpoint on improvement, return True to stop."""
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
            save_checkpoint(model, path)
            if self.verbose:
                print(f"  EarlyStopping: val_loss improved to {val_loss:.4f} -> {path}")
            return False
        self.counter += 1
        if self.verbose:
            print(f"  EarlyStopping: no improvement ({self.counter}/{self.patience})")
        return self.counter >= self.patience