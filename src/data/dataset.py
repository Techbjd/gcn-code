"""Orchestrate download -> featurization -> scaffold split."""

from __future__ import annotations

import os

import pandas as pd

from .download_chembl import download_mdm2_activity
from .featurize import build_graph_dataset, scaffold_split


def load_datasets(cfg: dict) -> dict:
    """Load MDM2 graphs as train/val/test splits with metadata per SPEC."""
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    raw_csv = data_cfg.get("raw_csv", "data/raw/chembl_mdm2.csv")
    if not os.path.exists(raw_csv):
        print("Raw CSV not found; downloading from ChEMBL...")
        download_mdm2_activity(raw_csv, use_cache=True)

    df = pd.read_csv(raw_csv)
    threshold = data_cfg.get("threshold_pchembl", 6.0)

    graphs, labels, pchembl_values = build_graph_dataset(df, threshold=threshold)

    seed = int(train_cfg.get("seed", 42))
    test_ratio = float(train_cfg.get("test_ratio", 0.2))
    val_ratio = float(train_cfg.get("val_ratio", 0.15))

    train, val, test = scaffold_split(
        graphs,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    n_active = int(labels.sum())
    n_inactive = int(len(labels) - n_active)
    pos_weight = float(n_inactive / n_active) if n_active > 0 else 1.0
    class_weight = [
        float(len(labels) / (2.0 * n_inactive)) if n_inactive > 0 else 1.0,
        float(len(labels) / (2.0 * n_active)) if n_active > 0 else 1.0,
    ]

    metadata = {
        "n_total": len(graphs),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_active": n_active,
        "n_inactive": n_inactive,
        "pos_weight": pos_weight,
        "class_weight": class_weight,
        "threshold": threshold,
    }

    print(
        f"Graphs: total={metadata['n_total']} train={metadata['n_train']} "
        f"val={metadata['n_val']} test={metadata['n_test']} "
        f"(active={n_active}, inactive={n_inactive}, threshold={threshold})"
    )

    return {
        "train": train,
        "val": val,
        "test": test,
        "num_node_features": 42,
        "num_edge_features": 10,
        "metadata": metadata,
    }