# Interface Specification (SPEC) — GCN MDM2 Inhibitor Pipeline

This file defines the EXACT public interfaces every module must implement so the
pipeline integrates cleanly. All agents must follow it. Do not rename functions
or change signatures. You may add private helpers.

## Goal
Replicate the ML classification step of the paper
"Machine Learning-Guided Discovery of Natural MDM2 Inhibitors"
(Budha et al., Adv. Theory Simul. 2026, DOI 10.1002/adts.202501502), replacing
the RandomForestClassifier with a Graph Convolutional Network (GCN).
Task: binary classification — is a molecule an MDM2 inhibitor?
Label: pChEMBL >= threshold (default 6.0) => active (class 1), else inactive (0).

## Environment facts (already installed)
- Python 3.13 (anaconda), PyTorch 2.13.0 CPU-only, PyTorch Geometric 2.8.0.post1,
  RDKit 2026.03.5, numpy 2.3.5, pandas 2.3.3, scikit-learn 1.8.0.
- No GPU on this machine. Code must use `torch.device` from config and work on CPU.

## File tree (root = /home/bijay/Desktop/ai-code/gcn-code)
```
src/
  __init__.py
  utils.py                 # config loading, seeding, metrics, EarlyStopping
  data/
    __init__.py
    download_chembl.py     # ChEMBL REST API -> raw CSV
    featurize.py           # SMILES -> torch_geometric Data (molecular graphs)
    dataset.py             # orchestrates download->featurize->split
  models/
    __init__.py
    gcn.py                 # GCN / GIN graph classifiers
  train.py                 # CLI entry: train + validate
  evaluate.py              # CLI entry: evaluate a checkpoint on test set
  predict.py               # CLI entry: score new SMILES with a checkpoint
config/
  cpu.yaml                 # device: cpu, small model (CPU-friendly)
  gpu.yaml                 # device: cuda if available, larger model
data/raw/                  # chembl_mdm2.csv
data/processed/            # cached graphs (optional)
checkpoints/               # best_model.pt
outputs/                   # metrics, plots, prediction CSVs
notebooks/colab_mdm2_gcn.ipynb
scripts/run_cpu.sh, run_gpu.sh
requirements.txt
README.md
```

## Data format contracts
### Raw CSV (data/raw/chembl_mdm2.csv), written by download_chembl.py
Columns (must be named exactly):
`molecule_chembl_id, smiles, standard_type, standard_value, standard_units, relation, pchembl_value`

Deduplicate by `molecule_chembl_id` (keep the row with max pchembl_value).
Rows with NULL pchembl_value or NULL smiles must be dropped.

### Featurization (featurize.py)
- `get_atom_features(atom: RDKit.Atom) -> list[int]`  length 42 (see below)
- `get_bond_features(bond: RDKit.Bond) -> list[int]`  length 10 (see below)
- `mol_to_graph(mol: RDKit.Mol) -> torch_geometric.data.Data` with fields:
  `x` [N,42] float32, `edge_index` [2,M] int64, `edge_attr` [M,10] float32.
  No self-loops. Undirected edges: add both directions.
- `smiles_to_data(smiles: str, pchembl: float|None, label: int|None) -> Data`
  Builds RDKit Mol from SMILES; if `mol is None` (invalid) return None.
  Sets `y = torch.tensor([label], dtype=torch.long)` and
  `y_pchembl = torch.tensor([pchembl], dtype=torch.float)` when not None.
  Sanitize with `Chem.SanitizeMol`, on failure return None.
- `build_graph_dataset(df: pd.DataFrame, threshold: float = 6.0)
  -> tuple[list[Data], np.ndarray, np.ndarray]`
  Returns (graphs, labels, pchembl_values). label = 1 if pchembl>=threshold else 0.
- `scaffold_split(graphs: list[Data], test_ratio=0.2, val_ratio=0.15, seed=42)
  -> tuple[list[Data], list[Data], list[Data]]` (train, val, test)
  Murcko scaffold grouping (RDKit `Scaffold.MurckoScaffoldSmiles`); assign each
  scaffold group wholly to one split; random tie-breaking by seed.

### Atom features (length 42)
1. atomic number one-hot (indices 1..18) = 18 values
2. degree one-hot (0..5) = 6
3. implicit H count one-hot (0..4) = 5
4. hybridization one-hot (SP,SP2,SP3,SP3D,SP3D2,OTHER) = 6
5. formal charge one-hot (-1,0,+1) = 3
6. is_aromatic (0/1) = 1
7. in_ring (0/1) = 1
8. total H count one-hot (0..3) = 4
Sum = 18+6+5+6+3+1+1+4 = 44? -> NO: keep exactly 42.
Use: 18 + 6 + 5 + 6 + 3 + 1 + 1 + 2 = 42
8. total H count clipped to 0..1 (i.e., binary has_any_H) = 2? -> NO.
SIMPLER, EXACT 42:
  1. atomic number one-hot (1..18) = 18
  2. degree one-hot (0..5) = 6
  3. implicit H count one-hot (0..4) = 5
  4. hybridization one-hot (SP,SP2,SP3,SP3D,SP3D2) = 5  (default index = OTHER=4)
  5. formal charge one-hot (-1,0,+1) = 3
  6. is_aromatic = 1
  7. in_ring = 1
  8. is_in_3_ring / in_any_small_ring = 1
  9. has_radical = 1
  10. chiral_center (0/1) = 1
Total: 18+6+5+5+3+1+1+1+1+1 = 42  ✓

### Bond features (length 10)
1. bond type one-hot (SINGLE,DOUBLE,TRIPLE,AROMATIC) = 4
2. is_conjugated = 1
3. is_in_ring = 1
4. stereo one-hot (STEREONONE,STEREOANY,E/Z,Z/E) = 3
5. is_aromatic = 1
Total: 4+1+1+3+1 = 10 ✓

### dataset.py
- `load_datasets(cfg: dict) -> dict` returning:
  ```
  {
    "train": list[Data], "val": list[Data], "test": list[Data],
    "num_node_features": int (42),
    "num_edge_features": int (10),
    "metadata": {n_total, n_train, n_val, n_test, n_active, n_inactive,
                 pos_weight (float), class_weight (list[float]), threshold}
  }
  ```
  Behavior: reads `cfg["data"]["raw_csv"]`; if missing, calls
  `download_mdm2_activity(out_path, use_cache=True)` from download_chembl.py.
  Applies threshold -> labels -> scaffold split with
  `cfg["train"]["seed"]`, ratios from cfg. Prints a short summary.

### download_chembl.py
- `download_mdm2_activity(out_path: str, use_cache: bool = True) -> pd.DataFrame`
  Uses ChEMBL REST API. Target MDM2 = CHEMBL5023.
  Query endpoint (paginate, limit 1000):
  `https://www.ebi.ac.uk/chembl/api/data/activity`
  params: `target_chembl_id=CHEMBL5023`,
  `standard_type__in=IC50,Ki,Kd,EC50`,
  `pchembl_value__isnull=false`, `format=json`.
  Set a polite `User-Agent`. Save deduplicated CSV at out_path.
  If use_cache and out_path exists, just load and return it.

## Model contract (src/models/gcn.py)
- `class GCNClassifier(nn.Module)`:
  `__init__(self, num_node_features: int, hidden_dim: int = 128,
            num_layers: int = 3, num_classes: int = 2, dropout: float = 0.2,
            pooling: str = "sum", model_type: str = "gcn")`
  model_type in {"gcn","gin"}:
    - "gcn": stack of `torch_geometric.nn.GCNConv`
    - "gin": stack of `torch_geometric.nn.GINConv` (MLP: Linear->ReLU->Linear)
  BatchNorm after each conv, ReLU, Dropout. Global pooling via
  `global_add_pool`/`global_mean_pool`/`global_max_pool` per `pooling`.
  Head: Linear(hidden, num_classes).
- `forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]`
  returns (logits [B,num_classes], embedding [B,hidden_dim]).
  Moves nothing; caller batches on device.
- `GINConv` must use `eps=0.0` fixed (no training eps) for stability.
- Save/load helpers:
  - `save_checkpoint(model, path)`
  - `load_checkpoint(model, path, device)` returns model with weights loaded.

## Training/eval/predict contracts
### train.py (CLI)
`python -m src.train --config config/cpu.yaml`
- loads config via `src.utils.load_config`
- trains GCNClassifier, Adam(lr, weight_decay), CrossEntropyLoss with
  `pos_weight` from metadata (handles class imbalance).
- Tracks val ROC-AUC; EarlyStopping on val loss with patience
  `cfg["train"]["early_stopping"]`; saves best to
  `cfg["train"]["checkpoint"]`.
- Every epoch: print train loss/acc, val loss/acc/auc.
- After training: evaluate on test set, print classification report,
  save `outputs/test_metrics.json` (accuracy, roc_auc, pr_auc, precision,
  recall, f1, confusion_matrix as list).
- Save `outputs/plots/{roc_curve,pr_curve,confusion_matrix}.png`.
- Save final `outputs/training_history.csv` (epoch, loss, acc, val_loss,
  val_acc, val_auc).
- Uses `set_seed(cfg["train"]["seed"])`. Device from cfg.

### evaluate.py (CLI)
`python -m src.evaluate --config config/cpu.yaml`
Loads checkpoint, evaluates on test split, prints report, saves metrics JSON.

### predict.py (CLI)
`python -m src.predict --config config/cpu.yaml --smiles_file data/raw/my_molecules.csv --output outputs/predictions.csv`
Input CSV must have a `smiles` column (optional `id` column).
Output CSV columns: `id, smiles, probability_active, class` where
class = 1 if probability >= `cfg["predict"]["threshold"]` (default 0.5).
Sort by probability descending.

### utils.py
- `load_config(path: str) -> dict` — YAML -> dict (safe load).
- `set_seed(seed: int)` — torch, numpy, random.
- `get_device(cfg) -> torch.device` — `cfg["train"]["device"]`; if "cuda" and
  not available, fall back to cpu with a printed warning.
- `compute_metrics(y_true, y_pred, y_prob) -> dict` — accuracy, roc_auc,
  pr_auc (average_precision), precision, recall, f1.
- `class EarlyStopping(patience, verbose=False)` with `.step(val_loss, model, path)`.

## Style requirements
- Python >=3.10, type hints on public functions, concise docstrings.
- NO comments unless truly necessary.
- All `import torch` guarded so module import is safe on CPU.
- Deterministic where possible (seeds).
- Every script must print clear progress messages to stdout.