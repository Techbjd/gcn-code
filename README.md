# GCN for MDM2 Inhibitor Classification

Binary classifier that predicts whether a molecule inhibits **MDM2** (target **CHEMBL5023**) from its SMILES string, using a **Graph Convolutional Network (GCN)** built with PyTorch Geometric.

![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue) ![framework](https://img.shields.io/badge/ML-GCN-blueviolet) ![license](https://img.shields.io/badge/license-MIT-green)

## What this project does

This repo replicates the **ML classification / virtual-screening step** of the paper

> **Machine Learning-Guided Discovery of Natural MDM2 Inhibitors: A Multistage In Silico Pipeline from Screening to ADMET Profiling**
> (Budha et al., *Advanced Theory and Simulations* 2026, DOI [10.1002/adts.202501502](https://doi.org/10.1002/adts.202501502))

The paper trained **40 traditional ML models** (best: RandomForestClassifier) on ChEMBL MDM2 bioactivity data, screened ~700,000 COCONUT natural products, and took the top hits through docking, MD, DFT, and ADMET profiling.

**What we replace:** the RandomForestClassifier (which used 2D fingerprints / descriptors) is swapped for a **Graph Convolutional Network** that learns directly on molecular graphs — atoms as nodes, bonds as edges — capturing the 3D-tolerant topology that flat fingerprints can miss. Everything else (docking, MD, DFT, ADMET) is downstream and out of scope here.

**Task:** binary classification — `active` (MDM2 inhibitor) iff `pChEMBL >= 6.0` (default threshold, ≈ 1 µM potency), else `inactive`.

### Pipeline diagram

```
                       ┌──────────────────────────────────────────────┐
                       │  ChEMBL REST API (CHEMBL5023, MDM2)          │
                       │  activity: IC50/Ki/Kd/EC50, pChEMBL not null │
                       └───────────────────┬──────────────────────────┘
                                           │ download_chembl.py
                                           ▼
                                 data/raw/chembl_mdm2.csv
                                           │ dedupe by molecule_chembl_id (max pChEMBL)
                                           │ drop NULL smiles / pChEMBL
                                           ▼
                               ┌───────────────────────────┐
                               │  label = pChEMBL >= 6.0 ? │  ← threshold_pchembl
                               └────────────┬──────────────┘
                                            ▼
                               SMILES → molecular graphs (featurize.py)
                               x [N,42] · edge_index [2,M] · edge_attr [M,10]
                                            │
                                            ▼
                              Murcko scaffold split (train/val/test)
                                            │
                    ┌───────────────────────┼────────────────────────┐
                    ▼                       ▼                         ▼
             GCN / GIN conv stack   early stopping (val loss)   ┌────────────┐
             global pooling → head  best → checkpoints/best.pt   │ new SMILES │
             outputs/test_metrics.json                           └─────┬──────┘
             outputs/plots/*.png                                          ▼
             outputs/training_history.csv                   predict.py → ranked CSV
                                                                          │
                                                                          ▼
                                                           docking → MD → DFT → ADMET
                                                              (as in the paper)
```

## What data is required

The pipeline needs one CSV of MDM2 bioactivity measurements:

| Column                 | Meaning                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `molecule_chembl_id`   | ChEMBL compound identifier (used for deduplication)                     |
| `smiles`               | Canonical SMILES of the molecule                                        |
| `standard_type`        | Assay readout type: `IC50`, `Ki`, `Kd`, or `EC50`                       |
| `standard_value`       | Raw potency value                                                       |
| `standard_units`       | Units (e.g. `nM`); converted to molar for pChEMBL                       |
| `relation`             | Comparison operator on the measurement (e.g. `=`, `<`, `>`)             |
| `pchembl_value`        | `-log10(potency in M)` — larger = more potent. **The label source**     |

Rows with NULL `pchembl_value` or NULL `smiles` are dropped; the dataset is deduplicated by `molecule_chembl_id` keeping the row with the max `pchembl_value`.

### Where to get it

**Automatic (recommended):** the pipeline downloads it for you. If `data/raw/chembl_mdm2.csv` is missing, `src.data.dataset.load_datasets` (or the `download` step in the scripts / notebook) calls `download_mdm2_activity` which queries the ChEMBL REST API:

```
GET https://www.ebi.ac.uk/chembl/api/data/activity
    target_chembl_id=CHEMBL5023
    standard_type__in=IC50,Ki,Kd,EC50
    pchembl_value__isnull=false
    format=json
```

**Manual fallback:** if you prefer a browser download,

1. Open <https://www.ebi.ac.uk/chembl/>
2. Search `MDM2` → open the target page for **CHEMBL5023** (E3 ubiquitin-protein ligase Mdm2).
3. Go to the **Bioactivities** tab; filter `Assay Type = B` (binding), `Standard Type` in `IC50, Ki, Kd, EC50`, and `pChEMBL` not null.
4. Use **Export → CSV**, then place the file at `data/raw/chembl_mdm2.csv`.

Either way the CSV must contain exactly the columns listed above (or use the auto-downloader, which writes them).

## References

- **Kipf, T. N. & Welling, M. (2017).** *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017. arXiv:1609.02907. — The GCN layer (GCNConv) that forms the backbone of our classifier.
- **Xu, K., Hu, W., Leskovec, J. & Jegelka, S. (2019).** *How Powerful are Graph Neural Networks?* ICLR 2019. arXiv:1810.00826. — Introduces GIN (also implemented here, `model_type: gin`) and the theoretical expressiveness of message-passing GNNs.
- **Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O. & Dahl, G. E. (2017).** *Neural Message Passing for Quantum Chemistry.* ICML 2017. arXiv:1704.01212. — Formalizes message passing on molecular graphs, the framework our GCN instantiates.
- **Duvenaud, D., Maclaurin, D., Aguilera-Iparraguirre, J., Gómez-Bombarelli, R., Hirzel, T., Aspuru-Guzik, A. & Adams, R. P. (2015).** *Convolutional Networks on Graphs for Learning Molecular Fingerprints.* NeurIPS 2015. arXiv:1509.09292. — Pioneering graph convolutions for molecular fingerprints; direct intellectual ancestor of molecular GCNs.
- **Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K. & Pande, V. (2018).** *MoleculeNet: a benchmark for molecular machine learning.* Chemical Science 9(2), 513–530. DOI: 10.1039/C8SC00175J. — Benchmark suite and best-practice guidance (splits, metrics) for molecular ML, which informs our scaffold split and ROC/PR-AUC evaluation.
- **Budha, S., et al. (2026).** *Machine Learning-Guided Discovery of Natural MDM2 Inhibitors: A Multistage In Silico Pipeline from Screening to ADMET Profiling.* Advanced Theory and Simulations. DOI: 10.1002/adts.202501502. — The paper this project replicates; we replace its RandomForestClassifier screening step with a GCN.

## Setup

Requires **Python 3.10–3.13** (3.13 verified working; 3.10–3.12 are the safest for wheel availability on some systems). The GPU stack needs a CUDA-enabled PyTorch build.

```bash
# 1. Clone
git clone <your-repo-url> gcn-code && cd gcn-code

# 2. Create an environment (conda recommended)
conda create -n mdm2gcn python=3.11 -y
conda activate mdm2gcn

# 3. Install dependencies (CPU build)
pip install -r requirements.txt

# GPU machines: install a CUDA PyTorch wheel first, then the rest
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The machine this was developed on already ships everything in its base env
(Python 3.13, torch 2.13.0 CPU, PyTorch Geometric 2.8.0.post1, RDKit 2026.03.5).

## Usage — CPU

Fastest path (download → train → evaluate → predict, uses `config/cpu.yaml`):

```bash
bash scripts/run_cpu.sh
```

Or step by step from the repo root:

```bash
mkdir -p data/raw outputs checkpoints
python -m src.data.download_chembl --config config/cpu.yaml   # fetch data (skips if cached)
python -m src.train      --config config/cpu.yaml             # train + validate
python -m src.evaluate   --config config/cpu.yaml             # score the test split
python -m src.predict    --config config/cpu.yaml \
    --smiles_file data/raw/example_smiles.csv \
    --output outputs/predictions.csv                          # rank new molecules
```

Example training output (abbreviated):

```
Epoch 001/060 | loss 0.6841 acc 0.602 | val_loss 0.6590 val_acc 0.631 val_auc 0.712
Epoch 002/060 | loss 0.6210 acc 0.671 | val_loss 0.5901 val_acc 0.694 val_auc 0.761
...
Best val loss 0.3721 at epoch 37 — early stopping patience 10
Test: accuracy 0.82 | roc_auc 0.91 | pr_auc 0.88 | precision 0.80 | recall 0.85 | f1 0.82
```

## Usage — GPU

```bash
bash scripts/run_gpu.sh            # device: cuda, auto-fallback to cpu if unavailable
```

`config/gpu.yaml` uses a bigger model (hidden 256, 4 layers, dropout 0.3) and larger batches (256) to exploit GPU parallelism; `num_workers` can be raised to 4–8 on GPU machines to keep the data pipeline ahead of the GPU. See the script header for SLURM/conda invocation examples.

## Usage — Google Colab

Open **`notebooks/colab_mdm2_gcn.ipynb`** in Colab (File → Upload notebook, or push the repo and open from GitHub). It detects the runtime (GPU or CPU), installs dependencies, gets the repo, downloads the data, trains with the correct device config, evaluates, and predicts on the example molecules. For a GPU runtime use **Runtime → Change runtime type → GPU**.

## Project structure

```
.
├── SPEC.md                        # interface contract for all modules
├── README.md                      # this file
├── requirements.txt
├── config/
│   ├── cpu.yaml                   # device: cpu, small model
│   └── gpu.yaml                   # device: cuda (fallback handled), larger model
├── scripts/
│   ├── run_cpu.sh                 # download → train → evaluate → predict (CPU)
│   └── run_gpu.sh                 # same, GPU config (+ SLURM/conda hints)
├── src/
│   ├── utils.py                   # config loading, seeding, metrics, EarlyStopping
│   ├── data/
│   │   ├── download_chembl.py     # ChEMBL REST API → raw CSV (CHEMBL5023)
│   │   ├── featurize.py           # SMILES → PyG graphs (42 atom / 10 bond feats)
│   │   └── dataset.py             # download → featurize → scaffold split
│   ├── models/
│   │   └── gcn.py                 # GCNClassifier (GCNConv / GINConv stack)
│   ├── train.py                   # CLI: train + validate, save best checkpoint
│   ├── evaluate.py                # CLI: score test split → metrics JSON + plots
│   └── predict.py                 # CLI: score new SMILES → ranked CSV
├── notebooks/
│   └── colab_mdm2_gcn.ipynb       # end-to-end Colab runbook
├── data/
│   ├── raw/                       # chembl_mdm2.csv, example_smiles.csv
│   └── processed/                 # cached graphs (optional)
├── checkpoints/                   # best_model.pt
└── outputs/                       # metrics, plots, predictions
```

## Results interpretation & next steps

- **Metrics:** accuracy, **ROC-AUC** and **PR-AUC** (the paper's primary discriminative metrics), precision/recall/F1, plus confusion matrix and training-history CSV — all under `outputs/`.
- **Class imbalance:** handled via `pos_weight` (CrossEntropy) and reported with PR-AUC, which is robust to imbalance.
- **Next steps (as in the paper):** rank your library with `predict.py`, then feed the top hits into **molecular docking** (e.g. AutoDock Vina/Glide) against MDM2, validate the best complexes with **molecular dynamics**, refine with **DFT**, and finish with **ADMET profiling**.

## License & disclaimer

Code is MIT licensed. **Research use only** — this is a screening prior trained on public bioactivity data, not a certified diagnostic. A high predicted probability means the molecule's graph resembles known MDM2 inhibitors; it does *not* guarantee binding, efficacy, or safety. Any candidate must be validated computationally (docking/MD) and experimentally before further consideration.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| ChEMBL download hangs / HTTP 5xx | Transient server load or blocked request | Retry with a delay; the downloader caches to `data/raw/chembl_mdm2.csv`, so re-runs skip it. Set a polite `User-Agent`. |
| `torch_geometric` version mismatch / missing `torch-scatter` | PyG wheel mismatch with the installed torch build | Upgrade everything together: `pip install -U torch torch-geometric` (PyG ≥ 2.4 bundles ops); on GPU, reinstall torch with `--index-url https://download.pytorch.org/whl/cu121` first. |
| `RuntimeError: CUDA not available` at train time | `device: cuda` but no GPU / wrong torch build | `get_device` auto-falls back to CPU with a warning; or reinstall the CUDA torch wheel, or switch to `config/cpu.yaml`. |
| `Some of your tests do not use a random device...` | torch-geometric metatest warning | Cosmetic; ignore, or pin `torch` to a release ≤ 2.3. |
| `ValueError: empty smiles` / failed featurization | Bad SMILES rows | Already dropped at download; for your own files, pre-filter with RDKit `Chem.MolFromSmiles`. |
| Slow CPU training | Large batch on a small model | Use `config/cpu.yaml` defaults (batch 32, hidden 128); reduce `epochs` if needed. |
| `ModuleNotFoundError: src` | Script run from outside the repo root | Run all `python -m src.*` commands from the repo root (see setup). |