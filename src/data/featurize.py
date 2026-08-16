"""SMILES -> torch_geometric molecular graph featurization."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from torch_geometric.data import Data

ATOM_FEATURE_DIM = 42
BOND_FEATURE_DIM = 10

_ATOM_NUMBERS = list(range(1, 19))
_HYBRIDIZATION = {
    Chem.HybridizationType.SP: 0,
    Chem.HybridizationType.SP2: 1,
    Chem.HybridizationType.SP3: 2,
    Chem.HybridizationType.SP3D: 3,
    Chem.HybridizationType.SP3D2: 4,
}
_SMALL_RING_SIZES = (3, 4, 5, 6, 7, 8)


def get_atom_features(atom: Chem.Atom) -> list[int]:
    """One-hot feature vector of length 42 for a single atom."""
    feats: list[int] = []

    atomic_feats = [0] * 18
    atomic_num = atom.GetAtomicNum()
    if 1 <= atomic_num <= 18:
        atomic_feats[atomic_num - 1] = 1
    feats.extend(atomic_feats)

    degree_feats = [0] * 6
    degree_feats[min(atom.GetDegree(), 5)] = 1
    feats.extend(degree_feats)

    h_feats = [0] * 5
    h_feats[min(atom.GetTotalNumHs(), 4)] = 1
    feats.extend(h_feats)

    hyb_feats = [0] * 5
    hyb_feats[_HYBRIDIZATION.get(atom.GetHybridization(), 4)] = 1
    feats.extend(hyb_feats)

    charge_feats = [0] * 3
    charge_feats[max(-1, min(atom.GetFormalCharge(), 1)) + 1] = 1
    feats.extend(charge_feats)

    feats.append(int(atom.GetIsAromatic()))
    feats.append(int(atom.IsInRing()))
    feats.append(int(any(atom.IsInRingSize(s) for s in _SMALL_RING_SIZES)))
    feats.append(int(atom.GetNumRadicalElectrons() > 0))
    feats.append(int(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED))

    return feats


def get_bond_features(bond: Chem.Bond) -> list[int]:
    """One-hot feature vector of length 10 for a single bond."""
    feats = [0] * 10

    bond_type = bond.GetBondType()
    type_idx = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }.get(bond_type, 0)
    feats[type_idx] = 1

    feats[4] = int(bond.GetIsConjugated())
    feats[5] = int(bond.IsInRing())

    stereo = bond.GetStereo()
    if stereo == Chem.BondStereo.STEREONONE:
        feats[6] = 1
    elif stereo == Chem.BondStereo.STEREOANY:
        feats[7] = 1
    else:
        feats[8] = 1

    feats[9] = int(bond.GetIsAromatic())
    return feats


def mol_to_graph(mol: Chem.Mol) -> Optional[Data]:
    """Convert an RDKit molecule into a torch_geometric Data graph."""
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    x = np.zeros((num_atoms, ATOM_FEATURE_DIM), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        x[i] = get_atom_features(atom)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i == j:
            continue
        feats = get_bond_features(bond)
        edge_index.append((i, j))
        edge_attr.append(feats)
        edge_index.append((j, i))
        edge_attr.append(feats)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)

    return Data(x=torch.tensor(x, dtype=torch.float32), edge_index=edge_index, edge_attr=edge_attr)


def smiles_to_data(
    smiles: str,
    pchembl: Optional[float],
    label: Optional[int],
) -> Optional[Data]:
    """Build a molecular graph Data from a SMILES string; None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    data = mol_to_graph(mol)
    if data is None:
        return None
    data.smiles = smiles
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)
    if pchembl is not None:
        data.y_pchembl = torch.tensor([pchembl], dtype=torch.float)
    return data


def build_graph_dataset(
    df: pd.DataFrame,
    threshold: float = 6.0,
) -> tuple[list[Data], np.ndarray, np.ndarray]:
    """Convert a SMILES/pchembl DataFrame into graphs, labels and pchembl arrays."""
    graphs: list[Data] = []
    labels: list[int] = []
    pchembl_values: list[float] = []
    for _, row in df.iterrows():
        smiles = row["smiles"]
        pchembl = float(row["pchembl_value"])
        label = 1 if pchembl >= threshold else 0
        g = smiles_to_data(smiles, pchembl, label)
        if g is None:
            continue
        graphs.append(g)
        labels.append(label)
        pchembl_values.append(pchembl)
    return graphs, np.asarray(labels, dtype=np.int64), np.asarray(pchembl_values, dtype=np.float64)


def scaffold_split(
    graphs: list[Data],
    test_ratio: float = 0.2,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Data], list[Data], list[Data]]:
    """Split graphs by Murcko scaffold group so groups stay wholly in one split."""
    scaffold_to_indices: dict[str, list[int]] = {}
    for i, g in enumerate(graphs):
        smiles = getattr(g, "smiles", None)
        if smiles is None:
            raise ValueError("Data object has no 'smiles' attribute; cannot scaffold split.")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            scaffold = smiles
        else:
            try:
                scaffold = MurckoScaffoldSmiles(mol)
            except Exception:
                scaffold = smiles
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    index_sets = list(scaffold_to_indices.values())
    random.Random(seed).shuffle(index_sets)

    n_total = len(graphs)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_test - n_val

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for idx_set in index_sets:
        if len(test_idx) + len(idx_set) <= n_test:
            test_idx.extend(idx_set)
        elif len(val_idx) + len(idx_set) <= n_val:
            val_idx.extend(idx_set)
        else:
            train_idx.extend(idx_set)

    train = [graphs[i] for i in train_idx]
    val = [graphs[i] for i in val_idx]
    test = [graphs[i] for i in test_idx]
    return train, val, test