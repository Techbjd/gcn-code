"""Hit curation CLI: summarize screening output, deduplicate, and build a diverse docking set.

Takes the combined predictions CSV from src.screen (plus the library CSV for molecule
details), prints a summary, drops structural duplicates by canonical SMILES, and keeps one
molecule per Murcko scaffold to produce a manageable, diverse hit list for docking.
"""

import argparse
import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold if scaffold is not None else smiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and curate screening predictions.")
    parser.add_argument("--predictions", required=True, help="Combined output CSV from src.screen.")
    parser.add_argument("--library", help="Library CSV to join SMILES/name (optional).")
    parser.add_argument("--id_col", default="id", help="ID column in the library CSV.")
    parser.add_argument("--smiles_col", default="canonical_smiles", help="SMILES column in the library CSV.")
    parser.add_argument("--n_diverse", type=int, default=200, help="Size of the diverse docking set.")
    parser.add_argument("--outdir", default="outputs", help="Directory for output files.")
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions)
    if args.library:
        lib = pd.read_csv(args.library, usecols=[args.smiles_col])
        if args.id_col in pd.read_csv(args.library, nrows=0).columns:
            lib = pd.read_csv(args.library, usecols=[args.id_col, args.smiles_col])
        else:
            lib[args.id_col] = range(len(lib))
        df = pred.merge(lib, left_on="id", right_on=args.id_col, how="left")
        df = df.rename(columns={args.smiles_col: "canonical_smiles"})
    else:
        df = pred
    if "class" not in df.columns:
        df["class"] = (df["probability_active"] >= 0.5).astype(int)
    df = df.dropna(subset=["canonical_smiles"])

    print("=== SCREEN SUMMARY ===")
    print("total scored:", len(df))
    print("predicted hits (prob>=0.5):", int((df["class"] == 1).sum()))
    print("count of prob==1.0:", int((df["probability_active"] == 1.0).sum()))
    print("prob distribution:\n", df["probability_active"].describe().round(4).to_string())

    df = df.drop_duplicates(subset=["canonical_smiles"], keep="first")
    hits = df[df["class"] == 1].sort_values("probability_active", ascending=False)
    print("unique structures:", len(df), "| unique hits:", len(hits))

    os.makedirs(args.outdir, exist_ok=True)
    hits_path = os.path.join(args.outdir, "hits.csv")
    hits.to_csv(hits_path, index=False)
    print("saved", hits_path)

    top = hits.head(5000).copy()
    top["scaffold"] = top["canonical_smiles"].apply(_scaffold)
    diverse = top.drop_duplicates(subset=["scaffold"]).head(args.n_diverse).reset_index(drop=True)
    diverse_path = os.path.join(args.outdir, "diverse_hits.csv")
    smi_path = os.path.join(args.outdir, "diverse_hits.smi")
    diverse.to_csv(diverse_path, index=False)
    diverse[["canonical_smiles"]].to_csv(smi_path, index=False, header=False)
    print(f"diverse docking set: {len(diverse)} -> {diverse_path} / {smi_path}")


if __name__ == "__main__":
    main()