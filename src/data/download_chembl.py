"""Download MDM2 inhibitor activity data from the ChEMBL REST API."""

from __future__ import annotations

import os

import pandas as pd
import requests

TARGET_CHEMBL_ID = "CHEMBL5023"
STANDARD_TYPES = "IC50,Ki,Kd,EC50"
BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/activity"
PAGE_SIZE = 1000
USER_AGENT = "GCN-MDM2-pipeline/1.0 (research; contact: local)"
COLUMNS = [
    "molecule_chembl_id",
    "smiles",
    "standard_type",
    "standard_value",
    "standard_units",
    "relation",
    "pchembl_value",
]


def _fetch_page(session: requests.Session, offset: int) -> list[dict]:
    params = {
        "target_chembl_id": TARGET_CHEMBL_ID,
        "standard_type__in": STANDARD_TYPES,
        "pchembl_value__isnull": "false",
        "format": "json",
        "limit": PAGE_SIZE,
        "offset": offset,
    }
    resp = session.get(BASE_URL, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json().get("activities", [])


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if "smiles" not in df.columns and "canonical_smiles" in df.columns:
        df = df.rename(columns={"canonical_smiles": "smiles"})
    df = df[df["smiles"].notna() & df["pchembl_value"].notna()]
    df = df.loc[df.groupby("molecule_chembl_id")["pchembl_value"].idxmax()]
    df = df[COLUMNS].reset_index(drop=True)
    return df


def download_mdm2_activity(out_path: str, use_cache: bool = True) -> pd.DataFrame:
    """Download MDM2 (CHEMBL5023) pChEMBL activities and cache them as CSV.

    If ``use_cache`` and ``out_path`` already exists, load and return it.
    """
    if use_cache and os.path.exists(out_path):
        return pd.read_csv(out_path)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    rows: list[dict] = []
    offset = 0
    while True:
        page = _fetch_page(session, offset)
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("ChEMBL API returned no activities for MDM2.")

    df = _clean_df(df)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Downloaded {len(df)} unique MDM2 activities -> {out_path}")
    return df


def _main() -> None:
    """CLI: download (or reload cached) MDM2 data to cfg['data']['raw_csv']."""
    import argparse

    from src.utils import load_config

    parser = argparse.ArgumentParser(description="Download MDM2 bioactivity data from ChEMBL.")
    parser.add_argument("--config", default="config/cpu.yaml", help="Path to YAML config.")
    parser.add_argument("--out", default=None, help="Output CSV path (overrides config).")
    parser.add_argument("--no-cache", action="store_true", help="Force a fresh download.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = args.out or cfg.get("data", {}).get("raw_csv", "data/raw/chembl_mdm2.csv")
    df = download_mdm2_activity(out, use_cache=not args.no_cache)
    print(f"{len(df)} rows ready at {out}")


if __name__ == "__main__":
    _main()