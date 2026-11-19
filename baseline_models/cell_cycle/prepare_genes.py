"""Resolve tinyatlas Ensembl IDs to gene symbols via the mygene.info API."""

import json
import os

import mygene
import pandas as pd

ATLAS_FILE = os.path.join(os.path.dirname(__file__), "tinyatlas_human_cell_cycle.csv")
OUT_FILE = os.path.join(os.path.dirname(__file__), "cell_cycle_genes.json")


def main() -> None:
    df = pd.read_csv(ATLAS_FILE)

    print(f"  {len(df)} genes loaded. Resolving Ensembl IDs to symbols via MyGene.info...")
    mg = mygene.MyGeneInfo()
    symbols = mg.querymany(
        df["geneID"].tolist(),
        scopes="ensembl.gene",
        fields="symbol",
        verbose=False,
    )
    df["symbol"] = [x.get("symbol", "") for x in symbols]

    s_genes = df[df["phase"] == "S"]["symbol"].dropna().tolist()
    g2m_genes = df[df["phase"] == "G2/M"]["symbol"].dropna().tolist()

    result = {"s_genes": s_genes, "g2m_genes": g2m_genes}
    with open(OUT_FILE, "w") as f:
        json.dump(result, f)

    total = len(set(s_genes + g2m_genes))
    print(f"  Saved {len(s_genes)} S-phase and {len(g2m_genes)} G2/M-phase genes "
          f"({total} unique) to {OUT_FILE}")


if __name__ == "__main__":
    main()
