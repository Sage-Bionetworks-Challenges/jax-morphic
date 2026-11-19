"""Cell-cycle GRN baseline.

This script is adapted from @Poirion's cell-cycle baseline notebook, located
at https://gitlab.com/jaxcomp/tutorial_dream_challenge.

Constructs a GRN by linking every TF to every cell-cycle gene (S-phase and
G2/M-phase) with a uniform weight of 1.0 for each cell type, restricted to
the precomputed top10Kfeatures.ygi list.
"""

import argparse
import json
import pandas as pd

from collections import defaultdict
from pathlib import Path


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/input",
        help="Path to directory containing input files",
    )
    parser.add_argument(
        "--output_dir",
        default="/output",
        help="Path to directory where predictions.tsv is written",
    )
    return parser.parse_args()


def load_cell_cycle_genes(genes_file: Path, top_feats_file: Path) -> list[str]:
    """Load the pre-built cell-cycle gene list, intersected with top10Kfeatures.ygi."""
    with open(genes_file) as f:
        data = json.load(f)
    s_genes = data.get("s_genes", [])
    g2m_genes = data.get("g2m_genes", [])
    cycle_genes = set(s_genes + g2m_genes)

    top_feats = set(
        pd.read_table(top_feats_file, header=None).iloc[:, 0].astype(str)
    )
    cycle_genes = sorted(cycle_genes & top_feats)
    print(f"  {len(cycle_genes)} cell-cycle genes remain after intersecting with "
          f"top10Kfeatures.ygi ({len(s_genes)} S-phase, {len(g2m_genes)} G2/M-phase before intersection)")
    return cycle_genes


def main() -> None:
    args = setup_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genes_file = Path(__file__).parent / "cell_cycle_genes.json"

    print("\n--- Step 1: Load cell-cycle gene list ---")
    cycle_genes = load_cell_cycle_genes(genes_file, input_dir / "top10Kfeatures.ygi")

    print("\n--- Step 2: Load TF list ---")
    tf_list = (
        pd.read_table(input_dir / "tf_list.refined.tsv", header=None)
        .iloc[:, 0]
        .astype(str)
        .tolist()
    )
    print(f"  {len(tf_list)} TFs loaded")

    print("\n--- Step 3: Load cell type labels ---")
    cell_types = sorted(
        pd.read_table(input_dir / "resolved.labels.tsv", index_col=0)
        .iloc[:, 0]
        .astype(str)
        .unique()
    )
    print(f"  {len(cell_types)} cell types found")

    print("\n--- Step 4: Build GRN ---")
    res: dict[str, list] = defaultdict(list)
    for celltype in cell_types:
        for tf in tf_list:
            for gene in cycle_genes:
                res["source"].append(tf)
                res["target"].append(gene)
                res["weight"].append(1.0)
                res["cell_type"].append(celltype)

    predictions = pd.DataFrame(res)

    out_path = output_dir / "predictions.tsv"
    print(f"\n--- Step 5: Output results ---")
    print(f"  Saving {len(predictions)} edges to {out_path}...")
    predictions.to_csv(out_path, sep="\t", index=False)
    print("  Done.")


if __name__ == "__main__":
    main()
