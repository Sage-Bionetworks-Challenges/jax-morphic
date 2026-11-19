"""GRN inference based on highly expressed or highly variable genes.

This script is adapted from @Poirion's high-expression / high-variance
baseline notebook, located at https://gitlab.com/jaxcomp/tutorial_dream_challenge.

Two methods are available and selected via --method:
  High_expr  — links every TF to the top --nb_feats genes ranked by total
               expression (sum across cells) within each cell type.
  High_var   — links every TF to the top --nb_feats genes ranked by
               expression standard deviation within each cell type.

In both cases the edge weight is the per-gene expression sum (High_expr)
or std (High_var) for that cell type.

Candidate genes are restricted to the precomputed top10Kfeatures.ygi list.
"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc

from collections import defaultdict
from pathlib import Path


VALID_METHODS = ("High_expr", "High_var")


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
    parser.add_argument(
        "--method",
        default="High_var",
        choices=VALID_METHODS,
        help="Ranking criterion for target gene selection",
    )
    parser.add_argument(
        "--nb_feats",
        type=int,
        default=100,
        help="Number of top target genes to link to each TF per cell type",
    )
    return parser.parse_args()


def load_data(input_dir: Path) -> tuple[sc.AnnData, list[str]]:
    """Load expression data, cell type labels, and TF list from input_dir."""
    print("  Loading expression data...")
    adata = sc.read_h5ad(input_dir / "df.H1diff.JointUMAP.rna.counts.h5ad")
    adata.var.index = adata.var["features"]

    print("  Loading cell type labels...")
    labels = pd.read_table(
        input_dir / "resolved.labels.tsv", index_col=0
    ).iloc[:, 0]
    common_cells = adata.obs_names.intersection(labels.index)
    adata = adata[common_cells].copy()
    adata.obs["Labels2"] = labels.reindex(adata.obs_names)

    print("  Loading TF list...")
    tf_list = (
        pd.read_table(input_dir / "tf_list.refined.tsv", header=None)
        .iloc[:, 0]
        .astype(str)
        .tolist()
    )
    return adata, tf_list


def preprocess(adata: sc.AnnData, input_dir: Path) -> sc.AnnData:
    """Normalize counts and subset to the precomputed top10Kfeatures.ygi list."""
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    top_feats = (
        pd.read_table(input_dir / "top10Kfeatures.ygi", header=None)
        .iloc[:, 0]
        .astype(str)
        .tolist()
    )
    print(f"  Selecting {len(top_feats)} genes from top10Kfeatures.ygi...")
    return adata[:, adata.var.index.isin(top_feats)].copy()


def build_grn(
    adata: sc.AnnData,
    tf_list: list[str],
    method: str,
    nb_feats: int,
) -> pd.DataFrame:
    """Link each TF to the top-ranked target genes per cell type."""
    cell_types = sorted(adata.obs["Labels2"].dropna().astype(str).unique())
    all_results = []

    for celltype in cell_types:
        print(f"  Cell type: {celltype}")
        mask = (adata.obs["Labels2"].astype(str) == celltype).values
        mat_cell = adata.X[mask]

        if method == "High_expr":
            scores = np.asarray(mat_cell.sum(axis=0)).reshape(-1)
        else:  # High_var
            scores = np.asarray(mat_cell.todense().std(axis=0)).reshape(-1)

        top_idx = np.argsort(scores)[-nb_feats:]
        top_genes = adata.var.index[top_idx].values
        top_weights = scores[top_idx]

        res: dict[str, list] = defaultdict(list)
        for tf in tf_list:
            for gene, weight in zip(top_genes, top_weights):
                res["source"].append(tf)
                res["target"].append(gene)
                res["weight"].append(float(weight))
                res["cell_type"].append(celltype)

        all_results.append(pd.DataFrame(res))

    return pd.concat(all_results, ignore_index=True)


def main() -> None:
    args = setup_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Step 1: Load data ---")
    adata, tf_list = load_data(input_dir)

    print(f"\n--- Step 2: Preprocess ---")
    adata = preprocess(adata, input_dir)

    print(f"\n--- Step 3: Build GRN (method={args.method}, nb_feats={args.nb_feats}) ---")
    predictions = build_grn(adata, tf_list, args.method, args.nb_feats)

    out_path = output_dir / "predictions.tsv"
    print(f"\n--- Step 4: Output results ---")
    print(f"  Saving {len(predictions)} edges to {out_path}...")
    predictions.to_csv(out_path, sep="\t", index=False)
    print("  Done.")


if __name__ == "__main__":
    main()
