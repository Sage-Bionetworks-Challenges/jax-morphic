"""Correlation-based GRN inference from single-cell RNA counts.

This script is adapted from @Poirion's correlation baseline notebook, located
at https://gitlab.com/jaxcomp/tutorial_dream_challenge.

Three methods are available and selected via --method:
  max_correlation  — one edge per TF: the single target with the highest
                     absolute Pearson correlation.
  all_correlation  — one edge per TF-target pair: all targets linked to each
                     TF using the Pearson correlation score as weight.
  top50pos_corr    — up to 50 edges per TF: the top-50 targets ranked by
                     positive Pearson correlation.

Candidate genes are restricted to the precomputed top10Kfeatures.ygi list.
"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc

from collections import defaultdict
from pathlib import Path
from sklearn.metrics.pairwise import pairwise_distances


VALID_METHODS = ("max_correlation", "all_correlation", "top50pos_corr")


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
        default="all_correlation",
        choices=VALID_METHODS,
        help="Correlation method to use for GRN inference",
    )
    return parser.parse_args()


def load_data(input_dir: Path) -> tuple[sc.AnnData, set[str]]:
    """Load expression data, cell type labels, and TF set from input_dir."""
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
    tf_set = set(
        pd.read_table(input_dir / "tf_list.refined.tsv", header=None)
        .iloc[:, 0]
        .astype(str)
        .tolist()
    )
    return adata, tf_set


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


def compute_correlations(
    adata: sc.AnnData,
    tf_set: set[str],
    method: str,
) -> pd.DataFrame:
    """Compute per-cell-type correlations between TFs and target genes."""
    is_tf = adata.var.index.isin(tf_set)
    is_target = ~is_tf

    tf_list = adata.var.index[is_tf].values
    targets_list = adata.var.index[is_target].values

    print(f"  {is_tf.sum()} TFs and {is_target.sum()} target genes after feature selection")

    cell_types = sorted(adata.obs["Labels2"].dropna().astype(str).unique())
    all_results = []

    for celltype in cell_types:
        print(f"  Cell type: {celltype}")
        mask = (adata.obs["Labels2"].astype(str) == celltype).values

        mat_cell = adata.X[mask]
        mat_tf = mat_cell[:, is_tf]
        mat_target = mat_cell[:, is_target]

        # pairwise_distances with "correlation" returns 1 - Pearson_r, range [0, 2].
        # Subtracting 1 centres to [-1, 1] where -1 = perfect positive correlation.
        res_cell = (
            pairwise_distances(
                mat_tf.T.toarray(),
                mat_target.T.toarray(),
                metric="correlation",
            )
            - 1.0
        )
        res_cell = np.nan_to_num(res_cell, nan=0)

        res: dict[str, list] = defaultdict(list)

        for tf, vect in zip(tf_list, res_cell):
            if method == "max_correlation":
                target_max = int(np.argmax(np.abs(vect)))
                res["source"].append(tf)
                res["target"].append(targets_list[target_max])
                res["weight"].append(float(vect[target_max]))
                res["cell_type"].append(celltype)

            elif method == "top50pos_corr":
                top50_idx = np.argsort(vect)[-50:]
                for tid in top50_idx:
                    res["source"].append(tf)
                    res["target"].append(targets_list[tid])
                    res["weight"].append(float(vect[tid]))
                    res["cell_type"].append(celltype)

            elif method == "all_correlation":
                for target, score in zip(targets_list, vect):
                    res["source"].append(tf)
                    res["target"].append(target)
                    res["weight"].append(float(score))
                    res["cell_type"].append(celltype)

        all_results.append(pd.DataFrame(res))

    return pd.concat(all_results, ignore_index=True)


def main() -> None:
    args = setup_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Step 1: Load data ---")
    adata, tf_set = load_data(input_dir)

    print(f"\n--- Step 2: Preprocess ---")
    adata = preprocess(adata, input_dir)

    print(f"\n--- Step 3: Compute correlations (method={args.method}) ---")
    predictions = compute_correlations(adata, tf_set, args.method)

    out_path = output_dir / "predictions.tsv"
    print(f"\n--- Step 4: Output results ---")
    print(f"  Saving {len(predictions)} edges to {out_path}...")
    predictions.to_csv(out_path, sep="\t", index=False)
    print("  Done.")


if __name__ == "__main__":
    main()
