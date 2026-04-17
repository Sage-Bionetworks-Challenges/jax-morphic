"""Run GRN inference from single-cell input files.

This script is adapted from @Poirion's GRNBoost2/SCENIC notebook, located
at https://gitlab.com/jaxcomp/tutorial_dream_challenge. Input files are
expected in input_dir, and final predictions file should be saved into
output_dir.
"""

import argparse
import pandas as pd
import scanpy as sc

from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from distributed import Client, LocalCluster
from pathlib import Path
from pyscenic.prune import df2regulons, prune2df
from pyscenic.utils import modules_from_adjacencies

DEBUG = True  # For testing locally. Set to False for final run to avoid saving large intermediate files.
Z_SCORE_CUTOFF = 2.0  # Arbitrary cutoff for filtering GRNBoost2 edges by importance


def load_data(
    input_dir: Path,
) -> tuple[sc.AnnData, list[str]]:
    """Load expression data, cell labels, and TF list from input_dir."""
    print("\tLoading expression data...")
    adata = sc.read_h5ad(input_dir / "df.H1diff.JointUMAP.rna.counts.h5ad")

    print("\tLoading cell metadata and labels...")
    meta = pd.read_csv(input_dir / "meta.all.tsv.gz", sep="\t", index_col=0)
    labels = pd.read_csv(input_dir / "resolved.labels.tsv", sep="\t", index_col=0).iloc[
        :, 0
    ]

    common_cells = adata.obs_names.intersection(labels.index)
    adata = adata[common_cells].copy()
    adata.obs = meta.reindex(adata.obs_names)
    adata.obs["Labels2"] = labels.reindex(adata.obs_names)

    print("\tLoading TF list...")
    tf_names = (
        pd.read_csv(input_dir / "tf_list.refined.tsv", sep="\t", header=None)
        .iloc[:, 0]
        .dropna()
        .astype(str)
        .tolist()
    )
    return adata, tf_names


def preprocess(
    adata: sc.AnnData, tf_names: list[str], n_feats: int
) -> tuple[sc.AnnData, list[str], list[str]]:
    """Normalize counts, select top-variance genes, and intersect with TF list."""
    sc.pp.normalize_total(adata)  # normalize by median total counts
    sc.pp.log1p(adata)

    print(f"\tSelecting top {n_feats} genes by variance...")

    # Calculate highly variable genes on log-normalized data, then filter.
    sc.pp.highly_variable_genes(adata, n_top_genes=n_feats, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    gene_names = adata.var["features"].values.tolist()

    tfs = [tf for tf in tf_names if tf in gene_names]
    print(f"\tFound {len(tfs)} TFs in the top {n_feats} genes")
    return adata, gene_names, tfs


def run_grnboost2(
    adata: sc.AnnData,
    gene_names: list[str],
    tfs: list[str],
    nthreads: int,
    grn_dir: Path = Path("grnboost2"),
) -> dict[str, pd.DataFrame]:
    """Run GRNBoost2 per cell type and filter edges by z-score cutoff."""
    adj_by_ct: dict[str, pd.DataFrame] = {}
    cell_types = sorted(adata.obs["Labels2"].dropna().astype(str).unique())

    if DEBUG:
        grn_dir.mkdir(parents=True, exist_ok=True)

    client = Client(LocalCluster(n_workers=nthreads, threads_per_worker=1))
    try:
        for cell_type in cell_types:
            mask = adata.obs["Labels2"].astype(str) == cell_type
            print(f"\tGRNBoost2: {cell_type} ({mask.sum()} cells)...")
            adj = grnboost2(
                expression_data=adata.X[mask.values],
                gene_names=gene_names,
                tf_names=tfs,
                client_or_address=client,
            )
            cutoff = Z_SCORE_CUTOFF * adj["importance"].std()
            adj = adj[adj["importance"] > cutoff]

            if DEBUG:
                adj.to_csv(grn_dir / f"{cell_type}.grn.tsv.gz", sep="\t", index=False)
            adj_by_ct[cell_type] = adj
    finally:
        client.close()
    return adj_by_ct


def run_scenic(
    adata: sc.AnnData,
    gene_names: list[str],
    adj_by_ct: dict[str, pd.DataFrame],
    input_dir: Path,
) -> pd.DataFrame:
    """Run RcisTarget motif enrichment per cell type using both motif databases."""
    dbs = [
        RankingDatabase(
            fname=str(
                input_dir
                / "hg38__refseq-r80__10kb_up_and_down_tss.mc9nr.genes_vs_motifs.rankings.feather"
            ),
            name="hg38__refseq-r80__10kb_up_and_down_tss.mc9nr",
        ),
        RankingDatabase(
            fname=str(
                input_dir
                / "hg38__refseq-r80__500bp_up_and_100bp_down_tss.mc9nr.genes_vs_motifs.rankings.feather"
            ),
            name="hg38__refseq-r80__500bp_up_and_100bp_down_tss.mc9nr",
        ),
    ]
    motif_annotations = str(input_dir / "motifs-v9-nr.hgnc-m0.001-o0.0.tbl")

    results = []
    for celltype, adj in adj_by_ct.items():
        mask = adata.obs["Labels2"].astype(str) == celltype
        print(f"\tSCENIC: {celltype}...")
        expr_df = pd.DataFrame.sparse.from_spmatrix(
            adata.X[mask.values], columns=gene_names
        )
        print("\tStarting regulon pruning, this may take a few minutes...")
        modules = list(modules_from_adjacencies(adj, expr_df))
        pruned = prune2df(dbs, modules, motif_annotations)
        if pruned.empty:
            print(f"\tNo regulons found for {celltype}, skipping...")
            continue
        regulons = df2regulons(pruned)
        print(
            f"\tFound {len(pruned)} regulons for {celltype}, converting to edge list..."
        )
        for regulon in regulons:
            tf_name = regulon.name.replace("(+)", "")
            for gene, score in regulon.gene2weight.items():
                results.append([tf_name, gene, score, celltype, "scenic"])
    return pd.DataFrame(results, columns=["source", "target", "weight", "CT", "method"])


def main():
    parser = argparse.ArgumentParser(
        description="Run GRN inference from single-cell input files."
    )
    parser.add_argument(
        "--input_dir", default="/input", help="Path to directory containing input files"
    )
    parser.add_argument(
        "--output_dir",
        default="/output",
        help="Path to directory where predictions.csv is written",
    )
    parser.add_argument(
        "--nthreads", type=int, default=8, help="Number of workers for GRNBoost2"
    )
    parser.add_argument(
        "--n_feats",
        type=int,
        default=10_000,
        help="Number of top-variance genes to keep",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    nthreads = args.nthreads
    n_feats = args.n_feats
    output_dir = Path(output_dir)

    print("\n--- Step 1: Data Loading ---")
    adata, tf_names = load_data(Path(input_dir))

    print("\n--- Step 2: Preprocessing ---")
    adata, gene_names, tfs = preprocess(adata, tf_names, n_feats)

    print("\n--- Step 3: GRNBoost2 co-expression inference (per cell type) ---")
    adj_by_ct = run_grnboost2(
        adata,
        gene_names,
        tfs,
        nthreads,
        output_dir / "grnboost2",
    )

    print("\n--- Step 4: RcisTarget motif enrichment (per cell type) ---")
    predictions = run_scenic(adata, gene_names, adj_by_ct, Path(input_dir))

    print(f"\n--- Step 5: Saving results to {output_dir} ---")
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    print(f"Saved {len(predictions)} edges to predictions.csv.")

    print(f"\n--- Inference done ---")


if __name__ == "__main__":
    main()
