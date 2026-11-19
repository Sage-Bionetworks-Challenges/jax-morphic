"""DoRothEA GRN baseline.

This script is adapted from @Poirion's DoRothEA baseline notebook, located at
https://gitlab.com/jaxcomp/tutorial_dream_challenge.

Constructs a GRN by looking up each TF in the DoRothEA curated regulon
database and linking it to its target genes using the MoR (Mode of
Regulation) score as edge weight. DoRothEA is a curated collection of
TF-target gene interactions organized into regulons with confidence scores
(A-E) based on supporting evidence, where the network is bundled in the
image as ctri_dorothea.csv.gz.
"""

import argparse
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


def main() -> None:
    args = setup_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    net_file = Path(__file__).parent / "ctri_dorothea.csv.gz"

    print("\n--- Step 1: Load DoRothEA network ---")
    net = pd.read_csv(net_file)  # columns: source, target, mor
    net_by_tf = net.groupby("source")
    print(f"  {len(net)} interactions across {net['source'].nunique()} TFs loaded")

    print("\n--- Step 2: Load TF list ---")
    tf_list = (
        pd.read_table(input_dir / "tf_list.refined.tsv", header=None)
        .iloc[:, 0]
        .astype(str)
        .tolist()
    )
    tf_set = set(tf_list)
    print(f"  {len(tf_list)} TFs in challenge TF list")

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
    tfs_with_regulon = 0
    for tf in tf_list:
        if tf not in net_by_tf.groups:
            continue
        tfs_with_regulon += 1
        regulon = net_by_tf.get_group(tf)
        for celltype in cell_types:
            for _, row in regulon.iterrows():
                res["source"].append(tf)
                res["target"].append(row["target"])
                res["weight"].append(float(row["mor"]))
                res["cell_type"].append(celltype)

    print(f"  {tfs_with_regulon}/{len(tf_list)} challenge TFs found in DoRothEA")

    predictions = pd.DataFrame(res)

    out_path = output_dir / "predictions.tsv"
    print(f"\n--- Step 5: Output results ---")
    print(f"  Saving {len(predictions)} edges to {out_path}...")
    predictions.to_csv(out_path, sep="\t", index=False)
    print("  Done.")


if __name__ == "__main__":
    main()
