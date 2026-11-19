"""TFLink GRN baseline.

This script is adapted from @Poirion's TFLink baseline notebook, located at
https://gitlab.com/jaxcomp/tutorial_dream_challenge.

Constructs a GRN using the TFLink database of experimentally validated
TF–target interactions. Two methods are available and selected via --method:
  Tflink_SS  — Small-Scale (SS) data: high-confidence interactions validated
               by targeted experiments. High precision, limited coverage.
  Tflink_LS  — Large-Scale (LS) data: genome-wide interactions inferred from
               high-throughput assays (ChIP-seq, motif analysis). Broad
               coverage, lower confidence.

All edges receive a uniform weight of 0.5.
"""

import argparse
import pandas as pd

from collections import defaultdict
from pathlib import Path


VALID_METHODS = ("Tflink_SS", "Tflink_LS")


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
        default="Tflink_SS",
        choices=VALID_METHODS,
        help="TFLink dataset to use: SS (small-scale) or LS (large-scale)",
    )
    return parser.parse_args()


def load_gmt(gmt_path: Path) -> dict[str, set[str]]:
    """Parse a GMT file into a dict of {TF: set_of_target_genes}."""
    tf_targets: dict[str, set[str]] = {}
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            tf = parts[0]
            # GMT format: TF <tab> description <tab> gene1 <tab> gene2 ...
            # TFLink GMT omits the description field; first token is TF name.
            tf_targets[tf] = set(parts[1:])
    return tf_targets


def main() -> None:
    args = setup_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gmt_file = Path(__file__).parent / (
        "tflink_SS.gmt" if args.method == "Tflink_SS" else "tflink_LS.gmt"
    )

    print(f"\n--- Step 1: Load TFLink {args.method} GMT ---")
    tf_targets = load_gmt(gmt_file)
    print(f"  {len(tf_targets)} TFs in TFLink {args.method}")

    print("\n--- Step 2: Load TF list ---")
    tf_list = (
        pd.read_table(input_dir / "tf_list.refined.tsv", header=None)
        .iloc[:, 0]
        .astype(str)
        .tolist()
    )
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
    tfs_with_interactions = 0
    for tf in tf_list:
        targets = tf_targets.get(tf, set())
        if not targets:
            continue
        tfs_with_interactions += 1
        for celltype in cell_types:
            for target in targets:
                res["source"].append(tf)
                res["target"].append(target)
                res["weight"].append(0.5)
                res["cell_type"].append(celltype)

    print(f"  {tfs_with_interactions}/{len(tf_list)} challenge TFs found in TFLink {args.method}")

    predictions = pd.DataFrame(res)

    out_path = output_dir / "predictions.tsv"
    print(f"\n--- Step 5: Output results ---")
    print(f"  Saving {len(predictions)} edges to {out_path}...")
    predictions.to_csv(out_path, sep="\t", index=False)
    print("  Done.")


if __name__ == "__main__":
    main()
