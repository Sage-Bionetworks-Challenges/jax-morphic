#!/usr/bin/env python

"""Scoring script.

Converts the submitted predictions TSV to the .tsv.gz format expected by
the scoring pipeline, then hands it off for scoring.

"""

import argparse
import gzip
import json
import shutil


def compress_to_tsv_gz(filepath, out_path="predictions.tsv.gz"):
    """Gzip-compress the submitted predictions file to .tsv.gz, as expected
    by the scoring module.
    """
    with open(filepath, "rb") as f_in, gzip.open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path


def main():
    """Main function."""

    pred_file = compress_to_tsv_gz(args.prediction_file)

    # TODO: pass pred_file (and args.groundtruth_file) to the scoring
    # pipeline.
    scores = None
    status = "SCORED"
    errors = ""

    result = {
        "accuracy": scores,
        "submission_status": status,
        "submission_errors": errors,
    }
    with open(args.output_file, "w") as o:
        o.write(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--prediction_file",
        required=True,
        help="Filepath to prediction TSV",
    )
    parser.add_argument(
        "-g",
        "--groundtruth_file",
        required=True,
        help="Filepath to groundtruth/goldstandard TSV",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        default="results.json",
        help="Output JSON file for scores and results",
    )
    args = parser.parse_args()
    main()
