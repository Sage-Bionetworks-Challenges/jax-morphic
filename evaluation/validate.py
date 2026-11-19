#!/usr/bin/env python

"""Validate prediction TSV file.

This challenge will expect a 4-column TSV file, with at least
one prediction row. There should be no missing or empty values
in the required columns.
"""

import argparse
import csv
import json


def validate_csv(filepath, delim="\t", expected_cols=["id"]):
    """
    Checks for expected colnames, at least one row, and non-empty values
    in the TSV file. The "weight" column must also parse as a float.
    """
    errors = []
    with open(filepath) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delim)
        for colname in expected_cols:
            if colname not in reader.fieldnames:
                errors.append(f"'{colname}' is missing from the prediction file")
        if errors:
            return "\n".join(errors)

        row_count = 0
        for row_num, row in enumerate(reader, start=2):
            row_count += 1
            for colname in expected_cols:
                if not row.get(colname):
                    errors.append(f"Row {row_num}: '{colname}' is missing or empty")
            weight = row.get("weight")
            if weight:
                try:
                    float(weight)
                except ValueError:
                    errors.append(f"Row {row_num}: 'weight' value '{weight}' is not a number")

        if row_count == 0:
            errors.append("Prediction file contains no data rows")
    return "\n".join(errors)


def main():
    """Main function."""

    if args.entity_type != "FileEntity":
        errors = f"Submission should be a file, not {args.entity_type}"
    else:
        errors = validate_csv(
            args.prediction_file,
            expected_cols=[
                "source",
                "target",
                "weight",
                "cell_type",
            ],
        )

    result = {
        "submission_status": "INVALID" if errors else "VALIDATED",
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
        "-e",
        "--entity_type",
        default="FileEntity",
        help="Submission type, based on Synapse entities",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        default="results.json",
        help="Output JSON file for scores and results",
    )
    args = parser.parse_args()
    main()
