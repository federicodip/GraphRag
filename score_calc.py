#!/usr/bin/env python
import argparse
import csv
from statistics import mean

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute accuracy/precision from a CSV of graded QA results."
    )
    parser.add_argument(
        "csv_path",
        help="Path to the CSV file (e.g. hybrid_results_ground_truth_failed.csv)",
    )
    parser.add_argument(
        "--score-column",
        default="score_hybrid",
        help="Name of the score column (default: score_hybrid). "
             "If not found, the script will fall back to the last column.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Threshold above which a score counts as 'acceptable' (default: 5.0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    scores = []
    tp = 0  # score > threshold
    fp = 0  # score <= threshold

    with open(args.csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("No rows found in CSV.")
        return

    # Try to detect header and score column
    header = rows[0]
    data_rows = rows[1:]

    # Decide which column to use for scores
    if args.score_column in header:
        score_idx = header.index(args.score_column)
    else:
        # Fall back: treat there as being NO header row,
        # and use the last column as the score.
        # Reinterpret all rows as data.
        print(
            f"[WARN] Score column '{args.score_column}' not found in header. "
            "Falling back to last column and treating all rows as data."
        )
        data_rows = rows  # include first row as data
        score_idx = -1

    for row in data_rows:
        if not row:
            continue
        if abs(score_idx) >= len(row):
            continue

        raw = (row[score_idx] or "").strip()
        if not raw:
            continue

        try:
            score = float(raw)
        except ValueError:
            # Not a numeric score; skip
            continue

        scores.append(score)
        if score > args.threshold:
            tp += 1
        else:
            fp += 1

    total = tp + fp
    if total == 0:
        print("No numeric scores found. Nothing to evaluate.")
        return

    avg_score = mean(scores)
    acceptability_ratio = tp / total  # this is exactly your previous 'ratio(tp, fp)'

    print("--- Evaluation Metrics ---")
    print(f"Total examples: {total}")
    print(f"Threshold for 'acceptable': score > {args.threshold}")
    print(f"Average score: {avg_score:.2f}")
    print(f"Accuracy (proportion score > {args.threshold}): {acceptability_ratio:.4f}")
    print(f"Precision (same set-up as accuracy here):     {acceptability_ratio:.4f}")


if __name__ == "__main__":
    main()
