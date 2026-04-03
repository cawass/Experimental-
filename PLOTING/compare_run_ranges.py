"""Print metric differences between explicit run pairs in terminal.

Difference definition for each pair:
    diff = value(run_left) - value(run_right)
for every numeric metric column (excluding 'run').
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_pair(text: str) -> tuple[int, int]:
    """Parse a run pair like '63-46' or '17-32' preserving order."""
    parts = text.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid pair '{text}'. Expected format like '63-46'.")
    left = int(parts[0].strip())
    right = int(parts[1].strip())
    return (left, right)


def _load_table(path: Path) -> pd.DataFrame:
    """Load combined aerodynamic table."""
    try:
        df = pd.read_csv(path, sep="\t", skiprows=1)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python", skiprows=1)

    if "run" not in df.columns:
        raise ValueError("Column 'run' was not found in input file.")

    df["run"] = pd.to_numeric(df["run"], errors="coerce")
    df = df.dropna(subset=["run"]).copy()
    df["run"] = df["run"].astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Print metric differences between explicit run pairs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("corrected_combined_output.txt"),
        help="Path to combined aerodynamic data file.",
    )
    parser.add_argument(
        "--pair",
        type=str,
        action="append",
        default=None,
        help="Run pair in form LEFT-RIGHT (repeat flag for multiple pairs). Example: --pair 63-46 --pair 17-32",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="metric",
        choices=["metric", "absdiff"],
        help="Sort each pair table by metric name or by absolute difference.",
    )
    args = parser.parse_args()

    pair_texts = args.pair if args.pair else ["63-46", "17-32"]
    pairs = [_parse_pair(text) for text in pair_texts]

    df = _load_table(args.input)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    metric_cols = [column for column in numeric_cols if column != "run"]
    if not metric_cols:
        raise RuntimeError("No numeric metric columns found.")

    print("Run-pair comparison")
    print("Difference shown as: value(LEFT) - value(RIGHT)")
    print("")

    for left_run, right_run in pairs:
        left_rows = df[df["run"] == int(left_run)].copy()
        right_rows = df[df["run"] == int(right_run)].copy()

        if left_rows.empty:
            raise RuntimeError(f"No rows found for left run: {left_run}")
        if right_rows.empty:
            raise RuntimeError(f"No rows found for right run: {right_run}")

        left_values = left_rows[metric_cols].mean(numeric_only=True)
        right_values = right_rows[metric_cols].mean(numeric_only=True)
        diff = left_values - right_values

        out = pd.DataFrame(
            {
                "metric": metric_cols,
                f"run_{left_run}": [left_values[m] for m in metric_cols],
                f"run_{right_run}": [right_values[m] for m in metric_cols],
                "diff_left_minus_right": [diff[m] for m in metric_cols],
            }
        )

        if args.sort == "absdiff":
            out["abs_diff"] = out["diff_left_minus_right"].abs()
            out = out.sort_values("abs_diff", ascending=False, kind="stable").drop(columns=["abs_diff"])
        else:
            out = out.sort_values("metric", kind="stable")

        print(f"Pair: {left_run} vs {right_run}  (n_left={len(left_rows)}, n_right={len(right_rows)})")
        print(
            out.to_string(
                index=False,
                float_format=lambda x: f"{x: .8f}",
            )
        )
        print("")


if __name__ == "__main__":
    main()
