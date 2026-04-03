"""Build reduced design-matrix tables with a J-based blocking strategy.

Uses uncorrected variables from the combined BAL export:
    - AoA_deg
    - elevator_deflection_deg
    - J_avg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_data(path: Path) -> pd.DataFrame:
    """Load required columns from the combined text export."""
    try:
        df = pd.read_csv(path, sep="\t", skiprows=1)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python", skiprows=1)

    required = ["run", "AoA_deg", "elevator_deflection_deg", "J_avg"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    out = df[required].copy()
    out["run"] = pd.to_numeric(out["run"], errors="coerce")
    out["AoA_deg"] = pd.to_numeric(out["AoA_deg"], errors="coerce")
    out["elevator_deflection_deg"] = pd.to_numeric(out["elevator_deflection_deg"], errors="coerce")
    out["J_avg"] = pd.to_numeric(out["J_avg"], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    return out


def _prepare_levels(df: pd.DataFrame, j_round_decimals: int) -> pd.DataFrame:
    """Create discrete design levels and block labels."""
    out = df.copy()
    out["alpha_level"] = np.rint(out["AoA_deg"]).astype(int)
    out["delta_e_level"] = np.rint(out["elevator_deflection_deg"]).astype(int)
    out["J_block_level"] = np.round(out["J_avg"], j_round_decimals)

    unique_blocks = sorted(out["J_block_level"].unique().tolist())
    block_map = {level: f"B{index + 1:02d}" for index, level in enumerate(unique_blocks)}
    out["block_id"] = out["J_block_level"].map(block_map)
    return out


def _build_reduced_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated runs for each block + design point."""
    grouped = (
        df.groupby(["block_id", "J_block_level", "alpha_level", "delta_e_level"], as_index=False)
        .agg(
            n_repeats=("run", "size"),
            representative_run=("run", "min"),
            J_mean=("J_avg", "mean"),
            J_std=("J_avg", "std"),
        )
        .sort_values(["J_block_level", "alpha_level", "delta_e_level"], kind="stable")
        .reset_index(drop=True)
    )

    grouped["J_std"] = grouped["J_std"].fillna(0.0)
    grouped["J_mean"] = grouped["J_mean"].round(4)
    grouped["J_std"] = grouped["J_std"].round(4)
    grouped["J_block_level"] = grouped["J_block_level"].round(3)
    return grouped


def _build_block_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary of the blocking strategy."""

    def _fmt_levels(values: pd.Series) -> str:
        sorted_vals = sorted(set(int(v) for v in values.tolist()))
        return "[" + ", ".join(str(v) for v in sorted_vals) + "]"

    summary = (
        df.groupby(["block_id", "J_block_level"], as_index=False)
        .agg(
            runs_in_block=("run", "size"),
            unique_design_points=("run", lambda s: int(s.index.size)),
            alpha_levels=("alpha_level", _fmt_levels),
            delta_e_levels=("delta_e_level", _fmt_levels),
        )
        .sort_values(["J_block_level"], kind="stable")
        .reset_index(drop=True)
    )

    unique_points = (
        df.groupby(["block_id", "J_block_level", "alpha_level", "delta_e_level"], as_index=False)
        .size()
        .groupby(["block_id", "J_block_level"], as_index=False)
        .agg(unique_design_points=("size", "size"))
    )

    summary = summary.drop(columns=["unique_design_points"]).merge(
        unique_points, on=["block_id", "J_block_level"], how="left"
    )
    summary["is_full_5x3_alpha_delta"] = summary["unique_design_points"] == 15
    summary["J_block_level"] = summary["J_block_level"].round(3)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reduced design-matrix tables with J-based blocking."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("corrected_combined_output.txt"),
        help="Combined aerodynamic export file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to save output tables.",
    )
    parser.add_argument(
        "--j-round-decimals",
        type=int,
        default=1,
        help="Decimals used to define J blocks (default: 1).",
    )
    args = parser.parse_args()

    data = _load_data(args.input)
    leveled = _prepare_levels(data, j_round_decimals=args.j_round_decimals)

    reduced = _build_reduced_table(leveled)
    summary = _build_block_summary(leveled)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reduced_path = args.output_dir / "design_matrix_blocked_reduced_uncorrected.csv"
    summary_path = args.output_dir / "design_matrix_block_summary_uncorrected.csv"
    reduced.to_csv(reduced_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Saved:")
    print(reduced_path)
    print(summary_path)


if __name__ == "__main__":
    main()
