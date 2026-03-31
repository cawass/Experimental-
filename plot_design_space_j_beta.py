from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 8-inch propeller diameter inferred from J values in the provided matrix comments.
PROP_DIAMETER_M = 0.2032

# (run_start, run_end, V_nominal [m/s], n_hz [Hz], label)
RUN_SCHEDULE = [
    (1, 5, 40.0, None, "FREE"),
    (6, 10, 40.0, 39.4, "BRK"),
    (11, 15, 40.0, 55.1, "BRK"),
    (16, 17, 40.0, 78.7, "ZERO"),
    (18, 18, 40.0, 123.0, "MANDATORY"),
    (19, 21, 40.0, 78.7, "ZERO"),
    (22, 26, 20.0, None, "FREE"),
    (27, 31, 20.0, 27.6, "BRK"),
    (32, 32, 40.0, 78.7, "REPLICATE"),
    (33, 37, 40.0, 39.4, "BRK"),
    (38, 42, 40.0, 55.1, "BRK"),
    (43, 47, 40.0, 78.7, "ZERO"),
    (48, 52, 40.0, None, "FREE"),
    (53, 57, 20.0, None, "FREE"),
    (58, 62, 20.0, 27.6, "BRK"),
    (63, 63, 40.0, 78.7, "REPLICATE"),
    (64, 64, 40.0, 123.0, "MANDATORY"),
    (65, 69, 40.0, 39.4, "CAL"),
    (70, 74, 40.0, 78.7, "CAL"),
    (75, 79, 20.0, 27.6, "CAL"),
    (80, 84, 20.0, 27.6, "BRK"),
    (85, 89, 20.0, 22.0, "BRK"),
    (90, 94, 40.0, None, "FREE"),
]


def resolve_base_dir() -> Path:
    cwd = Path.cwd()
    if (cwd / "BAL" / "compiled_bal_points.csv").exists():
        return cwd
    return Path(__file__).resolve().parent


def lookup_schedule(run_nr: int) -> tuple[float, float | None, str]:
    for start, end, v_nom, n_hz, label in RUN_SCHEDULE:
        if start <= run_nr <= end:
            return v_nom, n_hz, label
    raise ValueError(f"Run {run_nr} not covered by RUN_SCHEDULE.")


def parse_elevator_deflection(source_file: str) -> int:
    match = re.search(r"elevator([p\-]?\d+)", source_file)
    if not match:
        raise ValueError(f"Could not parse elevator deflection from '{source_file}'.")

    token = match.group(1)
    if token.startswith("p"):
        return int(token[1:])
    return int(token)


def build_design_space_table(data: pd.DataFrame) -> pd.DataFrame:
    mapped = data["Run_nr"].apply(lookup_schedule)
    data["V_nominal"] = mapped.apply(lambda x: x[0])
    data["n_hz"] = mapped.apply(lambda x: x[1])
    data["mode"] = mapped.apply(lambda x: x[2])
    data["delta_e_deg"] = data["source_file"].map(parse_elevator_deflection)

    data["J"] = np.where(
        data["n_hz"].notna(),
        data["V_nominal"] / (data["n_hz"] * PROP_DIAMETER_M),
        np.nan,
    )
    return data


def make_plot(points: pd.DataFrame, output_path: Path) -> None:
    plot_data = points.dropna(subset=["J", "delta_e_deg"]).copy()

    summary = (
        plot_data.groupby(["J", "delta_e_deg"], as_index=False)
        .agg(run_count=("Run_nr", "count"))
        .sort_values("J")
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        summary["J"],
        summary["delta_e_deg"],
        s=80 + 10 * summary["run_count"],
        c=summary["run_count"],
        cmap="viridis",
        edgecolors="black",
        linewidths=0.6,
    )

    for _, row in summary.iterrows():
        ax.annotate(
            f"n={int(row['run_count'])}",
            (row["J"], row["delta_e_deg"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    free_count = int(points["n_hz"].isna().sum())
    ax.text(
        0.02,
        0.97,
        f"FREE points (J undefined): {free_count}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    ax.set_xlabel("Advance Ratio J")
    ax.set_ylabel("Elevator Deflection delta_e (deg)")
    ax.set_title("Design Space: J vs Elevator Deflection")
    ax.grid(True, linestyle="--", alpha=0.35)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Number of runs at (J, delta_e)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    base_dir = resolve_base_dir()
    input_file = base_dir / "BAL" / "compiled_bal_points.csv"

    data = pd.read_csv(input_file)
    required_cols = {"Run_nr", "source_file"}
    missing = required_cols.difference(data.columns)
    if missing:
        raise ValueError(f"compiled_bal_points.csv missing required columns: {sorted(missing)}")

    data["Run_nr"] = pd.to_numeric(data["Run_nr"], errors="coerce")
    data = data.dropna(subset=["Run_nr"]).copy()
    data["Run_nr"] = data["Run_nr"].astype(int)
    data = data.sort_values("Run_nr").reset_index(drop=True)

    points = build_design_space_table(data)

    points_output = base_dir / "BAL" / "design_space_j_elevator_points.csv"
    summary_output = base_dir / "BAL" / "design_space_j_elevator_summary.csv"
    plot_output = base_dir / "BAL" / "design_space_j_elevator.png"

    points.to_csv(points_output, index=False)
    (
        points.dropna(subset=["J", "delta_e_deg"])
        .groupby(["J", "delta_e_deg"], as_index=False)
        .agg(run_count=("Run_nr", "count"))
        .sort_values("J")
        .to_csv(summary_output, index=False)
    )

    make_plot(points, plot_output)

    print(f"Saved design-space points: {points_output}")
    print(f"Saved design-space summary: {summary_output}")
    print(f"Saved design-space plot: {plot_output}")


if __name__ == "__main__":
    main()
