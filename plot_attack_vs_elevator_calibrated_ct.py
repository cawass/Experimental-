from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUTS = [
    ("m10", -10, Path("BAL") / "ct_minus_ct0_elevator_m10_calibrated_custom.csv"),
    ("0", 0, Path("BAL") / "ct_minus_ct0_elevator_0_calibrated_custom.csv"),
    ("p10", 10, Path("BAL") / "ct_minus_ct0_elevator_p10_calibrated_custom.csv"),
]

OUTPUT_POINTS = Path("BAL") / "attack_elevator_calibrated_ct_points.csv"
OUTPUT_PLOT = Path("BAL") / "attack_elevator_calibrated_ct_map.png"


def main() -> None:
    frames = []
    for tag, elevator_deg, csv_path in INPUTS:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input file: {csv_path}")

        df = pd.read_csv(csv_path)
        required = {"Alpha", "Ct_minus_Ct0", "Run_nr"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{csv_path.name} missing required columns: {sorted(missing)}")

        df["Alpha"] = pd.to_numeric(df["Alpha"], errors="coerce")
        df["Ct_minus_Ct0"] = pd.to_numeric(df["Ct_minus_Ct0"], errors="coerce")
        df["Run_nr"] = pd.to_numeric(df["Run_nr"], errors="coerce")
        df = df.dropna(subset=["Alpha", "Ct_minus_Ct0", "Run_nr"]).copy()
        df["Run_nr"] = df["Run_nr"].astype(int)

        df["elevator_tag"] = tag
        df["delta_e_deg"] = int(elevator_deg)
        frames.append(df)

    points = pd.concat(frames, ignore_index=True)
    if points.empty:
        raise ValueError("No calibrated points available to plot.")

    points = points.sort_values(["delta_e_deg", "Alpha", "Run_nr"]).reset_index(drop=True)
    points.to_csv(OUTPUT_POINTS, index=False)

    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)

    # Use measured alpha/elevator coordinates only (no interpolation).
    display_points = (
        points.groupby(["Alpha", "delta_e_deg"], as_index=False)
        .agg(
            sample_count=("Ct_minus_Ct0", "size"),
        )
        .sort_values(["delta_e_deg", "Alpha"])
    )

    z_min = float(display_points["sample_count"].min())
    z_max = float(display_points["sample_count"].max())
    if z_min == z_max:
        z_min -= 1e-9
        z_max += 1e-9

    sc = ax.scatter(
        display_points["Alpha"],
        display_points["delta_e_deg"],
        c=display_points["sample_count"],
        cmap="viridis",
        vmin=z_min,
        vmax=z_max,
        s=92,
        edgecolors="black",
        linewidths=0.25,
        alpha=0.95,
        zorder=3,
    )

    for row in display_points.itertuples(index=False):
        ax.annotate(
            f"n={int(row.sample_count)}",
            (row.Alpha, row.delta_e_deg),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="0.15",
        )

    ax.set_xlabel("Attack, alpha (deg)")
    ax.set_ylabel("Elevator deflection, delta_e (deg)")
    ax.set_title("Attack x Elevator Deflection (number of points at each location)")
    ax.set_yticks([-10, 0, 10])
    unique_alpha = sorted(display_points["Alpha"].round(3).unique().tolist())
    if len(unique_alpha) <= 12:
        ax.set_xticks(unique_alpha)
    ax.grid(True, linestyle="--", alpha=0.35)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Number of points")

    fig.savefig(OUTPUT_PLOT, dpi=240)
    plt.close(fig)

    print(f"Saved points: {OUTPUT_POINTS}")
    print(f"Saved plot: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
