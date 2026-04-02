from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUTS = [
    ("m10", -10, Path("BAL") / "ct_minus_ct0_elevator_m10_calibrated_custom.csv"),
    ("0", 0, Path("BAL") / "ct_minus_ct0_elevator_0_calibrated_custom.csv"),
    ("p10", 10, Path("BAL") / "ct_minus_ct0_elevator_p10_calibrated_custom.csv"),
]

OUTPUT_PLOT = Path("BAL") / "elevator_deflection_vs_calibrated_ct_points.png"
OUTPUT_CSV = Path("BAL") / "elevator_deflection_vs_calibrated_ct_points.csv"


def main() -> None:
    frames = []
    for tag, elevator_deg, path in INPUTS:
        if not path.exists():
            raise FileNotFoundError(f"Missing calibrated input: {path}")

        df = pd.read_csv(path)
        required = {"Run_nr", "Ct_minus_Ct0"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

        df["Run_nr"] = pd.to_numeric(df["Run_nr"], errors="coerce")
        df["Ct_minus_Ct0"] = pd.to_numeric(df["Ct_minus_Ct0"], errors="coerce")
        df = df.dropna(subset=["Run_nr", "Ct_minus_Ct0"]).copy()
        df["Run_nr"] = df["Run_nr"].astype(int)

        df["elevator_tag"] = tag
        df["elevator_deg"] = int(elevator_deg)
        frames.append(df)

    all_points = pd.concat(frames, ignore_index=True)
    if all_points.empty:
        raise ValueError("No valid points found across calibrated input files.")

    # Small deterministic horizontal jitter so overlapping points are visible.
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.33, 0.33, size=len(all_points))
    all_points["elevator_deg_jittered"] = all_points["elevator_deg"] + jitter

    all_points = all_points.sort_values(["elevator_deg", "Run_nr"]).reset_index(drop=True)
    all_points.to_csv(OUTPUT_CSV, index=False)

    fig, ax = plt.subplots(figsize=(9.6, 5.8), constrained_layout=True)
    color_by_elevator = {-10: "#1f77b4", 0: "#2ca02c", 10: "#d62728"}

    for elevator in [-10, 0, 10]:
        subset = all_points[all_points["elevator_deg"] == elevator]
        if subset.empty:
            continue
        ax.scatter(
            subset["elevator_deg_jittered"],
            subset["Ct_minus_Ct0"],
            s=56,
            alpha=0.84,
            color=color_by_elevator.get(elevator, "black"),
            edgecolors="black",
            linewidths=0.25,
            label=f"elevator {elevator:+d} deg",
        )

    ax.axhline(0.0, color="0.25", linestyle="--", linewidth=0.9, alpha=0.8)
    ax.set_xlabel("Elevator deflection (deg)")
    ax.set_ylabel("Calculated Ct (Ct_minus_Ct0)")
    ax.set_title("Elevator Deflection vs Calculated Ct (Calibrated, All Points)")
    ax.set_xticks([-10, 0, 10])
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.95)

    fig.savefig(OUTPUT_PLOT, dpi=240)
    plt.close(fig)

    print(f"Saved points table: {OUTPUT_CSV}")
    print(f"Saved plot: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
