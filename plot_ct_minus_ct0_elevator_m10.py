from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data_Loader import add_operating_columns, read_bal_file


INPUT_FILE = Path("BAL") / "corr_elevator-10rudder0.txt"
OUTPUT_PLOT = Path("BAL") / "ct_minus_ct0_elevator_m10.png"
OUTPUT_CSV = Path("BAL") / "ct_minus_ct0_elevator_m10.csv"
ZERO_THRUST_RUNS = {16, 17, 19, 20, 21}
EXCLUDE_SPEED_MPS = 20.0
SPEED_TOL_MPS = 3.0


def main() -> None:
    rows = read_bal_file(INPUT_FILE)
    if not rows:
        raise ValueError(f"No data rows parsed from {INPUT_FILE}")

    df = pd.DataFrame(rows)
    for col in ["Run_nr", "Alpha", "Ct", "Cm_pitch", "V"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Run_nr", "Alpha", "Ct", "Cm_pitch", "V"]).copy()
    df["Run_nr"] = df["Run_nr"].astype(int)
    df = df.sort_values("Run_nr").reset_index(drop=True)
    df = df[np.abs(df["V"] - EXCLUDE_SPEED_MPS) > SPEED_TOL_MPS].copy()
    if df.empty:
        raise ValueError("No rows left after removing 20 m/s runs.")

    df = add_operating_columns(df)

    baseline = df[df["Run_nr"].isin(ZERO_THRUST_RUNS)].copy()
    if baseline.empty:
        raise ValueError(f"No baseline runs found: {sorted(ZERO_THRUST_RUNS)}")

    baseline_curve = (
        baseline.groupby("Alpha", as_index=False)["Ct"]
        .mean()
        .sort_values("Alpha")
    )

    x_ref = baseline_curve["Alpha"].to_numpy(dtype=float)
    y_ref = baseline_curve["Ct"].to_numpy(dtype=float)
    if len(x_ref) < 2:
        raise ValueError("Need at least 2 baseline alpha points for interpolation.")

    df["Ct_zero_thrust_ref"] = np.interp(
        df["Alpha"].to_numpy(dtype=float),
        x_ref,
        y_ref,
        left=y_ref[0],
        right=y_ref[-1],
    )
    df["Ct_minus_Ct0"] = df["Ct"] - df["Ct_zero_thrust_ref"]

    df.to_csv(OUTPUT_CSV, index=False)

    fig, ax = plt.subplots(figsize=(10.4, 6.0), constrained_layout=True)

    cm_min = float(df["Cm_pitch"].min())
    cm_max = float(df["Cm_pitch"].max())
    if cm_min == cm_max:
        cm_min -= 1e-9
        cm_max += 1e-9
    cm_norm = plt.Normalize(vmin=cm_min, vmax=cm_max)
    cm_cmap = plt.colormaps["coolwarm"]

    ax.scatter(
        df["Alpha"],
        df["Ct_minus_Ct0"],
        s=55,
        c=df["Cm_pitch"],
        cmap=cm_cmap,
        norm=cm_norm,
        alpha=0.82,
        edgecolors="black",
        linewidths=0.25,
        label="All runs (filled by Cm_pitch)",
    )

    baseline_rows = df[df["Run_nr"].isin(ZERO_THRUST_RUNS)]
    ax.scatter(
        baseline_rows["Alpha"],
        baseline_rows["Ct_minus_Ct0"],
        s=90,
        marker="D",
        c=baseline_rows["Cm_pitch"],
        cmap=cm_cmap,
        norm=cm_norm,
        edgecolors="black",
        linewidths=1.2,
        label="Zero-thrust baseline runs (16,17,19,20,21)",
        zorder=3,
    )

    # Interpolate curves for each propeller-speed setting.
    for n_hz, group in df.groupby("prop_speed_hz", dropna=False):
        grouped = group.groupby("Alpha", as_index=False)["Ct_minus_Ct0"].mean().sort_values("Alpha")
        if grouped["Alpha"].nunique() < 2:
            continue
        x = grouped["Alpha"].to_numpy(dtype=float)
        y = grouped["Ct_minus_Ct0"].to_numpy(dtype=float)
        x_dense = np.linspace(float(x.min()), float(x.max()), 180)
        y_dense = np.interp(x_dense, x, y)

        if pd.isna(n_hz):
            line_label = "Interpolated FREE"
            line_color = "0.35"
            line_style = ":"
        else:
            line_label = f"Interpolated {float(n_hz):.1f} Hz"
            line_color = None
            line_style = "-"

        ax.plot(
            x_dense,
            y_dense,
            linestyle=line_style,
            linewidth=1.05,
            alpha=0.72,
            color=line_color,
            label=line_label,
            zorder=1,
        )

    # Label each point with run number and propeller speed.
    for k, row in enumerate(df.itertuples(index=False), start=0):
        if pd.isna(row.prop_speed_hz):
            label = f"run {row.Run_nr}, FREE"
        else:
            label = f"run {row.Run_nr}, {row.prop_speed_hz:.1f} Hz"
        y_offset = 8 if (k % 2 == 0) else -10
        ax.annotate(
            label,
            (row.Alpha, row.Ct_minus_Ct0),
            textcoords="offset points",
            xytext=(5, y_offset),
            fontsize=7.0,
            color="0.15",
        )

    ax.axhline(0.0, color="0.25", linewidth=0.9, linestyle="--", alpha=0.8)
    ax.set_xlabel("AoA, alpha (deg)")
    ax.set_ylabel("Ct - Ct0")
    ax.set_title("Elevator -10 deg: Ct - Ct0 (points filled by Cm_pitch)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=7.8, framealpha=0.95)

    sm = plt.cm.ScalarMappable(norm=cm_norm, cmap=cm_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Cm_pitch")

    fig.savefig(OUTPUT_PLOT, dpi=240)
    plt.close(fig)

    print(f"Saved plot: {OUTPUT_PLOT}")
    print(f"Saved data: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
