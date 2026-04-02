from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data_Loader import add_operating_columns, read_bal_file


TARGET_CONFIGS = [
    {
        "tag": "m10",
        "elevator_deg": -10,
        "file": Path("BAL") / "corr_elevator-10rudder0.txt",
        "calibration_runs": {16, 17, 19, 20, 21},
    },
    {
        "tag": "0",
        "elevator_deg": 0,
        "file": Path("BAL") / "corr_elevator0rudder0.txt",
        "calibration_runs": {70, 71, 72, 73, 74},
    },
    {
        "tag": "p10",
        "elevator_deg": 10,
        "file": Path("BAL") / "corr_elevatorp10rudder0.txt",
        "calibration_runs": {43, 44, 45, 46, 47},
    },
]

EXCLUDE_SPEED_MPS = 20.0
SPEED_TOL_MPS = 3.0

OUTPUT_SIDE_BY_SIDE = Path("BAL") / "ct_minus_ct0_elevator_m10_0_p10_side_by_side.png"


def load_bal_dataframe(file_path: Path) -> pd.DataFrame:
    rows = read_bal_file(file_path)
    if not rows:
        raise ValueError(f"No data rows parsed from {file_path}")

    df = pd.DataFrame(rows)
    for col in ["Run_nr", "Alpha", "Ct", "Cm_pitch", "V"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Run_nr", "Alpha", "Ct", "Cm_pitch", "V"]).copy()
    df["Run_nr"] = df["Run_nr"].astype(int)
    df = df.sort_values("Run_nr").reset_index(drop=True)

    # Keep only ~40 m/s block, as requested previously.
    df = df[np.abs(df["V"] - EXCLUDE_SPEED_MPS) > SPEED_TOL_MPS].copy()
    if df.empty:
        raise ValueError(f"No rows left after removing ~20 m/s runs in {file_path.name}")

    df = add_operating_columns(df)
    return df


def build_calibration_curve(df: pd.DataFrame, calibration_runs: set[int], file_name: str) -> tuple[np.ndarray, np.ndarray]:
    baseline = df[df["Run_nr"].isin(calibration_runs)].copy()
    if baseline.empty:
        raise ValueError(f"Calibration runs {sorted(calibration_runs)} were not found in {file_name}.")

    baseline_curve = baseline.groupby("Alpha", as_index=False)["Ct"].mean().sort_values("Alpha")
    x_ref = baseline_curve["Alpha"].to_numpy(dtype=float)
    y_ref = baseline_curve["Ct"].to_numpy(dtype=float)
    if len(x_ref) < 2:
        raise ValueError(f"Need at least 2 baseline alpha points in {file_name}.")
    return x_ref, y_ref


def annotate_points(ax: plt.Axes, df: pd.DataFrame) -> None:
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
            fontsize=6.8,
            color="0.15",
        )


def plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    elevator_deg: int,
    calibration_runs: set[int],
    cm_norm: plt.Normalize,
    cm_cmap,
) -> None:
    ax.scatter(
        df["Alpha"],
        df["Ct_minus_Ct0"],
        s=55,
        c=df["Cm_pitch"],
        cmap=cm_cmap,
        norm=cm_norm,
        alpha=0.86,
        edgecolors="black",
        linewidths=0.25,
        label="All runs (filled by Cm_pitch)",
        zorder=2,
    )

    # Interpolate by each propeller speed block.
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

    annotate_points(ax, df)

    run_text = ",".join(str(r) for r in sorted(calibration_runs))
    ax.axhline(0.0, color="0.25", linewidth=0.9, linestyle="--", alpha=0.8)
    ax.set_xlabel("AoA, alpha (deg)")
    ax.set_title(f"Elevator {elevator_deg:+d} deg\nCt0 calibration runs: {run_text}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=7.3, framealpha=0.95)


def main() -> None:
    processed = []
    for cfg in TARGET_CONFIGS:
        df = load_bal_dataframe(cfg["file"])
        x_ref, y_ref = build_calibration_curve(df, cfg["calibration_runs"], cfg["file"].name)

        df = df.copy()
        df["Ct_zero_thrust_ref"] = np.interp(
            df["Alpha"].to_numpy(dtype=float),
            x_ref,
            y_ref,
            left=y_ref[0],
            right=y_ref[-1],
        )
        df["Ct_minus_Ct0"] = df["Ct"] - df["Ct_zero_thrust_ref"]
        processed.append((cfg, df))

        csv_out = Path("BAL") / f"ct_minus_ct0_elevator_{cfg['tag']}_calibrated_custom.csv"
        df.to_csv(csv_out, index=False)
        print(f"Saved data: {csv_out}")

    combined = pd.concat([df for _, df in processed], ignore_index=True)
    cm_min = float(combined["Cm_pitch"].min())
    cm_max = float(combined["Cm_pitch"].max())
    if cm_min == cm_max:
        cm_min -= 1e-9
        cm_max += 1e-9
    cm_norm = plt.Normalize(vmin=cm_min, vmax=cm_max)
    cm_cmap = plt.colormaps["coolwarm"]

    fig, axes = plt.subplots(
        1,
        len(processed),
        figsize=(8.8 * len(processed), 6.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (cfg, df) in zip(axes, processed):
        plot_panel(
            ax=ax,
            df=df,
            elevator_deg=cfg["elevator_deg"],
            calibration_runs=cfg["calibration_runs"],
            cm_norm=cm_norm,
            cm_cmap=cm_cmap,
        )

    axes[0].set_ylabel("Ct - Ct0")
    fig.suptitle("Ct - Ct0 Calibration for Elevator -10, 0, and +10 (side by side)")

    sm = plt.cm.ScalarMappable(norm=cm_norm, cmap=cm_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.02)
    cbar.set_label("Cm_pitch")

    fig.savefig(OUTPUT_SIDE_BY_SIDE, dpi=240)
    plt.close(fig)
    print(f"Saved plot: {OUTPUT_SIDE_BY_SIDE}")


if __name__ == "__main__":
    main()
