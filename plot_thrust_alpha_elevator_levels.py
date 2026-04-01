from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path("BAL") / "compiled_data.xlsx"
OUTPUT_FILE = Path("BAL") / "thrust_vs_alpha_elevator_levels.png"
TARGET_ELEVATOR_LEVELS = [-10, 0, 10]
THRUST_COLUMN = "thrust_total_2props_N"
CM_COLUMN = "Cm_pitch"
VELOCITY_COLUMN = "V"


def parse_elevator_deflection(source_file: str) -> int:
    match = re.search(r"elevator([p\-]?\d+)", source_file)
    if not match:
        raise ValueError(f"Could not parse elevator deflection from '{source_file}'.")
    token = match.group(1)
    return int(token[1:]) if token.startswith("p") else int(token)


def round_to_nominal_elevator(value: float) -> int:
    levels = np.array(TARGET_ELEVATOR_LEVELS, dtype=float)
    return int(levels[np.argmin(np.abs(levels - float(value)))])


def main() -> None:
    data = pd.read_excel(INPUT_FILE)

    required_cols = {"source_file", "Alpha", THRUST_COLUMN, CM_COLUMN, VELOCITY_COLUMN}
    missing = required_cols.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_FILE}: {sorted(missing)}")

    for col in ["Alpha", THRUST_COLUMN, CM_COLUMN, VELOCITY_COLUMN]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["source_file", "Alpha", THRUST_COLUMN, CM_COLUMN, VELOCITY_COLUMN]).copy()
    data["delta_e_raw"] = data["source_file"].map(parse_elevator_deflection)
    data["delta_e_deg"] = data["delta_e_raw"].map(round_to_nominal_elevator)
    data = data[data["delta_e_deg"].isin(TARGET_ELEVATOR_LEVELS)].copy()

    if data.empty:
        raise ValueError("No valid rows found for elevator levels -10, 0, +10.")

    cm_min = float(data[CM_COLUMN].min())
    cm_max = float(data[CM_COLUMN].max())
    if cm_min == cm_max:
        cm_min -= 1e-9
        cm_max += 1e-9
    cm_norm = mpl.colors.Normalize(vmin=cm_min, vmax=cm_max)
    cm_cmap = mpl.colormaps["coolwarm"]

    v_min = float(data[VELOCITY_COLUMN].min())
    v_max = float(data[VELOCITY_COLUMN].max())
    if v_min == v_max:
        v_min -= 1e-9
        v_max += 1e-9
    v_norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    v_cmap = mpl.colormaps["viridis"]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), sharex=True, sharey=True, constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, elevator in zip(axes, TARGET_ELEVATOR_LEVELS):
        subset = data[data["delta_e_deg"] == elevator].copy().sort_values("Alpha")
        if subset.empty:
            ax.text(
                0.5,
                0.5,
                f"No data for delta_e={elevator:+d}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.grid(True, linestyle="--", alpha=0.35)
            continue

        # Marker fill = moment coefficient Cm_pitch.
        ax.scatter(
            subset["Alpha"],
            subset[THRUST_COLUMN],
            c=subset[CM_COLUMN],
            cmap=cm_cmap,
            norm=cm_norm,
            s=60,
            alpha=0.95,
            linewidths=0.25,
            edgecolors="black",
            zorder=2,
        )

        # Outer ring = measurement velocity.
        ring_colors = v_cmap(v_norm(subset[VELOCITY_COLUMN].to_numpy(dtype=float)))
        ax.scatter(
            subset["Alpha"],
            subset[THRUST_COLUMN],
            s=140,
            facecolors="none",
            edgecolors=ring_colors,
            linewidths=1.5,
            alpha=0.95,
            zorder=3,
        )

        ax.axhline(0.0, color="0.2", linewidth=0.9, alpha=0.6)
        ax.set_title(f"delta_e = {elevator:+d} deg")
        ax.set_xlabel("Alpha (deg)")
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Thrust total (2 propellers) [N]")

    sm_cm = mpl.cm.ScalarMappable(norm=cm_norm, cmap=cm_cmap)
    sm_cm.set_array([])
    cbar_cm = fig.colorbar(sm_cm, ax=axes.ravel().tolist(), shrink=0.96, pad=0.02)
    cbar_cm.set_label("Cm_pitch (marker fill)")

    sm_v = mpl.cm.ScalarMappable(norm=v_norm, cmap=v_cmap)
    sm_v.set_array([])
    cbar_v = fig.colorbar(sm_v, ax=axes.ravel().tolist(), shrink=0.96, pad=0.10)
    cbar_v.set_label("Velocity V [m/s] (outer ring color)")

    fig.suptitle("Thrust vs Alpha by Elevator Level (-10, 0, +10)")
    fig.savefig(OUTPUT_FILE, dpi=240)
    plt.close(fig)

    print(f"Saved plot: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
