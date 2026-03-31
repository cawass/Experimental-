from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROP_DIAMETER_M = 0.2032
CM_COLUMN = "Cm_pitch"
TARGET_VELOCITIES = [20.0, 40.0]
TARGET_ALPHAS = [-2.0, 4.0, 8.0]
ALPHA_TOLERANCE_DEG = 0.25
COLORMAP = "magma"

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


def lookup_schedule(run_nr: int) -> tuple[float, float | None]:
    for start, end, v_nom, n_hz, _ in RUN_SCHEDULE:
        if start <= run_nr <= end:
            return v_nom, n_hz
    raise ValueError(f"Run {run_nr} not covered by RUN_SCHEDULE.")


def parse_elevator_deflection(source_file: str) -> int:
    match = re.search(r"elevator([p\-]?\d+)", source_file)
    if not match:
        raise ValueError(f"Could not parse elevator deflection from '{source_file}'.")
    token = match.group(1)
    return int(token[1:]) if token.startswith("p") else int(token)


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    mapped = data["Run_nr"].apply(lookup_schedule)
    data = data.copy()
    data["V_nominal"] = mapped.apply(lambda x: x[0])
    data["n_hz"] = mapped.apply(lambda x: x[1])
    data["delta_e_deg"] = data["source_file"].map(parse_elevator_deflection)
    data["J"] = np.where(
        data["n_hz"].notna(),
        data["V_nominal"] / (data["n_hz"] * PROP_DIAMETER_M),
        np.nan,
    )
    return data


def aggregate_for_contour(data: pd.DataFrame, velocity: float, alpha_target: float) -> pd.DataFrame:
    subset = data[np.isclose(data["V_nominal"], velocity)].copy()
    subset = subset[np.abs(subset["Alpha"] - alpha_target) <= ALPHA_TOLERANCE_DEG]
    subset = subset.dropna(subset=["J", "delta_e_deg", CM_COLUMN])
    if subset.empty:
        return pd.DataFrame(columns=["J", "delta_e_deg", "cm_mean", "sample_count"])

    return (
        subset.groupby(["J", "delta_e_deg"], as_index=False)
        .agg(cm_mean=(CM_COLUMN, "mean"), sample_count=("Run_nr", "count"))
        .sort_values(["J", "delta_e_deg"])
    )


def save_grid_contours(summaries: dict[tuple[float, float], pd.DataFrame], output_path: Path) -> None:
    n_rows = len(TARGET_ALPHAS)
    n_cols = len(TARGET_VELOCITIES)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.8 * n_cols, 3.8 * n_rows), sharex=True, sharey=True, constrained_layout=True
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    non_empty = [df for df in summaries.values() if len(df) >= 3]
    levels = None
    if non_empty:
        cm_values = pd.concat([df["cm_mean"] for df in non_empty], ignore_index=True)
        vmin = float(cm_values.min())
        vmax = float(cm_values.max())
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6
        levels = np.linspace(vmin, vmax, 14)

    mappable = None
    for i, alpha_target in enumerate(TARGET_ALPHAS):
        for j, velocity in enumerate(TARGET_VELOCITIES):
            ax = axes[i, j]
            summary = summaries.get((velocity, alpha_target), pd.DataFrame())

            if levels is None or len(summary) < 3:
                ax.text(
                    0.5,
                    0.5,
                    f"Not enough data\nV={int(velocity)} m/s, alpha={alpha_target:g} deg",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
                ax.grid(True, linestyle="--", alpha=0.25)
            else:
                contourf = ax.tricontourf(
                    summary["J"],
                    summary["delta_e_deg"],
                    summary["cm_mean"],
                    levels=levels,
                    cmap=COLORMAP,
                )
                ax.tricontour(
                    summary["J"],
                    summary["delta_e_deg"],
                    summary["cm_mean"],
                    levels=levels,
                    colors="white",
                    linewidths=0.35,
                    alpha=0.6,
                )
                ax.scatter(
                    summary["J"],
                    summary["delta_e_deg"],
                    s=35 + 10 * summary["sample_count"],
                    c="black",
                    alpha=0.8,
                )
                for _, row in summary.iterrows():
                    ax.annotate(
                        f"n={int(row['sample_count'])}",
                        (row["J"], row["delta_e_deg"]),
                        textcoords="offset points",
                        xytext=(0, 6),
                        ha="center",
                        fontsize=7,
                        color="black",
                    )
                mappable = contourf

            if i == 0:
                ax.set_title(f"V = {int(velocity)} m/s")
            if j == 0:
                ax.set_ylabel(f"alpha={alpha_target:g} deg\n" + "delta_e (deg)")
            if i == n_rows - 1:
                ax.set_xlabel("Advance ratio J")
            ax.grid(True, linestyle="--", alpha=0.3)

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, shrink=0.98, pad=0.02)
        cbar.set_label(f"Mean {CM_COLUMN}")

    fig.suptitle(
        f"CM Contour Maps ({CM_COLUMN}) by Velocity and Alpha\nColormap: {COLORMAP}, alpha tol = ±{ALPHA_TOLERANCE_DEG} deg"
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    base_dir = resolve_base_dir()
    input_file = base_dir / "BAL" / "compiled_bal_points.csv"
    data = pd.read_csv(input_file)

    required_cols = {"Run_nr", "source_file", "Alpha", CM_COLUMN}
    missing = required_cols.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns in compiled data: {sorted(missing)}")

    data["Run_nr"] = pd.to_numeric(data["Run_nr"], errors="coerce")
    data = data.dropna(subset=["Run_nr"]).copy()
    data["Run_nr"] = data["Run_nr"].astype(int)
    data = data.sort_values("Run_nr").reset_index(drop=True)

    prepared = prepare_data(data)

    summaries: dict[tuple[float, float], pd.DataFrame] = {}
    export_rows = []
    for velocity in TARGET_VELOCITIES:
        for alpha_target in TARGET_ALPHAS:
            summary = aggregate_for_contour(prepared, velocity, alpha_target)
            summaries[(velocity, alpha_target)] = summary

            summary_export = summary.copy()
            summary_export["V_nominal"] = velocity
            summary_export["alpha_target"] = alpha_target
            export_rows.append(summary_export)

    all_summary = pd.concat(export_rows, ignore_index=True) if export_rows else pd.DataFrame()
    summary_output = base_dir / "BAL" / "cm_contour_alpha_velocity_summary.csv"
    all_summary.to_csv(summary_output, index=False)

    plot_output = base_dir / "BAL" / "cm_contour_alpha_velocity_grid.png"
    save_grid_contours(summaries, plot_output)

    print(f"Saved summary: {summary_output}")
    print(f"Saved grid plot: {plot_output}")


if __name__ == "__main__":
    main()
