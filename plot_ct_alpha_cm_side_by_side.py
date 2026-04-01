from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd


INPUT_FILE = Path("BAL") / "compiled_data.xlsx"
OUTPUT_ALPHA_FILE = Path("BAL") / "delta_cd_vs_alpha_side_by_side.png"
OUTPUT_DESIGN_SPACE_FILE = Path("BAL") / "delta_cd_design_space_interpolated.png"
OUTPUT_POINTS_FILE = Path("BAL") / "delta_cd_points.csv"
OUTPUT_INTERPOLATED_GRID_FILE = Path("BAL") / "delta_cd_interpolated_grid.csv"

TARGET_ELEVATOR_LEVELS = [-10, 0, 10]
TARGET_VELOCITIES = [40.0, 20.0]
CM_COLUMN = "Cm_pitch"
PROP_DIAMETER_M = 0.2032


def parse_elevator_deflection(source_file: str) -> int:
    match = re.search(r"elevator([p\-]?\d+)", source_file)
    if not match:
        raise ValueError(f"Could not parse elevator deflection from '{source_file}'.")
    token = match.group(1)
    return int(token[1:]) if token.startswith("p") else int(token)


def as_2d_axes(axes, n_rows: int, n_cols: int):
    if n_rows == 1 and n_cols == 1:
        return np.array([[axes]])
    if n_rows == 1:
        return np.array([axes])
    if n_cols == 1:
        return np.array([[ax] for ax in axes])
    return axes


def interpolate_panel_grid(subset: pd.DataFrame, n_alpha: int = 90, n_j: int = 90) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_points = subset[["Alpha", "J"]].drop_duplicates()
    if len(unique_points) < 3:
        return np.array([]), np.array([]), np.array([[]])

    tri = mtri.Triangulation(
        subset["Alpha"].to_numpy(dtype=float),
        subset["J"].to_numpy(dtype=float),
    )
    interp = mtri.LinearTriInterpolator(tri, subset["delta_CD"].to_numpy(dtype=float))

    alpha_min = float(subset["Alpha"].min())
    alpha_max = float(subset["Alpha"].max())
    j_min = float(subset["J"].min())
    j_max = float(subset["J"].max())

    alpha_grid = np.linspace(alpha_min, alpha_max, n_alpha)
    j_grid = np.linspace(j_min, j_max, n_j)
    aa, jj = np.meshgrid(alpha_grid, j_grid)
    zz = interp(aa, jj)

    # Convert masked array to ndarray with NaN for outside-triangulation cells.
    if np.ma.isMaskedArray(zz):
        zz = zz.filled(np.nan)
    else:
        zz = np.asarray(zz, dtype=float)

    return aa, jj, zz


def interpolate_free_cd(free_group: pd.DataFrame, alpha_query: pd.Series) -> np.ndarray:
    baseline = (
        free_group.groupby("Alpha", as_index=False)["CD"]
        .mean()
        .sort_values("Alpha")
    )
    x = baseline["Alpha"].to_numpy(dtype=float)
    y = baseline["CD"].to_numpy(dtype=float)
    if len(x) < 2:
        return np.full(len(alpha_query), np.nan)
    return np.interp(alpha_query.to_numpy(dtype=float), x, y, left=y[0], right=y[-1])


def compute_delta_cd_from_free(data: pd.DataFrame) -> pd.DataFrame:
    free = data[data["prop_speed_hz"].isna()].copy()
    powered = data[data["prop_speed_hz"].notna()].copy()

    if free.empty:
        raise ValueError("No FREE-propeller points found. Cannot apply T=0 baseline method.")
    if powered.empty:
        raise ValueError("No powered points found to compute delta Cd.")

    cd_free_est = np.full(len(powered), np.nan, dtype=float)
    baseline_mode = np.full(len(powered), "missing", dtype=object)

    # Baseline hierarchy:
    # 1) same elevator and same nominal velocity
    # 2) same elevator across all FREE velocities
    for idx, row in powered.iterrows():
        de = int(row["delta_e_deg"])
        v_nom = float(row["V_nominal"]) if pd.notna(row["V_nominal"]) else np.nan
        alpha = float(row["Alpha"])

        exact = free[
            (free["delta_e_deg"] == de)
            & np.isclose(free["V_nominal"], v_nom, equal_nan=False)
        ]
        if len(exact) >= 2:
            cd_free_est_val = float(interpolate_free_cd(exact, pd.Series([alpha]))[0])
            cd_free_est[powered.index.get_loc(idx)] = cd_free_est_val
            baseline_mode[powered.index.get_loc(idx)] = "same_delta_e_same_V"
            continue

        fallback = free[free["delta_e_deg"] == de]
        if len(fallback) >= 2:
            cd_free_est_val = float(interpolate_free_cd(fallback, pd.Series([alpha]))[0])
            cd_free_est[powered.index.get_loc(idx)] = cd_free_est_val
            baseline_mode[powered.index.get_loc(idx)] = "same_delta_e_any_V"

    powered = powered.copy()
    powered["CD_free_est"] = cd_free_est
    powered["baseline_mode"] = baseline_mode
    powered = powered.dropna(subset=["CD_free_est"]).copy()

    # Requested metric: difference in drag coefficient from FREE case.
    powered["delta_CD"] = powered["CD"] - powered["CD_free_est"]

    v_for_ct = powered["V"].fillna(powered["V_nominal"])
    n_hz = powered["prop_speed_hz"]
    powered["advance_ratio_J"] = v_for_ct / (n_hz * PROP_DIAMETER_M)

    powered = powered.replace([np.inf, -np.inf], np.nan)
    powered = powered.dropna(subset=["delta_CD", "advance_ratio_J"]).copy()
    powered["point_type"] = "powered_delta"

    free_points = free.copy()
    free_points["CD_free_est"] = free_points["CD"]
    free_points["delta_CD"] = 0.0
    free_points["baseline_mode"] = "free_reference"
    free_points["advance_ratio_J"] = np.nan
    free_points["point_type"] = "free_reference"

    return powered, free_points


def plot_delta_cd_vs_alpha(powered_points: pd.DataFrame, free_points: pd.DataFrame) -> None:
    panel_powered = powered_points[powered_points["delta_e_deg"].isin(TARGET_ELEVATOR_LEVELS)].copy()
    panel_free = free_points[free_points["delta_e_deg"].isin(TARGET_ELEVATOR_LEVELS)].copy()
    panel_data_for_color = pd.concat([panel_powered, panel_free], ignore_index=True)
    if panel_data_for_color.empty:
        raise ValueError("No rows found for elevator levels -10, 0, and +10.")

    cm_min = float(panel_data_for_color[CM_COLUMN].min())
    cm_max = float(panel_data_for_color[CM_COLUMN].max())
    if cm_min == cm_max:
        cm_min -= 1e-9
        cm_max += 1e-9
    cm_levels = np.linspace(cm_min, cm_max, 15)

    n_rows = len(TARGET_VELOCITIES)
    n_cols = len(TARGET_ELEVATOR_LEVELS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 3.8 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = as_2d_axes(axes, n_rows, n_cols)
    mappable = None

    for i, velocity in enumerate(TARGET_VELOCITIES):
        for j, elevator in enumerate(TARGET_ELEVATOR_LEVELS):
            ax = axes[i, j]
            subset_powered = panel_powered[
                (panel_powered["delta_e_deg"] == elevator)
                & np.isclose(panel_powered["V_nominal"], velocity)
            ].copy()
            subset_powered = subset_powered.sort_values("Alpha")
            subset_free = panel_free[
                (panel_free["delta_e_deg"] == elevator)
                & np.isclose(panel_free["V_nominal"], velocity)
            ].copy()
            subset_free = subset_free.sort_values("Alpha")
            subset_all = pd.concat(
                [
                    subset_powered[["Alpha", "delta_CD", CM_COLUMN]],
                    subset_free[["Alpha", "delta_CD", CM_COLUMN]],
                ],
                ignore_index=True,
            )
            subset_all = (
                subset_all.groupby(["Alpha", "delta_CD"], as_index=False)[CM_COLUMN]
                .mean()
                .sort_values(["Alpha", "delta_CD"])
            )

            if subset_powered.empty and subset_free.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No data\nV={int(velocity)} m/s, delta_e={elevator:+d}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9.5,
                )
            else:
                if len(subset_all) >= 3:
                    try:
                        cf = ax.tricontourf(
                            subset_all["Alpha"],
                            subset_all["delta_CD"],
                            subset_all[CM_COLUMN],
                            levels=cm_levels,
                            cmap="viridis",
                        )
                        ax.tricontour(
                            subset_all["Alpha"],
                            subset_all["delta_CD"],
                            subset_all[CM_COLUMN],
                            levels=cm_levels,
                            colors="black",
                            linewidths=0.42,
                            alpha=0.55,
                        )
                        mappable = cf
                    except Exception:
                        pass

                if not subset_powered.empty:
                    ax.scatter(
                        subset_powered["Alpha"],
                        subset_powered["delta_CD"],
                        s=28,
                        c="black",
                        alpha=0.78,
                        linewidths=0.2,
                        edgecolors="black",
                    )
                if not subset_free.empty:
                    ax.scatter(
                        subset_free["Alpha"],
                        subset_free["delta_CD"],
                        s=92,
                        marker="D",
                        alpha=0.98,
                        facecolors="white",
                        linewidths=0.65,
                        edgecolors="black",
                    )

            ax.axhline(0.0, color="0.2", linewidth=0.9, alpha=0.65)
            if i == 0:
                ax.set_title(f"delta_e = {elevator:+d} deg")
            if i == n_rows - 1:
                ax.set_xlabel("Alpha (deg)")
            if j == 0:
                ax.set_ylabel(f"V = {int(velocity)} m/s\nDelta CD = CD - CD_free")
            ax.grid(True, linestyle="--", alpha=0.35)

    if mappable is None:
        norm = mpl.colors.Normalize(vmin=cm_min, vmax=cm_max)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
        sm.set_array([])
        colorbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.94, pad=0.02)
    else:
        colorbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.94, pad=0.02)
    colorbar.set_label(f"{CM_COLUMN} (interpolated)")

    fig.suptitle("Interpolated CM Map on Delta CD vs Alpha (with CM isocurves)")
    fig.savefig(OUTPUT_ALPHA_FILE, dpi=240)
    plt.close(fig)


def plot_design_space_interpolation(powered_points: pd.DataFrame, free_points: pd.DataFrame) -> pd.DataFrame:
    panel_powered = powered_points[powered_points["delta_e_deg"].isin(TARGET_ELEVATOR_LEVELS)].copy()
    panel_free = free_points[free_points["delta_e_deg"].isin(TARGET_ELEVATOR_LEVELS)].copy()
    if panel_powered.empty and panel_free.empty:
        raise ValueError("No rows available for design-space interpolation.")

    # Reduce duplicates before interpolation.
    design = panel_powered.copy()
    design["Alpha_bin"] = design["Alpha"].round(3)
    design["J_bin"] = design["advance_ratio_J"].round(4)
    design = (
        design.groupby(["V_nominal", "delta_e_deg", "Alpha_bin", "J_bin"], as_index=False)
        .agg(
            delta_CD=("delta_CD", "mean"),
            sample_count=("delta_CD", "size"),
        )
        .rename(columns={"Alpha_bin": "Alpha", "J_bin": "J"})
    )
    free_design = panel_free.copy()
    free_design["Alpha_bin"] = free_design["Alpha"].round(3)
    free_design = (
        free_design.groupby(["V_nominal", "delta_e_deg", "Alpha_bin"], as_index=False)
        .agg(
            delta_CD=("delta_CD", "mean"),
            sample_count=("delta_CD", "size"),
        )
        .rename(columns={"Alpha_bin": "Alpha"})
    )

    if design.empty:
        ct_min = -1e-6
        ct_max = 1e-6
    else:
        ct_min = float(design["delta_CD"].min())
        ct_max = float(design["delta_CD"].max())
        if ct_min == ct_max:
            ct_min -= 1e-9
            ct_max += 1e-9
    levels = np.linspace(ct_min, ct_max, 16)

    if design.empty:
        j_min = 0.0
        j_max = 1.0
    else:
        j_min = float(design["J"].min())
        j_max = float(design["J"].max())
    j_span = max(j_max - j_min, 0.1)
    j_free_line = j_max + 0.18 * j_span

    n_rows = len(TARGET_VELOCITIES)
    n_cols = len(TARGET_ELEVATOR_LEVELS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 3.9 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = as_2d_axes(axes, n_rows, n_cols)
    mappable = None
    interpolated_rows: list[dict] = []

    for i, velocity in enumerate(TARGET_VELOCITIES):
        for j, elevator in enumerate(TARGET_ELEVATOR_LEVELS):
            ax = axes[i, j]
            subset = design[
                np.isclose(design["V_nominal"], velocity)
                & (design["delta_e_deg"] == elevator)
            ].copy()
            subset_free = free_design[
                np.isclose(free_design["V_nominal"], velocity)
                & (free_design["delta_e_deg"] == elevator)
            ].copy()

            if len(subset) < 3 and subset_free.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"Not enough points\nV={int(velocity)} m/s, delta_e={elevator:+d}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9.5,
                )
            else:
                if len(subset) >= 3:
                    aa, jj, zz = interpolate_panel_grid(subset, n_alpha=95, n_j=95)
                    if aa.size > 0 and np.isfinite(zz).any():
                        cf = ax.contourf(aa, jj, zz, levels=levels, cmap="turbo")
                        ax.contour(
                            aa,
                            jj,
                            zz,
                            levels=levels,
                            colors="white",
                            linewidths=0.4,
                            alpha=0.65,
                        )
                        mappable = cf

                        flat_alpha = aa.ravel()
                        flat_j = jj.ravel()
                        flat_delta_cd = zz.ravel()
                        finite_mask = np.isfinite(flat_delta_cd)
                        for a_val, j_val, d_val in zip(
                            flat_alpha[finite_mask],
                            flat_j[finite_mask],
                            flat_delta_cd[finite_mask],
                        ):
                            interpolated_rows.append(
                                {
                                    "V_nominal": float(velocity),
                                    "delta_e_deg": int(elevator),
                                    "Alpha": float(a_val),
                                    "J": float(j_val),
                                    "delta_CD_interpolated": float(d_val),
                                }
                            )

                    ax.scatter(
                        subset["Alpha"],
                        subset["J"],
                        s=35 + 10 * subset["sample_count"],
                        c="black",
                        alpha=0.8,
                        linewidths=0.2,
                    )
                if not subset_free.empty:
                    ax.scatter(
                        subset_free["Alpha"],
                        np.full(len(subset_free), j_free_line),
                        s=70 + 10 * subset_free["sample_count"],
                        marker="D",
                        facecolors="white",
                        edgecolors="black",
                        linewidths=0.8,
                        alpha=0.95,
                    )
                ax.axhline(j_free_line, color="0.25", linestyle=":", linewidth=0.9, alpha=0.7)

            if i == 0:
                ax.set_title(f"delta_e = {elevator:+d} deg")
            if i == n_rows - 1:
                ax.set_xlabel("Alpha (deg)")
            if j == 0:
                ax.set_ylabel(f"V = {int(velocity)} m/s\nAdvance ratio J")
            ax.grid(True, linestyle="--", alpha=0.30)
            ax.set_ylim(j_min - 0.08 * j_span, j_free_line + 0.10 * j_span)

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.95, pad=0.02)
        cbar.set_label("Interpolated delta CD")

    fig.suptitle("Interpolated Delta-CD Design Space (+ FREE points at top dashed line, J undefined)")
    fig.savefig(OUTPUT_DESIGN_SPACE_FILE, dpi=240)
    plt.close(fig)

    if interpolated_rows:
        return pd.DataFrame(interpolated_rows).sort_values(["V_nominal", "delta_e_deg", "Alpha", "J"])
    return pd.DataFrame(columns=["V_nominal", "delta_e_deg", "Alpha", "J", "delta_CD_interpolated"])


def main() -> None:
    data = pd.read_excel(INPUT_FILE)

    required_cols = {
        "source_file",
        "Alpha",
        "CD",
        "V",
        "V_nominal",
        "prop_speed_hz",
        CM_COLUMN,
    }
    missing = required_cols.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_FILE}: {sorted(missing)}")

    data["Alpha"] = pd.to_numeric(data["Alpha"], errors="coerce")
    data["CD"] = pd.to_numeric(data["CD"], errors="coerce")
    data["V"] = pd.to_numeric(data["V"], errors="coerce")
    data["V_nominal"] = pd.to_numeric(data["V_nominal"], errors="coerce")
    data["prop_speed_hz"] = pd.to_numeric(data["prop_speed_hz"], errors="coerce")
    data[CM_COLUMN] = pd.to_numeric(data[CM_COLUMN], errors="coerce")
    data = data.dropna(subset=["source_file", "Alpha", "CD", "V_nominal", CM_COLUMN]).copy()
    data["delta_e_deg"] = data["source_file"].map(parse_elevator_deflection)

    delta_cd_points, free_points = compute_delta_cd_from_free(data)
    all_points = pd.concat([delta_cd_points, free_points], ignore_index=True)
    all_points.to_csv(OUTPUT_POINTS_FILE, index=False)

    plot_delta_cd_vs_alpha(delta_cd_points, free_points)
    interpolated_grid = plot_design_space_interpolation(delta_cd_points, free_points)
    interpolated_grid.to_csv(OUTPUT_INTERPOLATED_GRID_FILE, index=False)

    fallback_stats = delta_cd_points["baseline_mode"].value_counts(dropna=False).to_dict()
    print(f"Saved delta-CD points: {OUTPUT_POINTS_FILE}")
    print(f"Saved plot: {OUTPUT_ALPHA_FILE}")
    print(f"Saved plot: {OUTPUT_DESIGN_SPACE_FILE}")
    print(f"Saved interpolated grid: {OUTPUT_INTERPOLATED_GRID_FILE}")
    print(f"Baseline usage: {fallback_stats}")


if __name__ == "__main__":
    main()
