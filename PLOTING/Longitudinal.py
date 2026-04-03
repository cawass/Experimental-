"""Plot Cm vs alpha at 40 m/s.

Creates one figure with two subplots:
1) elevator deflection = -10 deg
2) elevator deflection = 10 deg

For each subplot, multiple J groups are shown.
Each J curve is a linear fit of Cm(alpha) using only core alpha points:
    [-2, 8]
Validation points (not used in the fit):
    [0, 4]
Outside linear-scope point (not used in the fit):
    [12]

Also creates a second figure with gradient (dCm/dalpha) vs J.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_data(path: Path) -> pd.DataFrame:
    """Load required columns from the combined file."""
    try:
        df = pd.read_csv(path, sep="\t", skiprows=1)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python", skiprows=1)

    required = ["AoA_deg", "AoA_corr_deg", "elevator_deflection_deg", "V_mps", "J_avg", "CMpitch_corr"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    out = df[required].copy()
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna().reset_index(drop=True)

    # Use nominal AoA bins for role selection (core/validation/outside),
    # while plotting/fitting with corrected AoA values.
    out["alpha_key"] = np.rint(out["AoA_deg"]).astype(int)
    out["delta_e_key"] = np.rint(out["elevator_deflection_deg"]).astype(int)
    out["J_key"] = np.round(out["J_avg"], 1)
    return out


def _apply_axis_style(ax: plt.Axes) -> None:
    """Apply style consistent with the existing reference plot settings."""
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    ax.set_facecolor("white")
    ax.patch.set_alpha(1.0)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.7, color="#BDBDBD", alpha=0.9)


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Linear fit y = m*x + b."""
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _slope_sigma_from_point_sigma(alpha_points: list[int], sigma_cm: float) -> float:
    """Propagate point sigma in Cm to a sigma for fitted slope dCm/dalpha."""
    x = np.array(alpha_points, dtype=float)
    sxx = np.sum((x - np.mean(x)) ** 2)
    return float(sigma_cm / np.sqrt(sxx))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Cm-alpha fits for delta_e = -10 and 10 at 40 m/s.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("corrected_combined_output.txt"),
        help="Path to combined aerodynamic data file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("longitudinal-cm-alpha-40mps-deltae-m10-10.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--fit-summary-output",
        type=Path,
        default=Path(__file__).with_name("cm_alpha_40mps_fit_summary.csv"),
        help="CSV output path for fitted slope/intercept summary.",
    )
    parser.add_argument(
        "--gradient-output",
        type=Path,
        default=Path(__file__).with_name("longitudinal-cm-alpha-gradient-vs-j-40mps.png"),
        help="Output figure path for gradient vs J plot.",
    )
    parser.add_argument(
        "--gradient-summary-output",
        type=Path,
        default=Path(__file__).with_name("cm_alpha_gradient_vs_J_regression_40mps.csv"),
        help="CSV output path for gradient-vs-J regression summary.",
    )
    parser.add_argument(
        "--target-speed",
        type=float,
        default=40.0,
        help="Target speed in m/s for filtering.",
    )
    parser.add_argument(
        "--speed-tol",
        type=float,
        default=2.0,
        help="Speed tolerance around target speed in m/s.",
    )
    parser.add_argument(
        "--sigma-cm",
        type=float,
        default=0.00122,
        help="One-sigma Cm uncertainty (kept for exported summary values).",
    )
    parser.add_argument(
        "--sigma-plot-factor",
        type=float,
        default=3.0,
        help="Multiplier applied to sigma_Cm for plotted error bars (default: 3).",
    )
    parser.add_argument(
        "--zero-thrust-j",
        type=float,
        default=2.5,
        help="J value used for the zero-thrust reference line.",
    )
    args = parser.parse_args()

    alpha_fit_points = [-2, 8]
    alpha_validation_points = [0, 4]
    alpha_outside_scope_points = [12]
    alpha_plot_points = sorted(alpha_fit_points + alpha_validation_points + alpha_outside_scope_points)
    elevator_levels = [-10, 10]

    data = _load_data(args.input)

    filtered = data[
        (np.abs(data["V_mps"] - args.target_speed) <= args.speed_tol)
        & (data["delta_e_key"].isin(elevator_levels))
        & (data["alpha_key"].isin(alpha_plot_points))
    ].copy()

    grouped = (
        filtered.groupby(["delta_e_key", "J_key", "alpha_key"], as_index=False)
        .agg(
            alpha_corr_mean=("AoA_corr_deg", "mean"),
            CMpitch_mean=("CMpitch_corr", "mean"),
            n_runs=("CMpitch_corr", "size"),
        )
        .sort_values(["delta_e_key", "J_key", "alpha_key"], kind="stable")
        .reset_index(drop=True)
    )

    # Keep only J groups where all required alpha points exist.
    valid_groups: list[tuple[int, float]] = []
    alpha_set = set(alpha_fit_points)
    for (delta_e_key, j_key), frame in grouped.groupby(["delta_e_key", "J_key"]):
        if alpha_set.issubset(set(frame["alpha_key"].tolist())):
            valid_groups.append((int(delta_e_key), float(j_key)))

    valid = grouped[
        grouped.apply(lambda r: (int(r["delta_e_key"]), float(r["J_key"])) in valid_groups, axis=1)
    ].copy()

    j_levels = sorted(valid["J_key"].unique().tolist())
    if not j_levels:
        raise RuntimeError("No complete J groups found for the requested filters.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9), sharey=False, facecolor="white")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)

    cmap = plt.get_cmap("tab10")
    color_map = {j: cmap(index % 10) for index, j in enumerate(j_levels)}

    fit_rows: list[dict[str, float | int]] = []
    for ax, delta_e in zip(axes, elevator_levels):
        _apply_axis_style(ax)
        panel = valid[valid["delta_e_key"] == delta_e].copy()
        added_validation_legend = False
        added_outside_scope_legend = False
        panel_x_values: list[float] = []

        for j in j_levels:
            curve = panel[panel["J_key"] == j].sort_values("alpha_key")
            fit_curve = curve[curve["alpha_key"].isin(alpha_fit_points)].sort_values("alpha_key")
            val_curve = curve[curve["alpha_key"].isin(alpha_validation_points)].sort_values("alpha_key")
            outside_curve = curve[curve["alpha_key"].isin(alpha_outside_scope_points)].sort_values("alpha_key")
            if len(fit_curve) != len(alpha_fit_points):
                continue

            x_fit_pts = fit_curve["alpha_corr_mean"].to_numpy(dtype=float)
            y_fit_pts = fit_curve["CMpitch_mean"].to_numpy(dtype=float)
            slope, intercept = _fit_line(x_fit_pts, y_fit_pts)

            # Draw solid fit in corrected core range and dashed extrapolation to corrected outside point.
            x_core_min = float(np.min(x_fit_pts))
            x_core_max = float(np.max(x_fit_pts))
            x_fit = np.linspace(x_core_min, x_core_max, 200)
            y_fit = slope * x_fit + intercept
            color = color_map[j]
            panel_x_values.extend(x_fit_pts.tolist())

            ax.plot(x_fit, y_fit, color=color, linewidth=1.8, label=rf"$J={j:.1f}$")
            ax.plot(
                x_fit_pts,
                y_fit_pts,
                linestyle="None",
                marker=".",
                markersize=5.0,
                color=color,
            )
            if len(val_curve) > 0:
                x_val = val_curve["alpha_corr_mean"].to_numpy(dtype=float)
                y_val = val_curve["CMpitch_mean"].to_numpy(dtype=float)
                ax.plot(
                    x_val,
                    y_val,
                    linestyle="None",
                    marker="x",
                    color=color,
                    markersize=4.5,
                    label="Validation points" if not added_validation_legend else None,
                )
                panel_x_values.extend(x_val.tolist())
                added_validation_legend = True

            if len(outside_curve) > 0:
                x_out = outside_curve["alpha_corr_mean"].to_numpy(dtype=float)
                y_out = outside_curve["CMpitch_mean"].to_numpy(dtype=float)
                x_extrap_end = float(np.max(x_out))
                if x_extrap_end > x_core_max:
                    x_extrap = np.linspace(x_core_max, x_extrap_end, 120)
                    y_extrap = slope * x_extrap + intercept
                    ax.plot(
                        x_extrap,
                        y_extrap,
                        color=color,
                        linewidth=1.6,
                        linestyle="--",
                    )
                ax.plot(
                    x_out,
                    y_out,
                    linestyle="None",
                    marker="s",
                    color=color,
                    markerfacecolor="none",
                    markersize=4.5,
                    label="Non-linear points" if not added_outside_scope_legend else None,
                )
                panel_x_values.extend(x_out.tolist())
                added_outside_scope_legend = True

            fit_rows.append(
                {
                    "delta_e_deg": int(delta_e),
                    "J_level": float(j),
                    "slope_dCm_dalpha": slope,
                    "intercept": intercept,
                    "sigma_Cm": float(args.sigma_cm),
                }
            )

        if panel_x_values:
            x_min = min(panel_x_values)
            x_max = max(panel_x_values)
            margin = max(0.35, 0.03 * (x_max - x_min))
            ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_xlabel(r"$\alpha$", fontweight="bold")
        subcaption = rf"$V={args.target_speed:.0f}\,\mathrm{{m/s}},\ \delta_e={delta_e:d}$"
        ax.text(0.5, -0.18, subcaption, transform=ax.transAxes, ha="center", va="top")
        ax.legend(
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0,
            fancybox=False,
        )

    axes[0].set_ylabel(r"$C_m$", fontweight="bold")

    fig.subplots_adjust(bottom=0.22, wspace=0.24)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(fit_rows).sort_values(["delta_e_deg", "J_level"], kind="stable")
    sigma_gradient = _slope_sigma_from_point_sigma(alpha_fit_points, args.sigma_cm)
    summary_df["sigma_gradient"] = sigma_gradient
    summary_df.to_csv(args.fit_summary_output, index=False)

    # Gradient (dCm/dalpha) vs J plot (data only).
    fig_g, ax_g = plt.subplots(figsize=(7.8, 4.8), facecolor="white")
    fig_g.patch.set_facecolor("white")
    fig_g.patch.set_alpha(1.0)
    _apply_axis_style(ax_g)

    regression_rows: list[dict[str, float | int]] = []
    de_color_map = {-10: plt.get_cmap("tab10")(0), 10: plt.get_cmap("tab10")(3)}

    for delta_e in elevator_levels:
        curve = summary_df[summary_df["delta_e_deg"] == delta_e].sort_values("J_level")
        if curve.empty:
            continue

        xj = curve["J_level"].to_numpy(dtype=float)
        yg = curve["slope_dCm_dalpha"].to_numpy(dtype=float)
        color = de_color_map.get(delta_e, "black")

        ax_g.plot(
            xj,
            yg,
            linestyle="-",
            linewidth=1.2,
            marker=".",
            color=color,
            markersize=9.0,
            label=rf"Data ($\delta_e={delta_e:d}$)",
        )

        reg_slope, reg_intercept = _fit_line(xj, yg)

        regression_rows.append(
            {
                "delta_e_deg": int(delta_e),
                "fit_slope_dgrad_dJ": float(reg_slope),
                "fit_intercept": float(reg_intercept),
                "sigma_gradient": float(sigma_gradient),
            }
        )

    zero_line = ax_g.axvline(
        x=args.zero_thrust_j,
        color="red",
        linewidth=1.3,
        linestyle="--",
        label=rf"Zero-thrust line ($J_0={args.zero_thrust_j:g}$)",
    )

    ax_g.set_xlabel(r"$J$", fontweight="bold")
    ax_g.set_ylabel(r"$\mathrm{d}C_m/\mathrm{d}\alpha$", fontweight="bold")
    ax_g.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
    )

    fig_g.tight_layout()
    args.gradient_output.parent.mkdir(parents=True, exist_ok=True)
    fig_g.savefig(args.gradient_output, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig_g)

    regression_df = pd.DataFrame(regression_rows).sort_values(["delta_e_deg"], kind="stable")
    regression_df.to_csv(args.gradient_summary_output, index=False)

    print("Saved outputs:")
    print(f"Plot - Cm vs alpha (delta_e=-10 and 10): {args.output}")
    print(f"Table - Cm-alpha fit summary: {args.fit_summary_output}")
    print(f"Plot - dCm/dalpha vs J: {args.gradient_output}")
    print(f"Table - Gradient-vs-J regression summary: {args.gradient_summary_output}")


if __name__ == "__main__":
    main()
