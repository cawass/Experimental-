"""Create combined elevator-effectiveness and control-power figures at ~40 m/s.

This script generates two figures:
1) Side-by-side: CL vs delta_e and Cm vs delta_e (shared legend)
2) Side-by-side: dCL/d(delta_e) vs J and dCm/d(delta_e) vs J (shared legend)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def _load_data(path: Path) -> pd.DataFrame:
    """Load required columns from the combined aerodynamic file."""
    try:
        df = pd.read_csv(path, sep="\t", skiprows=1)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python", skiprows=1)

    required = ["AoA_deg", "AoA_corr_deg", "elevator_deflection_deg", "V_mps", "J_avg", "CL_corr", "CMpitch_corr"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    out = df[required].copy()
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna().reset_index(drop=True)

    out["alpha_key"] = np.rint(out["AoA_deg"]).astype(int)
    out["delta_e_key"] = np.rint(out["elevator_deflection_deg"]).astype(int)
    out["J_key"] = np.round(out["J_avg"], 1)
    return out


def _apply_axis_style(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    ax.set_facecolor("white")
    ax.patch.set_alpha(1.0)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.7, color="#BDBDBD", alpha=0.9)


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _build_valid_grouped(data: pd.DataFrame, target_speed: float, speed_tol: float, excluded_j: float | None) -> pd.DataFrame:
    """Group data and keep alpha/J groups with complete endpoint points."""
    deltae_points = [-10, 0, 10]
    deltae_fit_points = [-10, 10]

    filtered = data[
        (np.abs(data["V_mps"] - target_speed) <= speed_tol)
        & (data["delta_e_key"].isin(deltae_points))
    ].copy()
    if excluded_j is not None:
        filtered = filtered[~np.isclose(filtered["J_key"], float(excluded_j))].copy()

    grouped = (
        filtered.groupby(["alpha_key", "J_key", "delta_e_key"], as_index=False)
        .agg(
            alpha_corr_mean=("AoA_corr_deg", "mean"),
            CL_mean=("CL_corr", "mean"),
            Cm_mean=("CMpitch_corr", "mean"),
        )
        .sort_values(["alpha_key", "J_key", "delta_e_key"], kind="stable")
        .reset_index(drop=True)
    )

    valid_groups: list[tuple[int, float]] = []
    for (alpha_key, j_key), frame in grouped.groupby(["alpha_key", "J_key"]):
        if set(deltae_fit_points).issubset(set(frame["delta_e_key"].tolist())):
            valid_groups.append((int(alpha_key), float(j_key)))

    valid = grouped[
        grouped.apply(lambda r: (int(r["alpha_key"]), float(r["J_key"])) in valid_groups, axis=1)
    ].copy()

    return valid


def _plot_deltae_pair(
    valid: pd.DataFrame,
    output_path: Path,
    target_speed: float,
    alpha_target_key: int,
) -> None:
    """Plot CL-vs-delta_e and Cm-vs-delta_e side by side with shared legend."""
    deltae_fit_points = [-10, 10]
    deltae_validation_points = [0]

    alpha_levels = sorted(valid["alpha_key"].unique().tolist())
    if not alpha_levels:
        raise RuntimeError("No valid alpha levels available for combined delta_e plots.")
    alpha_panel = min(alpha_levels, key=lambda a: abs(a - alpha_target_key))
    panel = valid[valid["alpha_key"] == alpha_panel].copy()
    if panel.empty:
        raise RuntimeError("No panel data available for combined delta_e plots.")

    j_levels = sorted(panel["J_key"].unique().tolist())
    cmap = plt.get_cmap("tab10")
    color_map = {j: cmap(i % 10) for i, j in enumerate(j_levels)}

    fig, (ax_cl, ax_cm) = plt.subplots(1, 2, figsize=(12.0, 4.9), sharex=True, facecolor="white")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    _apply_axis_style(ax_cl)
    _apply_axis_style(ax_cm)

    for j in j_levels:
        curve = panel[panel["J_key"] == j].sort_values("delta_e_key")
        fit_curve = curve[curve["delta_e_key"].isin(deltae_fit_points)].sort_values("delta_e_key")
        val_curve = curve[curve["delta_e_key"].isin(deltae_validation_points)].sort_values("delta_e_key")
        if len(fit_curve) != len(deltae_fit_points):
            continue

        x_fit_pts = fit_curve["delta_e_key"].to_numpy(dtype=float)
        x_fit = np.linspace(min(deltae_fit_points), max(deltae_fit_points), 220)
        color = color_map[j]

        cl_fit_pts = fit_curve["CL_mean"].to_numpy(dtype=float)
        cl_slope, cl_intercept = _fit_line(x_fit_pts, cl_fit_pts)
        ax_cl.plot(x_fit, cl_slope * x_fit + cl_intercept, color=color, linewidth=1.6)
        ax_cl.plot(x_fit_pts, cl_fit_pts, linestyle="None", marker=".", markersize=6.0, color=color)
        if len(val_curve) > 0:
            ax_cl.plot(
                val_curve["delta_e_key"].to_numpy(dtype=float),
                val_curve["CL_mean"].to_numpy(dtype=float),
                linestyle="None",
                marker="x",
                markersize=5.0,
                color=color,
            )

        cm_fit_pts = fit_curve["Cm_mean"].to_numpy(dtype=float)
        cm_slope, cm_intercept = _fit_line(x_fit_pts, cm_fit_pts)
        ax_cm.plot(x_fit, cm_slope * x_fit + cm_intercept, color=color, linewidth=1.6)
        ax_cm.plot(x_fit_pts, cm_fit_pts, linestyle="None", marker=".", markersize=6.0, color=color)
        if len(val_curve) > 0:
            ax_cm.plot(
                val_curve["delta_e_key"].to_numpy(dtype=float),
                val_curve["Cm_mean"].to_numpy(dtype=float),
                linestyle="None",
                marker="x",
                markersize=5.0,
                color=color,
            )

    ax_cl.set_xlabel(r"$\delta_e$", fontweight="bold")
    ax_cm.set_xlabel(r"$\delta_e$", fontweight="bold")
    ax_cl.set_ylabel(r"$C_L$", fontweight="bold")
    ax_cm.set_ylabel(r"$C_m$", fontweight="bold")

    alpha_corr_panel = float(panel["alpha_corr_mean"].mean())
    ax_cl.text(
        0.5,
        -0.18,
        rf"$V={target_speed:.0f}\,\mathrm{{m/s}},\ \alpha\approx{alpha_corr_panel:.2f}$",
        transform=ax_cl.transAxes,
        ha="center",
        va="top",
    )
    ax_cm.text(
        0.5,
        -0.18,
        rf"$V={target_speed:.0f}\,\mathrm{{m/s}},\ \alpha\approx{alpha_corr_panel:.2f}$",
        transform=ax_cm.transAxes,
        ha="center",
        va="top",
    )

    j_handles = [
        Line2D([0], [0], color=color_map[j], linestyle="-", linewidth=1.6, label=rf"$J={j:.1f}$")
        for j in j_levels
    ]
    marker_handles = [
        Line2D([0], [0], linestyle="None", marker=".", markersize=6.0, color="black", label="Fit points"),
        Line2D([0], [0], linestyle="None", marker="x", markersize=5.0, color="black", label="Validation"),
    ]
    fig.legend(
        handles=j_handles + marker_handles,
        loc="center left",
        bbox_to_anchor=(0.89, 0.50),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
    )

    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.22, top=0.95, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)


def _plot_slope_pair(valid: pd.DataFrame, output_path: Path, zero_thrust_j: float) -> None:
    """Plot dCL/d(delta_e)-vs-J and dCm/d(delta_e)-vs-J with shared legend."""
    deltae_fit_points = [-10, 10]

    slope_rows: list[dict[str, float | int]] = []
    for (alpha_key, j_key), curve in valid.groupby(["alpha_key", "J_key"]):
        fit_curve = curve[curve["delta_e_key"].isin(deltae_fit_points)].sort_values("delta_e_key")
        if len(fit_curve) != len(deltae_fit_points):
            continue

        x_fit = fit_curve["delta_e_key"].to_numpy(dtype=float)
        cl_fit = fit_curve["CL_mean"].to_numpy(dtype=float)
        cm_fit = fit_curve["Cm_mean"].to_numpy(dtype=float)
        cl_slope, _ = _fit_line(x_fit, cl_fit)
        cm_slope, _ = _fit_line(x_fit, cm_fit)

        slope_rows.append(
            {
                "alpha_key": int(alpha_key),
                "alpha_corr_mean_for_panel": float(curve["alpha_corr_mean"].mean()),
                "J_level": float(j_key),
                "slope_dCL_ddeltae": float(cl_slope),
                "slope_dCm_ddeltae": float(cm_slope),
            }
        )

    summary_df = pd.DataFrame(slope_rows).sort_values(["alpha_key", "J_level"], kind="stable")
    if summary_df.empty:
        raise RuntimeError("No valid slope data available for combined slope plots.")

    alpha_levels = sorted(summary_df["alpha_key"].unique().tolist())
    cmap = plt.get_cmap("tab10")
    alpha_color_map = {a: cmap(i % 10) for i, a in enumerate(alpha_levels)}

    fig, (ax_cls, ax_cms) = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True, facecolor="white")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    _apply_axis_style(ax_cls)
    _apply_axis_style(ax_cms)

    for alpha_key in alpha_levels:
        curve = summary_df[summary_df["alpha_key"] == alpha_key].sort_values("J_level")
        if curve.empty:
            continue

        color = alpha_color_map[alpha_key]
        ax_cls.plot(
            curve["J_level"].to_numpy(dtype=float),
            curve["slope_dCL_ddeltae"].to_numpy(dtype=float),
            linestyle="-",
            linewidth=1.2,
            marker=".",
            markersize=9.0,
            color=color,
        )
        ax_cms.plot(
            curve["J_level"].to_numpy(dtype=float),
            curve["slope_dCm_ddeltae"].to_numpy(dtype=float),
            linestyle="-",
            linewidth=1.2,
            marker=".",
            markersize=9.0,
            color=color,
        )

    ax_cls.axvline(x=zero_thrust_j, color="red", linewidth=1.3, linestyle="--")
    ax_cms.axvline(x=zero_thrust_j, color="red", linewidth=1.3, linestyle="--")
    ax_cls.set_xlabel(r"$J$", fontweight="bold")
    ax_cms.set_xlabel(r"$J$", fontweight="bold")
    ax_cls.set_ylabel(r"$\mathrm{d}C_L/\mathrm{d}\delta_e$", fontweight="bold")
    ax_cms.set_ylabel(r"$\mathrm{d}C_m/\mathrm{d}\delta_e$", fontweight="bold")

    alpha_handles = []
    for alpha_key in alpha_levels:
        curve = summary_df[summary_df["alpha_key"] == alpha_key]
        alpha_corr = float(curve["alpha_corr_mean_for_panel"].mean())
        alpha_handles.append(
            Line2D(
                [0],
                [0],
                color=alpha_color_map[alpha_key],
                linestyle="-",
                linewidth=1.2,
                marker=".",
                markersize=8.5,
                label=rf"Data ($\alpha\approx{alpha_corr:.2f}$)",
            )
        )
    zero_handle = Line2D(
        [0],
        [0],
        color="red",
        linestyle="--",
        linewidth=1.3,
        label=rf"Zero-thrust line ($J_0={zero_thrust_j:g}$)",
    )

    fig.legend(
        handles=alpha_handles + [zero_handle],
        loc="center left",
        bbox_to_anchor=(0.89, 0.50),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
    )

    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.14, top=0.95, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create combined elevator-effectiveness and control-power plot pages."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("corrected_combined_output.txt"),
        help="Path to combined aerodynamic data file.",
    )
    parser.add_argument(
        "--output-deltae-pair",
        type=Path,
        default=Path(__file__).with_name("combined-elevator-effectiveness-control-power-deltae-40mps.png"),
        help="Output path for combined CL/Cm vs delta_e figure.",
    )
    parser.add_argument(
        "--output-slope-pair",
        type=Path,
        default=Path(__file__).with_name("combined-elevator-effectiveness-control-power-slopes-vs-j-40mps.png"),
        help="Output path for combined slope-vs-J figure.",
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
        "--alpha-target",
        type=int,
        default=0,
        help="Nominal AoA (deg) used for combined delta_e plots.",
    )
    parser.add_argument(
        "--zero-thrust-j",
        type=float,
        default=2.5,
        help="J value used for the zero-thrust reference line in slope-vs-J plots.",
    )
    parser.add_argument(
        "--exclude-j",
        type=float,
        default=1.6,
        help="J level to exclude from both combined figures.",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    data = _load_data(args.input)
    valid = _build_valid_grouped(
        data=data,
        target_speed=args.target_speed,
        speed_tol=args.speed_tol,
        excluded_j=args.exclude_j,
    )
    if valid.empty:
        raise RuntimeError("No valid grouped data found for combined plots. Check filters.")

    _plot_deltae_pair(
        valid=valid,
        output_path=args.output_deltae_pair,
        target_speed=args.target_speed,
        alpha_target_key=args.alpha_target,
    )
    _plot_slope_pair(
        valid=valid,
        output_path=args.output_slope_pair,
        zero_thrust_j=args.zero_thrust_j,
    )

    print("Saved outputs:")
    print(f"Plot - Combined CL/Cm vs delta_e: {args.output_deltae_pair}")
    print(f"Plot - Combined dCL/d(delta_e) & dCm/d(delta_e) vs J: {args.output_slope_pair}")


if __name__ == "__main__":
    main()

