"""Plot all uncorrected CL-alpha and Cm-alpha curves with linear/non-linear regions.

Uses uncorrected columns from the combined export:
    - AoA_deg
    - CL
    - CMpitch
    - J_avg
    - elevator_deflection_deg

Shaded regions:
    - Linear region:   -2 <= alpha <= 8
    - Non-linear region: 8 <= alpha <= 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def _load_uncorrected(path: Path) -> pd.DataFrame:
    """Load required uncorrected columns."""
    try:
        df = pd.read_csv(path, sep="\t", skiprows=1)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python", skiprows=1)

    required = ["AoA_deg", "elevator_deflection_deg", "J_avg", "V_mps", "CL", "CMpitch"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    out = df[required].copy()
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna().reset_index(drop=True)

    out["alpha_key"] = np.rint(out["AoA_deg"]).astype(int)
    out["delta_e_key"] = np.rint(out["elevator_deflection_deg"]).astype(int)
    out["J_key"] = np.round(out["J_avg"], 2)
    return out


def _apply_axis_style(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    ax.set_facecolor("white")
    ax.patch.set_alpha(1.0)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.7, color="#BDBDBD", alpha=0.9)


def _plot_all_curves(
    grouped: pd.DataFrame,
    y_col: str,
    y_label: str,
    output_path: Path,
    title: str,
) -> None:
    """Plot one curve per elevator angle with linear-fit + join-to-12 construction."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(11.5, 6.0), facecolor="white")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    _apply_axis_style(ax)

    # Shaded regions.
    linear_color = "#6FBF73"
    nonlinear_color = "#E9A15B"
    ax.axvspan(-2, 8, color=linear_color, alpha=0.14, zorder=0)
    ax.axvspan(8, 12, color=nonlinear_color, alpha=0.14, zorder=0)
    ax.axvline(-2, color="#4D4D4D", linestyle=":", linewidth=0.9)
    ax.axvline(8, color="#4D4D4D", linestyle=":", linewidth=0.9)
    ax.axvline(12, color="#4D4D4D", linestyle=":", linewidth=0.9)

    # Style maps: one line per elevator angle.
    de_levels = sorted(grouped["delta_e_key"].unique().tolist())
    cmap = plt.get_cmap("tab10")
    de_color_map = {int(de): cmap(i % 10) for i, de in enumerate(de_levels)}
    marker_cycle = ["o", "s", "D", "^", "v"]
    de_marker_map = {int(de): marker_cycle[i % len(marker_cycle)] for i, de in enumerate(de_levels)}

    # Aggregate over selected J families -> one representative curve per delta_e.
    agg = (
        grouped.groupby(["delta_e_key", "alpha_key"], as_index=False)
        .agg(y_mean=(y_col, "mean"))
        .sort_values(["delta_e_key", "alpha_key"], kind="stable")
        .reset_index(drop=True)
    )

    # Build each delta_e curve as:
    # linear fit in [-2, 8], then straight segment from alpha=8 to alpha=12.
    for de_key in de_levels:
        curve = agg[agg["delta_e_key"] == de_key].sort_values("alpha_key")
        if curve.empty:
            continue

        x_all = curve["alpha_key"].to_numpy(dtype=float)
        y_all = curve["y_mean"].to_numpy(dtype=float)
        linear_mask = (x_all >= -2.0) & (x_all <= 8.0)
        x_lin = x_all[linear_mask]
        y_lin = y_all[linear_mask]
        if len(x_lin) < 2:
            continue

        m, b = np.polyfit(x_lin, y_lin, 1)
        x_fit = np.linspace(float(np.min(x_lin)), float(np.max(x_lin)), 220)
        y_fit = m * x_fit + b
        color = de_color_map[int(de_key)]
        marker = de_marker_map[int(de_key)]

        ax.plot(
            x_fit,
            y_fit,
            color=color,
            linestyle="-",
            linewidth=1.8,
            alpha=0.98,
        )

        # Show all aggregated points used to build the curve.
        ax.plot(
            x_all,
            y_all,
            linestyle="None",
            marker=marker,
            markersize=4.6,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            alpha=0.98,
        )

        idx12 = np.where(np.isclose(x_all, 12.0))[0]
        if len(idx12) > 0:
            idx8 = np.where(np.isclose(x_all, 8.0))[0]
            if len(idx8) > 0:
                y8 = float(y_all[int(idx8[0])])
            else:
                y8 = float(m * 8.0 + b)
            y12 = float(y_all[int(idx12[0])])
            ax.plot(
                [8.0, 12.0],
                [y8, y12],
                color=color,
                linestyle="--",
                linewidth=1.6,
                alpha=0.98,
            )

    ax.set_xlabel(r"$\alpha$ (deg)", fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")

    # Region legend.
    region_handles = [
        mpatches.Patch(facecolor=linear_color, alpha=0.14, edgecolor="none", label="Linear region"),
        mpatches.Patch(facecolor=nonlinear_color, alpha=0.14, edgecolor="none", label="Non-linear region"),
    ]
    region_legend = ax.legend(
        handles=region_handles,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
    )
    ax.add_artist(region_legend)

    # Compact style legend on the right.
    de_handles = [
        Line2D(
            [0],
            [0],
            color=de_color_map[int(de)],
            linestyle="-",
            marker=de_marker_map[int(de)],
            markersize=5.0,
            markerfacecolor="white",
            markeredgecolor=de_color_map[int(de)],
            markeredgewidth=0.9,
            label=rf"$\delta_e={int(de)}$",
        )
        for de in de_levels
    ]
    model_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.4, label="Linear fit (-2 to 8)"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.4, label="Join to 12"),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="None",
            marker="o",
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.9,
            label="Points",
        ),
    ]

    legend_de = ax.legend(
        handles=de_handles + model_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
        title=r"$\delta_e$",
        ncol=1,
    )
    ax.add_artist(legend_de)

    fig.subplots_adjust(left=0.09, right=0.78, top=0.93, bottom=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", transparent=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot all uncorrected CL-alpha and Cm-alpha curves with linear/non-linear regions."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("corrected_combined_output.txt"),
        help="Combined aerodynamic file path.",
    )
    parser.add_argument(
        "--output-cl",
        type=Path,
        default=Path(__file__).with_name("cl_alpha_uncorrected_all_curves_regions.png"),
        help="Output path for CL-alpha figure.",
    )
    parser.add_argument(
        "--output-cm",
        type=Path,
        default=Path(__file__).with_name("cm_alpha_uncorrected_all_curves_regions.png"),
        help="Output path for Cm-alpha figure.",
    )
    parser.add_argument(
        "--target-speed",
        type=float,
        default=None,
        help="Optional target speed in m/s (if omitted, all speeds are used).",
    )
    parser.add_argument(
        "--speed-tol",
        type=float,
        default=2.0,
        help="Speed tolerance in m/s.",
    )
    parser.add_argument(
        "--only-j",
        type=float,
        nargs="*",
        default=[1.9, 2.5, 3.5, 5.0],
        help="Only keep these J levels (default: 1.9 2.5 3.5 5.0).",
    )
    parser.add_argument(
        "--j-tol",
        type=float,
        default=0.15,
        help="Absolute tolerance used to match J levels (default: 0.15).",
    )
    args = parser.parse_args()

    data = _load_uncorrected(args.input)
    filtered = data.copy()
    if args.target_speed is not None:
        filtered = data[np.abs(data["V_mps"] - args.target_speed) <= args.speed_tol].copy()
    if args.only_j:
        targets = np.array([float(x) for x in args.only_j], dtype=float)
        j_values = filtered["J_avg"].to_numpy(dtype=float)
        diffs = np.abs(j_values[:, None] - targets[None, :])
        nearest_idx = np.argmin(diffs, axis=1)
        nearest_diff = diffs[np.arange(len(j_values)), nearest_idx]
        keep = nearest_diff <= float(args.j_tol)
        filtered = filtered[keep].copy()
        filtered["J_family"] = targets[nearest_idx[keep]]
    else:
        filtered["J_family"] = np.round(filtered["J_avg"], 2)

    grouped = (
        filtered.groupby(["J_family", "J_key", "delta_e_key", "alpha_key"], as_index=False)
        .agg(
            CL=("CL", "mean"),
            CMpitch=("CMpitch", "mean"),
        )
        .sort_values(["J_family", "J_key", "delta_e_key", "alpha_key"], kind="stable")
        .reset_index(drop=True)
    )

    _plot_all_curves(
        grouped=grouped,
        y_col="CL",
        y_label=r"$C_L$",
        output_path=args.output_cl,
        title=r"Uncorrected $C_L$-$\alpha$ Curves",
    )
    _plot_all_curves(
        grouped=grouped,
        y_col="CMpitch",
        y_label=r"$C_m$",
        output_path=args.output_cm,
        title=r"Uncorrected $C_m$-$\alpha$ Curves",
    )

    print("Saved:")
    print(args.output_cl)
    print(args.output_cm)
    print("J families used:")
    print(sorted(grouped["J_family"].unique().tolist()))
    print("Raw J levels used:")
    print(sorted(grouped["J_key"].unique().tolist()))


if __name__ == "__main__":
    main()
