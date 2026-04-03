"""Evaluate Reynolds effects at selected J using corrected aerodynamic coefficients.

Creates one image with:
1) A 2x2 block (left): CL vs alpha, CD vs alpha, Cm vs alpha, CL/CD vs alpha
2) One additional plot on the right: Cm vs delta_e for different velocities
   at fixed J and fixed alpha.

Line style by speed:
- V = 40 m/s: solid
- V = 20 m/s: dashed

Color by elevator deflection:
- delta_e = 10, -10 deg
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

    required = [
        "AoA_deg",
        "AoA_corr_deg",
        "elevator_deflection_deg",
        "V_mps",
        "J_avg",
        "CL_corr",
        "CD_corr",
        "CMpitch_corr",
    ]
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


def _speed_key(v: float, tol: float) -> float | None:
    """Map measured speed to 20 or 40 within tolerance."""
    if abs(v - 40.0) <= tol:
        return 40.0
    if abs(v - 20.0) <= tol:
        return 20.0
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Reynolds-effect comparison at J~5.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("corrected_combined_output.txt"),
        help="Path to combined aerodynamic data file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("reynolds-j1p9-cl-cd-cm-v20-v40.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--target-j",
        type=float,
        default=1.9,
        help="Target J level.",
    )
    parser.add_argument(
        "--j-tol",
        type=float,
        default=0.3,
        help="Tolerance around target J.",
    )
    parser.add_argument(
        "--speed-tol",
        type=float,
        default=2.0,
        help="Tolerance for mapping measured speed to 20 or 40 m/s.",
    )
    parser.add_argument(
        "--alpha-target",
        type=float,
        default=0.0,
        help="Target alpha used for the fixed-alpha Cm-vs-delta_e panel.",
    )
    args = parser.parse_args()

    delta_levels = [10, -10]
    speed_levels = [40.0, 20.0]
    speed_style = {40.0: "-", 20.0: "--"}

    data = _load_data(args.input)

    filtered = data[
        (np.abs(data["J_avg"] - args.target_j) <= args.j_tol)
        & (data["delta_e_key"].isin(delta_levels))
    ].copy()
    if filtered.empty:
        raise RuntimeError("No data found for requested J and elevator settings.")

    filtered["speed_key"] = filtered["V_mps"].map(lambda v: _speed_key(float(v), args.speed_tol))
    filtered = filtered.dropna(subset=["speed_key"]).copy()
    filtered["speed_key"] = filtered["speed_key"].astype(float)
    if filtered.empty:
        raise RuntimeError("No data found near V=20/40 m/s within the provided speed tolerance.")

    grouped = (
        filtered.groupby(["speed_key", "delta_e_key", "alpha_key"], as_index=False)
        .agg(
            alpha_corr_mean=("AoA_corr_deg", "mean"),
            CL_mean=("CL_corr", "mean"),
            CD_mean=("CD_corr", "mean"),
            CM_mean=("CMpitch_corr", "mean"),
        )
        .sort_values(["speed_key", "delta_e_key", "alpha_key"], kind="stable")
        .reset_index(drop=True)
    )

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

    grouped["CLCD_mean"] = np.where(np.abs(grouped["CD_mean"]) > 0.0, grouped["CL_mean"] / grouped["CD_mean"], np.nan)

    fig = plt.figure(figsize=(16.0, 8.0), facecolor="white")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.95], wspace=0.30, hspace=0.34)
    axes_flat = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    ax_cm_delta = fig.add_subplot(gs[:, 2])

    cmap = plt.get_cmap("tab10")
    de_color = {10: cmap(0), -10: cmap(2)}

    metric_specs = [
        ("CL_mean", r"$C_L$", r"$C_L$ vs $\alpha$"),
        ("CD_mean", r"$C_D$", r"$C_D$ vs $\alpha$"),
        ("CM_mean", r"$C_m$", r"$C_m$ vs $\alpha$"),
        ("CLCD_mean", r"$C_L/C_D$", r"$C_L/C_D$ vs $\alpha$"),
    ]

    for ax, (metric_column, ylabel, title) in zip(axes_flat, metric_specs):
        _apply_axis_style(ax)
        for delta_e in delta_levels:
            for speed in speed_levels:
                curve = grouped[
                    (grouped["delta_e_key"] == int(delta_e)) & (np.isclose(grouped["speed_key"], float(speed)))
                ].sort_values("alpha_corr_mean")
                if curve.empty:
                    continue

                ax.plot(
                    curve["alpha_corr_mean"].to_numpy(dtype=float),
                    curve[metric_column].to_numpy(dtype=float),
                    color=de_color[int(delta_e)],
                    linestyle=speed_style[float(speed)],
                    linewidth=1.5,
                    marker=".",
                    markersize=5.0,
                )

        ax.set_xlabel(r"$\alpha$", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title)

    delta_handles = [
        Line2D([0], [0], color=de_color[int(delta_e)], linestyle="-", linewidth=1.8, label=rf"$\delta_e={delta_e:d}$")
        for delta_e in delta_levels
    ]
    speed_handles = [
        Line2D([0], [0], color="black", linestyle=speed_style[float(v)], linewidth=1.5, label=rf"$V={int(v):d}$")
        for v in speed_levels
    ]

    legend_delta = ax_cm_delta.legend(
        handles=delta_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
        title=r"$\delta_e$",
    )
    ax_cm_delta.add_artist(legend_delta)
    ax_cm_delta.legend(
        handles=speed_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.68),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
        title=r"$V$",
    )

    # Right panel: Cm vs delta_e for different V at fixed J and fixed alpha.
    _apply_axis_style(ax_cm_delta)
    alpha_levels = sorted(grouped["alpha_key"].unique().tolist())
    alpha_fixed = min(alpha_levels, key=lambda a: abs(float(a) - float(args.alpha_target)))
    panel = grouped[grouped["alpha_key"] == alpha_fixed].copy()

    for speed in speed_levels:
        curve = panel[np.isclose(panel["speed_key"], float(speed))].sort_values("delta_e_key")
        if curve.empty:
            continue
        ax_cm_delta.plot(
            curve["delta_e_key"].to_numpy(dtype=float),
            curve["CM_mean"].to_numpy(dtype=float),
            color="black",
            linestyle=speed_style[float(speed)],
            linewidth=1.8,
            marker=".",
            markersize=6.0,
        )

    alpha_corr_fixed = float(panel["alpha_corr_mean"].mean()) if len(panel) > 0 else float(alpha_fixed)
    ax_cm_delta.set_xlabel(r"$\delta_e$", fontweight="bold")
    ax_cm_delta.set_ylabel(r"$C_m$", fontweight="bold")
    ax_cm_delta.set_title(r"$C_m$ vs $\delta_e$")
    ax_cm_delta.text(
        0.02,
        0.47,
        rf"$J\approx{args.target_j:g},\ \alpha\approx{alpha_corr_fixed:.2f}$",
        transform=ax_cm_delta.transAxes,
        ha="left",
        va="top",
    )

    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor="white", transparent=False)
    plt.close(fig)

    print("Saved outputs:")
    print(
        f"Plot - Reynolds effect at J~{args.target_j:g} "
        f"(CL/CD/Cm/CLCD vs alpha + Cm vs delta_e at fixed alpha): {args.output}"
    )


if __name__ == "__main__":
    main()
