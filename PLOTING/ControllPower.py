"""Plot Cm vs delta_e and dCm/d(delta_e) vs J at ~40 m/s using corrected Cm data."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Cm-delta_e fits and slope-vs-J at 40 m/s.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("corrected_combined_output.txt"),
        help="Path to combined aerodynamic data file.",
    )
    parser.add_argument(
        "--cm-deltae-output",
        type=Path,
        default=Path(__file__).with_name("controll-power-cm-deltae-40mps.png"),
        help="Output figure path for Cm vs delta_e.",
    )
    parser.add_argument(
        "--slope-output",
        type=Path,
        default=Path(__file__).with_name("controll-power-cm-slope-deltae-vs-j-40mps.png"),
        help="Output figure path for dCm/d(delta_e) vs J.",
    )
    parser.add_argument(
        "--fit-summary-output",
        type=Path,
        default=Path(__file__).with_name("cm_deltae_40mps_fit_summary.csv"),
        help="CSV output path for fitted slope/intercept summary.",
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
        "--zero-thrust-j",
        type=float,
        default=2.5,
        help="J value used for the zero-thrust reference line in slope-vs-J.",
    )
    parser.add_argument(
        "--alpha-target",
        type=int,
        default=0,
        help="Nominal AoA (deg) used for the control-power plots (default: 0).",
    )
    args = parser.parse_args()

    # Use one AoA case for control-power plots.
    alpha_panels = [int(args.alpha_target)]
    deltae_fit_points = [-10, 10]
    deltae_validation_points = [0]

    data = _load_data(args.input)

    filtered = data[
        (np.abs(data["V_mps"] - args.target_speed) <= args.speed_tol)
        & (data["delta_e_key"].isin(deltae_fit_points + deltae_validation_points))
    ].copy()

    grouped = (
        filtered.groupby(["alpha_key", "J_key", "delta_e_key"], as_index=False)
        .agg(
            alpha_corr_mean=("AoA_corr_deg", "mean"),
            Cm_mean=("CMpitch_corr", "mean"),
            n_runs=("CMpitch_corr", "size"),
        )
        .sort_values(["alpha_key", "J_key", "delta_e_key"], kind="stable")
        .reset_index(drop=True)
    )

    # Keep J groups with both fit endpoints present.
    valid_groups: list[tuple[int, float]] = []
    for (alpha_key, j_key), frame in grouped.groupby(["alpha_key", "J_key"]):
        if set(deltae_fit_points).issubset(set(frame["delta_e_key"].tolist())):
            valid_groups.append((int(alpha_key), float(j_key)))

    valid = grouped[
        grouped.apply(lambda r: (int(r["alpha_key"]), float(r["J_key"])) in valid_groups, axis=1)
    ].copy()

    j_levels = sorted(valid["J_key"].unique().tolist())
    if not j_levels:
        raise RuntimeError("No complete J groups found for the requested filters.")

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

    # Figure 1: Cm vs delta_e (single alpha panel).
    fig_width = 7.8 if len(alpha_panels) == 1 else 11.0
    fig, axes = plt.subplots(1, len(alpha_panels), figsize=(fig_width, 4.9), sharey=False, facecolor="white")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    if len(alpha_panels) == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    color_map = {j: cmap(index % 10) for index, j in enumerate(j_levels)}

    # Build slope/intercept summary for all available alpha/J groups.
    fit_rows: list[dict[str, float | int]] = []
    for (alpha_key, j_key), curve in valid.groupby(["alpha_key", "J_key"]):
        fit_curve = curve[curve["delta_e_key"].isin(deltae_fit_points)].sort_values("delta_e_key")
        if len(fit_curve) != len(deltae_fit_points):
            continue

        x_fit_pts = fit_curve["delta_e_key"].to_numpy(dtype=float)
        y_fit_pts = fit_curve["Cm_mean"].to_numpy(dtype=float)
        slope, intercept = _fit_line(x_fit_pts, y_fit_pts)

        fit_rows.append(
            {
                "alpha_key": int(alpha_key),
                "alpha_corr_mean_for_panel": float(curve["alpha_corr_mean"].mean()),
                "J_level": float(j_key),
                "slope_dCm_ddeltae": float(slope),
                "intercept": float(intercept),
            }
        )

    for ax, alpha_key in zip(axes, alpha_panels):
        _apply_axis_style(ax)
        panel = valid[valid["alpha_key"] == alpha_key].copy()
        added_validation_legend = False

        for j in j_levels:
            curve = panel[panel["J_key"] == j].sort_values("delta_e_key")
            fit_curve = curve[curve["delta_e_key"].isin(deltae_fit_points)].sort_values("delta_e_key")
            val_curve = curve[curve["delta_e_key"].isin(deltae_validation_points)].sort_values("delta_e_key")
            if len(fit_curve) != len(deltae_fit_points):
                continue

            x_fit_pts = fit_curve["delta_e_key"].to_numpy(dtype=float)
            y_fit_pts = fit_curve["Cm_mean"].to_numpy(dtype=float)
            slope, intercept = _fit_line(x_fit_pts, y_fit_pts)

            x_fit = np.linspace(min(deltae_fit_points), max(deltae_fit_points), 200)
            y_fit = slope * x_fit + intercept
            color = color_map[j]

            ax.plot(x_fit, y_fit, color=color, linewidth=1.8, label=rf"$J={j:.1f}$")
            ax.plot(
                x_fit_pts,
                y_fit_pts,
                linestyle="None",
                marker=".",
                markersize=6.0,
                color=color,
            )

            if len(val_curve) > 0:
                x_val = val_curve["delta_e_key"].to_numpy(dtype=float)
                y_val = val_curve["Cm_mean"].to_numpy(dtype=float)
                ax.plot(
                    x_val,
                    y_val,
                    linestyle="None",
                    marker="x",
                    markersize=5.0,
                    color=color,
                    label="Validation point" if not added_validation_legend else None,
                )
                added_validation_legend = True

        ax.set_xlabel(r"$\delta_e$", fontweight="bold")
        alpha_corr_panel = float(panel["alpha_corr_mean"].mean()) if len(panel) > 0 else float(alpha_key)
        subcaption = rf"$V={args.target_speed:.0f}\,\mathrm{{m/s}},\ \alpha\approx{alpha_corr_panel:.2f}$"
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

    args.cm_deltae_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.cm_deltae_output, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(fit_rows).sort_values(["alpha_key", "J_level"], kind="stable")
    summary_df.to_csv(args.fit_summary_output, index=False)

    # Figure 2: dCm/d(delta_e) vs J (no curve fit).
    fig2, ax2 = plt.subplots(figsize=(7.8, 4.8), facecolor="white")
    fig2.patch.set_facecolor("white")
    fig2.patch.set_alpha(1.0)
    _apply_axis_style(ax2)

    alpha_levels = sorted(summary_df["alpha_key"].unique().tolist())
    alpha_cmap = plt.get_cmap("tab10")
    alpha_color_map = {alpha_key: alpha_cmap(i % 10) for i, alpha_key in enumerate(alpha_levels)}
    for alpha_key in alpha_levels:
        curve = summary_df[summary_df["alpha_key"] == alpha_key].sort_values("J_level")
        if curve.empty:
            continue
        alpha_corr_panel = float(curve["alpha_corr_mean_for_panel"].mean())
        ax2.plot(
            curve["J_level"].to_numpy(dtype=float),
            curve["slope_dCm_ddeltae"].to_numpy(dtype=float),
            linestyle="-",
            linewidth=1.2,
            marker=".",
            markersize=9.0,
            color=alpha_color_map.get(alpha_key, "black"),
            label=rf"Data ($\alpha\approx{alpha_corr_panel:.2f}$)",
        )

    ax2.axvline(
        x=args.zero_thrust_j,
        color="red",
        linewidth=1.3,
        linestyle="--",
        label=rf"Zero-thrust line ($J_0={args.zero_thrust_j:g}$)",
    )

    ax2.set_xlabel(r"$J$", fontweight="bold")
    ax2.set_ylabel(r"$\mathrm{d}C_m/\mathrm{d}\delta_e$", fontweight="bold")
    ax2.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
    )

    fig2.tight_layout()
    args.slope_output.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(args.slope_output, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig2)

    print("Saved outputs:")
    print(f"Plot - Cm vs delta_e: {args.cm_deltae_output}")
    print(f"Plot - dCm/d(delta_e) vs J: {args.slope_output}")
    print(f"Table - Cm-delta_e fit summary: {args.fit_summary_output}")


if __name__ == "__main__":
    main()
