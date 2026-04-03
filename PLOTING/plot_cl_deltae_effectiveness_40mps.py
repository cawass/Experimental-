"""Plot CL/CD delta-e behavior and effectiveness from corrected data at ~40 m/s.

Outputs:
1) CL vs delta_e for multiple J (alpha=-2 and alpha=8 panels).
2) Combined CL/CD vs delta_e figure (CL left, CD right; both alpha rows).
3) dCL/d(delta_e) vs J.
4) dCD/d(delta_e) vs J.

Modeling choices:
- CL(delta_e): linear approximation.
- CD(delta_e): quadratic approximation when >=3 delta_e points are available
  (fallback to linear when not enough points).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_data(path: Path) -> pd.DataFrame:
    """Load required columns from the combined aerodynamic file."""
    try:
        df = pd.read_csv(path, sep="\t", skiprows=1)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python", skiprows=1)

    required = ["AoA_deg", "AoA_corr_deg", "elevator_deflection_deg", "V_mps", "J_avg", "CL_corr", "CD_corr"]
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


def _fit_quadratic(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    q2, q1, q0 = np.polyfit(x, y, 2)
    return float(q2), float(q1), float(q0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CL/CD delta_e effectiveness from corrected data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("corrected_combined_output.txt"),
        help="Path to combined aerodynamic data file.",
    )
    parser.add_argument(
        "--cl-deltae-output",
        type=Path,
        default=Path(__file__).with_name("cl_deltae_40mps_alpha_m2_8.png"),
        help="Output figure path for CL vs delta_e.",
    )
    parser.add_argument(
        "--cl-effectiveness-output",
        type=Path,
        default=Path(__file__).with_name("cl_effectiveness_vs_J_40mps_alpha_m2_8.png"),
        help="Output figure path for dCL/d(delta_e) vs J.",
    )
    parser.add_argument(
        "--cd-deltae-output",
        type=Path,
        default=Path(__file__).with_name("2_cl_cd_deltae_40mps_alpha_m2_8.png"),
        help="Output figure path for combined CL/CD vs delta_e.",
    )
    parser.add_argument(
        "--cd-effectiveness-output",
        type=Path,
        default=Path(__file__).with_name("cd_effectiveness_vs_J_40mps_alpha_m2_8.png"),
        help="Output figure path for dCD/d(delta_e) vs J.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(__file__).with_name("cl_cd_deltae_40mps_effectiveness_summary.csv"),
        help="CSV output path for slope/intercept summary.",
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
        help="J value used for zero-thrust reference line in slope-vs-J plots.",
    )
    args = parser.parse_args()

    alpha_panels = [-2, 8]
    deltae_fit_points = [-10, 10]
    deltae_validation_points = [0]
    excluded_j = 1.6
    data = _load_data(args.input)

    filtered = data[(np.abs(data["V_mps"] - args.target_speed) <= args.speed_tol)].copy()

    grouped = (
        filtered.groupby(["alpha_key", "J_key", "delta_e_key"], as_index=False)
        .agg(
            alpha_corr_mean=("AoA_corr_deg", "mean"),
            CL_mean=("CL_corr", "mean"),
            CD_mean=("CD_corr", "mean"),
            n_runs=("CL_corr", "size"),
        )
        .sort_values(["alpha_key", "J_key", "delta_e_key"], kind="stable")
        .reset_index(drop=True)
    )

    grouped = grouped[~np.isclose(grouped["J_key"], excluded_j)].copy()

    # Keep alpha/J groups with both endpoint points for consistent CL-based slope estimates.
    valid_groups: list[tuple[int, float]] = []
    for (alpha_key, j_key), frame in grouped.groupby(["alpha_key", "J_key"]):
        if set(deltae_fit_points).issubset(set(frame["delta_e_key"].tolist())):
            valid_groups.append((int(alpha_key), float(j_key)))

    valid = grouped[
        grouped.apply(lambda r: (int(r["alpha_key"]), float(r["J_key"])) in valid_groups, axis=1)
    ].copy()

    if valid.empty:
        raise RuntimeError("No valid alpha/J groups found for the requested filters.")

    j_levels = sorted(valid["J_key"].unique().tolist())
    cmap = plt.get_cmap("tab10")
    color_map = {j: cmap(index % 10) for index, j in enumerate(j_levels)}

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

    # Build slope summary.
    slope_rows: list[dict[str, float | int]] = []
    for (alpha_key, j_key), curve in valid.groupby(["alpha_key", "J_key"]):
        fit_endpoints = curve[curve["delta_e_key"].isin(deltae_fit_points)].sort_values("delta_e_key")
        if len(fit_endpoints) != len(deltae_fit_points):
            continue

        x_end = fit_endpoints["delta_e_key"].to_numpy(dtype=float)
        y_cl_end = fit_endpoints["CL_mean"].to_numpy(dtype=float)
        slope_cl, intercept_cl = _fit_line(x_end, y_cl_end)

        cd_curve = curve.sort_values("delta_e_key")
        x_cd = cd_curve["delta_e_key"].to_numpy(dtype=float)
        y_cd = cd_curve["CD_mean"].to_numpy(dtype=float)
        if len(np.unique(x_cd)) >= 3:
            q2, q1, q0 = _fit_quadratic(x_cd, y_cd)
            slope_cd = q1  # derivative at delta_e=0
            intercept_cd = q0
        else:
            slope_cd, intercept_cd = _fit_line(x_end, fit_endpoints["CD_mean"].to_numpy(dtype=float))

        slope_rows.append(
            {
                "alpha_key": int(alpha_key),
                "alpha_corr_mean_for_panel": float(curve["alpha_corr_mean"].mean()),
                "J_level": float(j_key),
                "slope_dCL_ddeltae": float(slope_cl),
                "intercept_CL": float(intercept_cl),
                "slope_dCD_ddeltae": float(slope_cd),
                "intercept_CD": float(intercept_cd),
            }
        )

    summary_df = pd.DataFrame(slope_rows).sort_values(["alpha_key", "J_level"], kind="stable")
    if summary_df.empty:
        raise RuntimeError("No slope data available after filtering. Check speed/J/alpha filters.")
    summary_df.to_csv(args.summary_output, index=False)

    def _draw_curve_and_points(
        ax: plt.Axes,
        curve: pd.DataFrame,
        metric_column: str,
        color: object,
        label: str | None,
        fit_kind: str,
        add_validation_label: bool,
    ) -> bool:
        """Draw one series and return updated validation-label state."""
        fit_endpoints = curve[curve["delta_e_key"].isin(deltae_fit_points)].sort_values("delta_e_key")
        val_curve = curve[curve["delta_e_key"].isin(deltae_validation_points)].sort_values("delta_e_key")
        if len(fit_endpoints) != len(deltae_fit_points):
            return add_validation_label

        x_end = fit_endpoints["delta_e_key"].to_numpy(dtype=float)
        y_end = fit_endpoints[metric_column].to_numpy(dtype=float)

        x_points = x_end
        y_points = y_end
        if fit_kind == "quadratic":
            x_full = curve["delta_e_key"].to_numpy(dtype=float)
            y_full = curve[metric_column].to_numpy(dtype=float)
            if len(np.unique(x_full)) >= 3:
                q2, q1, q0 = _fit_quadratic(x_full, y_full)
                x_fit = np.linspace(float(np.min(x_full)), float(np.max(x_full)), 260)
                y_fit = q2 * x_fit**2 + q1 * x_fit + q0
                x_points = x_full
                y_points = y_full
            else:
                slope, intercept = _fit_line(x_end, y_end)
                x_fit = np.linspace(min(deltae_fit_points), max(deltae_fit_points), 220)
                y_fit = slope * x_fit + intercept
        else:
            slope, intercept = _fit_line(x_end, y_end)
            x_fit = np.linspace(min(deltae_fit_points), max(deltae_fit_points), 220)
            y_fit = slope * x_fit + intercept

        ax.plot(x_fit, y_fit, color=color, linewidth=1.6, alpha=0.9)
        ax.plot(
            x_points,
            y_points,
            linestyle="None",
            marker=".",
            markersize=6.0,
            color=color,
            label=label,
        )

        if fit_kind == "linear" and len(val_curve) > 0:
            x_val = val_curve["delta_e_key"].to_numpy(dtype=float)
            y_val = val_curve[metric_column].to_numpy(dtype=float)
            ax.plot(
                x_val,
                y_val,
                linestyle="None",
                marker="x",
                markersize=5.0,
                color=color,
                label="Validation points" if not add_validation_label else None,
            )
            add_validation_label = True

        return add_validation_label

    def _plot_cl_vs_deltae(output_path: Path) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9), sharey=False, facecolor="white")
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)

        for ax, alpha_key in zip(axes, alpha_panels):
            _apply_axis_style(ax)
            panel = valid[valid["alpha_key"] == alpha_key].copy()
            if panel.empty:
                continue
            added_validation = False

            for j in j_levels:
                curve = panel[panel["J_key"] == j].sort_values("delta_e_key")
                added_validation = _draw_curve_and_points(
                    ax=ax,
                    curve=curve,
                    metric_column="CL_mean",
                    color=color_map[j],
                    label=rf"$J={j:.1f}$",
                    fit_kind="linear",
                    add_validation_label=added_validation,
                )

            ax.set_xlabel(r"$\delta_e$", fontweight="bold")
            ax.set_ylabel(r"$C_L$", fontweight="bold")
            alpha_corr_panel = float(panel["alpha_corr_mean"].mean())
            ax.text(
                0.5,
                -0.18,
                rf"$V={args.target_speed:.0f}\,\mathrm{{m/s}},\ \alpha\approx{alpha_corr_panel:.2f}$",
                transform=ax.transAxes,
                ha="center",
                va="top",
            )
            ax.legend(
                loc="center right",
                frameon=True,
                facecolor="white",
                edgecolor="black",
                framealpha=1.0,
                fancybox=False,
            )

        fig.subplots_adjust(bottom=0.22, wspace=0.24)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
        plt.close(fig)

    def _plot_cl_cd_deltae_side_by_side(output_path: Path) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.6), sharex=True, facecolor="white")
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)

        metric_specs = [
            ("CL_mean", r"$C_L$", "linear", 0),
            ("CD_mean", r"$C_D$", "quadratic", 1),
        ]

        for row, alpha_key in enumerate(alpha_panels):
            panel = valid[valid["alpha_key"] == alpha_key].copy()
            if panel.empty:
                continue
            alpha_corr_panel = float(panel["alpha_corr_mean"].mean())

            for metric_column, ylabel, fit_kind, col in metric_specs:
                ax = axes[row, col]
                _apply_axis_style(ax)
                added_validation = False

                for j in j_levels:
                    curve = panel[panel["J_key"] == j].sort_values("delta_e_key")
                    added_validation = _draw_curve_and_points(
                        ax=ax,
                        curve=curve,
                        metric_column=metric_column,
                        color=color_map[j],
                        label=rf"$J={j:.1f}$",
                        fit_kind=fit_kind,
                        add_validation_label=added_validation,
                    )

                if row == len(alpha_panels) - 1:
                    ax.set_xlabel(r"$\delta_e$", fontweight="bold")
                ax.set_ylabel(ylabel, fontweight="bold")
                ax.text(
                    0.5,
                    -0.20,
                    rf"$\alpha\approx{alpha_corr_panel:.2f}$",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                )
                ax.legend(
                    loc="center right",
                    frameon=True,
                    facecolor="white",
                    edgecolor="black",
                    framealpha=1.0,
                    fancybox=False,
                )

        fig.subplots_adjust(bottom=0.12, wspace=0.24, hspace=0.35)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
        plt.close(fig)

    def _plot_slope_vs_j(slope_column: str, ylabel: str, output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7.8, 4.8), facecolor="white")
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)
        _apply_axis_style(ax)

        alpha_levels = sorted(summary_df["alpha_key"].unique().tolist())
        cmap_alpha = plt.get_cmap("tab10")
        alpha_color_map = {a: cmap_alpha(index % 10) for index, a in enumerate(alpha_levels)}

        for alpha_key in alpha_levels:
            curve = summary_df[summary_df["alpha_key"] == alpha_key].sort_values("J_level")
            if curve.empty:
                continue

            alpha_corr_panel = float(curve["alpha_corr_mean_for_panel"].mean())
            if alpha_key in [0, 4]:
                label = rf"$\alpha\approx{alpha_corr_panel:.2f}$ (Validation points)"
                marker = "x"
                markersize = 6.0
            elif alpha_key == 12:
                label = rf"$\alpha\approx{alpha_corr_panel:.2f}$ (Outside linear scope)"
                marker = "s"
                markersize = 5.0
            else:
                label = rf"$\alpha\approx{alpha_corr_panel:.2f}$"
                marker = "."
                markersize = 8.0

            ax.plot(
                curve["J_level"].to_numpy(dtype=float),
                curve[slope_column].to_numpy(dtype=float),
                linestyle="-",
                linewidth=1.2,
                marker=marker,
                markersize=markersize,
                color=alpha_color_map.get(alpha_key, "black"),
                label=label,
            )

        ax.axvline(
            x=args.zero_thrust_j,
            color="red",
            linewidth=1.3,
            linestyle="--",
            label=rf"Zero-thrust line ($J_0={args.zero_thrust_j:g}$)",
        )
        ax.set_xlabel(r"$J$", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0,
            fancybox=False,
        )

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
        plt.close(fig)

    _plot_cl_vs_deltae(args.cl_deltae_output)
    _plot_cl_cd_deltae_side_by_side(args.cd_deltae_output)
    _plot_slope_vs_j("slope_dCL_ddeltae", r"$\mathrm{d}C_L/\mathrm{d}\delta_e$", args.cl_effectiveness_output)
    _plot_slope_vs_j("slope_dCD_ddeltae", r"$\mathrm{d}C_D/\mathrm{d}\delta_e$", args.cd_effectiveness_output)

    print("Saved:")
    print(args.cl_deltae_output)
    print(args.cl_effectiveness_output)
    print(args.cd_deltae_output)
    print(args.cd_effectiveness_output)
    print(args.summary_output)


if __name__ == "__main__":
    main()
