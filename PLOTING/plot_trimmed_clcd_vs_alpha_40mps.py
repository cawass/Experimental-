"""Generate the complete trimmed-condition plot set at ~40 m/s.

All outputs are strictly filtered to J = 2.5 and J = 5.0.

Output files:
1) Cm vs delta_e (all alpha, both J)
2) CL vs delta_e and CD vs delta_e (side by side, all alpha, both J)
3) Trimmed CL vs alpha
4) Trimmed CD vs alpha
5) Trimmed CL/CD vs alpha
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REQUIRED_J_LEVELS = [2.5, 5.0]
CM_FIT_DELTAE = [-10, 10]
AERO_FIT_DELTAE = [-10, 0, 10]
MAX_TRIM_ABS_DELTAE = 10.0

POINT_CLASS_ORDER = ["fit", "val", "out"]
POINT_CLASS_STYLE = {
    "fit": {"marker": "o", "label": "Fit pts"},
    "val": {"marker": "D", "label": "Val"},
    "out": {"marker": "s", "label": "Out"},
}


def _load_data(path: Path) -> pd.DataFrame:
    """Load required columns from combined aerodynamic data file."""
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
        "CMpitch_corr",
        "CL_corr",
        "CD_corr",
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


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _fit_quad(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    q2, q1, q0 = np.polyfit(x, y, 2)
    return float(q2), float(q1), float(q0)


def _apply_axis_style(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    ax.set_facecolor("white")
    ax.patch.set_alpha(1.0)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.7, color="#BDBDBD", alpha=0.9)


def _lighten_color(color: tuple[float, float, float, float] | str, amount: float = 0.55) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    return tuple(rgb + (1.0 - rgb) * amount)


def _enforce_required_j(df: pd.DataFrame) -> pd.DataFrame:
    keep_mask = np.zeros(len(df), dtype=bool)
    for j_keep in REQUIRED_J_LEVELS:
        keep_mask |= np.isclose(df["J_key"].to_numpy(dtype=float), float(j_keep))
    return df[keep_mask].copy()


def _point_class_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for key in POINT_CLASS_ORDER:
        style = POINT_CLASS_STYLE[key]
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker=style["marker"],
                markersize=5.6,
                markerfacecolor="#DDDDDD",
                markeredgecolor="black",
                markeredgewidth=0.6,
                color="black",
                label=style["label"],
            )
        )
    return handles


def _build_point_class_map(summary_df: pd.DataFrame) -> dict[tuple[float, int], str]:
    """Classify each (J, alpha) point as fit / val / out using trimmed rules."""
    point_class_map: dict[tuple[float, int], str] = {}

    for j in sorted(summary_df["J_level"].unique().tolist()):
        curve = summary_df[np.isclose(summary_df["J_level"], float(j))].sort_values("alpha_key")
        if curve.empty:
            continue

        in_range = curve[np.abs(curve["delta_e_trim"]) <= MAX_TRIM_ABS_DELTAE].copy()
        fit_curve = in_range.sort_values("alpha_key").head(3).copy()
        fit_alpha_set = set(fit_curve["alpha_key"].astype(int).tolist())

        for row in curve.itertuples(index=False):
            alpha_key = int(row.alpha_key)
            if abs(float(row.delta_e_trim)) > MAX_TRIM_ABS_DELTAE:
                cls = "out"
            elif alpha_key in fit_alpha_set:
                cls = "fit"
            else:
                cls = "val"
            point_class_map[(float(j), alpha_key)] = cls

    return point_class_map


def _plot_cm_deltae_all_alpha(
    grouped: pd.DataFrame,
    output_path: Path,
    j_color_map: dict[float, object],
    point_class_map: dict[tuple[float, int], str],
) -> None:
    """Plot Cm vs delta_e in one axis (all alpha, both J)."""
    fig = plt.figure(figsize=(8.8, 5.0), facecolor="white")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    ax = fig.add_subplot(1, 1, 1)
    _apply_axis_style(ax)

    alpha_levels = sorted(grouped["alpha_key"].unique().tolist())
    line_styles = ["-", "--", ":", "-."]
    alpha_style_map = {int(a): line_styles[i % len(line_styles)] for i, a in enumerate(alpha_levels)}

    for j in REQUIRED_J_LEVELS:
        j_frame = grouped[np.isclose(grouped["J_key"], float(j))]
        if j_frame.empty:
            continue

        for alpha_key in alpha_levels:
            curve = j_frame[j_frame["alpha_key"] == alpha_key].sort_values("delta_e_key")
            if curve.empty:
                continue

            x = curve["delta_e_key"].to_numpy(dtype=float)
            y = curve["Cm_mean"].to_numpy(dtype=float)
            alpha_keys = curve["alpha_key"].to_numpy(dtype=int)
            color = j_color_map[float(j)]
            fill_color = _lighten_color(color, amount=0.60)

            # Explicit linear interpolation/fit over delta_e = [-10, 0, 10].
            m_interp, b_interp = _fit_line(x, y)
            x_interp = np.linspace(float(np.min(x)), float(np.max(x)), 180)
            y_interp = m_interp * x_interp + b_interp
            ax.plot(
                x_interp,
                y_interp,
                color=color,
                linestyle=alpha_style_map[int(alpha_key)],
                linewidth=1.5,
            )

            for cls in POINT_CLASS_ORDER:
                mask = np.array(
                    [point_class_map.get((float(j), int(a)), "val") == cls for a in alpha_keys],
                    dtype=bool,
                )
                if not np.any(mask):
                    continue
                ax.plot(
                    x[mask],
                    y[mask],
                    linestyle="None",
                    marker=POINT_CLASS_STYLE[cls]["marker"],
                    markersize=5.2,
                    markerfacecolor=fill_color,
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    zorder=5,
                )

    ax.set_xlabel(r"$\delta_e$", fontweight="bold")
    ax.set_ylabel(r"$C_m$", fontweight="bold")

    j_handles = [
        Line2D([0], [0], color=j_color_map[float(j)], linestyle="-", linewidth=1.8, label=rf"$J={j:.1f}$")
        for j in REQUIRED_J_LEVELS
    ]
    alpha_handles = [
        Line2D([0], [0], color="black", linestyle=alpha_style_map[int(a)], linewidth=1.3, label=rf"$a={int(a)}$")
        for a in alpha_levels
    ]

    legend_main = ax.legend(
        handles=j_handles + alpha_handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
    )
    ax.add_artist(legend_main)
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.3, label="Lin interp."),
            *_point_class_handles(),
        ],
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
        title="pts",
    )

    fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)


def _plot_cl_cd_vs_deltae_all_alpha(
    grouped: pd.DataFrame,
    output_path: Path,
    point_class_map: dict[tuple[float, int], str],
) -> None:
    """Plot side-by-side CL(delta_e) and CD(delta_e) with requested fit models."""
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), sharex=True, facecolor="white")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    ax_cl, ax_cd = axes
    _apply_axis_style(ax_cl)
    _apply_axis_style(ax_cd)

    alpha_levels = sorted(grouped["alpha_key"].unique().tolist())
    alpha_cmap = plt.get_cmap("tab10")
    alpha_color_map = {int(alpha): alpha_cmap(i % 10) for i, alpha in enumerate(alpha_levels)}

    j_linestyle_map = {2.5: "-", 5.0: "--"}

    for j in REQUIRED_J_LEVELS:
        j_frame = grouped[np.isclose(grouped["J_key"], float(j))]
        if j_frame.empty:
            continue

        for alpha_key in alpha_levels:
            curve = j_frame[j_frame["alpha_key"] == alpha_key].sort_values("delta_e_key")
            if len(curve) < 2:
                continue

            x = curve["delta_e_key"].to_numpy(dtype=float)
            y_cl = curve["CL_mean"].to_numpy(dtype=float)
            y_cd = curve["CD_mean"].to_numpy(dtype=float)
            color = alpha_color_map[int(alpha_key)]
            fill_color = _lighten_color(color, amount=0.60)
            cls = point_class_map.get((float(j), int(alpha_key)), "val")
            marker = POINT_CLASS_STYLE[cls]["marker"]

            # CL(delta_e): linear fit with all available points
            cl_m, cl_b = _fit_line(x, y_cl)
            x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 220)
            y_fit_cl = cl_m * x_fit + cl_b
            ax_cl.plot(
                x_fit,
                y_fit_cl,
                color=color,
                linestyle=j_linestyle_map.get(float(j), "-"),
                linewidth=1.2,
            )
            ax_cl.plot(
                x,
                y_cl,
                linestyle="None",
                marker=marker,
                markersize=5.2,
                markerfacecolor=fill_color,
                markeredgecolor="black",
                markeredgewidth=0.6,
            )

            # CD(delta_e): quadratic fit with all available points (fallback to linear if needed)
            if len(np.unique(x)) >= 3:
                q2, q1, q0 = _fit_quad(x, y_cd)
                y_fit_cd = q2 * x_fit**2 + q1 * x_fit + q0
            else:
                cd_m, cd_b = _fit_line(x, y_cd)
                y_fit_cd = cd_m * x_fit + cd_b

            ax_cd.plot(
                x_fit,
                y_fit_cd,
                color=color,
                linestyle=j_linestyle_map.get(float(j), "-"),
                linewidth=1.2,
            )
            ax_cd.plot(
                x,
                y_cd,
                linestyle="None",
                marker=marker,
                markersize=5.2,
                markerfacecolor=fill_color,
                markeredgecolor="black",
                markeredgewidth=0.6,
            )

    ax_cl.set_xlabel(r"$\delta_e$", fontweight="bold")
    ax_cl.set_ylabel(r"$C_L$", fontweight="bold")
    ax_cd.set_xlabel(r"$\delta_e$", fontweight="bold")
    ax_cd.set_ylabel(r"$C_D$", fontweight="bold")

    alpha_handles = [
        Line2D(
            [0],
            [0],
            color=alpha_color_map[int(alpha)],
            linestyle="-",
            linewidth=1.6,
            label=rf"$a={int(alpha)}$",
        )
        for alpha in alpha_levels
    ]
    j_handles = [
        Line2D([0], [0], color="black", linestyle=j_linestyle_map[float(j)], linewidth=1.3, label=rf"$J={j:.1f}$")
        for j in REQUIRED_J_LEVELS
    ]

    legend_alpha = ax_cd.legend(
        handles=alpha_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.55),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
        title="a",
    )
    ax_cd.add_artist(legend_alpha)
    legend_j = ax_cd.legend(
        handles=j_handles,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
        title="J",
    )
    ax_cd.add_artist(legend_j)
    ax_cd.legend(
        handles=_point_class_handles(),
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
        title="pts",
    )

    fig.subplots_adjust(left=0.08, right=0.79, top=0.96, bottom=0.14, wspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all trimmed-condition plots with fixed J=2.5 and 5.0.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("corrected_combined_output.txt"),
        help="Path to combined aerodynamic data file.",
    )
    parser.add_argument(
        "--output-cm-alpha",
        type=Path,
        default=Path(__file__).with_name("1_cm_deltae_40mps_all_alpha_j2p5_j5.png"),
        help="Output path for Cm vs delta_e plot.",
    )
    parser.add_argument(
        "--output-clcd-deltae",
        type=Path,
        default=Path(__file__).with_name("2_cl_cd_deltae_40mps_allalpha_j2p5_j5.png"),
        help="Output path for side-by-side CL/CD vs delta_e plot.",
    )
    parser.add_argument(
        "--output-cl",
        type=Path,
        default=Path(__file__).with_name("3_trimmed_cl_vs_alpha_40mps.png"),
        help="Output path for trimmed CL vs alpha.",
    )
    parser.add_argument(
        "--output-cd",
        type=Path,
        default=Path(__file__).with_name("4_trimmed_cd_vs_alpha_40mps.png"),
        help="Output path for trimmed CD vs alpha.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("5_trimmed_clcd_vs_alpha_40mps.png"),
        help="Output path for trimmed CL/CD vs alpha.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(__file__).with_name("trimmed_clcd_condition_summary_40mps.csv"),
        help="CSV output path with trimmed-condition values.",
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
    args = parser.parse_args()

    data = _load_data(args.input)
    filtered = data[
        (np.abs(data["V_mps"] - args.target_speed) <= args.speed_tol)
        & (data["delta_e_key"].isin(AERO_FIT_DELTAE))
    ].copy()

    grouped = (
        filtered.groupby(["alpha_key", "J_key", "delta_e_key"], as_index=False)
        .agg(
            alpha_corr_mean=("AoA_corr_deg", "mean"),
            Cm_mean=("CMpitch_corr", "mean"),
            CL_mean=("CL_corr", "mean"),
            CD_mean=("CD_corr", "mean"),
            n_runs=("CMpitch_corr", "size"),
        )
        .sort_values(["alpha_key", "J_key", "delta_e_key"], kind="stable")
        .reset_index(drop=True)
    )
    grouped = _enforce_required_j(grouped)
    if grouped.empty:
        raise RuntimeError("No data available for mandatory J levels (2.5 and 5.0).")

    trim_rows: list[dict[str, float | int]] = []
    for (alpha_key, j_key), frame in grouped.groupby(["alpha_key", "J_key"]):
        fit_frame = frame.sort_values("delta_e_key")
        cm_frame = fit_frame[fit_frame["delta_e_key"].isin(CM_FIT_DELTAE)].sort_values("delta_e_key")
        aero_frame = fit_frame[fit_frame["delta_e_key"].isin(AERO_FIT_DELTAE)].sort_values("delta_e_key")
        if len(cm_frame) != len(CM_FIT_DELTAE) or len(aero_frame) != len(AERO_FIT_DELTAE):
            continue

        x_cm = cm_frame["delta_e_key"].to_numpy(dtype=float)
        cm = cm_frame["Cm_mean"].to_numpy(dtype=float)
        x_aero = aero_frame["delta_e_key"].to_numpy(dtype=float)
        cl = aero_frame["CL_mean"].to_numpy(dtype=float)
        cd = aero_frame["CD_mean"].to_numpy(dtype=float)

        cm_m, cm_b = _fit_line(x_cm, cm)
        if np.isclose(cm_m, 0.0):
            continue
        delta_e_trim = -cm_b / cm_m

        cl_m, cl_b = _fit_line(x_aero, cl)
        cl_trim = cl_m * delta_e_trim + cl_b

        cd_q2, cd_q1, cd_q0 = _fit_quad(x_aero, cd)
        cd_trim = cd_q2 * (delta_e_trim**2) + cd_q1 * delta_e_trim + cd_q0
        clcd_trim = np.nan if np.isclose(cd_trim, 0.0) else cl_trim / cd_trim

        trim_rows.append(
            {
                "alpha_key": int(alpha_key),
                "alpha_corr_plot": float(fit_frame["alpha_corr_mean"].mean()),
                "J_level": float(j_key),
                "delta_e_trim": float(delta_e_trim),
                "Cm_slope": float(cm_m),
                "Cm_intercept": float(cm_b),
                "CL_trim": float(cl_trim),
                "CD_trim": float(cd_trim),
                "CLCD_trim": float(clcd_trim),
            }
        )

    summary_df = pd.DataFrame(trim_rows).sort_values(["J_level", "alpha_key"], kind="stable")
    if summary_df.empty:
        raise RuntimeError("No trimmed-condition points available after filtering.")
    summary_df = summary_df[np.isfinite(summary_df["CLCD_trim"])].copy()
    summary_df.to_csv(args.summary_output, index=False)

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

    point_class_map = _build_point_class_map(summary_df)

    cm_j_cmap = plt.get_cmap("tab10")
    cm_j_color_map = {j: cm_j_cmap(i % 10) for i, j in enumerate(REQUIRED_J_LEVELS)}
    _plot_cm_deltae_all_alpha(
        grouped=grouped,
        output_path=args.output_cm_alpha,
        j_color_map=cm_j_color_map,
        point_class_map=point_class_map,
    )
    _plot_cl_cd_vs_deltae_all_alpha(
        grouped=grouped,
        output_path=args.output_clcd_deltae,
        point_class_map=point_class_map,
    )

    j_levels = sorted(summary_df["J_level"].unique().tolist())
    cmap = plt.get_cmap("tab10")
    color_map = {j: cmap(index % 10) for index, j in enumerate(j_levels)}

    plot_data: list[dict[str, object]] = []
    for j in j_levels:
        curve = summary_df[np.isclose(summary_df["J_level"], float(j))].sort_values("alpha_key")
        if curve.empty:
            continue

        in_range = curve[np.abs(curve["delta_e_trim"]) <= MAX_TRIM_ABS_DELTAE].copy()
        fit_curve = in_range.sort_values("alpha_key").head(3).copy()
        if len(fit_curve) < 3:
            continue

        series_color = color_map[j]
        fill_color = _lighten_color(series_color, amount=0.60)

        alpha_fit = fit_curve["alpha_key"].to_numpy(dtype=float)
        cl_fit = fit_curve["CL_trim"].to_numpy(dtype=float)
        cd_fit = fit_curve["CD_trim"].to_numpy(dtype=float)

        cl_lin_coeff = np.polyfit(alpha_fit, cl_fit, 1)
        cd_quad_coeff = np.polyfit(alpha_fit, cd_fit, 2)

        fit_alpha_min = float(np.min(alpha_fit))
        fit_alpha_max = float(np.max(alpha_fit))
        alpha_solid = np.linspace(fit_alpha_min, fit_alpha_max, 220)
        cl_solid = np.polyval(cl_lin_coeff, alpha_solid)
        cd_solid = np.polyval(cd_quad_coeff, alpha_solid)
        clcd_solid = np.where(cd_solid > 0.0, cl_solid / cd_solid, np.nan)

        alpha_max = float(np.max(curve["alpha_key"].to_numpy(dtype=float)))
        has_extrap = alpha_max > fit_alpha_max
        if has_extrap:
            alpha_dash = np.linspace(fit_alpha_max, alpha_max, 120)
            cl_dash = np.polyval(cl_lin_coeff, alpha_dash)
            cd_dash = np.polyval(cd_quad_coeff, alpha_dash)
            clcd_dash = np.where(cd_dash > 0.0, cl_dash / cd_dash, np.nan)

        interpolation = fit_curve
        validation = in_range[~in_range["alpha_key"].isin(fit_curve["alpha_key"].tolist())]
        outside_scope = curve[np.abs(curve["delta_e_trim"]) > MAX_TRIM_ABS_DELTAE]

        item: dict[str, object] = {
            "j": float(j),
            "series_color": series_color,
            "fill_color": fill_color,
            "alpha_solid": alpha_solid,
            "cl_solid": cl_solid,
            "cd_solid": cd_solid,
            "clcd_solid": clcd_solid,
            "has_extrap": bool(has_extrap),
            "interpolation": interpolation.copy(),
            "validation": validation.copy(),
            "outside": outside_scope.copy(),
        }
        if has_extrap:
            item["alpha_dash"] = alpha_dash
            item["cl_dash"] = cl_dash
            item["cd_dash"] = cd_dash
            item["clcd_dash"] = clcd_dash
        plot_data.append(item)

    if not plot_data:
        raise RuntimeError("No valid regression groups available after trim filtering.")

    def _plot_one(metric_col: str, solid_key: str, dash_key: str, ylabel: str, output_path: Path) -> None:
        fig = plt.figure(figsize=(7.8, 4.8), facecolor="white")
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)
        ax = fig.add_subplot(1, 1, 1)
        _apply_axis_style(ax)

        j_handles: list[Line2D] = []
        for item in plot_data:
            x_solid = np.asarray(item["alpha_solid"], dtype=float)
            y_solid = np.asarray(item[solid_key], dtype=float)
            series_color = item["series_color"]
            fill_color = item["fill_color"]
            j_value = float(item["j"])

            line, = ax.plot(
                x_solid,
                y_solid,
                color=series_color,
                linewidth=1.6,
                linestyle="-",
            )
            if bool(item["has_extrap"]):
                ax.plot(
                    np.asarray(item["alpha_dash"], dtype=float),
                    np.asarray(item[dash_key], dtype=float),
                    color=series_color,
                    linewidth=1.6,
                    linestyle="--",
                )

            interpolation = item["interpolation"]
            validation = item["validation"]
            outside = item["outside"]

            if len(interpolation) > 0:
                ax.plot(
                    interpolation["alpha_key"].to_numpy(dtype=float),
                    interpolation[metric_col].to_numpy(dtype=float),
                    linestyle="None",
                    marker="o",
                    markersize=5.2,
                    markerfacecolor=fill_color,
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    zorder=5,
                )
            if len(validation) > 0:
                ax.plot(
                    validation["alpha_key"].to_numpy(dtype=float),
                    validation[metric_col].to_numpy(dtype=float),
                    linestyle="None",
                    marker="D",
                    markersize=5.7,
                    markerfacecolor=fill_color,
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                )
            if len(outside) > 0:
                ax.plot(
                    outside["alpha_key"].to_numpy(dtype=float),
                    outside[metric_col].to_numpy(dtype=float),
                    linestyle="None",
                    marker="s",
                    markersize=5.6,
                    markerfacecolor=fill_color,
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                )

            j_handles.append(
                Line2D([0], [0], color=series_color, linestyle="-", linewidth=1.6, label=rf"$J={j_value:.1f}$")
            )

        ax.set_xlabel(r"$\alpha$", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")

        model_handles = [
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.2, label="Fit"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Ext"),
        ]

        ax.legend(
            handles=j_handles + model_handles + _point_class_handles(),
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0,
            fancybox=False,
        )
        fig.subplots_adjust(left=0.10, right=0.78, top=0.96, bottom=0.14)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
        plt.close(fig)

    _plot_one(
        metric_col="CL_trim",
        solid_key="cl_solid",
        dash_key="cl_dash",
        ylabel=r"$C_L$",
        output_path=args.output_cl,
    )
    _plot_one(
        metric_col="CD_trim",
        solid_key="cd_solid",
        dash_key="cd_dash",
        ylabel=r"$C_D$",
        output_path=args.output_cd,
    )
    _plot_one(
        metric_col="CLCD_trim",
        solid_key="clcd_solid",
        dash_key="clcd_dash",
        ylabel=r"$C_L/C_D$",
        output_path=args.output,
    )

    print("Saved:")
    print(args.output_cm_alpha)
    print(args.output_clcd_deltae)
    print(args.output_cl)
    print(args.output_cd)
    print(args.output)
    print(args.summary_output)


if __name__ == "__main__":
    main()
