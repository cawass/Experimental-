"""Create design-space plots from uncorrected aerodynamic variables.

The script reads the combined BAL export and uses only uncorrected fields:
    - AoA_deg
    - elevator_deflection_deg
    - J_avg

Outputs (PNG):
    1) side-by-side figure with:
        - delta_elevator vs alpha
        - J vs alpha
        - J vs delta_elevator

Point colors represent how many repeated points exist for each plotted pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


def _load_uncorrected_columns(path: Path) -> pd.DataFrame:
    """Load the required uncorrected columns from the combined export file."""
    try:
        df = pd.read_csv(path, sep="\t", skiprows=1)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python", skiprows=1)

    required_columns = ["AoA_deg", "elevator_deflection_deg", "J_avg"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required columns in '{path}': {missing_text}")

    clean = df[required_columns].copy()
    for column in required_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna()

    return clean


def _prepare_design_keys(data: pd.DataFrame, j_round_decimals: int) -> pd.DataFrame:
    """Snap noisy measurements to design-space keys for robust repeat counting."""
    result = data.copy()
    result["alpha_key"] = np.rint(result["AoA_deg"]).astype(int)
    result["delta_e_key"] = np.rint(result["elevator_deflection_deg"]).astype(int)
    result["J_key"] = np.round(result["J_avg"], j_round_decimals)
    return result


def _add_repeat_counts(data: pd.DataFrame) -> pd.DataFrame:
    """Add repeat-count columns for each pair of plotted design variables."""
    result = data.copy()
    result["repeat_alpha_delta_e"] = (
        result.groupby(["alpha_key", "delta_e_key"])["alpha_key"].transform("size").astype(int)
    )
    result["repeat_alpha_J"] = (
        result.groupby(["alpha_key", "J_key"])["alpha_key"].transform("size").astype(int)
    )
    result["repeat_J_delta_e"] = (
        result.groupby(["J_key", "delta_e_key"])["alpha_key"].transform("size").astype(int)
    )
    return result


def _apply_reference_style(ax: plt.Axes) -> None:
    """Use the same styling conventions as plot_reference_style.py."""
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    ax.set_facecolor("white")
    ax.patch.set_alpha(1.0)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.7, color="#BDBDBD", alpha=0.9)


def _scatter_design_space(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    count_col: str,
    x_label: str,
    y_label: str,
    title: str,
    ax: plt.Axes,
    norm: Normalize,
) -> plt.Collection:
    _apply_reference_style(ax)

    scatter = ax.scatter(
        data[x_col],
        data[y_col],
        c=data[count_col],
        cmap="viridis",
        norm=norm,
        s=58,
        marker="o",
        edgecolors="#222222",
        linewidths=0.5,
    )

    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")

    return scatter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create design-space scatter plots using uncorrected aerodynamic variables."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("corrected_combined_output.txt"),
        help="Combined BAL export file path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory where plot images are saved.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="design_space_uncorrected_side_by_side.png",
        help="Filename for the combined side-by-side figure.",
    )
    parser.add_argument(
        "--j-round-decimals",
        type=int,
        default=2,
        help="Decimal precision used to group J for repeat counting.",
    )
    parser.add_argument(
        "--zero-thrust-j",
        type=float,
        default=2.5,
        help="J value used for the zero-thrust reference line.",
    )
    args = parser.parse_args()

    raw = _load_uncorrected_columns(args.input)
    keyed = _prepare_design_keys(raw, j_round_decimals=args.j_round_decimals)
    data = _add_repeat_counts(keyed)

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

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), facecolor="white", constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)

    plot_configs = [
        {
            "x_col": "alpha_key",
            "y_col": "delta_e_key",
            "count_col": "repeat_alpha_delta_e",
            "x_label": r"$\alpha$ (deg)",
            "y_label": r"$\delta_e$ (deg)",
            "title": "",
            "zero_thrust_horizontal": False,
        },
        {
            "x_col": "alpha_key",
            "y_col": "J_key",
            "count_col": "repeat_alpha_J",
            "x_label": r"$\alpha$ (deg)",
            "y_label": r"$J$",
            "title": "",
            "zero_thrust_horizontal": True,
        },
        {
            "x_col": "delta_e_key",
            "y_col": "J_key",
            "count_col": "repeat_J_delta_e",
            "x_label": r"$\delta_e$ (deg)",
            "y_label": r"$J$",
            "title": "",
            "zero_thrust_horizontal": True,
        },
    ]

    count_columns = ["repeat_alpha_delta_e", "repeat_alpha_J", "repeat_J_delta_e"]
    count_min = min(float(data[col].min()) for col in count_columns)
    count_max = max(float(data[col].max()) for col in count_columns)
    shared_norm = Normalize(vmin=count_min, vmax=count_max)

    scatter_for_colorbar = None
    for ax, cfg in zip(axes, plot_configs):
        scatter_for_colorbar = _scatter_design_space(
            data=data,
            x_col=cfg["x_col"],
            y_col=cfg["y_col"],
            count_col=cfg["count_col"],
            x_label=cfg["x_label"],
            y_label=cfg["y_label"],
            title=cfg["title"],
            ax=ax,
            norm=shared_norm,
        )
        if cfg["zero_thrust_horizontal"]:
            line = ax.axhline(
                y=args.zero_thrust_j,
                color="red",
                linewidth=1.3,
                linestyle="--",
                label=rf"Zero-thrust line ($J_0={args.zero_thrust_j:g}$)",
            )
            ax.legend(handles=[line], loc="upper right", frameon=False)

    if scatter_for_colorbar is not None:
        colorbar = fig.colorbar(scatter_for_colorbar, ax=axes, pad=0.02)
        colorbar.set_label("Number of points", fontweight="bold")
        colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        colorbar.update_ticks()

    output_path = args.output_dir / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", transparent=False)
    plt.close(fig)

    print("Saved:")
    print(output_path)


if __name__ == "__main__":
    main()
