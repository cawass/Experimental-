"""Plot aerodynamic coefficients with the style of the provided reference figure.

Expected input format (text or CSV):
    alpha, cl_cd, cl, cd

If no input file is provided, the script uses built-in sample data that mimics
the reference plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_four_column_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load 4-column numeric data from a text/CSV file."""
    with path.open("r", encoding="utf-8") as file:
        first_data_line = ""
        for line in file:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                first_data_line = stripped
                break

    if not first_data_line:
        raise ValueError(f"No numeric content found in '{path}'.")

    delimiter = "," if "," in first_data_line else None

    try:
        data = np.loadtxt(path, comments="#", delimiter=delimiter)
    except ValueError:
        data = np.loadtxt(path, comments="#", delimiter=delimiter, skiprows=1)

    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    if data.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns in '{path}', got {data.shape[1]}.")

    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def plot_reference_style(
    alpha: np.ndarray,
    cl_cd: np.ndarray,
    cl: np.ndarray,
    cd: np.ndarray,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    """Create the styled dual-axis plot."""
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

    golden_ratio = (1 + np.sqrt(5)) / 2
    fig_height = 5.0
    fig_width = fig_height * golden_ratio
    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height), facecolor="white")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    ax1.set_facecolor("white")
    ax1.patch.set_alpha(1.0)

    for spine in ax1.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    ax2 = ax1.twinx()
    ax2.patch.set_alpha(0.0)
    for spine in ax2.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    line_cl_cd, = ax1.plot(
        alpha,
        cl_cd,
        color="#222222",
        linewidth=1.7,
        marker="^",
        markersize=5.0,
        markerfacecolor="none",
        label="Cl/Cd",
    )

    line_cl, = ax2.plot(
        alpha,
        cl,
        color="#80AFD4",
        linewidth=1.7,
        marker="o",
        markersize=4.5,
        markerfacecolor="none",
        label="Coefficient of Lift (Cl)",
    )

    line_cd, = ax2.plot(
        alpha,
        cd,
        color="#D89A63",
        linewidth=1.7,
        marker="o",
        markersize=4.5,
        markerfacecolor="none",
        label="Coefficient of Drag (Cd)",
    )

    ax1.set_xlim(0, 60)
    ax1.set_ylim(0, 14)
    ax2.set_ylim(0, 1.6)

    ax1.set_xticks(np.arange(0, 61, 10))
    ax1.set_yticks(np.arange(0, 15, 2))
    ax2.set_yticks(np.arange(0, 1.61, 0.2))

    ax1.set_xlabel("Angle of Attack(α)", fontweight="bold")
    ax1.set_ylabel("Cl/Cd", fontweight="bold")
    ax2.set_ylabel("Cl, Cd", fontweight="bold")

    ax1.xaxis.grid(False)
    ax1.yaxis.grid(True, which="major", linestyle="--", linewidth=0.7, color="#BDBDBD", alpha=0.9)

    ax1.legend(
        handles=[line_cl_cd, line_cl, line_cd],
        loc="center right",
        bbox_to_anchor=(0.97, 0.38),
        frameon=False,
        fontsize=9,
        handlelength=2.0,
        handletextpad=0.5,
    )

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor="white", transparent=False)

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a dual-axis aerodynamic plot with reference styling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to text/CSV file with columns: alpha, cl_cd, cl, cd",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("reference_style_plot.png"),
        help="Path to save the generated figure (default: PLOTING/reference_style_plot.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figure without opening the interactive window.",
    )
    args = parser.parse_args()

    if args.input is not None:
        alpha, cl_cd, cl, cd = _load_four_column_data(args.input)
    else:
        alpha = np.array([0, 2, 5, 10, 20, 28, 33, 38, 43, 50, 60], dtype=float)
        cl_cd = np.array([2.8, 8.2, 11.7, 9.2, 4.9, 3.5, 2.9, 2.4, 2.0, 1.6, 1.0], dtype=float)
        cl = np.array([0.0, 0.08, 0.28, 0.55, 1.02, 1.33, 1.42, 1.43, 1.40, 1.32, 1.02], dtype=float)
        cd = np.array([0.0, 0.01, 0.03, 0.07, 0.22, 0.45, 0.60, 0.75, 0.88, 1.00, 1.15], dtype=float)

    plot_reference_style(alpha, cl_cd, cl, cd, output_path=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
