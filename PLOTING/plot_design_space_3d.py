"""Create a 3D design-space plot with J on the z-axis and a zero-thrust plane."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _load_columns(path: Path) -> pd.DataFrame:
    """Load and clean the needed columns from the combined file."""
    try:
        df = pd.read_csv(path, sep="\t", skiprows=1)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python", skiprows=1)

    required = ["AoA_deg", "elevator_deflection_deg", "J_avg"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    out = df[required].copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    return out


def _prepare_points(data: pd.DataFrame, j_round_decimals: int) -> pd.DataFrame:
    """Round to design levels and compute point multiplicity."""
    out = data.copy()
    out["alpha"] = np.rint(out["AoA_deg"]).astype(int)
    out["delta_e"] = np.rint(out["elevator_deflection_deg"]).astype(int)
    out["J"] = np.round(out["J_avg"], j_round_decimals)

    counts = (
        out.groupby(["delta_e", "alpha", "J"], as_index=False)
        .size()
        .rename(columns={"size": "n_points"})
        .sort_values(["delta_e", "alpha", "J"], kind="stable")
        .reset_index(drop=True)
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 3D design-space plot (x=delta_e, y=alpha, z=J).")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("corrected_combined_output.txt"),
        help="Path to combined aerodynamic data file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("design_space_3d.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--j-round-decimals",
        type=int,
        default=2,
        help="Decimal precision used to group J levels.",
    )
    parser.add_argument(
        "--zero-thrust-j",
        type=float,
        default=2.5,
        help="J value used for the zero-thrust plane.",
    )
    args = parser.parse_args()

    raw = _load_columns(args.input)
    points = _prepare_points(raw, j_round_decimals=args.j_round_decimals)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig = plt.figure(figsize=(9.0, 6.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    scatter = ax.scatter(
        points["delta_e"],
        points["alpha"],
        points["J"],
        c=points["n_points"],
        cmap="viridis",
        s=62,
        marker="o",
        edgecolors="#222222",
        linewidths=0.45,
        depthshade=False,
    )

    x_min, x_max = points["delta_e"].min(), points["delta_e"].max()
    y_min, y_max = points["alpha"].min(), points["alpha"].max()
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, 2),
        np.linspace(y_min, y_max, 2),
    )
    z_grid = np.full_like(x_grid, fill_value=args.zero_thrust_j, dtype=float)

    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color="red",
        alpha=0.16,
        linewidth=0,
        antialiased=True,
    )

    # Dashed intersection edges improve visibility of the reference plane.
    ax.plot(
        [x_min, x_max],
        [y_min, y_min],
        [args.zero_thrust_j, args.zero_thrust_j],
        color="red",
        linestyle="--",
        linewidth=1.3,
    )
    ax.plot(
        [x_min, x_max],
        [y_max, y_max],
        [args.zero_thrust_j, args.zero_thrust_j],
        color="red",
        linestyle="--",
        linewidth=1.3,
    )

    ax.set_xlabel(r"$\delta_e$ (deg)", labelpad=10, fontweight="bold")
    ax.set_ylabel(r"$\alpha$ (deg)", labelpad=10, fontweight="bold")
    ax.set_zlabel(r"$J$", labelpad=8, fontweight="bold")

    ax.view_init(elev=22, azim=-56)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
    ax.text2D(0.03, 0.95, r"Zero thrust plane: $J_0=2.5$", transform=ax.transAxes, color="red")

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.08, shrink=0.85)
    colorbar.set_label("Number of points", fontweight="bold")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor="white", transparent=False)
    plt.close(fig)

    print("Saved:")
    print(args.output)


if __name__ == "__main__":
    main()
