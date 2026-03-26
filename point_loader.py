from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float | None = None


class PointCloud:
    def __init__(self, points: Iterable[Point] | None = None) -> None:
        self.points = list(points or [])

    @classmethod
    def from_txt(cls, file_path: str | Path) -> "PointCloud":
        path = Path(file_path)
        points: list[Point] = []

        with path.open("r", encoding="utf-8") as file:
            for line_no, raw_line in enumerate(file, start=1):
                line = raw_line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = [p for p in re.split(r"[,\s]+", line) if p]
                if len(parts) not in (2, 3):
                    raise ValueError(
                        f"Invalid point format on line {line_no}: '{raw_line.rstrip()}'"
                    )

                try:
                    values = [float(value) for value in parts]
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric value on line {line_no}: '{raw_line.rstrip()}'"
                    ) from exc

                if len(values) == 2:
                    points.append(Point(values[0], values[1]))
                else:
                    points.append(Point(values[0], values[1], values[2]))

        return cls(points)

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def angle_of_attack_degrees(self) -> list[float]:
        if not self.points:
            raise ValueError("No points available to compute angle of attack.")

        return [math.degrees(math.atan2(point.y, point.x)) for point in self.points]

    def plot_angle_of_attack(
        self,
        save_path: str | Path = "angle_of_attack.png",
        show: bool = False,
    ) -> Path:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required to plot angle of attack. Install it with "
                "'pip install matplotlib'."
            ) from exc

        angles = self.angle_of_attack_degrees()
        x_axis = range(1, len(angles) + 1)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(x_axis, angles, marker="o", linewidth=1.5)
        ax.set_title("Angle of Attack per Point")
        ax.set_xlabel("Point Index")
        ax.set_ylabel("Angle of Attack (deg)")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        output_path = Path(save_path)
        fig.savefig(output_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)

        return output_path


def _split_columns(line: str) -> list[str]:
    return [part for part in re.split(r"\s+", line.strip()) if part]


def _file_has_bal_header(file_path: str | Path) -> bool:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as file:
        for _ in range(6):
            line = file.readline()
            if not line:
                break
            if "Run_nr" in line and "Alpha" in line:
                return True
    return False


def load_bal_alpha(file_path: str | Path) -> list[float]:
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8").splitlines()

    alpha_column: int | None = None
    header_index: int | None = None

    for line_no, line in enumerate(lines):
        if "Run_nr" in line and "Alpha" in line:
            columns = _split_columns(line)
            try:
                alpha_column = columns.index("Alpha")
            except ValueError as exc:
                raise ValueError(f"Could not find 'Alpha' column in {path.name}.") from exc
            header_index = line_no
            break

    if alpha_column is None or header_index is None:
        raise ValueError(f"BAL header with 'Alpha' not found in {path.name}.")

    alpha_values: list[float] = []
    for line in lines[header_index + 1 :]:
        if not line.strip():
            continue
        parts = _split_columns(line)
        if len(parts) <= alpha_column:
            continue
        try:
            alpha_values.append(float(parts[alpha_column]))
        except ValueError:
            # Skip unit/header rows and any malformed lines.
            continue

    if not alpha_values:
        raise ValueError(f"No numeric Alpha values found in {path.name}.")

    return alpha_values


def load_bal_alpha_series(path: str | Path) -> dict[str, list[float]]:
    target = Path(path)
    if target.is_dir():
        files = sorted(target.glob("*.txt"))
        if not files:
            raise ValueError(f"No .txt files found in folder: {target}")
    elif target.is_file():
        files = [target]
    else:
        raise FileNotFoundError(f"Input path does not exist: {target}")

    series: dict[str, list[float]] = {}
    errors: list[str] = []

    for file_path in files:
        try:
            series[file_path.stem] = load_bal_alpha(file_path)
        except ValueError as exc:
            errors.append(f"{file_path.name}: {exc}")

    if not series:
        details = "; ".join(errors) if errors else "No readable BAL files."
        raise ValueError(f"Could not load any BAL Alpha series. {details}")

    return series


def plot_bal_alpha_series(
    series: dict[str, list[float]],
    save_path: str | Path = "bal_alpha.png",
    show: bool = False,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to plot angle of attack. Install it with "
            "'pip install matplotlib'."
        ) from exc

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, values in series.items():
        x_axis = range(1, len(values) + 1)
        ax.plot(x_axis, values, marker="o", markersize=3, linewidth=1.2, label=label)

    ax.set_title("Angle of Attack (Alpha) - BAL Data")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Alpha (deg)")
    ax.grid(True, linestyle="--", alpha=0.4)
    if len(series) > 1:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    output_path = Path(save_path)
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return output_path


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot angle of attack from point files or BAL table files."
    )
    parser.add_argument(
        "input_path",
        help="Input .txt file or folder (BAL folder supported).",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "points", "bal"),
        default="auto",
        help="Input format: auto-detect, points (x y [z]), or BAL table files.",
    )
    parser.add_argument(
        "--out",
        default="angle_of_attack.png",
        help="Output image path (default: angle_of_attack.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the figure.",
    )
    return parser


if __name__ == "__main__":
    args = _build_cli().parse_args()
    input_path = Path(args.input_path)

    source = args.source
    if source == "auto":
        if input_path.is_dir():
            source = "bal"
        elif input_path.is_file() and _file_has_bal_header(input_path):
            source = "bal"
        else:
            source = "points"

    if source == "points":
        cloud = PointCloud.from_txt(input_path)
        saved_plot = cloud.plot_angle_of_attack(save_path=args.out, show=args.show)
    else:
        alpha_series = load_bal_alpha_series(input_path)
        saved_plot = plot_bal_alpha_series(alpha_series, save_path=args.out, show=args.show)

    print(f"Saved angle-of-attack plot to: {saved_plot}")
