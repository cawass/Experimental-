from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data_Loader import read_bal_file


CM_COLUMN = "Cm_pitch"
ALPHA_COLUMN = "Alpha"
VELOCITY_COLUMN = "V"
BETA_COLUMN = "Beta"


def resolve_base_dir() -> Path:
    cwd = Path.cwd()
    if (cwd / "BAL").exists():
        return cwd
    return Path(__file__).resolve().parent


def parse_elevator_deflection(file_name: str) -> int:
    match = re.search(r"elevator([p\-]?\d+)", file_name)
    if not match:
        raise ValueError(f"Could not parse elevator deflection from '{file_name}'.")
    token = match.group(1)
    return int(token[1:]) if token.startswith("p") else int(token)


def load_corrected_rows(bal_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for file_path in sorted(bal_dir.glob("corr_*.txt")):
        parsed_rows = read_bal_file(file_path)
        for row in parsed_rows:
            row["source_file"] = file_path.name
            row["delta_e_deg"] = parse_elevator_deflection(file_path.name)
        rows.extend(parsed_rows)

    data = pd.DataFrame(rows)
    if data.empty:
        return data

    for col in [ALPHA_COLUMN, BETA_COLUMN, CM_COLUMN, VELOCITY_COLUMN]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=[ALPHA_COLUMN, BETA_COLUMN, CM_COLUMN, VELOCITY_COLUMN]).copy()

    # Collapse tiny measurement jitter (-0.004, 7.996, 11.995, etc.) onto nominal alpha test points.
    data["alpha_nominal_deg"] = data[ALPHA_COLUMN].round(0).astype(int)
    data["V_nominal_ms"] = np.where(data[VELOCITY_COLUMN] < 30.0, 20, 40)
    return data


def build_curve_summary(data: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        data.groupby(["delta_e_deg", "V_nominal_ms", "alpha_nominal_deg"], as_index=False)
        .agg(
            cm_mean=(CM_COLUMN, "mean"),
            cm_std=(CM_COLUMN, "std"),
            sample_count=(CM_COLUMN, "size"),
            beta_mean=(BETA_COLUMN, "mean"),
        )
        .sort_values(["delta_e_deg", "V_nominal_ms", "alpha_nominal_deg"])
    )

    grouped["cm_std"] = grouped["cm_std"].fillna(0.0)
    return grouped


def plot_curves(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.0), constrained_layout=True)

    colors = {-10: "#1f77b4", 0: "#ff7f0e", 10: "#2ca02c"}
    line_styles = {20: "--", 40: "-"}
    markers = {20: "s", 40: "o"}

    for (delta_e_deg, v_nominal), curve in summary.groupby(["delta_e_deg", "V_nominal_ms"], sort=True):
        curve = curve.sort_values("alpha_nominal_deg")
        label = (
            f"delta_e={delta_e_deg:+d} deg, "
            f"V~{v_nominal:d} m/s, "
            f"beta~{curve['beta_mean'].mean():.1f} deg, "
            f"samples={int(curve['sample_count'].sum())}"
        )

        ax.plot(
            curve["alpha_nominal_deg"],
            curve["cm_mean"],
            linestyle=line_styles.get(v_nominal, "-"),
            marker=markers.get(v_nominal, "o"),
            color=colors.get(delta_e_deg, None),
            linewidth=1.9,
            markersize=5.5,
            label=label,
        )
        ax.fill_between(
            curve["alpha_nominal_deg"],
            curve["cm_mean"] - curve["cm_std"],
            curve["cm_mean"] + curve["cm_std"],
            color=colors.get(delta_e_deg, "#666666"),
            alpha=0.12,
            linewidth=0.0,
        )

    ax.set_title("Corrected BAL Data: Cm vs Alpha Curves")
    ax.set_xlabel("alpha (deg)")
    ax.set_ylabel("Cm (Cm_pitch)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Curve meaning", loc="best", fontsize=8.8, title_fontsize=9.2, framealpha=0.95)

    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def main() -> None:
    base_dir = resolve_base_dir()
    bal_dir = base_dir / "BAL"
    if not bal_dir.exists():
        raise FileNotFoundError(f"BAL directory not found: {bal_dir}")

    data = load_corrected_rows(bal_dir)
    if data.empty:
        raise ValueError("No corrected rows found in BAL/corr_*.txt")

    summary = build_curve_summary(data)

    summary_path = bal_dir / "cm_alpha_corrected_summary.csv"
    summary.to_csv(summary_path, index=False)

    output_path = bal_dir / "cm_alpha_corrected_curves.png"
    plot_curves(summary, output_path)

    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
