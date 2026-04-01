from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data_Loader import add_operating_columns, read_bal_file


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

    for col in ["Run_nr", ALPHA_COLUMN, BETA_COLUMN, CM_COLUMN, VELOCITY_COLUMN]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Run_nr", ALPHA_COLUMN, BETA_COLUMN, CM_COLUMN]).copy()
    data["Run_nr"] = data["Run_nr"].astype(int)
    data = add_operating_columns(data)

    # Collapse tiny measurement jitter (-0.004, 7.996, 11.995, etc.) onto nominal alpha test points.
    data["alpha_nominal_deg"] = data[ALPHA_COLUMN].round(0).astype(int)
    return data


def join_unique_tags(values: pd.Series) -> str:
    tags = sorted({str(v) for v in values.dropna() if str(v).strip()})
    return "/".join(tags) if tags else "NA"


def build_curve_summary(data: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        data.groupby(
            ["delta_e_deg", "V_nominal", "prop_speed_hz", "alpha_nominal_deg"],
            as_index=False,
            dropna=False,
        )
        .agg(
            cm_mean=(CM_COLUMN, "mean"),
            cm_std=(CM_COLUMN, "std"),
            sample_count=(CM_COLUMN, "size"),
            beta_mean=(BETA_COLUMN, "mean"),
            j_mean=("advance_ratio_J", "mean"),
            j_nominal=("advance_ratio_nominal_J", "median"),
            test_block=("test_block", join_unique_tags),
        )
        .sort_values(["delta_e_deg", "V_nominal", "prop_speed_hz", "alpha_nominal_deg"])
    )

    grouped["cm_std"] = grouped["cm_std"].fillna(0.0)
    return grouped


def plot_curves(summary: pd.DataFrame, output_path: Path) -> None:
    elevator_levels = sorted(summary["delta_e_deg"].dropna().unique().astype(int).tolist())
    if not elevator_levels:
        raise ValueError("No elevator levels found for plotting.")

    fig, axes = plt.subplots(
        len(elevator_levels),
        1,
        figsize=(11.2, 3.6 * len(elevator_levels)),
        sharex=True,
        constrained_layout=True,
    )
    if len(elevator_levels) == 1:
        axes = [axes]

    marker_by_velocity = {20: "s", 40: "o"}
    style_cycle = ["-", "--", "-.", ":"]

    for ax, delta_e in zip(axes, elevator_levels):
        subplot = summary[summary["delta_e_deg"] == delta_e].copy()
        if subplot.empty:
            continue

        n_levels = sorted(subplot["prop_speed_hz"].dropna().unique().tolist())
        style_by_n = {float(n): style_cycle[i % len(style_cycle)] for i, n in enumerate(n_levels)}
        cmap = plt.get_cmap("tab10")

        for i, ((v_nominal, n_hz), curve) in enumerate(
            subplot.groupby(["V_nominal", "prop_speed_hz"], sort=True, dropna=False)
        ):
            curve = curve.sort_values("alpha_nominal_deg")
            total_samples = int(curve["sample_count"].sum())
            block_tag = join_unique_tags(curve["test_block"])
            beta_mean = float(curve["beta_mean"].mean())

            if pd.isna(n_hz):
                n_text = "n=FREE"
                j_text = "J=n/a"
                linestyle = "-"
            else:
                n_hz_float = float(n_hz)
                n_text = f"n={n_hz_float:.1f} Hz"
                j_value = float(curve["j_nominal"].median())
                if np.isnan(j_value):
                    j_value = float(curve["j_mean"].mean())
                j_text = f"J={j_value:.2f}"
                linestyle = style_by_n.get(n_hz_float, "-")

            marker = marker_by_velocity.get(int(round(float(v_nominal))), "o")
            label = (
                f"V~{int(round(float(v_nominal)))} m/s, {n_text}, {j_text}, "
                f"block={block_tag}, beta~{beta_mean:.1f}, samples={total_samples}"
            )
            color = cmap(i % 10)

            ax.plot(
                curve["alpha_nominal_deg"],
                curve["cm_mean"],
                linestyle=linestyle,
                marker=marker,
                color=color,
                linewidth=1.8,
                markersize=5.1,
                label=label,
            )
            ax.fill_between(
                curve["alpha_nominal_deg"],
                curve["cm_mean"] - curve["cm_std"],
                curve["cm_mean"] + curve["cm_std"],
                color=color,
                alpha=0.10,
                linewidth=0.0,
            )

        ax.set_title(f"delta_e = {delta_e:+d} deg")
        ax.set_ylabel("Cm (Cm_pitch)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(title="Curve meaning", loc="best", fontsize=7.8, title_fontsize=8.5, framealpha=0.95)

    axes[-1].set_xlabel("alpha (deg)")
    fig.suptitle("Corrected BAL Data: Cm-alpha Curves by Elevator and Advance Ratio")
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
