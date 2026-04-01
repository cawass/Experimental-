from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data_Loader import add_operating_columns, read_bal_file


CM_COLUMN = "Cm_pitch"
ALPHA_COLUMN = "Alpha"
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


def join_unique_tags(values: pd.Series) -> str:
    tags = sorted({str(v) for v in values.dropna() if str(v).strip()})
    return "/".join(tags) if tags else "NA"


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

    for col in ["Run_nr", ALPHA_COLUMN, BETA_COLUMN, CM_COLUMN]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Run_nr", ALPHA_COLUMN, CM_COLUMN]).copy()
    data["Run_nr"] = data["Run_nr"].astype(int)
    data["alpha_nominal_deg"] = data[ALPHA_COLUMN].round(0).astype(int)
    data = add_operating_columns(data)
    return data


def build_condition_cm_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
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
        .sort_values(["V_nominal", "prop_speed_hz", "alpha_nominal_deg", "delta_e_deg"])
    )
    summary["cm_std"] = summary["cm_std"].fillna(0.0)
    return summary


def compute_effectiveness(condition_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = condition_summary.groupby(["V_nominal", "prop_speed_hz", "alpha_nominal_deg"], dropna=False)
    for (v_nominal, n_hz, alpha_deg), group in grouped:
        group = group.sort_values("delta_e_deg")
        if group["delta_e_deg"].nunique() < 2:
            continue

        x = group["delta_e_deg"].to_numpy(dtype=float)
        y = group["cm_mean"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x + intercept
        ss_res = float(np.sum((y - y_fit) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = np.nan if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot

        if pd.isna(n_hz):
            j_nominal = np.nan
        else:
            j_candidates = group["j_nominal"].dropna()
            if not j_candidates.empty:
                j_nominal = float(j_candidates.median())
            else:
                j_mean_candidates = group["j_mean"].dropna()
                j_nominal = float(j_mean_candidates.mean()) if not j_mean_candidates.empty else np.nan

        rows.append(
            {
                "V_nominal": float(v_nominal),
                "prop_speed_hz": float(n_hz) if pd.notna(n_hz) else np.nan,
                "alpha_nominal_deg": int(alpha_deg),
                "cm_delta_e_per_deg": float(slope),
                "r_squared": r_squared,
                "n_elevator_levels": int(group["delta_e_deg"].nunique()),
                "sample_count": int(group["sample_count"].sum()),
                "j_nominal": j_nominal,
                "test_block": join_unique_tags(group["test_block"]),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["V_nominal", "prop_speed_hz", "alpha_nominal_deg"])


def plot_effectiveness(effectiveness: pd.DataFrame, output_path: Path) -> None:
    velocities = sorted(effectiveness["V_nominal"].dropna().unique().tolist())
    if not velocities:
        raise ValueError("No velocity groups found for elevator effectiveness plotting.")

    fig, axes = plt.subplots(
        len(velocities),
        1,
        figsize=(10.8, 3.8 * len(velocities)),
        sharex=True,
        constrained_layout=True,
    )
    if len(velocities) == 1:
        axes = [axes]

    style_cycle = ["-", "--", "-.", ":"]
    for ax, velocity in zip(axes, velocities):
        subset = effectiveness[effectiveness["V_nominal"] == velocity].copy()
        n_levels = sorted(subset["prop_speed_hz"].dropna().unique().tolist())
        style_by_n = {float(n): style_cycle[i % len(style_cycle)] for i, n in enumerate(n_levels)}
        cmap = plt.get_cmap("tab10")

        for i, (n_hz, curve) in enumerate(subset.groupby("prop_speed_hz", dropna=False, sort=True)):
            curve = curve.sort_values("alpha_nominal_deg")
            color = cmap(i % 10)

            if pd.isna(n_hz):
                label = (
                    "n=FREE, J=n/a, "
                    f"block={join_unique_tags(curve['test_block'])}, "
                    f"elev_levels~{int(curve['n_elevator_levels'].max())}"
                )
                linestyle = "-"
            else:
                n_hz_float = float(n_hz)
                j_candidates = curve["j_nominal"].dropna()
                j_value = float(j_candidates.median()) if not j_candidates.empty else float("nan")
                label = (
                    f"n={n_hz_float:.1f} Hz, J={j_value:.2f}, "
                    f"block={join_unique_tags(curve['test_block'])}, "
                    f"elev_levels~{int(curve['n_elevator_levels'].max())}"
                )
                linestyle = style_by_n.get(n_hz_float, "-")

            ax.plot(
                curve["alpha_nominal_deg"],
                curve["cm_delta_e_per_deg"],
                marker="o",
                linestyle=linestyle,
                linewidth=1.9,
                markersize=5.0,
                color=color,
                label=label,
            )

        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.55)
        ax.set_title(f"V ~ {int(round(float(velocity)))} m/s")
        ax.set_ylabel("dCm/d(delta_e) [1/deg]")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(title="Condition", fontsize=8.0, title_fontsize=8.7, loc="best", framealpha=0.95)

    axes[-1].set_xlabel("alpha (deg)")
    fig.suptitle("Elevator Effectiveness from Corrected Data")
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

    condition_summary = build_condition_cm_summary(data)
    effectiveness = compute_effectiveness(condition_summary)
    if effectiveness.empty:
        raise ValueError("Not enough overlapping elevator conditions to compute effectiveness.")

    summary_path = bal_dir / "elevator_effectiveness_summary.csv"
    effectiveness.to_csv(summary_path, index=False)

    output_path = bal_dir / "elevator_effectiveness_curves.png"
    plot_effectiveness(effectiveness, output_path)

    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
