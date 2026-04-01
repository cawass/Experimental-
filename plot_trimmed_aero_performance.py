from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data_Loader import add_operating_columns, read_bal_file


ALPHA_COLUMN = "Alpha"
BETA_COLUMN = "Beta"
CM_COLUMN = "Cm_pitch"
CL_COLUMN = "CL"
CD_COLUMN = "CD"

TRIM_RANGE_TOL_DEG = 0.5


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

    numeric_cols = ["Run_nr", ALPHA_COLUMN, BETA_COLUMN, CM_COLUMN, CL_COLUMN, CD_COLUMN, "V"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Run_nr", ALPHA_COLUMN, CM_COLUMN, CL_COLUMN, CD_COLUMN]).copy()
    data["Run_nr"] = data["Run_nr"].astype(int)
    data["alpha_nominal_deg"] = data[ALPHA_COLUMN].round(0).astype(int)
    data = add_operating_columns(data)
    return data


def aggregate_by_condition_and_elevator(data: pd.DataFrame) -> pd.DataFrame:
    def join_unique_tags(values: pd.Series) -> str:
        tags = sorted({str(v) for v in values.dropna() if str(v).strip()})
        return "/".join(tags) if tags else "NA"

    grouped = (
        data.groupby(
            ["V_nominal", "prop_speed_hz", "alpha_nominal_deg", "delta_e_deg"],
            as_index=False,
            dropna=False,
        )
        .agg(
            cm_mean=(CM_COLUMN, "mean"),
            cl_mean=(CL_COLUMN, "mean"),
            cd_mean=(CD_COLUMN, "mean"),
            beta_mean=(BETA_COLUMN, "mean"),
            j_mean=("advance_ratio_J", "mean"),
            j_nominal=("advance_ratio_nominal_J", "median"),
            sample_count=("Run_nr", "size"),
            test_block=("test_block", join_unique_tags),
        )
        .sort_values(["V_nominal", "prop_speed_hz", "alpha_nominal_deg", "delta_e_deg"])
    )
    return grouped


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def compute_trimmed_performance(grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, block in grouped.groupby(["V_nominal", "prop_speed_hz", "alpha_nominal_deg"], dropna=False):
        block = block.sort_values("delta_e_deg")
        if block["delta_e_deg"].nunique() < 2:
            continue

        x = block["delta_e_deg"].to_numpy(dtype=float)
        cm = block["cm_mean"].to_numpy(dtype=float)
        cl = block["cl_mean"].to_numpy(dtype=float)
        cd = block["cd_mean"].to_numpy(dtype=float)

        m_cm, b_cm = linear_fit(x, cm)
        if abs(m_cm) < 1e-12:
            continue

        m_cl, b_cl = linear_fit(x, cl)
        m_cd, b_cd = linear_fit(x, cd)

        delta_e_trim = -b_cm / m_cm
        cl_trim = m_cl * delta_e_trim + b_cl
        cd_trim = m_cd * delta_e_trim + b_cd
        ld_trim = np.nan if abs(cd_trim) < 1e-12 else cl_trim / cd_trim

        cm_fit = m_cm * x + b_cm
        ss_res = float(np.sum((cm - cm_fit) ** 2))
        ss_tot = float(np.sum((cm - cm.mean()) ** 2))
        r_squared_cm = np.nan if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot

        de_min = float(np.min(x))
        de_max = float(np.max(x))
        trim_within_range = (de_min - TRIM_RANGE_TOL_DEG) <= delta_e_trim <= (de_max + TRIM_RANGE_TOL_DEG)

        j_candidates = block["j_nominal"].dropna()
        j_nominal = float(j_candidates.median()) if not j_candidates.empty else np.nan
        j_measured = float(block["j_mean"].mean())

        rows.append(
            {
                "V_nominal": float(keys[0]),
                "prop_speed_hz": float(keys[1]) if pd.notna(keys[1]) else np.nan,
                "test_block": "/".join(sorted({str(v) for v in block["test_block"] if str(v) != "nan"})),
                "alpha_nominal_deg": int(keys[2]),
                "delta_e_trim_deg": float(delta_e_trim),
                "delta_e_data_min_deg": de_min,
                "delta_e_data_max_deg": de_max,
                "trim_within_data_range": bool(trim_within_range),
                "cl_trim": float(cl_trim),
                "cd_trim": float(cd_trim),
                "ld_trim": float(ld_trim),
                "beta_mean_deg": float(block["beta_mean"].mean()),
                "j_nominal": j_nominal,
                "j_measured_mean": j_measured,
                "n_elevator_levels": int(block["delta_e_deg"].nunique()),
                "sample_count": int(block["sample_count"].sum()),
                "r_squared_cm_fit": r_squared_cm,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["V_nominal", "prop_speed_hz", "alpha_nominal_deg"])


def condition_label(v_nominal: float, prop_speed_hz: float | float, j_nominal: float) -> str:
    v_text = f"V~{int(round(v_nominal))} m/s"
    if pd.isna(prop_speed_hz):
        return f"{v_text}, n=FREE, J=n/a"
    if np.isnan(j_nominal):
        return f"{v_text}, n={prop_speed_hz:.1f} Hz"
    return f"{v_text}, n={prop_speed_hz:.1f} Hz, J={j_nominal:.2f}"


def plot_trimmed_performance(trimmed: pd.DataFrame, output_path: Path) -> None:
    valid = trimmed[trimmed["trim_within_data_range"]].copy()
    if valid.empty:
        raise ValueError("No trimmed points within the available elevator range.")

    fig, (ax_polar, ax_ld) = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)

    conditions = list(valid.groupby(["V_nominal", "prop_speed_hz"], dropna=False))
    cmap = plt.get_cmap("tab10")
    markers_by_v = {20: "s", 40: "o"}

    for i, ((v_nominal, n_hz), curve) in enumerate(conditions):
        curve = curve.sort_values("alpha_nominal_deg")
        color = cmap(i % 10)
        j_for_label_candidates = curve["j_nominal"].dropna()
        j_for_label = float(j_for_label_candidates.median()) if not j_for_label_candidates.empty else np.nan
        label = condition_label(float(v_nominal), float(n_hz) if pd.notna(n_hz) else np.nan, j_for_label)
        marker = markers_by_v.get(int(round(float(v_nominal))), "o")

        ax_polar.plot(
            curve["cd_trim"],
            curve["cl_trim"],
            marker=marker,
            linewidth=1.8,
            markersize=5.0,
            color=color,
            label=label,
        )
        ax_ld.plot(
            curve["alpha_nominal_deg"],
            curve["ld_trim"],
            marker=marker,
            linewidth=1.8,
            markersize=5.0,
            color=color,
            label=label,
        )

    ax_polar.set_title("Trimmed Polar")
    ax_polar.set_xlabel("CD_trim")
    ax_polar.set_ylabel("CL_trim")
    ax_polar.grid(True, linestyle="--", alpha=0.35)

    ax_ld.set_title("Trimmed L/D vs Alpha")
    ax_ld.set_xlabel("alpha (deg)")
    ax_ld.set_ylabel("(L/D)_trim")
    ax_ld.grid(True, linestyle="--", alpha=0.35)

    ax_ld.legend(title="Condition", fontsize=8.0, title_fontsize=8.8, loc="best", framealpha=0.95)
    fig.suptitle("Aerodynamic Performance in Trimmed Condition (Cm = 0)")
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

    grouped = aggregate_by_condition_and_elevator(data)
    trimmed = compute_trimmed_performance(grouped)
    if trimmed.empty:
        raise ValueError("Unable to compute trimmed performance from available corrected data.")

    summary_path = bal_dir / "trimmed_aero_performance_summary.csv"
    trimmed.to_csv(summary_path, index=False)

    plot_path = bal_dir / "trimmed_aero_performance.png"
    plot_trimmed_performance(trimmed, plot_path)

    valid_count = int(trimmed["trim_within_data_range"].sum())
    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Trimmed points within elevator range: {valid_count} / {len(trimmed)}")


if __name__ == "__main__":
    main()
