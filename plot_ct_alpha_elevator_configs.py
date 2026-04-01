from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path("BAL") / "compiled_data.xlsx"
OUTPUT_PLOT = Path("BAL") / "ct_alpha_elevator_configs.png"
OUTPUT_SUMMARY = Path("BAL") / "ct_alpha_elevator_configs_summary.csv"
TARGET_ELEVATOR_CONFIGS = np.array([-10, 0, 10], dtype=int)


def parse_elevator_deflection(source_file: str) -> int:
    match = re.search(r"elevator([p\-]?\d+)", source_file)
    if not match:
        raise ValueError(f"Could not parse elevator deflection from '{source_file}'.")
    token = match.group(1)
    return int(token[1:]) if token.startswith("p") else int(token)


def round_to_elevator_config(delta_e: float) -> int:
    idx = int(np.argmin(np.abs(TARGET_ELEVATOR_CONFIGS - float(delta_e))))
    return int(TARGET_ELEVATOR_CONFIGS[idx])


def main() -> None:
    data = pd.read_excel(INPUT_FILE)

    required_cols = {"source_file", "Alpha", "Ct"}
    missing = required_cols.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_FILE}: {sorted(missing)}")

    data["Alpha"] = pd.to_numeric(data["Alpha"], errors="coerce")
    data["Ct"] = pd.to_numeric(data["Ct"], errors="coerce")
    data = data.dropna(subset=["source_file", "Alpha", "Ct"]).copy()

    data["delta_e_raw"] = data["source_file"].map(parse_elevator_deflection)
    data["delta_e_cfg"] = data["delta_e_raw"].map(round_to_elevator_config)

    # Collapse tiny alpha jitter (e.g., 8.004) onto nominal points.
    data["alpha_nominal_deg"] = data["Alpha"].round(0).astype(int)

    summary = (
        data.groupby(["delta_e_cfg", "alpha_nominal_deg"], as_index=False)
        .agg(
            ct_mean=("Ct", "mean"),
            ct_std=("Ct", "std"),
            sample_count=("Ct", "size"),
        )
        .sort_values(["delta_e_cfg", "alpha_nominal_deg"])
    )
    summary["ct_std"] = summary["ct_std"].fillna(0.0)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    fig, ax = plt.subplots(figsize=(10.6, 6.0), constrained_layout=True)
    color_by_cfg = {-10: "#1f77b4", 0: "#2ca02c", 10: "#d62728"}

    for cfg in [-10, 0, 10]:
        curve = summary[summary["delta_e_cfg"] == cfg].copy()
        if curve.empty:
            continue
        curve = curve.sort_values("alpha_nominal_deg")
        color = color_by_cfg.get(cfg, "black")

        ax.plot(
            curve["alpha_nominal_deg"],
            curve["ct_mean"],
            marker="o",
            linewidth=1.9,
            markersize=5.0,
            color=color,
            label=f"elevator {cfg:+d} deg",
        )
        ax.fill_between(
            curve["alpha_nominal_deg"],
            curve["ct_mean"] - curve["ct_std"],
            curve["ct_mean"] + curve["ct_std"],
            color=color,
            alpha=0.14,
            linewidth=0.0,
        )

    ax.set_title("Ct vs Alpha for Elevator Configurations (-10, 0, +10)")
    ax.set_xlabel("Alpha (deg)")
    ax.set_ylabel("Ct")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Rounded elevator config", framealpha=0.95)

    fig.savefig(OUTPUT_PLOT, dpi=240)
    plt.close(fig)

    print(f"Saved summary: {OUTPUT_SUMMARY}")
    print(f"Saved plot: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
