from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


file_path = Path("BAL") / "compiled_data.xlsx"
data = pd.read_excel(file_path)

required_cols = {"source_file", "Alpha", "Cm_pitch"}
missing = required_cols.difference(data.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

plot_data = data.dropna(subset=["Alpha", "Cm_pitch"]).copy()


def split_internal_curves(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("Run_nr").reset_index(drop=True)

    curve_ids = []
    current_curve = 1
    seen_alpha = set()
    prev_alpha = None
    prev_v = None

    for _, row in group.iterrows():
        alpha = float(row["Alpha"])
        alpha_key = round(alpha, 2)
        v = float(row["V"]) if "V" in group.columns and pd.notna(row.get("V")) else None

        start_new_curve = False
        if prev_alpha is not None:
            if v is not None and prev_v is not None and abs(v - prev_v) > 5:
                start_new_curve = True
            elif len(seen_alpha) >= 4 and alpha_key in seen_alpha:
                start_new_curve = True
            elif len(seen_alpha) >= 4 and abs(alpha - prev_alpha) >= 8:
                start_new_curve = True

        if start_new_curve:
            current_curve += 1
            seen_alpha = set()

        curve_ids.append(current_curve)
        seen_alpha.add(alpha_key)
        prev_alpha = alpha
        if v is not None:
            prev_v = v

    group["curve_id"] = curve_ids
    return group


if "Run_nr" not in plot_data.columns:
    plot_data["Run_nr"] = range(1, len(plot_data) + 1)

segmented_groups = []
for _, source_group in plot_data.groupby("source_file"):
    segmented_groups.append(split_internal_curves(source_group))
plot_data = pd.concat(segmented_groups, ignore_index=True)

sources = sorted(plot_data["source_file"].unique())
fig, axes = plt.subplots(len(sources), 1, figsize=(10, 4 * len(sources)), sharex=True)
if len(sources) == 1:
    axes = [axes]

for ax, source in zip(axes, sources):
    source_data = plot_data[plot_data["source_file"] == source]
    for curve_id, curve in source_data.groupby("curve_id"):
        curve = (
            curve.groupby("Alpha", as_index=False)["Cm_pitch"]
            .mean()
            .sort_values("Alpha")
        )
        ax.plot(curve["Alpha"], curve["Cm_pitch"], marker="o", linewidth=1.6, label=f"curve {curve_id}")

    title = source.replace("corr_", "").replace(".txt", "")
    ax.set_title(title)
    ax.set_ylabel("Cm_pitch")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(ncol=2, fontsize=8)

axes[-1].set_xlabel("Alpha (deg)")
fig.suptitle("Cm-alpha Curves (All Curves per TXT File)", y=0.995)
fig.tight_layout()

output_plot = Path("BAL") / "cm_alpha_all_curves.png"
fig.savefig(output_plot, dpi=200)
print(f"Saved plot: {output_plot}")


def parse_elevator_deflection(source_file: str) -> int:
    match = re.search(r"elevator([p\-]?\d+)", source_file)
    if not match:
        raise ValueError(f"Could not parse elevator deflection from '{source_file}'")

    token = match.group(1)
    if token.startswith("p"):
        return int(token[1:])
    return int(token)


power_data = plot_data.copy()
power_data["delta_e_deg"] = power_data["source_file"].map(parse_elevator_deflection)
power_data["alpha_bin"] = power_data["Alpha"].round(1)

mean_curves = (
    power_data.groupby(["source_file", "delta_e_deg", "curve_id", "alpha_bin"], as_index=False)["Cm_pitch"]
    .mean()
)

power_rows = []
for (curve_id, alpha_bin), group in mean_curves.groupby(["curve_id", "alpha_bin"]):
    # Use linear fit Cm = a*delta_e + b -> elevator power is a = dCm/d(delta_e)
    if group["delta_e_deg"].nunique() < 2:
        continue
    x = group["delta_e_deg"].to_numpy(dtype=float)
    y = group["Cm_pitch"].to_numpy(dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    power_rows.append({"curve_id": int(curve_id), "Alpha": float(alpha_bin), "Cm_delta_e": slope})

power_df = pd.DataFrame(power_rows)
if power_df.empty:
    raise ValueError("Could not compute elevator power curves from available data.")

fig2, ax2 = plt.subplots(figsize=(10, 5))
for curve_id, curve in power_df.groupby("curve_id"):
    curve = curve.sort_values("Alpha")
    if len(curve) < 2:
        continue
    ax2.plot(curve["Alpha"], curve["Cm_delta_e"], marker="o", linewidth=1.8, label=f"curve {curve_id}")

ax2.set_title("Elevator Power Curves")
ax2.set_xlabel("Alpha (deg)")
ax2.set_ylabel("dCm / d(delta_e) (1/deg)")
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(ncol=3, fontsize=8)
fig2.tight_layout()

power_plot = Path("BAL") / "elevator_power_all_curves.png"
fig2.savefig(power_plot, dpi=200)
print(f"Saved plot: {power_plot}")
