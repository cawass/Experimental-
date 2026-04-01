from pathlib import Path
import re

import numpy as np
import pandas as pd


PROP_DIAMETER_M = 0.2032

# (run_start, run_end, V_nominal [m/s], n_hz [Hz], label)
RUN_SCHEDULE = [
    (1, 5, 40.0, None, "FREE"),
    (6, 10, 40.0, 39.4, "BRK"),
    (11, 15, 40.0, 55.1, "BRK"),
    (16, 17, 40.0, 78.7, "ZERO"),
    (18, 18, 40.0, 123.0, "MANDATORY"),
    (19, 21, 40.0, 78.7, "ZERO"),
    (22, 26, 20.0, None, "FREE"),
    (27, 31, 20.0, 27.6, "BRK"),
    (32, 32, 40.0, 78.7, "REPLICATE"),
    (33, 37, 40.0, 39.4, "BRK"),
    (38, 42, 40.0, 55.1, "BRK"),
    (43, 47, 40.0, 78.7, "ZERO"),
    (48, 52, 40.0, None, "FREE"),
    (53, 57, 20.0, None, "FREE"),
    (58, 62, 20.0, 27.6, "BRK"),
    (63, 63, 40.0, 78.7, "REPLICATE"),
    (64, 64, 40.0, 123.0, "MANDATORY"),
    (65, 69, 40.0, 39.4, "CAL"),
    (70, 74, 40.0, 78.7, "CAL"),
    (75, 79, 20.0, 27.6, "CAL"),
    (80, 84, 20.0, 27.6, "BRK"),
    (85, 89, 20.0, 22.0, "BRK"),
    (90, 94, 40.0, None, "FREE"),
]


def parse_value(token: str):
    token = token.strip()
    if token in {"", "/"}:
        return None

    try:
        value = float(token)
    except ValueError:
        return token

    if re.fullmatch(r"[+-]?\d+", token):
        return int(value)
    return value


def read_bal_file(file_path: Path) -> list[dict]:
    lines = file_path.read_text(encoding="utf-8").splitlines()

    header_index = None
    columns = None
    for i, line in enumerate(lines):
        if "Run_nr" in line and "Alpha" in line:
            header_index = i
            columns = [part for part in re.split(r"\s+", line.strip()) if part]
            break

    if header_index is None or columns is None:
        return []

    rows = []
    for line in lines[header_index + 1 :]:
        if not line.strip():
            continue

        parts = [part for part in re.split(r"\s+", line.strip()) if part]
        if not parts or parts[0] in {"/", "Run_nr"} or len(parts) < len(columns):
            continue

        row = {"source_file": file_path.name}
        for idx, col in enumerate(columns):
            row[col] = parse_value(parts[idx])

        run_nr = row.get("Run_nr")
        if run_nr in {None, 0}:
            continue

        rows.append(row)

    return rows


def build_compiled_dataframe(bal_dir: Path) -> pd.DataFrame:
    rows = []
    for file_path in sorted(bal_dir.glob("*.txt")):
        if file_path.name.lower().startswith("zer_"):
            continue
        rows.extend(read_bal_file(file_path))

    data = pd.DataFrame(rows)
    if data.empty:
        return data

    if "Run_nr" in data.columns:
        data["Run_nr"] = pd.to_numeric(data["Run_nr"], errors="coerce")
        data = data.dropna(subset=["Run_nr"])
        data["Run_nr"] = data["Run_nr"].astype(int)
        data = data.sort_values(by=["Run_nr", "source_file"]).reset_index(drop=True)
        data = add_operating_columns(data)

    return data


def lookup_schedule(run_nr: int) -> tuple[float | None, float | None, str | None]:
    for start, end, v_nominal, n_hz, label in RUN_SCHEDULE:
        if start <= run_nr <= end:
            return v_nominal, n_hz, label
    return None, None, None


def add_operating_columns(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    if "Run_nr" not in data.columns:
        raise ValueError("Data must contain a 'Run_nr' column to map operating points.")

    enriched = data.copy()
    mapped = enriched["Run_nr"].astype(int).apply(lookup_schedule)
    enriched["V_nominal"] = mapped.apply(lambda item: item[0])
    enriched["prop_speed_hz"] = mapped.apply(lambda item: item[1])
    enriched["test_block"] = mapped.apply(lambda item: item[2])

    if "V" in enriched.columns:
        measured_v = pd.to_numeric(enriched["V"], errors="coerce")
    else:
        measured_v = pd.Series(np.nan, index=enriched.index, dtype=float)

    speed_for_j = measured_v.fillna(enriched["V_nominal"])

    enriched["advance_ratio_J"] = np.where(
        enriched["prop_speed_hz"].notna(),
        speed_for_j / (enriched["prop_speed_hz"] * PROP_DIAMETER_M),
        np.nan,
    )
    enriched["advance_ratio_nominal_J"] = np.where(
        enriched["prop_speed_hz"].notna(),
        enriched["V_nominal"] / (enriched["prop_speed_hz"] * PROP_DIAMETER_M),
        np.nan,
    )
    return enriched


def normalize_dataframe(data: pd.DataFrame, exclude_columns: set[str] | None = None) -> pd.DataFrame:
    if data.empty:
        return data.copy()

    exclude_columns = exclude_columns or set()
    normalized = data.copy()

    numeric_columns = normalized.select_dtypes(include=["number"]).columns
    columns_to_normalize = [col for col in numeric_columns if col not in exclude_columns]

    for col in columns_to_normalize:
        col_min = normalized[col].min(skipna=True)
        col_max = normalized[col].max(skipna=True)

        if pd.isna(col_min) or pd.isna(col_max):
            continue

        if col_max == col_min:
            normalized[col] = 0.0
            continue

        normalized[col] = 2.0 * (normalized[col] - col_min) / (col_max - col_min) - 1.0

    return normalized


def main():
    bal_dir = Path("BAL")
    data = build_compiled_dataframe(bal_dir)

    if data.empty:
        raise ValueError("No BAL test data rows were found.")

    csv_output = bal_dir / "compiled_bal_points.csv"
    data.to_csv(csv_output, index=False)
    print(f"Saved pandas-compatible CSV: {csv_output}")

    excel_output = bal_dir / "compiled_data.xlsx"
    data.to_excel(excel_output, index=False)
    print(f"Saved Excel file: {excel_output}")

    normalized_data = normalize_dataframe(data, exclude_columns={"Run_nr"})
    normalized_csv_output = bal_dir / "compiled_bal_points_normalized.csv"
    normalized_data.to_csv(normalized_csv_output, index=False)
    print(f"Saved normalized CSV: {normalized_csv_output}")

    print(data.head())
    print(f"Total rows: {len(data)}")


if __name__ == "__main__":
    main()
