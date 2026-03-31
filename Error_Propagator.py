from pathlib import Path

import pandas as pd


# First pair and second pair used for replication-error calculations.
PAIR_RUNS = [(17,32), (63, 46)]

def load_compiled_data(file_path: Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    if "Run_nr" not in data.columns:
        raise ValueError("Input file must contain a 'Run_nr' column.")

    data["Run_nr"] = pd.to_numeric(data["Run_nr"], errors="coerce")
    data = data.dropna(subset=["Run_nr"]).copy()
    data["Run_nr"] = data["Run_nr"].astype(int)
    return data


def get_numeric_columns(data: pd.DataFrame) -> list[str]:
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    return [col for col in numeric_cols if col != "Run_nr"]


def validate_runs_exist(data: pd.DataFrame, pairs: list[tuple[int, int]]) -> None:
    requested_runs = sorted({run for pair in pairs for run in pair})
    available_runs = set(data["Run_nr"].tolist())
    missing_runs = [run for run in requested_runs if run not in available_runs]
    if missing_runs:
        raise ValueError(f"Missing requested run numbers: {missing_runs}")


def build_pair_stats_table(
    data: pd.DataFrame, pairs: list[tuple[int, int]], value_columns: list[str]
) -> pd.DataFrame:
    rows = []
    indexed = data.set_index("Run_nr")

    for run_a, run_b in pairs:
        for col in value_columns:
            value_a = pd.to_numeric(indexed.at[run_a, col], errors="coerce")
            value_b = pd.to_numeric(indexed.at[run_b, col], errors="coerce")
            if pd.isna(value_a) or pd.isna(value_b):
                continue

            values = pd.Series([float(value_a), float(value_b)], dtype=float)
            mean_value = float(values.mean())
            std_deviation = float(values.std(ddof=1))
            abs_diff = float(abs(value_a - value_b))
            if mean_value == 0:
                validation_deviation_pct = 0.0 if abs_diff == 0 else float("nan")
            else:
                validation_deviation_pct = abs_diff / abs(mean_value) * 100.0

            rows.append(
                {
                    "pair": f"{run_a}-{run_b}",
                    "run_a": run_a,
                    "run_b": run_b,
                    "variable": col,
                    "value_run_a": float(value_a),
                    "value_run_b": float(value_b),
                    "mean_value": mean_value,
                    "std_deviation": std_deviation,
                    "validation_deviation_pct": validation_deviation_pct,
                }
            )

    return pd.DataFrame(rows)


def resolve_base_dir() -> Path:
    cwd_candidate = Path.cwd()
    if (cwd_candidate / "BAL" / "compiled_bal_points.csv").exists():
        return cwd_candidate
    return Path(__file__).resolve().parent


def main() -> None:
    base_dir = resolve_base_dir()
    input_file = base_dir / "BAL" / "compiled_bal_points.csv"

    data = load_compiled_data(input_file)
    validate_runs_exist(data, PAIR_RUNS)
    value_columns = get_numeric_columns(data)

    pair_stats = build_pair_stats_table(data, PAIR_RUNS, value_columns)
    if pair_stats.empty:
        raise ValueError("No valid numeric values found to compute pair statistics.")

    output_file = base_dir / "BAL" / "replication_pair_stats.csv"
    pair_stats.to_csv(output_file, index=False)

    print(f"Saved pair stats: {output_file}")
    print("\nColumns: value_run_a, value_run_b, mean_value, std_deviation, validation_deviation_pct")
    print(pair_stats.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
