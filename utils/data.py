import io
import numpy as np
import pandas as pd

EXPECTED_COLUMNS = ["timestamp", "heart_rate", "steps", "calories", "sleep_hours", "glucose"]

def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")

    first_line = text.splitlines()[0] if text else ""
    sep = "\t" if "\t" in first_line else ","
    return pd.read_csv(io.StringIO(text), sep=sep)

def prepare_health_data(input_df: pd.DataFrame):
    df = input_df.copy()
    notes = []
    original_rows = len(df)

    df.columns = [str(col).strip().lower() for col in df.columns]
    notes.append("Standardized column names to lowercase and removed extra spaces.")

    if "timestamp" not in df.columns:
        raise ValueError("The dataset must include a timestamp column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    invalid_timestamps = int(df["timestamp"].isna().sum())
    if invalid_timestamps:
        notes.append(f"Dropped {invalid_timestamps} row(s) with invalid timestamps.")
    df = df.dropna(subset=["timestamp"])

    for col in df.columns:
        if col != "timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    duplicate_rows = original_rows - len(df.drop_duplicates())
    if duplicate_rows:
        notes.append(f"Removed {duplicate_rows} duplicate row(s).")
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        missing_before = int(df[col].isna().sum())
        if missing_before:
            df[col] = df[col].interpolate(limit_direction="both")
            df[col] = df[col].fillna(df[col].median())
            notes.append(f"Filled missing values in {col} using interpolation and median fallback.")

    for col in [c for c in ["heart_rate", "steps", "calories", "sleep_hours", "glucose"] if c in df.columns]:
        low = df[col].quantile(0.01)
        high = df[col].quantile(0.99)
        df[col] = df[col].clip(low, high)
    notes.append("Clipped extreme outliers at the 1st and 99th percentiles.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    missing_expected = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_expected:
        notes.append(f"Optional columns not found: {', '.join(missing_expected)}")

    return df, notes

def summarize_daily(scored_df: pd.DataFrame) -> pd.DataFrame:
    daily = scored_df.copy()
    daily["date"] = daily["timestamp"].dt.date

    summary = (
        daily.groupby("date", as_index=False)
        .agg({
            "heart_rate": "mean",
            "steps": "mean",
            "calories": "mean",
            "sleep_hours": "sum",
            "glucose": "mean",
            "anomaly_flag": "sum",
            "risk_score": "mean",
        })
        .rename(columns={"anomaly_flag": "anomalies", "risk_score": "avg_risk_score"})
    )

    return summary.round({
        "heart_rate": 2,
        "steps": 2,
        "calories": 2,
        "sleep_hours": 2,
        "glucose": 2,
        "avg_risk_score": 2,
    })
