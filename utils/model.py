import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

FEATURE_COLUMNS = ["heart_rate", "steps", "calories", "sleep_hours", "glucose"]

def available_features(df: pd.DataFrame):
    return [col for col in FEATURE_COLUMNS if col in df.columns]

def build_model(contamination: float = 0.02):
    return IsolationForest(n_estimators=200, contamination=contamination, random_state=7)

def _forecast_next_risk(series: pd.Series) -> float:
    recent = series.tail(24).reset_index(drop=True)
    if len(recent) < 6:
        return float(recent.iloc[-1]) if len(recent) else 0.0
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent, 1)
    forecast = slope * len(recent) + intercept
    return float(np.clip(forecast, 0, 100))

def add_predictive_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    weights = {
        "heart_rate": 0.30,
        "glucose": 0.30,
        "steps": 0.20,
        "sleep_hours": 0.10,
        "calories": 0.10,
    }

    total_signal = np.zeros(len(out), dtype=float)

    for metric, weight in weights.items():
        if metric not in out.columns:
            continue

        baseline_mean = out[metric].shift(1).rolling(window=7, min_periods=3).mean()
        baseline_std = out[metric].shift(1).rolling(window=7, min_periods=3).std().replace(0, np.nan)
        z_score = ((out[metric] - baseline_mean) / baseline_std).abs()
        z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 4)

        out[f"{metric}_baseline"] = baseline_mean
        out[f"{metric}_z"] = z_score
        total_signal += z_score * weight

    anomaly_component = out.get("anomaly_score", pd.Series(np.zeros(len(out)))).fillna(0) * 2.0
    out["risk_score"] = ((total_signal + anomaly_component) / 6.0 * 100).clip(0, 100)

    def classify(score: float) -> str:
        if score >= 65:
            return "High"
        if score >= 40:
            return "Moderate"
        return "Low"

    out["risk_level"] = out["risk_score"].apply(classify)

    drivers = []
    for _, row in out.iterrows():
        ranked = []
        for metric in ["heart_rate", "glucose", "steps", "sleep_hours", "calories"]:
            z_col = f"{metric}_z"
            if z_col in out.columns:
                ranked.append((metric, row.get(z_col, 0)))
        ranked.sort(key=lambda item: item[1], reverse=True)
        top = [name.replace("_", " ") for name, value in ranked[:2] if value > 1]
        drivers.append(", ".join(top) if top else "baseline variation")
    out["risk_drivers"] = drivers

    rolling_forecasts = []
    for idx in range(len(out)):
        rolling_forecasts.append(_forecast_next_risk(out.loc[:idx, "risk_score"]))
    out["forecast_risk_24h"] = rolling_forecasts

    return out

def score_anomalies(clean_df: pd.DataFrame, model):
    df = clean_df.copy()
    features = available_features(df)
    if not features:
        raise ValueError("No valid numeric health features were found.")

    model.fit(df[features])
    raw_pred = model.predict(df[features])
    df["anomaly_flag"] = np.where(raw_pred == -1, 1, 0)

    decision_scores = model.decision_function(df[features])
    df["anomaly_score"] = (-decision_scores - (-decision_scores).min()) / (
        ((-decision_scores).max() - (-decision_scores).min()) + 1e-9
    )

    return add_predictive_signals(df)

def compare_contamination_settings(clean_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = available_features(clean_df)
    has_truth = "true_anomaly" in clean_df.columns

    for contamination in [0.02, 0.03, 0.05]:
        model = build_model(contamination=contamination)
        model.fit(clean_df[features])
        preds = np.where(model.predict(clean_df[features]) == -1, 1, 0)

        row = {
            "contamination": contamination,
            "predicted_anomalies": int(preds.sum()),
        }

        if has_truth:
            truth = clean_df["true_anomaly"].astype(int)
            row.update({
                "accuracy": round(accuracy_score(truth, preds), 3),
                "precision": round(precision_score(truth, preds, zero_division=0), 3),
                "recall": round(recall_score(truth, preds, zero_division=0), 3),
                "f1": round(f1_score(truth, preds, zero_division=0), 3),
            })

        rows.append(row)

    return pd.DataFrame(rows)
