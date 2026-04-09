import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(7)

n_days = 60
dates = pd.date_range("2026-01-01", periods=n_days, freq="D")

df = pd.DataFrame({"timestamp": dates})
df["dayofweek"] = df["timestamp"].dt.dayofweek

# Heart rate
df["heart_rate"] = np.random.normal(72, 5, n_days)

weekend = df["dayofweek"].isin([5, 6])
df.loc[weekend, "heart_rate"] -= np.random.normal(2, 1, weekend.sum())

# Steps 
df["steps"] = np.random.normal(7500, 2000, n_days).clip(2000, 15000)
df.loc[weekend, "steps"] *= np.random.normal(0.9, 0.1, weekend.sum())

# Sleep
df["sleep_hours"] = np.random.normal(7, 1, n_days).clip(4.5, 9)

# Glucose
df["glucose"] = np.random.normal(100, 10, n_days)

# Calories
df["calories"] = (
    (df["steps"] * 0.04) +
    (df["heart_rate"] * 8) +
    np.random.normal(0, 50, n_days)
)

# Inject realistic anomalies
# Elevated glucose
df.loc[10:12, "glucose"] += 30

# Poor sleep + high heart rate
df.loc[25:27, "sleep_hours"] -= 2
df.loc[25:27, "heart_rate"] += 10

# Low activity
df.loc[40:42, "steps"] -= 4000

# High activity 
df.loc[50:51, "steps"] += 5000
df.loc[50:51, "calories"] += 300


df = df.drop(columns=["dayofweek"])

for col in ["heart_rate", "steps", "calories", "sleep_hours", "glucose"]:
    df[col] = df[col].round(2)

# Save
output_path = Path("data") / "sample_health_data.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Saved realistic daily dataset to {output_path}")
print(df.head())
