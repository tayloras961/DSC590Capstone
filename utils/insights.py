import pandas as pd
# user profile sidebar
def build_profile(age: int, gender: str, height_in: int, weight_lb: int, baseline_steps_goal: int, wellness_focus: str) -> dict:
    height_m = height_in * 0.0254
    weight_kg = weight_lb * 0.453592
    bmi = weight_kg / (height_m ** 2)
    return {
        "age": age,
        "gender": gender,
        "height_in": height_in,
        "weight_lb": weight_lb,
        "height_m": height_m,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "baseline_steps_goal": baseline_steps_goal,
        "wellness_focus": wellness_focus,
    }

def build_health_snapshot(scored_df: pd.DataFrame, profile: dict) -> dict:
    daily = scored_df.copy()
    daily["date"] = daily["timestamp"].dt.date
    daily_summary = daily.groupby("date", as_index=False).agg({
        "steps": "sum",
        "sleep_hours": "sum",
        "heart_rate": "mean",
        "glucose": "mean",
    })
    avg_daily_steps = float(daily_summary["steps"].mean()) if not daily_summary.empty else 0.0
    avg_daily_sleep = float(daily_summary["sleep_hours"].mean()) if not daily_summary.empty else 0.0
    avg_heart_rate = float(scored_df["heart_rate"].mean())
    avg_glucose = float(scored_df["glucose"].mean()) if "glucose" in scored_df.columns else 0.0
    step_goal_progress = (avg_daily_steps / max(profile["baseline_steps_goal"], 1)) * 100

    return {
        "bmi": profile["bmi"],
        "avg_daily_steps": avg_daily_steps,
        "avg_daily_sleep": avg_daily_sleep,
        "avg_heart_rate": avg_heart_rate,
        "avg_glucose": avg_glucose,
        "step_goal_progress": step_goal_progress,
    }

def build_personalized_recommendations(scored_df: pd.DataFrame, profile: dict) -> list[str]:
    snapshot = build_health_snapshot(scored_df, profile)
    latest = scored_df.iloc[-1]
    recommendations = []

    if latest["risk_level"] == "High":
        recommendations.append(
            "Recent readings differ substantially from your recent baseline. Review the flagged period closely and continue monitoring over the next 24 hours."
        )
    elif latest["risk_level"] == "Moderate":
        recommendations.append(
            "Some recent readings are moderately different from your baseline. Watch the next several readings to confirm whether the pattern persists."
        )
    else:
        recommendations.append(
            "Recent readings appear relatively stable compared with your recent baseline. Continue routine tracking."
        )

    if snapshot["step_goal_progress"] < 80:
        recommendations.append(
            f"Average daily steps are below your selected goal. A practical next step would be to gradually move toward {profile['baseline_steps_goal']:,} steps per day."
        )

    if snapshot["avg_daily_sleep"] < 7:
        recommendations.append(
            "Average daily sleep is below the common 7-hour target. Improving consistency in sleep timing may help stabilize other health indicators."
        )

    if snapshot["avg_glucose"] > 110:
        recommendations.append(
            "Average glucose appears somewhat elevated in the recent window. Reviewing meal timing, activity, and follow-up monitoring may be helpful."
        )

    if snapshot["bmi"] >= 30:
        recommendations.append(
            "The estimated BMI suggests that weight-management goals may be relevant. Consistent activity and gradual lifestyle changes may provide the most sustainable path forward."
        )
    elif snapshot["bmi"] < 18.5:
        recommendations.append(
            "The estimated BMI is on the lower end. Additional nutritional or wellness review may be helpful depending on the broader context."
        )

    if profile["wellness_focus"] == "Improve sleep":
        recommendations.append(
            "Because sleep is your main focus, continue tracking whether low-sleep periods overlap with higher risk scores or elevated heart rate."
        )
    elif profile["wellness_focus"] == "Increase activity":
        recommendations.append(
            "Because activity is your main focus, compare weekly step totals to your baseline goal and watch whether higher activity reduces risk over time."
        )
    elif profile["wellness_focus"] == "Reduce glucose variability":
        recommendations.append(
            "Because glucose stability is your main focus, pay close attention to periods when glucose-related risk drivers appear in the dashboard."
        )
    elif profile["wellness_focus"] == "Manage stress / recovery":
        recommendations.append(
            "Because recovery is your main focus, monitor whether elevated heart rate and low sleep occur together, as this may signal reduced recovery."
        )

    if latest.get("forecast_risk_24h", 0) > latest.get("risk_score", 0) + 5:
        recommendations.append(
            "The short-term forecast suggests rising risk over the next day. Continued monitoring is recommended rather than relying on a single current reading."
        )

    return recommendations

def generate_ai_summary(scored_df: pd.DataFrame, profile: dict) -> str:
    latest = scored_df.iloc[-1]

    risk = latest.get("risk_level", "Unknown")
    risk_score = latest.get("risk_score", 0)

    # Trend
    recent = scored_df.tail(3)
    trend = "stable"
    if len(recent) >= 2:
        if recent["risk_score"].iloc[-1] > recent["risk_score"].iloc[0]:
            trend = "increasing"
        elif recent["risk_score"].iloc[-1] < recent["risk_score"].iloc[0]:
            trend = "decreasing"

    # Key drivers
    drivers = latest.get("risk_drivers", "")
    driver_text = f" Key contributing factors include {drivers}." if drivers else ""

    # Summary
    return (
        f"Recent data indicates a {risk.lower()} level of risk with a {trend} trend over time. "
        f"The current risk score is {round(risk_score, 2)}.{driver_text} "
        f"This suggests a change from the user's recent baseline that may require continued monitoring."
    )
