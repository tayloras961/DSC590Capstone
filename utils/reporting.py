import plotly.graph_objects as go
from utils.insights import build_health_snapshot, build_personalized_recommendations


def build_anomaly_summary(scored_df):
    anomaly_count = int(scored_df["anomaly_flag"].sum())
    latest = scored_df.iloc[-1]
    return {
        "records": int(len(scored_df)),
        "anomalies": anomaly_count,
        "anomaly_rate": (anomaly_count / max(len(scored_df), 1)) * 100,
        "avg_heart_rate": float(scored_df["heart_rate"].mean()) if "heart_rate" in scored_df.columns else None,
        "latest_risk_level": latest.get("risk_level", "N/A"),
        "latest_risk_score": float(latest.get("risk_score", 0)),
        "latest_drivers": latest.get("risk_drivers", "baseline variation"),
    }


def build_metric_chart(scored_df, metric, title):
    normal_rows = scored_df.loc[scored_df["anomaly_flag"] == 0]
    anomaly_rows = scored_df.loc[scored_df["anomaly_flag"] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal_rows["timestamp"],
        y=normal_rows[metric],
        mode="lines+markers",
        name="Normal"
    ))

    if not anomaly_rows.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_rows["timestamp"],
            y=anomaly_rows[metric],
            mode="markers",
            name="Anomaly",
            marker=dict(size=10, symbol="circle-open")
        ))

    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        height=340,
        xaxis_title="Timestamp",
        yaxis_title=title,
    )
    return fig


def build_risk_chart(scored_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=scored_df["timestamp"],
        y=scored_df["risk_score"],
        mode="lines+markers",
        name="Risk score"
    ))

    fig.add_trace(go.Scatter(
        x=scored_df["timestamp"],
        y=scored_df["forecast_risk_24h"],
        mode="lines",
        name="24h forecast",
        line=dict(dash="dash")
    ))

    fig.add_hline(y=40, line_dash="dash", annotation_text="Moderate")
    fig.add_hline(y=65, line_dash="dash", annotation_text="High")

    fig.update_layout(
        title="Rolling Risk Score and Short-Term Forecast",
        height=340,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Timestamp",
        yaxis_title="Risk Score",
    )
    return fig


def build_weekly_report(scored_df, profile):
    recent = scored_df.tail(7).copy() if len(scored_df) > 7 else scored_df.copy()
    latest = recent.iloc[-1]
    anomaly_days = int((recent.groupby(recent["timestamp"].dt.date)["anomaly_flag"].sum() > 0).sum())
    snapshot = build_health_snapshot(recent, profile)

    lines = [
        f"Current risk level: {latest.get('risk_level', 'N/A')} ({latest.get('risk_score', 0):.1f}/100).",
        f"Forecast risk over the next day: {latest.get('forecast_risk_24h', 0):.1f}/100.",
        f"Days with at least one anomaly in the recent window: {anomaly_days}.",
        f"Average daily steps: {snapshot['avg_daily_steps']:.0f}.",
        f"Average daily sleep: {snapshot['avg_daily_sleep']:.1f} hours.",
        f"Average glucose: {snapshot['avg_glucose']:.1f}.",
        f"Main current risk drivers: {latest.get('risk_drivers', 'baseline variation')}.",
    ]
    return lines


def build_download_report(scored_df, profile):
    lines = ["AI-Powered Proactive Health Monitoring System", ""]
    lines.append("Weekly Summary")
    lines.extend([f"- {line}" for line in build_weekly_report(scored_df, profile)])
    lines.append("")
    lines.append("Suggested Next Steps")
    lines.extend([f"- {item}" for item in build_personalized_recommendations(scored_df, profile)])
    return "\n".join(lines)


def build_full_report(scored_df, profile):
    lines = []

    summary = build_anomaly_summary(scored_df)
    lines.append("AI-Powered Proactive Health Monitoring System")
    lines.append("")
    lines.append("Model Evaluation Summary")
    lines.append(f"- Total records: {summary['records']}")
    lines.append(f"- Anomalies detected: {summary['anomalies']}")
    lines.append(f"- Anomaly rate: {summary['anomaly_rate']:.1f}%")
    lines.append(f"- Current risk level: {scored_df.iloc[-1].get('risk_level', 'N/A')}")
    lines.append(f"- Current risk score: {scored_df.iloc[-1].get('risk_score', 0):.1f}")
    lines.append(f"- Forecast risk (24h): {scored_df.iloc[-1].get('forecast_risk_24h', 0):.1f}")
    lines.append("")

    lines.append("Weekly Summary")
    lines.extend([f"- {line}" for line in build_weekly_report(scored_df, profile)])
    lines.append("")

    lines.append("Recommendations")
    recommendations = build_personalized_recommendations(scored_df, profile)
    for item in recommendations:
        lines.append(f"- {item}")

    return "\n".join(lines)