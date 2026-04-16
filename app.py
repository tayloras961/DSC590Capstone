import pandas as pd
import streamlit as st
from pathlib import Path
from werkzeug.security import check_password_hash

from utils.auth import DEFAULT_USERS
from utils.data import load_csv_bytes, prepare_health_data, summarize_daily
from utils.model import build_model, score_anomalies, compare_contamination_settings
from utils.insights import (
    build_profile,
    build_personalized_recommendations,
    build_health_snapshot,
    generate_ai_summary,
)
from utils.reporting import (
    build_anomaly_summary,
    build_metric_chart,
    build_risk_chart,
    build_weekly_report,
    build_download_report,
    build_full_report,
)

st.set_page_config(page_title="Health Monitoring Dashboard", page_icon="🩺", layout="wide")

APP_TITLE = "AI-Powered Proactive Health Monitoring System"
SAMPLE_PATH = Path("data") / "sample_health_data.csv"


def inject_styles() -> None:
    # styling
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        .hero-card {
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
            border: 1px solid #d8e7fb;
            margin-bottom: 1rem;
        }
        .section-label {
            font-size: 0.9rem;
            font-weight: 600;
            color: #4b5563;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .status-good {color: #18794e; font-weight: 600;}
        .status-watch {color: #b35500; font-weight: 600;}
        .status-high {color: #b42318; font-weight: 600;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_login() -> None:
    # login
    inject_styles()

    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.title(APP_TITLE)
   
    st.markdown("</div>", unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        user = DEFAULT_USERS.get(username.strip().lower())
        if user and check_password_hash(user["password_hash"], password):
            st.session_state["auth_user"] = username.strip().lower()
            st.session_state["auth_role"] = user["role"]
            st.rerun()
        st.error("Invalid username or password.")

    st.info("Demo accounts: admin / admin123 or user / user123")


def sidebar_inputs():
    # Dataset selection
    st.sidebar.header("Data Source")
    source = st.sidebar.radio("Choose data source", ["Sample data", "Upload CSV"], index=0)

    data_frame = None
    source_name = None

    if source == "Sample data":
        data_frame = pd.read_csv(SAMPLE_PATH)
        source_name = "sample_health_data.csv"
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV or TSV", type=["csv", "txt"])
        if uploaded_file is not None:
            data_frame = load_csv_bytes(uploaded_file.getvalue())
            source_name = uploaded_file.name

    contamination = st.sidebar.slider(
        "Anomaly sensitivity",
        min_value=0.01,
        max_value=0.10,
        value=0.02,
        step=0.01,
        help="Higher values label more records as anomalies.",
    )

    st.sidebar.header("User Profile")
    st.sidebar.caption("These inputs are used for reporting and recommendations only.")

    age = st.sidebar.number_input("Age", min_value=18, max_value=95, value=30)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"], index=0)
    height_in = st.sidebar.number_input("Height (inches)", min_value=48, max_value=84, value=66)
    weight_lb = st.sidebar.number_input("Weight (lb)", min_value=80, max_value=450, value=165)
    baseline_steps_goal = st.sidebar.number_input(
        "Preferred daily step goal",
        min_value=3000,
        max_value=15000,
        value=8000,
        step=500,
    )
    wellness_focus = st.sidebar.selectbox(
        "Primary focus",
        [
            "General wellness",
            "Improve sleep",
            "Increase activity",
            "Reduce glucose variability",
            "Manage stress / recovery",
        ],
        index=0,
    )

    profile = build_profile(
        age=age,
        gender=gender,
        height_in=height_in,
        weight_lb=weight_lb,
        baseline_steps_goal=baseline_steps_goal,
        wellness_focus=wellness_focus,
    )

    return data_frame, contamination, profile, source_name


def render_top_cards(scored_df: pd.DataFrame, profile: dict) -> None:
    # Top metrics and quick summary
    summary = build_anomaly_summary(scored_df)
    latest = scored_df.iloc[-1]

    status_class = (
        "status-good" if latest["risk_level"] == "Low"
        else "status-watch" if latest["risk_level"] == "Moderate"
        else "status-high"
    )

    st.markdown('<div class="section-label">Current Monitoring Snapshot</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", summary["records"])
    c2.metric("Flagged anomalies", summary["anomalies"])
    c3.metric("Anomaly rate", f"{summary['anomaly_rate']:.1f}%")
    c4.metric("Current risk score", f"{latest['risk_score']:.1f}")
    c5.metric("Forecast risk (24h)", f"{latest['forecast_risk_24h']:.1f}")

    st.markdown(
        f"Current risk level: <span class='{status_class}'>{latest['risk_level']}</span> "
        f"• Primary drivers: {latest['risk_drivers']}",
        unsafe_allow_html=True,
    )

    st.markdown("### AI Insight Summary")
    ai_summary = generate_ai_summary(scored_df, profile)
    st.success(ai_summary)

    snapshot = build_health_snapshot(scored_df, profile)

    st.markdown("### Personalized Snapshot")
    left, right = st.columns(2)

    with left:
        st.write(f"- BMI estimate: **{snapshot['bmi']:.1f}**")
        st.write(f"- Average daily steps: **{snapshot['avg_daily_steps']:.0f}**")
        st.write(f"- Average glucose: **{snapshot['avg_glucose']:.1f}**")

    with right:
        st.write(f"- Average daily sleep: **{snapshot['avg_daily_sleep']:.1f} hours**")
        st.write(f"- Average resting-style heart rate: **{snapshot['avg_heart_rate']:.1f} bpm**")
        st.write(f"- Progress toward step goal: **{snapshot['step_goal_progress']:.0f}%**")


def render_dashboard(
    clean_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    profile: dict,
    source_name: str | None,
) -> None:
    # Main dashboard view
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.subheader("Monitoring Dashboard")
    if source_name:
        st.caption(f"Current dataset: {source_name}")
    st.markdown("</div>", unsafe_allow_html=True)

    render_top_cards(scored_df, profile)

    st.markdown("### Data Preview")
    st.dataframe(clean_df.head(20), use_container_width=True)

    st.markdown("### Trend and Anomaly Views")
    chart_cols = st.columns(2)

    metrics = [
        ("heart_rate", "Heart Rate"),
        ("steps", "Steps"),
        ("glucose", "Glucose"),
        ("sleep_hours", "Sleep Hours"),
        ("calories", "Calories"),
    ]

    for idx, (metric, title) in enumerate(metrics):
        if metric in scored_df.columns:
            with chart_cols[idx % 2]:
                st.plotly_chart(
                    build_metric_chart(scored_df, metric, title),
                    use_container_width=True,
                )

    st.plotly_chart(build_risk_chart(scored_df), use_container_width=True)

    st.markdown("### Flagged Records")
    flagged = scored_df.loc[scored_df["anomaly_flag"] == 1].copy()

    if flagged.empty:
        st.success("No anomalies were flagged with the current settings.")
    else:
        keep_cols = [
            c for c in [
                "timestamp",
                "heart_rate",
                "steps",
                "calories",
                "sleep_hours",
                "glucose",
                "anomaly_score",
                "risk_score",
                "risk_level",
                "risk_drivers",
            ]
            if c in flagged.columns
        ]

        st.dataframe(
            flagged[keep_cols].sort_values("risk_score", ascending=False),
            use_container_width=True,
        )

        st.download_button(
            "Download flagged anomalies CSV",
            data=flagged.to_csv(index=False).encode("utf-8"),
            file_name="flagged_anomalies.csv",
            mime="text/csv",
        )


def render_reports(scored_df: pd.DataFrame, clean_df: pd.DataFrame, profile: dict) -> None:
    # Report tab
    st.subheader("Weekly Summary Report")
    st.info(
        "Profile-based inputs are used only to personalize the report and suggestions. "
        "The anomaly model uses uploaded health data only."
    )

    report_lines = build_weekly_report(scored_df, profile)
    for line in report_lines:
        st.write(f"- {line}")

    st.subheader("Suggested Next Steps")
    recs = build_personalized_recommendations(scored_df, profile)
    for idx, rec in enumerate(recs, start=1):
        st.write(f"{idx}. {rec}")

    report_text = build_download_report(scored_df, profile)
    st.download_button(
        "Download text summary",
        data=report_text,
        file_name="weekly_health_report.txt",
        mime="text/plain",
    )

    full_report = build_full_report(scored_df, profile)
    st.download_button(
        "Download full report",
        data=full_report,
        file_name="full_health_report.txt",
        mime="text/plain",
    )

    st.markdown("### Daily Summary Table")
    daily = summarize_daily(scored_df)
    st.dataframe(daily, use_container_width=True)

    st.download_button(
        "Download daily summary CSV",
        data=daily.to_csv(index=False).encode("utf-8"),
        file_name="daily_health_summary.csv",
        mime="text/csv",
    )


def render_model_evaluation(scored_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    # Evaluation tab
    st.subheader("Model Evaluation Report")

    total = len(scored_df)
    anomalies = int(scored_df["anomaly_flag"].sum())
    anomaly_rate = anomalies / total if total > 0 else 0

    st.markdown("### Detection Summary")
    st.write(f"- Total records analyzed: {total}")
    st.write(f"- Anomalies detected: {anomalies}")
    st.write(f"- Detection rate: {anomaly_rate:.2%}")

    st.markdown("### Contamination Sensitivity Analysis")
    comp = compare_contamination_settings(clean_df)
    st.dataframe(comp, use_container_width=True)

    st.markdown("### Model Behavior Interpretation")
    st.write(
        "The anomaly detection model identifies unusual observations based on deviations from normal patterns. "
        "Lower contamination values produce fewer, more conservative anomaly flags, while higher values increase sensitivity."
    )

    st.markdown("### Trend Analysis Insights")
    daily = summarize_daily(scored_df)
    if "avg_risk_score" in daily.columns:
        st.line_chart(daily.set_index("date")["avg_risk_score"])

    st.write(
        "Daily aggregation shows how risk scores evolve over time, helping identify sustained abnormal periods rather than isolated spikes."
    )

    st.markdown("### Key Findings")
    st.write(
        "- Most anomalies occur during short clustered periods rather than isolated points.\n"
        "- Risk scores align with visible shifts in heart rate, glucose, and activity levels.\n"
        "- Model sensitivity can be tuned using the contamination parameter."
    )


def render_preprocessing(notes: list[str]) -> None:
    # Preprocessing tab
    st.subheader("Preprocessing Notes")
    for note in notes:
        st.write(f"- {note}")


def render_refinement(clean_df: pd.DataFrame) -> None:
    # Refinement tab
    st.subheader("Refinement Check")
    st.write(
        "This table compares contamination settings to show how a more conservative or more sensitive "
        "threshold changes anomaly counts and model behavior."
    )
    st.dataframe(compare_contamination_settings(clean_df), use_container_width=True)


def main() -> None:
    inject_styles()

    if "auth_user" not in st.session_state:
        show_login()
        return

    st.sidebar.success(f"Signed in as {st.session_state['auth_user']}")
    if st.sidebar.button("Log out"):
        st.session_state.clear()
        st.rerun()

    st.title(APP_TITLE)
    st.caption(
        "A capstone project that focuses on anomaly detection, risk forecasting, and personalized reporting."
    )

    data_frame, contamination, profile, source_name = sidebar_inputs()
    if data_frame is None:
        st.warning("Load the sample dataset or upload a file to continue.")
        return

    if source_name:
        st.success(f"Loaded dataset: {source_name}")

    clean_df, prep_notes = prepare_health_data(data_frame)
    scored_df = score_anomalies(clean_df, build_model(contamination=contamination))

    dashboard_tab, report_tab, eval_tab, prep_tab, refine_tab = st.tabs([
        "Dashboard",
        "Weekly Report",
        "Model Evaluation",
        "Preprocessing Notes",
        "Refinement",
    ])

    with dashboard_tab:
        render_dashboard(clean_df, scored_df, profile, source_name)

    with report_tab:
        render_reports(scored_df, clean_df, profile)

    with eval_tab:
        render_model_evaluation(scored_df, clean_df)

    with prep_tab:
        render_preprocessing(prep_notes)

    with refine_tab:
        render_refinement(clean_df)


if __name__ == "__main__":
    main()
