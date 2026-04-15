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
    # Basic styling for layout and cards
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
    # Simple login screen for demo purposes
    inject_styles()

    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.title(APP_TITLE)
    st.write(
        "This prototype combines anomaly detection, lightweight predictive risk scoring, "
        "and personalized health recommendations in a single interactive dashboard."
    )
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
    # Handles dataset selection + user profile inputs
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
    )

    st.sidebar.header("User Profile")

    profile = build_profile(
        age=st.sidebar.number_input("Age", 18, 95, 30),
        gender=st.sidebar.selectbox("Gender", ["Female", "Male"]),
        height_in=st.sidebar.number_input("Height (inches)", 48, 84, 66),
        weight_lb=st.sidebar.number_input("Weight (lb)", 80, 450, 165),
        baseline_steps_goal=st.sidebar.number_input("Step goal", 3000, 15000, 8000, step=500),
        wellness_focus=st.sidebar.selectbox(
            "Primary focus",
            ["General wellness", "Improve sleep", "Increase activity", "Reduce glucose variability", "Manage stress / recovery"],
        ),
    )

    return data_frame, contamination, profile, source_name


def render_top_cards(scored_df: pd.DataFrame, profile: dict) -> None:
    # Top summary metrics + quick insight
    summary = build_anomaly_summary(scored_df)
    latest = scored_df.iloc[-1]

    st.markdown('<div class="section-label">Current Monitoring Snapshot</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", summary["records"])
    c2.metric("Flagged anomalies", summary["anomalies"])
    c3.metric("Anomaly rate", f"{summary['anomaly_rate']:.1f}%")
    c4.metric("Risk score", f"{latest['risk_score']:.1f}")
    c5.metric("24h forecast", f"{latest['forecast_risk_24h']:.1f}")

    # Simple AI-style summary
    st.markdown("### AI Insight Summary")
    st.success(generate_ai_summary(scored_df, profile))

    # Personalized snapshot
    snapshot = build_health_snapshot(scored_df, profile)

    st.markdown("### Personalized Snapshot")
    left, right = st.columns(2)

    with left:
        st.write(f"- BMI: **{snapshot['bmi']:.1f}**")
        st.write(f"- Avg steps: **{snapshot['avg_daily_steps']:.0f}**")

    with right:
        st.write(f"- Avg sleep: **{snapshot['avg_daily_sleep']:.1f} hrs**")
        st.write(f"- Avg HR: **{snapshot['avg_heart_rate']:.1f} bpm**")


def render_dashboard(clean_df, scored_df, profile, source_name):
    # Main dashboard view
    st.subheader("Monitoring Dashboard")

    render_top_cards(scored_df, profile)

    st.markdown("### Data Preview")
    st.dataframe(clean_df.head(20))

    st.markdown("### Trends")
    st.plotly_chart(build_risk_chart(scored_df), use_container_width=True)


def main():
    inject_styles()

    if "auth_user" not in st.session_state:
        show_login()
        return

    st.title(APP_TITLE)

    data_frame, contamination, profile, source_name = sidebar_inputs()
    if data_frame is None:
        st.warning("Load data to continue.")
        return

    clean_df, _ = prepare_health_data(data_frame)
    scored_df = score_anomalies(clean_df, build_model(contamination))

    tab1, tab2 = st.tabs(["Dashboard", "Report"])

    with tab1:
        render_dashboard(clean_df, scored_df, profile, source_name)

    with tab2:
        st.write("Reports here...")


if __name__ == "__main__":
    main()