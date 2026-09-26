"""
app.py
------
Main Streamlit dashboard tying together the four required pieces:
  1. Real-time visibility  -> raw stock/bed/staff table + status colors
  2. Demand forecasting    -> forecast.py (risk table)
  3. Early warnings        -> risk_level flags (Critical / Warning)
  4. Redistribution        -> redistribution.py (transfer suggestions)
  5. Google GenAI layer    -> gemini_assistant.py (natural-language, multilingual report)

Run with:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

from forecast import build_risk_table
from redistribution import build_redistribution_plan
from gemini_assistant import generate_situation_report

st.set_page_config(page_title="PHC National AI Platform", layout="wide")

LANGUAGES = ["English", "Hindi", "Bengali", "Tamil", "Marathi", "Telugu"]


@st.cache_data
def load_data():
    stock_df = pd.read_csv("data/phc_stock_data.csv")
    risk_df = build_risk_table(stock_df)
    plan_df = build_redistribution_plan(risk_df)
    return stock_df, risk_df, plan_df


stock_df, risk_df, plan_df = load_data()

st.title("🏥 National PHC Health Resource & Supply Chain Platform")
st.caption(
    "Real-time visibility, demand forecasting, early warnings, and automated "
    "cross-district redistribution — built for India's Primary Health Centre network."
)

# ---- Sidebar filters ----
st.sidebar.header("Filters")
states = sorted(risk_df["state"].unique())
selected_states = st.sidebar.multiselect("State", states, default=states)
filtered_risk = risk_df[risk_df["state"].isin(selected_states)]
filtered_plan = plan_df[plan_df["shortage_state"].isin(selected_states)]

# ---- Top-line metrics ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("PHCs monitored", stock_df["phc_id"].nunique())
col2.metric("Critical stockout risks", (filtered_risk["risk_level"] == "Critical").sum())
col3.metric("Warning-level risks", (filtered_risk["risk_level"] == "Warning").sum())
col4.metric("Redistribution plans generated", len(filtered_plan))

st.divider()

# ---- Early warning table ----
st.subheader("⚠️ Early Warning — Stockout Risk by PHC")
risk_display = filtered_risk[filtered_risk["risk_level"] != "Stable / Surplus"]


def highlight_risk(row):
    color = {"Critical": "#ffcccc", "Warning": "#fff3cd"}.get(row["risk_level"], "")
    return [f"background-color: {color}"] * len(row)


st.dataframe(
    risk_display[[
        "phc_name", "district", "state", "medicine", "current_stock",
        "days_to_stockout", "risk_level"
    ]].style.apply(highlight_risk, axis=1),
    use_container_width=True,
    height=300,
)

# ---- Redistribution plan ----
st.subheader("🔄 Recommended Cross-District Redistribution")
st.dataframe(
    filtered_plan[[
        "medicine", "shortage_phc", "shortage_district", "risk_level",
        "donor_phc", "donor_district", "distance_km", "recommended_transfer_qty", "unit"
    ]],
    use_container_width=True,
    height=280,
)

st.divider()

# ---- Gemini-powered situation report ----
st.subheader("🤖 AI Situation Report (Google Gemini)")
st.caption(
    "Turns the raw tables above into a plain-language brief a district health "
    "officer can read or have read aloud — in the language of their choice."
)

lang = st.selectbox("Report language", LANGUAGES)
if st.button("Generate AI Situation Report"):
    with st.spinner("Asking Gemini..."):
        report = generate_situation_report(filtered_risk, filtered_plan, language=lang)
    st.info(report)

st.divider()

# ---- Raw live-status view ----
with st.expander("📋 Raw real-time PHC status (latest day, all metrics)"):
    latest_date = stock_df["date"].max()
    latest = stock_df[stock_df["date"] == latest_date]
    st.dataframe(
        latest[[
            "phc_name", "district", "state", "medicine", "stock_level",
            "beds_occupied", "sanctioned_beds", "staff_present", "sanctioned_staff"
        ]],
        use_container_width=True,
        height=300,
    )

st.caption(
    "Prototype data is synthetic, generated to mirror real HMIS patterns for demo "
    "purposes. In production this connects to live state HMIS feeds and data.gov.in."
)
