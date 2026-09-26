"""
app.py
------
Main Streamlit dashboard tying together:

1. Real-time visibility
2. Demand forecasting
3. Early warnings
4. Redistribution
5. Google Gemini AI situation reports

Run with:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from forecast import build_risk_table
from redistribution import build_redistribution_plan
from gemini_assistant import generate_situation_report


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="PHC National AI Platform",
    page_icon="🏥",
    layout="wide"
)


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

LANGUAGES = [
    "English",
    "Hindi",
    "Bengali",
    "Tamil",
    "Marathi",
    "Telugu"
]


# ---------------------------------------------------------
# FILE PATH
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "phc_stock_data.csv"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    # Check whether CSV exists
    if not DATA_FILE.exists():

        st.error("❌ PHC stock data file was not found.")

        st.code(
            f"""
Expected file:

{DATA_FILE}

Your project should look like:

buildwai-2026/
├── app.py
├── forecast.py
├── redistribution.py
├── gemini_assistant.py
├── requirements.txt
└── data/
    └── phc_stock_data.csv
""",
            language="text"
        )

        st.stop()

    # Load CSV
    stock_df = pd.read_csv("data/phc_stock_data.csv")
    # Validate that the CSV is not empty
    if stock_df.empty:
        st.error("❌ phc_stock_data.csv is empty.")
        st.stop()

    # Build risk table
    risk_df = build_risk_table(stock_df)

    # Build redistribution plan
    plan_df = build_redistribution_plan(risk_df)

    return stock_df, risk_df, plan_df


stock_df, risk_df, plan_df = load_data()


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🏥 National PHC Health Resource & Supply Chain Platform")

st.caption(
    "Real-time visibility, demand forecasting, early warnings, "
    "and automated cross-district redistribution — built for "
    "India's Primary Health Centre network."
)


# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------

st.sidebar.header("Filters")

states = sorted(risk_df["state"].dropna().unique())

selected_states = st.sidebar.multiselect(
    "State",
    states,
    default=states
)

filtered_risk = risk_df[
    risk_df["state"].isin(selected_states)
]

filtered_plan = plan_df[
    plan_df["shortage_state"].isin(selected_states)
]


# ---------------------------------------------------------
# TOP-LINE METRICS
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "PHCs monitored",
    stock_df["phc_id"].nunique()
)

col2.metric(
    "Critical stockout risks",
    (filtered_risk["risk_level"] == "Critical").sum()
)

col3.metric(
    "Warning-level risks",
    (filtered_risk["risk_level"] == "Warning").sum()
)

col4.metric(
    "Redistribution plans generated",
    len(filtered_plan)
)


st.divider()


# ---------------------------------------------------------
# EARLY WARNING TABLE
# ---------------------------------------------------------

st.subheader("⚠️ Early Warning — Stockout Risk by PHC")

risk_display = filtered_risk[
    filtered_risk["risk_level"] != "Stable / Surplus"
]


def highlight_risk(row):

    color = {
        "Critical": "#ffcccc",
        "Warning": "#fff3cd"
    }.get(row["risk_level"], "")

    return [
        f"background-color: {color}"
    ] * len(row)


if not risk_display.empty:

    st.dataframe(
        risk_display[
            [
                "phc_name",
                "district",
                "state",
                "medicine",
                "current_stock",
                "days_to_stockout",
                "risk_level"
            ]
        ].style.apply(
            highlight_risk,
            axis=1
        ),
        use_container_width=True,
        height=300
    )

else:

    st.success(
        "✅ No critical or warning-level stockout risks "
        "for the selected states."
    )


# ---------------------------------------------------------
# REDISTRIBUTION PLAN
# ---------------------------------------------------------

st.subheader("🔄 Recommended Cross-District Redistribution")

if not filtered_plan.empty:

    st.dataframe(
        filtered_plan[
            [
                "medicine",
                "shortage_phc",
                "shortage_district",
                "risk_level",
                "donor_phc",
                "donor_district",
                "distance_km",
                "recommended_transfer_qty",
                "unit"
            ]
        ],
        use_container_width=True,
        height=280
    )

else:

    st.info(
        "No redistribution plans are currently required "
        "for the selected states."
    )


st.divider()


# ---------------------------------------------------------
# GEMINI AI SITUATION REPORT
# ---------------------------------------------------------

st.subheader("🤖 AI Situation Report (Google Gemini)")

st.caption(
    "Turns the raw tables above into a plain-language brief "
    "a district health officer can read or have read aloud — "
    "in the language of their choice."
)


lang = st.selectbox(
    "Report language",
    LANGUAGES
)


if st.button("Generate AI Situation Report"):

    with st.spinner("Asking Gemini..."):

        report = generate_situation_report(
            filtered_risk,
            filtered_plan,
            language=lang
        )

    st.info(report)


st.divider()


# ---------------------------------------------------------
# RAW PHC STATUS
# ---------------------------------------------------------

with st.expander(
    "📋 Raw real-time PHC status (latest day, all metrics)"
):

    latest_date = stock_df["date"].max()

    latest = stock_df[
        stock_df["date"] == latest_date
    ]

    st.dataframe(
        latest[
            [
                "phc_name",
                "district",
                "state",
                "medicine",
                "stock_level",
                "beds_occupied",
                "sanctioned_beds",
                "staff_present",
                "sanctioned_staff"
            ]
        ],
        use_container_width=True,
        height=300
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.caption(
    "Prototype data is synthetic, generated to mirror real "
    "HMIS patterns for demo purposes. In production this "
    "connects to live state HMIS feeds and data.gov.in."
)
