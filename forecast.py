"""
forecast.py
-----------
The "predictive modelling" component. For a hackathon MVP this uses a
transparent, explainable linear-trend model (easy to justify to judges)
rather than a black-box deep model — swap in Vertex AI AutoML Forecasting
later for production without changing the interface below.

For each (PHC, medicine) pair:
  1. Fit a linear trend to the last N days of stock levels.
  2. Project forward to estimate days-until-stockout.
  3. Classify into a risk tier.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

LOOKBACK_DAYS = 14
CRITICAL_DAYS = 3   # red:  will run out within 3 days
WARNING_DAYS = 7     # amber: will run out within 7 days


def _forecast_single_series(df_series: pd.DataFrame) -> dict:
    """Fit a linear trend on one PHC+medicine's recent stock history."""
    df_series = df_series.sort_values("date").tail(LOOKBACK_DAYS).reset_index(drop=True)
    if len(df_series) < 3:
        return {"days_to_stockout": None, "trend_per_day": 0.0}

    X = np.arange(len(df_series)).reshape(-1, 1)
    y = df_series["stock_level"].values
    model = LinearRegression().fit(X, y)

    slope = model.coef_[0]
    current_stock = y[-1]

    if slope >= -0.01:  # flat or growing — not heading toward stockout
        return {"days_to_stockout": None, "trend_per_day": round(slope, 2)}

    days_to_stockout = current_stock / abs(slope)
    return {
        "days_to_stockout": round(days_to_stockout, 1),
        "trend_per_day": round(slope, 2),
    }


def build_risk_table(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Run the forecast for every PHC+medicine combination in the dataset."""
    results = []
    latest_date = stock_df["date"].max()

    for (phc_id, medicine), group in stock_df.groupby(["phc_id", "medicine"]):
        forecast = _forecast_single_series(group)
        latest_row = group[group["date"] == latest_date].iloc[0]

        days = forecast["days_to_stockout"]
        if days is None:
            risk = "Stable / Surplus"
        elif days <= CRITICAL_DAYS:
            risk = "Critical"
        elif days <= WARNING_DAYS:
            risk = "Warning"
        else:
            risk = "Watch"

        results.append({
            "phc_id": phc_id,
            "phc_name": latest_row["phc_name"],
            "district": latest_row["district"],
            "state": latest_row["state"],
            "lat": latest_row["lat"],
            "lon": latest_row["lon"],
            "medicine": medicine,
            "unit": latest_row["unit"],
            "current_stock": latest_row["stock_level"],
            "trend_per_day": forecast["trend_per_day"],
            "days_to_stockout": days,
            "risk_level": risk,
        })

    risk_df = pd.DataFrame(results)
    risk_order = {"Critical": 0, "Warning": 1, "Watch": 2, "Stable / Surplus": 3}
    risk_df["_sort"] = risk_df["risk_level"].map(risk_order)
    risk_df = risk_df.sort_values(["_sort", "days_to_stockout"]).drop(columns="_sort")
    return risk_df.reset_index(drop=True)


if __name__ == "__main__":
    stock_df = pd.read_csv("data/phc_stock_data.csv")
    risk_df = build_risk_table(stock_df)
    print(risk_df["risk_level"].value_counts())
    print(risk_df.head(10))
