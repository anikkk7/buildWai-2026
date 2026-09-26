"""
data_generator.py
------------------
Generates a synthetic but realistic dataset standing in for India's PHC
(Primary Health Centre) network. In a real deployment this would be
replaced by live feeds from state HMIS (Health Management Information
System) systems, many of which publish structured data via data.gov.in.

Run directly to (re)generate data/phc_stock_data.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

STATES_DISTRICTS = {
    "Uttar Pradesh": ["Lucknow", "Varanasi", "Kanpur"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur"],
    "West Bengal": ["Kolkata", "Howrah", "Malda"],
    "Maharashtra": ["Pune", "Nagpur", "Nashik"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur"],
    "Tamil Nadu": ["Chennai", "Madurai", "Coimbatore"],
    "Assam": ["Guwahati", "Dibrugarh", "Silchar"],
}

MEDICINES = [
    ("Paracetamol", 500, "units"),
    ("ORS Sachets", 300, "packets"),
    ("Insulin", 100, "vials"),
    ("Amoxicillin", 400, "strips"),
    ("Iron Folic Acid Tablets", 600, "strips"),
]

DAYS = 30


def generate_phc_list():
    """Create ~3 PHCs per district across the sample states."""
    phcs = []
    phc_id = 1
    for state, districts in STATES_DISTRICTS.items():
        for district in districts:
            for i in range(3):
                phcs.append({
                    "phc_id": f"PHC-{phc_id:03d}",
                    "phc_name": f"{district} PHC-{i+1}",
                    "district": district,
                    "state": state,
                    "lat": round(RNG.uniform(8, 32), 4),   # rough India bounding box
                    "lon": round(RNG.uniform(70, 90), 4),
                    "sanctioned_beds": int(RNG.integers(10, 30)),
                    "sanctioned_staff": int(RNG.integers(5, 15)),
                })
                phc_id += 1
    return pd.DataFrame(phcs)


def generate_time_series(phcs_df):
    """Generate 30 days of stock / bed / staff records per PHC per medicine."""
    rows = []
    today = datetime.today()
    start_date = today - timedelta(days=DAYS - 1)

    for _, phc in phcs_df.iterrows():
        # Each PHC gets a "trend profile" — some are trending toward stockout,
        # some are stable, some are oversupplied. This creates realistic variety
        # for the forecasting model to detect.
        for med_name, base_stock, unit in MEDICINES:
            trend_type = RNG.choice(
                ["declining_fast", "declining_slow", "stable", "surplus"],
                p=[0.15, 0.25, 0.40, 0.20],
            )
            stock = base_stock * RNG.uniform(0.6, 1.4)

            for day in range(DAYS):
                date = start_date + timedelta(days=day)
                daily_usage = base_stock * RNG.uniform(0.01, 0.03)

                if trend_type == "declining_fast":
                    stock -= daily_usage * 3
                elif trend_type == "declining_slow":
                    stock -= daily_usage * 1.3
                elif trend_type == "stable":
                    stock += RNG.normal(0, daily_usage * 0.5)
                else:  # surplus
                    stock += daily_usage * 0.5

                stock = max(stock, 0)

                # Bed occupancy and staff attendance (independent daily signal)
                beds_occupied = int(np.clip(
                    RNG.normal(phc["sanctioned_beds"] * 0.6, phc["sanctioned_beds"] * 0.15),
                    0, phc["sanctioned_beds"]
                ))
                staff_present = int(np.clip(
                    RNG.normal(phc["sanctioned_staff"] * 0.85, 1.2),
                    0, phc["sanctioned_staff"]
                ))

                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "phc_id": phc["phc_id"],
                    "phc_name": phc["phc_name"],
                    "district": phc["district"],
                    "state": phc["state"],
                    "lat": phc["lat"],
                    "lon": phc["lon"],
                    "medicine": med_name,
                    "unit": unit,
                    "stock_level": round(stock, 1),
                    "sanctioned_beds": phc["sanctioned_beds"],
                    "beds_occupied": beds_occupied,
                    "sanctioned_staff": phc["sanctioned_staff"],
                    "staff_present": staff_present,
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    phcs_df = generate_phc_list()
    ts_df = generate_time_series(phcs_df)
    ts_df.to_csv("data/phc_stock_data.csv", index=False)
    print(f"Generated {len(ts_df)} rows across {phcs_df.shape[0]} PHCs "
          f"in {len(STATES_DISTRICTS)} states.")
    print(ts_df.head())
