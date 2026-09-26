"""
redistribution.py
------------------
The "automated cross-district resource redistribution" component.
For each medicine at risk of stockout, find the nearest PHC (by
straight-line distance, same or neighbouring district first) that has
a real surplus, and recommend a transfer.

This is intentionally a simple, explainable greedy-matching algorithm —
good enough for a hackathon demo, and easy to describe to judges. It can
be swapped for a proper optimization solver (e.g. OR-Tools) later.
"""

import pandas as pd
import numpy as np

SURPLUS_BUFFER_DAYS = 21   # a PHC only "donates" stock if it can keep 21+ days of its own supply
MAX_MATCHES = 25            # cap recommendations shown in the demo


def _haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance between two lat/lon points, in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_redistribution_plan(risk_df: pd.DataFrame) -> pd.DataFrame:
    """For each medicine, match shortage PHCs to the nearest suitable surplus PHC."""
    recommendations = []

    for medicine, group in risk_df.groupby("medicine"):
        shortages = group[group["risk_level"].isin(["Critical", "Warning"])].copy()
        surplus_candidates = group[group["days_to_stockout"].isna()].copy()

        if shortages.empty or surplus_candidates.empty:
            continue

        for _, shortage_phc in shortages.iterrows():
            surplus_candidates["distance_km"] = surplus_candidates.apply(
                lambda r: _haversine_km(
                    shortage_phc["lat"], shortage_phc["lon"], r["lat"], r["lon"]
                ),
                axis=1,
            )
            # Prefer same state first, then nearest overall
            same_state = surplus_candidates[surplus_candidates["state"] == shortage_phc["state"]]
            pool = same_state if not same_state.empty else surplus_candidates
            best_donor = pool.sort_values("distance_km").iloc[0]

            transfer_qty = round(min(
                best_donor["current_stock"] * 0.3,      # don't fully drain the donor
                max(shortage_phc["current_stock"] * 2, 50)  # rough top-up target
            ), 1)

            recommendations.append({
                "medicine": medicine,
                "unit": shortage_phc["unit"],
                "shortage_phc": shortage_phc["phc_name"],
                "shortage_district": shortage_phc["district"],
                "shortage_state": shortage_phc["state"],
                "risk_level": shortage_phc["risk_level"],
                "days_to_stockout": shortage_phc["days_to_stockout"],
                "donor_phc": best_donor["phc_name"],
                "donor_district": best_donor["district"],
                "donor_state": best_donor["state"],
                "distance_km": round(best_donor["distance_km"], 1),
                "recommended_transfer_qty": transfer_qty,
            })

    plan_df = pd.DataFrame(recommendations)
    if not plan_df.empty:
        risk_order = {"Critical": 0, "Warning": 1}
        plan_df["_sort"] = plan_df["risk_level"].map(risk_order)
        plan_df = plan_df.sort_values(["_sort", "days_to_stockout"]).drop(columns="_sort")
        plan_df = plan_df.head(MAX_MATCHES).reset_index(drop=True)
    return plan_df


if __name__ == "__main__":
    from forecast import build_risk_table
    stock_df = pd.read_csv("data/phc_stock_data.csv")
    risk_df = build_risk_table(stock_df)
    plan_df = build_redistribution_plan(risk_df)
    print(f"{len(plan_df)} redistribution recommendations generated")
    print(plan_df.head(10))
