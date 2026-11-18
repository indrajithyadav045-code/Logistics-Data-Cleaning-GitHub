"""
exploratory_analysis.py
=======================
Diagnostic analytics, spatial flow analysis, root-cause delay attribution,
and customer impact analysis for Week 6.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week6" / "config"))
from project_config import DATA_PROCESSED, DATA_RESULTS


def run_exploratory_analysis(df: pd.DataFrame | None = None) -> dict:
    """Perform diagnostic exploratory logistics analysis and extract summary metrics.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Consolidated logistics DataFrame. If None, loaded from disk.

    Returns
    -------
    dict: Comprehensive dictionary of measured EDA metrics and findings.
    """
    print("\n--- [Pillar 2: Diagnostic & Spatial Exploratory Analysis] ---")
    if df is None:
        input_path = DATA_PROCESSED / "consolidated_logistics_orders.csv"
        df = pd.read_csv(input_path, low_memory=False)

    n_orders = len(df)
    total_freight_brl = float(df["total_freight"].sum())
    total_goods_value_brl = float(df["total_price"].sum())
    mean_freight_brl = float(df["total_freight"].mean())
    mean_price_brl = float(df["total_price"].mean())

    # Delivery durations
    mean_delivery_days = float(df["delivery_days"].mean())
    median_delivery_days = float(df["delivery_days"].median())
    std_delivery_days = float(df["delivery_days"].std())
    p95_delivery_days = float(df["delivery_days"].quantile(0.95))

    mean_carrier_days = float(df["carrier_dispatch_days"].mean())
    mean_transit_days = float(df["in_transit_days"].mean())
    mean_estimated_days = float(df["estimated_delivery_days"].mean())

    # Delays (SLA breach)
    delayed_orders = int(df["is_delayed"].sum())
    delay_rate_pct = float(delayed_orders / n_orders * 100.0)
    mean_delay_severity_days = float(
        df.loc[df["is_delayed"] == 1, "delay_days"].mean()
    ) if delayed_orders > 0 else 0.0

    print(f"Total Evaluated Orders      : {n_orders:,}")
    print(f"Mean Delivery Duration      : {mean_delivery_days:.2f} days (Median: {median_delivery_days:.1f}d)")
    print(f"Carrier Dispatch Latency    : {mean_carrier_days:.2f} days")
    print(f"In-Transit Carrier Latency  : {mean_transit_days:.2f} days")
    print(f"On-Time Delivery Rate       : {100.0 - delay_rate_pct:.2f}%")
    print(f"Late Deliveries (SLA Breach): {delayed_orders:,} ({delay_rate_pct:.2f}%)")
    print(f"Mean Late Delay Severity    : {mean_delay_severity_days:.2f} days beyond promise")

    # 1. Interstate vs Intrastate Performance
    interstate_summary = df.groupby("is_interstate").agg(
        order_count=("order_id", "count"),
        mean_delivery_days=("delivery_days", "mean"),
        median_delivery_days=("delivery_days", "median"),
        delay_rate_pct=("is_delayed", lambda x: float(np.mean(x) * 100.0)),
        mean_freight=("total_freight", "mean"),
        mean_distance_km=("distance_km", "mean"),
    ).reset_index()
    interstate_summary["route_type"] = interstate_summary["is_interstate"].map(
        {0: "Intrastate (Same State)", 1: "Interstate (Cross State)"}
    )
    interstate_path = DATA_RESULTS / "interstate_vs_intrastate.csv"
    interstate_summary.to_csv(interstate_path, index=False)
    print("\nInterstate vs Intrastate Flow Dynamics:")
    print(interstate_summary[["route_type", "order_count", "mean_delivery_days", "delay_rate_pct", "mean_freight"]].to_string(index=False))

    # 2. Regional Origin-Destination (O-D) Matrix
    od_matrix = pd.crosstab(
        df["seller_region"],
        df["customer_region"],
        values=df["delivery_days"],
        aggfunc="mean",
    ).round(2)
    od_path = DATA_RESULTS / "regional_od_matrix_delivery_days.csv"
    od_matrix.to_csv(od_path)

    # 3. Customer State Bottlenecks (Top 15 States by volume)
    state_perf = df.groupby("customer_state").agg(
        orders=("order_id", "count"),
        mean_days=("delivery_days", "mean"),
        delay_pct=("is_delayed", lambda x: float(np.mean(x) * 100.0)),
        mean_freight=("total_freight", "mean"),
        avg_distance=("distance_km", "mean"),
    ).sort_values("orders", ascending=False).reset_index()
    state_path = DATA_RESULTS / "state_logistics_performance.csv"
    state_perf.to_csv(state_path, index=False)

    # 4. Impact of Delays on Customer Review Scores
    ontime_reviews = df[df["is_delayed"] == 0]["review_score"]
    delayed_reviews = df[df["is_delayed"] == 1]["review_score"]
    mean_ontime_rating = float(ontime_reviews.mean())
    mean_delayed_rating = float(delayed_reviews.mean())
    satisfaction_drop_pct = float((mean_ontime_rating - mean_delayed_rating) / mean_ontime_rating * 100.0)

    print(f"\nCustomer Experience Impact:")
    print(f"On-Time Mean Review Score   : {mean_ontime_rating:.2f} / 5.0")
    print(f"Delayed Mean Review Score   : {mean_delayed_rating:.2f} / 5.0")
    print(f"Customer Rating Penalty     : -{mean_ontime_rating - mean_delayed_rating:.2f} stars ({satisfaction_drop_pct:.1f}% drop)")

    # Distance tier analysis
    bins = [0, 200, 500, 1000, 2000, 5000]
    labels = ["< 200 km", "200-500 km", "500-1000 km", "1000-2000 km", "> 2000 km"]
    df["distance_tier"] = pd.cut(df["distance_km"], bins=bins, labels=labels, include_lowest=True)
    dist_tier = df.groupby("distance_tier", observed=False).agg(
        orders=("order_id", "count"),
        mean_days=("delivery_days", "mean"),
        delay_rate=("is_delayed", lambda x: float(np.mean(x) * 100.0)),
        mean_freight=("total_freight", "mean"),
    ).reset_index()
    dist_tier_path = DATA_RESULTS / "distance_tier_performance.csv"
    dist_tier.to_csv(dist_tier_path, index=False)

    eda_metrics = {
        "dataset_summary": {
            "total_orders": n_orders,
            "total_freight_brl": total_freight_brl,
            "total_goods_value_brl": total_goods_value_brl,
            "mean_freight_brl": mean_freight_brl,
            "mean_price_brl": mean_price_brl,
            "mean_delivery_days": mean_delivery_days,
            "median_delivery_days": median_delivery_days,
            "std_delivery_days": std_delivery_days,
            "p95_delivery_days": p95_delivery_days,
            "mean_carrier_days": mean_carrier_days,
            "mean_transit_days": mean_transit_days,
            "mean_estimated_days": mean_estimated_days,
            "delayed_orders": delayed_orders,
            "delay_rate_pct": delay_rate_pct,
            "mean_delay_severity_days": mean_delay_severity_days,
        },
        "customer_satisfaction": {
            "mean_ontime_rating": mean_ontime_rating,
            "mean_delayed_rating": mean_delayed_rating,
            "satisfaction_drop_pct": satisfaction_drop_pct,
        },
        "interstate_breakdown": interstate_summary.to_dict(orient="records"),
        "top_states_performance": state_perf.head(10).to_dict(orient="records"),
    }

    out_json = DATA_RESULTS / "eda_metrics.json"
    with open(out_json, "w") as f:
        json.dump(eda_metrics, f, indent=2)
    print(f"EDA metrics saved: {out_json}")

    return eda_metrics


if __name__ == "__main__":
    run_exploratory_analysis()
