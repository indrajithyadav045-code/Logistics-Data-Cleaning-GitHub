"""
data_pipeline.py
================
End-to-end data ingestion, relational consolidation, data cleaning,
anomaly handling, and feature engineering for the Week 6 Logistics Project.
"""
from __future__ import annotations

import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week6" / "config"))
from project_config import (
    ORDERS_CSV,
    ITEMS_CSV,
    CUSTOMERS_CSV,
    SELLERS_CSV,
    PRODUCTS_CSV,
    REVIEWS_CSV,
    DATA_PROCESSED,
    VALID_ORDER_STATUSES,
    STATE_COORDINATES,
    REGION_MAP,
)


def calculate_haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate the great-circle distance between two points in kilometers."""
    R = 6371.0  # Earth's radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def get_state_distance(seller_state: str, customer_state: str) -> float:
    """Compute Haversine distance between state centroids.
    Intrastate deliveries (same state) receive a baseline floor distance of 45.0 km.
    """
    if seller_state == customer_state:
        return 45.0
    coord1 = STATE_COORDINATES.get(seller_state)
    coord2 = STATE_COORDINATES.get(customer_state)
    if coord1 is None or coord2 is None:
        return 500.0  # default fallback if state unmapped
    return calculate_haversine_distance(coord1[0], coord1[1], coord2[0], coord2[1])


def load_and_clean_data() -> pd.DataFrame:
    """Ingest, clean, merge, and engineer features across all Olist relational tables.

    Returns
    -------
    pd.DataFrame: Cleaned, consolidated order-level logistics dataset.
    """
    print("--- [Pillar 1: Data Ingestion & Relational Cleaning] ---")

    # 1. Orders
    print(f"Loading orders from: {ORDERS_CSV}")
    orders = pd.read_csv(ORDERS_CSV, low_memory=False)
    n_raw_orders = len(orders)
    orders = orders[orders["order_status"].isin(VALID_ORDER_STATUSES)].copy()
    print(f"Delivered orders retained: {len(orders)} / {n_raw_orders}")

    # Parse timestamps
    timestamp_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in timestamp_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # Drop orders missing customer delivery date or carrier date
    orders = orders.dropna(
        subset=[
            "order_purchase_timestamp",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ).copy()
    print(f"Orders with complete delivery timestamps: {len(orders)}")

    # 2. Items: Aggregate to order level
    print(f"Loading order items from: {ITEMS_CSV}")
    items = pd.read_csv(ITEMS_CSV, low_memory=False)
    
    # Calculate item and volume metrics
    items_agg = items.groupby("order_id").agg(
        total_items=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        primary_seller_id=("seller_id", "first"),
        primary_product_id=("product_id", "first"),
    ).reset_index()

    # 3. Products
    print(f"Loading products from: {PRODUCTS_CSV}")
    products = pd.read_csv(PRODUCTS_CSV, low_memory=False)
    # Compute volume in cm3
    products["product_volume_cm3"] = (
        products["product_length_cm"]
        * products["product_height_cm"]
        * products["product_width_cm"]
    )
    # Impute missing weights/volumes with medians
    median_weight = products["product_weight_g"].median()
    median_volume = products["product_volume_cm3"].median()
    products["product_weight_g"] = products["product_weight_g"].fillna(median_weight)
    products["product_volume_cm3"] = products["product_volume_cm3"].fillna(median_volume)
    products["product_category_name"] = products["product_category_name"].fillna("other")

    prod_cols = ["product_id", "product_category_name", "product_weight_g", "product_volume_cm3"]
    items_agg = items_agg.merge(
        products[prod_cols], left_on="primary_product_id", right_on="product_id", how="left"
    )
    items_agg["product_weight_g"] = items_agg["product_weight_g"].fillna(median_weight)
    items_agg["product_volume_cm3"] = items_agg["product_volume_cm3"].fillna(median_volume)
    items_agg["product_category_name"] = items_agg["product_category_name"].fillna("other")

    # 4. Customers
    print(f"Loading customers from: {CUSTOMERS_CSV}")
    customers = pd.read_csv(CUSTOMERS_CSV, low_memory=False)
    cust_cols = ["customer_id", "customer_state", "customer_city", "customer_zip_code_prefix"]

    # 5. Sellers
    print(f"Loading sellers from: {SELLERS_CSV}")
    sellers = pd.read_csv(SELLERS_CSV, low_memory=False)
    seller_cols = ["seller_id", "seller_state", "seller_city", "seller_zip_code_prefix"]

    # 6. Reviews
    print(f"Loading reviews from: {REVIEWS_CSV}")
    reviews = pd.read_csv(REVIEWS_CSV, low_memory=False)
    review_agg = reviews.groupby("order_id")["review_score"].mean().reset_index()

    # Consolidate merges
    df = orders.merge(items_agg, on="order_id", how="inner")
    df = df.merge(customers[cust_cols], on="customer_id", how="left")
    df = df.merge(sellers[seller_cols], left_on="primary_seller_id", right_on="seller_id", how="left")
    df = df.merge(review_agg, on="order_id", how="left")
    df["review_score"] = df["review_score"].fillna(df["review_score"].median())

    # -----------------------------------------------------------------------
    # Feature Engineering & Duration Calculations
    # -----------------------------------------------------------------------
    # Durations in days
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    df["carrier_dispatch_days"] = (
        df["order_delivered_carrier_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    df["in_transit_days"] = (
        df["order_delivered_customer_date"] - df["order_delivered_carrier_date"]
    ).dt.total_seconds() / 86400.0

    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    # Delay relative to promise (positive means late delivery)
    df["delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0

    df["is_delayed"] = (df["delay_days"] > 0.0).astype(int)

    # Clean anomalous durations (e.g. delivery before purchase or extreme outliers > 120 days)
    initial_len = len(df)
    valid_mask = (
        (df["delivery_days"] > 0)
        & (df["delivery_days"] <= 120)
        & (df["carrier_dispatch_days"] >= 0)
        & (df["in_transit_days"] >= 0)
        & (df["seller_state"].notna())
        & (df["customer_state"].notna())
    )
    df = df[valid_mask].copy()
    print(f"Data anomaly cleaning: removed {initial_len - len(df)} rows with negative/extreme transit times.")

    # Spatial features
    df["is_interstate"] = (df["seller_state"] != df["customer_state"]).astype(int)
    df["seller_region"] = df["seller_state"].map(REGION_MAP).fillna("Other")
    df["customer_region"] = df["customer_state"].map(REGION_MAP).fillna("Other")

    # Haversine distance
    df["distance_km"] = [
        get_state_distance(s, c)
        for s, c in zip(df["seller_state"], df["customer_state"])
    ]

    # Physical features
    df["weight_kg"] = df["product_weight_g"] / 1000.0
    df["volume_m3"] = df["product_volume_cm3"] / 1_000_000.0
    df["density_kg_m3"] = df["weight_kg"] / (df["volume_m3"] + 1e-5)
    df["freight_ratio"] = df["total_freight"] / (df["total_price"] + df["total_freight"] + 1e-4)

    # Temporal features
    df["purchase_dow"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["purchase_is_weekend"] = df["purchase_dow"].isin([5, 6]).astype(int)

    # Save processed dataset
    output_path = DATA_PROCESSED / "consolidated_logistics_orders.csv"
    df.to_csv(output_path, index=False)
    print(f"Consolidated dataset saved: {output_path} (Shape: {df.shape})")

    return df


if __name__ == "__main__":
    load_and_clean_data()
