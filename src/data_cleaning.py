
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD ORIGINAL DATA
# ==========================================

orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv")
items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
customers = pd.read_csv(RAW_DIR / "olist_customers_dataset.csv")

print("Original datasets loaded successfully.")

# ==========================================
# 3. REMOVE EXACT DUPLICATES
# ==========================================

before = {
    "orders": len(orders),
    "items": len(items),
    "customers": len(customers)
}

orders = orders.drop_duplicates().copy()
items = items.drop_duplicates().copy()
customers = customers.drop_duplicates().copy()

print("\nExact duplicate removal:")
print("Orders removed:", before["orders"] - len(orders))
print("Items removed:", before["items"] - len(items))
print("Customers removed:", before["customers"] - len(customers))

# ==========================================
# 4. STANDARDIZE TEXT FORMATS
# ==========================================

for df in [orders, items, customers]:
    text_columns = df.select_dtypes(
        include=["object", "str"]
    ).columns

    for col in text_columns:
        df[col] = df[col].str.strip()

orders["order_status"] = (
    orders["order_status"].str.lower()
)

customers["customer_city"] = (
    customers["customer_city"].str.strip().str.lower()
)

customers["customer_state"] = (
    customers["customer_state"].str.upper()
)

# Preserve ZIP codes as text for future use
customers["customer_zip_code_prefix"] = (
    customers["customer_zip_code_prefix"]
    .astype("string")
    .str.zfill(5)
)

# ==========================================
# 5. CONVERT DATE COLUMNS
# ==========================================

order_date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in order_date_columns:
    orders[col] = pd.to_datetime(
        orders[col],
        errors="coerce"
    )

items["shipping_limit_date"] = pd.to_datetime(
    items["shipping_limit_date"],
    errors="coerce"
)

# ==========================================
# 6. VALIDATE NUMERIC COLUMNS
# ==========================================

items["price"] = pd.to_numeric(
    items["price"], errors="coerce"
)

items["freight_value"] = pd.to_numeric(
    items["freight_value"], errors="coerce"
)

# ==========================================
# 7. CREATE DATA QUALITY FLAGS
# ==========================================

orders["missing_approval_date"] = (
    orders["order_approved_at"].isna()
)

orders["missing_carrier_date"] = (
    orders["order_delivered_carrier_date"].isna()
)

orders["missing_customer_delivery_date"] = (
    orders["order_delivered_customer_date"].isna()
)

orders["invalid_carrier_sequence"] = (
    orders["order_delivered_carrier_date"]
    < orders["order_purchase_timestamp"]
).fillna(False)

orders["invalid_delivery_sequence"] = (
    orders["order_delivered_customer_date"]
    < orders["order_delivered_carrier_date"]
).fillna(False)

items["invalid_price"] = (
    items["price"].isna() | (items["price"] <= 0)
)

items["invalid_freight"] = (
    items["freight_value"].isna()
    | (items["freight_value"] < 0)
)

# ==========================================
# 8. CALCULATE DELIVERY METRICS
# ==========================================

orders["delivery_days"] = (
    orders["order_delivered_customer_date"]
    - orders["order_purchase_timestamp"]
).dt.total_seconds() / 86400

orders["delivery_delay_days"] = (
    orders["order_delivered_customer_date"]
    - orders["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

# ==========================================
# 9. CREATE ORDER ITEM SUMMARY
# ==========================================

order_summary = (
    items.groupby("order_id")
    .agg(
        total_items=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum")
    )
    .reset_index()
)

# ==========================================
# 10. CREATE ANALYSIS DATASET
# ==========================================

analysis = orders.merge(
    order_summary,
    on="order_id",
    how="left",
    validate="one_to_one"
)

analysis = analysis.merge(
    customers[
        [
            "customer_id",
            "customer_city",
            "customer_state",
            "customer_zip_code_prefix"
        ]
    ],
    on="customer_id",
    how="left",
    validate="one_to_one"
)

# ==========================================
# 11. EXPORT CLEANED DATASETS
# ==========================================

orders.to_csv(
    OUT_DIR / "cleaned_orders.csv",
    index=False
)

items.to_csv(
    OUT_DIR / "cleaned_order_items.csv",
    index=False
)

customers.to_csv(
    OUT_DIR / "cleaned_customers.csv",
    index=False
)

order_summary.to_csv(
    OUT_DIR / "order_item_summary.csv",
    index=False
)

analysis.to_csv(
    OUT_DIR / "logistics_analysis_ready.csv",
    index=False
)

# ==========================================
# 12. FINAL SUMMARY
# ==========================================

print("\n========== CLEANING SUMMARY ==========")

print("Orders:", orders.shape)
print("Items:", items.shape)
print("Customers:", customers.shape)
print("Analysis dataset:", analysis.shape)

print("\nMissing values in orders:")
print(orders.isna().sum())

print("\nTimestamp anomaly flags:")
print(
    "Invalid carrier sequences:",
    orders["invalid_carrier_sequence"].sum()
)

print(
    "Invalid delivery sequences:",
    orders["invalid_delivery_sequence"].sum()
)

print("\nCLEANING PIPELINE COMPLETED SUCCESSFULLY!")
