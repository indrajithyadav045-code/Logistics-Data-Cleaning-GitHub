
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"

OUT.mkdir(parents=True, exist_ok=True)

orders = pd.read_csv(
    RAW / "olist_orders_dataset.csv"
)

date_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in date_cols:
    orders[col] = pd.to_datetime(
        orders[col], errors="coerce"
    )

# Identify carrier timestamp anomalies
carrier_anomalies = orders[
    orders["order_delivered_carrier_date"]
    < orders["order_purchase_timestamp"]
].copy()

# Identify delivery timestamp anomalies
delivery_anomalies = orders[
    orders["order_delivered_customer_date"]
    < orders["order_delivered_carrier_date"]
].copy()

print("\nCARRIER DATE ANOMALIES")
print("Total:", len(carrier_anomalies))

print(carrier_anomalies[
    ["order_id", "order_status",
     "order_purchase_timestamp",
     "order_delivered_carrier_date"]
].head(15).to_string(index=False))

print("\nDELIVERY DATE ANOMALIES")
print("Total:", len(delivery_anomalies))

print(delivery_anomalies[
    ["order_id", "order_status",
     "order_delivered_carrier_date",
     "order_delivered_customer_date"]
].head(15).to_string(index=False))

# Export evidence for documentation
carrier_anomalies.to_csv(
    OUT / "carrier_date_anomalies.csv",
    index=False
)

delivery_anomalies.to_csv(
    OUT / "delivery_date_anomalies.csv",
    index=False
)

# Order status distribution
print("\nORDER STATUS DISTRIBUTION")

print(
    orders["order_status"]
    .value_counts(dropna=False)
)

print("\nANOMALY INVESTIGATION COMPLETED")
