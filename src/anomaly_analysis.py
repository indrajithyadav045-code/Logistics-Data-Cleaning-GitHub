
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

# Calculate time differences in minutes
orders["carrier_minus_purchase_min"] = (
    orders["order_delivered_carrier_date"]
    - orders["order_purchase_timestamp"]
).dt.total_seconds() / 60

orders["delivery_minus_carrier_min"] = (
    orders["order_delivered_customer_date"]
    - orders["order_delivered_carrier_date"]
).dt.total_seconds() / 60

# Identify anomalies
carrier_bad = orders[
    orders["carrier_minus_purchase_min"] < 0
].copy()

delivery_bad = orders[
    orders["delivery_minus_carrier_min"] < 0
].copy()

# Add same-day indicators
carrier_bad["same_calendar_day"] = (
    carrier_bad["order_purchase_timestamp"].dt.date
    == carrier_bad["order_delivered_carrier_date"].dt.date
)

# Categorize anomaly severity
carrier_bad["anomaly_category"] = (
    carrier_bad["same_calendar_day"]
    .map({
        True: "Same-day timestamp reversal",
        False: "Cross-day timestamp reversal"
    })
)

# Export detailed evidence
carrier_bad.to_csv(
    OUT / "carrier_anomaly_analysis.csv",
    index=False
)

delivery_bad.to_csv(
    OUT / "delivery_anomaly_analysis.csv",
    index=False
)

# Print summary
print("\nCARRIER ANOMALY ANALYSIS")
print(carrier_bad["anomaly_category"].value_counts())

print("\nCarrier time difference (minutes)")
print(carrier_bad["carrier_minus_purchase_min"].describe())

print("\nDELIVERY ANOMALY ANALYSIS")
print(
    delivery_bad["delivery_minus_carrier_min"]
    .describe()
)

print("\nAnomalies by order status")
print(
    carrier_bad["order_status"].value_counts()
)

print("\nDelivery anomalies by status")
print(
    delivery_bad["order_status"].value_counts()
)

print("\nANALYSIS EXPORT COMPLETED")
