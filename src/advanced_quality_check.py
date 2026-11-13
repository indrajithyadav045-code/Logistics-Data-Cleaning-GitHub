
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"

orders = pd.read_csv(RAW / "olist_orders_dataset.csv")
items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
customers = pd.read_csv(RAW / "olist_customers_dataset.csv")

print("=" * 60)
print("ADVANCED DATA QUALITY INVESTIGATION")
print("=" * 60)

# 1. Primary key validation
print("\n1. PRIMARY KEY CHECKS")

print("Orders duplicate IDs:",
      orders["order_id"].duplicated().sum())

print("Customers duplicate IDs:",
      customers["customer_id"].duplicated().sum())

print("Order item duplicate composite keys:",
      items.duplicated(
          subset=["order_id", "order_item_id"]
      ).sum())

# 2. Foreign key validation
print("\n2. RELATIONSHIP VALIDATION")

orphan_orders = (
    ~items["order_id"].isin(orders["order_id"])
).sum()

orphan_customers = (
    ~orders["customer_id"].isin(customers["customer_id"])
).sum()

print("Items without matching orders:", orphan_orders)
print("Orders without matching customers:", orphan_customers)

# 3. Customer identity analysis
print("\n3. CUSTOMER IDENTITY ANALYSIS")

print("Customer records:", len(customers))
print("Unique customer identities:",
      customers["customer_unique_id"].nunique())

# 4. Date sequence validation
print("\n4. DATE SEQUENCE ANALYSIS")

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in date_columns:
    orders[col] = pd.to_datetime(
        orders[col], errors="coerce"
    )

invalid_approval = (
    orders["order_approved_at"]
    < orders["order_purchase_timestamp"]
).fillna(False)

invalid_carrier = (
    orders["order_delivered_carrier_date"]
    < orders["order_purchase_timestamp"]
).fillna(False)

invalid_delivery = (
    orders["order_delivered_customer_date"]
    < orders["order_delivered_carrier_date"]
).fillna(False)

print("Approval before purchase:", invalid_approval.sum())
print("Carrier date before purchase:", invalid_carrier.sum())
print("Customer delivery before carrier:", invalid_delivery.sum())

# 5. Numeric validation
print("\n5. NUMERIC VALIDATION")

print("Zero or negative prices:",
      (items["price"] <= 0).sum())

print("Negative freight values:",
      (items["freight_value"] < 0).sum())

# 6. Export investigation summary
summary = pd.DataFrame([
    ["Duplicate order IDs", orders["order_id"].duplicated().sum()],
    ["Duplicate customer IDs", customers["customer_id"].duplicated().sum()],
    ["Duplicate item keys", items.duplicated(
        subset=["order_id", "order_item_id"]).sum()],
    ["Orphan order items", orphan_orders],
    ["Orphan customers", orphan_customers],
    ["Invalid approval sequence", invalid_approval.sum()],
    ["Invalid carrier sequence", invalid_carrier.sum()],
    ["Invalid delivery sequence", invalid_delivery.sum()],
    ["Non-positive prices", (items["price"] <= 0).sum()],
    ["Negative freight values", (items["freight_value"] < 0).sum()]
], columns=["Quality Check", "Count"])

OUT.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    OUT / "advanced_quality_report.csv",
    index=False
)

print("\nADVANCED INVESTIGATION COMPLETED")
print(summary.to_string(index=False))
