
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"

orders = pd.read_csv(OUT / "cleaned_orders.csv")
items = pd.read_csv(OUT / "cleaned_order_items.csv")
customers = pd.read_csv(OUT / "cleaned_customers.csv")
analysis = pd.read_csv(OUT / "logistics_analysis_ready.csv")

checks = {
    "Unique order IDs": orders["order_id"].is_unique,
    "Unique customer IDs": customers["customer_id"].is_unique,
    "Unique order-item keys": not items.duplicated(
        ["order_id", "order_item_id"]
    ).any(),
    "No orphan order items": items["order_id"].isin(
        orders["order_id"]
    ).all(),
    "No orphan orders": orders["customer_id"].isin(
        customers["customer_id"]
    ).all(),
    "One analysis row per order": (
        len(analysis) == len(orders)
        and analysis["order_id"].is_unique
    )
}

print("\n========== FINAL INTEGRITY CHECK ==========")

for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

print(
    "\nOverall:",
    "PASS" if all(checks.values()) else "REVIEW REQUIRED"
)
