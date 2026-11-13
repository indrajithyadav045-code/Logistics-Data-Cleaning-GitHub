"""Week 2 logistics exploratory data analysis for the Olist dataset."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RAW = Path(r"D:\Logistics-Data-Cleaning\data\raw")
OUT = Path(r"C:\Users\indra\Documents\Codex\2026-11-13\referenced-chatgpt-conversation-this-is-an\work\week2")
OUT.mkdir(parents=True, exist_ok=True)
PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
palette = {"blue": "#1F4E78", "teal": "#2A9D8F", "orange": "#E76F51"}

orders = pd.read_csv(RAW / "olist_orders_dataset.csv", parse_dates=[
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date"
])
items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
customers = pd.read_csv(RAW / "olist_customers_dataset.csv")

orders["delivery_days"] = (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.total_seconds() / 86400
orders["delay_days"] = (orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]).dt.total_seconds() / 86400
orders["valid_delivery_sequence"] = (
    orders["order_delivered_customer_date"].notna()
    & orders["order_delivered_carrier_date"].notna()
    & (orders["order_delivered_customer_date"] >= orders["order_delivered_carrier_date"])
    & (orders["order_delivered_carrier_date"] >= orders["order_purchase_timestamp"])
)
valid = orders.loc[(orders["order_status"] == "delivered") & orders["valid_delivery_sequence"]].copy()
valid = valid[(valid["delivery_days"] >= 0) & (valid["delivery_days"] <= 90)].copy()
valid["purchase_month"] = valid["order_purchase_timestamp"].dt.to_period("M").astype(str)

order_summary = items.groupby("order_id", as_index=False).agg(
    total_items=("order_item_id", "count"), total_price=("price", "sum"), total_freight=("freight_value", "sum")
)
valid = valid.merge(order_summary, on="order_id", how="left").merge(
    customers[["customer_id", "customer_state"]], on="customer_id", how="left"
)

# 1. Order status distribution
status = orders["order_status"].value_counts().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8.2, 4.7))
ax.barh(status.index, status.values, color=palette["blue"])
ax.set_title("Order Status Distribution")
ax.set_xlabel("Number of orders")
for i, v in enumerate(status.values): ax.text(v + max(status.values)*0.005, i, f"{v:,}", va="center", fontsize=8)
fig.tight_layout(); fig.savefig(PLOTS / "01_order_status_distribution.png", dpi=220); plt.close(fig)

# 2. Monthly purchases
monthly = orders.assign(month=orders["order_purchase_timestamp"].dt.to_period("M").astype(str)).groupby("month").size()
fig, ax = plt.subplots(figsize=(9.2, 4.7))
ax.plot(range(len(monthly)), monthly.values, color=palette["teal"], marker="o", markersize=3, linewidth=2)
ticks = list(range(0, len(monthly), 4)); ax.set_xticks(ticks, [monthly.index[i] for i in ticks], rotation=35, ha="right")
ax.set_title("Monthly Order Volume")
ax.set_ylabel("Orders purchased")
fig.tight_layout(); fig.savefig(PLOTS / "02_monthly_order_volume.png", dpi=220); plt.close(fig)

# 3. Delivery time distribution
fig, ax = plt.subplots(figsize=(8.2, 4.7))
sns.histplot(valid["delivery_days"], bins=35, color=palette["blue"], kde=True, ax=ax)
ax.axvline(valid["delivery_days"].median(), color=palette["orange"], linestyle="--", label=f"Median: {valid['delivery_days'].median():.1f} days")
ax.set_xlim(0, min(45, valid["delivery_days"].quantile(.99)))
ax.set_title("Distribution of Valid Delivered Order Times")
ax.set_xlabel("Purchase to customer delivery (days)"); ax.legend()
fig.tight_layout(); fig.savefig(PLOTS / "03_delivery_time_distribution.png", dpi=220); plt.close(fig)

# 4. Actual vs estimated delivery sample
sample = valid.sample(min(3500, len(valid)), random_state=42)
actual_from_estimate = (sample["order_delivered_customer_date"] - sample["order_estimated_delivery_date"]).dt.total_seconds()/86400
fig, ax = plt.subplots(figsize=(8.2, 4.7))
ax.scatter(sample["delivery_days"], actual_from_estimate, s=9, alpha=.28, color=palette["teal"], edgecolors="none")
ax.axhline(0, color=palette["orange"], linewidth=1.4)
ax.set_title("Actual Delivery Position Relative to Estimate")
ax.set_xlabel("Purchase to customer delivery (days)")
ax.set_ylabel("Actual date minus estimated date (days)")
fig.tight_layout(); fig.savefig(PLOTS / "04_actual_vs_estimated_delivery.png", dpi=220); plt.close(fig)

# 5. Freight versus order value
sample2 = valid.sample(min(5000, len(valid)), random_state=7)
fig, ax = plt.subplots(figsize=(8.2, 4.7))
ax.scatter(sample2["total_price"], sample2["total_freight"], s=9, alpha=.25, color=palette["blue"], edgecolors="none")
ax.set_xlim(0, sample2["total_price"].quantile(.99)); ax.set_ylim(0, sample2["total_freight"].quantile(.99))
ax.set_title("Freight Cost and Order Value")
ax.set_xlabel("Total item price per order (BRL)"); ax.set_ylabel("Total freight per order (BRL)")
fig.tight_layout(); fig.savefig(PLOTS / "05_freight_vs_order_value.png", dpi=220); plt.close(fig)

# 6. State delivery times (top states by valid order count)
top_states = valid["customer_state"].value_counts().head(10).index
state_data = valid[valid["customer_state"].isin(top_states)]
order_states = state_data.groupby("customer_state")["delivery_days"].median().sort_values().index
fig, ax = plt.subplots(figsize=(9.2, 5.0))
sns.boxplot(data=state_data, x="customer_state", y="delivery_days", order=order_states, color="#B9D6EA", showfliers=False, ax=ax)
ax.set_ylim(0, state_data["delivery_days"].quantile(.95))
ax.set_title("Delivery Time Variation Across High Volume Customer States")
ax.set_xlabel("Customer state"); ax.set_ylabel("Delivery days")
fig.tight_layout(); fig.savefig(PLOTS / "06_delivery_by_state.png", dpi=220); plt.close(fig)

metrics = {
    "orders": int(len(orders)), "items": int(len(items)), "customers": int(len(customers)),
    "delivered_valid_orders": int(len(valid)),
    "delivered_share": round(100 * (orders["order_status"] == "delivered").mean(), 2),
    "delivery_mean": round(float(valid["delivery_days"].mean()), 2),
    "delivery_median": round(float(valid["delivery_days"].median()), 2),
    "delivery_p90": round(float(valid["delivery_days"].quantile(.90)), 2),
    "early_or_on_time_share": round(100 * (valid["delay_days"] <= 0).mean(), 2),
    "late_share": round(100 * (valid["delay_days"] > 0).mean(), 2),
    "freight_price_corr": round(float(valid[["total_freight", "total_price"]].corr().iloc[0,1]), 3),
    "top_state": str(valid["customer_state"].value_counts().idxmax()),
    "top_state_median": round(float(valid.groupby("customer_state")["delivery_days"].median().loc[valid["customer_state"].value_counts().idxmax()]), 2),
    "status_top": str(orders["order_status"].value_counts().idxmax()),
    "status_top_count": int(orders["order_status"].value_counts().max()),
    "monthly_peak": str(monthly.idxmax()), "monthly_peak_count": int(monthly.max())
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
pd.DataFrame([metrics]).to_csv(OUT / "summary_metrics.csv", index=False)
print(json.dumps(metrics, indent=2))
