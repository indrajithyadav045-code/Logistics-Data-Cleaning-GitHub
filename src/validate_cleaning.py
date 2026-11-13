
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"

datasets = {
    "Orders": (
        "olist_orders_dataset.csv",
        "cleaned_orders.csv"
    ),
    "Items": (
        "olist_order_items_dataset.csv",
        "cleaned_order_items.csv"
    ),
    "Customers": (
        "olist_customers_dataset.csv",
        "cleaned_customers.csv"
    )
}

results = []

for name, (raw_file, clean_file) in datasets.items():
    original = pd.read_csv(RAW / raw_file)
    cleaned = pd.read_csv(OUT / clean_file)

    source_columns = original.columns.tolist()

    results.append({
        "Dataset": name,
        "Original Rows": len(original),
        "Cleaned Rows": len(cleaned),
        "Original Duplicates": original.duplicated().sum(),
        "Cleaned Duplicates": cleaned[source_columns].duplicated().sum(),
        "Original Missing": original.isna().sum().sum(),
        "Cleaned Missing": cleaned[source_columns].isna().sum().sum(),
        "Row Count Preserved": len(original) == len(cleaned)
    })

report = pd.DataFrame(results)

print("\n========== VALIDATION REPORT ==========")
print(report.to_string(index=False))

report.to_csv(
    OUT / "before_after_validation.csv",
    index=False
)

print("\nValidation report saved successfully!")
