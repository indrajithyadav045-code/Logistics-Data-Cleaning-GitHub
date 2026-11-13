
from pathlib import Path
import pandas as pd

# Project directory
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"

# Dataset files
datasets = {
    "Orders": "olist_orders_dataset.csv",
    "Order Items": "olist_order_items_dataset.csv",
    "Customers": "olist_customers_dataset.csv"
}

for name, filename in datasets.items():

    print("\n" + "=" * 60)
    print(f"DATASET: {name}")
    print("=" * 60)

    file_path = RAW_DIR / filename

    df = pd.read_csv(file_path)

    print("\nDataset dimensions:", df.shape)

    print("\nFirst 5 records:")
    print(df.head())

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nStatistical summary:")
    print(df.describe(include="all").T)

print("\nINITIAL DATA INSPECTION COMPLETED!")
