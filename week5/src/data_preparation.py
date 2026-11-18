"""
data_preparation.py
===================
Loads and cleans the raw Olist orders dataset.

Steps
-----
1. Load olist_orders_dataset.csv.
2. Parse order_purchase_timestamp as a datetime.
3. Remove rows with missing or unparseable timestamps.
4. Deduplicate on order_id (keep first occurrence).
5. Filter to operationally relevant order statuses.
6. Aggregate to a continuous daily demand time series (zero-filling gaps).
7. Report dataset statistics and save processed files.

Design rationale
----------------
order_purchase_timestamp is used (not delivery dates) because it represents
the moment customer demand is revealed to the warehouse.  Delivery dates lag
by several days and would obscure when stock was actually consumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week5" / "config"))
from simulation_config import (
    RAW_ORDERS,
    DATA_PROCESSED,
    ORDER_STATUS_FILTER,
)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_raw_orders() -> pd.DataFrame:
    """Read the raw Olist orders CSV and return a DataFrame.

    Returns
    -------
    pd.DataFrame with all original columns.

    Raises
    ------
    FileNotFoundError if the raw data file is missing.
    """
    if not RAW_ORDERS.exists():
        raise FileNotFoundError(
            f"Raw orders file not found: {RAW_ORDERS}\n"
            "Please place olist_orders_dataset.csv at that path."
        )
    df = pd.read_csv(RAW_ORDERS, low_memory=False)
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse order_purchase_timestamp to datetime; drop unparseable rows.

    Parameters
    ----------
    df : raw DataFrame from load_raw_orders()

    Returns
    -------
    DataFrame with a parsed 'purchase_date' column (date only).
    """
    df = df.copy()
    df["purchase_dt"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )
    n_before = len(df)
    df = df.dropna(subset=["purchase_dt"])
    n_after = len(df)
    print(f"[data_preparation] Timestamp parsing: {n_before} → {n_after} rows "
          f"({n_before - n_after} dropped for bad/missing timestamp).")
    df["purchase_date"] = df["purchase_dt"].dt.normalize()
    return df


def deduplicate_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate order_ids, keeping first occurrence.

    Parameters
    ----------
    df : DataFrame after parse_timestamps()

    Returns
    -------
    DataFrame with unique order_ids.
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=["order_id"], keep="first")
    n_after = len(df)
    print(f"[data_preparation] Deduplication: {n_before} → {n_after} rows "
          f"({n_before - n_after} duplicate order_ids removed).")
    return df


def filter_valid_statuses(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only operationally relevant order statuses.

    Cancelled and unavailable orders are excluded because they were
    never fulfilled and should not appear as demand in the simulation.

    Parameters
    ----------
    df : DataFrame after deduplicate_orders()
    """
    n_before = len(df)
    df = df[df["order_status"].isin(ORDER_STATUS_FILTER)].copy()
    n_after = len(df)
    print(f"[data_preparation] Status filter: {n_before} → {n_after} rows "
          f"({n_before - n_after} removed for excluded order_status).")
    return df


def build_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate orders into a continuous daily demand time series.

    Zero-fills days with no orders so that the series has no gaps.

    Parameters
    ----------
    df : DataFrame after filter_valid_statuses()

    Returns
    -------
    pd.DataFrame with columns ['date', 'demand'].
    """
    daily = (
        df.groupby("purchase_date")["order_id"]
        .count()
        .rename("demand")
        .reset_index()
        .rename(columns={"purchase_date": "date"})
    )
    # Build a complete calendar from first to last day
    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = (
        daily.set_index("date")
        .reindex(full_range, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    zero_days = (daily["demand"] == 0).sum()
    print(
        f"[data_preparation] Daily demand: {len(daily)} calendar days, "
        f"{zero_days} zero-demand days."
    )
    return daily


def compute_demand_statistics(daily: pd.DataFrame) -> dict:
    """Compute descriptive statistics for daily demand.

    Parameters
    ----------
    daily : output of build_daily_demand()

    Returns
    -------
    dict with keys: mean, median, std, min, max, p05, p25, p75, p95,
    zero_days, total_orders, n_days, date_start, date_end.
    """
    d = daily["demand"]
    stats = {
        "date_start": str(daily["date"].min().date()),
        "date_end": str(daily["date"].max().date()),
        "n_days": int(len(daily)),
        "total_orders": int(d.sum()),
        "mean": float(d.mean()),
        "median": float(d.median()),
        "std": float(d.std()),
        "min": int(d.min()),
        "max": int(d.max()),
        "p05": float(d.quantile(0.05)),
        "p25": float(d.quantile(0.25)),
        "p75": float(d.quantile(0.75)),
        "p95": float(d.quantile(0.95)),
        "zero_days": int((d == 0).sum()),
        "skewness": float(d.skew()),
        "kurtosis": float(d.kurtosis()),
    }
    return stats


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add weekday and month columns for pattern analysis.

    Parameters
    ----------
    daily : output of build_daily_demand()
    """
    daily = daily.copy()
    daily["weekday"] = daily["date"].dt.day_name()
    daily["weekday_num"] = daily["date"].dt.dayofweek   # 0=Monday
    daily["month"] = daily["date"].dt.month
    daily["month_name"] = daily["date"].dt.month_name()
    daily["year"] = daily["date"].dt.year
    return daily


def run_data_preparation() -> tuple[pd.DataFrame, dict]:
    """Execute the complete data preparation pipeline.

    Returns
    -------
    (daily_demand, stats) where daily_demand is a DataFrame and
    stats is a dict of descriptive statistics.
    """
    print("\n=== DATA PREPARATION ===")
    raw = load_raw_orders()
    df = parse_timestamps(raw)
    df = deduplicate_orders(df)
    df = filter_valid_statuses(df)
    daily = build_daily_demand(df)
    daily = add_calendar_features(daily)
    stats = compute_demand_statistics(daily)

    # Save processed data
    out_path = DATA_PROCESSED / "daily_demand.csv"
    daily.to_csv(out_path, index=False)
    print(f"[data_preparation] Saved daily demand to {out_path}")

    # Print summary
    print(f"\nDataset date range : {stats['date_start']} to {stats['date_end']}")
    print(f"Total calendar days: {stats['n_days']}")
    print(f"Total valid orders : {stats['total_orders']:,}")
    print(f"Zero-demand days   : {stats['zero_days']}")
    print(f"Mean daily demand  : {stats['mean']:.2f}")
    print(f"Std  daily demand  : {stats['std']:.2f}")
    print(f"Max  daily demand  : {stats['max']}")

    return daily, stats


if __name__ == "__main__":
    run_data_preparation()
