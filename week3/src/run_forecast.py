"""Reproducible leakage-safe 7-day demand-forecasting workflow for Olist.

Run from the repository root:
    python week3/src/run_forecast.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib

# Generate files without invoking the Windows interactive Tk backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ORDERS = PROJECT_ROOT / "data" / "raw" / "olist_orders_dataset.csv"
WEEK3 = PROJECT_ROOT / "week3"
OUTPUTS = WEEK3 / "outputs"
REPORTS = WEEK3 / "reports"
FORECAST_HORIZON = 7
TEST_FRACTION = 0.20
RANDOM_STATE = 42
LAGS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 28)


def safe_mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """MAPE on nonzero actuals only; return NaN if no eligible observations exist."""
    actual = np.asarray(list(actual), dtype=float)
    predicted = np.asarray(list(predicted), dtype=float)
    nonzero = actual != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)


def metrics(actual: pd.Series, predicted: np.ndarray) -> Dict[str, float]:
    """Return complementary error measures without unsafe zero division.

    MAPE is retained only on non-zero actuals for comparability, while WAPE
    divides total absolute error by total observed demand and therefore remains
    defined when individual calendar days have zero orders.
    """
    absolute_error = np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))
    total_actual = float(np.abs(np.asarray(actual, dtype=float)).sum())
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE_nonzero_pct": safe_mape(actual, predicted),
        "WAPE_pct": float(absolute_error.sum() / total_actual * 100) if total_actual else float("nan"),
        "MAPE_eligible_days": int((np.asarray(actual, dtype=float) != 0).sum()),
        "Zero_actual_days": int((np.asarray(actual, dtype=float) == 0).sum()),
    }


def load_daily_demand(path: Path) -> pd.DataFrame:
    """Aggregate all order purchases to a complete daily order-count series."""
    orders = pd.read_csv(path, usecols=["order_id", "order_purchase_timestamp"])
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )
    orders = orders.dropna(subset=["order_id", "order_purchase_timestamp"])
    orders["date"] = orders["order_purchase_timestamp"].dt.normalize()
    daily = orders.groupby("date", as_index=True)["order_id"].nunique().rename("demand")
    daily = daily.asfreq("D", fill_value=0).to_frame()
    return daily


def make_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Build calendar and historical-only features.

    Every lag/rolling statistic uses demand shifted by one day. Therefore, a row
    at date t contains no information from demand at t or any later date.
    """
    frame = daily.copy()
    index = frame.index
    frame["day_of_week"] = index.day_of_week
    frame["day_of_month"] = index.day
    frame["month"] = index.month
    frame["week_of_year"] = index.isocalendar().week.astype(int)
    frame["time_index"] = np.arange(len(frame))
    for lag in LAGS:
        frame[f"lag_{lag}"] = frame["demand"].shift(lag)
    prior = frame["demand"].shift(1)
    for window in ROLLING_WINDOWS:
        frame[f"rolling_mean_{window}"] = prior.rolling(window=window, min_periods=window).mean()
        frame[f"rolling_std_{window}"] = prior.rolling(window=window, min_periods=window).std()
    return frame.dropna().copy()


def split_chronologically(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_at = int(len(frame) * (1 - TEST_FRACTION))
    if split_at < 1 or split_at >= len(frame):
        raise ValueError("Not enough observations for a chronological train/test split.")
    return frame.iloc[:split_at].copy(), frame.iloc[split_at:].copy()


def plot_eda(daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily.index, daily["demand"], color="#4C78A8", linewidth=0.8, label="Daily orders")
    ax.plot(daily.index, daily["demand"].rolling(28, min_periods=1).mean(), color="#F58518", linewidth=1.8, label="28-day mean")
    ax.set(title="Olist daily demand from purchase timestamps", xlabel="Purchase date", ylabel="Unique orders")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS / "eda_daily_demand.png", dpi=160)
    plt.close(fig)


def plot_predictions(test: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test.index, test["demand"], color="black", linewidth=2, label="Actual")
    for name, values in predictions.items():
        ax.plot(test.index, values, linewidth=1.4, label=name)
    ax.set(title="Chronological test set: actual versus predicted daily demand", xlabel="Date", ylabel="Orders")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)


def plot_importance(model: RandomForestRegressor, feature_columns: list[str]) -> pd.DataFrame:
    importance = pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_}).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance["feature"], importance["importance"], color="#54A24B")
    ax.set(title="Random forest feature importance", xlabel="Impurity-based importance")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "feature_importance.png", dpi=160)
    plt.close(fig)
    return importance.sort_values("importance", ascending=False)


def recursive_forecast(model: RandomForestRegressor, daily: pd.DataFrame, feature_columns: list[str], horizon: int) -> pd.DataFrame:
    """Generate a multi-step forecast, feeding each prediction back as past demand."""
    history = daily.astype({"demand": float}).copy()
    results = []
    for _ in range(horizon):
        next_date = history.index.max() + pd.Timedelta(days=1)
        # The placeholder target is never used as a feature for its own row:
        # all historical features are shifted. A numeric value lets the shared
        # feature builder retain this date before its prediction replaces it.
        placeholder = pd.DataFrame({"demand": [0.0]}, index=pd.DatetimeIndex([next_date]))
        candidate = pd.concat([history, placeholder])
        next_features = make_features(candidate).loc[[next_date], feature_columns]
        prediction = max(0.0, float(model.predict(next_features)[0]))
        history.loc[next_date, "demand"] = prediction
        results.append({"date": next_date.date().isoformat(), "predicted_demand": prediction})
    return pd.DataFrame(results)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not RAW_ORDERS.exists():
        raise FileNotFoundError(f"Expected Olist raw data at {RAW_ORDERS}")

    daily = load_daily_demand(RAW_ORDERS)
    feature_frame = make_features(daily)
    train, test = split_chronologically(feature_frame)
    feature_columns = [column for column in feature_frame.columns if column != "demand"]
    x_train, y_train = train[feature_columns], train["demand"]
    x_test, y_test = test[feature_columns], test["demand"]

    baseline = test["lag_7"].to_numpy()  # seasonal-naive weekly baseline
    linear = LinearRegression().fit(x_train, y_train)
    forest = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1).fit(x_train, y_train)
    predictions = {
        "Seasonal naive (lag 7)": baseline,
        "Linear regression": linear.predict(x_test),
        "Random forest": forest.predict(x_test),
    }
    metric_rows = []
    for name, prediction in predictions.items():
        metric_rows.append({"model": name, **metrics(y_test, prediction)})
    metric_frame = pd.DataFrame(metric_rows).sort_values("MAE")

    plot_eda(daily)
    plot_predictions(test, predictions)
    importance = plot_importance(forest, feature_columns)
    forecast = recursive_forecast(forest, daily, feature_columns, FORECAST_HORIZON)

    daily.rename_axis("date").to_csv(OUTPUTS / "daily_demand.csv")
    metric_frame.to_csv(OUTPUTS / "model_metrics.csv", index=False)
    importance.to_csv(OUTPUTS / "feature_importance.csv", index=False)
    forecast.to_csv(OUTPUTS / "recursive_7_day_forecast.csv", index=False)
    test_export = test[["demand"]].copy()
    for name, prediction in predictions.items():
        test_export[name] = prediction
    test_export.rename_axis("date").to_csv(OUTPUTS / "test_predictions.csv")

    summary = {
        "source": str(RAW_ORDERS.relative_to(PROJECT_ROOT)),
        "demand_definition": "unique order_id count per calendar day of order_purchase_timestamp",
        "date_range": {"start": daily.index.min().date().isoformat(), "end": daily.index.max().date().isoformat()},
        "daily_rows": int(len(daily)),
        "feature_rows_after_history_requirement": int(len(feature_frame)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_end": train.index.max().date().isoformat(),
        "test_start": test.index.min().date().isoformat(),
        "test_end": test.index.max().date().isoformat(),
        "forecast_horizon_days": FORECAST_HORIZON,
        "daily_demand_summary": {
            "total_orders": int(daily["demand"].sum()),
            "mean": float(daily["demand"].mean()),
            "median": float(daily["demand"].median()),
            "maximum": int(daily["demand"].max()),
            "maximum_date": daily["demand"].idxmax().date().isoformat(),
            "zero_demand_days": int((daily["demand"] == 0).sum()),
        },
        "models": metric_frame.round(4).to_dict(orient="records"),
        "forecast_model": "Random forest selected for recursive forecast; its test MAE is compared with the baseline and linear regression in model_metrics.csv.",
        "leakage_controls": ["chronological split", "lags use prior demand", "rolling features use demand.shift(1)", "models fit only on training rows"],
        "caveat": "Olist is historical, finite, and may be sparse at its boundaries. This is an educational demonstration, not a live operational forecast.",
    }
    with (REPORTS / "measured_results.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
