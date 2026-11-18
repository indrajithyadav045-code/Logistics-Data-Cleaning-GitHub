# Week 3 Logistics Demand Forecasting

This project forecasts daily Olist order demand seven days ahead for an educational logistics analytics exercise. Demand is the number of unique `order_id` values per calendar day, aggregated from `order_purchase_timestamp` in the untouched source file at `data/raw/olist_orders_dataset.csv`.

## Run

From the project root, create or activate a Python environment and install dependencies:

```powershell
venv\Scripts\python.exe -m pip install -r week3\requirements.txt
venv\Scripts\python.exe week3\src\run_forecast.py
```

The script writes reproducible CSVs, plots, and measured metadata beneath `week3/outputs/` and `week3/reports/`. It never changes the raw data or Week 1 artifacts.

## Method

1. Load raw orders and count unique purchases per day. Missing calendar days are retained with zero demand.
2. Inspect the daily series and its 28-day moving average.
3. Make leakage-safe features: calendar fields; lags of 1, 7, 14, and 28 days; and 7/28-day means and standard deviations calculated only from `demand.shift(1)`.
4. Split chronologically: the earliest 80% of usable feature rows are training data and the latest 20% are test data. Models see no test rows during fitting.
5. Evaluate a weekly seasonal-naive (`lag_7`) baseline, `LinearRegression`, and `RandomForestRegressor` using MAE, RMSE, and MAPE. MAPE excludes zero-actual days to avoid division by zero.
6. Produce a seven-day recursive forecast with the random forest: each predicted value becomes available as history for the next day. This is appropriate when actual future demand is unavailable at forecast time.

## Outputs

- `outputs/eda_daily_demand.png` - daily-demand exploratory plot
- `outputs/actual_vs_predicted.png` - chronological test-set comparison
- `outputs/feature_importance.png` and `.csv` - random-forest feature importance
- `outputs/model_metrics.csv` - measured baseline and model performance
- `outputs/recursive_7_day_forecast.csv` - seven-day recursive forecast
- `reports/measured_results.json` - run metadata and measured outcomes
- `reports/week3_report.md` - report-ready interpretation generated after a successful run

## Caveat

Olist is a fixed historical marketplace dataset, not a live feed. It can be sparse near the beginning or end of the observed period and does not include forward-looking operational drivers. This workflow demonstrates sound time-series validation and forecasting mechanics; it is not a live operational forecast.
