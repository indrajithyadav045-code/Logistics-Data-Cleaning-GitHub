# Week 3 Logistics Demand Forecasting Report

## Purpose and result

This report forecasts seven future daily Olist order arrivals using unique order IDs grouped by `order_purchase_timestamp`. Random forest had the lowest chronological holdout MAE (42.96 orders) and RMSE (51.63); it was therefore used for the recursive forecast. Its WAPE was 31.4%. This is a historical modelling exercise, not a live operational commitment.

## Data and worked example

The untouched raw orders file yielded 774 calendar days from 2016-09-04 through 2018-10-17: 99,441 orders, mean 128.5/day, median 124.0/day, and 140 zero-demand dates. The peak is 1176 orders on 2017-11-24.

For the first held-out date, 2018-05-21, actual demand was 206 orders. `lag_7` is the count seven days earlier; the 7-day rolling mean is (364 + 352 + 357 + 228 + 236 + 139 + 161) / 7 = 262.43. Both use only preceding days because rolling features are computed after `demand.shift(1)`, so the target-day count is not leaked into its predictors.

## Validation and evaluation

The first 596 usable rows through 2018-05-20 train the models. The following 150 rows, 2018-05-21 through 2018-10-17, are held out chronologically. The weekly seasonal-naive baseline repeats lag 7. MAE gives typical daily error, RMSE emphasizes big misses, MAPE is calculated only on non-zero actuals, and WAPE is a zero-safe aggregate percentage.

| Model | MAE | RMSE | WAPE | MAPE nonzero |
| --- | ---: | ---: | ---: | ---: |
| Random forest | 42.96 | 51.63 | 31.4% | 588.6% |
| Seasonal naive (lag 7) | 44.54 | 66.87 | 32.6% | 182.0% |
| Linear regression | 65.70 | 77.84 | 48.0% | 1506.9% |

MAPE is volatile here because small non-zero actual counts magnify percentages; MAE and RMSE drive selection. The forest improves MAE by 1.58 orders/day versus the weekly baseline.

## Chart rationale and interpretation

The historical plot uses a line chart because the order of daily observations matters: it shows continuity, spikes, and the 28-day smoothing trend. A bar chart would obscure movement over 774 days. The actual-versus-predicted line chart uses the time-ordered holdout to show timing and amplitude mistakes that a metric table hides. The horizontal importance chart was selected because long feature names remain legible while rank comparison is easy. `lag_1` is highest (0.449), but importance is a fitted-model diagnostic, not causal evidence; correlated lags and rolling means can share signal.

## Seven day recursive forecast

Each prediction is appended as history before the next day is predicted. This matches production conditions where future actuals do not exist, but error can compound.

| Date | Predicted orders |
| --- | ---: |
| 2018-10-18 | 40.4 |
| 2018-10-19 | 58.2 |
| 2018-10-20 | 63.6 |
| 2018-10-21 | 70.2 |
| 2018-10-22 | 86.3 |
| 2018-10-23 | 87.7 |
| 2018-10-24 | 98.9 |

## Limitations and actions

The Olist history is fixed and sparse at the endpoint. It lacks future promotions, holidays, price, inventory, stock-out, and capacity signals. One holdout split does not establish stability over many origins. For operational use, refresh data, add known future drivers, run rolling-origin backtests, monitor against seasonal naive, and publish prediction intervals.
