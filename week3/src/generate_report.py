"""Build a data-derived Week 3 Markdown and DOCX report after run_forecast.py."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
WEEK3, OUT, REPORTS = ROOT / "week3", ROOT / "week3" / "outputs", ROOT / "week3" / "reports"
RAW = ROOT / "data" / "raw" / "olist_orders_dataset.csv"

def paragraph(doc, text):
    doc.add_paragraph(text)

def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = "Light Shading Accent 1"
    for c, v in zip(t.rows[0].cells, headers): c.text = str(v)
    for row in rows:
        cells = t.add_row().cells
        for c, v in zip(cells, row): c.text = str(v)

def load_context():
    summary = json.loads((REPORTS / "measured_results.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(OUT / "daily_demand.csv", parse_dates=["date"]).set_index("date")
    metrics = pd.read_csv(OUT / "model_metrics.csv").sort_values("MAE")
    forecast = pd.read_csv(OUT / "recursive_7_day_forecast.csv")
    importance = pd.read_csv(OUT / "feature_importance.csv")
    test = pd.read_csv(OUT / "test_predictions.csv", parse_dates=["date"]).set_index("date")
    date = test.index.min()
    prior = daily.loc[date - pd.Timedelta(days=7):date - pd.Timedelta(days=1), "demand"]
    raw = pd.read_csv(RAW, usecols=["order_id", "order_purchase_timestamp"])
    raw["order_purchase_timestamp"] = pd.to_datetime(raw["order_purchase_timestamp"], errors="coerce")
    examples = raw.loc[raw["order_purchase_timestamp"].dt.normalize() == date].head(3).copy()
    examples["purchase_timestamp"] = examples["order_purchase_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return dict(summary=summary, daily=daily, metrics=metrics, forecast=forecast, importance=importance,
                date=date, actual=int(daily.loc[date, "demand"]), prior=prior,
                examples=examples[["order_id", "purchase_timestamp"]])

def narrative(c):
    s, m, f, imp = c["summary"], c["metrics"], c["forecast"], c["importance"]
    d, best = s["daily_demand_summary"], m.iloc[0]
    metric_rows = "\n".join(f"| {r.model} | {r.MAE:.2f} | {r.RMSE:.2f} | {r.WAPE_pct:.1f}% | {r.MAPE_nonzero_pct:.1f}% |" for r in m.itertuples(index=False))
    forecast_rows = "\n".join(f"| {r.date} | {r.predicted_demand:.1f} |" for r in f.itertuples(index=False))
    prior_txt = " + ".join(str(int(v)) for v in c["prior"])
    return f"""# Week 3 Logistics Demand Forecasting Report

## Purpose and result

This report forecasts seven future daily Olist order arrivals using unique order IDs grouped by `order_purchase_timestamp`. Random forest had the lowest chronological holdout MAE ({best.MAE:.2f} orders) and RMSE ({best.RMSE:.2f}); it was therefore used for the recursive forecast. Its WAPE was {best.WAPE_pct:.1f}%. This is a historical modelling exercise, not a live operational commitment.

## Data and worked example

The untouched raw orders file yielded {s['daily_rows']} calendar days from {s['date_range']['start']} through {s['date_range']['end']}: {d['total_orders']:,} orders, mean {d['mean']:.1f}/day, median {d['median']:.1f}/day, and {d['zero_demand_days']} zero-demand dates. The peak is {d['maximum']} orders on {d['maximum_date']}.

For the first held-out date, {c['date'].date()}, actual demand was {c['actual']} orders. `lag_7` is the count seven days earlier; the 7-day rolling mean is ({prior_txt}) / 7 = {c['prior'].mean():.2f}. Both use only preceding days because rolling features are computed after `demand.shift(1)`, so the target-day count is not leaked into its predictors.

## Validation and evaluation

The first {s['train_rows']} usable rows through {s['train_end']} train the models. The following {s['test_rows']} rows, {s['test_start']} through {s['test_end']}, are held out chronologically. The weekly seasonal-naive baseline repeats lag 7. MAE gives typical daily error, RMSE emphasizes big misses, MAPE is calculated only on non-zero actuals, and WAPE is a zero-safe aggregate percentage.

| Model | MAE | RMSE | WAPE | MAPE nonzero |
| --- | ---: | ---: | ---: | ---: |
{metric_rows}

MAPE is volatile here because small non-zero actual counts magnify percentages; MAE and RMSE drive selection. The forest improves MAE by {m.loc[m.model.str.contains('Seasonal'), 'MAE'].iloc[0]-best.MAE:.2f} orders/day versus the weekly baseline.

## Chart rationale and interpretation

The historical plot uses a line chart because the order of daily observations matters: it shows continuity, spikes, and the 28-day smoothing trend. A bar chart would obscure movement over {s['daily_rows']} days. The actual-versus-predicted line chart uses the time-ordered holdout to show timing and amplitude mistakes that a metric table hides. The horizontal importance chart was selected because long feature names remain legible while rank comparison is easy. `{imp.iloc[0].feature}` is highest ({imp.iloc[0].importance:.3f}), but importance is a fitted-model diagnostic, not causal evidence; correlated lags and rolling means can share signal.

## Seven day recursive forecast

Each prediction is appended as history before the next day is predicted. This matches production conditions where future actuals do not exist, but error can compound.

| Date | Predicted orders |
| --- | ---: |
{forecast_rows}

## Limitations and actions

The Olist history is fixed and sparse at the endpoint. It lacks future promotions, holidays, price, inventory, stock-out, and capacity signals. One holdout split does not establish stability over many origins. For operational use, refresh data, add known future drivers, run rolling-origin backtests, monitor against seasonal naive, and publish prediction intervals.
"""

def build_docx(c):
    s, m, f, imp = c["summary"], c["metrics"], c["forecast"], c["importance"]
    d, best = s["daily_demand_summary"], m.iloc[0]
    doc = Document(); sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Inches(.65); sec.left_margin = sec.right_margin = Inches(.75)
    for name in ["Normal", "Heading 1", "Heading 2"]:
        style = doc.styles[name]; style.font.name = "Aptos"; style.font.size = Pt(10.5 if name == "Normal" else 14); style.font.color.rgb = RGBColor(0,0,0)
    title = doc.add_paragraph(style="Title"); title.alignment = WD_ALIGN_PARAGRAPH.CENTER; title.add_run("Week 3 Logistics Demand Forecasting Report")
    sub = doc.add_paragraph("Olist daily purchase demand  Seven day recursive forecast"); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_heading("Purpose and conclusion", 1)
    paragraph(doc, f"This project forecasts daily marketplace order arrivals from `order_purchase_timestamp`. Random forest had the best chronological holdout MAE ({best.MAE:.2f} orders) and RMSE ({best.RMSE:.2f}), so it produced the seven-day forecast. WAPE was {best.WAPE_pct:.1f}%. These results demonstrate a reproducible method, not live operational accuracy.")
    doc.add_heading("Demand definition and data preparation", 1)
    paragraph(doc, f"Demand is distinct `order_id` count on each purchase date, rather than delivery count: purchase time measures incoming demand whereas delivery time measures fulfilment. Raw data remains untouched. The series contains {s['daily_rows']} dates from {s['date_range']['start']} through {s['date_range']['end']}. Missing calendar dates are zero observed orders, not invented timestamps; retaining them preserves a valid daily lag timeline.")
    table(doc, ["Measure", "Observed result"], [["Unique orders", f"{d['total_orders']:,}"], ["Mean daily demand", f"{d['mean']:.1f}"], ["Median daily demand", f"{d['median']:.1f}"], ["Peak day", f"{d['maximum_date']}  {d['maximum']} orders"], ["Zero-demand calendar dates", d['zero_demand_days']]])
    doc.add_heading("Worked example using actual records", 1)
    paragraph(doc, f"The first holdout date is {c['date'].date()} and its actual demand is {c['actual']} orders. The three raw records below demonstrate the source field and aggregation date.")
    table(doc, ["Order ID", "Purchase timestamp"], [[r.order_id, r.purchase_timestamp] for r in c['examples'].itertuples(index=False)])
    vals = " + ".join(str(int(v)) for v in c['prior'])
    paragraph(doc, f"For this prediction, `lag_7` is {int(c['prior'].iloc[0])}, the order count on {(c['date']-pd.Timedelta(days=7)).date()}. The 7-day rolling mean is ({vals}) / 7 = {c['prior'].mean():.2f}; the rolling standard deviation is {c['prior'].std():.2f}. Because features use `demand.shift(1)`, none includes the target-day count.")
    doc.add_heading("Exploration and chart choice", 1)
    paragraph(doc, f"A line chart is appropriate for chronological observations: it exposes continuity, peaks, and the 28-day smoothed reference. A bar chart would compress {s['daily_rows']} dates into unreadable categories. The peak of {d['maximum']} orders on {d['maximum_date']} should be investigated as a possible event, not called seasonality without recurrence.")
    doc.add_picture(str(OUT / "eda_daily_demand.png"), width=Inches(6.7))
    doc.add_heading("Leakage safe validation and model comparison", 1)
    paragraph(doc, f"Rows without 28 days of history are excluded. The earliest {s['train_rows']} usable rows through {s['train_end']} are training data; {s['test_rows']} later rows from {s['test_start']} through {s['test_end']} are a chronological holdout. Calendar features are known ahead of time; lags 1, 7, 14, 28 and rolling 7/28 statistics use only prior demand. The seasonal-naive baseline repeats seven-day-old demand, which a complex model must beat.")
    table(doc, ["Model", "MAE", "RMSE", "WAPE", "MAPE nonzero"], [[r.model, f"{r.MAE:.2f}", f"{r.RMSE:.2f}", f"{r.WAPE_pct:.1f}%", f"{r.MAPE_nonzero_pct:.1f}%"] for r in m.itertuples(index=False)])
    baseline = m.loc[m["model"].str.contains("Seasonal"), "MAE"].iloc[0]
    paragraph(doc, f"MAE communicates typical daily error and drives model selection; RMSE adds penalty for large misses. MAPE uses only {int(best.MAPE_eligible_days)} non-zero actual days, so it is volatile when demand is very small. WAPE remains defined with zero-demand days. Random forest improved MAE by {baseline-best.MAE:.2f} orders/day over seasonal naive.")
    paragraph(doc, "The actual-versus-predicted line chart is deliberately chronological. It complements the table by showing whether models miss turns or over-smooth the sparse late period, rather than hiding all behaviour behind one number.")
    doc.add_picture(str(OUT / "actual_vs_predicted.png"), width=Inches(6.7))
    doc.add_heading("Feature importance interpretation", 1)
    paragraph(doc, f"A horizontal bar chart was selected to keep feature names readable while showing relative rank. The highest fitted importance is `{imp.iloc[0].feature}` ({imp.iloc[0].importance:.3f}), followed by `{imp.iloc[1].feature}` ({imp.iloc[1].importance:.3f}). These values describe tree split behaviour, not causal impact; correlated lags and rolling means can redistribute importance.")
    doc.add_picture(str(OUT / "feature_importance.png"), width=Inches(5.8))
    doc.add_heading("Seven day recursive forecast", 1)
    paragraph(doc, "Each predicted day is appended to the historical series before features for the next day are calculated. That avoids using unavailable future actuals, but can compound error across the horizon.")
    table(doc, ["Forecast date", "Predicted orders"], [[r.date, f"{r.predicted_demand:.1f}"] for r in f.itertuples(index=False)])
    doc.add_heading("Limitations and actionable next steps", 1)
    paragraph(doc, "Olist is a fixed historical dataset and has sparse endpoint behaviour. It lacks known future promotions, public holidays, price, inventory, seller capacity, cancellations, and stock-outs. One holdout split is honest but insufficient to establish long-run stability. Operational deployment should refresh data, use rolling-origin backtests, monitor MAE against seasonal naive, add known future drivers, and issue prediction intervals beside point forecasts.")
    doc.save(REPORTS / "week3_demand_forecasting_report.docx")

def main():
    REPORTS.mkdir(exist_ok=True); c = load_context()
    (REPORTS / "week3_report.md").write_text(narrative(c), encoding="utf-8")
    build_docx(c)

if __name__ == "__main__": main()
