# Week 2 Exploratory Data Analysis and Visualization

This folder contains the Week 2 YuvaIntern deliverables for exploratory data analysis of Olist logistics data.

## Contents

- `src/week2_eda.py` - reproducible pandas, Matplotlib, and Seaborn analysis
- `plots/` - six generated logistics visualizations
- `summary_metrics.csv` - key EDA metrics
- `report/` - submission-ready DOCX report

## Results at a Glance

- 99,441 orders analysed at source level
- 96,204 chronologically valid delivered orders used for delivery-duration metrics
- Median delivery time: 10.22 days
- Mean delivery time: 12.47 days
- On-time or early share: 91.95%
- Late share: 8.05%
- Freight versus order-value Pearson correlation: 0.411

## Reproduction

Place the three Olist raw CSV files in `data/raw/` at the repository root, then run:

```bash
python week-2/src/week2_eda.py
```

The raw dataset is excluded from this repository. Source and terms: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
