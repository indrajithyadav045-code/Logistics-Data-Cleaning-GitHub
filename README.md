# Logistics Data Cleaning

Week 1 project for YuvaIntern: **Data Collection and Cleaning Strategies in Logistics**.

## Overview

This project prepares the Olist Brazilian E-Commerce Public Dataset for logistics analysis. It uses orders, order items, and customers data to produce clean source tables, an order-level analysis table, data-quality flags, validation outputs, and anomaly evidence.

## Results

- Orders: 99,441 rows and 15 columns
- Order items: 112,650 rows and 9 columns
- Customers: 99,441 rows and 5 columns
- Analysis dataset: 99,441 rows and 21 columns
- Exact duplicate rows removed: 0 in all three source datasets
- Missing order timestamps retained: 160 approval, 1,783 carrier, and 2,965 customer-delivery values
- Timestamp anomalies flagged and retained: 166 carrier-before-purchase and 23 delivery-before-carrier cases
- Six final integrity checks: PASS

## Project Structure

```text
src/                 Python workflow scripts
reports/             Submission-ready Week 1 report
data/processed/      Compact quality, validation, and anomaly evidence CSVs
```

## Running the Workflow

1. Download the Olist Brazilian E-Commerce Public Dataset from Kaggle.
2. Put the required CSV files in `data/raw/` (this directory is excluded from the repository because the raw dataset is large and subject to its source terms).
3. Install dependencies: `pip install pandas numpy`.
4. Run the scripts in this order:

```bash
python src/inspect_data.py
python src/data_cleaning.py
python src/investigate_anomalies.py
python src/anomaly_analysis.py
python src/validate_cleaning.py
python src/advanced_quality_check.py
python src/final_integrity_check.py
```

## Data Attribution

Source: Olist, *Brazilian E-Commerce Public Dataset*, Kaggle: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

The data is a historical public dataset for learning and analysis. Follow the dataset page and Kaggle terms when downloading or using the raw files. This repository does not include the raw data, does not claim live logistics collection, and is not a production deployment.
