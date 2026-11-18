# Week 6: Integrated Data Science Project for Logistics Efficiency Improvement

## Project Title
**End-to-End Predictive Delay Modeling, Bottleneck Diagnostics, and Multi-Hub Prescriptive Network Optimization in E-Commerce Supply Chains**

---

## Executive Overview
This capstone project combines four integrated pillars of logistics data science to eliminate fulfillment delays, reduce freight costs, and minimize environmental impact across the Brazilian e-commerce logistics network:
1. **Data Cleaning & Entity Consolidation**: Ingested and harmonized 6 relational datasets (96,238 delivered orders across 730 continuous calendar days) with zero data leakage.
2. **Exploratory Data Analysis & Spatial Diagnostics**: Uncovered that 64.0% of orders are cross-state shipments experiencing a 90% transit latency penalty (15.07d vs 7.93d) and quantified customer satisfaction collapse (-1.68 review stars for late deliveries).
3. **Predictive Machine Learning**: Built classification (HistGradientBoosting ROC-AUC 0.798) and regression (MAE 4.08d) models to forecast delivery delays prior to SLA failure.
4. **Prescriptive Operations Research**: Formulated and solved an 8-hub linear programming network allocation model, reducing logistics costs by 37.15% (R$ 320,315.57 savings), cutting average transit distance by 75.9% (518.2 km down to 125.0 km), and decreasing diesel fleet carbon emissions by 75.9%.

---

## Directory Architecture
```
week6/
├── config/
│   └── project_config.py          # Central paths, state centroids, cost factors, and settings
├── data/
│   ├── processed/
│   │   └── consolidated_logistics_orders.csv
│   └── results/
│       ├── classification_model_metrics.csv
│       ├── distance_tier_performance.csv
│       ├── eda_metrics.json
│       ├── feature_importance.csv
│       ├── hub_throughput_distribution.csv
│       ├── interstate_vs_intrastate.csv
│       ├── network_optimization_metrics.json
│       ├── optimized_hub_assignments.csv
│       ├── predictive_model_metrics.json
│       ├── regional_od_matrix_delivery_days.csv
│       ├── regression_model_metrics.csv
│       └── state_logistics_performance.csv
├── plots/
│   ├── 01_delivery_duration_distribution.png
│   ├── 02_interstate_vs_intrastate_performance.png
│   ├── 03_review_score_vs_delivery_delay.png
│   ├── 04_regional_od_heatmap.png
│   ├── 05_state_logistics_bottlenecks.png
│   ├── 06_predictive_roc_pr_curves.png
│   ├── 07_feature_importance_ranking.png
│   ├── 08_regression_actual_vs_predicted.png
│   ├── 09_optimization_cost_distance_comparison.png
│   └── 10_hub_volume_and_carbon_savings.png
├── reports/
│   └── Week6_Integrated_Logistics_Data_Science_Report.docx
├── src/
│   ├── data_pipeline.py           # Ingestion, cleaning, feature engineering
│   ├── exploratory_analysis.py    # Spatial O-D matrix, latency diagnostics
│   ├── predictive_modeling.py     # Delay risk classification & duration regression
│   ├── network_optimization.py    # Multi-hub linear programming solver
│   ├── visualization.py           # 10 publication charts
│   ├── generate_report.py         # 25-chapter Word technical report generator
│   └── main.py                    # End-to-end pipeline orchestrator
├── tests/
│   └── test_integrated_pipeline.py# 19 automated unit & integration tests
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1. Prerequisites & Virtual Environment
Ensure Python 3.11+ is installed. In Windows PowerShell:
```powershell
cd D:\Logistics-Data-Cleaning-GitHub
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r week6/requirements.txt
```

---

## Executing the Project

### Run the Full Integrated Pipeline
To re-run the entire pipeline from data ingestion to report generation:
```powershell
python week6/src/main.py
```
*Expected runtime: ~45–60 seconds.*

### Run the Automated Test Suite
To verify data invariants, modeling accuracy, and optimization constraints:
```powershell
python -m pytest week6/tests/test_integrated_pipeline.py -v
```
*Expected test outcome: 19 passed.*

---

## Key Results Summary

| Metric | As-Is Centralized (SP Origin) | Optimized Multi-Hub (8 Regional Hubs) | Operational Benefit |
|---|---|---|---|
| **Annual Logistics Cost** | R$ 862,188.22 | R$ 541,872.64 | **-R$ 320,315.57 (-37.15%)** |
| **Mean Transit Distance** | 518.2 km | 125.0 km | **-393.2 km (-75.9%)** |
| **Mean Transit Duration** | 2.80 days | 1.81 days | **-0.98 days (-35.2%)** |
| **Diesel CO2 Emissions** | 6.4 Tons | 1.6 Tons | **-4.9 Tons (-75.9%)** |
| **Delay Prediction ROC-AUC**| — | **0.7979** (70.1% Recall) | Early warning capability |
| **Review Rating Collapse** | **4.30 ★** (On-time) | **2.61 ★** (Delayed) | Avoids customer rating loss |

---

## Deliverables
- **Technical Report**: `week6/reports/Week6_Integrated_Logistics_Data_Science_Report.docx` (611 KB, 25 chapters with embedded figures and tables).
- **Tested Source Code**: Modular, annotated Python scripts adhering to PEP 8.
- **Figures**: 10 high-resolution charts in `week6/plots/`.
- **Reproducible Data**: Processed datasets and tabular outputs in `week6/data/results/`.
