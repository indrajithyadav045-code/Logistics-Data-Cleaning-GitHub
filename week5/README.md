# Week 5 – Logistics Simulation: Inventory Management and Replenishment Optimisation

## Project Objective

This project builds a stochastic inventory simulation calibrated from the Olist Brazilian
e-commerce dataset. Six scenarios are evaluated using Monte Carlo analysis (150 replications each)
to identify the optimal replenishment policy balancing cost and customer service level.

---

## Dataset Requirements

| File | Location |
|------|----------|
| `olist_orders_dataset.csv` | `D:\Logistics-Data-Cleaning\data\raw\` |

The simulation reads demand data from the Olist orders dataset. The raw file must be present
at the path configured in `config/simulation_config.py` (`RAW_ORDERS`).

---

## Folder Structure

```
week5/
├── config/
│   └── simulation_config.py    # All parameters, paths, scenario settings
├── src/
│   ├── data_preparation.py     # Data loading, cleaning, daily aggregation
│   ├── demand_analysis.py      # Statistical demand modelling, distribution fitting
│   ├── inventory_policy.py     # (s,Q) replenishment policy implementation
│   ├── simulation_engine.py    # Monte Carlo simulation core engine
│   ├── scenario_analysis.py    # Six scenarios + grid-search optimisation
│   ├── statistical_analysis.py # Hypothesis tests, confidence intervals, effect sizes
│   ├── visualization.py        # 10 publication-quality charts
│   ├── generate_report.py      # Professional DOCX report generator
│   └── main.py                 # Pipeline orchestrator
├── tests/
│   └── test_simulation.py      # 20+ unit and integration tests
├── data/
│   ├── processed/              # Cleaned daily demand CSV, summaries
│   └── results/                # Scenario replication results, grid search output
├── plots/                      # 10 PNG charts (150 dpi)
├── reports/
│   └── Week5_Logistics_Simulation_Report.docx
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Create and activate a virtual environment (Windows PowerShell)

```powershell
cd D:\Logistics-Data-Cleaning-GitHub
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r week5/requirements.txt
```

---

## Running the Simulation

From the repository root (PowerShell):

```powershell
cd D:\Logistics-Data-Cleaning-GitHub
python week5/src/main.py
```

Expected runtime: approximately 8–15 minutes depending on hardware
(150 replications × 6 scenarios + 375 grid-search combinations × 30 replications).

---

## Running Tests

```powershell
cd D:\Logistics-Data-Cleaning-GitHub
pytest week5/tests/test_simulation.py -v
```

Tests cover:
- Inventory conservation
- Demand fulfillment calculations
- Replenishment timing and lead-time bounds
- Cost calculation accuracy
- Stockout metric consistency
- Reproducibility (same seed = same result)
- Scenario configuration validity
- Statistical summary correctness

---

## Output Descriptions

| Output | Location | Description |
|--------|----------|-------------|
| Daily demand CSV | `week5/data/processed/daily_demand.csv` | Cleaned daily demand time series |
| Weekday summary | `week5/data/processed/weekday_demand_summary.csv` | Demand by day of week |
| Grid search results | `week5/data/results/grid_search_results.csv` | All (s,Q) combinations evaluated |
| Scenario summaries | `week5/data/results/scenario_summary.csv` | Aggregated metrics per scenario |
| Measured results | `week5/data/results/measured_results.json` | Key metrics in JSON for reporting |
| Pairwise tests | `week5/data/results/pairwise_tests.csv` | Statistical comparisons vs baseline |
| Charts | `week5/plots/01_*.png` … `10_*.png` | 10 visualisations (150 dpi) |
| DOCX report | `week5/reports/Week5_Logistics_Simulation_Report.docx` | Full technical report |

---

## Configuration Changes

All scenario parameters are in `week5/config/simulation_config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_REPLICATIONS` | 150 | Monte Carlo replications per scenario |
| `SIM_HORIZON_DAYS` | 365 | Simulation length in days |
| `WARMUP_DAYS` | 30 | Days excluded from metrics |
| `BASELINE_REORDER_POINT` | 1,200 | Reorder trigger level (units) |
| `BASELINE_ORDER_QUANTITY` | 500 | Fixed order batch size (units) |
| `UNIT_HOLDING_COST_PER_DAY` | 0.50 | R$ per unit per day |
| `UNIT_ORDERING_COST` | 500.00 | R$ per purchase order |
| `UNIT_SHORTAGE_COST` | 25.00 | R$ per unfulfilled unit |

To run a faster test, reduce `N_REPLICATIONS` to 20 in `simulation_config.py`.

---

## Assumption Documentation

The following values are **simulation assumptions** (not measured from Olist data):

- Initial on-hand inventory: 2,000 units
- Unit holding cost: R$0.50/unit/day
- Fixed ordering cost: R$500/order
- Lost-sale shortage penalty: R$25/unit
- Baseline lead time: Uniform(5, 9) days
- Demand surge multiplier: 1.40 (40%)
- Supplier disruption probability: 5%/day
- Disruption duration: Uniform(1, 7) days

---

## Reproducibility Instructions

1. Use the same Python version: 3.11.x
2. Install exact package versions from `requirements.txt`
3. Do **not** change `RANDOM_SEED_BASE` in `simulation_config.py`
4. Run `python week5/src/main.py` from the repository root
5. Results will be bit-for-bit identical across runs with the same configuration

---

## Important Notes

- **Never modify** Week 1, 2, 3 or 4 files
- **Never commit** raw datasets, API keys or virtual environment folders
- The `.gitignore` excludes `.venv/`, `__pycache__/`, `*.pyc` and `data/raw/`
