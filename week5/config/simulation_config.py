"""
simulation_config.py
====================
Central configuration for the Week 5 Logistics Simulation project.

All scenario parameters, cost assumptions, random seeds and file paths
are defined here so that every module reads from a single source of truth.
Nothing inside this file is loaded from an external secret store.

IMPORTANT – clearly labelled assumptions
-----------------------------------------
Parameters marked [ASSUMPTION] are not measured from the Olist dataset.
They are intentional modelling choices that reflect plausible warehouse
operating conditions for a medium-volume Brazilian e-commerce distributor.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# 1.  FILE PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
# The raw Olist dataset lives in the sibling repo directory, not this repo.
# Adjust this path if your data is stored elsewhere.
_DATA_FOLDER = Path(r"D:\Logistics-Data-Cleaning\data\raw")
RAW_ORDERS: Path = _DATA_FOLDER / "olist_orders_dataset.csv"

WEEK5: Path = PROJECT_ROOT / "week5"
DATA_PROCESSED: Path = WEEK5 / "data" / "processed"
DATA_RESULTS: Path = WEEK5 / "data" / "results"
PLOTS_DIR: Path = WEEK5 / "plots"
REPORTS_DIR: Path = WEEK5 / "reports"

# Ensure output directories exist at import time
for _d in (DATA_PROCESSED, DATA_RESULTS, PLOTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2.  DATA PREPARATION
# ---------------------------------------------------------------------------
# Only "delivered" orders represent fulfilled customer demand entering the
# warehouse fulfilment pipeline.  Cancelled / unavailable orders are excluded.
ORDER_STATUS_FILTER: list[str] = ["delivered", "shipped", "invoiced", "processing", "approved"]

# ---------------------------------------------------------------------------
# 3.  SIMULATION HORIZON
# ---------------------------------------------------------------------------
SIM_HORIZON_DAYS: int = 365          # one calendar year per replication
N_REPLICATIONS: int = 100            # Monte Carlo replications per scenario
RANDOM_SEED_BASE: int = 42           # replication k uses seed = BASE + k
WARMUP_DAYS: int = 30                # days discarded from each replication's statistics

# ---------------------------------------------------------------------------
# 4.  INVENTORY COST PARAMETERS  [ASSUMPTION]
# ---------------------------------------------------------------------------
# These are plausible unit costs for a medium-volume logistics warehouse.
# They are NOT measured from the Olist dataset.
UNIT_HOLDING_COST_PER_DAY: float = 0.50    # R$ per unit per day on hand
UNIT_ORDERING_COST: float = 500.00         # R$ fixed cost per purchase order placed
UNIT_SHORTAGE_COST: float = 25.00          # R$ penalty per unit of unmet demand (lost-sales)

# ---------------------------------------------------------------------------
# 5.  BASELINE REPLENISHMENT POLICY  [ASSUMPTION]
# ---------------------------------------------------------------------------
# Continuous-review (s, Q) policy:
#   s = reorder point (units)   – order when inventory position falls to or below s
#   Q = order quantity (units)  – fixed replenishment batch size
#
# EOQ-inspired initial values: Q ≈ sqrt(2*D*K / h)
# With D ≈ 128 units/day * 365 = 46 720 units/year,
#      K = 500, h = 0.50 * 365 = 182.5  → EOQ ≈ 506 units
# Reorder point: s = mean_demand_per_day * (mean_lead_time + z * sigma_lt)
#   mean_lt = 7, sigma_lt = 2, z(95%) ≈ 1.65
#   demand variability: sigma_d = 60 (from historical data)
#   safety stock ≈ 1.65 * sqrt(7) * 60 ≈ 262
#   s ≈ 128 * 7 + 262 ≈ 1158 → rounded to 1200 for safety
BASELINE_REORDER_POINT: int = 1_200        # units
BASELINE_ORDER_QUANTITY: int = 500         # units per replenishment batch
INITIAL_INVENTORY: int = 2_000            # units at simulation start [ASSUMPTION]

# Demand model: "bootstrap" | "poisson" | "negbinom"
DEMAND_MODEL: str = "bootstrap"           # justified in demand_analysis.py

# Unfulfilled demand model: "lost_sales" | "backorder"
# Lost-sales is used: once a day ends with unmet demand, those units are gone.
# This is realistic for perishable / time-sensitive consumer goods.
FULFILLMENT_MODEL: str = "lost_sales"

# ---------------------------------------------------------------------------
# 6.  SUPPLIER LEAD TIME  [ASSUMPTION]
# ---------------------------------------------------------------------------
# Lead time in days follows a discrete uniform distribution: Uniform(LT_MIN, LT_MAX)
BASELINE_LT_MIN: int = 5     # minimum supplier lead time (days)
BASELINE_LT_MAX: int = 9     # maximum supplier lead time (days)  → mean 7 days

# ---------------------------------------------------------------------------
# 7.  SCENARIO PARAMETERS
# ---------------------------------------------------------------------------
SCENARIOS: dict = {
    "S1_Baseline": {
        "label": "Baseline",
        "demand_multiplier": 1.0,
        "lt_min": BASELINE_LT_MIN,
        "lt_max": BASELINE_LT_MAX,
        "disruption_prob": 0.0,
        "disruption_duration_max": 0,
        "reorder_point": BASELINE_REORDER_POINT,
        "order_quantity": BASELINE_ORDER_QUANTITY,
        "safety_stock_extra": 0,
    },
    "S2_Demand_Surge": {
        "label": "Demand Surge",
        "demand_multiplier": 1.40,       # 40 % demand increase [ASSUMPTION]
        "lt_min": BASELINE_LT_MIN,
        "lt_max": BASELINE_LT_MAX,
        "disruption_prob": 0.0,
        "disruption_duration_max": 0,
        "reorder_point": BASELINE_REORDER_POINT,
        "order_quantity": BASELINE_ORDER_QUANTITY,
        "safety_stock_extra": 0,
    },
    "S3_Supplier_Delay": {
        "label": "Supplier Delay",
        "demand_multiplier": 1.0,
        "lt_min": 10,                    # extended lead time [ASSUMPTION]
        "lt_max": 18,
        "disruption_prob": 0.0,
        "disruption_duration_max": 0,
        "reorder_point": BASELINE_REORDER_POINT,
        "order_quantity": BASELINE_ORDER_QUANTITY,
        "safety_stock_extra": 0,
    },
    "S4_Supplier_Disruption": {
        "label": "Supplier Disruption",
        "demand_multiplier": 1.0,
        "lt_min": BASELINE_LT_MIN,
        "lt_max": BASELINE_LT_MAX,
        "disruption_prob": 0.05,         # 5 % daily probability of disruption [ASSUMPTION]
        "disruption_duration_max": 7,    # disruption lasts 1–7 days [ASSUMPTION]
        "reorder_point": BASELINE_REORDER_POINT,
        "order_quantity": BASELINE_ORDER_QUANTITY,
        "safety_stock_extra": 0,
    },
    "S5_Safety_Stock": {
        "label": "Safety Stock Strategy",
        "demand_multiplier": 1.0,
        "lt_min": BASELINE_LT_MIN,
        "lt_max": BASELINE_LT_MAX,
        "disruption_prob": 0.0,
        "disruption_duration_max": 0,
        "reorder_point": 1_800,          # elevated reorder point [ASSUMPTION]
        "order_quantity": BASELINE_ORDER_QUANTITY,
        "safety_stock_extra": 600,       # additional buffer units [ASSUMPTION]
    },
    "S6_Optimized": {
        "label": "Optimized Replenishment",
        "demand_multiplier": 1.0,
        "lt_min": BASELINE_LT_MIN,
        "lt_max": BASELINE_LT_MAX,
        "disruption_prob": 0.0,
        "disruption_duration_max": 0,
        "reorder_point": 950,            # grid-search result (filled in by scenario_analysis.py)
        "order_quantity": 650,
        "safety_stock_extra": 0,
    },
}

# Grid-search ranges for Scenario 6 optimisation
OPT_ROP_RANGE: list[int] = list(range(600, 1801, 100))   # 13 values
OPT_QTY_RANGE: list[int] = list(range(300, 1001, 100))   # 8 values
OPT_OBJECTIVE: str = "total_cost"   # minimise total_cost subject to fill_rate >= OPT_MIN_FILL
OPT_MIN_FILL_RATE: float = 0.95     # service-level constraint

# ---------------------------------------------------------------------------
# 8.  PLOT STYLING
# ---------------------------------------------------------------------------
PLOT_STYLE: str = "seaborn-v0_8-whitegrid"
PLOT_FIGSIZE_WIDE: tuple[int, int] = (14, 6)
PLOT_FIGSIZE_SQUARE: tuple[int, int] = (10, 8)
PLOT_DPI: int = 150
PLOT_PALETTE: str = "tab10"
