"""
project_config.py
=================
Central configuration for Week 6: Integrated Data Science Project for Logistics Efficiency.
Defines file paths, modeling hyperparameters, optimization constraints, cost factors,
and geographic centroid coordinates for Brazilian federative units.
"""
from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. FILE PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DATA_DIR: Path = Path(r"D:\Logistics-Data-Cleaning\data\raw")

WEEK6_DIR: Path = PROJECT_ROOT / "week6"
DATA_PROCESSED: Path = WEEK6_DIR / "data" / "processed"
DATA_RESULTS: Path = WEEK6_DIR / "data" / "results"
PLOTS_DIR: Path = WEEK6_DIR / "plots"
REPORTS_DIR: Path = WEEK6_DIR / "reports"

# Ensure output directories exist
for _d in (DATA_PROCESSED, DATA_RESULTS, PLOTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. RAW DATASETS
# ---------------------------------------------------------------------------
ORDERS_CSV = RAW_DATA_DIR / "olist_orders_dataset.csv"
ITEMS_CSV = RAW_DATA_DIR / "olist_order_items_dataset.csv"
CUSTOMERS_CSV = RAW_DATA_DIR / "olist_customers_dataset.csv"
SELLERS_CSV = RAW_DATA_DIR / "olist_sellers_dataset.csv"
PRODUCTS_CSV = RAW_DATA_DIR / "olist_products_dataset.csv"
REVIEWS_CSV = RAW_DATA_DIR / "olist_order_reviews_dataset.csv"

# ---------------------------------------------------------------------------
# 3. EXPERIMENTAL & MODELING PARAMETERS
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# Delivered orders filter
VALID_ORDER_STATUSES: list[str] = ["delivered"]

# Target delay threshold: actual delivery date > estimated delivery date
DELAY_MARGIN_DAYS: float = 0.0  # > 0.0 days late is considered SLA breach

# ---------------------------------------------------------------------------
# 4. BRAZILIAN STATE CENTROIDS (Latitude, Longitude) for Haversine Distance
# ---------------------------------------------------------------------------
STATE_COORDINATES: dict[str, tuple[float, float]] = {
    "AC": (-9.0238, -70.8120),  # Acre
    "AL": (-9.5713, -36.7820),  # Alagoas
    "AP": (0.9020, -52.0030),   # Amapá
    "AM": (-3.4168, -65.8561),  # Amazonas
    "BA": (-12.5797, -41.7007), # Bahia
    "CE": (-5.4984, -39.3206),  # Ceará
    "DF": (-15.7998, -47.8645), # Distrito Federal
    "ES": (-19.1834, -40.3089), # Espírito Santo
    "GO": (-15.8270, -49.8362), # Goiás
    "MA": (-4.9609, -45.2744),  # Maranhão
    "MT": (-12.6819, -55.6244), # Mato Grosso
    "MS": (-20.7722, -54.7852), # Mato Grosso do Sul
    "MG": (-18.5122, -44.5550), # Minas Gerais
    "PA": (-1.9981, -54.9306),  # Pará
    "PB": (-7.2399, -36.7820),  # Paraíba
    "PR": (-25.2521, -52.0215), # Paraná
    "PE": (-8.8137, -36.9541),  # Pernambuco
    "PI": (-7.7183, -42.7289),  # Piauí
    "RJ": (-22.9068, -43.1729), # Rio de Janeiro
    "RN": (-5.7945, -36.5684),  # Rio Grande do Norte
    "RS": (-30.0346, -51.2177), # Rio Grande do Sul
    "RO": (-11.5057, -63.5806), # Rondônia
    "RR": (2.7376, -62.0751),   # Roraima
    "SC": (-27.2423, -50.2189), # Santa Catarina
    "SP": (-23.5505, -46.6333), # São Paulo
    "SE": (-10.5741, -37.3857), # Sergipe
    "TO": (-10.1753, -48.2982), # Tocantins
}

# ---------------------------------------------------------------------------
# 5. REGIONAL CLUSTERS (Brazilian Macro-regions)
# ---------------------------------------------------------------------------
REGION_MAP: dict[str, str] = {
    "SP": "Southeast", "RJ": "Southeast", "MG": "Southeast", "ES": "Southeast",
    "PR": "South", "SC": "South", "RS": "South",
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West", "MS": "Central-West",
    "BA": "Northeast", "PE": "Northeast", "CE": "Northeast", "MA": "Northeast",
    "PB": "Northeast", "RN": "Northeast", "AL": "Northeast", "SE": "Northeast", "PI": "Northeast",
    "AM": "North", "PA": "North", "RO": "North", "TO": "North", "AC": "North", "AP": "North", "RR": "North"
}

# ---------------------------------------------------------------------------
# 6. OPTIMIZATION ASSUMPTIONS & COST PARAMETERS
# ---------------------------------------------------------------------------
# Proposed Forward-Stocking / Regional Fulfillment Hub Locations
CANDIDATE_HUBS: list[str] = ["SP", "RJ", "MG", "PR", "BA", "PE", "RS", "DF"]

# Estimated transportation costs per km per kg (BRL)
COST_PER_KM_PER_KG: float = 0.0025  # R$ 0.0025 per km per kg
BASE_HANDLING_COST_PER_ORDER: float = 4.50  # Fixed cross-docking / handling per order
INTERSTATE_TAX_PENALTY: float = 3.00       # Fixed regulatory / interstate documentation cost
LATE_DELIVERY_PENALTY_PER_DAY: float = 15.00 # Goodwill / concession cost per day late
AVERAGE_TRUCK_SPEED_KM_PER_DAY: float = 400.0 # Realistic road freight average travel per day
CARBON_EMISSIONS_KG_PER_TON_KM: float = 0.062 # CO2 footprint for diesel road haulage
