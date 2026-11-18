"""
test_integrated_pipeline.py
===========================
Comprehensive unit and integration test suite for the Week 6 Integrated
Logistics Data Science Project.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Insert paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week6" / "config"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week6" / "src"))

from project_config import (
    DATA_PROCESSED,
    DATA_RESULTS,
    PLOTS_DIR,
    STATE_COORDINATES,
    CANDIDATE_HUBS,
)
from data_pipeline import calculate_haversine_distance, get_state_distance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def processed_data() -> pd.DataFrame:
    """Load the processed consolidated dataset."""
    path = DATA_PROCESSED / "consolidated_logistics_orders.csv"
    if not path.exists():
        from data_pipeline import load_and_clean_data
        return load_and_clean_data()
    return pd.read_csv(path, low_memory=False)


@pytest.fixture(scope="module")
def eda_results() -> dict:
    """Load the EDA summary results JSON."""
    path = DATA_RESULTS / "eda_metrics.json"
    if not path.exists():
        from exploratory_analysis import run_exploratory_analysis
        return run_exploratory_analysis()
    with open(path, "r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ml_results() -> dict:
    """Load predictive ML metrics JSON."""
    path = DATA_RESULTS / "predictive_model_metrics.json"
    if not path.exists():
        from predictive_modeling import train_predictive_models
        return train_predictive_models()
    with open(path, "r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def opt_results() -> dict:
    """Load network optimization metrics JSON."""
    path = DATA_RESULTS / "network_optimization_metrics.json"
    if not path.exists():
        from network_optimization import run_network_optimization
        return run_network_optimization()
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Data Pipeline & Spatial Integrity Tests
# ---------------------------------------------------------------------------
class TestDataPipeline:
    def test_dataset_non_empty(self, processed_data):
        """Ensure consolidated dataset is substantial (> 80k rows)."""
        assert len(processed_data) > 80_000

    def test_no_negative_delivery_durations(self, processed_data):
        """Delivery days and transit days must be strictly positive."""
        assert (processed_data["delivery_days"] > 0).all()
        assert (processed_data["carrier_dispatch_days"] >= 0).all()
        assert (processed_data["in_transit_days"] >= 0).all()

    def test_binary_delay_flag_consistency(self, processed_data):
        """is_delayed must be 1 if and only if delay_days > 0."""
        expected = (processed_data["delay_days"] > 0).astype(int)
        assert (processed_data["is_delayed"] == expected).all()

    def test_haversine_distance_properties(self):
        """Haversine distance between same coordinates must be zero."""
        assert calculate_haversine_distance(-23.55, -46.63, -23.55, -46.63) == 0.0

    def test_state_distance_symmetry_and_floor(self):
        """State distance must be positive and intrastate must return floor distance."""
        dist_sp_rj = get_state_distance("SP", "RJ")
        dist_rj_sp = get_state_distance("RJ", "SP")
        assert dist_sp_rj > 0.0
        assert abs(dist_sp_rj - dist_rj_sp) < 1e-6
        assert get_state_distance("SP", "SP") == 45.0

    def test_all_state_coordinates_valid(self):
        """All 27 Brazilian states must have valid lat/lon."""
        assert len(STATE_COORDINATES) == 27
        for state, (lat, lon) in STATE_COORDINATES.items():
            assert -35.0 <= lat <= 5.0
            assert -75.0 <= lon <= -30.0


# ---------------------------------------------------------------------------
# 2. Exploratory Analysis & Customer Satisfaction Tests
# ---------------------------------------------------------------------------
class TestExploratoryAnalysis:
    def test_delay_rate_within_bounds(self, eda_results):
        """Delay rate percentage must be between 0 and 100%."""
        rate = eda_results["dataset_summary"]["delay_rate_pct"]
        assert 0.0 <= rate <= 100.0

    def test_customer_rating_collapse(self, eda_results):
        """Delayed deliveries must receive lower ratings than on-time deliveries."""
        ontime = eda_results["customer_satisfaction"]["mean_ontime_rating"]
        delayed = eda_results["customer_satisfaction"]["mean_delayed_rating"]
        assert ontime > delayed
        assert ontime >= 4.0
        assert delayed < 3.5

    def test_interstate_vs_intrastate_counts(self, eda_results):
        """Interstate + intrastate orders must sum to total orders."""
        rows = eda_results["interstate_breakdown"]
        total = sum(r["order_count"] for r in rows)
        assert total == eda_results["dataset_summary"]["total_orders"]


# ---------------------------------------------------------------------------
# 3. Machine Learning & Predictive Modeling Tests
# ---------------------------------------------------------------------------
class TestPredictiveModeling:
    def test_classification_auc_above_baseline(self, ml_results):
        """Ensemble models must achieve ROC-AUC >= 0.70."""
        for m in ml_results["classification_metrics"]:
            assert m["roc_auc"] >= 0.65, f"Model {m['model']} ROC-AUC too low: {m['roc_auc']}"

    def test_best_model_performance(self, ml_results):
        """HistGradientBoosting must achieve strong discrimination (AUC > 0.75)."""
        hgb = next(m for m in ml_results["classification_metrics"] if "HistGradientBoosting" in m["model"])
        assert hgb["roc_auc"] > 0.75
        assert hgb["recall"] > 0.60

    def test_regression_mae_sanity(self, ml_results):
        """Regression MAE must be realistic (between 2 and 6 days)."""
        for m in ml_results["regression_metrics"]:
            assert 2.0 <= m["mae_days"] <= 6.0
            assert m["r2_score"] > 0.30

    def test_top_feature_importance_valid(self, ml_results):
        """Feature importance must be non-empty with carrier_dispatch or distance among top."""
        top_feats = [f["feature"] for f in ml_results["top_features"]]
        assert len(top_feats) == 10
        assert any(f in top_feats for f in ["carrier_dispatch_days", "distance_km", "total_freight"])


# ---------------------------------------------------------------------------
# 4. Operations Research & Optimization Tests
# ---------------------------------------------------------------------------
class TestNetworkOptimization:
    def test_optimization_reduces_cost(self, opt_results):
        """Optimized multi-hub network must achieve lower cost than centralized baseline."""
        as_is = opt_results["as_is_centralized"]["total_cost_brl"]
        opt = opt_results["optimized_multi_hub"]["total_cost_brl"]
        assert opt < as_is
        assert opt_results["improvements"]["annual_cost_savings_brl"] > 100_000.0

    def test_optimization_reduces_transit_distance(self, opt_results):
        """Optimized network must reduce average transit distance."""
        as_is_dist = opt_results["as_is_centralized"]["mean_distance_km"]
        opt_dist = opt_results["optimized_multi_hub"]["mean_distance_km"]
        assert opt_dist < as_is_dist

    def test_optimization_reduces_carbon_footprint(self, opt_results):
        """Optimized network must reduce total diesel CO2 emissions."""
        co2_as_is = opt_results["as_is_centralized"]["co2_emissions_tons"]
        co2_opt = opt_results["optimized_multi_hub"]["co2_emissions_tons"]
        assert co2_opt < co2_as_is
        assert opt_results["improvements"]["co2_reduction_pct"] > 50.0

    def test_hub_throughput_distribution(self, opt_results):
        """All hub throughput percentages must sum to 100%."""
        throughputs = [h["throughput_pct"] for h in opt_results["hub_throughput"]]
        assert abs(sum(throughputs) - 100.0) < 1e-4

    def test_assignment_file_validity(self):
        """Assignment CSV must cover all 27 destination states."""
        assign_csv = pd.read_csv(DATA_RESULTS / "optimized_hub_assignments.csv")
        assert len(assign_csv) == 27
        assert set(assign_csv["assigned_hub"]).issubset(set(CANDIDATE_HUBS))


# ---------------------------------------------------------------------------
# 5. Visualizations & Artifacts Verification
# ---------------------------------------------------------------------------
class TestArtifacts:
    def test_all_ten_plots_exist(self):
        """Verify that all 10 PNG charts were generated and have non-zero size."""
        expected_plots = [
            "01_delivery_duration_distribution.png",
            "02_interstate_vs_intrastate_performance.png",
            "03_review_score_vs_delivery_delay.png",
            "04_regional_od_heatmap.png",
            "05_state_logistics_bottlenecks.png",
            "06_predictive_roc_pr_curves.png",
            "07_feature_importance_ranking.png",
            "08_regression_actual_vs_predicted.png",
            "09_optimization_cost_distance_comparison.png",
            "10_hub_volume_and_carbon_savings.png",
        ]
        for plot_name in expected_plots:
            plot_path = PLOTS_DIR / plot_name
            assert plot_path.exists(), f"Missing plot: {plot_name}"
            assert plot_path.stat().st_size > 10_000, f"Plot {plot_name} is abnormally small"
