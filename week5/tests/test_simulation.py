"""
test_simulation.py
==================
Unit and integration tests for the Week 5 inventory simulation.

Test groups
-----------
1. Inventory conservation  – inventory never negative; fulfilled ≤ demand
2. Demand fulfillment      – fill_rate in [0, 1]
3. Replenishment timing    – orders arrive after the correct lead time
4. Lead-time handling      – within configured bounds
5. Cost calculations       – components sum to total; no negative costs
6. Stockout calculations   – consistent with fulfilled/demand
7. Reproducibility         – identical seed → identical result
8. Scenario configuration  – all six scenarios have valid PolicyParameters
9. Statistical summaries   – CI bounds, mean within [min, max]

Usage
-----
    pytest week5/tests/test_simulation.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from src and config
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week5" / "config"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week5" / "src"))

from simulation_config import (
    SCENARIOS,
    INITIAL_INVENTORY,
    UNIT_HOLDING_COST_PER_DAY,
    UNIT_ORDERING_COST,
    UNIT_SHORTAGE_COST,
    RANDOM_SEED_BASE,
)
from inventory_policy import PolicyParameters, InventoryPolicy, PendingOrder
from simulation_engine import (
    run_single_replication,
    run_monte_carlo,
    results_to_dataframe,
    DemandSampler,
    ReplicationResult,
)
from statistical_analysis import descriptive_stats, cohens_d, _t_ci


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_pool() -> np.ndarray:
    """Small demand pool: 50 values drawn from N(100, 20)."""
    rng = np.random.default_rng(99)
    return np.abs(rng.normal(100, 20, size=200)).astype(float)


@pytest.fixture
def simple_weekday_pools(simple_pool) -> dict:
    """Weekday pools: same pool for all weekdays."""
    return {d: simple_pool for d in range(7)}


@pytest.fixture
def baseline_params() -> PolicyParameters:
    """Standard baseline PolicyParameters."""
    cfg = SCENARIOS["S1_Baseline"]
    return PolicyParameters(
        reorder_point=cfg["reorder_point"],
        order_quantity=cfg["order_quantity"],
        lt_min=cfg["lt_min"],
        lt_max=cfg["lt_max"],
        disruption_prob=0.0,
        disruption_duration_max=0,
        safety_stock_extra=0,
    )


@pytest.fixture
def baseline_result(baseline_params, simple_pool, simple_weekday_pools) -> ReplicationResult:
    """Single replication for unit tests."""
    return run_single_replication(
        replication=0,
        policy_params=baseline_params,
        global_pool=simple_pool,
        weekday_pools=simple_weekday_pools,
        sim_days=200,
        warmup_days=20,
        store_daily=True,
    )


# ---------------------------------------------------------------------------
# 1. Inventory conservation
# ---------------------------------------------------------------------------

class TestInventoryConservation:
    def test_inventory_never_negative(self, baseline_result):
        """Closing inventory must be >= 0 on every day."""
        for rec in baseline_result.daily_records:
            assert rec.closing_inventory >= 0, (
                f"Negative inventory on day {rec.day}: {rec.closing_inventory}"
            )

    def test_fulfilled_never_exceeds_demand(self, baseline_result):
        """Fulfilled units cannot exceed demand."""
        for rec in baseline_result.daily_records:
            assert rec.fulfilled <= rec.demand + 1e-9, (
                f"Day {rec.day}: fulfilled ({rec.fulfilled}) > demand ({rec.demand})"
            )

    def test_fulfilled_never_exceeds_opening_inventory(self, baseline_result):
        """Fulfilled units cannot exceed what was available."""
        for rec in baseline_result.daily_records:
            available = rec.opening_inventory + rec.incoming_replenishment
            assert rec.fulfilled <= available + 1e-9, (
                f"Day {rec.day}: fulfilled ({rec.fulfilled}) > available ({available})"
            )

    def test_inventory_conservation_equation(self, baseline_result):
        """closing ≈ opening + incoming - fulfilled (± 1 unit for int rounding).

        opening_inventory in DayRecord = on-hand BEFORE replenishments and fulfilment.
        Conservation identity:
            closing_inventory = opening_inventory + incoming_replenishment - fulfilled
        A tolerance of 1 unit covers floor() rounding of float demand values.
        """
        for rec in baseline_result.daily_records:
            expected = rec.opening_inventory + rec.incoming_replenishment - rec.fulfilled
            assert abs(rec.closing_inventory - expected) <= 1.0, (
                f"Day {rec.day}: conservation violated. "
                f"opening={rec.opening_inventory} + incoming={rec.incoming_replenishment} "
                f"- fulfilled={rec.fulfilled:.1f} = {expected:.1f} "
                f"but closing={rec.closing_inventory}"
            )


# ---------------------------------------------------------------------------
# 2. Demand fulfillment
# ---------------------------------------------------------------------------

class TestDemandFulfillment:
    def test_fill_rate_in_unit_interval(self, baseline_result):
        """Fill rate must be between 0 and 1 (inclusive)."""
        assert 0.0 <= baseline_result.fill_rate <= 1.0

    def test_lost_sales_non_negative(self, baseline_result):
        """Lost sales (unfulfilled) must be >= 0."""
        assert baseline_result.total_unfulfilled >= 0.0

    def test_fill_rate_consistent(self, baseline_result):
        """fill_rate == demand_fulfillment_rate by definition."""
        assert abs(baseline_result.fill_rate - baseline_result.demand_fulfillment_rate) < 1e-9


# ---------------------------------------------------------------------------
# 3. Replenishment timing
# ---------------------------------------------------------------------------

class TestReplenishmentTiming:
    def test_orders_arrive_within_lead_time(self):
        """Orders placed on day d must arrive within [lt_min, lt_max] days."""
        rng = np.random.default_rng(42)
        params = PolicyParameters(reorder_point=100, order_quantity=50, lt_min=3, lt_max=7)
        policy = InventoryPolicy(params=params, rng=rng)
        for day in range(10):
            order = policy.create_order(current_day=day)
            lt = order.arrival_day - day
            assert params.lt_min <= lt <= params.lt_max, (
                f"Lead time {lt} out of bounds [{params.lt_min}, {params.lt_max}]"
            )

    def test_no_self_arrival(self):
        """An order placed on day d cannot arrive on the same day (lt >= 1)."""
        rng = np.random.default_rng(1)
        params = PolicyParameters(reorder_point=0, order_quantity=10, lt_min=1, lt_max=1)
        policy = InventoryPolicy(params=params, rng=rng)
        order = policy.create_order(current_day=5)
        assert order.arrival_day > 5


# ---------------------------------------------------------------------------
# 4. Lead-time handling
# ---------------------------------------------------------------------------

class TestLeadTimeHandling:
    def test_lead_time_within_bounds(self):
        """Sampled lead times must always be in [lt_min, lt_max]."""
        rng = np.random.default_rng(7)
        params = PolicyParameters(reorder_point=100, order_quantity=50, lt_min=5, lt_max=10)
        policy = InventoryPolicy(params=params, rng=rng)
        samples = [policy.sample_lead_time() for _ in range(1000)]
        assert min(samples) >= 5
        assert max(samples) <= 10

    def test_lead_time_variety(self):
        """With a range > 0, we should see more than one unique value."""
        rng = np.random.default_rng(8)
        params = PolicyParameters(reorder_point=100, order_quantity=50, lt_min=5, lt_max=10)
        policy = InventoryPolicy(params=params, rng=rng)
        samples = {policy.sample_lead_time() for _ in range(100)}
        assert len(samples) > 1


# ---------------------------------------------------------------------------
# 5. Cost calculations
# ---------------------------------------------------------------------------

class TestCostCalculations:
    def test_total_cost_equals_sum_of_components(self, baseline_result):
        """Total cost must equal holding + ordering + shortage."""
        expected = (
            baseline_result.total_holding_cost
            + baseline_result.total_ordering_cost
            + baseline_result.total_shortage_cost
        )
        assert abs(baseline_result.total_cost - expected) < 1.0

    def test_all_cost_components_non_negative(self, baseline_result):
        """No cost component can be negative."""
        assert baseline_result.total_holding_cost >= 0.0
        assert baseline_result.total_ordering_cost >= 0.0
        assert baseline_result.total_shortage_cost >= 0.0

    def test_holding_cost_proportional_to_inventory(self, baseline_result):
        """Holding cost ≈ sum(daily_inventory) × UNIT_HOLDING_COST_PER_DAY."""
        total_inv = sum(
            r.closing_inventory
            for r in baseline_result.daily_records
            if r.day >= 20   # warmup boundary
        )
        expected_holding = total_inv * UNIT_HOLDING_COST_PER_DAY
        # Allow 5% tolerance for edge effects
        tol = 0.05 * expected_holding + 1
        assert abs(baseline_result.total_holding_cost - expected_holding) <= tol

    def test_ordering_cost_proportional_to_n_orders(self, baseline_result):
        """Total ordering cost = n_orders × UNIT_ORDERING_COST."""
        expected = baseline_result.n_orders * UNIT_ORDERING_COST
        assert abs(baseline_result.total_ordering_cost - expected) < 1.0


# ---------------------------------------------------------------------------
# 6. Stockout calculations
# ---------------------------------------------------------------------------

class TestStockoutCalculations:
    def test_stockout_days_consistent(self, baseline_result):
        """stockout_days == number of days where lost_sales > 0."""
        actual = sum(1 for r in baseline_result.daily_records
                     if r.day >= 20 and r.lost_sales > 0)
        assert baseline_result.stockout_days == actual

    def test_stockout_frequency_in_unit_interval(self, baseline_result):
        """Stockout frequency must be in [0, 1]."""
        assert 0.0 <= baseline_result.stockout_frequency <= 1.0

    def test_total_unfulfilled_matches_records(self, baseline_result):
        """total_unfulfilled ≈ sum of daily lost_sales (post-warmup)."""
        actual = sum(r.lost_sales for r in baseline_result.daily_records if r.day >= 20)
        assert abs(baseline_result.total_unfulfilled - actual) < 1e-6


# ---------------------------------------------------------------------------
# 7. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result(self, baseline_params, simple_pool, simple_weekday_pools):
        """Running with the same seed produces identical fill_rate."""
        r1 = run_single_replication(0, baseline_params, simple_pool, simple_weekday_pools,
                                    sim_days=100, warmup_days=10, store_daily=False)
        r2 = run_single_replication(0, baseline_params, simple_pool, simple_weekday_pools,
                                    sim_days=100, warmup_days=10, store_daily=False)
        assert r1.fill_rate == r2.fill_rate
        assert r1.total_cost == r2.total_cost

    def test_different_seeds_different_results(self, baseline_params, simple_pool, simple_weekday_pools):
        """Different replication indices should (usually) produce different costs."""
        r0 = run_single_replication(0, baseline_params, simple_pool, simple_weekday_pools,
                                    sim_days=100, warmup_days=10, store_daily=False)
        r1 = run_single_replication(1, baseline_params, simple_pool, simple_weekday_pools,
                                    sim_days=100, warmup_days=10, store_daily=False)
        # Not guaranteed but highly probable over 90 post-warmup days
        assert r0.total_cost != r1.total_cost


# ---------------------------------------------------------------------------
# 8. Scenario configuration
# ---------------------------------------------------------------------------

class TestScenarioConfiguration:
    @pytest.mark.parametrize("scenario_key", list(SCENARIOS.keys()))
    def test_valid_policy_params(self, scenario_key):
        """Every scenario's config should build a valid PolicyParameters."""
        cfg = SCENARIOS[scenario_key]
        params = PolicyParameters(
            reorder_point=cfg["reorder_point"],
            order_quantity=cfg["order_quantity"],
            lt_min=cfg["lt_min"],
            lt_max=cfg["lt_max"],
            disruption_prob=cfg["disruption_prob"],
            disruption_duration_max=cfg["disruption_duration_max"],
            safety_stock_extra=cfg["safety_stock_extra"],
        )
        assert params.reorder_point >= 0
        assert params.order_quantity > 0
        assert params.lt_min <= params.lt_max

    def test_demand_surge_multiplier(self):
        """Demand surge scenario must have multiplier > 1."""
        assert SCENARIOS["S2_Demand_Surge"]["demand_multiplier"] > 1.0

    def test_supplier_delay_longer_lead_time(self):
        """Supplier delay scenario must have longer max lead time than baseline."""
        assert SCENARIOS["S3_Supplier_Delay"]["lt_max"] > SCENARIOS["S1_Baseline"]["lt_max"]

    def test_disruption_probability_valid(self):
        """Disruption probability must be in (0, 1]."""
        p = SCENARIOS["S4_Supplier_Disruption"]["disruption_prob"]
        assert 0 < p <= 1.0

    def test_safety_stock_higher_reorder_point(self):
        """Safety stock scenario must have higher reorder point than baseline."""
        assert (SCENARIOS["S5_Safety_Stock"]["reorder_point"]
                > SCENARIOS["S1_Baseline"]["reorder_point"])


# ---------------------------------------------------------------------------
# 9. Statistical summary calculations
# ---------------------------------------------------------------------------
class TestStatisticalSummaries:
    def test_ci_bounds_ordered(self):
        """Lower CI bound must be <= mean <= upper CI bound."""
        values = np.random.default_rng(10).normal(100, 15, 150)
        lo, hi = _t_ci(values)
        mean = float(values.mean())
        assert lo <= mean <= hi

    def test_descriptive_stats_range(self):
        """Mean must lie between min and max."""
        df = pd.DataFrame({"x": np.random.default_rng(11).exponential(50, 100)})
        s = descriptive_stats(df, "x")
        assert s["min"] <= s["mean"] <= s["max"]

    def test_cohens_d_zero_for_identical(self):
        """Cohen's d must be 0 when two arrays are identical."""
        a = np.ones(50) * 100
        d = cohens_d(a, a)
        assert abs(d) < 1e-9

    def test_fill_rate_aggregate_in_unit_interval(self, baseline_params, simple_pool, simple_weekday_pools):
        """Mean fill rate across replications must be in [0, 1]."""
        reps = run_monte_carlo(
            baseline_params, simple_pool, simple_weekday_pools,
            n_replications=20, sim_days=100, warmup_days=10, store_first_daily=False
        )
        df = results_to_dataframe(reps)
        assert 0.0 <= df["fill_rate"].mean() <= 1.0

    def test_policy_params_validation(self):
        """Invalid PolicyParameters should raise ValueError."""
        with pytest.raises(ValueError):
            PolicyParameters(reorder_point=-1, order_quantity=100, lt_min=5, lt_max=10)
        with pytest.raises(ValueError):
            PolicyParameters(reorder_point=100, order_quantity=0, lt_min=5, lt_max=10)
        with pytest.raises(ValueError):
            PolicyParameters(reorder_point=100, order_quantity=50, lt_min=10, lt_max=5)
