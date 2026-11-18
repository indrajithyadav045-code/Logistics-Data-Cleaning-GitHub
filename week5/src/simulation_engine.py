"""
simulation_engine.py
====================
Discrete-time stochastic inventory simulation engine.

Simulation logic (one day, lost-sales model)
--------------------------------------------
1. Receive any replenishments arriving today (add to on-hand inventory).
2. Generate daily demand from the demand model.
3. Fulfil demand up to available on-hand inventory.
   - Units fulfilled = min(demand, on-hand).
   - Shortfall = demand - fulfilled  → lost forever (lost-sales model).
4. Update on-hand inventory: on-hand_new = on-hand - fulfilled.
5. Compute inventory position = on-hand_new + on-order.
6. Check disruption state.
7. If inventory_position <= reorder_point and no disruption: place order.
8. Record all state variables.

Lost-sales vs backorders
------------------------
Lost-sales is used because:
- Consumer e-commerce orders are typically fulfilled from a competitor
  if the retailer is out of stock (demand is not deferred).
- It avoids accumulating unbounded backlog, which is unrealistic for
  the demand magnitudes observed in Olist (~128 units/day average).
- Shortage cost is applied per lost unit on the day of stockout.

Inventory accounting convention
--------------------------------
Closing on-hand >= 0 always.  Inventory never goes negative.

Replication
-----------
Each replication is completely independent:
- Demand is re-sampled from scratch.
- Inventory state is reset to INITIAL_INVENTORY + safety_stock_extra.
- Random seed = RANDOM_SEED_BASE + replication_index.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week5" / "config"))
from simulation_config import (
    SIM_HORIZON_DAYS,
    N_REPLICATIONS,
    RANDOM_SEED_BASE,
    WARMUP_DAYS,
    INITIAL_INVENTORY,
    UNIT_HOLDING_COST_PER_DAY,
    UNIT_ORDERING_COST,
    UNIT_SHORTAGE_COST,
    DEMAND_MODEL,
)
from inventory_policy import InventoryPolicy, PolicyParameters, PendingOrder


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DayRecord:
    """Stores all state variables for a single simulation day."""
    day: int
    opening_inventory: int
    incoming_replenishment: int
    demand: float
    fulfilled: float
    lost_sales: float
    closing_inventory: int
    inventory_position: int
    order_placed: bool
    order_quantity: int
    on_order: int
    disruption_active: bool
    holding_cost: float
    ordering_cost: float
    shortage_cost: float
    total_cost: float


@dataclass
class ReplicationResult:
    """Aggregated metrics for a single Monte Carlo replication."""
    replication: int
    # Inventory metrics
    avg_inventory: float
    max_inventory: float
    ending_inventory: float
    total_replenishment_qty: int
    n_orders: int
    # Service metrics
    fill_rate: float
    demand_fulfillment_rate: float
    stockout_days: int
    total_unfulfilled: float
    stockout_frequency: float
    # Financial metrics
    total_holding_cost: float
    total_ordering_cost: float
    total_shortage_cost: float
    total_cost: float
    # Raw day-by-day records (optional, for time series plots)
    daily_records: Optional[List[DayRecord]] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Demand sampling
# ---------------------------------------------------------------------------

class DemandSampler:
    """Generates daily demand from either bootstrap, Poisson or NegBin model.

    Parameters
    ----------
    model        : 'bootstrap' | 'poisson' | 'negbinom'
    global_pool  : observed daily demand values (used for bootstrap)
    weekday_pools: dict mapping weekday int → demand pool (bootstrap only)
    rng          : numpy random Generator
    multiplier   : demand scaling factor (> 1 for demand surge scenarios)
    """

    def __init__(
        self,
        model: str,
        global_pool: np.ndarray,
        weekday_pools: dict,
        rng: np.random.Generator,
        multiplier: float = 1.0,
    ) -> None:
        valid_models = {"bootstrap", "poisson", "negbinom"}
        if model not in valid_models:
            raise ValueError(f"model must be one of {valid_models}; got '{model}'.")
        self._model = model
        self._global_pool = global_pool.astype(float)
        self._weekday_pools = {k: v.astype(float) for k, v in weekday_pools.items()}
        self._rng = rng
        self._multiplier = float(multiplier)
        # Pre-compute Poisson/NegBin parameters from pool
        self._poisson_lambda = float(global_pool.mean()) * self._multiplier
        mu = float(global_pool.mean())
        var = float(global_pool.var())
        self._nb_r = mu ** 2 / max(var - mu, 1e-6)
        self._nb_p = self._nb_r / (self._nb_r + mu)

    def sample(self, day_of_week: int) -> float:
        """Generate demand for a specific weekday.

        Parameters
        ----------
        day_of_week : int in [0, 6] (0 = Monday)

        Returns
        -------
        Non-negative demand value (may be 0).
        """
        if self._model == "bootstrap":
            pool = self._weekday_pools.get(day_of_week, self._global_pool)
            raw = float(self._rng.choice(pool))
            return max(0.0, raw * self._multiplier)
        elif self._model == "poisson":
            lam = self._poisson_lambda
            return float(self._rng.poisson(lam))
        elif self._model == "negbinom":
            # scipy's nbinom: n=r, p=p  → mean = r*(1-p)/p
            # To preserve mean with multiplier, adjust lambda via Poisson override
            return float(self._rng.negative_binomial(self._nb_r, self._nb_p))
        else:
            raise RuntimeError(f"Unknown model: {self._model}")


# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------

def run_single_replication(
    replication: int,
    policy_params: PolicyParameters,
    global_pool: np.ndarray,
    weekday_pools: dict,
    demand_multiplier: float = 1.0,
    sim_days: int = SIM_HORIZON_DAYS,
    warmup_days: int = WARMUP_DAYS,
    store_daily: bool = True,
) -> ReplicationResult:
    """Execute one complete inventory simulation replication.

    Parameters
    ----------
    replication      : replication index (0-based)
    policy_params    : PolicyParameters instance
    global_pool      : observed daily demand values for bootstrap
    weekday_pools    : weekday-conditioned demand pools
    demand_multiplier: scaling factor applied to sampled demand
    sim_days         : total simulation horizon in days
    warmup_days      : initial days excluded from metric calculations
    store_daily      : if True, include day-by-day DayRecord list in result

    Returns
    -------
    ReplicationResult
    """
    seed = RANDOM_SEED_BASE + replication
    rng = np.random.default_rng(seed)

    # Initialise policy and demand sampler
    policy = InventoryPolicy(params=policy_params, rng=rng)
    sampler = DemandSampler(
        model=DEMAND_MODEL,
        global_pool=global_pool,
        weekday_pools=weekday_pools,
        rng=rng,
        multiplier=demand_multiplier,
    )

    # State variables
    on_hand: int = INITIAL_INVENTORY + policy_params.safety_stock_extra
    pending_orders: List[PendingOrder] = []
    disruption_days_remaining: int = 0

    # Accumulators (post-warmup only)
    daily_records: List[DayRecord] = []
    total_demand: float = 0.0
    total_fulfilled: float = 0.0
    total_lost: float = 0.0
    stockout_days: int = 0
    inventory_levels: List[int] = []
    n_orders: int = 0
    total_replenishment_qty: int = 0
    total_holding: float = 0.0
    total_ordering: float = 0.0
    total_shortage: float = 0.0

    for day in range(sim_days):
        # --- Step 1: Receive arriving replenishments ---
        incoming = sum(o.quantity for o in pending_orders if o.arrival_day == day)
        on_hand = on_hand + incoming
        pending_orders = [o for o in pending_orders if o.arrival_day != day]

        # --- Step 2: Sample demand ---
        dow = day % 7   # day-of-week proxy (0=Mon)
        demand = sampler.sample(dow)

        # --- Step 3: Fulfil demand (lost-sales model) ---
        fulfilled = min(demand, float(on_hand))
        lost = demand - fulfilled
        on_hand_new = on_hand - int(np.floor(fulfilled))
        # Safety guard: on-hand cannot go negative due to float rounding
        on_hand = max(0, on_hand_new)

        # --- Step 4: Disruption state ---
        if disruption_days_remaining > 0:
            disruption_days_remaining -= 1
            disruption_active = True
        else:
            disruption_active = False
            if policy.disruption_occurs():
                dur = policy.sample_disruption_duration()
                disruption_days_remaining = dur - 1
                disruption_active = True

        # --- Step 5: Compute inventory position and reorder decision ---
        on_order = sum(o.quantity for o in pending_orders)
        inv_position = on_hand + on_order
        order_placed = False
        order_qty = 0

        if policy.should_reorder(inv_position, disruption_active):
            new_order = policy.create_order(current_day=day)
            pending_orders.append(new_order)
            order_placed = True
            order_qty = new_order.quantity

        # --- Step 6: Costs ---
        holding = float(on_hand) * UNIT_HOLDING_COST_PER_DAY
        ordering = UNIT_ORDERING_COST if order_placed else 0.0
        shortage = float(lost) * UNIT_SHORTAGE_COST
        day_total = holding + ordering + shortage

        # --- Step 7: Accumulate metrics (skip warm-up) ---
        if day >= warmup_days:
            total_demand += demand
            total_fulfilled += fulfilled
            total_lost += lost
            if lost > 0:
                stockout_days += 1
            inventory_levels.append(on_hand)
            if order_placed:
                n_orders += 1
                total_replenishment_qty += order_qty
            total_holding += holding
            total_ordering += ordering
            total_shortage += shortage

        # --- Step 8: Record ---
        # opening_inventory = on-hand BEFORE receiving replenishments and before fulfilment
        # Conservation identity: opening + incoming - fulfilled = closing  (± 1 rounding unit)
        opening_before = on_hand + int(np.floor(fulfilled)) - incoming
        opening_before = max(0, opening_before)
        rec = DayRecord(
            day=day,
            opening_inventory=opening_before,
            incoming_replenishment=incoming,
            demand=demand,
            fulfilled=fulfilled,
            lost_sales=lost,
            closing_inventory=on_hand,
            inventory_position=inv_position,
            order_placed=order_placed,
            order_quantity=order_qty,
            on_order=on_order,
            disruption_active=disruption_active,
            holding_cost=holding,
            ordering_cost=ordering,
            shortage_cost=shortage,
            total_cost=day_total,
        )
        if store_daily:
            daily_records.append(rec)

    # --- Compute aggregate metrics ---
    measured_days = sim_days - warmup_days
    avg_inventory = float(np.mean(inventory_levels)) if inventory_levels else 0.0
    max_inventory = float(np.max(inventory_levels)) if inventory_levels else 0.0
    ending_inventory = float(inventory_levels[-1]) if inventory_levels else 0.0
    fill_rate = total_fulfilled / total_demand if total_demand > 0 else 1.0
    demand_fulfillment_rate = fill_rate  # synonym; kept explicit
    stockout_frequency = stockout_days / measured_days if measured_days > 0 else 0.0

    return ReplicationResult(
        replication=replication,
        avg_inventory=avg_inventory,
        max_inventory=max_inventory,
        ending_inventory=ending_inventory,
        total_replenishment_qty=total_replenishment_qty,
        n_orders=n_orders,
        fill_rate=fill_rate,
        demand_fulfillment_rate=demand_fulfillment_rate,
        stockout_days=stockout_days,
        total_unfulfilled=total_lost,
        stockout_frequency=stockout_frequency,
        total_holding_cost=total_holding,
        total_ordering_cost=total_ordering,
        total_shortage_cost=total_shortage,
        total_cost=total_holding + total_ordering + total_shortage,
        daily_records=daily_records if store_daily else None,
    )


# ---------------------------------------------------------------------------
# Multi-replication runner
# ---------------------------------------------------------------------------

def run_monte_carlo(
    policy_params: PolicyParameters,
    global_pool: np.ndarray,
    weekday_pools: dict,
    n_replications: int = N_REPLICATIONS,
    demand_multiplier: float = 1.0,
    sim_days: int = SIM_HORIZON_DAYS,
    warmup_days: int = WARMUP_DAYS,
    store_first_daily: bool = True,
) -> List[ReplicationResult]:
    """Run n_replications independent Monte Carlo replications.

    Parameters
    ----------
    policy_params     : PolicyParameters for this scenario
    global_pool       : historical demand pool (numpy array)
    weekday_pools     : weekday-conditioned demand pools
    n_replications    : number of independent runs
    demand_multiplier : demand scaling factor
    sim_days          : simulation horizon (days)
    warmup_days       : initial days excluded from metrics
    store_first_daily : if True, store DayRecord list only for replication 0
                        (reduces memory footprint)

    Returns
    -------
    List[ReplicationResult] with length n_replications
    """
    results: List[ReplicationResult] = []
    for rep in range(n_replications):
        store = store_first_daily and (rep == 0)
        result = run_single_replication(
            replication=rep,
            policy_params=policy_params,
            global_pool=global_pool,
            weekday_pools=weekday_pools,
            demand_multiplier=demand_multiplier,
            sim_days=sim_days,
            warmup_days=warmup_days,
            store_daily=store,
        )
        results.append(result)
        if (rep + 1) % 50 == 0:
            print(f"  Completed {rep + 1}/{n_replications} replications …")
    return results


def results_to_dataframe(results: List[ReplicationResult]) -> pd.DataFrame:
    """Convert a list of ReplicationResult objects to a summary DataFrame.

    Parameters
    ----------
    results : list from run_monte_carlo()

    Returns
    -------
    pd.DataFrame with one row per replication.
    """
    rows = []
    for r in results:
        rows.append({
            "replication": r.replication,
            "avg_inventory": r.avg_inventory,
            "max_inventory": r.max_inventory,
            "ending_inventory": r.ending_inventory,
            "total_replenishment_qty": r.total_replenishment_qty,
            "n_orders": r.n_orders,
            "fill_rate": r.fill_rate,
            "demand_fulfillment_rate": r.demand_fulfillment_rate,
            "stockout_days": r.stockout_days,
            "total_unfulfilled": r.total_unfulfilled,
            "stockout_frequency": r.stockout_frequency,
            "total_holding_cost": r.total_holding_cost,
            "total_ordering_cost": r.total_ordering_cost,
            "total_shortage_cost": r.total_shortage_cost,
            "total_cost": r.total_cost,
        })
    return pd.DataFrame(rows)
