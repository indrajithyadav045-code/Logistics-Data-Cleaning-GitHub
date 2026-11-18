"""
scenario_analysis.py
====================
Runs all six scenarios and the grid-search optimisation for Scenario 6.

Scenarios
---------
S1  Baseline           – normal demand, normal lead times
S2  Demand Surge       – 40 % demand increase
S3  Supplier Delay     – extended lead times (10–18 days)
S4  Supplier Disruption – random interruptions (5 % daily probability)
S5  Safety Stock       – elevated reorder point + 600-unit buffer
S6  Optimized          – grid-search best (s, Q) over cost/service trade-off

Grid search (Scenario 6)
------------------------
Searches reorder points in OPT_ROP_RANGE and quantities in OPT_QTY_RANGE.
For each (s, Q) combination runs N_OPT_REPS replications and computes
mean total cost.  Only combinations achieving fill_rate >= OPT_MIN_FILL_RATE
are considered feasible.  The (s, Q) with minimum mean total cost among
feasible combinations is selected.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week5" / "config"))
from simulation_config import (
    SCENARIOS,
    N_REPLICATIONS,
    OPT_ROP_RANGE,
    OPT_QTY_RANGE,
    OPT_MIN_FILL_RATE,
    DATA_RESULTS,
)
from inventory_policy import PolicyParameters
from simulation_engine import (
    run_monte_carlo,
    results_to_dataframe,
    ReplicationResult,
)

# Fast replication count for grid search (fewer reps for speed)
N_OPT_REPS: int = 10


def build_policy_params(scenario_key: str) -> Tuple[PolicyParameters, float]:
    """Build PolicyParameters and demand_multiplier from a scenario config dict.

    Parameters
    ----------
    scenario_key : key in SCENARIOS dict (e.g. 'S1_Baseline')

    Returns
    -------
    (PolicyParameters, demand_multiplier)
    """
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
    multiplier = cfg["demand_multiplier"]
    return params, multiplier


def run_scenario(
    scenario_key: str,
    global_pool: np.ndarray,
    weekday_pools: dict,
    n_reps: int = N_REPLICATIONS,
) -> Tuple[pd.DataFrame, List[ReplicationResult]]:
    """Run a single named scenario and return summary DataFrame + raw results.

    Parameters
    ----------
    scenario_key  : key in SCENARIOS dict
    global_pool   : historical demand pool
    weekday_pools : weekday-conditioned demand pools
    n_reps        : number of Monte Carlo replications

    Returns
    -------
    (summary_df, raw_results)
    """
    label = SCENARIOS[scenario_key]["label"]
    print(f"\n--- Running: {label} ({n_reps} replications) ---")
    params, multiplier = build_policy_params(scenario_key)
    raw_results = run_monte_carlo(
        policy_params=params,
        global_pool=global_pool,
        weekday_pools=weekday_pools,
        n_replications=n_reps,
        demand_multiplier=multiplier,
        store_first_daily=True,
    )
    df = results_to_dataframe(raw_results)
    df["scenario"] = scenario_key
    df["scenario_label"] = label
    return df, raw_results


def grid_search_optimisation(
    global_pool: np.ndarray,
    weekday_pools: dict,
) -> Tuple[int, int, pd.DataFrame]:
    """Grid-search optimal (reorder_point, order_quantity) for Scenario 6.

    Objective : minimise mean total cost
    Constraint: mean fill_rate >= OPT_MIN_FILL_RATE

    Parameters
    ----------
    global_pool   : historical demand pool
    weekday_pools : weekday-conditioned pools

    Returns
    -------
    (best_rop, best_qty, grid_results_df)
    """
    print("\n--- Grid search for Scenario 6 (Optimized Replenishment) ---")
    records = []
    n_combos = len(OPT_ROP_RANGE) * len(OPT_QTY_RANGE)
    done = 0
    for rop in OPT_ROP_RANGE:
        for qty in OPT_QTY_RANGE:
            params = PolicyParameters(
                reorder_point=rop,
                order_quantity=qty,
                lt_min=5,
                lt_max=9,
            )
            reps = run_monte_carlo(
                policy_params=params,
                global_pool=global_pool,
                weekday_pools=weekday_pools,
                n_replications=N_OPT_REPS,
                store_first_daily=False,
            )
            df = results_to_dataframe(reps)
            records.append({
                "reorder_point": rop,
                "order_quantity": qty,
                "mean_fill_rate": df["fill_rate"].mean(),
                "mean_total_cost": df["total_cost"].mean(),
                "mean_stockout_days": df["stockout_days"].mean(),
                "feasible": df["fill_rate"].mean() >= OPT_MIN_FILL_RATE,
            })
            done += 1
            if done % 50 == 0:
                print(f"  Grid search: {done}/{n_combos} combinations done …")

    grid_df = pd.DataFrame(records)
    feasible = grid_df[grid_df["feasible"]]
    if feasible.empty:
        print("  WARNING: No feasible combination found; relaxing constraint.")
        feasible = grid_df

    best_row = feasible.loc[feasible["mean_total_cost"].idxmin()]
    best_rop = int(best_row["reorder_point"])
    best_qty = int(best_row["order_quantity"])
    print(f"  Best (s={best_rop}, Q={best_qty}): "
          f"cost={best_row['mean_total_cost']:.0f}, "
          f"fill={best_row['mean_fill_rate']:.4f}")

    # Update S6 config with discovered optimum
    SCENARIOS["S6_Optimized"]["reorder_point"] = best_rop
    SCENARIOS["S6_Optimized"]["order_quantity"] = best_qty

    return best_rop, best_qty, grid_df


def run_all_scenarios(
    global_pool: np.ndarray,
    weekday_pools: dict,
    n_reps: int = N_REPLICATIONS,
) -> Dict[str, Tuple[pd.DataFrame, List[ReplicationResult]]]:
    """Run all six scenarios and return a dict of results.

    Scenario 6 runs grid search first to identify optimal parameters,
    then runs the full replication set.

    Parameters
    ----------
    global_pool   : historical demand pool
    weekday_pools : weekday-conditioned demand pools
    n_reps        : replications per scenario (except grid-search)

    Returns
    -------
    dict mapping scenario_key → (summary_df, raw_results_list)
    """
    # Grid search for S6 first
    best_rop, best_qty, grid_df = grid_search_optimisation(global_pool, weekday_pools)
    grid_path = DATA_RESULTS / "grid_search_results.csv"
    grid_df.to_csv(grid_path, index=False)
    print(f"  Grid search results saved to {grid_path}")

    scenario_results: Dict[str, Tuple[pd.DataFrame, List[ReplicationResult]]] = {}
    for key in SCENARIOS:
        df, raws = run_scenario(key, global_pool, weekday_pools, n_reps=n_reps)
        scenario_results[key] = (df, raws)

        # Save per-scenario raw results
        out_path = DATA_RESULTS / f"{key}_replications.csv"
        df.to_csv(out_path, index=False)

    return scenario_results


def compile_scenario_summary(
    scenario_results: Dict[str, Tuple[pd.DataFrame, List[ReplicationResult]]],
) -> pd.DataFrame:
    """Compile a cross-scenario summary table of key metrics.

    Parameters
    ----------
    scenario_results : output of run_all_scenarios()

    Returns
    -------
    pd.DataFrame with one row per scenario.
    """
    rows = []
    for key, (df, _) in scenario_results.items():
        label = SCENARIOS[key]["label"]
        ci_lo, ci_hi = _confidence_interval(df["total_cost"].values)
        fill_ci_lo, fill_ci_hi = _confidence_interval(df["fill_rate"].values)
        rows.append({
            "scenario_key": key,
            "scenario_label": label,
            "mean_avg_inventory": df["avg_inventory"].mean(),
            "mean_fill_rate": df["fill_rate"].mean(),
            "fill_rate_ci_lo": fill_ci_lo,
            "fill_rate_ci_hi": fill_ci_hi,
            "mean_stockout_days": df["stockout_days"].mean(),
            "mean_stockout_freq": df["stockout_frequency"].mean(),
            "mean_total_unfulfilled": df["total_unfulfilled"].mean(),
            "mean_holding_cost": df["total_holding_cost"].mean(),
            "mean_ordering_cost": df["total_ordering_cost"].mean(),
            "mean_shortage_cost": df["total_shortage_cost"].mean(),
            "mean_total_cost": df["total_cost"].mean(),
            "std_total_cost": df["total_cost"].std(),
            "cost_ci_lo": ci_lo,
            "cost_ci_hi": ci_hi,
            "mean_n_orders": df["n_orders"].mean(),
        })
    summary_df = pd.DataFrame(rows)
    summary_path = DATA_RESULTS / "scenario_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[scenario_analysis] Scenario summary saved to {summary_path}")
    return summary_df


def _confidence_interval(
    values: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute two-sided t-based confidence interval for the mean.

    Parameters
    ----------
    values     : 1-D array of observations
    confidence : confidence level (default 0.95)

    Returns
    -------
    (lower, upper) confidence interval bounds
    """
    from scipy import stats as sps
    n = len(values)
    if n < 2:
        return (float(values[0]), float(values[0])) if n == 1 else (0.0, 0.0)
    se = sps.sem(values)
    t_crit = sps.t.ppf((1 + confidence) / 2, df=n - 1)
    mean = float(values.mean())
    margin = float(t_crit * se)
    return (mean - margin, mean + margin)
