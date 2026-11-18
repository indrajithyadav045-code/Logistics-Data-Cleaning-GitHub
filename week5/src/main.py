"""
main.py
=======
Orchestrates the complete Week 5 logistics simulation pipeline.

Execution order
---------------
1.  Data preparation  → daily_demand DataFrame + statistics
2.  Demand analysis   → distribution fitting, weekday pools, model selection
3.  Scenario analysis → 6 Monte Carlo scenario runs + grid-search optimisation
4.  Statistical analysis → pairwise t-tests, effect sizes, confidence intervals
5.  Visualisation     → 10 publication-quality charts
6.  Results export    → CSV summaries + JSON metrics
7.  Report generation → professional DOCX report

Usage (from repository root)
-----------------------------
    python week5/src/main.py

Or from week5/src:
    python main.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week5" / "config"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from simulation_config import (
    DATA_RESULTS,
    PLOTS_DIR,
    REPORTS_DIR,
    SIM_HORIZON_DAYS,
    N_REPLICATIONS,
    WARMUP_DAYS,
    RANDOM_SEED_BASE,
    SCENARIOS,
)
from data_preparation import run_data_preparation
from demand_analysis import run_demand_analysis
from scenario_analysis import run_all_scenarios, compile_scenario_summary
from statistical_analysis import run_statistical_analysis
from visualization import generate_all_plots


def export_metrics_json(
    demand_stats: dict,
    summary_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
) -> Path:
    """Export key measured metrics to a JSON file for the report.

    Parameters
    ----------
    demand_stats  : dict from data_preparation.compute_demand_statistics()
    summary_df    : scenario summary DataFrame
    pairwise_df   : pairwise t-test results DataFrame

    Returns
    -------
    Path to the saved JSON file.
    """
    metrics = {
        "simulation_config": {
            "n_replications": N_REPLICATIONS,
            "sim_horizon_days": SIM_HORIZON_DAYS,
            "warmup_days": WARMUP_DAYS,
            "random_seed_base": RANDOM_SEED_BASE,
        },
        "dataset_statistics": demand_stats,
        "scenario_results": [],
        "pairwise_tests_total_cost": [],
    }
    for _, row in summary_df.iterrows():
        metrics["scenario_results"].append(row.to_dict())
    for _, row in pairwise_df[pairwise_df["metric"] == "total_cost"].iterrows():
        metrics["pairwise_tests_total_cost"].append(row.to_dict())

    out_path = DATA_RESULTS / "measured_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[main] Metrics exported to {out_path}")
    return out_path


def print_summary_table(summary_df: pd.DataFrame) -> None:
    """Print a formatted results table to stdout."""
    print("\n" + "=" * 90)
    print("SCENARIO RESULTS SUMMARY")
    print("=" * 90)
    fmt = "{:<30s} {:>8.4f} {:>14,.0f} {:>10.1f} {:>12.1f}"
    header = f"{'Scenario':<30s} {'Fill Rate':>8s} {'Total Cost R$':>14s} {'Stockout d':>10s} {'Avg Inv':>12s}"
    print(header)
    print("-" * 90)
    for _, row in summary_df.iterrows():
        print(fmt.format(
            row["scenario_label"],
            row["mean_fill_rate"],
            row["mean_total_cost"],
            row["mean_stockout_days"],
            row["mean_avg_inventory"],
        ))
    print("=" * 90)


def main() -> None:
    """Main pipeline entry point."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("  WEEK 5 – LOGISTICS SIMULATION & INVENTORY OPTIMISATION")
    print("=" * 70)

    # --- Stage 1: Data preparation ---
    daily, demand_stats = run_data_preparation()

    # --- Stage 2: Demand analysis ---
    demand_info = run_demand_analysis(daily)
    global_pool = demand_info["global_pool"]
    weekday_pools = demand_info["weekday_pools"]

    # --- Stage 3: Scenario analysis ---
    print("\n=== SCENARIO ANALYSIS ===")
    print(f"Running {len(SCENARIOS)} scenarios × {N_REPLICATIONS} replications each …")
    scenario_results = run_all_scenarios(global_pool, weekday_pools, n_reps=N_REPLICATIONS)

    # --- Stage 4: Compile summary ---
    summary_df = compile_scenario_summary(scenario_results)
    print_summary_table(summary_df)

    # Flatten per-scenario DataFrames for statistical analysis
    per_scenario_dfs = {key: df for key, (df, _) in scenario_results.items()}

    # --- Stage 5: Statistical analysis ---
    descriptive_df, pairwise_df = run_statistical_analysis(per_scenario_dfs)

    # --- Stage 6: Visualisation ---
    plot_paths = generate_all_plots(daily, scenario_results, summary_df)

    # --- Stage 7: Export metrics JSON ---
    json_path = export_metrics_json(demand_stats, summary_df, pairwise_df)

    # --- Stage 8: Report generation ---
    try:
        from generate_report import generate_report
        generate_report(demand_stats, summary_df, pairwise_df, plot_paths)
    except Exception as exc:
        print(f"\n[main] WARNING: Report generation failed: {exc}")
        print("[main] All other outputs are intact.")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE  –  elapsed: {elapsed:.1f}s")
    print(f"  Results: {DATA_RESULTS}")
    print(f"  Plots  : {PLOTS_DIR}")
    print(f"  Report : {REPORTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
