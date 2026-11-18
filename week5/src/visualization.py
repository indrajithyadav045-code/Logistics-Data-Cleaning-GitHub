"""
visualization.py
================
Generates all 10 required publication-quality charts for the Week 5 project.

Chart inventory
---------------
01  Historical daily demand distribution (histogram + KDE)
02  Inventory level over time (replication 0, baseline)
03  Demand vs fulfilled demand (replication 0, baseline)
04  Scenario-wise service level comparison (bar chart)
05  Scenario-wise total cost comparison (bar + CI)
06  Cost breakdown: holding / ordering / shortage (stacked bar)
07  Stockout frequency comparison (bar chart)
08  Confidence interval comparison across scenarios (line/errorbar)
09  Replenishment order activity (baseline replication 0)
10  Cost vs service-level trade-off (scatter)

Design principles
-----------------
- Seaborn whitegrid style for professional readability.
- Colour-blind-safe palette (tab10).
- Explicit axis labels with units.
- Titles are specific, not generic.
- Every chart is saved to week5/plots/ at 150 dpi.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week5" / "config"))
from simulation_config import (
    PLOTS_DIR,
    PLOT_STYLE,
    PLOT_FIGSIZE_WIDE,
    PLOT_FIGSIZE_SQUARE,
    PLOT_DPI,
    PLOT_PALETTE,
    SCENARIOS,
    WARMUP_DAYS,
)

try:
    plt.style.use(PLOT_STYLE)
except OSError:
    plt.style.use("seaborn-v0_8-whitegrid")

PALETTE = sns.color_palette(PLOT_PALETTE, n_colors=len(SCENARIOS))
SCENARIO_COLORS = {key: PALETTE[i] for i, key in enumerate(SCENARIOS.keys())}
SCENARIO_LABELS = {key: cfg["label"] for key, cfg in SCENARIOS.items()}


def _save(fig: plt.Figure, filename: str) -> Path:
    """Save a figure to the plots directory."""
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# Chart 01 – Historical daily demand distribution
# ---------------------------------------------------------------------------
def plot_demand_distribution(daily: pd.DataFrame) -> Path:
    """Histogram + KDE of observed daily demand.

    Why: Reveals the shape, skewness and presence of zero-demand days.
    Question: What does baseline demand look like?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    d = daily["demand"]
    ax = axes[0]
    ax.hist(d, bins=60, color="#2196F3", edgecolor="white", alpha=0.85, density=True)
    d.plot.kde(ax=ax, color="#E53935", linewidth=2.5, label="KDE")
    ax.axvline(d.mean(), color="#FF9800", linewidth=2, linestyle="--", label=f"Mean = {d.mean():.1f}")
    ax.axvline(d.median(), color="#4CAF50", linewidth=2, linestyle=":", label=f"Median = {d.median():.1f}")
    ax.set_xlabel("Daily Order Count (units)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Historical Daily Demand Distribution\n(Olist, 2016–2018)", fontsize=13)
    ax.legend()

    # Second panel: weekday mean demand
    ax2 = axes[1]
    wd_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wd_means = daily.groupby("weekday")["demand"].mean().reindex(wd_order)
    colors = sns.color_palette("Blues_d", n_colors=7)
    bars = ax2.bar(range(7), wd_means.values, color=colors, edgecolor="white")
    ax2.set_xticks(range(7))
    ax2.set_xticklabels([w[:3] for w in wd_order], fontsize=11)
    ax2.set_xlabel("Day of Week", fontsize=12)
    ax2.set_ylabel("Mean Daily Orders", fontsize=12)
    ax2.set_title("Mean Daily Demand by Weekday", fontsize=13)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return _save(fig, "01_demand_distribution.png")


# ---------------------------------------------------------------------------
# Chart 02 – Inventory level over time (baseline replication 0)
# ---------------------------------------------------------------------------
def plot_inventory_over_time(daily_records: list, scenario_label: str = "Baseline") -> Path:
    """Line chart of daily closing inventory for replication 0.

    Why: Shows the sawtooth inventory pattern and identifies stockout periods.
    Question: How does inventory deplete and replenish over time?
    """
    df = pd.DataFrame([
        {"day": r.day, "closing_inventory": r.closing_inventory,
         "reorder_point": None, "order_placed": r.order_placed}
        for r in daily_records if r.day >= WARMUP_DAYS
    ])

    # Read reorder point from config
    rop = SCENARIOS["S1_Baseline"]["reorder_point"]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_WIDE)
    ax.fill_between(df["day"], df["closing_inventory"], alpha=0.25, color="#2196F3")
    ax.plot(df["day"], df["closing_inventory"], color="#1565C0", linewidth=1.2,
            label="Closing Inventory (units)")
    ax.axhline(rop, color="#FF9800", linewidth=1.8, linestyle="--", label=f"Reorder Point = {rop}")
    ax.axhline(0, color="#E53935", linewidth=1.2, linestyle=":", label="Zero Inventory")

    # Mark order days
    order_days = df[df["order_placed"]]["day"]
    ax.scatter(order_days, [rop] * len(order_days), marker="v", color="#FF5722",
               s=40, zorder=5, label="Order Placed")

    ax.set_xlabel("Simulation Day", fontsize=12)
    ax.set_ylabel("On-Hand Inventory (units)", fontsize=12)
    ax.set_title(f"Inventory Level Over Time – {scenario_label}\n(Replication 0, post-warmup)", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return _save(fig, "02_inventory_over_time.png")


# ---------------------------------------------------------------------------
# Chart 03 – Demand vs fulfilled demand (baseline rep 0)
# ---------------------------------------------------------------------------
def plot_demand_vs_fulfilled(daily_records: list) -> Path:
    """Line chart overlaying raw demand and fulfilled demand.

    Why: Visually reveals shortfall periods where demand > inventory.
    Question: When and how severely does the system fail to meet demand?
    """
    df = pd.DataFrame([
        {"day": r.day, "demand": r.demand, "fulfilled": r.fulfilled, "lost": r.lost_sales}
        for r in daily_records if r.day >= WARMUP_DAYS
    ])

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_WIDE)
    ax.plot(df["day"], df["demand"], color="#1565C0", linewidth=1.2, alpha=0.8, label="Demand")
    ax.plot(df["day"], df["fulfilled"], color="#2E7D32", linewidth=1.2, alpha=0.8, label="Fulfilled")
    ax.fill_between(df["day"], df["fulfilled"], df["demand"],
                    where=df["demand"] > df["fulfilled"],
                    color="#E53935", alpha=0.35, label="Unfulfilled (Lost Sales)")
    ax.set_xlabel("Simulation Day", fontsize=12)
    ax.set_ylabel("Units", fontsize=12)
    ax.set_title("Daily Demand vs Fulfilled Demand – Baseline (Replication 0)", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return _save(fig, "03_demand_vs_fulfilled.png")


# ---------------------------------------------------------------------------
# Chart 04 – Scenario-wise service level comparison
# ---------------------------------------------------------------------------
def plot_service_level_comparison(summary_df: pd.DataFrame) -> Path:
    """Horizontal bar chart of mean fill rate per scenario.

    Why: Provides an at-a-glance service-level ranking.
    Question: Which scenario best serves customers?
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [SCENARIO_COLORS[k] for k in summary_df["scenario_key"]]
    bars = ax.barh(summary_df["scenario_label"], summary_df["mean_fill_rate"] * 100,
                   color=colors, edgecolor="white", height=0.6)
    ax.axvline(95, color="#E53935", linewidth=2, linestyle="--", label="95% target")
    ax.set_xlabel("Mean Fill Rate (%)", fontsize=12)
    ax.set_title("Service Level Comparison Across Scenarios", fontsize=13)
    ax.set_xlim(0, 105)
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.2f}%", va="center", fontsize=10)
    ax.legend()
    plt.tight_layout()
    return _save(fig, "04_service_level_comparison.png")


# ---------------------------------------------------------------------------
# Chart 05 – Scenario-wise total cost comparison
# ---------------------------------------------------------------------------
def plot_total_cost_comparison(summary_df: pd.DataFrame) -> Path:
    """Bar chart of mean total cost with 95% CI error bars.

    Why: Shows the financial impact of each scenario.
    Question: Which scenario is most/least expensive to operate?
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [SCENARIO_COLORS[k] for k in summary_df["scenario_key"]]
    x = range(len(summary_df))
    bars = ax.bar(x, summary_df["mean_total_cost"], color=colors, edgecolor="white", width=0.6)
    # CI error bars
    lo_errs = summary_df["mean_total_cost"] - summary_df["cost_ci_lo"]
    hi_errs = summary_df["cost_ci_hi"] - summary_df["mean_total_cost"]
    ax.errorbar(x, summary_df["mean_total_cost"],
                yerr=[lo_errs.values, hi_errs.values],
                fmt="none", ecolor="black", capsize=5, linewidth=1.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary_df["scenario_label"], rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Mean Total Operating Cost (R$)", fontsize=12)
    ax.set_title("Total Operating Cost Comparison Across Scenarios\n(Mean ± 95% CI)", fontsize=13)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"R${bar.get_height():,.0f}", ha="center", va="bottom", fontsize=8.5)
    plt.tight_layout()
    return _save(fig, "05_total_cost_comparison.png")


# ---------------------------------------------------------------------------
# Chart 06 – Cost breakdown by component
# ---------------------------------------------------------------------------
def plot_cost_breakdown(summary_df: pd.DataFrame) -> Path:
    """Stacked bar chart of holding, ordering and shortage costs.

    Why: Identifies the dominant cost driver in each scenario.
    Question: Where does the money go – storage, procurement or stockouts?
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(summary_df))
    labels = summary_df["scenario_label"].tolist()
    h = summary_df["mean_holding_cost"].values
    o = summary_df["mean_ordering_cost"].values
    s = summary_df["mean_shortage_cost"].values

    ax.bar(x, h, label="Holding Cost", color="#1565C0", edgecolor="white")
    ax.bar(x, o, bottom=h, label="Ordering Cost", color="#43A047", edgecolor="white")
    ax.bar(x, s, bottom=h + o, label="Shortage Cost", color="#E53935", edgecolor="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Mean Cost (R$)", fontsize=12)
    ax.set_title("Cost Component Breakdown by Scenario", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return _save(fig, "06_cost_breakdown.png")


# ---------------------------------------------------------------------------
# Chart 07 – Stockout frequency comparison
# ---------------------------------------------------------------------------
def plot_stockout_frequency(summary_df: pd.DataFrame) -> Path:
    """Bar chart of mean fraction of days with at least one lost sale.

    Why: Directly quantifies unreliability.
    Question: How often does the warehouse run out of stock?
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [SCENARIO_COLORS[k] for k in summary_df["scenario_key"]]
    bars = ax.bar(
        summary_df["scenario_label"],
        summary_df["mean_stockout_freq"] * 100,
        color=colors, edgecolor="white", width=0.6
    )
    ax.set_ylabel("Stockout Frequency (% of days)", fontsize=12)
    ax.set_title("Stockout Frequency Comparison Across Scenarios", fontsize=13)
    ax.set_xticks(range(len(summary_df)))
    ax.set_xticklabels(summary_df["scenario_label"].tolist(), rotation=15, ha="right")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    return _save(fig, "07_stockout_frequency.png")


# ---------------------------------------------------------------------------
# Chart 08 – Confidence interval comparison
# ---------------------------------------------------------------------------
def plot_confidence_intervals(summary_df: pd.DataFrame) -> Path:
    """Errorbar chart showing mean total cost ± 95 % CI per scenario.

    Why: Quantifies uncertainty around each scenario's cost estimate.
    Question: How stable and certain are the cost estimates?
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(summary_df))
    colors = [SCENARIO_COLORS[k] for k in summary_df["scenario_key"]]
    for i, row in summary_df.iterrows():
        idx = list(summary_df.index).index(i)
        err_lo = row["mean_total_cost"] - row["cost_ci_lo"]
        err_hi = row["cost_ci_hi"] - row["mean_total_cost"]
        ax.errorbar(idx, row["mean_total_cost"],
                    yerr=[[err_lo], [err_hi]],
                    fmt="o", color=colors[idx], capsize=8,
                    markersize=10, linewidth=2.5,
                    label=row["scenario_label"])

    ax.set_xticks(list(x))
    ax.set_xticklabels(summary_df["scenario_label"], rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Mean Total Cost (R$) ± 95% CI", fontsize=12)
    ax.set_title("95% Confidence Intervals for Total Operating Cost", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _save(fig, "08_confidence_intervals.png")


# ---------------------------------------------------------------------------
# Chart 09 – Replenishment order activity
# ---------------------------------------------------------------------------
def plot_replenishment_activity(daily_records: list) -> Path:
    """Stem plot of replenishment order arrival events (baseline rep 0).

    Why: Shows the spacing between orders and order timing relative to demand.
    Question: How frequently and regularly does the warehouse reorder?
    """
    df = pd.DataFrame([
        {"day": r.day, "order_placed": r.order_placed,
         "order_quantity": r.order_quantity, "incoming": r.incoming_replenishment}
        for r in daily_records if r.day >= WARMUP_DAYS
    ])
    orders = df[df["order_placed"]]
    arrivals = df[df["incoming"] > 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Order placements
    ax1.stem(orders["day"], orders["order_quantity"],
             linefmt="C0-", markerfmt="C0o", basefmt=" ")
    ax1.set_ylabel("Order Quantity (units)", fontsize=11)
    ax1.set_title("Replenishment Order Placements – Baseline (Replication 0)", fontsize=12)

    # Arrivals
    ax2.bar(arrivals["day"], arrivals["incoming"], color="#43A047", width=1.5, alpha=0.8)
    ax2.set_xlabel("Simulation Day", fontsize=12)
    ax2.set_ylabel("Units Received", fontsize=11)
    ax2.set_title("Replenishment Arrivals", fontsize=12)

    plt.tight_layout()
    return _save(fig, "09_replenishment_activity.png")


# ---------------------------------------------------------------------------
# Chart 10 – Cost vs service level trade-off
# ---------------------------------------------------------------------------
def plot_cost_service_tradeoff(summary_df: pd.DataFrame) -> Path:
    """Scatter plot of mean total cost vs mean fill rate per scenario.

    Why: Reveals the efficiency frontier – scenarios in the bottom-right
    corner provide high service at low cost.
    Question: Which policy offers the best cost-service balance?
    """
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SQUARE)
    colors = [SCENARIO_COLORS[k] for k in summary_df["scenario_key"]]
    ax.scatter(
        summary_df["mean_fill_rate"] * 100,
        summary_df["mean_total_cost"],
        c=colors, s=200, zorder=5, edgecolors="black", linewidth=0.8
    )
    for _, row in summary_df.iterrows():
        ax.annotate(
            row["scenario_label"],
            xy=(row["mean_fill_rate"] * 100, row["mean_total_cost"]),
            xytext=(6, 6), textcoords="offset points", fontsize=9
        )
    ax.axvline(95, color="#E53935", linewidth=1.5, linestyle="--", alpha=0.6, label="95% fill rate target")
    ax.set_xlabel("Mean Fill Rate (%)", fontsize=12)
    ax.set_ylabel("Mean Total Operating Cost (R$)", fontsize=12)
    ax.set_title("Cost vs Service Level Trade-off\nAcross Replenishment Scenarios", fontsize=13)
    ax.legend()
    plt.tight_layout()
    return _save(fig, "10_cost_service_tradeoff.png")


# ---------------------------------------------------------------------------
# Main plot runner
# ---------------------------------------------------------------------------

def generate_all_plots(
    daily: pd.DataFrame,
    scenario_results: dict,
    summary_df: pd.DataFrame,
    baseline_key: str = "S1_Baseline",
) -> Dict[str, Path]:
    """Generate all 10 required charts.

    Parameters
    ----------
    daily            : historical daily demand DataFrame
    scenario_results : dict from scenario_analysis.run_all_scenarios()
    summary_df       : compiled scenario summary table
    baseline_key     : key for the baseline scenario

    Returns
    -------
    dict mapping chart name → file path
    """
    print("\n=== GENERATING VISUALIZATIONS ===")
    paths: Dict[str, Path] = {}

    # Baseline daily records (replication 0)
    baseline_raws = scenario_results[baseline_key][1]
    baseline_records = baseline_raws[0].daily_records or []

    paths["01_demand_distribution"] = plot_demand_distribution(daily)
    if baseline_records:
        paths["02_inventory_over_time"] = plot_inventory_over_time(baseline_records)
        paths["03_demand_vs_fulfilled"] = plot_demand_vs_fulfilled(baseline_records)
        paths["09_replenishment_activity"] = plot_replenishment_activity(baseline_records)
    paths["04_service_level"] = plot_service_level_comparison(summary_df)
    paths["05_total_cost"] = plot_total_cost_comparison(summary_df)
    paths["06_cost_breakdown"] = plot_cost_breakdown(summary_df)
    paths["07_stockout_frequency"] = plot_stockout_frequency(summary_df)
    paths["08_confidence_intervals"] = plot_confidence_intervals(summary_df)
    paths["10_cost_service_tradeoff"] = plot_cost_service_tradeoff(summary_df)

    print(f"\n[visualization] All {len(paths)} charts saved to {PLOTS_DIR}")
    return paths
