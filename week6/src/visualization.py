"""
visualization.py
================
Generates 10 publication-quality charts for the Week 6 Integrated Logistics Project.
"""
from __future__ import annotations

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week6" / "config"))
from project_config import PLOTS_DIR, DATA_PROCESSED, DATA_RESULTS

# Professional styling
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass
sns.set_palette("tab10")
DPI = 150


def _save_fig(fig: plt.Figure, filename: str) -> Path:
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out_path.name}")
    return out_path


def generate_all_visualizations(
    df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Generate and export all 10 diagnostic, predictive, and optimization figures."""
    print("\n--- [Generating Publication-Quality Visualizations] ---")
    if df is None:
        df = pd.read_csv(DATA_PROCESSED / "consolidated_logistics_orders.csv", low_memory=False)

    saved_plots = {}

    # -----------------------------------------------------------------------
    # 01. Delivery Duration Distribution (KDE & Histogram)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        df["delivery_days"], bins=50, kde=True, color="#1976D2", ax=ax, stat="density", alpha=0.6
    )
    mean_val = df["delivery_days"].mean()
    median_val = df["delivery_days"].median()
    p95_val = df["delivery_days"].quantile(0.95)

    ax.axvline(mean_val, color="#D32F2F", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.1f}d")
    ax.axvline(median_val, color="#388E3C", linestyle="-", linewidth=2, label=f"Median: {median_val:.1f}d")
    ax.axvline(p95_val, color="#F57C00", linestyle=":", linewidth=2, label=f"95th Pct: {p95_val:.1f}d")

    ax.set_title("Distribution of Customer Delivery Duration (Days)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Delivery Duration (Days from Purchase to Delivery)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0, 50)
    ax.legend(frameon=True, fontsize=10)
    saved_plots["01_delivery_duration_distribution"] = _save_fig(fig, "01_delivery_duration_distribution.png")

    # -----------------------------------------------------------------------
    # 02. Interstate vs Intrastate Performance
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    interstate_csv = pd.read_csv(DATA_RESULTS / "interstate_vs_intrastate.csv")

    # Transit days
    sns.barplot(data=interstate_csv, x="route_type", y="mean_delivery_days", palette=["#388E3C", "#D32F2F"], ax=axes[0])
    axes[0].set_title("Mean Delivery Days", fontweight="bold")
    axes[0].set_ylabel("Days")
    axes[0].set_xlabel("")
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}d", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=11, fontweight="bold", xytext=(0, 3), textcoords='offset points')

    # Delay rate %
    sns.barplot(data=interstate_csv, x="route_type", y="delay_rate_pct", palette=["#388E3C", "#D32F2F"], ax=axes[1])
    axes[1].set_title("Late Delivery Rate (%)", fontweight="bold")
    axes[1].set_ylabel("Delay Rate (%)")
    axes[1].set_xlabel("")
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=11, fontweight="bold", xytext=(0, 3), textcoords='offset points')

    # Mean freight
    sns.barplot(data=interstate_csv, x="route_type", y="mean_freight", palette=["#388E3C", "#D32F2F"], ax=axes[2])
    axes[2].set_title("Mean Freight Cost (R$)", fontweight="bold")
    axes[2].set_ylabel("Freight (BRL)")
    axes[2].set_xlabel("")
    for p in axes[2].patches:
        axes[2].annotate(f"R$ {p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=11, fontweight="bold", xytext=(0, 3), textcoords='offset points')

    plt.suptitle("Logistics Friction: Interstate vs Intrastate Fulfillment", fontsize=14, fontweight="bold")
    plt.tight_layout()
    saved_plots["02_interstate_vs_intrastate_performance"] = _save_fig(fig, "02_interstate_vs_intrastate_performance.png")

    # -----------------------------------------------------------------------
    # 03. Customer Review Score vs Delivery Delay
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    df["delay_status"] = df["is_delayed"].map({0: "On-Time Delivery", 1: "Late Delivery (SLA Breach)"})
    sns.boxplot(data=df, x="delay_status", y="review_score", palette=["#1976D2", "#D32F2F"], ax=ax, width=0.4, showmeans=True)
    ax.set_title("Customer Review Score Collapse Caused by Delivery Delays", fontsize=13, fontweight="bold")
    ax.set_ylabel("Customer Review Rating (1 to 5 Stars)", fontsize=11)
    ax.set_xlabel("")
    ontime_mean = df[df["is_delayed"] == 0]["review_score"].mean()
    late_mean = df[df["is_delayed"] == 1]["review_score"].mean()
    ax.text(0, ontime_mean + 0.25, f"Mean: {ontime_mean:.2f} stars", ha="center", fontweight="bold", color="#0D47A1", fontsize=11)
    ax.text(1, late_mean + 0.25, f"Mean: {late_mean:.2f} stars", ha="center", fontweight="bold", color="#B71C1C", fontsize=11)
    saved_plots["03_review_score_vs_delivery_delay"] = _save_fig(fig, "03_review_score_vs_delivery_delay.png")

    # -----------------------------------------------------------------------
    # 04. Regional Origin-Destination Heatmap
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    od_csv = pd.read_csv(DATA_RESULTS / "regional_od_matrix_delivery_days.csv", index_col="seller_region")
    sns.heatmap(od_csv, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Mean Delivery Days'}, ax=ax, linewidths=0.5)
    ax.set_title("Regional Origin-to-Destination (O-D) Delivery Duration (Days)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Customer Destination Region", fontsize=11)
    ax.set_ylabel("Seller Origin Region", fontsize=11)
    saved_plots["04_regional_od_heatmap"] = _save_fig(fig, "04_regional_od_heatmap.png")

    # -----------------------------------------------------------------------
    # 05. State Logistics Bottlenecks
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    state_csv = pd.read_csv(DATA_RESULTS / "state_logistics_performance.csv").head(15)
    sns.scatterplot(
        data=state_csv, x="mean_days", y="delay_pct", size="orders", sizes=(100, 1000),
        hue="customer_state", palette="turbo", ax=ax, legend=False
    )
    for _, row in state_csv.iterrows():
        ax.annotate(f"{row['customer_state']} ({row['delay_pct']:.1f}%)", (row['mean_days'] + 0.3, row['delay_pct']), fontsize=9, fontweight="bold")
    ax.set_title("Top 15 Customer Destination States: Delivery Latency vs Delay Rate", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Delivery Days", fontsize=11)
    ax.set_ylabel("Late Delivery Rate (%)", fontsize=11)
    saved_plots["05_state_logistics_bottlenecks"] = _save_fig(fig, "05_state_logistics_bottlenecks.png")

    # -----------------------------------------------------------------------
    # 06. Predictive Model Comparison (ROC & PR metrics)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    cls_csv = pd.read_csv(DATA_RESULTS / "classification_model_metrics.csv")
    x = np.arange(len(cls_csv))
    width = 0.25

    ax.bar(x - width, cls_csv["roc_auc"], width, label="ROC-AUC", color="#1976D2")
    ax.bar(x, cls_csv["pr_auc"], width, label="PR-AUC", color="#388E3C")
    ax.bar(x + width, cls_csv["recall"], width, label="Recall (Delayed Class)", color="#F57C00")

    ax.set_xticks(x)
    ax.set_xticklabels(cls_csv["model"], rotation=12, ha="right", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Predictive Classification Performance: SLA Delay Breach Models", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)
    saved_plots["06_predictive_roc_pr_curves"] = _save_fig(fig, "06_predictive_roc_pr_curves.png")

    # -----------------------------------------------------------------------
    # 07. Top Predictive Feature Importance
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    feat_csv = pd.read_csv(DATA_RESULTS / "feature_importance.csv").head(10).sort_values("importance", ascending=True)
    ax.barh(feat_csv["feature"], feat_csv["importance"], color="#00897B")
    ax.set_title("Top 10 Feature Importances for Delay Prediction (Random Forest)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Gini Feature Importance", fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.3f}", (p.get_width() + 0.005, p.get_y() + p.get_height() / 2.),
                    va='center', fontsize=9, fontweight="bold")
    saved_plots["07_feature_importance_ranking"] = _save_fig(fig, "07_feature_importance_ranking.png")

    # -----------------------------------------------------------------------
    # 08. Regression Performance (Actual vs Predicted Delivery Days)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    reg_csv = pd.read_csv(DATA_RESULTS / "regression_model_metrics.csv")
    bars = ax.bar(reg_csv["model"], reg_csv["mae_days"], color=["#5C6BC0", "#26A69A"], width=0.5)
    ax.set_title("Transit Duration Regression: Mean Absolute Error (Days)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (Days)", fontsize=11)
    for bar in bars:
        ax.annotate(f"{bar.get_height():.2f} days MAE", (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight="bold", xytext=(0, 4), textcoords='offset points')
    saved_plots["08_regression_actual_vs_predicted"] = _save_fig(fig, "08_regression_actual_vs_predicted.png")

    # -----------------------------------------------------------------------
    # 09. Network Optimization: Cost & Distance Comparison
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    opt_json_path = DATA_RESULTS / "network_optimization_metrics.json"
    import json
    with open(opt_json_path, "r") as f:
        opt_data = json.load(f)

    # Cost comparison
    costs = [opt_data["as_is_centralized"]["total_cost_brl"], opt_data["optimized_multi_hub"]["total_cost_brl"]]
    labels = ["As-Is Centralized\n(São Paulo Origin)", "Optimized Multi-Hub\n(8 Regional Hubs)"]
    bars1 = axes[0].bar(labels, costs, color=["#E53935", "#43A047"], width=0.5)
    axes[0].set_title("Annual Transportation & Handling Cost", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Cost (BRL)", fontsize=11)
    for b in bars1:
        axes[0].annotate(f"R$ {b.get_height():,.0f}", (b.get_x() + b.get_width()/2., b.get_height()),
                         ha='center', va='bottom', fontsize=11, fontweight="bold", xytext=(0, 4), textcoords='offset points')

    # Distance comparison
    distances = [opt_data["as_is_centralized"]["mean_distance_km"], opt_data["optimized_multi_hub"]["mean_distance_km"]]
    bars2 = axes[1].bar(labels, distances, color=["#E53935", "#43A047"], width=0.5)
    axes[1].set_title("Average Transit Distance per Order", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Kilometers (km)", fontsize=11)
    for b in bars2:
        axes[1].annotate(f"{b.get_height():.1f} km", (b.get_x() + b.get_width()/2., b.get_height()),
                         ha='center', va='bottom', fontsize=11, fontweight="bold", xytext=(0, 4), textcoords='offset points')

    plt.suptitle("Logistics Efficiency: As-Is Centralized vs Optimized Multi-Hub", fontsize=14, fontweight="bold")
    plt.tight_layout()
    saved_plots["09_optimization_cost_distance_comparison"] = _save_fig(fig, "09_optimization_cost_distance_comparison.png")

    # -----------------------------------------------------------------------
    # 10. Regional Hub Throughput & Carbon Footprint
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    hub_df = pd.read_csv(DATA_RESULTS / "hub_throughput_distribution.csv")
    sns.barplot(data=hub_df, x="hub", y="throughput_pct", palette="mako", ax=axes[0])
    axes[0].set_title("Optimized Throughput Share by Regional Hub (%)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Share of Total National Orders (%)", fontsize=11)
    axes[0].set_xlabel("Fulfillment Hub Location", fontsize=11)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2., p.get_height()),
                         ha='center', va='bottom', fontsize=10, fontweight="bold", xytext=(0, 3), textcoords='offset points')

    # Carbon Emissions
    co2 = [opt_data["as_is_centralized"]["co2_emissions_tons"], opt_data["optimized_multi_hub"]["co2_emissions_tons"]]
    bars3 = axes[1].bar(["As-Is Centralized", "Optimized Multi-Hub"], co2, color=["#8D6E63", "#2E7D32"], width=0.5)
    axes[1].set_title("Diesel Fleet CO2 Emissions (Tons)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Tons of CO2", fontsize=11)
    for b in bars3:
        axes[1].annotate(f"{b.get_height():.1f} Tons", (b.get_x() + b.get_width()/2., b.get_height()),
                         ha='center', va='bottom', fontsize=11, fontweight="bold", xytext=(0, 4), textcoords='offset points')

    plt.suptitle("Environmental Sustainability & Hub Allocation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    saved_plots["10_hub_volume_and_carbon_savings"] = _save_fig(fig, "10_hub_volume_and_carbon_savings.png")

    print(f"All {len(saved_plots)} visualizations generated successfully in {PLOTS_DIR}")
    return saved_plots


if __name__ == "__main__":
    generate_all_visualizations()
