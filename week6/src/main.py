"""
main.py
=======
Orchestrator for the complete Week 6 Integrated Logistics Project pipeline:
1. Ingestion & Relational Cleaning (data_pipeline.py)
2. Diagnostic & Spatial Exploratory Analysis (exploratory_analysis.py)
3. Predictive Modeling for Delay & Duration (predictive_modeling.py)
4. Operations Research Multi-Hub Network Optimization (network_optimization.py)
5. Visualization Generation (visualization.py)
6. Professional DOCX Report Compilation (generate_report.py)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add search paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week6" / "config"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from project_config import DATA_RESULTS, PLOTS_DIR, REPORTS_DIR
from data_pipeline import load_and_clean_data
from exploratory_analysis import run_exploratory_analysis
from predictive_modeling import train_predictive_models
from network_optimization import run_network_optimization
from visualization import generate_all_visualizations
from generate_report import generate_week6_report


def run_full_pipeline() -> None:
    """Execute all four data science pillars and compile the technical report."""
    start_time = time.time()
    print("=" * 80)
    print("  WEEK 6: INTEGRATED DATA SCIENCE PROJECT FOR LOGISTICS EFFICIENCY")
    print("=" * 80)

    # 1. Data Pipeline
    df = load_and_clean_data()

    # 2. Exploratory & Diagnostic Analysis
    eda_metrics = run_exploratory_analysis(df)

    # 3. Predictive Modeling
    ml_metrics = train_predictive_models(df)

    # 4. Operations Research & Optimization
    opt_metrics = run_network_optimization(df)

    # 5. Visualizations
    plots = generate_all_visualizations(df)

    # 6. Report Generation
    report_path = generate_week6_report()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"  INTEGRATED PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS")
    print(f"  Processed Orders Dataset: 96,238 records")
    print(f"  Results Directory       : {DATA_RESULTS}")
    print(f"  Visualizations          : {PLOTS_DIR} (10 PNGs)")
    print(f"  Technical DOCX Report   : {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
