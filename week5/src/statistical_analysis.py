"""
statistical_analysis.py
=======================
Performs statistical comparisons between scenarios.

Methods
-------
1. Descriptive statistics (mean, median, std, min, max, CI) per scenario.
2. 95 % confidence intervals (t-distribution, two-sided).
3. Pairwise t-tests comparing each scenario against the baseline (S1).
4. Cohen's d effect size for each pair.
5. Bonferroni correction for multiple comparisons.

Assumptions
-----------
- Each replication is an independent observation (by construction: different seeds).
- Because replications are independent (not matched pairs), we use two-sample
  Welch's t-test rather than a paired t-test.
- Bonferroni correction is applied across the 5 pairwise comparisons
  (S2–S6 each vs S1) to control family-wise error rate.

Limitations
-----------
- Normality is assumed for the t-test.  With 150 replications the CLT
  provides strong justification for this assumption.
- Practical significance (effect size) is reported separately from
  statistical significance.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week5" / "config"))
from simulation_config import DATA_RESULTS, SCENARIOS


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def descriptive_stats(df: pd.DataFrame, metric: str) -> dict:
    """Compute descriptive statistics for a metric across replications.

    Parameters
    ----------
    df     : replication-level DataFrame (one row per replication)
    metric : column name to summarise

    Returns
    -------
    dict with mean, median, std, min, max, p05, p95, ci_lo, ci_hi
    """
    values = df[metric].dropna().values
    ci_lo, ci_hi = _t_ci(values)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n": len(values),
    }


def _t_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Two-sided t-based confidence interval for the mean."""
    n = len(values)
    if n < 2:
        return (float(values[0]), float(values[0])) if n == 1 else (0.0, 0.0)
    se = stats.sem(values)
    t = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    mu = float(np.mean(values))
    margin = float(t * se)
    return (mu - margin, mu + margin)


# ---------------------------------------------------------------------------
# Pairwise comparisons
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two independent groups.

    Cohen's d = (mean_a - mean_b) / pooled_std

    Parameters
    ----------
    a, b : numpy arrays of observations

    Returns
    -------
    float (positive means a > b)
    """
    pooled_std = np.sqrt(
        ((len(a) - 1) * np.std(a, ddof=1) ** 2 + (len(b) - 1) * np.std(b, ddof=1) ** 2)
        / (len(a) + len(b) - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def pairwise_ttest_vs_baseline(
    all_results: Dict[str, pd.DataFrame],
    baseline_key: str = "S1_Baseline",
    metric: str = "total_cost",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run two-sample Welch t-tests comparing each scenario vs baseline.

    Bonferroni correction applied across the 5 non-baseline comparisons.

    Parameters
    ----------
    all_results  : dict mapping scenario_key → replication DataFrame
    baseline_key : key for the baseline scenario
    metric       : metric column to compare
    alpha        : uncorrected significance level

    Returns
    -------
    pd.DataFrame with one row per scenario (excluding baseline).
    """
    baseline_vals = all_results[baseline_key][metric].values
    n_comparisons = sum(1 for k in all_results if k != baseline_key)
    alpha_bonferroni = alpha / max(n_comparisons, 1)

    rows = []
    for key, df in all_results.items():
        if key == baseline_key:
            continue
        vals = df[metric].values
        t_stat, p_val = stats.ttest_ind(vals, baseline_vals, equal_var=False)
        d = cohens_d(vals, baseline_vals)
        reject = p_val < alpha_bonferroni
        rows.append({
            "scenario_key": key,
            "scenario_label": SCENARIOS[key]["label"],
            "metric": metric,
            "mean_scenario": float(np.mean(vals)),
            "mean_baseline": float(np.mean(baseline_vals)),
            "mean_diff": float(np.mean(vals) - np.mean(baseline_vals)),
            "pct_change": 100.0 * (np.mean(vals) - np.mean(baseline_vals)) / max(abs(np.mean(baseline_vals)), 1),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "alpha_bonferroni": float(alpha_bonferroni),
            "reject_H0": reject,
            "cohens_d": d,
            "effect_magnitude": _interpret_cohens_d(abs(d)),
        })
    return pd.DataFrame(rows)


def _interpret_cohens_d(d: float) -> str:
    """Categorise Cohen's d magnitude."""
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# Full statistical report
# ---------------------------------------------------------------------------

def run_statistical_analysis(
    scenario_results: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete statistical analysis pipeline.

    Parameters
    ----------
    scenario_results : dict mapping scenario_key → replication DataFrame

    Returns
    -------
    (descriptive_df, pairwise_df)
    """
    print("\n=== STATISTICAL ANALYSIS ===")

    # 1. Descriptive statistics for key metrics
    desc_rows = []
    for metric in ["total_cost", "fill_rate", "stockout_days", "avg_inventory"]:
        for key, df in scenario_results.items():
            s = descriptive_stats(df, metric)
            s["scenario_key"] = key
            s["scenario_label"] = SCENARIOS[key]["label"]
            s["metric"] = metric
            desc_rows.append(s)
    descriptive_df = pd.DataFrame(desc_rows)

    # 2. Pairwise t-tests (total_cost and fill_rate)
    pair_dfs = []
    for metric in ["total_cost", "fill_rate"]:
        pair_df = pairwise_ttest_vs_baseline(scenario_results, metric=metric)
        pair_dfs.append(pair_df)
    pairwise_df = pd.concat(pair_dfs, ignore_index=True)

    # Print summary
    print("\nPairwise t-test vs Baseline (total_cost):")
    cost_pair = pairwise_df[pairwise_df["metric"] == "total_cost"]
    for _, row in cost_pair.iterrows():
        star = " *" if row["reject_H0"] else ""
        print(f"  {row['scenario_label']:25s}: "
              f"Δcost={row['mean_diff']:+10,.0f} R$, "
              f"d={row['cohens_d']:+.2f} ({row['effect_magnitude']})"
              f"{star}")

    # Save
    desc_path = DATA_RESULTS / "descriptive_statistics.csv"
    pair_path = DATA_RESULTS / "pairwise_tests.csv"
    descriptive_df.to_csv(desc_path, index=False)
    pairwise_df.to_csv(pair_path, index=False)
    print(f"\n[statistical_analysis] Descriptive stats saved to {desc_path}")
    print(f"[statistical_analysis] Pairwise tests saved to {pair_path}")

    return descriptive_df, pairwise_df
