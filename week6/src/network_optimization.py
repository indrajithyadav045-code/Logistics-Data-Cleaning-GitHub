"""
network_optimization.py
=======================
Prescriptive operations research optimization for multi-hub regional fulfillment
allocation and logistics network redesign for the Week 6 Logistics Project.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week6" / "config"))
from project_config import (
    DATA_PROCESSED,
    DATA_RESULTS,
    CANDIDATE_HUBS,
    STATE_COORDINATES,
    COST_PER_KM_PER_KG,
    BASE_HANDLING_COST_PER_ORDER,
    INTERSTATE_TAX_PENALTY,
    LATE_DELIVERY_PENALTY_PER_DAY,
    AVERAGE_TRUCK_SPEED_KM_PER_DAY,
    CARBON_EMISSIONS_KG_PER_TON_KM,
)
from data_pipeline import get_state_distance


def run_network_optimization(df: pd.DataFrame | None = None) -> dict:
    """Formulate and solve the multi-hub regional fulfillment allocation optimization.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Processed orders dataset.

    Returns
    -------
    dict: Comparative performance metrics between As-Is and Optimized networks.
    """
    print("\n--- [Pillar 4: Prescriptive Network Re-allocation & Optimization] ---")
    if df is None:
        input_path = DATA_PROCESSED / "consolidated_logistics_orders.csv"
        df = pd.read_csv(input_path, low_memory=False)

    # 1. State Demand & Characteristics
    state_demand = df.groupby("customer_state").agg(
        demand_orders=("order_id", "count"),
        mean_weight_kg=("weight_kg", "mean"),
        as_is_delivery_days=("delivery_days", "mean"),
        as_is_freight=("total_freight", "mean"),
        as_is_delay_rate=("is_delayed", "mean"),
    ).reset_index()

    customer_states = state_demand["customer_state"].tolist()
    demands = state_demand["demand_orders"].values
    mean_weights = state_demand["mean_weight_kg"].values
    n_states = len(customer_states)
    n_hubs = len(CANDIDATE_HUBS)

    print(f"Customer Destinations: {n_states} Brazilian states")
    print(f"Candidate Fulfillment Hubs: {CANDIDATE_HUBS} ({n_hubs} hubs)")

    # 2. Cost Matrix C[i, j] and Duration Matrix T[i, j]
    # i = hub index, j = customer state index
    cost_matrix = np.zeros((n_hubs, n_states))
    duration_matrix = np.zeros((n_hubs, n_states))
    distance_matrix = np.zeros((n_hubs, n_states))

    standard_sla_days = 12.0  # Promised customer delivery SLA benchmark

    for i, hub in enumerate(CANDIDATE_HUBS):
        for j, c_state in enumerate(customer_states):
            dist = get_state_distance(hub, c_state)
            distance_matrix[i, j] = dist

            # Predicted transit duration = 1.5 days dispatch + road transit
            travel_days = dist / AVERAGE_TRUCK_SPEED_KM_PER_DAY
            pred_days = 1.5 + travel_days
            duration_matrix[i, j] = pred_days

            # Transit cost per order = base handling + distance * weight * rate + interstate friction
            weight = mean_weights[j]
            transport_cost = BASE_HANDLING_COST_PER_ORDER + (dist * weight * COST_PER_KM_PER_KG)
            if hub != c_state:
                transport_cost += INTERSTATE_TAX_PENALTY

            # Expected SLA breach penalty
            late_days = max(0.0, pred_days - standard_sla_days)
            penalty_cost = late_days * LATE_DELIVERY_PENALTY_PER_DAY

            cost_matrix[i, j] = transport_cost + penalty_cost

    # 3. Solve Linear Programming / Optimal Assignment
    # Objective: Minimize sum_{i, j} D_j * cost_matrix[i, j] * x_ij
    # Subject to: sum_i x_ij = 1 for each j
    # Capacity constraint: Hub volume <= max_capacity (e.g. 40% of total national volume per hub)
    total_volume = demands.sum()
    max_hub_capacity = 0.45 * total_volume  # Prevents any single hub from overloading

    c_vector = []
    for i in range(n_hubs):
        for j in range(n_states):
            c_vector.append(demands[j] * cost_matrix[i, j])
    c_vector = np.array(c_vector)

    # Equality constraints: sum_i x_ij = 1 for each j (n_states constraints)
    A_eq = np.zeros((n_states, n_hubs * n_states))
    b_eq = np.ones(n_states)
    for j in range(n_states):
        for i in range(n_hubs):
            A_eq[j, i * n_states + j] = 1.0

    # Inequality constraints: sum_j D_j * x_ij <= max_hub_capacity for each hub (n_hubs constraints)
    A_ub = np.zeros((n_hubs, n_hubs * n_states))
    b_ub = np.ones(n_hubs) * max_hub_capacity
    for i in range(n_hubs):
        for j in range(n_states):
            A_ub[i, i * n_states + j] = demands[j]

    # Bounds: 0 <= x_ij <= 1
    bounds = [(0, 1) for _ in range(n_hubs * n_states)]

    res = linprog(
        c_vector,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise RuntimeError(f"Linear programming optimization failed: {res.message}")

    solution = res.x.reshape((n_hubs, n_states))
    optimal_cost_brl = float(res.fun)

    # 4. As-Is Centralized Network Baseline (All orders shipped from SP hub)
    sp_idx = CANDIDATE_HUBS.index("SP")
    as_is_cost_brl = float(sum(demands[j] * cost_matrix[sp_idx, j] for j in range(n_states)))

    # Compute operational impact comparisons
    # As-Is metrics
    as_is_mean_dist_km = float(sum(demands[j] * distance_matrix[sp_idx, j] for j in range(n_states)) / total_volume)
    as_is_mean_days = float(sum(demands[j] * duration_matrix[sp_idx, j] for j in range(n_states)) / total_volume)
    as_is_late_orders = int(sum(
        demands[j] for j in range(n_states) if duration_matrix[sp_idx, j] > standard_sla_days
    ))
    as_is_late_rate = float(as_is_late_orders / total_volume * 100.0)

    # Optimized metrics
    opt_mean_dist_km = float(sum(
        solution[i, j] * demands[j] * distance_matrix[i, j]
        for i in range(n_hubs) for j in range(n_states)
    ) / total_volume)
    opt_mean_days = float(sum(
        solution[i, j] * demands[j] * duration_matrix[i, j]
        for i in range(n_hubs) for j in range(n_states)
    ) / total_volume)
    opt_late_orders = int(sum(
        solution[i, j] * demands[j]
        for i in range(n_hubs) for j in range(n_states)
        if duration_matrix[i, j] > standard_sla_days
    ))
    opt_late_rate = float(opt_late_orders / total_volume * 100.0)

    # Savings
    annual_cost_savings_brl = as_is_cost_brl - optimal_cost_brl
    cost_reduction_pct = (annual_cost_savings_brl / as_is_cost_brl) * 100.0
    delivery_time_reduction_days = as_is_mean_days - opt_mean_days
    delivery_time_reduction_pct = (delivery_time_reduction_days / as_is_mean_days) * 100.0
    distance_reduction_km = as_is_mean_dist_km - opt_mean_dist_km

    # Environmental Carbon Emissions (Tons of CO2)
    # CO2 = distance_km * (weight_kg / 1000) * factor
    mean_national_weight_ton = float(np.mean(mean_weights)) / 1000.0
    as_is_co2_tons = (as_is_mean_dist_km * total_volume * mean_national_weight_ton * CARBON_EMISSIONS_KG_PER_TON_KM) / 1000.0
    opt_co2_tons = (opt_mean_dist_km * total_volume * mean_national_weight_ton * CARBON_EMISSIONS_KG_PER_TON_KM) / 1000.0
    co2_reduction_tons = as_is_co2_tons - opt_co2_tons
    co2_reduction_pct = (co2_reduction_tons / as_is_co2_tons) * 100.0

    print(f"\n--- OPTIMIZATION RESULTS SUMMARY ---")
    print(f"As-Is Centralized Cost (SP Only): R$ {as_is_cost_brl:,.2f}")
    print(f"Optimized Multi-Hub Cost        : R$ {optimal_cost_brl:,.2f}")
    print(f"Annual Logistics Cost Savings   : R$ {annual_cost_savings_brl:,.2f} ({cost_reduction_pct:.2f}% savings)")
    print(f"Mean Delivery Transit Duration  : {as_is_mean_days:.2f}d → {opt_mean_days:.2f}d (-{delivery_time_reduction_days:.2f}d, {delivery_time_reduction_pct:.1f}% faster)")
    print(f"Average Transit Distance        : {as_is_mean_dist_km:.1f} km → {opt_mean_dist_km:.1f} km (-{distance_reduction_km:.1f} km)")
    print(f"Late Delivery Rate (SLA Breach) : {as_is_late_rate:.2f}% → {opt_late_rate:.2f}%")
    print(f"Carbon Emissions (Diesel CO2)   : {as_is_co2_tons:.1f} tons → {opt_co2_tons:.1f} tons (-{co2_reduction_tons:.1f} tons, -{co2_reduction_pct:.1f}%)")

    # 5. Extract Optimal State-to-Hub Assignments
    assignment_records = []
    for j, c_state in enumerate(customer_states):
        # Find dominant assigned hub
        best_hub_idx = int(np.argmax(solution[:, j]))
        best_hub = CANDIDATE_HUBS[best_hub_idx]
        alloc_pct = float(solution[best_hub_idx, j] * 100.0)
        dist = distance_matrix[best_hub_idx, j]
        days = duration_matrix[best_hub_idx, j]

        assignment_records.append({
            "customer_state": c_state,
            "assigned_hub": best_hub,
            "allocation_pct": alloc_pct,
            "demand_orders": int(demands[j]),
            "as_is_distance_km": float(distance_matrix[sp_idx, j]),
            "optimized_distance_km": float(dist),
            "as_is_delivery_days": float(duration_matrix[sp_idx, j]),
            "optimized_delivery_days": float(days),
        })

    assign_df = pd.DataFrame(assignment_records)
    assign_path = DATA_RESULTS / "optimized_hub_assignments.csv"
    assign_df.to_csv(assign_path, index=False)

    # Hub throughput distribution
    hub_throughput = []
    for i, hub in enumerate(CANDIDATE_HUBS):
        orders_handled = float(sum(solution[i, j] * demands[j] for j in range(n_states)))
        hub_throughput.append({
            "hub": hub,
            "orders_handled": orders_handled,
            "throughput_pct": (orders_handled / total_volume) * 100.0,
        })
    hub_df = pd.DataFrame(hub_throughput).sort_values("orders_handled", ascending=False)
    hub_path = DATA_RESULTS / "hub_throughput_distribution.csv"
    hub_df.to_csv(hub_path, index=False)

    opt_summary = {
        "as_is_centralized": {
            "total_cost_brl": as_is_cost_brl,
            "mean_delivery_days": as_is_mean_days,
            "mean_distance_km": as_is_mean_dist_km,
            "late_delivery_rate_pct": as_is_late_rate,
            "co2_emissions_tons": as_is_co2_tons,
        },
        "optimized_multi_hub": {
            "total_cost_brl": optimal_cost_brl,
            "mean_delivery_days": opt_mean_days,
            "mean_distance_km": opt_mean_dist_km,
            "late_delivery_rate_pct": opt_late_rate,
            "co2_emissions_tons": opt_co2_tons,
        },
        "improvements": {
            "annual_cost_savings_brl": annual_cost_savings_brl,
            "cost_reduction_pct": cost_reduction_pct,
            "delivery_time_reduction_days": delivery_time_reduction_days,
            "delivery_time_reduction_pct": delivery_time_reduction_pct,
            "distance_reduction_km": distance_reduction_km,
            "co2_reduction_tons": co2_reduction_tons,
            "co2_reduction_pct": co2_reduction_pct,
        },
        "hub_throughput": hub_df.to_dict(orient="records"),
    }

    with open(DATA_RESULTS / "network_optimization_metrics.json", "w") as f:
        json.dump(opt_summary, f, indent=2)

    return opt_summary


if __name__ == "__main__":
    run_network_optimization()
