"""Solve a publicly inspired Olist distribution-network MILP with SciPy HiGHS.

The demand counts are actual historical Olist purchase orders by customer state.
Facility costs, capacities, transport costs, and transit times are transparent
scenario assumptions, included to demonstrate network-design optimisation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[2]
OUT, REPORTS = ROOT / "week4" / "outputs", ROOT / "week4" / "reports"
DC = ["Sao Paulo", "Rio de Janeiro", "Curitiba", "Belo Horizonte"]
DC_STATE = ["SP", "RJ", "PR", "MG"]
FIXED_COST = np.array([180000, 125000, 110000, 105000], dtype=float)
BASE_CAPACITY = np.array([48000, 30000, 28000, 26000], dtype=float)
# Assumed per-order transport costs and one-way transit days; not observed Olist rates.
COST = np.array([[7, 15, 13, 12, 22, 20], [14, 7, 17, 15, 25, 23], [15, 20, 23, 21, 14, 11], [13, 16, 14, 8, 24, 22]], dtype=float)
TIME = np.array([[1.0, 2.0, 2.0, 1.5, 3.0, 3.0], [2.0, 1.0, 2.5, 2.0, 3.5, 3.5], [2.5, 3.0, 3.0, 2.5, 1.0, 1.5], [2.0, 2.5, 2.5, 1.0, 3.5, 3.5]], dtype=float)
TIME_LIMIT = 2.35

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--data-root", type=Path, required=True, help="Folder holding the two raw Olist CSV files")
    return p.parse_args()

def load_demand(data_root: Path) -> pd.Series:
    orders = pd.read_csv(data_root / "olist_orders_dataset.csv", usecols=["order_id", "customer_id", "order_purchase_timestamp"])
    customers = pd.read_csv(data_root / "olist_customers_dataset.csv", usecols=["customer_id", "customer_state"])
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
    valid = orders.dropna(subset=["order_id", "customer_id", "order_purchase_timestamp"]).drop_duplicates("order_id")
    merged = valid.merge(customers.drop_duplicates("customer_id"), on="customer_id", how="inner")
    return merged.groupby("customer_state")["order_id"].nunique().sort_values(ascending=False)

def solve_scenario(name: str, states: list[str], demand: np.ndarray, capacity: np.ndarray) -> tuple[pd.DataFrame, dict]:
    n_i, n_j = len(DC), len(states); n_flow = n_i*n_j; n_var = n_flow+n_i
    objective = np.r_[COST.ravel(), FIXED_COST]; integrality = np.r_[np.zeros(n_flow), np.ones(n_i)]
    constraints = []
    dm = lil_matrix((n_j, n_var))
    for j in range(n_j):
        for i in range(n_i): dm[j, i*n_j+j] = 1
    constraints.append(LinearConstraint(dm.tocsr(), demand, demand))
    cm = lil_matrix((n_i, n_var))
    for i in range(n_i): cm[i, i*n_j:(i+1)*n_j] = 1; cm[i, n_flow+i] = -capacity[i]
    constraints.append(LinearConstraint(cm.tocsr(), -np.inf*np.ones(n_i), np.zeros(n_i)))
    constraints.append(LinearConstraint(np.r_[TIME.ravel(), np.zeros(n_i)][None,:], -np.inf, [TIME_LIMIT*demand.sum()]))
    result = milp(c=objective, integrality=integrality, bounds=Bounds(np.zeros(n_var), np.r_[np.full(n_flow, np.inf), np.ones(n_i)]), constraints=constraints, options={"time_limit":60})
    if not result.success: raise RuntimeError(f"{name} did not solve: {result.message}")
    flow, opened = result.x[:n_flow].reshape(n_i,n_j), result.x[n_flow:]
    rows=[]
    for i in range(n_i):
        for j in range(n_j):
            if flow[i,j] > 1e-5: rows.append({"scenario":name,"dc":DC[i],"dc_state":DC_STATE[i],"demand_state":states[j],"allocated_orders":flow[i,j],"unit_cost_assumption":COST[i,j],"transit_days_assumption":TIME[i,j]})
    variable=float((flow*COST).sum()); fixed=float((opened*FIXED_COST).sum())
    return pd.DataFrame(rows), {"scenario":name,"status":result.message,"objective_cost":float(result.fun),"variable_transport_cost":variable,"fixed_facility_cost":fixed,"total_orders":int(demand.sum()),"weighted_avg_transit_days":float((flow*TIME).sum()/demand.sum()),"service_limit_days":TIME_LIMIT,"open_distribution_centres":", ".join(DC[i] for i,v in enumerate(opened) if v>.5),"opened_count":int((opened>.5).sum()),"capacity_utilisation":"; ".join(f"{DC[i]}: {flow[i].sum()/capacity[i]:.1%}" for i in range(n_i) if opened[i]>.5)}

def plots(flows: pd.DataFrame, summary: pd.DataFrame, states: list[str]) -> None:
    base=flows[flows.scenario=="Base case"].pivot_table(index="demand_state",columns="dc",values="allocated_orders",aggfunc="sum",fill_value=0).reindex(states)
    ax=base.plot(kind="bar",stacked=True,figsize=(10,5),colormap="tab20c"); ax.set(title="Base case order allocation by demand state",xlabel="Customer state",ylabel="Historical orders allocated"); ax.legend(title="Distribution centre"); plt.tight_layout(); plt.savefig(OUT/"base_allocation.png",dpi=170); plt.close()
    fig,ax=plt.subplots(figsize=(9,4.5)); ax.bar(summary.scenario,summary.objective_cost,color=["#4C78A8","#F58518","#54A24B"]); ax.set(title="Optimised total cost across scenarios",ylabel="Assumed cost units"); ax.tick_params(axis="x",rotation=12)
    for i,v in enumerate(summary.objective_cost): ax.text(i,v,f"{v:,.0f}",ha="center",va="bottom",fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/"scenario_cost_comparison.png",dpi=170); plt.close(fig)

def main() -> None:
    args=parse_args(); OUT.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True)
    demand_by_state=load_demand(args.data_root); states=demand_by_state.head(6).index.tolist(); base=demand_by_state.loc[states].to_numpy(dtype=float)
    scenarios=[("Base case",base,BASE_CAPACITY),("Southern demand surge",base*np.array([1,1,1,1,1.25,1.25]),BASE_CAPACITY),("Sao Paulo capacity disruption",base,BASE_CAPACITY*np.array([.65,1,1,1]))]
    flows=[]; summaries=[]
    for name,demand,capacity in scenarios:
        flow,summary=solve_scenario(name,states,demand,capacity); flows.append(flow); summaries.append(summary)
    flows,summary=pd.concat(flows,ignore_index=True),pd.DataFrame(summaries); inputs=pd.DataFrame({"demand_state":states,"historical_order_demand":base.astype(int)})
    inputs.to_csv(OUT/"demand_inputs.csv",index=False); flows.to_csv(OUT/"allocation_flows.csv",index=False); summary.to_csv(OUT/"scenario_summary.csv",index=False); plots(flows,summary,states)
    metadata={"demand_source":"Actual unique Olist order_id counts joined to customer_state","source_files":["olist_orders_dataset.csv","olist_customers_dataset.csv"],"states":states,"facility_assumptions":{"candidate_dcs":DC,"fixed_cost":FIXED_COST.tolist(),"capacity":BASE_CAPACITY.tolist(),"time_limit_days":TIME_LIMIT},"scenario_results":summary.round(4).to_dict(orient="records"),"important_caveat":"Costs, transit times, capacities and facility fixed costs are hypothetical educational assumptions; only demand counts are derived from Olist."}
    (REPORTS/"measured_results.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8"); print(json.dumps(metadata,indent=2))

if __name__ == "__main__": main()
