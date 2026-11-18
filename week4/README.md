# Week 4 Supply Chain Network Optimization

This reproducible project selects distribution centres and allocates Olist-order demand to them with a mixed-integer linear program (MILP). It uses actual historical demand counts by customer state from the Olist orders and customers files. Candidate-centre capacities, fixed operating costs, transport costs, and transit times are clearly labelled scenario assumptions, not observed commercial rates.

## Run

Raw source data is intentionally not committed. Download the Olist dataset and point the script at the folder containing `olist_orders_dataset.csv` and `olist_customers_dataset.csv`:

```powershell
python week4\src\run_network_optimization.py --data-root D:\path\to\data\raw
python week4\src\generate_report.py
```

The solver uses `scipy.optimize.milp` (HiGHS). It writes compact, reproducible results to `week4/outputs/` and the Word report to `week4/reports/`.

## Model

For each candidate distribution centre `i` and demand state `j`, `x[i,j]` is allocated historical order volume and `y[i]` is a binary open/close decision. The model minimizes transport cost plus fixed facility cost, satisfies every demand node, observes capacity, and keeps weighted average transit time below the stated service limit. Three scenarios test base conditions, a southern-demand surge, and reduced Sao Paulo capacity.

Raw data is never changed by this project.
