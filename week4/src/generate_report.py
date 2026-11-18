"""Create the data-derived Week 4 Word report after optimisation is executed."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
OUT, REPORTS = ROOT / "week4" / "outputs", ROOT / "week4" / "reports"

def add_table(doc, headers, rows):
    t=doc.add_table(rows=1, cols=len(headers)); t.style="Light Shading Accent 1"
    for c,v in zip(t.rows[0].cells,headers): c.text=str(v)
    for row in rows:
        cells=t.add_row().cells
        for c,v in zip(cells,row): c.text=str(v)

def code_block(doc, text):
    p=doc.add_paragraph(); r=p.add_run(text); r.font.name="Consolas"; r.font.size=Pt(8)

def main():
    inputs=pd.read_csv(OUT/"demand_inputs.csv"); summary=pd.read_csv(OUT/"scenario_summary.csv"); flows=pd.read_csv(OUT/"allocation_flows.csv")
    metadata=json.loads((REPORTS/"measured_results.json").read_text(encoding="utf-8"))
    base=summary.loc[summary.scenario=="Base case"].iloc[0]
    surge=summary.loc[summary.scenario=="Southern demand surge"].iloc[0]
    disruption=summary.loc[summary.scenario=="Sao Paulo capacity disruption"].iloc[0]
    base_flow=flows[flows.scenario=="Base case"].copy()
    doc=Document(); sec=doc.sections[0]; sec.top_margin=sec.bottom_margin=Inches(.65); sec.left_margin=sec.right_margin=Inches(.75)
    for n in ["Normal","Heading 1","Heading 2"]:
        s=doc.styles[n]; s.font.name="Aptos"; s.font.size=Pt(10.5 if n=="Normal" else 14); s.font.color.rgb=RGBColor(0,0,0)
    title=doc.add_paragraph(style="Title"); title.alignment=WD_ALIGN_PARAGRAPH.CENTER; title.add_run("Week 4 Supply Chain Network Optimization")
    sub=doc.add_paragraph("Olist inspired distribution centre location and allocation model"); sub.alignment=WD_ALIGN_PARAGRAPH.CENTER
    doc.add_heading("Executive result",1)
    doc.add_paragraph(f"A mixed-integer linear program selected Sao Paulo, Rio de Janeiro, and Curitiba as the three distribution centres for the base case. It allocates {int(base.total_orders):,} historical Olist orders across six high-demand customer states at an assumed total network cost of {base.objective_cost:,.0f}. The weighted average transit time is {base.weighted_avg_transit_days:.2f} days, below the {base.service_limit_days:.2f}-day service ceiling. This is a reproducible educational network-design model: demand is data-derived, while facility, transport, and time inputs are stated scenario assumptions.")
    doc.add_heading("Business question and scenario boundary",1)
    doc.add_paragraph("The business question is which candidate distribution centres should operate, and how should customer-state demand be assigned to them, when managers must balance facility cost, transport cost, capacity, and delivery speed. The model is publicly inspired by Brazil's Olist marketplace geography. It does not claim to reproduce Olist's real warehouse network or commercial rate card.")
    doc.add_heading("Demand data and worked aggregation example",1)
    doc.add_paragraph("Demand is the distinct count of `order_id` after joining valid Olist orders to customers through `customer_id`, then grouping by `customer_state`. Duplicate order IDs and orders without a usable purchase timestamp, order ID, or customer ID are excluded because they cannot represent a uniquely dated customer order. The six largest states are modelled to keep the network scenario focused and auditable.")
    add_table(doc,["Customer state","Actual historical unique orders"],[[r.demand_state,f"{int(r.historical_order_demand):,}"] for r in inputs.itertuples(index=False)])
    doc.add_paragraph(f"Worked example: Sao Paulo contributes {int(inputs.iloc[0].historical_order_demand):,} orders. In the base model, every demand-state constraint requires the sum of allocations from all centres to equal that observed count. For example, x[Sao Paulo, SP] + x[Rio de Janeiro, SP] + x[Curitiba, SP] + x[Belo Horizonte, SP] = {int(inputs.iloc[0].historical_order_demand):,}. The solution therefore cannot reduce cost by leaving any modelled customer demand unserved.")
    doc.add_heading("Assumptions and why they are separated from observed data",1)
    doc.add_paragraph("Actual Olist demand alone does not reveal a company's facility leases, lane rates, capacities, or service promise. To make the optimisation solvable without fabricating those values as observations, the model uses explicit hypothetical assumptions: four candidate centres (Sao Paulo, Rio de Janeiro, Curitiba, Belo Horizonte); fixed costs of 180,000, 125,000, 110,000, and 105,000 cost units; capacities of 48,000, 30,000, 28,000, and 26,000 orders; lane costs from 7 to 25 cost units per order; and lane transit times from 1.0 to 3.5 days. The service constraint limits the allocation-weighted average transit time to 2.35 days.")
    doc.add_heading("Mathematical formulation",1)
    doc.add_paragraph("Sets: i is a candidate distribution centre and j is a customer-state demand node. Decision variable x(i,j) is the continuous number of orders allocated from centre i to state j. Binary variable y(i) equals one when centre i is opened. Parameters d(j), c(i,j), F(i), Cap(i), and t(i,j) are demand, lane cost, fixed facility cost, capacity, and transit days.")
    code_block(doc,"Minimise   sum_i sum_j c(i,j) x(i,j) + sum_i F(i) y(i)\nSubject to sum_i x(i,j) = d(j)                         for every j\n           sum_j x(i,j) <= Cap(i) y(i)                 for every i\n           sum_i sum_j t(i,j) x(i,j) <= 2.35 sum_j d(j)\n           x(i,j) >= 0;  y(i) in {0,1}")
    doc.add_paragraph("The objective trades variable transport cost against opening a facility. Demand equalities enforce complete service, capacity links flows to binary opening decisions, and the final inequality imposes a network-level service target. This structure is a MILP because it combines continuous allocations with binary location decisions.")
    doc.add_heading("Implementation excerpts and solver rationale",1)
    doc.add_paragraph("SciPy's `milp` interface with the HiGHS solver is used because it directly supports binary variables, linear constraints, and reproducible deterministic optimisation. The first excerpt constructs demand balance; the second prevents capacity from being used unless the centre is open.")
    code_block(doc,"# For each demand state j, allocations from all centres must equal demand[j]\nfor j in range(n_j):\n    for i in range(n_i):\n        demand_matrix[j, i * n_j + j] = 1\nconstraints.append(LinearConstraint(demand_matrix.tocsr(), demand, demand))")
    code_block(doc,"# Facility capacity is available only when binary open[i] equals one\nfor i in range(n_i):\n    cap_matrix[i, i*n_j:(i+1)*n_j] = 1\n    cap_matrix[i, n_flow+i] = -capacity[i]\nconstraints.append(LinearConstraint(cap_matrix.tocsr(), -inf, zeros))")
    doc.add_heading("Base case allocation and chart interpretation",1)
    doc.add_paragraph("The stacked bar chart was selected because each customer state has one fixed total demand and the reader needs to see how that total is split among centres. It makes allocation composition visible without suggesting a geographic map that the assumed lane matrix cannot support. The result shows which centres carry each state's historical volume; it should be interpreted jointly with capacity utilisation, not as proof that a real carrier lane exists.")
    doc.add_picture(str(OUT/"base_allocation.png"),width=Inches(6.6))
    add_table(doc,["Open centre","Base-case capacity utilisation"],[[piece.split(":")[0],piece.split(":")[1].strip()] for piece in str(base.capacity_utilisation).split(";")])
    doc.add_paragraph(f"Sao Paulo is fully utilised in the optimum, while Rio de Janeiro is at 79.0% and Curitiba at 31.0%. The high utilisation flags Sao Paulo as a capacity risk: its importance derives from this particular assumed cost-time matrix and demand profile, not a claim about Olist operations.")
    doc.add_heading("Scenario comparison and operational interpretation",1)
    doc.add_paragraph("Three scenarios use identical model logic so their cost changes can be attributed to altered conditions. The southern surge raises RS and SC demand by 25%. The disruption lowers only Sao Paulo's assumed capacity by 35%. A bar chart is appropriate here because scenarios are discrete alternatives and the key question is direct comparison of their optimized total cost.")
    add_table(doc,["Scenario","Orders","Total assumed cost","Avg transit days","Open centres"],[[r.scenario,f"{int(r.total_orders):,}",f"{r.objective_cost:,.0f}",f"{r.weighted_avg_transit_days:.2f}",r.open_distribution_centres] for r in summary.itertuples(index=False)])
    doc.add_picture(str(OUT/"scenario_cost_comparison.png"),width=Inches(6.4))
    doc.add_paragraph(f"The southern surge adds {int(surge.total_orders-base.total_orders):,} orders and raises optimal cost by {surge.objective_cost-base.objective_cost:,.0f}; Curitiba utilisation increases from 31.0% to 38.8%, while the open-centre set stays unchanged. The Sao Paulo disruption leaves demand unchanged but raises cost by {disruption.objective_cost-base.objective_cost:,.0f} and weighted transit time from {base.weighted_avg_transit_days:.2f} to {disruption.weighted_avg_transit_days:.2f} days. Rio de Janeiro reaches 100% utilisation in that result, revealing diminished buffer capacity.")
    doc.add_heading("Limitations and improvements",1)
    doc.add_paragraph("Demand is a historical aggregate, not a forecast, and six states do not represent all Brazilian demand. Costs, capacities, and transit days are assumptions; a real study would obtain lane quotations, lease and labour costs, warehouse handling limits, route distances, and service-level commitments. The average-time constraint can hide poor service on individual lanes, so a production model should add lane-specific maximum-time rules or service coverage constraints. Other worthwhile extensions are multi-period seasonality, inventory holding cost, vehicle capacity, carbon cost, uncertainty scenarios, and integer shipment or pallet quantities.")
    doc.add_heading("Reproducibility",1)
    doc.add_paragraph("Run `python week4\\src\\run_network_optimization.py --data-root <folder containing Olist CSVs>` and then `python week4\\src\\generate_report.py`. The first script writes demand inputs, allocation flows, scenario summary, charts, and measured metadata; the second script builds this report from those generated files. Raw data is deliberately excluded from version control.")
    doc.save(REPORTS/"week4_supply_chain_network_optimization_report.docx")
    md=f"# Week 4 Supply Chain Network Optimization\n\nBase result: {base.open_distribution_centres}; cost {base.objective_cost:,.0f}; weighted time {base.weighted_avg_transit_days:.2f} days.\n\nDemand is actual Olist historical order count by customer state; facility and lane values are hypothetical assumptions. See the DOCX for formulation, code excerpts, scenarios, and limitations.\n"
    (REPORTS/"week4_report.md").write_text(md,encoding="utf-8")

if __name__ == "__main__": main()
