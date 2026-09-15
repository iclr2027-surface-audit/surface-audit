#!/usr/bin/env python3
"""Table 5: per-category retention of TruthfulQA-476 (retained / total pairs per category)."""
import json, pathlib, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
tqa = pd.read_csv(ROOT / "TruthfulQA.csv").reset_index(drop=True)
keep = set(json.load(open(ROOT / "data/subsets/TruthfulQA-Audited/surface6/pair_ids/pair_ids_theta053.json"))["pair_ids"])
tqa["retained"] = tqa.index.isin(keep)
g = tqa.groupby("Category")["retained"].agg(["sum", "count"]).sort_values("sum")
g["rate"] = g["sum"] / g["count"]
assert g["sum"].sum() == 476 and g["count"].sum() == 790 and (g["sum"] > 0).all()
for cat, r in g.sort_values("rate").iterrows():
    print(f"{cat:28s} {int(r['sum']):3d}/{int(r['count']):3d}  {100*r['rate']:.0f}%")
adv = tqa.groupby("Type")["retained"].agg(["sum", "count"])
print("\nby type:", {t: f"{int(r['sum'])}/{int(r['count'])}" for t, r in adv.iterrows()}, "| categories:", len(g))
