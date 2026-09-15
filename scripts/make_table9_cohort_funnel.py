#!/usr/bin/env python3
"""Table 9: acceptance funnel of the surface-flipped cohort, per batch, from the released pipeline files.

Columns: Generated (raw generations), Validator (rule-based surface validator passes), Polarity (gate
passes; the pilot predates the gate), Judge >= 0.8 (joint truth-judge passes at confidence 0.8),
Drift-flagged (advisory drift screen; pilot not screened), In cohort (released pairs after
de-duplication and author verification, attributed to batches by id block: pilot < 1000, batch 4 =
3001-3020, batch b = b*1000+k otherwise).
"""
import json, pathlib, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; A = ROOT / "audits"
released = pd.read_csv(ROOT / "hf_release/SurfaceFlipped-131.csv")["id"]
def batch_of(i): return "pilot" if i < 1000 else ("4" if 3000 <= i < 4000 else str(i // 1000))
incohort = released.map(batch_of).value_counts().to_dict()
rows, tot = [], [0] * 6
for b in ["pilot"] + [str(i) for i in range(4, 14)]:
    p = "_eval_dataset_v3" if b == "pilot" else f"_eval_dataset_v3_batch{b}"
    gen = len(json.load(open(A / f"{p}_raw.json"))["rows"])
    val = int(pd.read_csv(A / f"{p}_validated.csv")["all_passes"].sum())
    pol = None if b == "pilot" else len(pd.read_csv(A / f"{p}_polarity_passed.csv"))
    jf = A / (f"{p}_judged.json" if b in ("pilot", "4") else f"{p}p_judged.json")
    jd = json.load(open(jf)); jud = jd.get("n_pass_joint_conf80", jd.get("n_pass_joint"))
    drf = None if b == "pilot" else json.load(open(A / f"{p}_drift_dropped.json"))["n_dropped"]
    inc = incohort.get(b, 0)
    rows.append((b, gen, val, pol, jud, drf, inc))
    for j, v in enumerate((gen, val, pol, jud, drf, inc)):
        tot[j] += (v or 0)
print(f"{'batch':6} {'Generated':>9} {'Validator':>9} {'Polarity':>8} {'Judge>=0.8':>10} {'Drift':>6} {'In cohort':>9}")
for r in rows: print(f"{r[0]:6} {r[1]:9d} {r[2]:9d} {str(r[3] if r[3] is not None else '---'):>8} {r[4]:10d} {str(r[5] if r[5] is not None else '---'):>6} {r[6]:9d}")
print(f"{'Total':6} {tot[0]:9d} {tot[1]:9d} {tot[2]:8d} {tot[3]:10d} {tot[4]:6d} {tot[5]:9d}")
assert tot == [240, 179, 149, 161, 53, 131], tot
