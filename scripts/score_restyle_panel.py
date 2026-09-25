#!/usr/bin/env python3
"""Score the plain-vs-inverted restyling panel (outputs of run_restyle_panel.py).

Per model: accuracy on each arm, discordant counts (lost = right on plain & wrong on
inverted; gained = the reverse), exact two-sided McNemar p, Holm-adjusted p over the
models scored, unparsed counts, and the share of 'A' answers per arm (position lock).
Usage: score_restyle_panel.py [--dir rebuttal_2026/restyle_panel] [--json out.json]
"""
import argparse, glob, json, os
import pandas as pd
from scipy.stats import binomtest
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_restyle_panel import parse_strict

def reparse(df):
    raw = df["raw"].fillna("").astype(str).str.replace("\\n", "\n", regex=False)
    choice = raw.map(parse_strict)
    df = df.copy(); df["choice"] = choice.fillna("NONE")
    df["correct"] = [0 if c == "NONE" else int((c == "A") == (t == 1)) for c, t in zip(df["choice"], df["true_is_a"])]
    return df

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [None]*len(ps); running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (len(ps)-rank) * ps[i]); running = max(running, val); adj[i] = running
    return adj

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "predictions", "restyle"))
    ap.add_argument("--json", default=None); a = ap.parse_args()
    rows = []
    for fp in sorted(glob.glob(os.path.join(a.dir, "*__plain131.csv"))):
        fi = fp.replace("__plain131", "__inverted131")
        if not os.path.exists(fi): continue
        p = reparse(pd.read_csv(fp)); q = reparse(pd.read_csv(fi)); m = p.merge(q, on="pair_id", suffixes=("_p", "_i"))
        lost = int(((m.correct_p == 1) & (m.correct_i == 0)).sum()); gained = int(((m.correct_p == 0) & (m.correct_i == 1)).sum())
        pval = binomtest(min(lost, gained), lost + gained, 0.5).pvalue if lost + gained else 1.0
        rows.append(dict(model=p.model_name.iloc[0], n=len(m), acc_plain=p.correct.mean(), acc_inverted=q.correct.mean(),
                         lost=lost, gained=gained, p=pval, none_plain=int((p.choice == "NONE").sum()), none_inverted=int((q.choice == "NONE").sum()),
                         rateA_plain=float((p.choice == "A").mean()), rateA_inverted=float((q.choice == "A").mean())))
    if not rows: print("no complete model pairs found in", a.dir); return
    adj = holm([r["p"] for r in rows])
    for r, h in zip(rows, adj): r["p_holm"] = h
    print(f"{'model':38s} {'plain':>6s} {'inv':>6s} {'lost':>4s} {'gain':>4s} {'p':>9s} {'p_holm':>8s} {'NONE p/i':>9s} {'A% p/i':>11s}")
    for r in rows:
        print(f"{r['model']:38s} {r['acc_plain']:6.3f} {r['acc_inverted']:6.3f} {r['lost']:4d} {r['gained']:4d} {r['p']:9.2g} {r['p_holm']:8.2g} {r['none_plain']:4d}/{r['none_inverted']:<4d} {r['rateA_plain']:5.2f}/{r['rateA_inverted']:<5.2f}")
    if a.json: json.dump(rows, open(a.json, "w"), indent=1); print("wrote", a.json)

if __name__ == "__main__":
    main()
