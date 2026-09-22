#!/usr/bin/env python3
"""D37 (1) — retained-vs-removed difficulty with instruments that have power.
Capable instruments (per-item correctness on all 790 pairs, same protocol as the panel): Qwen2.5-14B-Instruct (mean of seeds 42/43/44),
and the four frontier runs (claude-opus-5, gpt-5.4-2026-03-05, gpt-5.4-mini-2026-03-17, claude-haiku-4-5). Per-item difficulty = mean
correctness over the five instruments. Tests: two-sided Mann-Whitney U on per-item difficulty (476 retained vs 314 removed), Fisher exact per
instrument, and item discrimination = point-biserial correlation between an item's correctness vector and model ability across the
19-model pool (14 open-weights panel + 5 capable), compared retained vs removed. Also reports the 14-model panel on its own (key panel_14). Output: results/d37_difficulty_powered.json"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact, pearsonr
ROOT = Path(__file__).resolve().parents[1]
ids476 = set(json.load(open(ROOT / "data/subsets/TruthfulQA-Audited/surface6/pair_ids/pair_ids_theta053.json"))["pair_ids"]); assert len(ids476) == 476
def vec(path):
    d = pd.read_csv(path); assert len(d) == 790; return d.set_index("pair_id")["correct"].astype(float).sort_index()
# capable instruments
cap = {}
q = [vec(f) for f in sorted(glob.glob(str(ROOT / "data/predictions/model_predictions_Qwen_Qwen2.5-14B-Instruct_seed4[234]*.csv")))]; assert len(q) == 3
cap["Qwen2.5-14B (mean of 3 seeds)"] = sum(q) / 3
for f in sorted(glob.glob(str(ROOT / "data/predictions/frontier/*__tqa790.csv"))): cap[Path(f).name.split("__")[0]] = vec(f)
# 14-model open-weights panel (seed-42 files + pythia), as in scripts/repair_rank_fidelity.py
files = sorted(glob.glob(str(ROOT / "data/predictions/model_predictions_*seed42*"))) + [str(ROOT / "data/predictions/model_predictions_pythia.csv")]
df = pd.concat([pd.read_csv(f) for f in files]); M = df.pivot_table(index="pair_id", columns="model_name", values="correct", aggfunc="first").sort_index(); assert M.shape == (790, 14)
idx = np.array(sorted(M.index)); ret = np.array([i in ids476 for i in idx]); rem = ~ret
out = {"instruments": {}, "per_item_difficulty": {}, "discrimination": {}}
for name, v in cap.items():
    v = v.reindex(idx).to_numpy(); a_r, a_m = v[ret].mean(), v[rem].mean()
    if set(np.unique(v)) <= {0.0, 1.0}:
        tab = [[int(v[ret].sum()), int((1 - v[ret]).sum())], [int(v[rem].sum()), int((1 - v[rem]).sum())]]; p = float(fisher_exact(tab)[1]); test = "Fisher exact"
    else:
        p = float(mannwhitneyu(v[ret], v[rem], alternative="two-sided").pvalue); test = "Mann-Whitney U (seed-averaged correctness)"
    out["instruments"][name] = dict(acc_full=float(v.mean()), acc_retained=float(a_r), acc_removed=float(a_m), gap=float(a_r - a_m), test=test, p=p)
    print(f"{name:28} full {v.mean():.3f} retained {a_r:.3f} removed {a_m:.3f} gap {a_r-a_m:+.3f} {test} p={p:.3f}", flush=True)
D = np.column_stack([cap[k].reindex(idx).to_numpy() for k in cap]); diff = D.mean(1)
u = mannwhitneyu(diff[ret], diff[rem], alternative="two-sided")
hard = diff < 0.5
out["per_item_difficulty"] = dict(n_instruments=len(cap), mean_retained=float(diff[ret].mean()), mean_removed=float(diff[rem].mean()), sd_retained=float(diff[ret].std(ddof=1)), sd_removed=float(diff[rem].std(ddof=1)),
                                  mannwhitney_U=float(u.statistic), p=float(u.pvalue), share_hard_retained=float(hard[ret].mean()), share_hard_removed=float(hard[rem].mean()),
                                  cliffs_delta=float((np.subtract.outer(diff[ret], diff[rem]) > 0).mean() - (np.subtract.outer(diff[ret], diff[rem]) < 0).mean()))
print(f"per-item difficulty (5 capable instruments): retained {diff[ret].mean():.3f} vs removed {diff[rem].mean():.3f}; MWU p={u.pvalue:.3f}; hard share {hard[ret].mean():.3f} vs {hard[rem].mean():.3f}", flush=True)
# item discrimination over the 19-model pool
P = np.column_stack([M.to_numpy(float)] + [cap[k].reindex(idx).to_numpy() for k in cap]); ability = P.mean(0)
disc = np.array([pearsonr(P[i], ability)[0] if P[i].std() > 0 else 0.0 for i in range(len(idx))])
u2 = mannwhitneyu(disc[ret], disc[rem], alternative="two-sided")
out["discrimination"] = dict(n_models=P.shape[1], mean_retained=float(disc[ret].mean()), mean_removed=float(disc[rem].mean()), mannwhitney_p=float(u2.pvalue), note="point-biserial r between item correctness and model ability (mean accuracy) across 14 open + 5 capable instruments")
print(f"item discrimination (19 models): retained {disc[ret].mean():.3f} vs removed {disc[rem].mean():.3f}; MWU p={u2.pvalue:.3f}", flush=True)
# the 14-model open-weights panel on its own (Appendix C quotes it; it sits near chance, hence the capable instruments above)
pd14 = M.to_numpy(float).mean(1)
u3 = mannwhitneyu(pd14[ret], pd14[rem], alternative="two-sided")
out["panel_14"] = dict(n_models=int(M.shape[1]), mean_acc_retained_476=float(pd14[ret].mean()), mean_acc_full_790=float(pd14.mean()),
                       per_pair_retained=float(pd14[ret].mean()), per_pair_removed=float(pd14[rem].mean()), mannwhitney_p=float(u3.pvalue),
                       note="per-pair difficulty = fraction of the 14 panel models answering the pair correctly")
print(f"14-model panel: mean accuracy {pd14[ret].mean():.4f} on 476 vs {pd14.mean():.4f} on 790; per-pair retained {pd14[ret].mean():.3f} vs removed {pd14[rem].mean():.3f}; MWU p={u3.pvalue:.2f}", flush=True)
json.dump(out, open(ROOT / "results/d37_difficulty_powered.json", "w"), indent=1); print("wrote results/d37_difficulty_powered.json")
