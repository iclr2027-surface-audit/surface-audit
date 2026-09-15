#!/usr/bin/env python3
"""Table 2 random-subset rows: Surface6 grouped-CV accuracy/AUC and 14-model ranking fidelity for ten
random subsets at each matched N (408, 476, 536, 588), plus the Audit-Prune rows for reference.

Random rule (as in the original run): np.random.default_rng(seed).choice(790, N, replace=False), seeds 0-9.
Checks itself against results/table2_random_rows.json.
"""
import glob, json, pathlib, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from truthfulqa_pruning_utils import load_candidates_with_features
from audit_subset_evaluator import df_pairs_from_ids
from run_aflite_baseline import audit6
full = load_candidates_with_features(ROOT / "TruthfulQA.csv", ROOT / "audits/truthfulqa_style_audit.csv")
files = sorted(glob.glob(str(ROOT / "data/predictions/model_predictions_*seed42*"))) + [str(ROOT / "data/predictions/model_predictions_pythia.csv")]
M = pd.concat([pd.read_csv(f) for f in files]).pivot_table(index="pair_id", columns="model_name", values="correct", aggfunc="first").sort_index()
assert M.shape == (790, 14), M.shape
def fidelity(ids):
    m = M.index.isin(ids); af, a4 = M.to_numpy(float).mean(0), M.to_numpy(float)[m].mean(0)
    return float(spearmanr(af, a4).correlation), float(kendalltau(af, a4).correlation)
ref = json.load(open(ROOT / "results/table2_random_rows.json"))
ap = json.load(open(ROOT / "data/subsets/TruthfulQA-Audited/surface6/pair_ids/pair_ids_theta053.json"))["pair_ids"]
auc, acc = audit6(df_pairs_from_ids(full, set(ap)), 42); rho, tau = fidelity(ap)
print(f"Audit-Prune 476: acc {acc:.4f} auc {auc:.4f} rho {rho:.4f} tau {tau:.4f}   (paper: 0.522 / 0.528 / 0.915)")
assert abs(auc - 0.5283) < 5e-4 and abs(rho - 0.9151) < 5e-4
out = {}
for N in ref["meta"]["sizes"]:
    vals = []
    for s in range(10):
        ids = sorted(int(x) for x in np.random.default_rng(s).choice(790, N, replace=False))
        a, c = audit6(df_pairs_from_ids(full, set(ids)), 42); r, t = fidelity(ids)
        vals.append((c, a, r, t)); exp = ref["random"][str(s)][str(N)]
        assert abs(a - exp["surface6_auc"]) < 2e-3 and abs(r - exp["rho"]) < 2e-3, (N, s, a, exp)
    v = np.array(vals); out[N] = {"acc": (v[:, 0].mean(), v[:, 0].std(ddof=1)), "auc": (v[:, 1].mean(), v[:, 1].std(ddof=1)), "rho": (v[:, 2].mean(), v[:, 2].std(ddof=1))}
    print(f"N={N}: Random acc {out[N]['acc'][0]:.3f} ({out[N]['acc'][1]:.3f})  auc {out[N]['auc'][0]:.3f} ({out[N]['auc'][1]:.3f})  rho {out[N]['rho'][0]:.3f} ({out[N]['rho'][1]:.3f})")
print("all random rows reproduce results/table2_random_rows.json")
