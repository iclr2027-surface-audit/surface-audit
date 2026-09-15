#!/usr/bin/env python3
"""D19 — AFLite arms for the 2x2 comparison, with ensemble seeds.

Arm  aflite_bge      : AFLite (Le Bras et al. 2020, pair-respecting adaptation of scripts/run_aflite_baseline.py) on frozen
                       bge-large-en-v1.5 answer embeddings. Seeds 42 (published) + 0..4.
Arm  aflite_surface6 : the SAME AFLite procedure on the six standardised SURFACE6 features (the missing 2x2 cell:
                       AFLite's selection rule in Audit-Prune's feature space). Seeds 42 + 0..4.
For every seed the full removal trajectory (k_remove=10 per round, floor 300) is stored, so a subset of any size N is
'all pairs minus the first 790-N removed'. At N in {300, 335, 401, 408, 476, 536, 588} we report residual SURFACE6 audit
AUC/accuracy and 14-model ranking fidelity (same protocol as the paper).
Output: results/d19_aflite_arms.json"""
import json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0, str(ROOT / "scripts"))
import run_aflite_baseline as AB
from truthfulqa_pruning_utils import load_candidates_with_features
from search_truthfulqa_pruned_improved import _ans_frame
from audit_subset_evaluator import df_pairs_from_ids
from sklearn.preprocessing import StandardScaler

SEEDS = [42, 0, 1, 2, 3, 4]; NS = [300, 335, 401, 408, 476, 536, 588]; T0 = time.time()
full = load_candidates_with_features(Path("TruthfulQA.csv"), Path("audits/truthfulqa_style_audit.csv"))
all_ids = sorted(int(i) for i in full["example_id"].unique())
mc = AB.build_fidelity_panel(); models, fid, _ = AB.fidelity_fns(mc); assert len(models) == 14
Xb = np.load("artifacts/embeddings/full_X.npy"); yb = np.load("artifacts/embeddings/full_y.npy").astype(int); pb = np.load("artifacts/embeddings/full_pair_id.npy").astype(int)
ans = _ans_frame(full); Xs = StandardScaler().fit_transform(ans[AB.S6].to_numpy(float)); ys = ans["label"].to_numpy(int); ps = ans["pair_id"].to_numpy(int)
out_path = ROOT / "results" / "d19_aflite_arms.json"
out = json.load(open(out_path)) if out_path.exists() else {"meta": dict(seeds=SEEDS, sizes=NS, aflite=dict(m=64, train_frac=0.5, k_remove=10, floor=300), audit="SURFACE6 grouped 5-fold LR (run_aflite_baseline.audit6)", fidelity="14-model Spearman/Kendall vs full 790"), "arms": {}}

def evaluate(removed, seed):
    rows = []
    for n in NS:
        rm = set(removed[:len(all_ids) - n]); ids = [i for i in all_ids if i not in rm]
        auc, acc = AB.audit6(df_pairs_from_ids(full, set(ids)), 42); rho, tau = fid(ids)
        rows.append(dict(N=n, surface6_auc=round(auc, 4), surface6_acc=round(acc, 4), rho=round(rho, 4), tau=round(tau, 4)))
    return rows

for arm, (X, y, pid) in (("aflite_bge", (Xb, yb, pb)), ("aflite_surface6", (Xs, ys, ps))):
    out["arms"].setdefault(arm, {})
    for seed in SEEDS:
        key = str(seed)
        if key in out["arms"][arm]: print(f"{arm} seed {seed}: cached", flush=True); continue
        t0 = time.time(); removed = AB.aflite_trajectory(X, y, pid, all_ids, 300, 64, 0.5, 10, seed)
        rows = evaluate(removed, seed)
        out["arms"][arm][key] = dict(removed_order=[int(p) for p in removed], at_N=rows, seconds=round(time.time() - t0, 1))
        json.dump(out, open(out_path, "w"))
        print(f"{arm} seed {seed}: " + " ".join(f"N{r['N']}={r['surface6_auc']:.3f}" for r in rows) + f" | {time.time()-t0:.0f}s", flush=True)
# summary: mean +/- sd over seeds
summ = {}
for arm in out["arms"]:
    summ[arm] = {}
    for i, n in enumerate(NS):
        a = np.array([out["arms"][arm][s]["at_N"][i]["surface6_auc"] for s in out["arms"][arm]]); c = np.array([out["arms"][arm][s]["at_N"][i]["surface6_acc"] for s in out["arms"][arm]]); r = np.array([out["arms"][arm][s]["at_N"][i]["rho"] for s in out["arms"][arm]])
        summ[arm][str(n)] = dict(n_seeds=len(a), auc_mean=round(float(a.mean()), 4), auc_sd=round(float(a.std(ddof=1)), 4) if len(a) > 1 else None, acc_mean=round(float(c.mean()), 4), acc_sd=round(float(c.std(ddof=1)), 4) if len(c) > 1 else None, rho_mean=round(float(r.mean()), 4), rho_sd=round(float(r.std(ddof=1)), 4) if len(r) > 1 else None)
out["summary"] = summ; out["meta"]["elapsed_s"] = round(time.time() - T0, 1)
json.dump(out, open(out_path, "w"))
print("wrote", out_path, "|", out["meta"]["elapsed_s"], "s", flush=True)
