#!/usr/bin/env python3
"""A8 numbers (read-only): (1) per-feature LOO ablation on TruthfulQA across 10 shuffled grouped partitions (mean +/- sd of dAcc/dAUC);
(2) HaluEval QA word counts: hallucinated vs accurate answers, mean gap and Cohen's d."""
import os, sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import reproduce_truthfulqa476 as RP
OUT = ROOT / "results"

af = RP.answer_frame(); y = af["label"].to_numpy(int); g = af["pair_id"].to_numpy()
def run(feats, seed):
    X = af[feats].to_numpy(float)
    cv = GroupKFold(5) if seed is None else GroupKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(RP.pipe(), X, y, cv=cv, groups=g, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p)), float(((p >= 0.5).astype(int) == y).mean())
rows = {}
for drop in [None] + RP.FEATS:
    feats = [f for f in RP.FEATS if f != drop]
    canon = run(feats, None); fresh = np.array([run(feats, s) for s in range(10)])
    rows[drop or "FULL"] = dict(auc=canon[0], acc=canon[1], auc_fresh_mean=fresh[:, 0].mean(), auc_fresh_sd=fresh[:, 0].std(ddof=1), acc_fresh_mean=fresh[:, 1].mean(), acc_fresh_sd=fresh[:, 1].std(ddof=1))
full = rows["FULL"]
for k, r in rows.items():
    r["dAUC"] = r["auc"] - full["auc"]; r["dAcc"] = r["acc"] - full["acc"]
    r["dAUC_fresh_mean"] = r["auc_fresh_mean"] - full["auc_fresh_mean"]; r["dAcc_fresh_mean"] = r["acc_fresh_mean"] - full["acc_fresh_mean"]
    print(f"{k:16} auc {r['auc']:.4f} dAUC {r['dAUC']:+.4f} (fresh {r['dAUC_fresh_mean']:+.4f} sd {r['auc_fresh_sd']:.4f}) | acc {r['acc']:.4f} dAcc {r['dAcc']:+.4f} (fresh {r['dAcc_fresh_mean']:+.4f} sd {r['acc_fresh_sd']:.4f})", flush=True)

# HaluEval word counts
from run_fever_audit import load_halueval, compute_features
hd = load_halueval(); F = compute_features(hd["claim_text"]); wc = np.asarray(F["word_count"], float); lab = hd["label"].to_numpy(int)
h, a = wc[lab == 0], wc[lab == 1]
d = (h.mean() - a.mean()) / np.sqrt((h.var(ddof=1) + a.var(ddof=1)) / 2)
halu = dict(n_pairs=int(len(h)), mean_words_hallucinated=float(h.mean()), mean_words_accurate=float(a.mean()), median_h=float(np.median(h)), median_a=float(np.median(a)), sd_h=float(h.std(ddof=1)), sd_a=float(a.std(ddof=1)), cohens_d=float(d), frac_pairs_halluc_longer=float(np.mean(h > a)))
print(json.dumps(halu), flush=True)
json.dump(dict(loo=rows, halueval_words=halu), open(OUT / "a8_numbers.json", "w"), indent=1)
