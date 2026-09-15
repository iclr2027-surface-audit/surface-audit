#!/usr/bin/env python3
"""
AFLite (Le Bras et al., 2020) as an alternative *cleaner* for paired TruthfulQA,
compared head-to-head against Audit-Prune (ours).

Why this script exists
----------------------
A reviewer may ask whether a generic, embedding-based adversarial-filtering method
would remove the same surface-form leakage that Audit-Prune targets. This script runs
a faithful, pair-respecting adaptation of AFLite and measures, on the surviving subset,
(i) residual surface-form audit AUC and (ii) 14-model ranking fidelity -- computed with
the *same* surface6 grouped-CV probe and the *same* cached per-pair model predictions
used everywhere else in the paper.

Faithful AFLite adaptation
--------------------------
* Representation: frozen BAAI/bge-large-en-v1.5 answer embeddings (artifacts/embeddings/full_X.npy),
  i.e. the strongest encoder in our classifier-family suite. No re-encoding.
* At each iteration, train an ensemble of m=64 linear classifiers on random *pair-disjoint*
  partitions (train_frac of retained pairs -> train, rest -> held-out), predicting held-out
  answers. Each answer's predictability = fraction of held-out predictions that are correct.
* Pair predictability = mean of its two answers' predictability. Greedily remove the
  k_remove most predictable pairs per round (the AFLite selection criterion), keeping pairs
  intact, down to a floor size. This yields a removal trajectory; a subset of any size N is
  the N least-predictable pairs.

Validation gates (must pass, else the comparison is not trustworthy)
--------------------------------------------------------------------
* surface6 audit: full-790 AUC ~ 0.715 ; canonical TruthfulQA-476 AUC ~ 0.528
* fidelity on 476: Spearman rho ~ 0.9151 , Kendall tau ~ 0.8268

Outputs
-------
results/aflite_comparison_surface6.json

Run:
    python scripts/run_aflite_baseline.py [--seed 42]
"""
from __future__ import annotations
import argparse, sys, os, json, csv, glob, random
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, kendalltau

from truthfulqa_pruning_utils import load_candidates_with_features, CV_SPLITS
from search_truthfulqa_pruned_improved import _ans_frame
from audit_subset_evaluator import df_pairs_from_ids

S6 = ["word_count", "neg_cnt", "avg_token_len", "type_token", "neg_lead", "hedge_rate"]
# Audit-Prune operating points from tab:t5c_sweep_with_fidelity_surface6 (N -> AUC, rho, tau)
AUDIT_PRUNE = {
    335: (0.4981, 0.9159, 0.8137), 401: (0.5091, 0.9193, 0.8203),
    408: (0.5189, 0.9248, 0.8476), 476: (0.5283, 0.9151, 0.8268),
    493: (0.5396, 0.9216, 0.8427), 492: (0.5498, 0.9239, 0.8492),
    536: (0.5556, 0.9757, 0.9162), 497: (0.5700, 0.9912, 0.9664),
    588: (0.5798, 0.9469, 0.8815),
}


def audit6(df_pairs, seed):
    ans = _ans_frame(df_pairs)
    X = ans[S6].to_numpy(); y = ans["label"].to_numpy(); g = ans["pair_id"].to_numpy()
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=seed, solver="liblinear"))
    proba = cross_val_predict(pipe, X, y, cv=GroupKFold(n_splits=CV_SPLITS),
                              groups=g, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba)), float(np.mean((proba >= 0.5) == y))


def build_fidelity_panel():
    files = [f for f in sorted(glob.glob("data/predictions/model_predictions*.csv"))
             if os.path.basename(f) != "example_model_predictions.csv"]
    by_model = defaultdict(list)
    for f in files:
        with open(f, newline="") as fh:
            row = next(csv.DictReader(fh), None)
        if row and "model_name" in row:
            by_model[row["model_name"]].append(f)
    model_correct = {}
    for mn, fs in by_model.items():
        pick = next((f for f in fs if "seed42" in os.path.basename(f)), sorted(fs)[0])
        d = {}
        with open(pick, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["model_name"] == mn:
                    d[int(r["pair_id"])] = int(r["correct"])
        model_correct[mn] = d
    return model_correct


def fidelity_fns(model_correct):
    models = sorted(model_correct)
    ids_full = sorted(set.intersection(*[set(model_correct[m]) for m in models]))
    full_acc = np.array([np.mean([model_correct[m][i] for i in ids_full]) for m in models])
    shared = set(ids_full)
    def fid(ids):
        ids = [i for i in ids if i in shared]
        sub = np.array([np.mean([model_correct[m][i] for i in ids]) for m in models])
        return float(spearmanr(full_acc, sub).statistic), float(kendalltau(full_acc, sub).statistic)
    return models, fid, ids_full


def aflite_trajectory(X, y, pid, pairs, floor_n, m, train_frac, k_remove, seed):
    rng = random.Random(seed)
    pairs = list(pairs)
    ans_by_pair = defaultdict(list)
    for ridx, p in enumerate(pid):
        ans_by_pair[int(p)].append(ridx)
    removed_order = []
    while len(pairs) > floor_n:
        correct = np.zeros(len(y)); seen = np.zeros(len(y))
        for _ in range(m):
            shuf = pairs[:]; rng.shuffle(shuf)
            n_tr = max(CV_SPLITS, int(round(train_frac * len(shuf))))
            tr_rows = [r for p in shuf[:n_tr] for r in ans_by_pair[p]]
            te_rows = [r for p in shuf[n_tr:] for r in ans_by_pair[p]]
            if len(set(y[tr_rows])) < 2 or not te_rows:
                continue
            clf = LogisticRegression(max_iter=1000, solver="liblinear",
                                     random_state=rng.randint(0, 1_000_000))
            clf.fit(X[tr_rows], y[tr_rows])
            pred = clf.predict(X[te_rows])
            for r, pr in zip(te_rows, pred):
                seen[r] += 1
                if pr == y[r]:
                    correct[r] += 1
        pa = np.where(seen > 0, correct / np.maximum(seen, 1), 0.0)
        score = {p: float(np.mean([pa[r] for r in ans_by_pair[p]])) for p in pairs}
        order = sorted(pairs, key=lambda p: (-score[p], p))
        k = min(k_remove, len(pairs) - floor_n)
        chunk = order[:k]; removed_order.extend(chunk)
        rm = set(chunk); pairs = [p for p in pairs if p not in rm]
    return removed_order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--k_remove", type=int, default=10)
    ap.add_argument("--floor", type=int, default=300)
    a = ap.parse_args()

    full = load_candidates_with_features(Path("TruthfulQA.csv"),
                                         Path("audits/truthfulqa_style_audit.csv"))
    all_ids = sorted(int(i) for i in full["example_id"].unique())

    # validation gates
    auc_full, _ = audit6(full, a.seed)
    ids476 = json.load(open("data/subsets/TruthfulQA-Audited/surface6/pair_ids/pair_ids_theta053.json"))["pair_ids"]
    auc476, _ = audit6(df_pairs_from_ids(full, set(ids476)), a.seed)
    mc = build_fidelity_panel(); models, fid, ids_shared = fidelity_fns(mc)
    rho476, tau476 = fid(ids476)
    gate1 = abs(auc_full - 0.715) < 0.02 and abs(auc476 - 0.528) < 0.01
    gate2 = abs(rho476 - 0.9151) < 0.03 and len(models) == 14
    print(f"GATE1(audit) full={auc_full:.4f} 476={auc476:.4f} -> {'PASS' if gate1 else 'FAIL'}")
    print(f"GATE2(fidelity) rho476={rho476:.4f} tau476={tau476:.4f} n_models={len(models)} -> {'PASS' if gate2 else 'FAIL'}")
    if not (gate1 and gate2):
        raise SystemExit("Validation gates failed; aborting.")

    X = np.load("artifacts/embeddings/full_X.npy")
    y = np.load("artifacts/embeddings/full_y.npy").astype(int)
    pid = np.load("artifacts/embeddings/full_pair_id.npy").astype(int)
    removed = aflite_trajectory(X, y, pid, all_ids, a.floor, a.m, a.train_frac, a.k_remove, a.seed)

    def retained_at(n):
        rm = set(removed[:len(all_ids) - n])
        return [i for i in all_ids if i not in rm]

    equal_n = []
    for n in sorted(AUDIT_PRUNE):
        ids = retained_at(n); auc, acc = audit6(df_pairs_from_ids(full, set(ids)), a.seed)
        rho, tau = fid(ids)
        ap_auc, ap_rho, ap_tau = AUDIT_PRUNE[n]
        equal_n.append({"N": n, "aflite_auc": round(auc, 4), "aflite_acc": round(acc, 4),
                        "aflite_rho": round(rho, 4), "aflite_tau": round(tau, 4),
                        "audit_prune_auc": ap_auc, "audit_prune_rho": ap_rho, "audit_prune_tau": ap_tau})
    traj = []
    for n in range(len(all_ids), a.floor - 1, -10):
        ids = retained_at(n); auc, acc = audit6(df_pairs_from_ids(full, set(ids)), a.seed)
        traj.append({"N": n, "auc": round(auc, 4), "acc": round(acc, 4)})
    min_auc = min(t["auc"] for t in traj)
    n_le_053 = max((t["N"] for t in traj if t["auc"] <= 0.53), default=None)

    out = {"meta": {"seed": a.seed, "embedding": "BAAI/bge-large-en-v1.5",
                    "aflite": {"m": a.m, "train_frac": a.train_frac, "k_remove": a.k_remove, "floor": a.floor},
                    "gates": {"auc_full": auc_full, "auc_476": auc476, "rho_476": rho476, "tau_476": tau476},
                    "n_models": len(models), "models": models},
           "equal_N": equal_n,
           "matched_auc": {"audit_prune_reaches_0.528_at_N": 476,
                           "aflite_min_auc_over_trajectory": min_auc,
                           "aflite_N_at_auc_le_0.53": n_le_053},
           "trajectory": traj}
    outp = ROOT / "results" / "aflite_comparison_surface6.json"
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nEqual-N residual surface AUC (Audit-Prune vs AFLite):")
    for r in equal_n:
        print(f"  N={r['N']:>4}  AP={r['audit_prune_auc']:.4f}  AFLite={r['aflite_auc']:.4f}")
    print(f"\nMatched-AUC: AFLite min AUC over trajectory = {min_auc:.4f}; "
          f"reaches <=0.53 at N={n_le_053}. Audit-Prune reaches 0.528 at N=476.")
    print("wrote", outp)


if __name__ == "__main__":
    main()
