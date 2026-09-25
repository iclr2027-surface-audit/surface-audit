#!/usr/bin/env python3
"""Verify HaluBench contamination claims for Section 2.2.

Claims to check:
  1. 10,000 of 14,900 HaluBench test rows come verbatim from HaluEval QA with original labels.
  2. Answer-only SURFACE6 probe on that slice: AUC ~0.971.
  3. Cross-validated in-slice accuracy ~94.4%.
"""
import sys, os, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + "/scripts")
os.chdir(ROOT)

from run_fever_audit import compute_features, load_halueval

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score

from datasets import load_dataset

FEATS = ["neg_lead", "neg_cnt", "hedge_rate", "word_count", "avg_token_len", "type_token"]

print("=== Load HaluBench ===", flush=True)
hb = load_dataset("PatronusAI/HaluBench")
print(hb, flush=True)
split = list(hb.keys())[0]
df = hb[split].to_pandas()
print("columns:", df.columns.tolist(), flush=True)
print("total rows:", len(df), flush=True)
print(df["source_ds"].value_counts(), flush=True)
print("label values:", df["label"].value_counts(), flush=True)

halu = df[df["source_ds"].str.lower().str.contains("halueval")].copy()
print("HaluEval slice rows:", len(halu), flush=True)

print("=== Load original HaluEval QA ===", flush=True)
he = load_halueval()  # canonical loader: two rows per pair (right=1, hallucinated=0)
print("HaluEval audit rows:", len(he), "cols:", he.columns.tolist(), flush=True)

# Build verbatim lookup: answer text -> set of labels in original HaluEval
def norm(s):
    return " ".join(str(s).split())

he_map = {}
for t, lab in zip(he["claim_text"], he["label"]):
    he_map.setdefault(norm(t), set()).add(int(lab))

hb_lab = halu["label"].map({"PASS": 1, "FAIL": 0})
verbatim = 0
label_match = 0
label_conflict = 0
miss = 0
for ans, lab in zip(halu["answer"], hb_lab):
    key = norm(ans)
    if key in he_map:
        verbatim += 1
        if int(lab) in he_map[key]:
            label_match += 1
        else:
            label_conflict += 1
    else:
        miss += 1
print(f"verbatim answer matches: {verbatim}/{len(halu)}  label-consistent: {label_match}  conflict: {label_conflict}  missing: {miss}", flush=True)

print("=== Answer-only SURFACE6 probe on HaluBench HaluEval slice ===", flush=True)
f = compute_features(halu["answer"])
f = f.rename(columns={"avg_word_len": "avg_token_len", "type_token_ratio": "type_token"})
X = f[FEATS].values.astype(float)
y = hb_lab.values.astype(int)

# group by question so paired answers to the same question stay in one fold
groups = pd.factorize(halu["question"].map(norm))[0]
n_groups = len(set(groups))
print("unique question groups:", n_groups, flush=True)

pipe = make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=2000, solver="liblinear", random_state=42))
if n_groups < len(halu):  # paired structure exists -> GroupKFold
    cv = GroupKFold(5)
    proba = cross_val_predict(pipe, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
else:
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

auc = roc_auc_score(y, proba)
acc = accuracy_score(y, (proba >= 0.5).astype(int))
print(f"SURFACE6 answer-only CV: AUC={auc:.4f}  Acc={acc:.4f}", flush=True)

out = {
    "halubench_total_rows": int(len(df)),
    "source_ds_counts": df["source_ds"].value_counts().to_dict(),
    "halueval_slice_rows": int(len(halu)),
    "verbatim_matches": int(verbatim),
    "label_consistent": int(label_match),
    "label_conflict": int(label_conflict),
    "missing": int(miss),
    "surface6_auc": round(float(auc), 4),
    "surface6_acc": round(float(acc), 4),
    "cv": "GroupKFold5_by_question" if n_groups < len(halu) else "StratifiedKFold5",
}
json.dump(out, open(os.path.join(ROOT, "results", "halubench_contamination_results.json"), "w"), indent=2)
print(json.dumps(out, indent=1), flush=True)
print("DONE", flush=True)
