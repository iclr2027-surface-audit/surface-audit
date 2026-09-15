#!/usr/bin/env python3
"""D7 addendum: VitaminC and SelfCheckGPT are GROUPED but not PAIRED (a case / item holds several claims of either label).
The original t4 run used a PLAIN label shuffle for these two (recorded in its null_protocol field) while the caption's
rule ('within-group swap for paired/grouped') implies a within-group swap. r5 applied the caption rule; this addendum
re-runs the two datasets (+ VitaminC ablations) under the plain-shuffle null so both nulls are on record.
CV is unchanged (GroupKFold on case_id / item_id); only the null changes."""
import os, sys, json, time
from pathlib import Path
import numpy as np
src = open(Path(__file__).with_name("repair_permutation_b10000.py")).read()
exec(src[:src.index("T0 = time.time(); rows = []")])          # loaders + audit core, no run
def audit_plain(name, X, y, groups, B, tag, ref_auc=None):
    t0 = time.time(); p = oof(X, y, groups); auc = float(roc_auc_score(y, p)); acc = float(((p >= 0.5).astype(int) == y).mean())
    lo, hi = boot_ci(y, p, groups); null = null_dist(X, y, None, B, tag)      # groups=None -> plain shuffle in the null only
    pv = float((1 + np.sum(null >= auc)) / (B + 1))
    rec = dict(dataset=name, ablation=tag, n=int(len(y)), groups_n=int(len(np.unique(groups))), protocol="plain_label_shuffle (CV still grouped)",
               auc=auc, acc=acc, ci_lo=lo, ci_hi=hi, null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)), p_value=pv, B=B,
               min_attainable_p=1/(B+1), ref_auc_t4=ref_auc, seconds=round(time.time()-t0, 1))
    np.save(OUT / f"r5b_null_{name.replace(' ','_')}_{tag.replace(' ','_').replace('+','_')}_plain.npy", null)
    print(f"  [{name} | {tag} | plain-shuffle null] AUC={auc:.4f} null={null.mean():.4f}±{null.std(ddof=1):.4f} p={pv:.5f} B={B}", flush=True); return rec
T0 = time.time(); rows = []
for name, loader in (("SelfCheckGPT", selfcheck), ("VitaminC", vitaminc)):
    df = loader(); F = pd.DataFrame([extract_surface10(t_) for t_ in df["text"]]); X = F[FEATS].to_numpy(float)
    y = df["label"].to_numpy(int); groups = df["group"].to_numpy()
    rows.append(audit_plain(name, X, y, groups, B_MAIN, "Full SURFACE6", ref_auc=float(T4[name]["auc"])))
    if name == "VitaminC":
        for tag, cols in ABL.items():
            if tag != "Full SURFACE6": rows.append(audit_plain(name, X[:, [FEATS.index(c) for c in cols]], y, groups, B_ABL, tag))
json.dump(dict(meta=dict(B_MAIN=B_MAIN, B_ABL=B_ABL, note="plain-shuffle null for grouped-not-paired datasets; CV grouped"), rows=rows),
          open(OUT / "r5b_d7_plain_shuffle_addendum.json", "w"), indent=1)
print(f"DONE in {(time.time()-T0)/60:.1f} min", flush=True)
