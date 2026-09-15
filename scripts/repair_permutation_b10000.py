#!/usr/bin/env python3
"""D7 — every permutation p-value in the paper regenerated from ONE pipeline at B=10,000 (approved 2026-08-25).

Covers: Table cross_dataset_surface6 (14 printed rows incl. TruthfulQA-790 and TruthfulQA-476), the TQA feature-group
ablation (Table 7) and the BoolQ / HaluEval / VitaminC feature-group ablations (appendix). FeverSymmetric is skipped:
its row is commented out in the tex and its t4 entry is degenerate (AUC 0.500, CI [0.500,0.500]).
Protocol (as the caption declares): per-text SURFACE6 features from scripts/surface_features_text.extract_surface10;
StandardScaler + LogisticRegression(liblinear, C=1.0, max_iter=2000, seed 42); GroupKFold(5) for paired/grouped data,
StratifiedKFold(5, shuffle, seed 42) otherwise; null = within-group label swap (paired/grouped) or plain shuffle (unpaired);
p = (1 + #{null >= observed}) / (B + 1)  [add-one rule]; cluster bootstrap 95% CI (n=1000) on groups / rows.
SANITY: each dataset's OOF AUC is compared with the published t4 value before any p-value is trusted.
Writes results/r5_d7_permutation_b10000.{json,csv} incrementally + r5_null_<dataset>[_<ablation>].npy
"""
import os, sys, json, time, platform
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))
from surface_features_text import extract_surface10   # canonical lexicons
import repair_common as C
OUT = ROOT / "results"
B_MAIN = int(os.environ.get("B_MAIN", "10000")); B_ABL = int(os.environ.get("B_ABL", "2000")); N_JOBS = int(os.environ.get("N_JOBS", "10"))
SEED = 42; FEATS = ["neg_lead", "neg_cnt", "hedge_rate", "word_count", "avg_token_len", "type_token"]
ABL = {"Full SURFACE6": FEATS, "No negation": ["hedge_rate", "word_count", "avg_token_len", "type_token"],
       "No hedging": ["neg_lead", "neg_cnt", "word_count", "avg_token_len", "type_token"],
       "No length+regularity": ["neg_lead", "neg_cnt", "hedge_rate"]}
T4 = {r["dataset_id"]: r for r in json.load(open("results/t4_cross_dataset_surface6.json"))["rows"]}

# ------------------------------------------------------------------ loaders -> DataFrame(text, label, group|None)
def _hf(*a, **k):
    from datasets import load_dataset; return load_dataset(*a, **k)
def tqa(key):
    af = C.answer_frame(C.load_datasets()[key]); return pd.DataFrame({"label": af["label"].astype(int), "group": af["pair_id"]}), af[FEATS].to_numpy(float)
def nli(hf, split):
    rows = [{"text": str(e["hypothesis"]).strip(), "label": 1 if int(e["label"]) == 0 else 0} for e in _hf(hf, split=split) if int(e["label"]) in (0, 2)]
    return pd.DataFrame(rows)
def anli():
    rows = []
    for sp in ("dev_r1", "dev_r2", "dev_r3"):
        rows += [{"text": str(e["hypothesis"]).strip(), "label": 1 if int(e["label"]) == 0 else 0} for e in _hf("facebook/anli", split=sp) if int(e["label"]) in (0, 2)]
    return pd.DataFrame(rows)
def rte():
    return pd.DataFrame([{"text": str(e["hypothesis"]).strip(), "label": 1 if int(e["label"]) == 0 else 0} for e in _hf("aps/super_glue", "rte", split="validation")])
def multirc():
    return pd.DataFrame([{"group": e["idx"]["paragraph"] * 1000 + e["idx"]["question"], "text": (str(e["question"]).strip() + " " + str(e["answer"]).strip()), "label": int(e["label"])} for e in _hf("aps/super_glue", "multirc", split="validation")])
def openbookqa():
    rows = []
    for i, e in enumerate(_hf("allenai/openbookqa", "main", split="validation")):
        q = str(e["question_stem"]).strip(); gold = str(e["answerKey"]).strip()
        rows += [{"group": i, "text": (q + " " + str(t)).strip(), "label": int(str(l).strip() == gold)} for l, t in zip(e["choices"]["label"], e["choices"]["text"])]
    return pd.DataFrame(rows)
def selfcheck():
    rows = []
    for i, e in enumerate(_hf("potsawee/wiki_bio_gpt3_hallucination", split="evaluation")):
        rows += [{"group": i, "text": str(s).strip(), "label": 1 if a == "accurate" else 0} for s, a in zip(e["gpt3_sentences"], e["annotation"])]
    return pd.DataFrame(rows)
def piqa():
    rows = []
    for i, e in enumerate(_hf("lighteval/piqa", split="validation")):
        rows += [{"group": i, "text": (str(e["goal"]).strip() + " " + str(s).strip()).strip(), "label": int(j == int(e["label"]))} for j, s in enumerate((e["sol1"], e["sol2"]))]
    return pd.DataFrame(rows)
def boolq():
    return pd.DataFrame([{"text": str(e["question"]).strip(), "label": int(bool(e["answer"]))} for e in _hf("google/boolq", split="validation")])
def halueval():
    rows = []
    for i, e in enumerate(_hf("pminervini/HaluEval", "qa", split="data")):
        rows += [{"group": i, "text": str(e["right_answer"]).strip(), "label": 1}, {"group": i, "text": str(e["hallucinated_answer"]).strip(), "label": 0}]
    return pd.DataFrame(rows)
def fever():
    rows = []
    for line in open("data/fever/shared_task_dev.jsonl"):
        j = json.loads(line); l = j.get("label") or ""
        if l in ("SUPPORTS", "REFUTES"): rows.append({"text": j["claim"], "label": int(l == "SUPPORTS")})
    return pd.DataFrame(rows)
def vitaminc():
    return pd.DataFrame([{"text": str(e["claim"]), "label": int(e["label"] == "SUPPORTS"), "group": str(e["case_id"])} for e in _hf("tals/vitaminc", split="validation") if e["label"] in ("SUPPORTS", "REFUTES")])

DATASETS = [  # (t4 id, loader, expected n)  smallest first; TQA first for the sanity anchor
    ("TruthfulQA", "tqa790", 1580), ("TruthfulQA-Audited", "tqa476", 952), ("RTE", rte, 277), ("SelfCheckGPT", selfcheck, 1908),
    ("OpenBookQA", openbookqa, 2000), ("ANLI", anli, 2132), ("BoolQ", boolq, 3270), ("PIQA", piqa, 3676), ("MultiRC", multirc, 4848),
    ("SNLI", lambda: nli("stanfordnlp/snli", "validation"), 6607), ("MultiNLI", lambda: nli("nyu-mll/multi_nli", "validation_matched"), 6692),
    ("FEVER", fever, 13332), ("HaluEval QA", halueval, 20000), ("VitaminC", vitaminc, 54012)]
ABL_ON = {"TruthfulQA", "BoolQ", "HaluEval QA", "VitaminC"}

# ------------------------------------------------------------------ audit core
def pipe(): return make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", C=1.0, max_iter=2000, penalty="l2", random_state=SEED))
def cv_for(groups): return (GroupKFold(n_splits=5), {"groups": groups}) if groups is not None else (StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED), {})
def oof(X, y, groups):
    cv, kw = cv_for(groups); return cross_val_predict(pipe(), X, y, cv=cv, method="predict_proba", **kw)[:, 1]
def perm_labels(y, groups, rng):
    if groups is None: return rng.permutation(y)
    yp = y.copy(); order = np.argsort(groups, kind="stable"); gs = groups[order]
    bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]; yp[idx] = y[idx][rng.permutation(b - a)]
    return yp
def null_chunk(X, y, groups, seeds):
    out = []
    for s in seeds:
        yp = perm_labels(y, groups, np.random.default_rng(s))
        if len(np.unique(yp)) < 2: out.append(0.5); continue
        out.append(float(roc_auc_score(yp, oof(X, yp, groups))))
    return out
def null_dist(X, y, groups, B, tag):
    t0 = time.time(); seeds = np.arange(B) + 1_000_000; chunks = np.array_split(seeds, max(1, min(B // 25, 400)))
    vals = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(null_chunk)(X, y, groups, c) for c in chunks)
    arr = np.concatenate([np.asarray(v) for v in vals]); print(f"      null B={B} done in {(time.time()-t0)/60:.1f} min", flush=True); return arr
def boot_ci(y, p, groups, n_boot=1000):
    rng = np.random.default_rng(SEED); units = np.unique(groups) if groups is not None else np.arange(len(y)); vals = []
    if groups is not None:
        idx_of = {g: np.flatnonzero(groups == g) for g in units}
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        idx = np.concatenate([idx_of[g] for g in pick]) if groups is not None else pick
        if len(np.unique(y[idx])) < 2: continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def audit(name, X, y, groups, B, tag, ref_auc=None):
    t0 = time.time(); p = oof(X, y, groups); auc = float(roc_auc_score(y, p)); acc = float(((p >= 0.5).astype(int) == y).mean())
    lo, hi = boot_ci(y, p, groups); null = null_dist(X, y, groups, B, tag); pv = float((1 + np.sum(null >= auc)) / (B + 1))
    rec = dict(dataset=name, ablation=tag, n=int(len(y)), groups_n=(int(len(np.unique(groups))) if groups is not None else None),
               protocol=("within_group_label_swap" if groups is not None else "plain_label_shuffle"), auc=auc, acc=acc, ci_lo=lo, ci_hi=hi,
               null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)), p_value=pv, B=B, min_attainable_p=1 / (B + 1),
               ref_auc_t4=ref_auc, auc_delta_vs_t4=(None if ref_auc is None else auc - ref_auc), seconds=round(time.time() - t0, 1))
    np.save(OUT / f"r5_null_{name.replace(' ', '_')}{'' if tag == 'Full SURFACE6' else '_' + tag.replace(' ', '_').replace('+', '_')}.npy", null)
    flag = "" if ref_auc is None else ("  OK" if abs(auc - ref_auc) <= 0.003 else f"  ** AUC differs from t4 by {auc-ref_auc:+.4f} **")
    print(f"  [{name} | {tag}] n={len(y)} AUC={auc:.4f} (t4 {ref_auc}) null={null.mean():.4f}±{null.std(ddof=1):.4f} p={pv:.5f} B={B}{flag}", flush=True)
    return rec

def dump(rows, final=False):
    meta = dict(B_MAIN=B_MAIN, B_ABL=B_ABL, n_jobs=N_JOBS, seed=SEED, add_one_rule=True, complete=final, python=sys.version.split()[0],
                numpy=np.__version__, sklearn=__import__("sklearn").__version__, platform=platform.platform(), total_seconds=round(time.time() - T0, 1))
    json.dump(dict(meta=meta, rows=rows), open(OUT / "r5_d7_permutation_b10000.json", "w"), indent=1)
    pd.DataFrame(rows).to_csv(OUT / "r5_d7_permutation_b10000.csv", index=False)

T0 = time.time(); rows = []
for name, loader, n_exp in DATASETS:
    t = time.time()
    if loader == "tqa790": df, X = tqa("TQA-790")
    elif loader == "tqa476": df, X = tqa("TQA-476")
    else:
        df = loader(); F = pd.DataFrame([extract_surface10(t_) for t_ in df["text"]]); X = F[FEATS].to_numpy(float)
    y = df["label"].to_numpy(int); groups = df["group"].to_numpy() if "group" in df.columns else None
    print(f"== {name}: n={len(y)} (expected {n_exp}) groups={None if groups is None else len(np.unique(groups))} loaded+featurized in {time.time()-t:.0f}s ==", flush=True)
    rows.append(audit(name, X, y, groups, B_MAIN, "Full SURFACE6", ref_auc=float(T4[name]["auc"]))); dump(rows)
    if name in ABL_ON:
        for tag, cols in ABL.items():
            if tag == "Full SURFACE6": continue
            Xa = X[:, [FEATS.index(c) for c in cols]]; rows.append(audit(name, Xa, y, groups, B_ABL, tag)); dump(rows)
dump(rows, final=True); print(f"\nDONE in {(time.time()-T0)/60:.0f} min", flush=True)
