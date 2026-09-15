#!/usr/bin/env python3
"""
CANONICAL Surface6 audit library.

Unlike audit_lib.py (screening), this imports the paper's RELEASED feature
extractor, scripts/surface_features_text.py, so every number is directly
comparable to the published table.

Validated: reproduces AUC 0.7145 on audits/truthfulqa_style_audit.csv (790
pairs), against the paper's reported 0.715.

Protocol: StandardScaler + LogisticRegression(liblinear, max_iter=2000,
seed 42), GroupKFold(5) grouped by question, pooled out-of-fold AUC.

Nulls: 'within_group' permutes labels inside each group; 'global' permutes
across all rows. Groups that are singletons or label-constant make the
within-group null degenerate, so this library picks the null automatically
and RECORDS which was used. Also reports cluster-bootstrap 95% CI.
"""
from __future__ import annotations
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]   # repository root (this file lives in audits/candidate_screen_2026-09-03/)
sys.path.insert(0, str(REPO / "scripts"))
from surface_features_text import extract_surface10, FEATURE_COLS_SURFACE6  # noqa: E402

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from datasets import load_dataset  # noqa: E402,F401  (re-exported for loaders)

SEED = 42
CV_SPLITS = 5
S6 = list(FEATURE_COLS_SURFACE6)


def featurize(texts) -> pd.DataFrame:
    """Canonical Surface6 features, from the paper's released extractor."""
    return pd.DataFrame([{k: extract_surface10(t)[k] for k in S6} for t in texts])


def _pipe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=SEED, solver="liblinear"),
    )


def _oof(X, y, g):
    return cross_val_predict(
        _pipe(), X, y, cv=GroupKFold(n_splits=CV_SPLITS), groups=g, method="predict_proba"
    )[:, 1]


def _auc(X, y, g):
    p = _oof(X, y, g)
    return float(roc_auc_score(y, p)), float(np.mean((p >= 0.5) == y)), p


def _within_group_permute(y, g, rng):
    y2 = y.copy()
    order = np.argsort(g, kind="stable")
    gs = g[order]
    bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        y2[idx] = y[rng.permutation(idx)]
    return y2


def audit(rows: pd.DataFrame, name: str, fmt: str, B: int = 200,
          null: str = "auto", n_boot: int = 1000, extra: dict | None = None) -> dict:
    """rows: DataFrame with columns text, label, group."""
    R = rows.dropna(subset=["text", "label", "group"]).copy()
    R["text"] = R["text"].astype(str)
    R["label"] = R["label"].astype(int)
    y = R["label"].to_numpy()
    g = R["group"].astype(str).to_numpy()
    n_groups = len(set(g))
    if y.min() == y.max():
        return {"name": name, "error": "single-class labels"}
    if n_groups < CV_SPLITS:
        return {"name": name, "error": f"only {n_groups} groups, need >= {CV_SPLITS}"}

    F = featurize(R["text"].tolist())
    X = F.to_numpy()
    t0 = time.time()
    auc, acc, proba = _auc(X, y, g)

    # ---- choose the null honestly ----
    dfg = pd.DataFrame({"y": y, "g": g})
    sizes = dfg.groupby("g").size()
    nunique = dfg.groupby("g")["y"].nunique()
    frac_permutable = float((nunique > 1).mean())
    if null == "auto":
        null_type = "within_group" if frac_permutable >= 0.5 else "global"
    else:
        null_type = null

    rng = np.random.default_rng(SEED)
    nulls = []
    for _ in range(B):
        yb = rng.permutation(y) if null_type == "global" else _within_group_permute(y, g, rng)
        if yb.min() == yb.max():
            continue
        nulls.append(_auc(X, yb, g)[0])
    nulls = np.array(nulls)
    exceed = int(np.sum(nulls >= auc))
    p = float((1 + exceed) / (1 + len(nulls))) if len(nulls) else None

    # ---- cluster bootstrap over groups ----
    groups = np.array(sorted(set(g)))
    gi = {k: np.flatnonzero(g == k) for k in groups}
    boots = []
    rb = np.random.default_rng(SEED)
    for _ in range(n_boot):
        pick = rb.choice(len(groups), len(groups), replace=True)
        idx = np.concatenate([gi[groups[j]] for j in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], proba[idx]))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None

    dirs = {}
    for f in S6:
        a, b_ = F.loc[y == 1, f], F.loc[y == 0, f]
        sd = F[f].std(ddof=1) or 1.0
        dirs[f] = {"mean_pos": float(a.mean()), "mean_neg": float(b_.mean()),
                   "d": float((a.mean() - b_.mean()) / sd)}
    # word-count-only baseline: the single most important control
    wc = F[["word_count"]].to_numpy()
    wc_auc = float(roc_auc_score(y, _oof(wc, y, g)))

    out = {
        "name": name, "format": fmt, "implementation": "canonical surface_features_text.extract_surface10",
        "n_rows": int(len(y)), "n_groups": int(n_groups), "pos_rate": float(y.mean()),
        "median_group_size": float(sizes.median()), "frac_groups_both_labels": frac_permutable,
        "auc": auc, "auc_ci95": ci, "accuracy": acc, "word_count_only_auc": wc_auc,
        "perm_B": int(len(nulls)), "null_type": null_type, "null_exceedances": exceed, "perm_p": p,
        "null_mean": float(nulls.mean()) if len(nulls) else None,
        "null_sd": float(nulls.std(ddof=1)) if len(nulls) > 1 else None,
        "feature_directions": dirs, "seconds": round(time.time() - t0, 1),
    }
    if extra:
        out.update(extra)
    return out


def paired_frac_pos_longer(rows: pd.DataFrame) -> float | None:
    """For paired data: fraction of groups where the positive text is longer."""
    R = rows.copy()
    R["wc"] = [extract_surface10(t)["word_count"] for t in R["text"].astype(str)]
    piv = R.pivot_table(index="group", columns="label", values="wc", aggfunc="first")
    if 0 not in piv.columns or 1 not in piv.columns:
        return None
    piv = piv.dropna()
    return float((piv[1] > piv[0]).mean()) if len(piv) else None


def mc_rows(qid, options, correct_idx):
    lens = [len(str(o).split()) for o in options]
    lmax = max(lens) if lens else 0
    return [{"text": o, "label": int(i == correct_idx), "group": qid,
             "is_longest": int(lens[i] == lmax and lens.count(lmax) == 1)}
            for i, o in enumerate(options)]


def run(loaders: dict, out_path: str, only=None):
    only = only or list(loaders)
    results = json.load(open(out_path)) if Path(out_path).exists() else {}
    for name in only:
        if name in results and "auc" in results[name]:
            print("skip cached", name); continue
        print("=" * 80); print(name, flush=True)
        try:
            rows, fmt = loaders[name]()
            B = 200 if len(rows) <= 20000 else 50
            res = audit(rows, name, fmt, B)
            if fmt.startswith("paired"):
                res["frac_pos_longer"] = paired_frac_pos_longer(rows)
        except Exception as e:
            res = {"name": name, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
        results[name] = res
        json.dump(results, open(out_path, "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k not in ("feature_directions", "trace")}, indent=1), flush=True)
    print("DONE")
