#!/usr/bin/env python3
"""
SUITE B common library: construction-stratified and conditional leakage.

Reuses the repo's own implementations wherever they exist:
  truthfulqa_pruning_utils.load_candidates_with_features  (pair frame + surface10 feats)
  audit_subset_evaluator.df_pairs_from_ids                (theta subset selection)
  audit_subset_evaluator.evaluate_subset_grouped_cv[_detailed]  (baseline analysis 1)
  audit_subset_evaluator._lr_pipeline                     (StandardScaler + liblinear LR)
  search_truthfulqa_pruned_improved._ans_frame            (answer-level stacking)
  truthfulqa_pruning_utils.CV_SPLITS                      (= 5)

REIMPLEMENTED here (no repo equivalent exists), stated explicitly per audit rule 4:
  * leave-one-category-out CV loop  (repo only ever does GroupKFold(n_splits=5) on pair_id)
  * paired delta-feature design      (repo has no pair-level delta classifier)
  * category x surface6 interaction design matrices
  * question-gated interaction-only design matrices
  * cluster (pair-level) bootstrap CIs (repo has no bootstrap CI utility for the audit probe)
All reimplementations reuse `_ans_frame` for feature extraction and `_lr_pipeline`
for the classifier, so the feature values and the estimator are the repo's, not mine.

Seeds: SEED_MODEL=42 (matches the canonical manifest seed), SEED_BOOT=12345,
SEED_SVD=42.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

import audit_subset_evaluator as ase  # noqa: E402
from surface_features_text import FEATURE_COLS_SURFACE6  # noqa: E402
from truthfulqa_pruning_utils import CV_SPLITS, load_candidates_with_features  # noqa: E402
from search_truthfulqa_pruned_improved import _ans_frame  # noqa: E402

SEED_MODEL = 42
SEED_BOOT = 12345
SEED_SVD = 42
B_BOOT = 2000

SURFACE6: List[str] = list(FEATURE_COLS_SURFACE6)
THETA053 = ROOT / "data" / "subsets" / "TruthfulQA-Audited" / "surface6" / "pair_ids" / "pair_ids_theta053.json"


# ----------------------------------------------------------------------------- data
def load_datasets() -> Dict[str, pd.DataFrame]:
    """Return {'TQA-790': full pair frame, 'TQA-476': theta=0.53 retained pairs}."""
    full = load_candidates_with_features(ROOT / "TruthfulQA.csv",
                                         ROOT / "audits" / "truthfulqa_style_audit.csv")
    ids = json.load(open(THETA053))["pair_ids"]
    sub = ase.df_pairs_from_ids(full, ids)
    return {"TQA-790": full, "TQA-476": sub}


def answer_frame(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """Answer-level frame (2 rows/pair) via the repo's _ans_frame, plus Category+question.

    _ans_frame emits pair_id / label / the 10 surface features. We attach the
    pair-level metadata (Category, question) by joining on pair_id.
    """
    ans = _ans_frame(df_pairs).copy()
    meta = df_pairs[["example_id", "Category", "question"]].rename(columns={"example_id": "pair_id"})
    ans = ans.merge(meta, on="pair_id", how="left", validate="many_to_one")
    assert ans["Category"].notna().all()
    return ans


# ----------------------------------------------------------------------------- metrics
def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with midranks for ties (matches roc_auc_score)."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def acc_at_half(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p >= 0.5).astype(int) == np.asarray(y)))


def _pair_row_matrix(pair_id: np.ndarray) -> Tuple[np.ndarray, int]:
    """Rows grouped by pair as an (n_pair, k) index matrix. Requires equal group sizes."""
    uniq, inv = np.unique(pair_id, return_inverse=True)
    n_pair = len(uniq)
    counts = np.bincount(inv, minlength=n_pair)
    k = int(counts[0])
    if not np.all(counts == k):
        raise ValueError(f"unequal rows per pair: {sorted(set(counts.tolist()))}")
    order = np.argsort(inv, kind="stable")
    return order.reshape(n_pair, k), n_pair


def cluster_bootstrap(y: np.ndarray, p: np.ndarray, pair_id: np.ndarray,
                      B: int = B_BOOT, seed: int = SEED_BOOT) -> Dict[str, float]:
    """Pair-level (cluster) bootstrap percentile CIs for AUC and accuracy.

    Resamples PAIRS with replacement (both answer rows of a pair move together),
    holding the fitted out-of-fold predictions fixed. This propagates sampling
    variability of the dataset but NOT model-refit variability -- stated as a
    limitation in the write-up.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    mat, n_pair = _pair_row_matrix(np.asarray(pair_id))
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    rng = np.random.default_rng(seed)
    aucs = np.empty(B)
    accs = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, n_pair, n_pair)
        idx = mat[pick].ravel()
        aucs[b] = fast_auc(y[idx], p[idx])
        accs[b] = correct[idx].mean()
    ok = ~np.isnan(aucs)
    return {
        "auc": fast_auc(y, p),
        "acc": float(correct.mean()),
        "auc_ci_lo": float(np.percentile(aucs[ok], 2.5)),
        "auc_ci_hi": float(np.percentile(aucs[ok], 97.5)),
        "acc_ci_lo": float(np.percentile(accs, 2.5)),
        "acc_ci_hi": float(np.percentile(accs, 97.5)),
        "n_pairs": int(n_pair),
        "n_rows": int(y.size),
        "B": int(B),
        "n_boot_valid": int(ok.sum()),
    }


# ----------------------------------------------------------------------------- rows
def row(dataset: str, analysis: str, stratum: str, res: Dict[str, float], **extra) -> Dict:
    r = {
        "dataset": dataset,
        "analysis": analysis,
        "stratum": stratum,
        "n_pairs": res.get("n_pairs"),
        "n_rows": res.get("n_rows"),
        "auc": res.get("auc"),
        "acc": res.get("acc"),
        "ci_lo": res.get("auc_ci_lo"),
        "ci_hi": res.get("auc_ci_hi"),
        "acc_ci_lo": res.get("acc_ci_lo"),
        "acc_ci_hi": res.get("acc_ci_hi"),
        "B": res.get("B"),
    }
    r.update(extra)
    return r


def write_rows(rows: Sequence[Dict], path: Path) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    print(f"wrote {path}  ({len(rows)} rows)")


# ------------------------------------------------- question representation (analysis 5)
# Identical definitions to those in cl04_qgated.py; kept here so cl05 can reuse them
# WITHOUT importing (and therefore re-executing) the cl04 script. Equivalence is
# verified numerically: cl05's observed AUC for delta|s_sxq(k=16) on TQA-476 must
# reproduce cl04's 0.6999.
def fit_q(train_text, k: int):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    Z = tf.fit_transform(train_text)
    k_eff = int(min(k, Z.shape[1] - 1))
    svd = TruncatedSVD(n_components=k_eff, random_state=SEED_SVD)
    Qm = svd.fit_transform(Z)
    sc = StandardScaler().fit(Qm)
    return tf, svd, sc, k_eff


def apply_q(tf, svd, sc, text):
    return sc.transform(svd.transform(tf.transform(text)))


def gated(S: np.ndarray, Qm: np.ndarray) -> np.ndarray:
    return np.einsum("nj,nm->njm", S, Qm).reshape(len(S), -1)
