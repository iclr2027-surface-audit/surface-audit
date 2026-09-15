#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import truthfulqa_paper_audit as tpa
from surface_features_text import FEATURE_COLS, extract_surface10, rel_len_gap

CV_SPLITS = 5

# Per-answer feature columns carried on the pair-level frame.
# `len_gap` is pair-level and stored once per pair (not duplicated true/false).
_PER_ANSWER_FEATS = [c for c in FEATURE_COLS if c != "len_gap"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_candidates_with_features(truthfulqa_csv: Path, audit_csv: Path) -> pd.DataFrame:
    """Load TruthfulQA pairs and compute canonical surface10 features per answer.

    Features are derived from raw text via `extract_surface10`, not from the
    audit CSV. The audit CSV is read only for the pair-level `style_violation`
    flag (used by downstream confounded/clean splitting).
    """
    tq = pd.read_csv(truthfulqa_csv)
    audit = pd.read_csv(audit_csv)
    if len(tq) != len(audit):
        raise ValueError(f"Row mismatch: TruthfulQA={len(tq)} audit={len(audit)}")

    best_col = tq["Best Answer"].fillna("").astype(str)
    incorrect_col = tq["Best Incorrect Answer"].fillna("").astype(str)
    n = len(tq)

    feats_true: dict[str, list[float]] = {f: [] for f in _PER_ANSWER_FEATS}
    feats_false: dict[str, list[float]] = {f: [] for f in _PER_ANSWER_FEATS}
    len_gaps: list[float] = []
    for i in range(n):
        t_text = best_col.iat[i]
        f_text = incorrect_col.iat[i]
        ft = extract_surface10(t_text)
        ff = extract_surface10(f_text)
        for k in _PER_ANSWER_FEATS:
            feats_true[k].append(ft[k])
            feats_false[k].append(ff[k])
        len_gaps.append(rel_len_gap(t_text, f_text))

    df = pd.DataFrame()
    df["example_id"] = np.arange(n, dtype=int)
    df["question"] = tq["Question"]
    df["answer_a"] = best_col
    df["answer_b"] = incorrect_col
    df["correct_label"] = "A"
    df["confounded_flag"] = audit["style_violation"].astype(int)

    for k in _PER_ANSWER_FEATS:
        df[f"{k}_true"] = np.asarray(feats_true[k], dtype=float)
        df[f"{k}_false"] = np.asarray(feats_false[k], dtype=float)
    df["length_gap"] = np.asarray(len_gaps, dtype=float)

    df["negation_flag"] = (
        (df["neg_lead_true"] + df["neg_lead_false"] + df["neg_cnt_true"] + df["neg_cnt_false"]) > 0
    ).astype(int)
    df["hedge_diff"] = (df["hedge_rate_true"] - df["hedge_rate_false"]).abs().astype(float)
    df["authority_diff"] = (df["auth_rate_true"] - df["auth_rate_false"]).abs().astype(float)
    df["Type"] = tq.get("Type", "")
    df["Category"] = tq.get("Category", "")
    return df


def audit_auc_subset(df: pd.DataFrame, profile: str, seed: int) -> float:
    """Grouped-CV OOF AUC for a candidate frame using canonical surface10.

    Expects either a candidate frame produced by `load_candidates_with_features`
    (with `<feat>_true` / `<feat>_false` columns + `length_gap`) or an
    audit-shaped frame already carrying COLS_TRUE_SURFACE10 / COLS_FALSE_SURFACE10
    + `len_gap`.
    """
    needs_remap = "length_gap" in df.columns and "len_gap" not in df.columns
    if needs_remap:
        cols: dict[str, np.ndarray] = {}
        for k in _PER_ANSWER_FEATS:
            cols[f"{k}_true"] = df[f"{k}_true"].to_numpy()
            cols[f"{k}_false"] = df[f"{k}_false"].to_numpy()
        cols["len_gap"] = df["length_gap"].to_numpy()
        a = pd.DataFrame(cols)
    else:
        a = df.copy()
    ans = tpa.build_answer_level_audit_frame(a, profile="surface10", copy_audit_meta=False)
    return float(tpa.paper_compatible_audit_oof_auc(ans, profile="surface10", seed=seed, n_splits=CV_SPLITS).auc_oof)

