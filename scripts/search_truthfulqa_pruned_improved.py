#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from truthfulqa_pruning_utils import CV_SPLITS


def _apply_prefix_keep(df_sorted: pd.DataFrame, keep_n: int) -> pd.DataFrame:
    k = int(max(0, min(keep_n, len(df_sorted))))
    return df_sorted.iloc[:k].copy()


def _ans_frame(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """Stack answer-level rows with canonical surface10 features (10 real values)."""
    pid = df_pairs["example_id"].to_numpy()
    lg = df_pairs["length_gap"].to_numpy()
    rows_t = pd.DataFrame(
        {
            "pair_id": pid,
            "label": 1,
            "neg_lead": df_pairs["neg_lead_true"].to_numpy(),
            "neg_cnt": df_pairs["neg_cnt_true"].to_numpy(),
            "hedge_rate": df_pairs["hedge_rate_true"].to_numpy(),
            "auth_rate": df_pairs["auth_rate_true"].to_numpy(),
            "len_gap": lg,
            "word_count": df_pairs["word_count_true"].to_numpy(),
            "sent_count": df_pairs["sent_count_true"].to_numpy(),
            "avg_token_len": df_pairs["avg_token_len_true"].to_numpy(),
            "type_token": df_pairs["type_token_true"].to_numpy(),
            "punc_rate": df_pairs["punc_rate_true"].to_numpy(),
        }
    )
    rows_f = pd.DataFrame(
        {
            "pair_id": pid,
            "label": 0,
            "neg_lead": df_pairs["neg_lead_false"].to_numpy(),
            "neg_cnt": df_pairs["neg_cnt_false"].to_numpy(),
            "hedge_rate": df_pairs["hedge_rate_false"].to_numpy(),
            "auth_rate": df_pairs["auth_rate_false"].to_numpy(),
            "len_gap": lg,
            "word_count": df_pairs["word_count_false"].to_numpy(),
            "sent_count": df_pairs["sent_count_false"].to_numpy(),
            "avg_token_len": df_pairs["avg_token_len_false"].to_numpy(),
            "type_token": df_pairs["type_token_false"].to_numpy(),
            "punc_rate": df_pairs["punc_rate_false"].to_numpy(),
        }
    )
    return pd.concat([rows_t, rows_f], ignore_index=True)


def _auc_pairs(df_pairs: pd.DataFrame, seed: int) -> float:
    if len(df_pairs) < CV_SPLITS:
        return float("nan")
    ans = _ans_frame(df_pairs)
    X = ans[
        [
            "neg_lead",
            "neg_cnt",
            "hedge_rate",
            "auth_rate",
            "len_gap",
            "word_count",
            "sent_count",
            "avg_token_len",
            "type_token",
            "punc_rate",
        ]
    ].to_numpy()
    y = ans["label"].to_numpy()
    g = ans["pair_id"].to_numpy()
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed, solver="liblinear"),
    )
    proba = cross_val_predict(pipe, X, y, cv=GroupKFold(n_splits=CV_SPLITS), groups=g, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def _evaluate_train_hold(
    df_sorted: pd.DataFrame, keep_n: int, train_ids: set[int], hold_ids: set[int], profile: str, seed: int
) -> Tuple[float, float]:
    kept = _apply_prefix_keep(df_sorted, keep_n)
    kt = kept[kept["example_id"].isin(train_ids)]
    kh = kept[kept["example_id"].isin(hold_ids)]
    return _auc_pairs(kt, seed), _auc_pairs(kh, seed)


def _pair_separability_train(train_df: pd.DataFrame, seed: int) -> Dict[int, float]:
    ans = _ans_frame(train_df)
    X = ans[
        [
            "neg_lead",
            "neg_cnt",
            "hedge_rate",
            "auth_rate",
            "len_gap",
            "word_count",
            "sent_count",
            "avg_token_len",
            "type_token",
            "punc_rate",
        ]
    ].to_numpy()
    y = ans["label"].to_numpy()
    g = ans["pair_id"].to_numpy()
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed, solver="liblinear"),
    )
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    out: Dict[int, List[float]] = {}
    for i, pid in enumerate(g):
        out.setdefault(int(pid), []).append(abs(float(p[i]) - 0.5))
    return {k: float(np.mean(v)) for k, v in out.items()}


def _priority_key(df: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "negation_only":
        return 10.0 * df["negation_flag"].to_numpy(float) + (
            df["neg_cnt_true"].to_numpy(float) + df["neg_cnt_false"].to_numpy(float)
        )
    if mode == "length_only":
        return df["length_gap"].to_numpy(float)
    if mode in ("neg_len", "neg_length"):
        return 8.0 * df["negation_flag"].to_numpy(float) + 4.0 * df["length_gap"].to_numpy(float)
    return (
        10.0 * df["confounded_flag"].to_numpy(float)
        + 8.0 * df["negation_flag"].to_numpy(float)
        + 4.0 * df["length_gap"].to_numpy(float)
        + 2.0 * df["hedge_diff"].to_numpy(float)
        + 2.0 * df["authority_diff"].to_numpy(float)
    )


def _imbalance_penalty(kept: pd.DataFrame, full: pd.DataFrame) -> float:
    if len(kept) == 0:
        return 10.0
    return (
        abs(float(kept["confounded_flag"].mean()) - float(full["confounded_flag"].mean()))
        + abs(float(kept["negation_flag"].mean()) - float(full["negation_flag"].mean()))
        + abs(float(kept["length_gap"].mean()) - float(full["length_gap"].mean()))
    )


def compute_objective(
    heldout_auc: float,
    keep_n: int,
    n_full: int,
    min_keep: int,
    imb: float,
    w_heldout: float,
    w_size: float,
    w_imbalance: float,
) -> float:
    if not np.isfinite(heldout_auc):
        return 1e9
    auc_pen = abs(float(heldout_auc) - 0.5)
    size_pen = 0.0 if keep_n >= min_keep else (min_keep - keep_n) / max(1, n_full)
    return w_heldout * auc_pen + w_size * size_pen + w_imbalance * float(imb)


def search_prefix_grid(
    df: pd.DataFrame,
    mode: str,
    train_ids: set[int],
    hold_ids: set[int],
    min_keep: int,
    max_drop_fraction: float,
    target_auc: float,
    profile: str,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Sweep keep_n from n_hi (keep all) down to n_lo (minimum allowed kept).
    """
    d = df.copy()
    d["_prio"] = _priority_key(d, mode)
    df_sorted = d.sort_values(["_prio", "example_id"], ascending=[True, True]).reset_index(drop=True)
    n_full = len(df_sorted)
    min_kept_from_drop_rule = int(np.ceil(n_full * (1.0 - max_drop_fraction)))
    n_lo = max(CV_SPLITS + 1, min_keep, min_kept_from_drop_rule)
    n_hi = n_full
    if n_lo > n_hi:
        n_lo = n_hi
    results = []
    step = max(1, (n_hi - n_lo) // 80)
    for keep_n in range(n_hi, n_lo - 1, -step):
        s_auc, h_auc = _evaluate_train_hold(df_sorted, keep_n, train_ids, hold_ids, profile, seed)
        results.append(
            {
                "method": "prefix_grid",
                "mode": mode,
                "keep_n": int(keep_n),
                "search_time_auc": float(s_auc),
                "heldout_auc": float(h_auc),
                "optimism_gap": float(s_auc - h_auc) if np.isfinite(s_auc) and np.isfinite(h_auc) else float("nan"),
            }
        )
    return results


def search_negation_first_stratified(
    df: pd.DataFrame, train_ids: set[int], hold_ids: set[int], min_keep: int, profile: str, seed: int
) -> list[dict[str, Any]]:
    d = df.copy()
    d["_prio"] = 10.0 * d["negation_flag"].to_numpy(float) + (
        d["neg_cnt_true"].to_numpy(float) + d["neg_cnt_false"].to_numpy(float)
    ) + 0.25 * d["length_gap"].to_numpy(float)
    df_sorted = d.sort_values(["_prio", "example_id"], ascending=[True, True]).reset_index(drop=True)
    n_full = len(df_sorted)
    out = []
    for keep_n in [n_full, max(min_keep, int(0.9 * n_full)), max(min_keep, int(0.8 * n_full)), min_keep]:
        s_auc, h_auc = _evaluate_train_hold(df_sorted, keep_n, train_ids, hold_ids, profile, seed)
        out.append(
            {
                "method": "negation_first",
                "mode": "all_features",
                "keep_n": int(keep_n),
                "search_time_auc": float(s_auc),
                "heldout_auc": float(h_auc),
                "optimism_gap": float(s_auc - h_auc) if np.isfinite(s_auc) and np.isfinite(h_auc) else float("nan"),
            }
        )
    return out


def search_feature_balanced(
    df: pd.DataFrame, train_ids: set[int], hold_ids: set[int], min_keep: int, profile: str, seed: int
) -> list[dict[str, Any]]:
    d = df.copy()
    d["len_bin"] = pd.qcut(d["length_gap"], q=4, labels=False, duplicates="drop")
    parts = []
    for b in sorted(d["len_bin"].dropna().unique()):
        sub = d[d["len_bin"] == b].sort_values(["negation_flag", "length_gap", "example_id"])
        parts.append(sub)
    df_sorted = pd.concat(parts, ignore_index=True) if parts else d.sort_values(["example_id"]).reset_index(drop=True)
    n_full = len(df_sorted)
    out = []
    for keep_n in [n_full, max(min_keep, int(0.9 * n_full)), max(min_keep, int(0.8 * n_full)), min_keep]:
        s_auc, h_auc = _evaluate_train_hold(df_sorted, keep_n, train_ids, hold_ids, profile, seed)
        out.append(
            {
                "method": "feature_balanced",
                "mode": "all_features",
                "keep_n": int(keep_n),
                "search_time_auc": float(s_auc),
                "heldout_auc": float(h_auc),
                "optimism_gap": float(s_auc - h_auc) if np.isfinite(s_auc) and np.isfinite(h_auc) else float("nan"),
            }
        )
    return out


def search_score_greedy(
    df: pd.DataFrame, train_ids: set[int], hold_ids: set[int], min_keep: int, profile: str, seed: int
) -> list[dict[str, Any]]:
    train_df = df[df["example_id"].isin(train_ids)].copy()
    sep = _pair_separability_train(train_df, seed)
    d = df.copy()
    d["_sep"] = d["example_id"].map(sep).fillna(0.0)
    df_sorted = d.sort_values(["_sep", "example_id"], ascending=[True, True]).reset_index(drop=True)
    n_full = len(df_sorted)
    out = []
    for keep_n in [n_full, max(min_keep, int(0.9 * n_full)), max(min_keep, int(0.8 * n_full)), min_keep]:
        s_auc, h_auc = _evaluate_train_hold(df_sorted, keep_n, train_ids, hold_ids, profile, seed)
        out.append(
            {
                "method": "score_greedy",
                "mode": "all_features",
                "keep_n": int(keep_n),
                "search_time_auc": float(s_auc),
                "heldout_auc": float(h_auc),
                "optimism_gap": float(s_auc - h_auc) if np.isfinite(s_auc) and np.isfinite(h_auc) else float("nan"),
            }
        )
    return out


def search_beam_multi_start(
    df: pd.DataFrame, train_ids: set[int], hold_ids: set[int], min_keep: int, profile: str, seed: int, n_starts: int = 5
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n_full = len(df)
    out = []
    for sid in range(n_starts):
        d = df.copy()
        base = _priority_key(d, "all_features")
        d["_prio"] = base + rng.normal(0.0, 1e-6, len(d))
        df_sorted = d.sort_values(["_prio", "example_id"], ascending=[True, True]).reset_index(drop=True)
        for keep_n in [n_full, max(min_keep, int(0.9 * n_full)), max(min_keep, int(0.8 * n_full)), min_keep]:
            s_auc, h_auc = _evaluate_train_hold(df_sorted, keep_n, train_ids, hold_ids, profile, seed + sid)
            out.append(
                {
                    "method": "multi_start",
                    "mode": "all_features",
                    "start_id": sid,
                    "keep_n": int(keep_n),
                    "search_time_auc": float(s_auc),
                    "heldout_auc": float(h_auc),
                    "optimism_gap": float(s_auc - h_auc) if np.isfinite(s_auc) and np.isfinite(h_auc) else float("nan"),
                }
            )
    return out


def enrich_search_rows(
    all_rows: list[dict[str, Any]],
    df: pd.DataFrame,
    train_ids: set[int],
    profile: str,
    seed: int,
    min_keep: int,
    w_heldout: float,
    w_size: float,
    w_imbalance: float,
    full_dataset_auc: float,
) -> list[dict[str, Any]]:
    n_full = len(df)
    enriched: list[dict[str, Any]] = []
    for r in all_rows:
        keep_n = int(r["keep_n"])
        mode = r.get("mode", "all_features")
        if r["method"] == "score_greedy":
            train_df = df[df["example_id"].isin(train_ids)].copy()
            sep = _pair_separability_train(train_df, seed)
            d = df.copy()
            d["_sep"] = d["example_id"].map(sep).fillna(0.0)
            d_sorted = d.sort_values(["_sep", "example_id"], ascending=[True, True]).reset_index(drop=True)
        elif r["method"] == "feature_balanced":
            d = df.copy()
            d["len_bin"] = pd.qcut(d["length_gap"], q=4, labels=False, duplicates="drop")
            parts = [d[d["len_bin"] == b].sort_values(["negation_flag", "length_gap"]) for b in sorted(d["len_bin"].dropna().unique())]
            d_sorted = pd.concat(parts, ignore_index=True) if parts else d.sort_values("example_id").reset_index(drop=True)
        elif r["method"] == "multi_start":
            sid = int(r.get("start_id", 0))
            base_prio = _priority_key(df, "all_features")
            noise = (sid + 1) * 1e-9 * np.arange(len(df), dtype=float)
            d = df.copy()
            d["_prio"] = base_prio + noise
            d_sorted = d.sort_values(["_prio", "example_id"], ascending=[True, True]).reset_index(drop=True)
        elif r["method"] == "negation_first":
            d = df.copy()
            d["_tox"] = d["negation_flag"].astype(float) * 1e6 + (d["neg_cnt_true"].astype(float) + d["neg_cnt_false"].astype(float))
            d_sorted = d.sort_values(["_tox", "example_id"], ascending=[True, True]).reset_index(drop=True)
        else:
            d = df.copy()
            d["_prio"] = _priority_key(d, mode if mode in ("negation_only", "length_only", "neg_len", "all_features") else "all_features")
            d_sorted = d.sort_values(["_prio", "example_id"], ascending=[True, True]).reset_index(drop=True)
        kept = _apply_prefix_keep(d_sorted, keep_n)
        dropped = df[~df["example_id"].isin(kept["example_id"])].copy()
        imb = _imbalance_penalty(kept, df)
        obj = compute_objective(
            float(r["heldout_auc"]),
            keep_n,
            n_full,
            min_keep,
            imb,
            w_heldout,
            w_size,
            w_imbalance,
        )
        enriched.append(
            {
                **r,
                "audit_profile": profile,
                "full_dataset_auc": full_dataset_auc,
                "retained_count": int(len(kept)),
                "dropped_count": int(len(dropped)),
                "retained_confounded_fraction": float(kept["confounded_flag"].mean()) if len(kept) else 0.0,
                "retained_negation_rate": float(kept["negation_flag"].mean()) if len(kept) else 0.0,
                "retained_length_gap_mean": float(kept["length_gap"].mean()) if len(kept) else 0.0,
                "retained_hedge_diff_mean": float(kept["hedge_diff"].mean()) if len(kept) else 0.0,
                "retained_authority_diff_mean": float(kept["authority_diff"].mean()) if len(kept) else 0.0,
                "retained_label_balance": 0.5,
                "imbalance_penalty": imb,
                "objective_score": obj,
            }
        )
    return enriched

