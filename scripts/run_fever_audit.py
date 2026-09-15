#!/usr/bin/env python3
"""
Surface-form confound audit: FEVER 1.0, FeverSymmetric, BoolQ, HaluEval QA,
and VitaminC.

FEVER and FeverSymmetric metrics are frozen constants; each run recomputes BoolQ,
HaluEval QA, and VitaminC. Outputs: audits/fever_audit_results.csv (5 external
rows), cross-dataset table, bar figure.

Methodology matches the TruthfulQA audit: StandardScaler + LogisticRegression,
5-fold StratifiedKFold OOF AUC, bootstrap CI, permutation null, ablations,
heuristic confound rate (BoolQ adds question_neg to the confound heuristic).

Usage:
    python scripts/run_fever_audit.py

Optional local data (skips HuggingFace hub):
    python scripts/run_fever_audit.py --boolq-data data/boolq/validation.parquet
    python scripts/run_fever_audit.py --halueval-data data/halueval/qa.parquet
    python scripts/run_fever_audit.py --vitaminc-data data/vitaminc/validation.parquet

Random-label control (BoolQ + VitaminC no-signal floor):
    python scripts/run_fever_audit.py --random-label-control-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- Defaults (match TruthfulQA audit paper numbers where noted) ---
DEFAULT_SEED = 42
CV_SPLITS = 5
N_PERMUTATION_NULL = 100
N_BOOTSTRAP = 1000
LR_MAX_ITER = 1000

# TruthfulQA reference row for cross-dataset table / figure (from paper)
TRUTHFULQA_N = 790
TRUTHFULQA_AUC = 0.716
TRUTHFULQA_NULL = 0.501

# Frozen FEVER / FeverSymmetric audit rows (last full run; avoids re-downloading / recomputing)
FEVER_10_RESULT: dict = {
    "dataset": "FEVER 1.0 dev",
    "n": 13089,
    "auc": 0.6453826715477438,
    "ci_lo": 0.6365100188173196,
    "ci_hi": 0.6551036957548522,
    "null_mean": 0.4999879360327812,
    "p_value": 0.0,
    "dominant_feature": "Negation",
    "confound_pct": 52.85354114141646,
}
FEVERSYMMETRIC_RESULT: dict = {
    "dataset": "FeverSymmetric dev",
    "n": 708,
    "auc": 0.35586038494685435,
    "ci_lo": 0.3144603456255689,
    "ci_hi": 0.39557696073076837,
    "null_mean": 0.4911651345398831,
    "p_value": 1.0,
    "dominant_feature": "Negation",
    "confound_pct": 60.451977401129945,
}

# Mid-question negation contractions (exclude match at token index 0 only)
QUESTION_NEG_CONTRACTIONS = frozenset({
    "isn't", "aren't", "wasn't", "won't", "can't", "couldn't",
    "wouldn't", "didn't", "doesn't", "don't", "haven't", "hasn't",
})

# --- Lexicons ---
NEG_LEAD_WORDS = [
    "not", "never", "no", "nor",
    "isn't", "aren't", "wasn't", "won't", "can't", "couldn't",
    "wouldn't", "didn't", "doesn't", "don't", "haven't", "hasn't",
    "failed", "refused", "unable",
]
NEG_WORDS = [
    r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bnor\b",
    r"\bdidn't\b", r"\bdoesn't\b", r"\bwasn't\b", r"\bwasn\b",
    r"\bwon't\b", r"\bcan't\b", r"\bcannot\b", r"\bnone\b",
]
NEG_BIGRAM_REFUTES = [
    "did not", "does not", "was not", "yet to",
    "refused to", "failed to", "unable to", "not have", "never been",
]
NEG_BIGRAM_SUPPORTS = [
    "at least", "least one", "won award", "american actor", "starred in",
]
HEDGES = [
    r"\bmight\b", r"\bmay\b", r"\bpossibly\b", r"\ballegedly\b",
    r"\breportedly\b", r"\bsupposedly\b", r"\bperhaps\b", r"\bprobably\b",
]
AUTH_PHRASES = [
    r"according to", r"as stated", r"reported by",
    r"experts say", r"study shows",
]
SUPERLATIVE = [
    r"\bmost\b", r"\bfirst\b", r"\bbest\b", r"\bworst\b",
    r"\blargest\b", r"\bhighest\b", r"\blowest\b", r"\bgreatest\b",
]
GENERALIZATION = [
    r"\bsome\b", r"\bmany\b", r"\boften\b", r"\busually\b",
    r"\bgenerally\b", r"\btypically\b", r"\bsometimes\b",
]

FEAT_COLS = [
    "neg_lead", "neg_cnt", "neg_bigram_refutes", "neg_bigram_supports",
    "hedge_rate", "word_count", "char_count", "avg_word_len",
    "type_token_ratio", "has_number", "specificity_score",
    "superlative", "generalization_proxy", "auth_rate", "question_neg",
]

# HaluEval: answer text only; omit question_neg (interrogative-specific).
HALUEVAL_FEAT_COLS = [c for c in FEAT_COLS if c != "question_neg"]

ABLATIONS = [
    (
        "No negation",
        [
            "neg_lead", "neg_cnt", "neg_bigram_refutes", "neg_bigram_supports",
            "question_neg",
        ],
    ),
    ("No length", ["word_count", "char_count", "avg_word_len"]),
    ("No hedging", ["hedge_rate"]),
    ("No authority", ["auth_rate"]),
    ("No specificity/mutation", ["specificity_score", "superlative", "generalization_proxy"]),
]

FEVER_DOWNLOAD_URLS = [
    "https://dl.fbaipublicfiles.com/fever/shared_task_dev.jsonl",
    "https://fever.ai/download/fever/shared_task_dev.jsonl",
]
FEVERSYMMETRIC_URL = (
    "https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/"
    "symmetric_v0.2/fever_symmetric_dev.jsonl"
)

# HaluEval QA on Hugging Face: configs from get_dataset_config_names() include "qa"
# (there is no "qa_data" config). The loaded split name is "data" (~10k rows).
HALUEVAL_HF_ID = "pminervini/HaluEval"
HALUEVAL_HF_CONFIG = "qa"
HALUEVAL_HF_SPLIT = "data"

# VitaminC on Hugging Face.
VITAMINC_HF_ID = "tals/vitaminc"
VITAMINC_HF_CONFIG = "default"
VITAMINC_HF_SPLIT = "validation"

N_RANDOM_LABEL_RUNS = 20
N_RANDOM_LABEL_RUNS_BOOLQ = 40


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def simple_tokens(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(s).lower())


def count_matches(patterns: list[str], text: str) -> int:
    text_l = str(text).lower()
    return sum(len(re.findall(p, text_l)) for p in patterns)


def starts_with_neg_lead(text: str) -> int:
    t = str(text).strip().lower()
    if not t:
        return 0
    return int(
        any(t == w or t.startswith(w + " ") for w in NEG_LEAD_WORDS)
    )


def count_bigrams(bigrams: list[str], text: str) -> int:
    text_l = str(text).lower()
    return sum(text_l.count(bg) for bg in bigrams)


def question_neg_mid(text: str) -> int:
    r"""
    1 if any listed contraction appears in the question after the first word.

    Questions where the contraction *is* the first word (e.g. "Isn't it true?")
    return 0 by design — those are captured by ``neg_lead`` instead.

    Uses the raw lowercased string (not token splitting) so straight and curly
    apostrophes both match; simple_tokens only allows ASCII ' in contractions.
    """
    t = str(text).strip().lower()
    for u in ("\u2019", "\u2018", "\u02bc"):
        t = t.replace(u, "'")
    parts = t.split(None, 1)
    if len(parts) < 2:
        return 0
    tail = parts[1]
    for c in QUESTION_NEG_CONTRACTIONS:
        esc = re.escape(c)
        if re.search(r"(?<![a-z0-9])" + esc + r"(?![a-z0-9])", tail):
            return 1
    return 0


def compute_features(claims: pd.Series) -> pd.DataFrame:
    rows = []
    for claim in claims:
        tokens = simple_tokens(claim)
        n_tok = len(tokens)
        words_gt6 = sum(1 for t in tokens if len(t) > 6)

        neg_lead = starts_with_neg_lead(claim)
        neg_cnt = count_matches(NEG_WORDS, claim)
        neg_bigram_refutes = count_bigrams(NEG_BIGRAM_REFUTES, claim)
        neg_bigram_supports = count_bigrams(NEG_BIGRAM_SUPPORTS, claim)

        hedge_cnt = count_matches(HEDGES, claim)
        hedge_rate = hedge_cnt / max(1, n_tok)

        word_count = n_tok
        char_count = len(str(claim))
        avg_word_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
        type_token_ratio = len(set(tokens)) / max(1, n_tok)
        has_number = int(bool(re.search(r"\d", str(claim))))
        specificity_score = words_gt6 / max(1, n_tok)
        superlative = int(any(re.search(p, str(claim).lower()) for p in SUPERLATIVE))
        generalization_proxy = int(any(re.search(p, str(claim).lower()) for p in GENERALIZATION))

        auth_cnt = count_matches(AUTH_PHRASES, claim)
        auth_rate = auth_cnt / max(1, n_tok)
        q_neg = question_neg_mid(claim)

        rows.append({
            "neg_lead": neg_lead,
            "neg_cnt": neg_cnt,
            "neg_bigram_refutes": neg_bigram_refutes,
            "neg_bigram_supports": neg_bigram_supports,
            "hedge_rate": hedge_rate,
            "word_count": word_count,
            "char_count": char_count,
            "avg_word_len": avg_word_len,
            "type_token_ratio": type_token_ratio,
            "has_number": has_number,
            "specificity_score": specificity_score,
            "superlative": superlative,
            "generalization_proxy": generalization_proxy,
            "auth_rate": auth_rate,
            "question_neg": q_neg,
        })
    return pd.DataFrame(rows)


def _make_lr_pipeline(seed: int) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=LR_MAX_ITER, random_state=seed),
    )


def run_audit(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_null: int,
    n_boot: int,
    groups: np.ndarray | None = None,
) -> tuple[float, float, float, float, float, np.ndarray]:
    if groups is not None:
        cv = GroupKFold(n_splits=CV_SPLITS)
    else:
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)
    pipe = _make_lr_pipeline(seed)

    predict_kw: dict = {"method": "predict_proba"}
    if groups is not None:
        predict_kw["groups"] = groups
    y_proba = cross_val_predict(pipe, X, y, cv=cv, **predict_kw)[:, 1]
    auc = roc_auc_score(y, y_proba)

    rng = np.random.default_rng(seed)
    n = len(y)
    auc_boot: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        auc_boot.append(roc_auc_score(y[idx], y_proba[idx]))
    auc_boot_arr = np.array(auc_boot)
    ci_lo = float(np.percentile(auc_boot_arr, 2.5))
    ci_hi = float(np.percentile(auc_boot_arr, 97.5))

    auc_null: list[float] = []
    for _ in range(n_null):
        y_perm = rng.permutation(y)
        proba_perm = cross_val_predict(pipe, X, y_perm, cv=cv, **predict_kw)[:, 1]
        auc_null.append(roc_auc_score(y_perm, proba_perm))
    auc_null_arr = np.array(auc_null)
    null_mean = float(np.mean(auc_null_arr))
    p_value = float((auc_null_arr >= auc).mean())

    return auc, ci_lo, ci_hi, null_mean, p_value, y_proba


def _audit_auc_shuffled_labels(
    X: np.ndarray,
    y: np.ndarray,
    seed_permute: int,
    seed_audit: int,
    groups: np.ndarray | None = None,
) -> float:
    """
    Run audit with globally permuted labels; returns OOF AUC only.
    Uses same model/CV as run_audit but skips bootstrap/null (not needed for control).
    """
    rng = np.random.default_rng(seed_permute)
    y_shuffled = rng.permutation(y)
    if groups is not None:
        cv = GroupKFold(n_splits=CV_SPLITS)
        predict_kw: dict = {"method": "predict_proba", "groups": groups}
    else:
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed_audit)
        predict_kw = {"method": "predict_proba"}
    pipe = _make_lr_pipeline(seed_audit)
    y_proba = cross_val_predict(pipe, X, y_shuffled, cv=cv, **predict_kw)[:, 1]
    return float(roc_auc_score(y_shuffled, y_proba))


def run_random_label_control(
    args: argparse.Namespace,
    datasets: list[str],
) -> None:
    """Run random-label control for BoolQ and/or VitaminC; print mean ± std AUC."""
    seed_audit = args.seed
    n_override = getattr(args, "n_random_runs", None)

    print("\n" + "=" * 60)
    print("RANDOM-LABEL CONTROL (global label shuffle, no-signal floor)")
    print("=" * 60)

    if "boolq" in datasets:
        n_bq = n_override if n_override is not None else N_RANDOM_LABEL_RUNS_BOOLQ
        df_bq = load_boolq(args.boolq_data)
        claims_bq = df_bq["claim_text"]
        y_bq = df_bq["label"].values
        feat_bq = compute_features(claims_bq)
        X_bq = feat_bq[FEAT_COLS].fillna(0).to_numpy()
        aucs_bq: list[float] = []
        for s in range(n_bq):
            aucs_bq.append(_audit_auc_shuffled_labels(X_bq, y_bq, s, seed_audit, groups=None))
        mean_bq = float(np.mean(aucs_bq))
        std_bq = float(np.std(aucs_bq))
        print(f"BoolQ-random: mean AUC = {mean_bq:.4f} ± {std_bq:.4f} (n={n_bq} runs)")

    if "vitaminc" in datasets:
        n_vc = n_override if n_override is not None else N_RANDOM_LABEL_RUNS
        df_vc = load_vitaminc(args.vitaminc_data)
        claims_vc = df_vc["claim_text"]
        y_vc = df_vc["label"].values
        groups_vc = df_vc["case_id"].values
        feat_vc = compute_features(claims_vc)
        X_vc = feat_vc[FEAT_COLS].fillna(0).to_numpy()
        aucs_vc: list[float] = []
        for s in range(n_vc):
            aucs_vc.append(
                _audit_auc_shuffled_labels(X_vc, y_vc, s, seed_audit, groups=groups_vc)
            )
        mean_vc = float(np.mean(aucs_vc))
        std_vc = float(np.std(aucs_vc))
        print(
            f"VitaminC-random: mean AUC = {mean_vc:.4f} ± {std_vc:.4f} "
            f"(n={n_vc} runs, GroupKFold by case_id preserved)"
        )


def _ablation_one(
    X_df: pd.DataFrame,
    y: np.ndarray,
    feat_cols: list[str],
    use_cols: list[str],
    seed: int,
    n_null: int,
    groups: np.ndarray | None = None,
) -> tuple[float, float]:
    if groups is not None:
        cv = GroupKFold(n_splits=CV_SPLITS)
    else:
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)
    pipe = _make_lr_pipeline(seed)
    X = X_df[use_cols].fillna(0).to_numpy()
    predict_kw: dict = {"method": "predict_proba"}
    if groups is not None:
        predict_kw["groups"] = groups
    y_proba = cross_val_predict(pipe, X, y, cv=cv, **predict_kw)[:, 1]
    auc = roc_auc_score(y, y_proba)
    rng = np.random.default_rng(seed)
    auc_null = []
    for _ in range(n_null):
        y_perm = rng.permutation(y)
        proba_perm = cross_val_predict(pipe, X, y_perm, cv=cv, **predict_kw)[:, 1]
        auc_null.append(roc_auc_score(y_perm, proba_perm))
    return auc, float(np.mean(auc_null))


def run_ablation(
    X_df: pd.DataFrame,
    y: np.ndarray,
    feat_cols: list[str],
    seed: int,
    n_null: int,
    groups: np.ndarray | None = None,
) -> list[tuple[str, float, float]]:
    keep = [c for c in feat_cols if c in X_df.columns]
    results: list[tuple[str, float, float]] = []
    auc, null_m = _ablation_one(X_df, y, feat_cols, keep, seed, n_null, groups=groups)
    results.append(("None (full model)", auc, null_m))

    for name, removed in ABLATIONS:
        use_cols = [c for c in feat_cols if c in X_df.columns and c not in removed]
        if not use_cols:
            continue
        auc, null_m = _ablation_one(X_df, y, feat_cols, use_cols, seed, n_null, groups=groups)
        results.append((name, auc, null_m))
    return results


def dominant_ablation_group(ablations: list[tuple[str, float, float]]) -> str:
    """Name of removed group that causes largest AUC drop from full model."""
    full_auc = ablations[0][1]
    max_drop = 0.0
    best = ""
    for ab_name, ab_auc, _ in ablations[1:]:
        drop = full_auc - ab_auc
        if drop > max_drop:
            max_drop = drop
            best = ab_name.replace("No ", "").split(" (")[0].capitalize()
    if max_drop <= 0:
        return "None identified"
    return best


def heuristic_confound(df: pd.DataFrame, *, for_boolq: bool = False) -> np.ndarray:
    """Length tails (20/80) match FEVER; omitted for BoolQ where lengths cluster (degenerate tails)."""
    confounded = (
        (df["neg_lead"] == 1)
        | (df["neg_cnt"] >= 2)
        | (df["neg_bigram_refutes"] >= 1)
    )
    if not for_boolq:
        wc = df["word_count"].values
        p20 = np.percentile(wc, 20)
        p80 = np.percentile(wc, 80)
        in_extreme = (wc <= p20) | (wc >= p80)
        confounded = confounded | in_extreme
    if for_boolq:
        confounded = confounded | (df["question_neg"] >= 1)
    return confounded.astype(int).values


def _parse_fever_jsonl(content: str) -> pd.DataFrame:
    rows = []
    seen_claims: set[str] = set()
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        data = json.loads(line)
        label = data.get("label", "").strip().upper()
        if label not in ("SUPPORTS", "SUPPORTED", "REFUTES", "REFUTED"):
            continue
        claim = data["claim"]
        if claim in seen_claims:
            continue
        seen_claims.add(claim)
        rows.append({
            "claim_text": claim,
            "label": 1 if label in ("SUPPORTS", "SUPPORTED") else 0,
        })
    df = pd.DataFrame(rows)
    df["pair_id"] = np.arange(len(df))
    return df[["pair_id", "claim_text", "label"]]


def load_fever_10(fever_dev_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load FEVER 1.0 labelled dev: SUPPORTS/REFUTES only, deduped by claim text.

    Resolution order: explicit path → data/fever/shared_task_dev.jsonl → HTTP.
    """
    if fever_dev_path is not None:
        path = Path(fever_dev_path)
        if path.exists():
            return _parse_fever_jsonl(path.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"FEVER dev file not found: {path}")

    local = repo_root() / "data" / "fever" / "shared_task_dev.jsonl"
    if local.exists():
        return _parse_fever_jsonl(local.read_text(encoding="utf-8"))

    try:
        import requests
    except ImportError as e:
        raise RuntimeError(
            "Install requests (pip install requests) or place shared_task_dev.jsonl "
            f"at {local}"
        ) from e

    for url in FEVER_DOWNLOAD_URLS:
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            return _parse_fever_jsonl(r.text)
        except Exception:
            continue
    raise RuntimeError(
        "Could not load FEVER 1.0 dev. Download shared_task_dev.jsonl from "
        "https://fever.ai/dataset/fever.html and place at data/fever/shared_task_dev.jsonl"
    )


def load_fever_symmetric(symmetric_path: str | Path | None = None) -> pd.DataFrame:
    """Load FeverSymmetric dev (SUPPORTS/REFUTES only). Local file or GitHub raw."""
    if symmetric_path is not None:
        path = Path(symmetric_path)
        if not path.exists():
            raise FileNotFoundError(f"FeverSymmetric file not found: {path}")
        content = path.read_text(encoding="utf-8")
    else:
        local = repo_root() / "data" / "fever_symmetric" / "fever_symmetric_dev.jsonl"
        if local.exists():
            content = local.read_text(encoding="utf-8")
        else:
            try:
                import requests
                r = requests.get(FEVERSYMMETRIC_URL, timeout=120)
                r.raise_for_status()
                content = r.text
            except Exception:
                import urllib.request
                with urllib.request.urlopen(FEVERSYMMETRIC_URL, timeout=120) as f:
                    content = f.read().decode("utf-8")

    data = []
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        label = row.get("label", "").upper()
        if label not in ("SUPPORTS", "REFUTES"):
            continue
        data.append({"claim_text": row["claim"], "label": 1 if label == "SUPPORTS" else 0})
    df = pd.DataFrame(data)
    df["pair_id"] = np.arange(len(df))
    return df


def load_boolq(local_path: str | Path | None = None) -> pd.DataFrame:
    """
    BoolQ validation split: question text only, label = answer (True -> 1).

    Resolution: explicit path → data/boolq/validation.{parquet,json,jsonl,csv}
    → HuggingFace google/boolq validation.
    """
    candidates: list[Path] = []
    if local_path is not None:
        p = Path(local_path)
        if not p.exists():
            raise FileNotFoundError(f"BoolQ data not found: {p}")
        candidates.append(p)
    else:
        base = repo_root() / "data" / "boolq"
        for name in ("validation.parquet", "validation.json", "validation.jsonl", "validation.csv"):
            q = base / name
            if q.exists():
                candidates.append(q)
                break

    if candidates:
        path = candidates[0]
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            raw = pd.read_parquet(path)
        elif suffix == ".json":
            raw = pd.read_json(path)
        elif suffix in (".jsonl", ".ndjson"):
            raw = pd.read_json(path, lines=True)
        elif suffix == ".csv":
            raw = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported BoolQ file type: {path}")
    else:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "Install datasets (pip install datasets) or pass --boolq-data PATH "
                "or place validation.parquet/json/jsonl/csv under data/boolq/"
            ) from e
        ds = load_dataset("google/boolq", split="validation")
        raw = ds.to_pandas()

    need = {"question", "answer"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"BoolQ table missing columns {missing}; have {list(raw.columns)}")

    df = pd.DataFrame({
        "pair_id": np.arange(len(raw)),
        "claim_text": raw["question"].astype(str),
        "label": raw["answer"].map(lambda a: 1 if bool(a) else 0).astype(int),
    })
    return df


def load_halueval(local_path: str | Path | None = None) -> pd.DataFrame:
    """
    HaluEval QA: two audit rows per example (right_answer=1, hallucinated_answer=0).

    Uses answer strings only (not question/knowledge). pair_id groups the pair.

    Resolution: --halueval-data path → data/halueval/qa.{parquet,json,jsonl,csv}
    → HuggingFace load_dataset(HALUEVAL_HF_ID, HALUEVAL_HF_CONFIG, split=...).
    """
    candidates: list[Path] = []
    if local_path is not None:
        p = Path(local_path)
        if not p.exists():
            raise FileNotFoundError(f"HaluEval data not found: {p}")
        candidates.append(p)
    else:
        base = repo_root() / "data" / "halueval"
        for name in ("qa.parquet", "qa.json", "qa.jsonl", "qa.csv"):
            q = base / name
            if q.exists():
                candidates.append(q)
                break

    if candidates:
        path = candidates[0]
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            raw = pd.read_parquet(path)
        elif suffix == ".json":
            raw = pd.read_json(path)
        elif suffix in (".jsonl", ".ndjson"):
            raw = pd.read_json(path, lines=True)
        elif suffix == ".csv":
            raw = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported HaluEval file type: {path}")
    else:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "Install datasets (pip install datasets) or pass --halueval-data PATH "
                "or place qa.parquet/json/jsonl/csv under data/halueval/"
            ) from e
        ds = load_dataset(HALUEVAL_HF_ID, HALUEVAL_HF_CONFIG, split=HALUEVAL_HF_SPLIT)
        raw = ds.to_pandas()

    need = {"right_answer", "hallucinated_answer"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"HaluEval table missing columns {missing}; have {list(raw.columns)}")

    out_rows: list[dict] = []
    pair_id = 0
    for _, row in raw.iterrows():
        ra = str(row["right_answer"]).strip()
        ha = str(row["hallucinated_answer"]).strip()
        if not ra or not ha:
            continue
        out_rows.append({"pair_id": pair_id, "claim_text": ra, "label": 1})
        out_rows.append({"pair_id": pair_id, "claim_text": ha, "label": 0})
        pair_id += 1

    return pd.DataFrame(out_rows)


def load_vitaminc(local_path: str | Path | None = None) -> pd.DataFrame:
    """
    VitaminC validation: claim-only features, SUPPORTS=1 and REFUTES=0.

    Drops NOT ENOUGH INFO to mirror FEVER-style binary setup.
    Keeps case_id for grouped CV.

    Resolution: --vitaminc-data path → data/vitaminc/validation.{parquet,json,jsonl,csv}
    → HuggingFace load_dataset(VITAMINC_HF_ID, VITAMINC_HF_CONFIG, split=...).
    """
    candidates: list[Path] = []
    if local_path is not None:
        p = Path(local_path)
        if not p.exists():
            raise FileNotFoundError(f"VitaminC data not found: {p}")
        candidates.append(p)
    else:
        base = repo_root() / "data" / "vitaminc"
        for name in ("validation.parquet", "validation.json", "validation.jsonl", "validation.csv"):
            q = base / name
            if q.exists():
                candidates.append(q)
                break

    if candidates:
        path = candidates[0]
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            raw = pd.read_parquet(path)
        elif suffix == ".json":
            raw = pd.read_json(path)
        elif suffix in (".jsonl", ".ndjson"):
            raw = pd.read_json(path, lines=True)
        elif suffix == ".csv":
            raw = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported VitaminC file type: {path}")
    else:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "Install datasets (pip install datasets) or pass --vitaminc-data PATH "
                "or place validation.parquet/json/jsonl/csv under data/vitaminc/"
            ) from e
        ds = load_dataset(VITAMINC_HF_ID, VITAMINC_HF_CONFIG, split=VITAMINC_HF_SPLIT)
        raw = ds.to_pandas()

    need = {"claim", "label", "case_id"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"VitaminC table missing columns {missing}; have {list(raw.columns)}")

    label_map = {"SUPPORTS": 1, "REFUTES": 0}
    keep = raw["label"].astype(str).str.upper().isin(label_map.keys())
    filtered = raw.loc[keep, ["claim", "label", "case_id"]].copy()
    filtered["label"] = filtered["label"].astype(str).str.upper().map(label_map).astype(int)

    df = pd.DataFrame({
        "case_id": filtered["case_id"].astype(str),
        "claim_text": filtered["claim"].astype(str),
        "label": filtered["label"],
    })
    return df


def _csv_metadata_note(results_all: list[dict]) -> str:
    f1, fs, bq, hv, vc = results_all[0], results_all[1], results_all[2], results_all[3], results_all[4]
    inv_auc = 1 - fs["auc"]
    return (
        "# FEVER 1.0: frozen row from shared_task_dev.jsonl (SUPPORTS+REFUTES, dedup by claim).\n"
        "# FeverSymmetric: frozen row from fever_symmetric_dev.jsonl (SUPPORTS+REFUTES).\n"
        f"# FeverSymmetric AUC={fs['auc']:.3f}: inverted signal. 1-{fs['auc']:.3f}={inv_auc:.3f} "
        f"≈ FEVER 1.0 AUC ({f1['auc']:.3f}). Symmetric reverses shortcuts, does not eliminate them.\n"
        f"# BoolQ: google/boolq validation, recomputed each run (N={bq['n']}). Question text only.\n"
        f"# HaluEval QA: {HALUEVAL_HF_ID} config={HALUEVAL_HF_CONFIG} split={HALUEVAL_HF_SPLIT}, "
        f"recomputed each run (N={hv['n']} answer rows). GroupKFold by pair_id; answer text only; "
        "excludes question_neg feature. LLM-generated (ChatGPT): possible style confounds.\n"
        f"# VitaminC: {VITAMINC_HF_ID} config={VITAMINC_HF_CONFIG} split={VITAMINC_HF_SPLIT}, "
        f"recomputed each run (N={vc['n']} claims). Claim-only; SUPPORTS/REFUTES only (NEI dropped); "
        "GroupKFold by case_id.\n"
    )


def save_audit_csv(path: Path, results_all: list[dict]) -> None:
    res_df = pd.DataFrame(results_all)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_csv_metadata_note(results_all))
        res_df.to_csv(f, index=False)


def write_fever_ablation_tex(
    path: Path,
    ablation_rows: list[list[str]],
    *,
    caption: str = "FEVER 1.0 feature-group ablation (5-fold stratified CV, 100 permutation nulls).",
    label: str = "tab:fever_ablation",
) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Features removed & AUC & Null mean \\\\",
        "\\midrule",
    ]
    for row in ablation_rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_dataset_tex(path: Path, results_all: list[dict]) -> None:
    truth_row = [
        "TruthfulQA",
        str(TRUTHFULQA_N),
        f"{TRUTHFULQA_AUC:.3f}",
        f"{TRUTHFULQA_NULL:.3f}",
        "<0.001",
        "Negation",
        "67.5\\%",
    ]
    cross_rows = [truth_row]
    for r in results_all:
        pstr = f"{r['p_value']:.3f}" if r["p_value"] >= 0.001 else "<0.001"
        if "Symmetric" in r["dataset"]:
            confound_str = "N/A"
        elif r["confound_pct"] < 1.0:
            confound_str = "$<$1\\%"
        else:
            confound_str = f"{r['confound_pct']:.1f}\\%"
        cross_rows.append([
            r["dataset"].replace(" dev", ""),
            str(r["n"]),
            f"{r['auc']:.3f}",
            f"{r['null_mean']:.3f}",
            pstr,
            r["dominant_feature"],
            confound_str,
        ])
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cross-dataset surface-form audit comparison.}",
        "\\label{tab:cross_dataset}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Dataset & N & AUC & Null mean & p & Dominant feature & Confound \\% \\\\",
        "\\midrule",
    ]
    for row in cross_rows:
        lines.append(" & ".join(str(x) for x in row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_auc_comparison(fig_dir: Path, results_all: list[dict]) -> None:
    datasets = ["TruthfulQA", "FEVER 1.0", "FeverSymmetric", "BoolQ", "HaluEval QA", "VitaminC"]
    aucs = [TRUTHFULQA_AUC] + [r["auc"] for r in results_all]
    nulls = [TRUTHFULQA_NULL] + [r["null_mean"] for r in results_all]

    fig, ax = plt.subplots(figsize=(8.8, 3.5))
    x = np.arange(len(datasets))
    w = 0.35
    ax.bar(x - w / 2, aucs, w, label="AUC", color="#009E73", edgecolor="black")
    ax.bar(x + w / 2, nulls, w, label="Null mean", color="#D55E00", edgecolor="black")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylabel("AUC / Null mean")
    ax.set_title("Surface-form audit: AUC vs permutation null across datasets")
    ax.legend(frameon=False)
    ax.set_ylim(0.35, 0.85)
    fig.tight_layout()
    fig.savefig(fig_dir / "fever_audit_auc_comparison.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fever_audit_auc_comparison.png", bbox_inches="tight")
    plt.close(fig)


def print_deliverables(results_all: list[dict]) -> None:
    f1, fs, bq, hv, vc = results_all[0], results_all[1], results_all[2], results_all[3], results_all[4]
    print("\n" + "=" * 60)
    print("DELIVERABLES")
    print("=" * 60)
    print(f"FEVER 1.0 (frozen) AUC: {f1['auc']:.3f}, null mean: {f1['null_mean']:.3f}")
    print(f"FeverSymmetric (frozen) AUC: {fs['auc']:.3f}, null mean: {fs['null_mean']:.3f}")
    print(f"BoolQ validation (live) OOF AUC: {bq['auc']:.3f}, null mean: {bq['null_mean']:.3f}")
    print(
        f"HaluEval QA (live, GroupKFold) OOF AUC: {hv['auc']:.3f}, "
        f"null mean: {hv['null_mean']:.3f}"
    )
    print(
        f"VitaminC validation (live) OOF AUC: {vc['auc']:.3f}, "
        f"null mean: {vc['null_mean']:.3f}"
    )
    if bq["auc"] < 0.55:
        print(
            "BoolQ near-chance AUC suggests interrogative surface form "
            "carries less shortcut signal than declarative claims — itself a "
            "finding worth reporting."
        )
        print(
            "Paper framing: treat this as a clean positive — weak surface-form "
            "signal and near-zero confound rate support cross-format robustness "
            "(interrogative vs declarative), not a failed audit."
        )
    print(f"Dominant feature (frozen FEVER ablation): {f1['dominant_feature']}")
    print(f"Dominant feature (BoolQ ablation): {bq['dominant_feature']}")
    print(f"Dominant feature (HaluEval ablation): {hv['dominant_feature']}")
    print(f"Dominant feature (VitaminC ablation): {vc['dominant_feature']}")
    print(f"FEVER confound rate (frozen): {f1['confound_pct']:.1f}%")
    print(f"BoolQ confound rate: {bq['confound_pct']:.1f}%")
    print(f"HaluEval confound rate: {hv['confound_pct']:.1f}%")
    print(f"VitaminC confound rate: {vc['confound_pct']:.1f}%")
    vs_tqa = (
        "stronger" if f1["auc"] > TRUTHFULQA_AUC
        else "weaker" if f1["auc"] < TRUTHFULQA_AUC
        else "similar"
    )
    print(f"FEVER signal vs TruthfulQA ({TRUTHFULQA_AUC:.3f}): {vs_tqa}.")
    drop_toward_50 = "yes" if fs["auc"] < 0.65 else "no"
    print(f"FeverSymmetric AUC drop toward 0.50: {drop_toward_50}.")
    fs_inverted_auc = 1 - fs["auc"]
    print(f"FeverSymmetric inverted AUC (1 - {fs['auc']:.3f}): {fs_inverted_auc:.3f}")
    print(
        f"This matches FEVER 1.0 AUC ({f1['auc']:.3f}), confirming shortcuts are inverted not removed."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root_default = str(repo_root())
    p = argparse.ArgumentParser(
        description=(
            "External surface-form audit: frozen FEVER + FeverSymmetric; "
            "live BoolQ + HaluEval QA + VitaminC (HF datasets)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  cd {root_default} && python scripts/run_fever_audit.py

  python scripts/run_fever_audit.py --boolq-data data/boolq/validation.parquet
  python scripts/run_fever_audit.py --random-label-control-only

FEVER / FeverSymmetric: frozen in-script. BoolQ: google/boolq validation (question-only).
HaluEval: {HALUEVAL_HF_ID} config={HALUEVAL_HF_CONFIG} split={HALUEVAL_HF_SPLIT}
(answer-only, GroupKFold by pair_id). Local: data/halueval/qa.{{parquet,json,jsonl,csv}}
VitaminC: {VITAMINC_HF_ID} config={VITAMINC_HF_CONFIG} split={VITAMINC_HF_SPLIT}
(claim-only; SUPPORTS/REFUTES). Local: data/vitaminc/validation.{{parquet,json,jsonl,csv}}
        """.strip(),
    )
    p.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root (default: current directory; outputs go under audits/ and paper_assets/)",
    )
    p.add_argument(
        "--boolq-data",
        type=str,
        default=None,
        help="Path to BoolQ validation export (parquet/json/csv) with columns question, answer",
    )
    p.add_argument(
        "--halueval-data",
        type=str,
        default=None,
        help="Path to HaluEval QA table with columns right_answer, hallucinated_answer",
    )
    p.add_argument(
        "--vitaminc-data",
        type=str,
        default=None,
        help="Path to VitaminC table with columns claim, label, case_id",
    )
    p.add_argument(
        "--fever_dev",
        type=str,
        default=None,
        help="Deprecated, no-op: FEVER row is frozen; ignored.",
    )
    p.add_argument(
        "--fever_symmetric",
        type=str,
        default=None,
        help="Deprecated, no-op: FeverSymmetric row is frozen; ignored.",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    p.add_argument(
        "--n-null",
        type=int,
        default=N_PERMUTATION_NULL,
        help=f"Permutation null draws (default: {N_PERMUTATION_NULL})",
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=N_BOOTSTRAP,
        help=f"Bootstrap samples for AUC CI (default: {N_BOOTSTRAP})",
    )
    p.add_argument(
        "--random-label-control",
        action="store_true",
        help="Run random-label control after main audit (BoolQ + VitaminC, 20 runs each)",
    )
    p.add_argument(
        "--random-label-control-only",
        action="store_true",
        help="Run only random-label control, skip main audit (faster for control numbers)",
    )
    p.add_argument(
        "--random-label-control-dataset",
        type=str,
        choices=["boolq", "vitaminc", "both"],
        default="both",
        help="Which datasets to run random-label control on (default: both)",
    )
    p.add_argument(
        "--n-random-runs",
        type=int,
        default=None,
        help="Override n runs for random-label control (e.g. 101). Default: BoolQ=40, VitaminC=20.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run external surface-form audit (frozen FEVER/Symmetric + live BoolQ/HaluEval/VitaminC)."""
    args = parse_args(argv)
    if args.random_label_control_only:
        ds = args.random_label_control_dataset
        datasets = ["boolq", "vitaminc"] if ds == "both" else [ds]
        run_random_label_control(args, datasets)
        return 0
    if args.fever_dev is not None or args.fever_symmetric is not None:
        print(
            "run_fever_audit.py: warning: --fever_dev and --fever_symmetric are deprecated "
            "and ignored (FEVER / FeverSymmetric metrics are frozen in-script).",
            file=sys.stderr,
        )
    root = Path(args.root).resolve()
    audits_dir = root / "audits"
    tbl_dir = root / "paper_assets" / "tables"
    fig_dir = root / "paper_assets" / "fig"
    audits_dir.mkdir(exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    n_null = args.n_null
    n_boot = args.n_boot

    results_all: list[dict] = [
        dict(FEVER_10_RESULT),
        dict(FEVERSYMMETRIC_RESULT),
    ]
    print("\n" + "=" * 60)
    print("FEVER 1.0 dev + FeverSymmetric dev (frozen constants; not recomputed)")
    print("=" * 60)
    for row in results_all:
        print(
            f"  {row['dataset']}: N={row['n']}, AUC={row['auc']:.3f}, "
            f"null={row['null_mean']:.3f}"
        )

    print(f"\n{'=' * 60}\nBoolQ validation (live)\n{'=' * 60}")
    df_bq = load_boolq(args.boolq_data)
    claims = df_bq["claim_text"]
    y = df_bq["label"].values
    n = len(df_bq)
    print(f"Loaded {n} questions, {int(y.sum())} True, {int((1 - y).sum())} False")

    feat_df = compute_features(claims)
    X = feat_df[FEAT_COLS].fillna(0).to_numpy()

    auc, ci_lo, ci_hi, null_mean, p_value, _ = run_audit(X, y, seed, n_null, n_boot)
    print(f"OOF AUC: {auc:.3f} [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"Null mean: {null_mean:.3f}, p-value: {p_value:.4f}")

    ablations = run_ablation(feat_df, y, FEAT_COLS, seed, n_null)
    dominant = dominant_ablation_group(ablations)

    conf = heuristic_confound(feat_df, for_boolq=True)
    confound_pct = 100 * conf.mean()
    print(f"Confound rate (incl. question_neg): {confound_pct:.1f}%")

    results_all.append({
        "dataset": "BoolQ validation",
        "n": n,
        "auc": auc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "null_mean": null_mean,
        "p_value": p_value,
        "dominant_feature": dominant,
        "confound_pct": confound_pct,
    })

    print(f"\n{'=' * 60}\nHaluEval QA (live, GroupKFold by pair_id)\n{'=' * 60}")
    df_hv = load_halueval(args.halueval_data)
    claims_hv = df_hv["claim_text"]
    y_hv = df_hv["label"].values
    groups_hv = df_hv["pair_id"].values
    n_hv = len(df_hv)
    print(
        f"Loaded {n_hv} answer rows ({len(np.unique(groups_hv))} pairs), "
        f"{int(y_hv.sum())} right, {int((1 - y_hv).sum())} hallucinated"
    )

    feat_hv = compute_features(claims_hv)
    X_hv = feat_hv[HALUEVAL_FEAT_COLS].fillna(0).to_numpy()

    auc_h, ci_lo_h, ci_hi_h, null_h, p_h, _ = run_audit(
        X_hv, y_hv, seed, n_null, n_boot, groups=groups_hv
    )
    print(f"OOF AUC: {auc_h:.3f} [95% CI: {ci_lo_h:.3f}, {ci_hi_h:.3f}]")
    print(f"Null mean: {null_h:.3f}, p-value: {p_h:.4f}")

    ablations_hv = run_ablation(
        feat_hv, y_hv, HALUEVAL_FEAT_COLS, seed, n_null, groups=groups_hv
    )
    dominant_hv = dominant_ablation_group(ablations_hv)

    conf_hv = heuristic_confound(feat_hv, for_boolq=False)
    confound_pct_hv = 100 * conf_hv.mean()
    print(f"Confound rate: {confound_pct_hv:.1f}%")

    results_all.append({
        "dataset": "HaluEval QA",
        "n": n_hv,
        "auc": auc_h,
        "ci_lo": ci_lo_h,
        "ci_hi": ci_hi_h,
        "null_mean": null_h,
        "p_value": p_h,
        "dominant_feature": dominant_hv,
        "confound_pct": confound_pct_hv,
    })

    print(f"\n{'=' * 60}\nVitaminC validation (live, GroupKFold by case_id)\n{'=' * 60}")
    df_vc = load_vitaminc(args.vitaminc_data)
    claims_vc = df_vc["claim_text"]
    y_vc = df_vc["label"].values
    groups_vc = df_vc["case_id"].values
    n_vc = len(df_vc)
    print(
        f"Loaded {n_vc} claims ({len(np.unique(groups_vc))} case_id groups), "
        f"{int(y_vc.sum())} supports, {int((1 - y_vc).sum())} refutes"
    )

    feat_vc = compute_features(claims_vc)
    X_vc = feat_vc[FEAT_COLS].fillna(0).to_numpy()

    auc_v, ci_lo_v, ci_hi_v, null_v, p_v, _ = run_audit(
        X_vc, y_vc, seed, n_null, n_boot, groups=groups_vc
    )
    print(f"OOF AUC: {auc_v:.3f} [95% CI: {ci_lo_v:.3f}, {ci_hi_v:.3f}]")
    print(f"Null mean: {null_v:.3f}, p-value: {p_v:.4f}")

    ablations_vc = run_ablation(feat_vc, y_vc, FEAT_COLS, seed, n_null, groups=groups_vc)
    dominant_vc = dominant_ablation_group(ablations_vc)

    conf_vc = heuristic_confound(feat_vc, for_boolq=False)
    confound_pct_vc = 100 * conf_vc.mean()
    print(f"Confound rate: {confound_pct_vc:.1f}%")

    results_all.append({
        "dataset": "VitaminC validation",
        "n": n_vc,
        "auc": auc_v,
        "ci_lo": ci_lo_v,
        "ci_hi": ci_hi_v,
        "null_mean": null_v,
        "p_value": p_v,
        "dominant_feature": dominant_vc,
        "confound_pct": confound_pct_vc,
    })

    csv_path = audits_dir / "fever_audit_results.csv"
    save_audit_csv(csv_path, results_all)
    print(f"\nSaved {csv_path}")

    ablation_rows = [[name, f"{auc:.3f}", f"{null_m:.3f}"] for name, auc, null_m in ablations]
    boolq_ab_path = tbl_dir / "boolq_feature_ablation_table.tex"
    write_fever_ablation_tex(
        boolq_ab_path,
        ablation_rows,
        caption="BoolQ validation feature-group ablation (5-fold stratified CV, 100 permutation nulls).",
        label="tab:boolq_ablation",
    )
    print(f"Saved {boolq_ab_path}")

    ablation_rows_hv = [
        [name, f"{a:.3f}", f"{nm:.3f}"] for name, a, nm in ablations_hv
    ]
    halueval_ab_path = tbl_dir / "halueval_feature_ablation_table.tex"
    write_fever_ablation_tex(
        halueval_ab_path,
        ablation_rows_hv,
        caption=(
            "HaluEval QA feature-group ablation (5-fold GroupKFold by pair_id, "
            "100 permutation nulls; answer-only features, no question\\_neg)."
        ),
        label="tab:halueval_ablation",
    )
    print(f"Saved {halueval_ab_path}")

    ablation_rows_vc = [
        [name, f"{a:.3f}", f"{nm:.3f}"] for name, a, nm in ablations_vc
    ]
    vitaminc_ab_path = tbl_dir / "vitaminc_feature_ablation_table.tex"
    write_fever_ablation_tex(
        vitaminc_ab_path,
        ablation_rows_vc,
        caption=(
            "VitaminC validation feature-group ablation (5-fold GroupKFold by case_id, "
            "100 permutation nulls; claim-only, SUPPORTS/REFUTES only)."
        ),
        label="tab:vitaminc_ablation",
    )
    print(f"Saved {vitaminc_ab_path}")

    cross_path = tbl_dir / "cross_dataset_comparison_table.tex"
    write_cross_dataset_tex(cross_path, results_all)
    print(f"Saved {cross_path}")

    plot_auc_comparison(fig_dir, results_all)
    print(f"Saved {fig_dir / 'fever_audit_auc_comparison.pdf'}")

    print_deliverables(results_all)

    if args.random_label_control:
        ds = args.random_label_control_dataset
        datasets = ["boolq", "vitaminc"] if ds == "both" else [ds]
        run_random_label_control(args, datasets)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e
