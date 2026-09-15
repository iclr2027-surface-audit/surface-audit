#!/usr/bin/env python3
"""
Curiosity experiment (NOT for paper) — fourth extra embedding family:
BAAI/bge-multilingual-gemma2 (9.24B-param Gemma2-based encoder).

Same protocol as audits/_exp_three_extra_embeddings.py:
  - encode 1580 (full) + 1056 (cleaned tau=0.52) TruthfulQA answer rows
  - StandardScaler + LogisticRegression head, GroupKFold(n=5) AUC by pair_id
  - save pickles to artifacts/classifiers/bge_multi_gemma2_lr_{full,cleaned}.pkl
  - score inverted-side top-1 accuracy on the n=135 surface-flipped set

Memory plan: try MPS bf16 with batch=1 first; on OOM fall back to CPU.
"""
from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "artifacts"
CLF_DIR = ARTIFACTS / "classifiers"
EMB_DIR = ARTIFACTS / "embeddings"
HF_CACHE = ARTIFACTS / "hf_cache"

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "hub")
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# leave room on MPS unified memory
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest

CLF_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR.mkdir(parents=True, exist_ok=True)

TRUTHFULQA_CSV = REPO / "TruthfulQA.csv"
TAU052 = REPO / "data" / "subsets" / "TruthfulQA-Audited" / "surface_audited" / "pair_ids" / "pair_ids_tau052.json"
EVAL_CSV = REPO / "audits" / "_eval_dataset_v3_inv_accuracy_full_final.csv"
EXISTING_RESULTS = REPO / "audits" / "_eval_dataset_v3_inv_accuracy_full_score_tables.json"

LABEL = "BGE-Multi-Gemma2"
PREFIX = "bge_multi_gemma2"
MODEL_ID = "BAAI/bge-multilingual-gemma2"

RANDOM_STATE = 42
CV_SPLITS = 5

# ---- helpers --------------------------------------------------------------

def make_pipe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, solver="liblinear",
                           random_state=RANDOM_STATE),
    )


def cv_auc(X, y, g):
    pipe = make_pipe()
    proba = cross_val_predict(pipe, X, y, cv=GroupKFold(CV_SPLITS),
                              groups=g, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def fit_and_save(prefix, X, y, g, out_pkl):
    auc = cv_auc(X, y, g)
    pipe = make_pipe(); pipe.fit(X, y)
    joblib.dump({
        "pipeline": pipe, "cv_auc_group5": auc,
        "n_rows": int(X.shape[0]), "n_pairs": int(len(np.unique(g))),
        "embedding_dim": int(X.shape[1]),
        "random_state": RANDOM_STATE, "cv_splits": CV_SPLITS,
        "recipe": "make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, solver='liblinear', random_state=42))",
        "source_prefix": prefix,
    }, out_pkl)
    return auc


def proba_truthful(pipe, X):
    clf = pipe.steps[-1][1]
    classes = list(getattr(clf, "classes_", [0, 1]))
    col = classes.index(1)
    return np.asarray(pipe.predict_proba(X)[:, col], dtype=float)


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n=n, p=0.5,
                           alternative="two-sided").pvalue)


# ---- BGE-Multilingual-Gemma2 encoder -------------------------------------

class GemmaEncoder:
    """Last-token pool, L2 normalize. Tries MPS first, falls back to CPU."""

    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"  loading model on {self.device} (bf16)...", flush=True)
        t0 = time.time()
        self.mdl = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        self.mdl.eval()
        try:
            self.mdl.to(self.device)
            print(f"    loaded on {self.device} in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"    MPS load failed ({type(e).__name__}: {e}); falling back to CPU", flush=True)
            self.device = torch.device("cpu")
            self.mdl = self.mdl.to(self.device)

    def encode(self, texts, batch_size=1, max_length=512, log_every=200):
        out = []
        n = len(texts)
        for s in range(0, n, batch_size):
            batch = texts[s:s + batch_size]
            try:
                enc = self.tok(batch, padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.no_grad():
                    o = self.mdl(**enc)
                pooled = o.last_hidden_state[:, -1, :]
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                out.append(pooled.detach().cpu().to(torch.float32).numpy())
            except RuntimeError as e:
                msg = str(e).lower()
                if ("out of memory" in msg or "memory" in msg) and self.device.type == "mps":
                    print(f"    MPS OOM at row {s}; switching to CPU for remainder", flush=True)
                    self.mdl = self.mdl.to("cpu")
                    self.device = torch.device("cpu")
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    enc = self.tok(batch, padding=True, truncation=True,
                                   max_length=max_length, return_tensors="pt")
                    with torch.no_grad():
                        o = self.mdl(**enc)
                    pooled = o.last_hidden_state[:, -1, :]
                    pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                    out.append(pooled.detach().to(torch.float32).numpy())
                else:
                    raise
            if s and s % log_every == 0:
                print(f"      encode: {s}/{n}", flush=True)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        return np.concatenate(out, axis=0).astype(np.float32)

    def close(self):
        del self.mdl, self.tok
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


# ---- training corpus ------------------------------------------------------

def build_truthfulqa_rows():
    tq = pd.read_csv(TRUTHFULQA_CSV)
    rows = []
    for pid in range(len(tq)):
        rows.append({"pair_id": pid, "y": 1, "text": str(tq["Best Answer"].iat[pid])})
        rows.append({"pair_id": pid, "y": 0, "text": str(tq["Best Incorrect Answer"].iat[pid])})
    return pd.DataFrame(rows)


def filter_tau052_keep():
    return set(int(p) for p in json.loads(TAU052.read_text())["pair_ids"])


# ---- main ----------------------------------------------------------------

def main():
    t_total = time.time()
    print(f"\n{'='*72}\n{LABEL}  ({PREFIX})\n{'='*72}", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)

    full_rows = build_truthfulqa_rows()
    tau_keep = filter_tau052_keep()
    eval_df = pd.read_csv(EVAL_CSV)
    print(f"  full rows={len(full_rows)}  cleaned subset target=1056  "
          f"eval rows={len(eval_df)}", flush=True)

    encoder = GemmaEncoder()

    # ---- encode train corpus ----
    t0 = time.time()
    print(f"  [encode] full train: {len(full_rows)} rows", flush=True)
    X_full = encoder.encode(full_rows.text.tolist())
    encode_train_time = time.time() - t0
    print(f"    encoded shape={X_full.shape} in {encode_train_time:.1f}s", flush=True)

    y_full = full_rows.y.to_numpy(int)
    g_full = full_rows.pair_id.to_numpy(int)
    keep_mask = full_rows.pair_id.isin(tau_keep).to_numpy()
    X_clean = X_full[keep_mask]; y_clean = y_full[keep_mask]; g_clean = g_full[keep_mask]
    assert X_clean.shape[0] == 1056

    np.save(EMB_DIR / f"{PREFIX}_full_X.npy", X_full)
    np.save(EMB_DIR / f"{PREFIX}_full_y.npy", y_full)
    np.save(EMB_DIR / f"{PREFIX}_full_pair_id.npy", g_full)
    np.save(EMB_DIR / f"{PREFIX}_cleaned_X.npy", X_clean)
    np.save(EMB_DIR / f"{PREFIX}_cleaned_y.npy", y_clean)
    np.save(EMB_DIR / f"{PREFIX}_cleaned_pair_id.npy", g_clean)

    # ---- fit + AUC ----
    t1 = time.time()
    out_full_pkl = CLF_DIR / f"{PREFIX}_lr_full.pkl"
    out_clean_pkl = CLF_DIR / f"{PREFIX}_lr_cleaned.pkl"
    auc_full = fit_and_save(f"{PREFIX}_full", X_full, y_full, g_full, out_full_pkl)
    auc_clean = fit_and_save(f"{PREFIX}_cleaned", X_clean, y_clean, g_clean, out_clean_pkl)
    train_time = time.time() - t1
    print(f"  [train] full CV AUC = {auc_full:.4f}  "
          f"cleaned CV AUC = {auc_clean:.4f}  "
          f"({'OK' if auc_full > auc_clean else 'unexpected'})  "
          f"({train_time:.1f}s)", flush=True)

    # ---- score n=135 inverted ----
    t2 = time.time()
    XT = encoder.encode(eval_df["true_text_surface_inverted"].astype(str).tolist())
    XF = encoder.encode(eval_df["false_text_surface_inverted"].astype(str).tolist())
    score_time = time.time() - t2
    encoder.close()

    pipe_full = joblib.load(out_full_pkl)["pipeline"]
    pipe_clean = joblib.load(out_clean_pkl)["pipeline"]
    pT_full = proba_truthful(pipe_full, XT); pF_full = proba_truthful(pipe_full, XF)
    pT_clean = proba_truthful(pipe_clean, XT); pF_clean = proba_truthful(pipe_clean, XF)
    full_correct = (pT_full > pF_full).astype(int)
    clean_correct = (pT_clean > pF_clean).astype(int)
    b = int(((full_correct == 1) & (clean_correct == 0)).sum())
    c = int(((full_correct == 0) & (clean_correct == 1)).sum())
    p = mcnemar_p(b, c)
    acc_full = float(full_correct.mean()); acc_clean = float(clean_correct.mean())
    print(f"  [score] acc_full={acc_full:.3f}  acc_clean={acc_clean:.3f}  "
          f"Δ={acc_clean-acc_full:+.3f}  b={b} c={c} p={p:.4f}  "
          f"({score_time:.1f}s)", flush=True)

    # ---- combined comparison ----
    EX_ORDER = ["surface_lr", "BGE-large", "ModernBERT-base",
                "Qwen2.5-0.5B", "Qwen2.5-1.5B", "SmolLM2-1.7B",
                "Qwen2.5-3B", "Phi-3.5-mini", "Llama-3.2-3B"]
    PRIOR_NEW = [
        ("BGE-M3", {"acc_full": 0.200, "acc_clean": 0.304, "delta": +0.104, "b": 7, "c": 21, "p": 0.0125}),
    ]

    existing = json.loads(EXISTING_RESULTS.read_text())
    rows = []
    for fam in EX_ORDER:
        r = existing["results"].get(fam, {})
        if "error" in r:
            continue
        rows.append({"label": fam, "src": "existing",
                     "acc_full": r["acc_full"], "acc_clean": r["acc_clean"],
                     "delta": r["delta_clean_minus_full"],
                     "b": r["mcnemar_b"], "c": r["mcnemar_c"], "p": r["mcnemar_p"]})
    for lbl, r in PRIOR_NEW:
        rows.append({"label": lbl, "src": "NEW (prior)",
                     "acc_full": r["acc_full"], "acc_clean": r["acc_clean"],
                     "delta": r["delta"], "b": r["b"], "c": r["c"], "p": r["p"]})
    rows.append({"label": LABEL, "src": "NEW",
                 "acc_full": acc_full, "acc_clean": acc_clean,
                 "delta": acc_clean - acc_full, "b": b, "c": c, "p": p})

    rows.sort(key=lambda r: -r["delta"])

    print()
    print("=" * 72)
    print(f"COMBINED COMPARISON ({len(rows)} families, sorted by Δ; n_eval={len(eval_df)})")
    print("=" * 72)
    print()
    print("| Family | Acc_full | Acc_clean | Δ | McNemar p (b, c) | source |")
    print("|---|---:|---:|---:|---|---|")
    for r in rows:
        sig = "***" if r["p"] < 0.001 else ("**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else ""))
        print(f"| {r['label']} | {r['acc_full']:.3f} | {r['acc_clean']:.3f} | "
              f"{r['delta']:+.3f} | {r['p']:.4f}{sig} (b={r['b']}, c={r['c']}) | {r['src']} |")

    print()
    print("=" * 72)
    print("TIMING")
    print("=" * 72)
    print(f"  encode train: {encode_train_time:7.1f}s")
    print(f"  train + CV  : {train_time:7.1f}s")
    print(f"  encode eval + score: {score_time:7.1f}s")
    print(f"  total       : {time.time()-t_total:7.1f}s")

    print(f"\n  AUC sanity: full={auc_full:.4f}  cleaned={auc_clean:.4f}  "
          f"({'OK (full>cleaned)' if auc_full > auc_clean else 'UNEXPECTED'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
