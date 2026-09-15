#!/usr/bin/env python3
"""
Build Qwen2.5-0.5B-Instruct embeddings for the TruthfulQA full (790 pairs)
and cleaned tau=0.52 (528 pairs) answer sets.

Motivation
----------
The three canonical families are:
  - surface_lr      (10 engineered surface features, LR)
  - BGE-large       (335M, encoder, cosine-optimized)
  - ModernBERT-base (149M, encoder, CLS token, not cosine)

Qwen2.5-0.5B-Instruct (~494M, causal LM) is added as a fourth family
immediately under ModernBERT on the capability ladder. It is:
  - Non-gated on HuggingFace
  - A causal decoder (no CLS token; we mean-pool last hidden state
    weighted by the attention mask)
  - A similar mid-band encoder-like representation, but from a
    generative pretraining objective

Config (fixed):
  MODEL_ID       "Qwen/Qwen2.5-0.5B-Instruct"
  pooling         attention-masked mean over last_hidden_state
  normalize       NO
  max_length      512
  batch_size      16   (smaller than MBERT because hidden=896 and
                       the decoder layers use more memory per token)
  precision       float32
  mode            eval(), torch.no_grad()
  device          MPS if available else CPU

Outputs (mirroring build_modernbert_embeddings.py):
  artifacts/embeddings/qwen_full_X.npy        (1580, 896) float32
  artifacts/embeddings/qwen_full_y.npy        (1580,)     int64
  artifacts/embeddings/qwen_full_pair_id.npy  (1580,)     int64
  artifacts/embeddings/qwen_cleaned_X.npy     (1056, 896) float32
  artifacts/embeddings/qwen_cleaned_y.npy     (1056,)     int64
  artifacts/embeddings/qwen_cleaned_pair_id.npy (1056,)   int64
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_HF_HOME_DEFAULT = REPO_ROOT / "artifacts" / "hf_cache"
os.environ.setdefault("HF_HOME", str(_HF_HOME_DEFAULT))
os.environ.setdefault("HF_HUB_CACHE", str(_HF_HOME_DEFAULT / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_HOME_DEFAULT / "hub"))
_HF_HOME_DEFAULT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_DIM_EXPECTED = 896
MAX_LENGTH = 512
BATCH_SIZE = 16

TRUTHFULQA_CSV = REPO_ROOT / "TruthfulQA.csv"
TAU052_PAIR_IDS = (
    REPO_ROOT
    / "data" / "subsets" / "TruthfulQA-Audited" / "surface_audited"
    / "pair_ids" / "pair_ids_tau052.json"
)
OUT_DIR = REPO_ROOT / "artifacts" / "embeddings"


def _build_answer_rows(tq: pd.DataFrame) -> pd.DataFrame:
    best = tq["Best Answer"].fillna("").astype(str)
    incorrect = tq["Best Incorrect Answer"].fillna("").astype(str)
    rows: list[dict] = []
    for pair_id in range(len(tq)):
        rows.append({"pair_id": pair_id, "y": 1, "text": best.iat[pair_id]})
        rows.append({"pair_id": pair_id, "y": 0, "text": incorrect.iat[pair_id]})
    return pd.DataFrame(rows)


def _pick_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _encode(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = _pick_device()
    print(f"Device: {device}", flush=True)

    print(f"Loading tokenizer for {MODEL_ID} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading model {MODEL_ID} (first run downloads ~1 GB) ...",
          flush=True)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    model.to(device)

    dim = getattr(model.config, "hidden_size", None)
    if dim != EMBED_DIM_EXPECTED:
        raise RuntimeError(
            f"Unexpected Qwen hidden_size={dim}, expected {EMBED_DIM_EXPECTED}."
        )
    print(f"  loaded; hidden_size={dim}", flush=True)

    out_rows: list[np.ndarray] = []
    n = len(texts)
    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            hidden = outputs.last_hidden_state  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
            summed = (hidden * mask).sum(dim=1)  # (B, H)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / denom  # (B, H)
            out_rows.append(pooled.detach().cpu().to(torch.float32).numpy())
            done = min(start + BATCH_SIZE, n)
            if (start // BATCH_SIZE) % 5 == 0 or done == n:
                print(f"  encoded {done}/{n} ...", flush=True)
    X = np.concatenate(out_rows, axis=0).astype(np.float32)
    if X.shape != (n, EMBED_DIM_EXPECTED):
        raise RuntimeError(f"Unexpected encoded shape: {X.shape}")
    return X


def _assert_finite(X: np.ndarray, label: str) -> None:
    if not np.isfinite(X).all():
        n_nan = int(np.isnan(X).sum())
        n_inf = int(np.isinf(X).sum())
        raise RuntimeError(
            f"{label}: non-finite values in embedding matrix "
            f"(NaN={n_nan}, Inf={n_inf})"
        )


def main() -> int:
    print("=" * 72, flush=True)
    print("build_qwen_embeddings.py", flush=True)
    print("=" * 72, flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Out dir: {OUT_DIR}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tq = pd.read_csv(TRUTHFULQA_CSV)
    print(f"TruthfulQA rows: {len(tq)}", flush=True)
    rows = _build_answer_rows(tq)
    if len(rows) != 2 * len(tq):
        raise RuntimeError(f"answer row count {len(rows)} != 2 * {len(tq)}")

    texts = rows["text"].tolist()
    X = _encode(texts)
    print(f"Encoded matrix: {X.shape}, dtype={X.dtype}", flush=True)
    _assert_finite(X, "full-encoded")

    y_full = rows["y"].to_numpy(dtype=np.int64)
    pair_full = rows["pair_id"].to_numpy(dtype=np.int64)

    np.save(OUT_DIR / "qwen_full_X.npy", X)
    np.save(OUT_DIR / "qwen_full_y.npy", y_full)
    np.save(OUT_DIR / "qwen_full_pair_id.npy", pair_full)
    print(f"Wrote qwen_full_* blobs ({X.shape[0]} rows)", flush=True)

    with open(TAU052_PAIR_IDS, "r", encoding="utf-8") as f:
        tau_ids = set(int(p) for p in json.load(f)["pair_ids"])
    keep = np.array([pid in tau_ids for pid in pair_full])
    X_c = X[keep]
    y_c = y_full[keep]
    p_c = pair_full[keep]
    if len(X_c) != 1056:
        raise RuntimeError(f"Cleaned row count {len(X_c)} != 1056")
    _assert_finite(X_c, "cleaned-encoded")
    np.save(OUT_DIR / "qwen_cleaned_X.npy", X_c)
    np.save(OUT_DIR / "qwen_cleaned_y.npy", y_c)
    np.save(OUT_DIR / "qwen_cleaned_pair_id.npy", p_c)
    print(f"Wrote qwen_cleaned_* blobs ({X_c.shape[0]} rows)", flush=True)

    for label, _X, y_, p_ in [
        ("full",    X,   y_full,  pair_full),
        ("cleaned", X_c, y_c,     p_c),
    ]:
        u, c = np.unique(p_, return_counts=True)
        if not np.all(c == 2):
            raise RuntimeError(
                f"{label}: pair_id counts not all 2 "
                f"(min={c.min()}, max={c.max()})"
            )
        print(f"  {label}: {len(u)} unique pair_ids, each appears twice. "
              f"y=1 count={int((y_==1).sum())}, y=0 count={int((y_==0).sum())}",
              flush=True)

    print("\nSanity:", flush=True)
    print(f"  shapes: full_X={X.shape}  cleaned_X={X_c.shape}", flush=True)
    print(f"  row-wise mean of first 3 full rows:    "
          f"{[float(f'{X[i].mean():.6f}') for i in range(3)]}", flush=True)
    print(f"  row-wise mean of first 3 cleaned rows: "
          f"{[float(f'{X_c[i].mean():.6f}') for i in range(3)]}", flush=True)
    print(f"  global min/max (full):  min={float(X.min()):.4f}  "
          f"max={float(X.max()):.4f}", flush=True)
    print("  finite-check: OK (no NaN, no Inf)", flush=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in build_qwen_embeddings.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
