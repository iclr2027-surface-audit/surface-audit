#!/usr/bin/env python3
"""
Step 6 — Encode Best Answer (y=1) and Best Incorrect Answer (y=0) for
the full 790-pair TruthfulQA set and the tau=0.52 528-pair cleaned
subset with `BAAI/bge-large-en-v1.5`, and persist .npy blobs for the
downstream embedding-LR.

Outputs (per plan):

    artifacts/embeddings/full_X.npy         (1580, 1024) float32
    artifacts/embeddings/full_y.npy         (1580,)      int
    artifacts/embeddings/full_pair_id.npy   (1580,)      int

    artifacts/embeddings/cleaned_X.npy      (1056, 1024) float32
    artifacts/embeddings/cleaned_y.npy      (1056,)      int
    artifacts/embeddings/cleaned_pair_id.npy (1056,)     int

Encoding settings (per plan §5.1):
    - `BAAI/bge-large-en-v1.5`
    - normalize_embeddings=True
    - No query/passage prefix (these are answer strings, not queries).

Row ordering matches the per-answer feature parquets written in Step 3
(pair_id ascending, y=1 row before y=0 row for each pair).
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ID = "BAAI/bge-large-en-v1.5"
EMBED_DIM = 1024
ENCODE_BATCH = 32

REPO_ROOT = Path(__file__).resolve().parent.parent

# Redirect HuggingFace hub cache into the workspace so model downloads do
# not require writing under ~/.cache (which may be sandboxed). Setting
# these before the first sentence_transformers / transformers import is
# what actually matters; callers can override by exporting HF_HOME first.
_HF_HOME_DEFAULT = REPO_ROOT / "artifacts" / "hf_cache"
os.environ.setdefault("HF_HOME", str(_HF_HOME_DEFAULT))
os.environ.setdefault("HF_HUB_CACHE", str(_HF_HOME_DEFAULT / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_HOME_DEFAULT / "hub"))
_HF_HOME_DEFAULT.mkdir(parents=True, exist_ok=True)
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


def _encode(texts: list[str]):
    from sentence_transformers import SentenceTransformer  # lazy import
    print(f"Loading {MODEL_ID} (first run may download ~1.3 GB) ...")
    model = SentenceTransformer(MODEL_ID)
    print(f"  loaded. embedding dim = {model.get_sentence_embedding_dimension()}")
    X = model.encode(
        texts,
        batch_size=ENCODE_BATCH,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.asarray(X, dtype=np.float32)


def main() -> int:
    print("=" * 72)
    print("STEP 6 — build_bge_embeddings.py")
    print("=" * 72)
    print(f"Model: {MODEL_ID}")
    print(f"Out dir: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tq = pd.read_csv(TRUTHFULQA_CSV)
    print(f"TruthfulQA rows: {len(tq)}")
    rows = _build_answer_rows(tq)
    assert len(rows) == 2 * len(tq)

    texts = rows["text"].tolist()
    X = _encode(texts)
    if X.shape != (len(rows), EMBED_DIM):
        raise RuntimeError(f"Unexpected embedding shape: {X.shape}")
    print(f"Encoded matrix: {X.shape}, dtype={X.dtype}")

    y_full = rows["y"].to_numpy(dtype=np.int64)
    pair_full = rows["pair_id"].to_numpy(dtype=np.int64)

    # Full ---------------------------------------------------------------
    np.save(OUT_DIR / "full_X.npy", X)
    np.save(OUT_DIR / "full_y.npy", y_full)
    np.save(OUT_DIR / "full_pair_id.npy", pair_full)
    print(f"Wrote full_* blobs ({X.shape[0]} rows)")

    # Cleaned tau=0.52 ---------------------------------------------------
    with open(TAU052_PAIR_IDS, "r", encoding="utf-8") as f:
        tau_ids = set(int(p) for p in json.load(f)["pair_ids"])
    keep = np.array([pid in tau_ids for pid in pair_full])
    X_c = X[keep]
    y_c = y_full[keep]
    p_c = pair_full[keep]
    if len(X_c) != 1056:
        raise RuntimeError(f"Cleaned row count {len(X_c)} != 1056")
    np.save(OUT_DIR / "cleaned_X.npy", X_c)
    np.save(OUT_DIR / "cleaned_y.npy", y_c)
    np.save(OUT_DIR / "cleaned_pair_id.npy", p_c)
    print(f"Wrote cleaned_* blobs ({X_c.shape[0]} rows)")

    # Sanity: check rows are still paired (y=1 immediately followed by y=0)
    # and that every unique pair_id appears twice.
    for label, X_, y_, p_ in [
        ("full", X, y_full, pair_full),
        ("cleaned", X_c, y_c, p_c),
    ]:
        u, c = np.unique(p_, return_counts=True)
        if not np.all(c == 2):
            raise RuntimeError(
                f"{label}: pair_id counts not all 2 (min={c.min()}, max={c.max()})"
            )
        print(f"  {label}: {len(u)} unique pair_ids, each appears twice. "
              f"y=1 count={int((y_==1).sum())}, y=0 count={int((y_==0).sum())}")

    print("\nStep 6 complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in build_bge_embeddings.py:", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
