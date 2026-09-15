#!/usr/bin/env python3
"""Rebuild the 10,000-pair MedHallu pool from the pinned Hugging Face revision.

Pair ordering is the one used by every MedHallu script in this directory and by the released
manifest: one row per upstream record, id ``mh_<config>_<row>``, rows sorted by that id as a string,
``pair_id`` = position after sorting.  ``rebuild_pool()`` therefore reproduces
``medhallu_panel/medhallu_pairs_with_questions.csv`` column for column (verified 2026-09-05).
"""
from __future__ import annotations
import os
import pandas as pd

HF_REPO = "UTAustin-AIHealth/MedHallu"
REVISION = "515060458a945c633debc6fd5baac7764416b724"   # pinned upstream revision (card: 5150604)
CONFIGS = ("pqa_labeled", "pqa_artificial")
_HERE = os.path.dirname(os.path.abspath(__file__))
POOL_CSV = os.path.join(_HERE, "medhallu_panel", "medhallu_pairs_with_questions.csv")


def rebuild_pool(revision: str = REVISION) -> pd.DataFrame:
    """Columns: pair_id, group, cfg, row, question, text_true, text_false, tier, cat, knowledge."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from datasets import load_dataset
    rows = []
    for cfg in CONFIGS:
        d = load_dataset(HF_REPO, cfg, split="train", revision=revision).to_pandas()
        for i, r in d.iterrows():
            rows.append({"group": f"mh_{cfg}_{i}", "cfg": cfg, "row": int(i), "question": r["Question"],
                         "text_true": r["Ground Truth"], "text_false": r["Hallucinated Answer"],
                         "tier": r["Difficulty Level"], "cat": r["Category of Hallucination"],
                         "knowledge": r["Knowledge"]})
    up = pd.DataFrame(rows).sort_values("group", kind="stable").reset_index(drop=True)
    up.insert(0, "pair_id", range(len(up)))
    assert len(up) == 10_000, len(up)
    return up


def check_against_repo(up: pd.DataFrame) -> None:
    """Assert the rebuilt pool equals the committed pool CSV (ids, texts, metadata)."""
    pq = pd.read_csv(POOL_CSV).sort_values("pair_id").reset_index(drop=True)
    assert (pq["pair_id"].values == up["pair_id"].values).all()
    assert (pq["medhallu_id"].values == up["group"].values).all(), "pair_id -> upstream id mapping differs"
    for c in ("text_true", "text_false", "cfg", "tier", "cat", "question"):
        n = int((pq[c].astype(str).values != up[c].astype(str).values).sum())
        assert n == 0, f"{n} mismatches in {c}"


if __name__ == "__main__":
    up = rebuild_pool()
    check_against_repo(up)
    print("pool rebuilt from", HF_REPO, "@", REVISION[:7], "and identical to", os.path.relpath(POOL_CSV))
