#!/usr/bin/env python3
"""
Build the n=full inverted-accuracy cohort.

Combines all judge-passing rows (i.e. before the drift screen) for batches
1, 4, 5..13. Drift-dropped rows are included intentionally: top-1 inverted
accuracy depends only on the inverted texts having opposite polarity, not
on the FALSE_plain → FALSE_inv proposition match that the drift screen
guards. Dedups by normalized question.

Inputs:
  audits/_eval_dataset_v3_clean_final.csv          (batch1 seed, n=20)
  audits/_eval_dataset_v3_batch{4..13}_final.csv   (judge-passing, pre-drift)

Output:
  audits/_eval_dataset_v3_inv_accuracy_full_final.csv
"""
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
AUD = REPO / "audits"
OUT = AUD / "_eval_dataset_v3_inv_accuracy_full_final.csv"

SOURCES = [("batch1", AUD / "_eval_dataset_v3_clean_final.csv")]
for n in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
    SOURCES.append((f"batch{n}", AUD / f"_eval_dataset_v3_batch{n}_final.csv"))


def norm_q(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def main() -> int:
    frames = []
    print("Per-source row counts:")
    for label, path in SOURCES:
        if not path.exists():
            raise SystemExit(f"missing source: {path}")
        df = pd.read_csv(path)
        df["__source"] = label
        frames.append(df)
        print(f"  {label:8s} {path.name:48s} {len(df):4d} rows")

    combined = pd.concat(frames, ignore_index=True)
    pre_dedup = len(combined)
    combined["__qnorm"] = combined["question"].apply(norm_q)
    deduped = combined.drop_duplicates("__qnorm", keep="first").copy()
    post_dedup = len(deduped)
    n_dropped = pre_dedup - post_dedup

    print()
    print(f"Pre-dedup total:  {pre_dedup}")
    print(f"Post-dedup total: {post_dedup}  (dropped {n_dropped} duplicate questions)")

    # source breakdown post-dedup
    print()
    print("Post-dedup rows kept per source:")
    for label, _ in SOURCES:
        print(f"  {label:8s} {(deduped['__source'] == label).sum():4d}")

    out_cols = [c for c in deduped.columns if c not in ("__source", "__qnorm")]
    deduped[out_cols].to_csv(OUT, index=False)
    print(f"\nWrote {OUT.relative_to(REPO)} ({post_dedup} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
