#!/usr/bin/env python3
"""
Combine v3 batch1 (clean, n=20) + v3 batch2 (drift-screened) into a single
final CSV with a `batch` column. Dedupe by normalized question (keep batch1
on collision).

Inputs:
  audits/_eval_dataset_v3_clean_final.csv             (batch1)
  audits/_eval_dataset_v3_batch2_drift_screened.csv  (batch2 kept rows)

Output:
  audits/_eval_dataset_v3_combined_final.csv
"""
from __future__ import annotations
import re, string, sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

B1_CSV = REPO / "audits" / "_eval_dataset_v3_clean_final.csv"
B2_CSV = REPO / "audits" / "_eval_dataset_v3_batch2_drift_screened.csv"
OUT_CSV = REPO / "audits" / "_eval_dataset_v3_combined_final.csv"


def _norm_q(q: str) -> str:
    q = (q or "").lower().strip()
    q = q.translate(str.maketrans("", "", string.punctuation))
    q = re.sub(r"\s+", " ", q).strip()
    return q


def main():
    b1 = pd.read_csv(B1_CSV)
    b2 = pd.read_csv(B2_CSV)
    b1 = b1.copy(); b1["batch"] = 1
    b2 = b2.copy(); b2["batch"] = 2
    print(f"batch1 rows: {len(b1)}")
    print(f"batch2 rows: {len(b2)}")

    seen = set()
    keep_rows = []
    n_dup = 0
    for _, r in pd.concat([b1, b2], ignore_index=True).iterrows():
        nq = _norm_q(str(r["question"]))
        if nq in seen:
            n_dup += 1
            continue
        seen.add(nq)
        keep_rows.append(r)
    out = pd.DataFrame(keep_rows)

    # Reorder columns: keep batch column near the end before any judge stats.
    cols = list(out.columns)
    if "batch" in cols:
        cols.remove("batch")
        cols.append("batch")
        out = out[cols]

    out.to_csv(OUT_CSV, index=False)
    print(f"\n=== Combine summary ===")
    print(f"  batch1            : {len(b1)}")
    print(f"  batch2 (kept)     : {len(b2)}")
    print(f"  duplicates removed: {n_dup}")
    print(f"  combined final    : {len(out)}")
    if "category" in out.columns:
        print(f"\nPer-category breakdown:")
        for k, v in out["category"].value_counts().sort_values(ascending=False).items():
            print(f"  {k:24s} {v}")
    print(f"\nWrote {OUT_CSV.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
