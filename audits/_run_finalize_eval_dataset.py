#!/usr/bin/env python3
"""
Final assembly: keep rows that pass mechanical + judge + confidence>=0.8,
emit audits/_eval_dataset_final.csv with a blank human_eyeball_pass column.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

VERSION = os.environ.get("EVAL_DATASET_VERSION", "v2")
VALIDATED_CSV = REPO_ROOT / "audits" / f"_eval_dataset_{VERSION}_validated.csv"
JUDGE_JSON    = REPO_ROOT / "audits" / f"_eval_dataset_{VERSION}_judged.json"
OUT_CSV       = REPO_ROOT / "audits" / f"_eval_dataset_{VERSION}_final.csv"

CONF_THRESHOLD = 0.8


def main():
    df = pd.read_csv(VALIDATED_CSV)
    n_total = len(df)
    n_mech = int(df["all_passes"].sum())

    jd = json.loads(JUDGE_JSON.read_text())
    verdicts = jd["verdicts"]
    by_id = {int(v["id"]): v for v in verdicts}

    rows = []
    for _, r in df[df["all_passes"] == True].iterrows():
        qid = int(r["id"])
        v = by_id.get(qid)
        if v is None:
            continue
        judge_passes = (v["answer_a_is_true"]
                        and v["answer_b_is_false"]
                        and v["same_question"])
        if not (judge_passes and v["confidence"] >= CONF_THRESHOLD):
            continue
        rows.append({
            "id": qid,
            "category": r["category"],
            "question": r["question"],
            "true_answer_plain":  r["true_answer_plain"],
            "false_answer_plain": r["false_answer_plain"],
            "true_text_surface_inverted":  r["true_text_surface_inverted"],
            "false_text_surface_inverted": r["false_text_surface_inverted"],
            "rationale": r["rationale"],
            "judge_confidence": v["confidence"],
            "judge_rationale":  v["rationale"],
            "true_neg_lead":   r["true_neg_lead"],
            "true_neg_cnt":    r["true_neg_cnt"],
            "true_hedge_rate": r["true_hedge_rate"],
            "true_auth_rate":  r["true_auth_rate"],
            "true_word_count": r["true_word_count"],
            "false_neg_lead":   r["false_neg_lead"],
            "false_neg_cnt":    r["false_neg_cnt"],
            "false_hedge_rate": r["false_hedge_rate"],
            "false_word_count": r["false_word_count"],
            "human_eyeball_pass": "",
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    n_judge = jd.get("n_pass_joint", 0)
    n_both  = len(rows)

    print(f"\n=== Final eval dataset summary ===")
    print(f"  generated      : {n_total}")
    print(f"  mechanical pass: {n_mech}")
    print(f"  judge pass     : {n_judge}")
    print(f"  both           : {n_both}")
    if n_both > 0:
        print(f"\nPer-category breakdown of final rows:")
        cat = out["category"].value_counts().sort_values(ascending=False)
        for k, v in cat.items():
            print(f"  {k:24s} {v}")
    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
