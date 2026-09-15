#!/usr/bin/env python3
"""
Mechanical surface10 validation of generated eval candidates.

Reads audits/_eval_dataset_raw.json. Writes
audits/_eval_dataset_validated.csv with surface10-derived check columns.
"""
from __future__ import annotations
import json, os, re, string, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from scripts.surface_features_text import extract_surface10

VERSION = os.environ.get("EVAL_DATASET_VERSION", "v2")
RAW_JSON = REPO_ROOT / "audits" / f"_eval_dataset_{VERSION}_raw.json"
OUT_CSV  = REPO_ROOT / "audits" / f"_eval_dataset_{VERSION}_validated.csv"
TQA_CSV  = REPO_ROOT / "TruthfulQA.csv"


def _norm_q(q: str) -> str:
    q = (q or "").lower().strip()
    q = q.translate(str.maketrans("", "", string.punctuation))
    q = re.sub(r"\s+", " ", q).strip()
    return q


def main():
    if OUT_CSV.exists():
        try:
            df = pd.read_csv(OUT_CSV)
            if len(df) > 0:
                print(f"Resume: {OUT_CSV.name} exists ({len(df)} rows). Skipping validation.")
                return 0
        except Exception:
            pass

    raw = json.loads(RAW_JSON.read_text())
    rows = raw["rows"] if isinstance(raw, dict) else raw
    print(f"Loaded {len(rows)} rows from {RAW_JSON.name}")

    tqa_qs = set(_norm_q(q) for q in pd.read_csv(TQA_CSV)["Question"].tolist())
    print(f"TQA reference: {len(tqa_qs)} normalized questions")

    out = []
    for r in rows:
        qid = r.get("id")
        question = r.get("question", "")
        t_text = r.get("true_text_surface_inverted", "")
        f_text = r.get("false_text_surface_inverted", "")

        t_feat = extract_surface10(t_text)
        f_feat = extract_surface10(f_text)

        q_in_tqa = _norm_q(question) in tqa_qs

        # v3 spec: TRUE 7-10 words / 0 negs / 0 hedges / 0 auth.
        #          FALSE 13-18 words / opener=1 / neg_cnt 2-3 / exactly 1 hedge
        #          (hedge_rate>0 and <=0.10 — 1/13=0.077, 2/13=0.154).
        #          PAIR: word_count(FALSE) > word_count(TRUE) AND
        #          word_count(FALSE) - word_count(TRUE) in [4, 10].
        # v2 spec (kept for legacy): TRUE 22-36 / FALSE 12-25, no hedge cap,
        #          no per-pair length constraint.
        if VERSION.startswith("v3"):
            true_passes = (
                t_feat["neg_lead"]   == 0.0 and
                t_feat["neg_cnt"]    == 0.0 and
                t_feat["hedge_rate"] == 0.0 and
                t_feat["auth_rate"]  == 0.0 and
                7 <= t_feat["word_count"] <= 10
            )
            false_passes = (
                f_feat["neg_lead"]   == 1.0 and
                2.0 <= f_feat["neg_cnt"] <= 3.0 and
                f_feat["hedge_rate"]  >  0.0 and
                f_feat["hedge_rate"]  <= 0.10 and
                13 <= f_feat["word_count"] <= 18
            )
            gap = f_feat["word_count"] - t_feat["word_count"]
            pair_passes = (gap >= 4) and (gap <= 10)
        else:
            true_passes = (
                t_feat["neg_lead"]   == 0.0 and
                t_feat["neg_cnt"]    == 0.0 and
                t_feat["hedge_rate"] == 0.0 and
                t_feat["auth_rate"]  == 0.0 and
                22 <= t_feat["word_count"] <= 36
            )
            false_passes = (
                f_feat["neg_lead"]   == 1.0 and
                f_feat["neg_cnt"]    >= 2.0 and
                f_feat["hedge_rate"] >  0.0 and
                12 <= f_feat["word_count"] <= 25
            )
            pair_passes = True
        all_passes = bool(true_passes and false_passes and pair_passes and not q_in_tqa)

        out.append({
            "id": qid,
            "category": r.get("category", ""),
            "question": question,
            "true_answer_plain":  r.get("true_answer_plain", ""),
            "false_answer_plain": r.get("false_answer_plain", ""),
            "true_text_surface_inverted":  t_text,
            "false_text_surface_inverted": f_text,
            "rationale": r.get("rationale", ""),
            "q_in_tqa": q_in_tqa,
            "true_neg_lead":   int(t_feat["neg_lead"]),
            "true_neg_cnt":    int(t_feat["neg_cnt"]),
            "true_hedge_rate": round(t_feat["hedge_rate"], 4),
            "true_auth_rate":  round(t_feat["auth_rate"],  4),
            "true_word_count": int(t_feat["word_count"]),
            "false_neg_lead":   int(f_feat["neg_lead"]),
            "false_neg_cnt":    int(f_feat["neg_cnt"]),
            "false_hedge_rate": round(f_feat["hedge_rate"], 4),
            "false_word_count": int(f_feat["word_count"]),
            "length_gap": int(f_feat["word_count"] - t_feat["word_count"]),
            "true_passes":  bool(true_passes),
            "false_passes": bool(false_passes),
            "pair_passes":  bool(pair_passes),
            "all_passes":   all_passes,
        })

    df = pd.DataFrame(out)
    df.to_csv(OUT_CSV, index=False)

    n = len(df)
    n_t = int(df["true_passes"].sum())
    n_f = int(df["false_passes"].sum())
    n_p = int(df["pair_passes"].sum())
    n_q = int(df["q_in_tqa"].sum())
    n_a = int(df["all_passes"].sum())
    print(f"\nValidation summary (VERSION={VERSION}):")
    print(f"  total          : {n}")
    print(f"  true_passes    : {n_t}/{n}  ({n_t/n:.3f})")
    print(f"  false_passes   : {n_f}/{n}  ({n_f/n:.3f})")
    print(f"  pair_passes    : {n_p}/{n}  ({n_p/n:.3f})")
    print(f"  q_in_tqa       : {n_q}/{n}")
    print(f"  all_passes     : {n_a}/{n}  ({n_a/n:.3f})")
    print(f"  length_gap     : min={int(df['length_gap'].min())}  "
          f"mean={df['length_gap'].mean():.2f}  max={int(df['length_gap'].max())}")
    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
