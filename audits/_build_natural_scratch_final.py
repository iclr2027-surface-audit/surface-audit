#!/usr/bin/env python3
"""
Stage 5 of the natural-from-scratch 135-pair build.

From audits/_eval_dataset_natural_scratch_judged.json, take judge-passing
rows (all three booleans true AND confidence >= 0.8), then for each
topic_domain take the first N rows up to that topic's target count.

Output: audits/_eval_dataset_natural_scratch_final.csv
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JUDGED = REPO_ROOT / "audits" / "_eval_dataset_natural_scratch_judged.json"
OUT_CSV = REPO_ROOT / "audits" / "_eval_dataset_natural_scratch_final.csv"

TARGETS = [
    ("Animal Myths",     "Animal biology and behavior",            24),
    ("Health Myths",     "Human health and medicine",              19),
    ("Food Myths",       "Food and nutrition",                     14),
    ("Pseudoscience",    "Science and physics",                    14),
    ("Body Myths",       "Human anatomy and physiology",           11),
    ("History Myths",    "Historical events and figures",          11),
    ("Superstitions",    "Cultural traditions and practices",      10),
    ("Folk Wisdom",      "Everyday knowledge and practical facts",  8),
    ("Misconceptions",   "General knowledge",                       8),
    ("Mandela Effect",   "Pop culture and media history",           7),
    ("Urban Legends",    "Modern history and society",              6),
    ("Language Myths",   "Language and linguistics",                3),
]


def main():
    data = json.loads(JUDGED.read_text(encoding="utf-8"))
    verdicts = data["verdicts"]

    passing = [v for v in verdicts
               if v["answer_a_is_true"] and v["answer_b_is_false"]
               and v["same_question"] and v["confidence"] >= 0.8]
    passing.sort(key=lambda v: int(v["id"]))

    by_topic: dict[str, list] = {}
    for v in passing:
        by_topic.setdefault(v["topic_domain"], []).append(v)

    final_rows = []
    short = []
    for source_cat, topic, target in TARGETS:
        pool = by_topic.get(topic, [])
        take = pool[:target]
        if len(take) < target:
            short.append((topic, len(take), target))
        for v in take:
            final_rows.append({
                "id": int(v["id"]),
                "source_category": source_cat,
                "topic_domain": topic,
                "question": v["question"],
                "true_answer_natural":  v["answer_a"],
                "false_answer_natural": v["answer_b"],
                "judge_confidence": float(v["confidence"]),
                "judge_rationale": v["rationale"],
            })

    final_rows.sort(key=lambda r: int(r["id"]))

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
        w.writeheader()
        for r in final_rows:
            w.writerow(r)

    print(f"Wrote {OUT_CSV.relative_to(REPO_ROOT)}  ({len(final_rows)} rows)")
    print()
    print("Per-topic counts (final vs target):")
    by_topic_final = {}
    for r in final_rows:
        by_topic_final[r["topic_domain"]] = by_topic_final.get(r["topic_domain"], 0) + 1
    for source_cat, topic, target in TARGETS:
        n = by_topic_final.get(topic, 0)
        ok = "OK" if n == target else f"SHORT(-{target-n})"
        print(f"  {source_cat:18s} -> {topic:42s}  n={n:>3}/{target:<3}  {ok}")

    if short:
        print()
        print("WARNING: short topics (took what was available, no regenerate):")
        for topic, got, target in short:
            print(f"  {topic:42s}  got={got}  target={target}  short=-{target-got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
