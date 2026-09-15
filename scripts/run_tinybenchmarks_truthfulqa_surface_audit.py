#!/usr/bin/env python3
"""
Audit TinyBenchmarks tinyTruthfulQA IRT anchors vs local TruthfulQA.csv using the
same grouped-CV surface audit as audit_subset_evaluator (surface10-style _ans_frame).

Also runs 10 random 100-pair controls from the full 790 for baseline AUC mean ± std.

Requires: pip install datasets (see requirements-paper-full.txt in this repo).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from audit_subset_evaluator import SubsetAuditMetrics, df_pairs_from_ids, evaluate_subset_grouped_cv
from truthfulqa_pruning_utils import load_candidates_with_features

PAPER_FULL_TABLE_AUC = 0.713  # paper_assets feature table (may differ from _ans_frame pipeline)

HF_DATASET_ID = "tinyBenchmarks/tinyTruthfulQA"
HF_VALIDATION_SPLIT = "validation"


def norm_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" in the california", " in california")
    s = s.rstrip(".")
    return s


def parse_mc1(mc: Dict[str, Any]) -> Tuple[str, List[str]]:
    choices = list(mc["choices"])
    labels = list(mc["labels"])
    correct: Optional[str] = None
    wrong: List[str] = []
    for c, lab in zip(choices, labels):
        if int(lab) == 1:
            if correct is not None:
                raise ValueError("multiple correct labels in mc1_targets")
            correct = norm_text(c)
        else:
            wrong.append(norm_text(c))
    if correct is None:
        raise ValueError("no correct label in mc1_targets")
    return correct, wrong


def build_answer_index(tq: pd.DataFrame) -> Dict[Tuple[str, str], List[Tuple[int, str]]]:
    """(norm_question, norm_best_answer) -> [(example_id, norm_incorrect), ...]"""
    idx: Dict[Tuple[str, str], List[Tuple[int, str]]] = defaultdict(list)
    for i, row in tq.iterrows():
        q = norm_text(row["Question"])
        a = norm_text(row["Best Answer"])
        b = norm_text(row["Best Incorrect Answer"])
        idx[(q, a)].append((int(i), b))
    return idx


def match_hf_row_to_pair_id(
    idx: Dict[Tuple[str, str], List[Tuple[int, str]]],
    question: str,
    correct: str,
    wrongs: List[str],
) -> Optional[int]:
    key = (norm_text(question), correct)
    if key not in idx:
        return None
    for pid, inc in idx[key]:
        if inc in wrongs:
            return pid
    return None


def load_hf_validation() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "The 'datasets' package is required. Install with: pip install datasets\n"
            "Or use the paper environment (requirements-paper-full.txt)."
        ) from e
    return load_dataset(HF_DATASET_ID)[HF_VALIDATION_SPLIT]


def _match_status_counts(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Return (n_unmatched, n_duplicate_skipped)."""
    u = sum(1 for r in rows if r["status"] == "unmatched")
    d = sum(1 for r in rows if r["status"] == "duplicate_pair_id_skipped")
    return u, d


def build_hf_match_report(
    hf_ds: Any,
    idx: Dict[Tuple[str, str], List[Tuple[int, str]]],
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Walk HF validation rows, map each to a local ``example_id`` (pair_id) when possible.

    First HF row that maps to a given pair_id is ``matched``; later HF rows that map to the
    same pair_id are ``duplicate_pair_id_skipped`` so anchors stay unique for evaluation.
    """
    match_rows: List[Dict[str, Any]] = []
    anchor_ids: List[int] = []
    seen_ids: Set[int] = set()
    for j in range(len(hf_ds)):
        ex = hf_ds[j]
        q = ex["question"]
        cor, wr = parse_mc1(ex["mc1_targets"])
        pid = match_hf_row_to_pair_id(idx, q, cor, wr)
        status = "matched" if pid is not None else "unmatched"
        if pid is not None and pid not in seen_ids:
            seen_ids.add(pid)
            anchor_ids.append(pid)
        elif pid is not None:
            status = "duplicate_pair_id_skipped"
        match_rows.append(
            {
                "hf_row_index": j,
                "hf_question": q,
                "status": status,
                "matched_pair_id": pid if pid is not None else "",
                "hf_correct_norm": cor,
                "hf_wrong_norm_count": len(wr),
            }
        )
    return match_rows, anchor_ids


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=str, default=".", help="Repository root")
    p.add_argument("--truthfulqa-csv", type=str, default="TruthfulQA.csv")
    p.add_argument("--audit-csv", type=str, default="audits/truthfulqa_style_audit.csv")
    p.add_argument("--lr-seed", type=int, default=42)
    p.add_argument(
        "--tau053-pair-json",
        type=str,
        default="truthfulqaAuditPrune/pair_ids/pair_ids_tau0530_imbalance_seed42.json",
        help="Reference audited subset pair_ids JSON",
    )
    p.add_argument("--n-random-controls", type=int, default=10)
    p.add_argument("--random-subset-size", type=int, default=100)
    p.add_argument("--random-base-seed", type=int, default=42)
    args = p.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / "results" / "tinybenchmarks_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load local benchmark + audit features (790 pairs).
    tq_path = (root / args.truthfulqa_csv).resolve()
    audit_path = (root / args.audit_csv).resolve()
    full_df = load_candidates_with_features(tq_path, audit_path)
    n_full = len(full_df)

    # Map HF IRT anchors to local pair_ids.
    hf_ds = load_hf_validation()
    if len(hf_ds) == 0:
        raise SystemExit(f"HF dataset {HF_DATASET_ID!r} split {HF_VALIDATION_SPLIT!r} is empty.")
    tq = pd.read_csv(tq_path)
    idx = build_answer_index(tq)
    match_rows, anchor_ids = build_hf_match_report(hf_ds, idx)
    anchor_ids_sorted = sorted(set(anchor_ids))
    n_unmatched_hf, n_dup_skipped = _match_status_counts(match_rows)

    # Persist HF→local match table and anchor id list for downstream use.
    with (out_dir / "anchor_pair_ids.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "hf_dataset": HF_DATASET_ID,
                "hf_split": HF_VALIDATION_SPLIT,
                "n_hf_rows": len(hf_ds),
                "n_matched_unique_pair_ids": len(anchor_ids_sorted),
                "n_unmatched": n_unmatched_hf,
                "pair_ids": anchor_ids_sorted,
                "lr_seed": args.lr_seed,
            },
            f,
            indent=2,
        )
        f.write("\n")

    match_path = out_dir / "anchor_match_report.csv"
    if match_rows:
        with match_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(match_rows[0].keys()))
            w.writeheader()
            for r in match_rows:
                w.writerow(r)

    # Grouped CV on anchors, full CSV, reference audited subset, then random baselines.
    df_anchor = df_pairs_from_ids(full_df, anchor_ids_sorted)
    m_anchor = evaluate_subset_grouped_cv(df_anchor, args.lr_seed)

    m_full = evaluate_subset_grouped_cv(full_df, args.lr_seed)

    tau_json = (root / args.tau053_pair_json).resolve()
    tau_ids = set(json.loads(tau_json.read_text(encoding="utf-8"))["pair_ids"])
    df_tau = df_pairs_from_ids(full_df, tau_ids)
    m_tau = evaluate_subset_grouped_cv(df_tau, args.lr_seed)

    rng_seeds = [args.random_base_seed + k for k in range(args.n_random_controls)]
    ctrl_aucs: List[float] = []
    ctrl_accs: List[float] = []
    summary_rows: List[Dict[str, Any]] = []

    def add_row(name: str, stype: str, n: int, m: SubsetAuditMetrics, notes: str = "") -> None:
        summary_rows.append(
            {
                "subset_name": name,
                "subset_type": stype,
                "n_pairs_evaluated": n,
                "grouped_cv_auc": m.auc,
                "grouped_cv_accuracy": m.accuracy,
                "lr_seed": args.lr_seed,
                "notes": notes,
            }
        )

    add_row(
        "full_truthfulqa_local_csv",
        "reference_full",
        m_full.n_pairs,
        m_full,
        f"Same pipeline as anchor; paper table full-model AUC≈{PAPER_FULL_TABLE_AUC} may differ slightly.",
    )
    add_row(
        "audit_prune_tau0530_imbalance",
        "reference_audited_subset",
        m_tau.n_pairs,
        m_tau,
        f"From {args.tau053_pair_json}",
    )
    add_row(
        "tinyTruthfulQA_hf_matched_to_local",
        "irt_anchor_matched",
        m_anchor.n_pairs,
        m_anchor,
        f"HF validation n={len(hf_ds)}; unique local matches={len(anchor_ids_sorted)}; see anchor_match_report.csv",
    )

    for k, rs in enumerate(rng_seeds):
        rng = np.random.default_rng(rs)
        pick = rng.choice(n_full, size=args.random_subset_size, replace=False)
        pids = [int(full_df.iloc[i]["example_id"]) for i in pick]
        m_r = evaluate_subset_grouped_cv(df_pairs_from_ids(full_df, pids), args.lr_seed)
        ctrl_aucs.append(m_r.auc)
        ctrl_accs.append(m_r.accuracy)
        add_row(f"random100_control_{k:02d}", "random_control", m_r.n_pairs, m_r, f"rng_seed={rs}")

    auc_mean = float(np.mean(ctrl_aucs))
    auc_std = float(np.std(ctrl_aucs, ddof=1)) if len(ctrl_aucs) > 1 else 0.0
    acc_mean = float(np.mean(ctrl_accs))
    acc_std = float(np.std(ctrl_accs, ddof=1)) if len(ctrl_accs) > 1 else 0.0
    summary_rows.append(
        {
            "subset_name": "random100_control_aggregate",
            "subset_type": "random_control_aggregate",
            "n_pairs_evaluated": args.random_subset_size,
            "grouped_cv_auc": auc_mean,
            "grouped_cv_accuracy": acc_mean,
            "lr_seed": args.lr_seed,
            "notes": f"mean±std over {args.n_random_controls} controls; auc_std={auc_std:.6f} acc_std={acc_std:.6f}; rng_seeds={rng_seeds}",
        }
    )

    main_csv = out_dir / "tinybenchmarks_surface_audit.csv"
    fields = list(summary_rows[0].keys())
    with main_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    cfg = {
        "hf_dataset": HF_DATASET_ID,
        "truthfulqa_csv": str(args.truthfulqa_csv),
        "audit_csv": str(args.audit_csv),
        "lr_seed": args.lr_seed,
        "n_hf_validation_rows": len(hf_ds),
        "n_matched_anchor_pair_ids": len(anchor_ids_sorted),
        "n_unmatched_hf_rows": n_unmatched_hf,
        "n_duplicate_hf_to_same_pair_id": n_dup_skipped,
        "random_control_runs": args.n_random_controls,
        "random_subset_size": args.random_subset_size,
        "random_rng_seeds": rng_seeds,
        "random_control_auc_mean": auc_mean,
        "random_control_auc_std": auc_std,
        "random_control_acc_mean": acc_mean,
        "random_control_acc_std": acc_std,
        "paper_feature_table_full_auc": PAPER_FULL_TABLE_AUC,
        "evaluator": "audit_subset_evaluator.evaluate_subset_grouped_cv + _ans_frame",
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    print("\n=== TinyTruthfulQA surface audit (grouped CV, same pipeline) ===\n")
    print(f"HF rows: {len(hf_ds)} | Matched unique local pair_ids: {len(anchor_ids_sorted)} | Unmatched: {cfg['n_unmatched_hf_rows']}")
    print(f"\n{'subset_name':<42} {'n':>4} {'AUC':>8} {'Acc':>8}")
    print("-" * 70)
    for r in summary_rows:
        if r["subset_type"] == "random_control_aggregate":
            print(
                f"{r['subset_name']:<42} {r['n_pairs_evaluated']:>4} "
                f"{float(r['grouped_cv_auc']):>8.4f} {float(r['grouped_cv_accuracy']):>8.4f}  (mean; std in notes)"
            )
        else:
            print(
                f"{r['subset_name']:<42} {r['n_pairs_evaluated']:>4} "
                f"{float(r['grouped_cv_auc']):>8.4f} {float(r['grouped_cv_accuracy']):>8.4f}"
            )
    print(f"\nRandom 100-pair baseline: AUC mean={auc_mean:.4f} std={auc_std:.4f} | Acc mean={acc_mean:.4f} std={acc_std:.4f}")
    print(f"IRT matched anchors: AUC={m_anchor.auc:.4f} (delta vs random mean: {m_anchor.auc - auc_mean:+.4f})")
    print(f"Paper feature-table full-dataset AUC (reference): {PAPER_FULL_TABLE_AUC:.3f} | Same-pipeline full CSV AUC: {m_full.auc:.4f}")
    print(f"Strong audited subset (tau=0.53 imbalance, n={m_tau.n_pairs}): AUC={m_tau.auc:.4f}")
    print(
        f"\nMatch summary: unmatched={cfg['n_unmatched_hf_rows']} "
        f"duplicate_hf_to_same_pair_id={cfg['n_duplicate_hf_to_same_pair_id']}"
    )
    print(f"\nWrote: {main_csv}")
    if match_rows:
        print(f"Wrote: {match_path}")
    print(f"Wrote: {out_dir / 'anchor_pair_ids.json'}")
    print(f"Wrote: {out_dir / 'config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
