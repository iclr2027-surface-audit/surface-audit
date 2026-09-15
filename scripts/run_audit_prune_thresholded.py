#!/usr/bin/env python3
"""
Classifier-guided subset selection: maximize retained pairs subject to grouped CV audit AUC <= tau.

Greedy removal (confidence or feature-imbalance score), then local add-back refinement.
Baseline comparison: maximum prefix length K on the released feature-balanced ordering (seed 42).

Does not modify truthfulqaPro/ or existing baseline scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from audit_subset_evaluator import (
    SubsetAuditMetrics,
    df_pairs_from_ids,
    evaluate_subset_grouped_cv,
    oof_pair_confidence_scores,
    oof_pair_imbalance_scores,
    pick_pair_to_remove,
)
from search_truthfulqa_pruned_improved import _apply_prefix_keep
from truthfulqa_pruning_utils import CV_SPLITS, load_candidates_with_features, repo_root

StrategyName = Literal["confidence", "imbalance"]

TAU_EPS = 1e-9

BASE_COLS = [
    "Type",
    "Category",
    "Question",
    "Best Answer",
    "Best Incorrect Answer",
]


def sorted_df_feature_balanced_baseline(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Same ordering as ``export_feature_balanced_subset_csvs.sorted_df_feature_balanced``:
    qcut length_gap quartiles, shuffle within bin with seed+ b, sort negation_flag, length_gap, example_id.
    """
    d = df.copy()
    d["len_bin"] = pd.qcut(d["length_gap"], q=4, labels=False, duplicates="drop")
    parts = []
    for b in sorted(d["len_bin"].dropna().unique()):
        sub = d[d["len_bin"] == b]
        parts.append(sub.sample(frac=1.0, random_state=seed + int(b)))
    mixed = pd.concat(parts, ignore_index=True)
    return mixed.sort_values(["negation_flag", "length_gap", "example_id"], ascending=[True, True, True]).reset_index(
        drop=True
    )


def baseline_max_prefix_k(
    d_sorted: pd.DataFrame, tau: float, seed: int
) -> Tuple[Optional[int], SubsetAuditMetrics]:
    """
    Largest K such that grouped CV OOF AUC on the first K pairs of ``d_sorted`` is <= tau.
    Scans K ascending and keeps the last feasible (maximum K).
    """
    n_full = len(d_sorted)
    best_k: Optional[int] = None
    best_m = SubsetAuditMetrics(auc=float("nan"), accuracy=float("nan"), n_pairs=0, n_answer_rows=0)
    for K in range(CV_SPLITS, n_full + 1):
        sub = _apply_prefix_keep(d_sorted, K)
        m = evaluate_subset_grouped_cv(sub, seed)
        if m.auc <= tau + TAU_EPS:
            best_k = K
            best_m = m
    return best_k, best_m


def score_pairs_for_strategy(df_retained: pd.DataFrame, strategy: StrategyName, seed: int) -> Dict[int, float]:
    if strategy == "confidence":
        return oof_pair_confidence_scores(df_retained, seed)
    if strategy == "imbalance":
        return oof_pair_imbalance_scores(df_retained, seed)
    raise ValueError(strategy)


def greedy_prune_until_threshold(
    full_df: pd.DataFrame,
    tau: float,
    strategy: StrategyName,
    seed: int,
    log_every: int = 25,
) -> Tuple[Set[int], List[int]]:
    """Returns (retained_ids, removal_order)."""
    retained: Set[int] = set(int(x) for x in full_df["example_id"].tolist())
    removal_order: List[int] = []
    step = 0
    while True:
        sub = df_pairs_from_ids(full_df, retained)
        m = evaluate_subset_grouped_cv(sub, seed)
        if m.auc <= tau + TAU_EPS:
            break
        if len(retained) <= CV_SPLITS:
            raise RuntimeError(
                f"Cannot reach AUC <= {tau}: still at {m.auc} with only {len(retained)} pairs (minimum {CV_SPLITS})."
            )
        scores = score_pairs_for_strategy(sub, strategy, seed)
        pid = pick_pair_to_remove(scores, retained)
        retained.remove(pid)
        removal_order.append(pid)
        step += 1
        if log_every and step % log_every == 0:
            m2 = evaluate_subset_grouped_cv(df_pairs_from_ids(full_df, retained), seed)
            print(f"  prune step {step}: removed up to pair {pid}, |S|={len(retained)}, auc={m2.auc:.4f}")
    return retained, removal_order


def add_back_refinement(
    full_df: pd.DataFrame,
    retained: Set[int],
    removed: Set[int],
    tau: float,
    seed: int,
) -> Set[int]:
    """Greedy add-back in deterministic order (ascending pair_id) until stable."""
    ret = set(retained)
    rem = set(removed)
    while True:
        added_any = False
        for pid in sorted(rem):
            trial = ret | {pid}
            sub = df_pairs_from_ids(full_df, trial)
            m = evaluate_subset_grouped_cv(sub, seed)
            if m.auc <= tau + TAU_EPS:
                ret.add(pid)
                rem.remove(pid)
                added_any = True
        if not added_any:
            break
    return ret


def run_one_threshold_cached_baseline(
    full_df: pd.DataFrame,
    tau: float,
    strategy: StrategyName,
    seed: int,
    bk: Optional[int],
    bm: SubsetAuditMetrics,
    n_full: int,
) -> Dict[str, Any]:
    """Prune + refine; dominance vs precomputed baseline max-prefix scan."""
    print(f"\n=== tau={tau} strategy={strategy} ===")
    retained, removal_order = greedy_prune_until_threshold(full_df, tau, strategy, seed)
    removed_before = set(int(x) for x in full_df["example_id"].tolist()) - retained
    retained_after = add_back_refinement(full_df, retained, removed_before, tau, seed)
    m_new = evaluate_subset_grouped_cv(df_pairs_from_ids(full_df, retained_after), seed)

    dominates = (
        bk is not None
        and len(retained_after) > bk
        and m_new.auc <= tau + TAU_EPS
        and np.isfinite(bm.auc)
        and bm.auc <= tau + TAU_EPS
    )
    return {
        "tau": tau,
        "strategy": strategy,
        "pruned_pair_count": len(retained_after),
        "pruned_auc": m_new.auc,
        "pruned_accuracy": m_new.accuracy,
        "pruned_retained_fraction": len(retained_after) / max(1, n_full),
        "removal_steps": len(removal_order),
        "add_back_count": len(retained_after) - len(retained),
        "dominates_baseline": dominates,
        "retained_ids": sorted(retained_after),
    }


def tau_slug(tau: float) -> str:
    return f"tau{int(round(tau * 1000)):04d}"


def write_subset_artifacts(
    root: Path,
    pair_ids: List[int],
    tau: float,
    strategy: StrategyName,
    seed: int,
    df_t: pd.DataFrame,
    style: np.ndarray,
    summary_row: Dict[str, Any],
) -> Tuple[str, str]:
    """Write CSV + JSON; return (csv_rel, json_rel) paths relative to root."""
    out_dir = root / "truthfulqaAuditPrune"
    json_dir = out_dir / "pair_ids"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    slug = f"{tau_slug(tau)}_{strategy}"
    subset_name = f"truthfulqaAuditPrune_{slug}"
    csv_name = f"{subset_name}.csv"
    jname = f"pair_ids_{slug}_seed{seed}.json"
    cpath = out_dir / csv_name
    jpath = json_dir / jname

    fieldnames = ["pair_id", *BASE_COLS, "style_violation", "subset_name"]
    with cpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pid in pair_ids:
            row = df_t.iloc[pid]
            w.writerow(
                {
                    "pair_id": pid,
                    **{c: row[c] for c in BASE_COLS},
                    "style_violation": int(style[pid]),
                    "subset_name": subset_name,
                }
            )

    meta = {
        "selection_method": "audit_prune_thresholded",
        "removal_scoring": strategy,
        "audit_threshold_auc": tau,
        "lr_random_state": seed,
        "baseline_prefix_ordering_seed": summary_row.get("baseline_prefix_seed"),
        "pair_ids": [int(x) for x in pair_ids],
        "n_pairs": len(pair_ids),
        "grouped_cv_oof_auc": summary_row.get("pruned_auc"),
        "grouped_cv_oof_accuracy": summary_row.get("pruned_accuracy"),
        "comparison": {
            "baseline_max_prefix_k": summary_row.get("baseline_max_k"),
            "baseline_auc": summary_row.get("baseline_auc"),
            "dominates_baseline": summary_row.get("dominates_baseline"),
        },
    }
    jpath.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return f"truthfulqaAuditPrune/{csv_name}", f"truthfulqaAuditPrune/pair_ids/{jname}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truthfulqa-csv", type=str, default="TruthfulQA.csv")
    p.add_argument("--audit-csv", type=str, default="audits/truthfulqa_style_audit.csv")
    p.add_argument("--seed", type=int, default=42, help="LR + CV random state for audit metrics and confidence OOF.")
    p.add_argument("--baseline-prefix-seed", type=int, default=42, help="Seed for qcut/shuffle/sort baseline ordering.")
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.60, 0.57, 0.55, 0.53],
    )
    p.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["confidence", "imbalance"],
        choices=["confidence", "imbalance"],
    )
    p.add_argument("--output-root", type=str, default=".", help="Repository root for truthfulqaAuditPrune/ and results/.")
    p.add_argument(
        "--results-subdir",
        type=str,
        default="audit_prune_thresholded",
        help="Subdirectory under results/ for summary CSVs and config.json",
    )
    p.add_argument("--skip-artifacts", action="store_true", help="Skip writing CSV/JSON/manifest (metrics only).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_root).resolve()
    tq_path = (root / args.truthfulqa_csv).resolve()
    audit_path = (root / args.audit_csv).resolve()
    full_df = load_candidates_with_features(tq_path, audit_path)
    df_t = pd.read_csv(tq_path)
    audit_t = pd.read_csv(audit_path)
    style = audit_t["style_violation"].to_numpy()

    d_sorted = sorted_df_feature_balanced_baseline(full_df, args.baseline_prefix_seed)

    results_dir = root / "results" / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "truthfulqa_csv": str(args.truthfulqa_csv),
        "audit_csv": str(args.audit_csv),
        "seed": args.seed,
        "baseline_prefix_seed": args.baseline_prefix_seed,
        "thresholds": list(args.thresholds),
        "strategies": list(args.strategies),
        "min_pairs_for_cv": CV_SPLITS,
        "evaluator": "audit_subset_evaluator.evaluate_subset_grouped_cv + search_truthfulqa_pruned_improved._ans_frame",
    }
    (results_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    summary_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    for tau in args.thresholds:
        bk, bm = baseline_max_prefix_k(d_sorted, tau, args.baseline_prefix_seed)
        n_full = len(full_df)
        baseline_row = {
            "baseline_max_prefix_k": bk,
            "baseline_auc": bm.auc if bk is not None else float("nan"),
            "baseline_accuracy": bm.accuracy if bk is not None else float("nan"),
            "baseline_retained_fraction": (bk / n_full) if bk is not None else float("nan"),
        }
        print(
            f"\n--- tau={tau}: baseline max-prefix K={bk} "
            f"auc={baseline_row['baseline_auc']} acc={baseline_row['baseline_accuracy']} ---"
        )
        for strategy in args.strategies:
            st: StrategyName = strategy  # type: ignore[assignment]
            out = run_one_threshold_cached_baseline(
                full_df, tau, st, args.seed, bk, bm, n_full
            )
            row = {
                "tau": out["tau"],
                "strategy": out["strategy"],
                "pruned_pair_count": out["pruned_pair_count"],
                "pruned_auc": out["pruned_auc"],
                "pruned_accuracy": out["pruned_accuracy"],
                "pruned_retained_fraction": out["pruned_retained_fraction"],
                "removal_steps": out["removal_steps"],
                "add_back_pairs_recovered": out["add_back_count"],
                "baseline_max_prefix_k": baseline_row["baseline_max_prefix_k"],
                "baseline_auc": baseline_row["baseline_auc"],
                "baseline_accuracy": baseline_row["baseline_accuracy"],
                "baseline_retained_fraction": baseline_row["baseline_retained_fraction"],
                "dominates_baseline": out["dominates_baseline"],
                "seed": args.seed,
                "baseline_prefix_seed": args.baseline_prefix_seed,
            }
            summary_rows.append(row)

            if not args.skip_artifacts:
                pair_ids = out["retained_ids"]
                csv_rel, json_rel = write_subset_artifacts(
                    root,
                    pair_ids,
                    tau,
                    st,
                    args.seed,
                    df_t,
                    style,
                    {**row, "baseline_prefix_seed": args.baseline_prefix_seed},
                )
                slug = f"{tau_slug(tau)}_{st}"
                manifest_rows.append(
                    {
                        "subset_name": f"truthfulqaAuditPrune_{slug}",
                        "tau": tau,
                        "strategy": st,
                        "csv_path": csv_rel,
                        "canonical_json": json_rel,
                        "pruned_pair_count": len(pair_ids),
                        "grouped_cv_oof_auc": out["pruned_auc"],
                        "grouped_cv_oof_accuracy": out["pruned_accuracy"],
                        "baseline_max_prefix_k": bk,
                        "baseline_auc": baseline_row["baseline_auc"],
                        "dominates_baseline": out["dominates_baseline"],
                        "paper_role": "audit_prune_thresholded",
                    }
                )

    summ_path = results_dir / "summary_table.csv"
    if summary_rows:
        with summ_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
    print(f"\nWrote {summ_path}")

    cmp_path = results_dir / "comparison_vs_baseline.csv"
    with cmp_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "tau",
            "strategy",
            "new_pair_count",
            "new_auc",
            "new_accuracy",
            "new_retained_fraction",
            "baseline_max_prefix_k",
            "baseline_auc",
            "baseline_accuracy",
            "baseline_retained_fraction",
            "dominates_baseline",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(
                {
                    "tau": r["tau"],
                    "strategy": r["strategy"],
                    "new_pair_count": r["pruned_pair_count"],
                    "new_auc": r["pruned_auc"],
                    "new_accuracy": r["pruned_accuracy"],
                    "new_retained_fraction": r["pruned_retained_fraction"],
                    "baseline_max_prefix_k": r["baseline_max_prefix_k"],
                    "baseline_auc": r["baseline_auc"],
                    "baseline_accuracy": r["baseline_accuracy"],
                    "baseline_retained_fraction": r["baseline_retained_fraction"],
                    "dominates_baseline": r["dominates_baseline"],
                }
            )
    print(f"Wrote {cmp_path}")

    if manifest_rows:
        man_path = root / "truthfulqaAuditPrune" / "subset_manifest.csv"
        man_path.parent.mkdir(parents=True, exist_ok=True)
        with man_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            for r in manifest_rows:
                w.writerow(r)
        print(f"Wrote {man_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
