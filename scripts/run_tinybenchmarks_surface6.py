#!/usr/bin/env python3
"""Table 8 (IRT anchors) with the paper's Surface6 feature list.

run_tinybenchmarks_truthfulqa_surface_audit.py audits with audit_subset_evaluator.FEATURE_COLS, the
ten-feature list of the original code base; the paper's Table 8 uses the six Surface6 features. This
wrapper sets that list (as run_audit_prune_surface6.py does) and points the reference subset at the
released TruthfulQA-476 manifest. Downloads tinyBenchmarks/tinyTruthfulQA (100 anchors) on first run;
writes results/tinybenchmarks_audit/.

    python scripts/run_tinybenchmarks_surface6.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import audit_subset_evaluator as ase
ase.FEATURE_COLS[:] = ["word_count", "neg_cnt", "avg_token_len", "type_token", "neg_lead", "hedge_rate"]
import run_tinybenchmarks_truthfulqa_surface_audit as drv
if __name__ == "__main__":
    if not any(a.startswith("--tau053-pair-json") for a in sys.argv[1:]):
        sys.argv += ["--tau053-pair-json", "data/subsets/TruthfulQA-Audited/surface6/pair_ids/pair_ids_theta053.json"]
    sys.exit(drv.main())
