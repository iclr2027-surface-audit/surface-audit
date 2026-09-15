#!/usr/bin/env python3
"""Run the generic Audit-Prune threshold driver with the paper's Surface6 feature list.

run_audit_prune_thresholded.py scores pairs with audit_subset_evaluator.FEATURE_COLS, which is the
ten-feature list of the original code base. The paper's runs use the six Surface6 features (the same
list reproduce_truthfulqa476.py uses). This wrapper sets that list and forwards all arguments, e.g.

    python scripts/run_audit_prune_surface6.py --thresholds 0.53
    python scripts/run_audit_prune_surface6.py --strategies confidence imbalance --thresholds 0.60 0.55 0.53 0.52
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import audit_subset_evaluator as ase
ase.FEATURE_COLS[:] = ["word_count", "neg_cnt", "avg_token_len", "type_token", "neg_lead", "hedge_rate"]
import run_audit_prune_thresholded as drv
if __name__ == "__main__":
    drv.main()
