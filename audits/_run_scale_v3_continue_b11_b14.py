#!/usr/bin/env python3
"""
Continuation: run batches 11..14 on top of existing _eval_dataset_v3_scale_combined.csv.

Changes vs the first scale run:
  - START_BATCH=11, END_BATCH=14, TARGET_N=100, MIN_YIELD_PCT=0.15
  - Drift screen runs with DRIFT_CONF_THRESHOLD=0.80 (only drop when
    drift=true AND drift_confidence >= 0.80)
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AUD = REPO / "audits"
sys.path.insert(0, str(AUD))

import _run_scale_v3_to_150 as base  # noqa: E402

# override config
base.START_BATCH = 11
base.END_BATCH = 14
base.TARGET_N = 100
base.MIN_YIELD_PCT = 0.15

# patch run_drift to pass DRIFT_CONF_THRESHOLD into the subprocess env
_orig_run_drift = base.run_drift


def run_drift_with_threshold(N: int) -> Path:
    in_csv = AUD / f"_eval_dataset_v3_batch{N}_final.csv"
    out_csv = AUD / f"_eval_dataset_v3_batch{N}_drift_screened.csv"
    out_json = AUD / f"_eval_dataset_v3_batch{N}_drift_dropped.json"
    code = (
        "import os, sys\n"
        "os.environ['DRIFT_CONF_THRESHOLD'] = '0.80'\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        f"sys.path.insert(0, {str(AUD)!r})\n"
        "import _run_drift_screen_batch2 as ds\n"
        "ds.DRIFT_CONF_THRESHOLD = float(os.environ['DRIFT_CONF_THRESHOLD'])\n"
        "from pathlib import Path\n"
        f"ds.IN_CSV  = Path({str(in_csv)!r})\n"
        f"ds.OUT_CSV = Path({str(out_csv)!r})\n"
        f"ds.OUT_JSON = Path({str(out_json)!r})\n"
        "raise SystemExit(ds.main())\n"
    )
    rc, _ = base.run([base.PY, "-c", code], label=f"drift screen batch{N} (conf>=0.80)")
    if rc != 0:
        raise RuntimeError(f"drift failed for batch{N}")
    return out_csv


base.run_drift = run_drift_with_threshold

# preserve historical batches 5..10 in the progress markdown
HISTORICAL = [
    {"batch": 5,  "gen": 20, "mech": 17, "polarity": 17, "judge": 15, "drift": 10, "new": 10, "yield_pct": 0.50, "cum_n": 41},
    {"batch": 6,  "gen": 20, "mech":  7, "polarity":  7, "judge":  6, "drift":  6, "new":  6, "yield_pct": 0.30, "cum_n": 47},
    {"batch": 7,  "gen": 20, "mech": 17, "polarity": 17, "judge": 17, "drift": 12, "new": 12, "yield_pct": 0.60, "cum_n": 59},
    {"batch": 8,  "gen": 20, "mech":  9, "polarity":  9, "judge":  9, "drift":  6, "new":  6, "yield_pct": 0.30, "cum_n": 65},
    {"batch": 9,  "gen": 20, "mech": 16, "polarity": 16, "judge": 15, "drift":  7, "new":  7, "yield_pct": 0.35, "cum_n": 72},
    {"batch": 10, "gen": 20, "mech": 14, "polarity": 14, "judge": 10, "drift":  4, "new":  4, "yield_pct": 0.20, "cum_n": 76},
]

_orig_update = base.update_progress


def update_progress_merge(stats, final=False):
    merged = HISTORICAL + list(stats)
    _orig_update(merged, final=final)


base.update_progress = update_progress_merge

if __name__ == "__main__":
    raise SystemExit(base.main())
