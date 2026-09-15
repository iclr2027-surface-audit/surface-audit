#!/usr/bin/env python3
"""
Random size-matched subset fidelity baseline, answering the Table-2 question
("what are rho/tau for the full dataset -- it's not 1, correct?").

The full dataset compared to itself is rho = tau = 1 by definition; the
informative calibration is a *random* 476-pair subset. This script computes
Spearman rho / Kendall tau between per-model accuracy on random 476-pair
subsets and on the full 790 pairs, across the same 14-model seed-42 panel used
everywhere else, and compares against Audit-Prune's canonical subset.

Sanity check: reproduces the paper's Audit-Prune-476 fidelity exactly
(rho = 0.9151, tau = 0.8268) before computing the baseline.

Run: python scripts/run_random_subset_fidelity.py
Writes: results/random_subset_fidelity.json
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

N_SEEDS = 50
SUBSET_SIZE = 476


def main() -> None:
    files = glob.glob("data/predictions/model_predictions_*seed42*")
    files.append("data/predictions/model_predictions_pythia.csv")  # pythia-2.8b-deduped
    df = pd.concat([pd.read_csv(f) for f in files])
    piv = df.pivot_table(index="pair_id", columns="model_name", values="correct")
    assert piv.shape == (790, 14), f"expected 790x14, got {piv.shape}"
    full_acc = piv.mean(axis=0)

    ap = json.load(open(
        "data/subsets/TruthfulQA-Audited/surface6/pair_ids/pair_ids_theta053.json"
    ))["pair_ids"]
    sub = piv.loc[ap].mean(axis=0)
    ours = {"rho": round(float(spearmanr(full_acc, sub).statistic), 4),
            "tau": round(float(kendalltau(full_acc, sub).statistic), 4)}
    if abs(ours["rho"] - 0.9151) > 5e-4 or abs(ours["tau"] - 0.8268) > 5e-4:
        sys.exit(f"sanity check failed: {ours} != paper 0.9151/0.8268")
    print(f"sanity ok: Audit-Prune-476 rho={ours['rho']} tau={ours['tau']}")

    rhos, taus = [], []
    for s in range(1, N_SEEDS + 1):
        ids = random.Random(s).sample(list(piv.index), SUBSET_SIZE)
        a = piv.loc[ids].mean(axis=0)
        rhos.append(float(spearmanr(full_acc, a).statistic))
        taus.append(float(kendalltau(full_acc, a).statistic))
    rhos, taus = np.array(rhos), np.array(taus)

    out = {
        "protocol": ("Per-model accuracy on subset vs full 790, Spearman rho / Kendall tau, "
                     "14-model seed-42 panel; random subsets of size 476, seeds 1..50."),
        "audit_prune_476": ours,
        "random_476": {
            "n_seeds": N_SEEDS,
            "rho_mean": round(float(rhos.mean()), 4), "rho_sd": round(float(rhos.std()), 4),
            "rho_min": round(float(rhos.min()), 4), "rho_max": round(float(rhos.max()), 4),
            "tau_mean": round(float(taus.mean()), 4), "tau_sd": round(float(taus.std()), 4),
            "tau_min": round(float(taus.min()), 4), "tau_max": round(float(taus.max()), 4),
            "frac_rho_below_ours": round(float((rhos < ours["rho"]).mean()), 2),
        },
    }
    json.dump(out, open("results/random_subset_fidelity.json", "w"), indent=2)
    print(json.dumps(out["random_476"], indent=1))
    print("wrote results/random_subset_fidelity.json")


if __name__ == "__main__":
    main()
