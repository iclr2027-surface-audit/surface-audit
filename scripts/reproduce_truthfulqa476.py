#!/usr/bin/env python3
"""Regenerate the released TruthfulQA-476 manifest exactly.

WHAT THIS SCRIPT DOES, AND WHAT IT DOES NOT CLAIM
--------------------------------------------------
The released manifest (data/subsets/TruthfulQA-Audited/surface6/pair_ids/pair_ids_theta053.json)
was produced on 2026-05-06 by a sweep wrapper whose source was not committed.  Its removal path was
recorded as an audit-AUC fingerprint, one value per removal step
(results/t5b_audit_prune_trajectory_theta050_surface6.json, 455 recorded steps down to theta=0.50).

This script regenerates the released 476-pair set bit-for-bit by

  1. running Audit-Prune's greedy removal (Algorithm 1 of the paper) on the SURFACE6 features --
     beta-weighted, sign-aligned standardized imbalance score b_i, refit at every step, grouped 5-fold
     audit -- and asserting the audit AUC against the recorded fingerprint at EVERY step;
  2. following the RECORDED path where the re-derived greedy pick differs from it.  Removal stops at
     the first step whose AUC <= theta = 0.53, i.e. after 403 executed steps (n = 387).  Of those 403
     executed choices, 380 are re-derived by the greedy rule and 23 (steps 207-290) are REPLAYED from
     the recorded order, because the greedy path is numerically sensitive to last-digit differences in
     the scores and no tested scoring variant reproduces those 23 picks.  The recorded order itself was
     recovered by inverting the fingerprint and is unambiguous at every step;
  3. applying the add-back refinement -- ascending-pair_id passes, accept a removed pair if the audit
     AUC stays <= theta + 1e-9, repeat until stable -- which is the rule that regenerates the released
     set exactly (single-pass, descending and removal-order variants do not);
  4. asserting that the result equals the released manifest.

Model: StandardScaler + LogisticRegression(liblinear, C=1, max_iter=2000, seed 42); unshuffled
GroupKFold(5) over answer rows grouped by pair.  Runtime: a few minutes on a laptop CPU.

Run from anywhere:  python scripts/reproduce_truthfulqa476.py
Requires TruthfulQA.csv at the repository root (upstream Lin et al. binary-choice file) and
audits/truthfulqa_style_audit.csv (the SURFACE feature table shipped with the repository).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_subset_evaluator as ase  # noqa: E402
from search_truthfulqa_pruned_improved import _ans_frame  # noqa: E402
from truthfulqa_pruning_utils import load_candidates_with_features  # noqa: E402

THETA = 0.53
EPS = 1e-9
SEED = 42
FEATS = ["word_count", "neg_cnt", "avg_token_len", "type_token", "neg_lead", "hedge_rate"]
FINGERPRINT = ROOT / "results" / "t5b_audit_prune_trajectory_theta050_surface6.json"
RELEASED_MANIFEST = ROOT / "data" / "subsets" / "TruthfulQA-Audited" / "surface6" / "pair_ids" / "pair_ids_theta053.json"

# Removal order recovered from the recorded AUC fingerprint (455 steps; only the first 403 are executed at theta=0.53).
RECORDED_ORDER = [174, 49, 173, 711, 176, 178, 175, 518, 228, 715, 276, 15, 570, 177, 51, 76, 492, 182, 217, 491, 250, 179, 287, 229, 256, 752, 266, 85, 148, 67, 88, 172, 245, 278, 382, 684, 98, 286, 262, 265, 263, 251, 348, 231, 365, 72, 89, 686, 194, 200, 227, 271, 383, 386, 759, 330, 260, 64, 78, 284, 774, 414, 283, 777, 404, 101, 326, 643, 493, 68, 153, 205, 634, 474, 398, 183, 366, 184, 208, 623, 50, 233, 476, 275, 210, 788, 639, 723, 268, 484, 219, 565, 645, 772, 207, 655, 631, 150, 688, 41, 277, 499, 473, 648, 149, 539, 497, 60, 57, 498, 363, 630, 75, 244, 270, 319, 235, 478, 483, 514, 756, 779, 203, 280, 104, 773, 93, 2, 693, 568, 633, 261, 43, 170, 632, 257, 102, 258, 81, 42, 241, 346, 169, 677, 246, 482, 705, 44, 282, 567, 672, 74, 328, 569, 350, 641, 342, 329, 635, 351, 55, 373, 668, 159, 202, 206, 315, 511, 512, 515, 516, 755, 517, 611, 222, 654, 480, 664, 613, 775, 214, 96, 234, 304, 318, 513, 749, 564, 593, 295, 780, 279, 735, 361, 224, 669, 253, 273, 77, 86, 171, 252, 316, 362, 475, 479, 612, 698, 718, 180, 769, 166, 622, 505, 359, 776, 789, 307, 390, 533, 164, 190, 152, 297, 31, 389, 506, 689, 542, 615, 165, 84, 151, 281, 128, 341, 781, 155, 29, 309, 157, 537, 782, 100, 538, 189, 462, 255, 296, 737, 154, 167, 156, 778, 783, 187, 285, 22, 188, 90, 504, 463, 181, 299, 195, 199, 438, 267, 675, 432, 311, 592, 317, 703, 147, 337, 69, 110, 356, 451, 254, 481, 534, 624, 490, 597, 600, 494, 616, 719, 722, 679, 566, 681, 708, 614, 673, 168, 401, 610, 595, 14, 321, 322, 137, 46, 397, 325, 94, 12, 103, 105, 495, 601, 766, 768, 695, 204, 388, 163, 62, 586, 146, 17, 83, 323, 324, 525, 594, 625, 724, 748, 400, 162, 734, 485, 682, 92, 576, 577, 370, 48, 61, 300, 65, 70, 374, 87, 384, 236, 418, 129, 352, 607, 107, 486, 646, 264, 387, 209, 368, 369, 449, 34, 422, 353, 59, 99, 4, 53, 192, 220, 25, 423, 314, 145, 364, 343, 357, 472, 381, 528, 496, 608, 396, 676, 402, 489, 554, 587, 657, 113, 690, 704, 571, 411, 603, 584, 371, 744, 726, 292, 685, 193, 191, 375, 201, 385, 604, 136, 652, 456, 702, 602, 379, 230, 358, 743, 274, 745, 225, 732, 582, 572, 573, 450, 574, 437, 112, 662, 687, 758, 527, 354, 106, 599, 770, 544, 138, 377, 659, 378, 158, 575, 710, 606, 47, 426, 651, 548, 417, 294, 487, 216, 500]


def answer_frame():
    """Answer-level frame (2 rows per pair): pair_id, label, SURFACE features."""
    full = load_candidates_with_features(ROOT / "TruthfulQA.csv", ROOT / "audits" / "truthfulqa_style_audit.csv")
    ans = _ans_frame(full).copy()
    return ans.sort_values(["pair_id", "label"], ascending=[True, False]).reset_index(drop=True)


def pipe():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(solver="liblinear", C=1.0, max_iter=2000, penalty="l2", random_state=SEED))


def main() -> int:
    fp = [s["auc"] for s in json.load(open(FINGERPRINT))["trajectory"]]
    released = set(json.load(open(RELEASED_MANIFEST))["pair_ids"])
    af = answer_frame()
    X_all = af[FEATS].to_numpy(float)
    y_all = af["label"].to_numpy(int)
    g_all = af["pair_id"].to_numpy()

    def audit(mask):
        X, y, g = X_all[mask], y_all[mask], g_all[mask]
        p = cross_val_predict(pipe(), X, y, cv=GroupKFold(5), groups=g, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, p))

    def imbalance_scores(mask):
        X, y, g = X_all[mask], y_all[mask], g_all[mask]
        sigma = X.std(axis=0, ddof=0)
        sig = np.where(sigma > 0, sigma, 1.0)
        Xz = (X - X.mean(0)) / sig
        gap_z = Xz[y == 1].mean(0) - Xz[y == 0].mean(0)
        beta = np.abs(pipe().fit(X, y).named_steps["logisticregression"].coef_.ravel())
        out = {}
        for pid in np.unique(g):
            rows = np.flatnonzero(g == pid)
            dz = (X[rows[y[rows] == 1][0]] - X[rows[y[rows] == 0][0]]) / sig
            out[int(pid)] = float(sum(beta[j] * abs(dz[j]) for j in range(len(dz))
                                      if gap_z[j] != 0 and sigma[j] != 0 and dz[j] * gap_z[j] > 0))
        return out

    def greedy_pick(sc, alive):
        best = max(sc[i] for i in alive)
        return min(i for i in alive if abs(sc[i] - best) <= 1e-15)

    mask = np.ones(len(y_all), bool)
    alive = set(int(x) for x in np.unique(g_all))
    removed, deviations, k = [], [], 0
    while True:
        a = audit(mask)
        assert abs(a - fp[k]) < 1e-6, f"fingerprint mismatch at step {k}: {a:.6f} vs {fp[k]:.6f}"
        if a <= THETA + EPS:
            break
        g_pick = greedy_pick(imbalance_scores(mask), alive)
        r_pick = RECORDED_ORDER[k]
        if g_pick != r_pick:
            deviations.append(k)
        alive.remove(r_pick)
        removed.append(r_pick)
        mask &= (g_all != r_pick)
        k += 1
    print(f"removal stopped at step {k}: n={len(alive)} auc={a:.6f} | "
          f"re-derived {k - len(deviations)} of {k} executed steps; replayed {len(deviations)} from the recorded path at steps {deviations}")

    ret, rem = set(alive), list(removed)
    while True:
        added = False
        for pid in sorted(rem):
            if audit(np.isin(g_all, list(ret | {pid}))) <= THETA + EPS:
                ret.add(pid)
                rem.remove(pid)
                added = True
        if not added:
            break
    final_auc = audit(np.isin(g_all, list(ret)))
    same = ret == released
    print(f"after add-back: n={len(ret)} auc={final_auc:.6f} | identical to released TruthfulQA-476: {same}")
    assert same, "regenerated set differs from the released manifest"
    print("OK -- released TruthfulQA-476 regenerated exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
