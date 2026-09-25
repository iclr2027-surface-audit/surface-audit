#!/usr/bin/env python3
"""D6 — rank-fidelity uncertainty. Item bootstrap with the 14-model panel held FIXED (models are not
exchangeable: sibling checkpoints), B=10,000; leave-one-family-out and leave-one-model-out; tie policy stated.
Matrix: the same 790x14 per-pair correctness matrix used by the rank-precision audit."""
import json, glob, time, sys, platform
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results"
B = 10_000; SEED = 42; T0 = time.time()
files = sorted(glob.glob(str(ROOT / "data/predictions/model_predictions_*seed42*"))) + [str(ROOT / "data/predictions/model_predictions_pythia.csv")]
df = pd.concat([pd.read_csv(f) for f in files]); M = df.pivot_table(index="pair_id", columns="model_name", values="correct", aggfunc="first")
assert M.shape == (790, 14) and not M.isna().any().any(), M.shape
ids = json.load(open(ROOT / "data/subsets/TruthfulQA-Audited/surface6/pair_ids/pair_ids_theta053.json")); ids = ids if isinstance(ids, list) else ids.get("pair_ids", ids.get("ids"))
in476 = M.index.isin(ids); assert in476.sum() == 476
fam = json.load(open(ROOT / "audits/iclr_new_checks/e01_rank_precision.json"))["meta"]["families"]
A = M.to_numpy(float); models = list(M.columns)
def fid(mask_rows, cols):
    af = A[:, cols].mean(0); a4 = A[mask_rows][:, cols].mean(0)
    return float(spearmanr(af, a4).correlation), float(kendalltau(af, a4).correlation)   # average ranks; tau-b
allc = list(range(14)); rho0, tau0 = fid(in476, allc)
print(f"point: rho={rho0:.4f} tau_b={tau0:.4f} (paper 0.9151 / 0.8268)")
rng = np.random.default_rng(SEED); rb, tb = [], []
for _ in range(B):
    idx = rng.integers(0, 790, 790); Ab = A[idx]; m4 = in476[idx]
    if m4.sum() < 50: continue
    af = Ab.mean(0); a4 = Ab[m4].mean(0); rb.append(spearmanr(af, a4).correlation); tb.append(kendalltau(af, a4).correlation)
rb, tb = np.array(rb), np.array(tb)
ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
rlo, rhi = ci(rb); tlo, thi = ci(tb)
print(f"item bootstrap B={B}: rho 95% CI [{rlo:.4f}, {rhi:.4f}] (mean {rb.mean():.4f}) | tau_b CI [{tlo:.4f}, {thi:.4f}]")
lofo = []
for f in sorted(set(fam.values())):
    cols = [i for i, m in enumerate(models) if fam[m] != f]; r, t = fid(in476, cols); lofo.append(dict(dropped_family=f, n_models=len(cols), rho=r, tau_b=t))
    print(f"  LOFO drop {f:10} n={len(cols):2} rho={r:.4f} tau={t:.4f}")
lomo = [dict(dropped_model=m, rho=fid(in476, [j for j in allc if j != i])[0]) for i, m in enumerate(models)]
lofo_min = min(x["rho"] for x in lofo); lomo_min = min(x["rho"] for x in lomo)
print(f"LOFO min rho={lofo_min:.4f} ({min(lofo, key=lambda x: x['rho'])['dropped_family']}) | leave-one-MODEL-out min rho={lomo_min:.4f} ({min(lomo, key=lambda x: x['rho'])['dropped_model']})")
grade = ("FANTASTIC" if rlo >= 0.90 and lofo_min >= 0.90 else "VERY GOOD" if rlo >= 0.85 and lofo_min >= 0.85 else "GOOD" if rlo >= 0.80 and lofo_min >= 0.80
         else "FAIR" if rlo >= 0.70 and lofo_min >= 0.70 else "NEUTRAL" if rlo >= 0.60 and lofo_min >= 0.60 else "BAD")
print("GRADE:", grade)
json.dump(dict(meta=dict(B=B, seed=SEED, unit="pairs (items) resampled with replacement; 14-model panel fixed", tie_policy="Spearman on average ranks (scipy default); Kendall tau-b",
                         matrix_shape=list(M.shape), prediction_files=[Path(f).name for f in files], python=sys.version.split()[0], platform=platform.platform(), seconds=round(time.time()-T0,1)),
               point=dict(rho=rho0, tau_b=tau0, published_rho=0.9151, published_tau=0.8268),
               item_bootstrap=dict(rho_ci95=[rlo, rhi], rho_mean=float(rb.mean()), rho_sd=float(rb.std(ddof=1)), tau_ci95=[tlo, thi], tau_mean=float(tb.mean()), n_effective=int(len(rb))),
               lofo=lofo, lomo=lomo, lofo_min_rho=lofo_min, lomo_min_rho=lomo_min, grade=grade),
          open(OUT / "r6_d6_rank_fidelity.json", "w"), indent=1)
np.save(OUT / "r6_boot_rho.npy", rb); print(f"wrote r6_d6_rank_fidelity.json ({time.time()-T0:.0f}s)")
