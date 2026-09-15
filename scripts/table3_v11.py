#!/usr/bin/env python3
"""v1.1 rescore step 2a — Table 3 (adversarial + natural paired accuracies,
McNemar cleaned-476 vs full-790) on the 131-row v1.1 cohorts, per the
f02_clustered_table4 protocol: published read-only heads from artifacts_790train/
artifacts_476train, rand-476 size-control heads refit from the cached frozen
training embeddings with the published manifests audits/random476_seeds_v3 and the
published recipe (StandardScaler + LR liblinear, seed 42; C=1.0/mi2000 for
embedding families, mi1000 for surface_lr).

Test matrices: results/v1_1_rescore/emb/{adv131,natural131}_{prefix}_{T,F}.npy
(edited rows re-encoded, dropped rows removed; bge_multi_gemma2 = PROVISIONAL,
stale embeddings on edited rows). surface_lr features recomputed from v1.1 text.

Output: results/v1_1_rescore/table3_v1_1.json + per-item correctness CSV.
"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("f02", ROOT / "scripts/f02_clustered_table4.py")
f02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f02)

EMB = ROOT / "artifacts" / "embeddings"
NEWEMB = ROOT / "results" / "v1_1_rescore" / "emb"
SEED = 42

sf1 = pd.read_csv(ROOT / "hf_release" / "SurfaceFlipped-131.csv")
na1 = pd.read_csv(ROOT / "hf_release" / "Natural-131.csv")
assert len(sf1) == len(na1) == 131
adv_T = sf1.true_text_surface_inverted.astype(str).tolist(); adv_F = sf1.false_text_surface_inverted.astype(str).tolist()
nat_T = na1.true_answer_natural.astype(str).tolist(); nat_F = na1.false_answer_natural.astype(str).tolist()
tqa = pd.read_csv(ROOT / "TruthfulQA.csv").reset_index(drop=True)

RAND = {s: sorted(int(x) for x in json.loads((ROOT / "audits" / "random476_seeds_v3" / f"random476_seed{s}.json").read_text())["pair_ids"]) for s in range(10)}

published = json.load(open(ROOT / "results" / "t15v3_adv_nat_theta053.json"))
pub_rows = dict(published["rows"])
gem_fill = json.load(open(ROOT / "results" / "t15v3_bge_gemma2_fill.json"))

out = {"meta": {"protocol": "f02_clustered_table4 (read-only published heads; rand-476x10 refits from audits/random476_seeds_v3; McNemar exact binomial)",
                "cohorts": "hf_release v1.1 (n=131 each)",
                "n_items": 131, "lr_seed": SEED,
                "test_embeddings": "results/v1_1_rescore/emb/ (edited rows re-encoded, 8 dropped rows removed)",
                "provisional": ["BGE-Multi-Gemma2 (stale v1.0 embeddings on edited rows until cluster re-encode)"]},
       "rows": {}}
per_item = []
T0 = time.time()
for label, full_stem, clean_stem, emb_prefix, test_prefix in f02.FAMILIES:
    t0 = time.time()
    full_pipe = f02.load_pipe(ROOT / "artifacts_790train" / f"{full_stem}.pkl")
    clean_pipe = f02.load_pipe(ROOT / "artifacts_476train" / f"{clean_stem}.pkl")
    if emb_prefix is None:
        Xtr_all = ytr_all = pid_all = None
        tests = {"adv": f02.surface10_pair_matrices(adv_T, adv_F),
                 "nat": f02.surface10_pair_matrices(nat_T, nat_F)}
    else:
        Xtr_all = np.load(EMB / f"{emb_prefix}_X.npy")
        ytr_all = np.load(EMB / f"{emb_prefix}_y.npy")
        pid_all = np.load(EMB / f"{emb_prefix}_pair_id.npy")
        tests = {"adv": (np.load(NEWEMB / f"adv131_{test_prefix}_T.npy"), np.load(NEWEMB / f"adv131_{test_prefix}_F.npy")),
                 "nat": (np.load(NEWEMB / f"natural131_{test_prefix}_T.npy"), np.load(NEWEMB / f"natural131_{test_prefix}_F.npy"))}
    cor = {}
    for tname, (XT, XF) in tests.items():
        assert XT.shape[0] == 131
        cor[(tname, "full")] = (f02.proba_truthful(full_pipe, XT) > f02.proba_truthful(full_pipe, XF)).astype(int)
        cor[(tname, "clean")] = (f02.proba_truthful(clean_pipe, XT) > f02.proba_truthful(clean_pipe, XF)).astype(int)
    rand_cor = {"adv": [], "nat": []}
    for s in range(10):
        keep = RAND[s]
        if emb_prefix is None:
            Xr, yr = f02.build_surface10_matrix(tqa.loc[keep].reset_index(drop=True))
            rp = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", max_iter=1000, random_state=SEED))
        else:
            m = np.isin(pid_all, keep)
            Xr, yr = Xtr_all[m], ytr_all[m]
            rp = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=SEED))
        rp.fit(Xr, yr)
        for tname, (XT, XF) in tests.items():
            rand_cor[tname].append((f02.proba_truthful(rp, XT) > f02.proba_truthful(rp, XF)).astype(int))
    row = {}
    for tname in ("adv", "nat"):
        rc = np.vstack(rand_cor[tname])
        # published t15v3 convention: b = cleaned-wrong & full-right, c = cleaned-right & full-wrong
        b, c, p = f02.mcnemar(cor[(tname, "full")], cor[(tname, "clean")])
        row[tname] = {"acc_full_790": float(cor[(tname, "full")].mean()),
                      "acc_cleaned_476": float(cor[(tname, "clean")].mean()),
                      "acc_rand476_mean": float(rc.mean()),
                      "acc_rand476_sd": float(rc.mean(1).std(ddof=1)),
                      "acc_rand476_seeds": [float(x) for x in rc.mean(1)],
                      "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p}
        ids = sf1.id if tname == "adv" else na1.id
        for i in range(131):
            per_item.append(dict(family=label, cohort=tname, item_id=int(ids.iloc[i]),
                                 correct_full=int(cor[(tname, "full")][i]),
                                 correct_cleaned=int(cor[(tname, "clean")][i]),
                                 correct_rand476_mean=float(rc[:, i].mean())))
    out["rows"][label] = row
    print(f"[{label}] adv full {row['adv']['acc_full_790']:.4f} cln {row['adv']['acc_cleaned_476']:.4f} p {row['adv']['mcnemar_p']:.4g} | "
          f"nat full {row['nat']['acc_full_790']:.4f} cln {row['nat']['acc_cleaned_476']:.4f} p {row['nat']['mcnemar_p']:.4g} | {time.time()-t0:.0f}s", flush=True)
    del Xtr_all, ytr_all, pid_all

# published comparison block (t15v3 + gemma2 fill live values)
comp = {}
for label in out["rows"]:
    pub = pub_rows.get(label) or (gem_fill if label == "BGE-Multi-Gemma2" else {})
    comp[label] = {}
    for t in ("adv", "nat"):
        pv = pub.get(t, {}) if pub else {}
        nv = out["rows"][label][t]
        comp[label][t] = {k: {"published": pv.get(k), "v1_1": nv.get(k),
                              "delta": (None if pv.get(k) is None or nv.get(k) is None else float(nv[k]) - float(pv[k]))}
                          for k in ("acc_full_790", "acc_cleaned_476", "acc_rand476_mean", "acc_rand476_sd", "mcnemar_b", "mcnemar_c", "mcnemar_p")}
out["published_comparison"] = comp
out["meta"]["elapsed_s"] = round(time.time() - T0, 1)
json.dump(out, open(ROOT / "results" / "v1_1_rescore" / "table3_v1_1.json", "w"), indent=1)
pd.DataFrame(per_item).to_csv(ROOT / "results" / "v1_1_rescore" / "table3_v1_1_per_item.csv", index=False)
print("wrote table3_v1_1.json |", out["meta"]["elapsed_s"], "s", flush=True)
