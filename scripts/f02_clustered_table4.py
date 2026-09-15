#!/usr/bin/env python3
"""SUITE F (2) -- clustered / batch-robust inference for Table 4.

Recomputes the Adversarial + Natural n=135 evaluation per item, then re-runs the
full-vs-cleaned and cleaned-vs-random effects under four inference schemes:

  (i)   ordinary McNemar (exact binomial on discordant pairs) -- reproduces the
        published p-values in results/t15_adv_nat_surface6.json and flags any
        mismatch;
  (ii)  CATEGORY-cluster bootstrap (12 clusters, resample clusters w/ repl.);
  (iii) generation-BATCH inference -- (a) batch-cluster bootstrap and
        (b) wild cluster bootstrap with Rademacher weights imposing H0
        (the small-G-robust option; Adversarial only, see f01: the Natural
        cohort has no batch structure);
  (iv)  sensitivity EXCLUDING the 20 pilot-template items (n=115).

All heads are the published, already-trained ones (READ ONLY):
  artifacts_790train/*_lr_full_790.pkl   (full-790 heads)
  artifacts_476train/*_lr_cleaned_476.pkl (cleaned TruthfulQA-476, theta=0.53)
Random-476 size-control heads are refit here from the cached frozen embeddings
with the published protocol (StandardScaler + LogisticRegression, liblinear,
random_state=42) and the published manifests
audits/random476_seeds/random476_seed{0..9}.json when present, else regenerated
with numpy.default_rng(seed).choice(790, 476, replace=False) exactly as the
published runner does.

Seeds: LR random_state=42; random-476 draws use seeds 0..9;
       cluster bootstrap rng seed 20260813; wild bootstrap rng seed 20260814.

Outputs (new files only):
  audits/iclr_new_checks/clustered_table4.csv
  audits/iclr_new_checks/f02_per_item_correct.csv
  audits/iclr_new_checks/f02_clustered_table4.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "audits" / "iclr_new_checks"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.surface_features_text import (  # noqa: E402
    extract_surface10, rel_len_gap, FEATURE_COLS as SURFACE10_COLS,
)
from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.pipeline import make_pipeline            # noqa: E402
from sklearn.preprocessing import StandardScaler      # noqa: E402

EMB = REPO / "artifacts" / "embeddings"
ART_FULL = REPO / "artifacts_790train"
ART_CLEAN = REPO / "artifacts_476train"
SEED = 42
N_TOTAL, N_RAND, RAND_SEEDS = 790, 476, list(range(10))
B_CLUSTER = 20000
B_WILD = 20000
RNG_CLUSTER = 20260813
RNG_WILD = 20260814

# label, full pkl stem, cleaned pkl stem, training-embedding prefix, cohort cache prefix
FAMILIES = [
    ("Llama-3.2-3B",     "llama32_3b_lr_full_790",       "llama32_3b_lr_cleaned_476",       "llama32_3b_full",       "llama32_3b"),
    ("SmolLM2-1.7B",     "smollm2_lr_full_790",          "smollm2_lr_cleaned_476",          "smollm2_full",          "smollm2"),
    ("BGE-large",        "embedding_lr_full_790",        "embedding_lr_cleaned_476",        "full",                  "bge_large"),
    ("Qwen2.5-0.5B",     "qwen_lr_full_790",             "qwen_lr_cleaned_476",             "qwen_full",             "qwen"),
    ("BGE-Multi-Gemma2", "bge_multi_gemma2_lr_full_790", "bge_multi_gemma2_lr_cleaned_476", "bge_multi_gemma2_full", "bge_multi_gemma2"),
    ("Phi-3.5-mini",     "phi35_lr_full_790",            "phi35_lr_cleaned_476",            "phi35_full",            "phi35"),
    ("ModernBERT-base",  "modernbert_lr_full_790",       "modernbert_lr_cleaned_476",       "modernbert_full",       "modernbert"),
    ("Qwen2.5-3B",       "qwen3b_lr_full_790",           "qwen3b_lr_cleaned_476",           "qwen3b_full",           "qwen3b"),
    ("Qwen2.5-1.5B",     "qwen15b_lr_full_790",          "qwen15b_lr_cleaned_476",          "qwen15b_full",          "qwen15b"),
    ("surface_lr",       "surface_lr_lr_full_790",       "surface_lr_lr_cleaned_476",       None,                    None),
]


# --------------------------------------------------------------------------- #
def proba_truthful(pipe, X):
    classes = list(getattr(pipe.steps[-1][1], "classes_", [0, 1]))
    return np.asarray(pipe.predict_proba(X)[:, classes.index(1)], dtype=float)


def load_pipe(p: Path):
    o = joblib.load(p)
    return o["pipeline"] if isinstance(o, dict) else o


def build_surface10_matrix(pairs_df):
    rows, y = [], []
    for _, r in pairs_df.iterrows():
        t, f = str(r["Best Answer"]), str(r["Best Incorrect Answer"])
        gap = float(rel_len_gap(t, f))
        ft = extract_surface10(t); ft["len_gap"] = gap
        ff = extract_surface10(f); ff["len_gap"] = gap
        rows.append([float(ft[c]) for c in SURFACE10_COLS]); y.append(1)
        rows.append([float(ff[c]) for c in SURFACE10_COLS]); y.append(0)
    return np.asarray(rows, float), np.asarray(y, int)


def surface10_pair_matrices(true_texts, false_texts):
    XA, XB = [], []
    for a, b in zip(true_texts, false_texts):
        gap = float(rel_len_gap(a, b))
        fa = extract_surface10(a); fa["len_gap"] = gap
        fb = extract_surface10(b); fb["len_gap"] = gap
        XA.append([float(fa[c]) for c in SURFACE10_COLS])
        XB.append([float(fb[c]) for c in SURFACE10_COLS])
    return np.asarray(XA, float), np.asarray(XB, float)


def mcnemar(cor_a, cor_b):
    """b = a-right/b-wrong, c = a-wrong/b-right; exact binomial (repo recipe)."""
    a = np.asarray(cor_a, int); bb = np.asarray(cor_b, int)
    b = int(np.sum((a == 1) & (bb == 0)))
    c = int(np.sum((a == 0) & (bb == 1)))
    n = b + c
    p = 1.0 if n == 0 else float(
        binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    return b, c, p


# --------------------------------------------------------------------------- #
# cluster inference helpers
# --------------------------------------------------------------------------- #
def cluster_bootstrap(d, clusters, B, seed):
    """Nonparametric cluster bootstrap of mean(d). Returns (lo, hi, p, se, n_eff)."""
    d = np.asarray(d, float)
    uc = np.unique(clusters)
    idx = [np.where(clusters == c)[0] for c in uc]
    G = len(uc)
    rng = np.random.default_rng(seed)
    stats = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, G, size=G)
        sel = np.concatenate([idx[j] for j in pick])
        stats[b] = d[sel].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    theta = d.mean()
    # percentile-bootstrap two-sided p: 2*min(frac<=0, frac>=0)
    p = 2.0 * min((stats <= 0).mean(), (stats >= 0).mean())
    return float(lo), float(hi), float(min(p, 1.0)), float(stats.std(ddof=1)), G


def wild_cluster_bootstrap(d, clusters, B, seed):
    """Wild cluster bootstrap-t imposing H0: E[d]=0, Rademacher weights.

    Test stat: t = mean(d) / cluster-robust SE.  Under H0 the restricted
    residual is d_i - 0 = d_i re-centred to the null, i.e. e_i = d_i - mean(d)
    is NOT used; the standard 'impose the null' construction for a mean is
    d*_i = (d_i - mean(d)) * w_g  (so E[d*]=0 by construction), then
    t* = mean(d*) / SE(d*).  p = fraction |t*| >= |t|.
    """
    d = np.asarray(d, float)
    n = len(d)
    uc = np.unique(clusters)
    gidx = {c: np.where(clusters == c)[0] for c in uc}
    G = len(uc)

    def crse(x):
        m = x.mean()
        s = np.array([ (x[gidx[c]] - m).sum() for c in uc ], float)
        # cluster-robust variance of the mean, with small-G correction G/(G-1)
        var = (G / max(G - 1, 1)) * (s ** 2).sum() / (n ** 2)
        return float(np.sqrt(var)) if var > 0 else float("nan")

    t_obs = d.mean() / crse(d) if np.isfinite(crse(d)) and crse(d) > 0 else np.nan
    e = d - d.mean()
    rng = np.random.default_rng(seed)
    t_star = np.empty(B)
    for b in range(B):
        w = rng.choice([-1.0, 1.0], size=G)
        ds = e.copy()
        for j, c in enumerate(uc):
            ds[gidx[c]] = e[gidx[c]] * w[j]
        se = crse(ds)
        t_star[b] = ds.mean() / se if se and np.isfinite(se) and se > 0 else np.nan
    ok = np.isfinite(t_star)
    p = float((np.abs(t_star[ok]) >= abs(t_obs)).mean()) if np.isfinite(t_obs) and ok.sum() else float("nan")
    return float(t_obs), float(p), int(G), float(crse(d))


def randomization_batch(d, clusters, B, seed):
    """Batch-level sign-flip randomization test on cluster means (exact-ish).

    Statistic: mean over items.  Null: the sign of each batch's mean effect is
    exchangeable (batch-level permutation).  p = fraction of sign assignments
    with |stat*| >= |stat_obs|.  With G<=11 batches we enumerate all 2^G sign
    vectors when 2^G <= B, else sample B of them.
    """
    d = np.asarray(d, float)
    uc = np.unique(clusters)
    gidx = [np.where(clusters == c)[0] for c in uc]
    G = len(uc)
    sizes = np.array([len(i) for i in gidx], float)
    means = np.array([d[i].mean() for i in gidx], float)
    n = sizes.sum()
    stat_obs = float((sizes * means).sum() / n)
    if 2 ** G <= B:
        signs = np.array([[1 if (k >> j) & 1 else -1 for j in range(G)]
                          for k in range(2 ** G)], float)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(B, G))
    stats = (signs * (sizes * means)).sum(axis=1) / n
    p = float((np.abs(stats) >= abs(stat_obs) - 1e-15).mean())
    return stat_obs, p, G, int(len(signs))


# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    adv = pd.read_csv(REPO / "audits" / "_eval_dataset_v3_inv_accuracy_full_final.csv")
    nat = pd.read_csv(REPO / "audits" / "_eval_dataset_natural_scratch_final.csv")
    bat = pd.read_csv(OUT / "adv135_batch_category.csv")
    assert len(adv) == len(nat) == len(bat) == 135
    assert list(bat["id"]) == list(adv["id"])

    adv_T = adv["true_text_surface_inverted"].astype(str).tolist()
    adv_F = adv["false_text_surface_inverted"].astype(str).tolist()
    nat_T = nat["true_answer_natural"].astype(str).tolist()
    nat_F = nat["false_answer_natural"].astype(str).tolist()

    adv_cat = adv["category"].astype(str).to_numpy()
    nat_cat = nat["source_category"].astype(str).to_numpy()
    adv_batch = bat["batch"].astype(str).to_numpy()
    is_pilot = bat["is_pilot"].to_numpy().astype(bool)

    tqa = pd.read_csv(REPO / "TruthfulQA.csv").reset_index(drop=True)
    assert len(tqa) == N_TOTAL

    # random-476 manifests: reuse published ones if present, else regenerate
    seeds_dir = REPO / "audits" / "random476_seeds_v3"
    rand_ids, rand_src = {}, {}
    for s in RAND_SEEDS:
        fp = seeds_dir / f"random476_seed{s}.json"
        if fp.exists():
            rand_ids[s] = sorted(int(x) for x in json.loads(fp.read_text())["pair_ids"])
            rand_src[s] = str(fp.relative_to(REPO))
        else:
            rng = np.random.default_rng(seed=s)
            rand_ids[s] = sorted(int(x) for x in
                                 rng.choice(N_TOTAL, size=N_RAND, replace=False))
            rand_src[s] = "regenerated numpy.default_rng(%d).choice(790,476,False)" % s
        assert len(rand_ids[s]) == N_RAND

    published = json.loads((REPO / "results" / "t15_adv_nat_surface6.json").read_text())

    per_item_rows = []
    table_rows = []
    detail = {}

    for label, full_stem, clean_stem, emb_prefix, test_prefix in FAMILIES:
        t_fam = time.time()
        full_pipe = load_pipe(ART_FULL / f"{full_stem}.pkl")
        clean_pipe = load_pipe(ART_CLEAN / f"{clean_stem}.pkl")

        if emb_prefix is None:
            Xtr_all, ytr_all, pid_all = None, None, None
            adv_XT, adv_XF = surface10_pair_matrices(adv_T, adv_F)
            nat_XT, nat_XF = surface10_pair_matrices(nat_T, nat_F)
        else:
            Xtr_all = np.load(EMB / f"{emb_prefix}_X.npy")
            ytr_all = np.load(EMB / f"{emb_prefix}_y.npy")
            pid_all = np.load(EMB / f"{emb_prefix}_pair_id.npy")
            adv_XT = np.load(EMB / f"adv135_{test_prefix}_T.npy")
            adv_XF = np.load(EMB / f"adv135_{test_prefix}_F.npy")
            nat_XT = np.load(EMB / f"natural135_{test_prefix}_T.npy")
            nat_XF = np.load(EMB / f"natural135_{test_prefix}_F.npy")

        cor = {}
        for tname, XT, XF in (("adv", adv_XT, adv_XF), ("nat", nat_XT, nat_XF)):
            cor[(tname, "full")] = (proba_truthful(full_pipe, XT) >
                                    proba_truthful(full_pipe, XF)).astype(int)
            cor[(tname, "clean")] = (proba_truthful(clean_pipe, XT) >
                                     proba_truthful(clean_pipe, XF)).astype(int)

        # ---- random-476 size-control heads -------------------------------- #
        rand_cor = {"adv": [], "nat": []}
        for s in RAND_SEEDS:
            keep = rand_ids[s]
            if emb_prefix is None:
                Xr, yr = build_surface10_matrix(tqa.loc[keep].reset_index(drop=True))
                rp = make_pipeline(StandardScaler(),
                                   LogisticRegression(solver="liblinear",
                                                      max_iter=1000, random_state=SEED))
            else:
                m = np.isin(pid_all, keep)
                Xr, yr = Xtr_all[m], ytr_all[m]
                rp = make_pipeline(StandardScaler(),
                                   LogisticRegression(C=1.0, solver="liblinear",
                                                      max_iter=2000, random_state=SEED))
            rp.fit(Xr, yr)
            for tname, XT, XF in (("adv", adv_XT, adv_XF), ("nat", nat_XT, nat_XF)):
                rand_cor[tname].append(
                    (proba_truthful(rp, XT) > proba_truthful(rp, XF)).astype(int))
        for tname in ("adv", "nat"):
            rand_cor[tname] = np.vstack(rand_cor[tname])          # (10, 135)

        # ---- per-item records --------------------------------------------- #
        for tname in ("adv", "nat"):
            ids = adv["id"] if tname == "adv" else nat["id"]
            cats = adv_cat if tname == "adv" else nat_cat
            for i in range(135):
                per_item_rows.append(dict(
                    family=label, cohort=tname, row=i, item_id=int(ids.iloc[i]),
                    category=cats[i],
                    batch=(adv_batch[i] if tname == "adv" else "n/a"),
                    is_pilot=(int(is_pilot[i]) if tname == "adv" else -1),
                    correct_full=int(cor[(tname, "full")][i]),
                    correct_cleaned=int(cor[(tname, "clean")][i]),
                    correct_rand476_mean=float(rand_cor[tname][:, i].mean()),
                ))

        # ---- inference ----------------------------------------------------- #
        for tname in ("adv", "nat"):
            cf, cc = cor[(tname, "full")], cor[(tname, "clean")]
            rc = rand_cor[tname]
            cats = adv_cat if tname == "adv" else nat_cat

            # contrast A: cleaned - full   (per-item difference in {-1,0,1})
            dA = cc.astype(float) - cf.astype(float)
            # contrast B: cleaned - random476 (averaged over the 10 seeds)
            dB = cc.astype(float) - rc.mean(axis=0)

            for contrast, d, (ca, cb) in (("cleaned_minus_full", dA, (cc, cf)),
                                          ("cleaned_minus_rand476", dB, (None, None))):
                subsets = [("all135", np.ones(135, bool))]
                if tname == "adv":
                    subsets.append(("no_pilot115", ~is_pilot))
                for sub_name, mask in subsets:
                    dm = d[mask]
                    row = dict(family=label, cohort=tname, contrast=contrast,
                               subset=sub_name, n_items=int(mask.sum()),
                               effect=float(dm.mean()))
                    if contrast == "cleaned_minus_full":
                        b, c, p = mcnemar(ca[mask], cb[mask])
                        row.update(mcnemar_b=b, mcnemar_c=c, mcnemar_p=p)
                        row["acc_full"] = float(cb[mask].mean())
                        row["acc_cleaned"] = float(ca[mask].mean())
                    else:
                        # per-seed McNemar cleaned vs each random head
                        ps, bs, cs_ = [], [], []
                        for s_i in range(len(RAND_SEEDS)):
                            b, c, p = mcnemar(cc[mask], rc[s_i][mask])
                            ps.append(p); bs.append(b); cs_.append(c)
                        row.update(mcnemar_b=float(np.mean(bs)),
                                   mcnemar_c=float(np.mean(cs_)),
                                   mcnemar_p=float(np.median(ps)))
                        row["mcnemar_p_min"] = float(np.min(ps))
                        row["mcnemar_p_max"] = float(np.max(ps))
                        row["n_seeds_p_lt_05"] = int(np.sum(np.array(ps) < 0.05))
                        row["acc_cleaned"] = float(cc[mask].mean())
                        row["acc_rand476_mean"] = float(rc[:, mask].mean())
                    # (ii) category-cluster bootstrap
                    lo, hi, pc, se, G = cluster_bootstrap(dm, cats[mask], B_CLUSTER, RNG_CLUSTER)
                    row.update(cat_boot_lo=lo, cat_boot_hi=hi, cat_boot_p=pc,
                               cat_boot_se=se, n_cat_clusters=G)
                    # (iii) batch inference (adversarial only)
                    if tname == "adv":
                        blo, bhi, bp, bse, bG = cluster_bootstrap(
                            dm, adv_batch[mask], B_CLUSTER, RNG_CLUSTER + 1)
                        t_obs, wp, wG, crse = wild_cluster_bootstrap(
                            dm, adv_batch[mask], B_WILD, RNG_WILD)
                        st, rp_, rG, n_signs = randomization_batch(
                            dm, adv_batch[mask], B_WILD, RNG_WILD + 1)
                        row.update(batch_boot_lo=blo, batch_boot_hi=bhi,
                                   batch_boot_p=bp, batch_boot_se=bse,
                                   n_batch_clusters=bG,
                                   wild_t=t_obs, wild_p=wp, wild_crse=crse,
                                   rand_infer_p=rp_, rand_infer_n_signs=n_signs)
                    else:
                        row.update(batch_boot_lo=np.nan, batch_boot_hi=np.nan,
                                   batch_boot_p=np.nan, batch_boot_se=np.nan,
                                   n_batch_clusters=0, wild_t=np.nan, wild_p=np.nan,
                                   wild_crse=np.nan, rand_infer_p=np.nan,
                                   rand_infer_n_signs=0)
                    table_rows.append(row)

        # published comparison
        pub = published["rows"].get(label, {})
        detail[label] = {
            "published": {t: {k: pub.get(t, {}).get(k) for k in
                              ("acc_full_790", "acc_cleaned_476", "acc_rand476_mean",
                               "mcnemar_b", "mcnemar_c", "mcnemar_p")}
                          for t in ("adv", "nat")},
            "fam_seconds": time.time() - t_fam,
        }
        print(f"[{label}] done in {time.time()-t_fam:.1f}s", flush=True)
        del Xtr_all
        if emb_prefix is not None:
            del ytr_all, pid_all

    tab = pd.DataFrame(table_rows)
    tab.to_csv(OUT / "clustered_table4.csv", index=False)
    pd.DataFrame(per_item_rows).to_csv(OUT / "f02_per_item_correct.csv", index=False)

    # ---- reproduction check vs published --------------------------------- #
    repro = []
    for label, *_ in FAMILIES:
        pub = published["rows"].get(label, {})
        for t in ("adv", "nat"):
            r = tab[(tab.family == label) & (tab.cohort == t) &
                    (tab.contrast == "cleaned_minus_full") & (tab.subset == "all135")]
            if len(r) != 1:
                continue
            r = r.iloc[0]
            pv = pub.get(t, {})
            repro.append(dict(
                family=label, cohort=t,
                acc_full_new=r["acc_full"], acc_full_pub=pv.get("acc_full_790"),
                acc_clean_new=r["acc_cleaned"], acc_clean_pub=pv.get("acc_cleaned_476"),
                p_new=r["mcnemar_p"], p_pub=pv.get("mcnemar_p"),
                b_new=r["mcnemar_b"], b_pub=pv.get("mcnemar_b"),
                c_new=r["mcnemar_c"], c_pub=pv.get("mcnemar_c"),
            ))
    rep = pd.DataFrame(repro)
    rep.to_csv(OUT / "f02_repro_vs_published.csv", index=False)

    meta = {
        "random476_manifest_source": rand_src,
        "B_cluster_bootstrap": B_CLUSTER, "B_wild": B_WILD,
        "rng_cluster": RNG_CLUSTER, "rng_wild": RNG_WILD,
        "lr_seed": SEED,
        "elapsed_sec": time.time() - t_start,
        "families": [f[0] for f in FAMILIES],
        "detail": detail,
    }
    (OUT / "f02_clustered_table4.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\n[done] {time.time()-t_start:.1f}s -> {OUT/'clustered_table4.csv'}")


if __name__ == "__main__":
    raise SystemExit(main())
