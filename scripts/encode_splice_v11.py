#!/usr/bin/env python3
"""v1.1 rescore step 1 — re-encode ONLY the edited rows of the two cohorts and
splice into copies of the cached test matrices; drop the 8 removed rows.

Encoder recipes are copied verbatim from scripts/d21_encode_plain_adv135.py
(itself copied from scripts/build_*_embeddings.py: normalised sentence-transformers
for BGE-large, CLS for ModernBERT, attention-masked mean-pool of last hidden state
for the causal LMs, max_length=512, published batch sizes / dtypes).

Validation per family per text column: re-encode 3 UNEDITED rows and require
mean/min cosine >= 0.9999 against the published cache before splicing.

Inputs (read-only):
  audits/_eval_dataset_v3_inv_accuracy_full_final.csv   (== hf_release/archive_v1.0/SurfaceFlipped-135.csv text fields, verified)
  audits/_eval_dataset_natural_scratch_final.csv        (== hf_release/archive_v1.0/Natural-135.csv text fields, verified)
  hf_release/SurfaceFlipped-131.csv, hf_release/Natural-131.csv
  hf_release/v1.1_change_manifest.csv
  artifacts/embeddings/{adv135,adv135plain,natural135}_{prefix}_{T,F}.npy

Outputs (new files only, under results/v1_1_rescore/emb/):
  adv131_{prefix}_{T,F}.npy, adv131plain_{prefix}_{T,F}.npy, natural131_{prefix}_{T,F}.npy
  (bge_multi_gemma2: drop-only splice of the stale cache, adv+natural only — PROVISIONAL
   until the cluster re-encode; no plain cache exists for that family.)
  results/v1_1_rescore/encode_validation.json
  results/v1_1_rescore/emb/PROVENANCE.json
"""
import gc, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "artifacts" / "hf_cache"
os.environ["HF_HOME"] = str(HF)
os.environ["HF_HUB_CACHE"] = str(HF / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, pandas as pd

EMB = ROOT / "artifacts" / "embeddings"
OUT = ROOT / "results" / "v1_1_rescore" / "emb"
OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results" / "v1_1_rescore"

adv0 = pd.read_csv(ROOT / "audits" / "_eval_dataset_v3_inv_accuracy_full_final.csv")
nat0 = pd.read_csv(ROOT / "audits" / "_eval_dataset_natural_scratch_final.csv")
sf1 = pd.read_csv(ROOT / "hf_release" / "SurfaceFlipped-131.csv")
na1 = pd.read_csv(ROOT / "hf_release" / "Natural-131.csv")
man = pd.read_csv(ROOT / "hf_release" / "v1.1_change_manifest.csv")
assert len(adv0) == len(nat0) == 135 and len(sf1) == len(na1) == 131

# keep masks over the 135-row frames (order verified preserved)
keep_adv = adv0.id.isin(set(sf1.id)).to_numpy()
keep_nat = nat0.id.isin(set(na1.id)).to_numpy()
assert keep_adv.sum() == 131 and keep_nat.sum() == 131
assert list(adv0.id[keep_adv]) == list(sf1.id) and list(nat0.id[keep_nat]) == list(na1.id)

# v1.1 texts aligned to the 135-row frames (dropped rows keep the old text; they
# are removed by the keep mask after splicing, so their values never matter)
sf1i = sf1.set_index("id"); na1i = na1.set_index("id")
def aligned(df0, df1i, col):
    return [str(df1i.loc[i, col]) if i in df1i.index else str(v)
            for i, v in zip(df0.id, df0[col])]

# (cache tag, side, 135-frame df, column, keep mask, v1.0 texts, v1.1 texts)
COLS = []
for side, col in (("T", "true_text_surface_inverted"), ("F", "false_text_surface_inverted")):
    COLS.append(("adv135", side, adv0, col, keep_adv))
for side, col in (("T", "true_answer_plain"), ("F", "false_answer_plain")):
    COLS.append(("adv135plain", side, adv0, col, keep_adv))
for side, col in (("T", "true_answer_natural"), ("F", "false_answer_natural")):
    COLS.append(("natural135", side, nat0, col, keep_nat))

# edited row indices per column = kept rows whose text changed v1.0 -> v1.1
PLAN = {}
for tag, side, df0, col, keep in COLS:
    old = df0[col].astype(str).tolist()
    new = aligned(df0, sf1i if df0 is adv0 else na1i, col)
    idx = [i for i in range(135) if keep[i] and old[i] != new[i]]
    PLAN[(tag, side)] = dict(col=col, edited_idx=idx, new_texts=[new[i] for i in idx],
                             old_texts=old, keep=keep)

# cross-check against the manifest
man_edit = man[man.kind == "edit"]
expect = {
    ("adv135", "T"): set(man_edit[(man_edit.cohort == "SurfaceFlipped") & (man_edit.field == "true_text_surface_inverted")].id),
    ("adv135", "F"): set(man_edit[(man_edit.cohort == "SurfaceFlipped") & (man_edit.field == "false_text_surface_inverted")].id),
    ("adv135plain", "T"): set(man_edit[(man_edit.cohort == "SurfaceFlipped") & (man_edit.field == "true_answer_plain")].id),
    ("adv135plain", "F"): set(man_edit[(man_edit.cohort == "SurfaceFlipped") & (man_edit.field == "false_answer_plain")].id),
    ("natural135", "T"): set(man_edit[(man_edit.cohort == "Natural") & (man_edit.field == "true_answer_natural")].id),
    ("natural135", "F"): set(man_edit[(man_edit.cohort == "Natural") & (man_edit.field == "false_answer_natural")].id),
}
for (tag, side), p in PLAN.items():
    df0 = adv0 if tag.startswith("adv") else nat0
    got = {int(df0.id.iloc[i]) for i in p["edited_idx"]}
    assert got == expect[(tag, side)], f"{tag}_{side}: diff-derived {got} != manifest {expect[(tag, side)]}"
print("[plan] edited-row sets match the manifest for all 6 columns", flush=True)
for k, p in PLAN.items():
    print("  ", k, "n_edited=", len(p["edited_idx"]), flush=True)

def _dev():
    import torch
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ---- encoder recipes (verbatim from scripts/d21_encode_plain_adv135.py), but
# ---- restructured so each family loads its model once and returns a closure.
def make_bge():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-large-en-v1.5")
    def enc(texts):
        X = m.encode(texts, batch_size=32, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(X, dtype=np.float32)
    return enc

def make_mbert():
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = _dev()
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    mdl = AutoModel.from_pretrained("answerdotai/ModernBERT-base"); mdl.eval(); mdl.to(dev)
    def enc(texts):
        out = []
        with torch.no_grad():
            for s in range(0, len(texts), 32):
                e = tok(texts[s:s+32], padding=True, truncation=True, max_length=512, return_tensors="pt")
                e = {k: v.to(dev) for k, v in e.items()}
                out.append(mdl(**e).last_hidden_state[:, 0, :].detach().cpu().to(torch.float32).numpy())
        return np.concatenate(out).astype(np.float32)
    return enc

def make_causal(model_id, batch_size, dtype_name):
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = _dev(); dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=dtype)
    mdl.eval(); mdl.to(dev)
    def enc(texts):
        out = []
        with torch.no_grad():
            for s in range(0, len(texts), batch_size):
                e = tok(texts[s:s+batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt")
                e = {k: v.to(dev) for k, v in e.items()}
                h = mdl(**e).last_hidden_state
                m = e["attention_mask"].unsqueeze(-1).to(h.dtype)
                out.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).detach().cpu().to(torch.float32).numpy())
        return np.concatenate(out).astype(np.float32)
    return enc

FAMILIES = [
    ("bge_large", make_bge, {}),
    ("modernbert", make_mbert, {}),
    ("qwen", make_causal, dict(model_id="Qwen/Qwen2.5-0.5B-Instruct", batch_size=16, dtype_name="fp32")),
    ("qwen15b", make_causal, dict(model_id="Qwen/Qwen2.5-1.5B-Instruct", batch_size=8, dtype_name="bf16")),
    ("smollm2", make_causal, dict(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", batch_size=8, dtype_name="bf16")),
    ("qwen3b", make_causal, dict(model_id="Qwen/Qwen2.5-3B-Instruct", batch_size=4, dtype_name="bf16")),
    ("phi35", make_causal, dict(model_id="microsoft/Phi-3.5-mini-instruct", batch_size=4, dtype_name="bf16")),
    ("llama32_3b", make_causal, dict(model_id="meta-llama/Llama-3.2-3B-Instruct", batch_size=4, dtype_name="bf16")),
]

OUTTAG = {"adv135": "adv131", "adv135plain": "adv131plain", "natural135": "natural131"}
COSINE_MIN = 0.9999
report = {}
provenance = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "note": "131-row v1.1 test matrices: published caches with edited rows re-encoded and the 8 dropped rows removed",
              "files": {}}
only = set(sys.argv[1:])
failures = []
for pref, maker, kw in FAMILIES:
    if only and pref not in only:
        continue
    t0 = time.time(); print(f"=== {pref}", flush=True)
    enc = maker(**kw)
    fam_rep = {}
    for (tag, side), p in PLAN.items():
        cache = np.load(EMB / f"{tag}_{pref}_{side}.npy")
        assert cache.shape[0] == 135
        # validation: first 3 kept, unedited rows of this column
        val_idx = [i for i in range(135) if p["keep"][i] and i not in p["edited_idx"]][:3]
        V = enc([p["old_texts"][i] for i in val_idx])
        C = cache[val_idx]
        cos = np.sum(V * C, 1) / (np.linalg.norm(V, axis=1) * np.linalg.norm(C, axis=1) + 1e-12)
        ok = bool(cos.min() >= COSINE_MIN)
        fam_rep[f"{tag}_{side}"] = dict(val_rows=val_idx, val_cos_min=float(cos.min()),
                                        val_cos_mean=float(cos.mean()),
                                        val_max_abs_diff=float(np.abs(V - C).max()),
                                        n_edited=len(p["edited_idx"]), cosine_ok=ok)
        if not ok:
            failures.append(f"{pref}:{tag}_{side} min cos {cos.min():.6f}")
            print(f"  !! VALIDATION FAILED {tag}_{side}: min cos {cos.min():.6f}", flush=True)
            continue
        new = cache.copy()
        if p["edited_idx"]:
            new[np.array(p["edited_idx"])] = enc(p["new_texts"])
        new = new[p["keep"]]
        assert new.shape[0] == 131
        fp = OUT / f"{OUTTAG[tag]}_{pref}_{side}.npy"
        np.save(fp, new.astype(np.float32))
        provenance["files"][fp.name] = dict(source=f"{tag}_{pref}_{side}.npy",
                                            re_encoded_rows_135frame=p["edited_idx"],
                                            status="exact")
        print(f"  {tag}_{side}: val_cos_min={cos.min():.6f} spliced {len(p['edited_idx'])} rows -> {fp.name}", flush=True)
    fam_rep["seconds"] = round(time.time() - t0, 1)
    report[pref] = fam_rep
    json.dump(report, open(RES / "encode_validation.json", "w"), indent=1)
    del enc; gc.collect()
    try:
        import torch; torch.mps.empty_cache()
    except Exception:
        pass

# ---- BGE-Multi-Gemma2: drop-only splice of the stale cache (PROVISIONAL) -----
for tag in ("adv135", "natural135"):
    for side in ("T", "F"):
        p = PLAN[(tag, side)]
        cache = np.load(EMB / f"{tag}_bge_multi_gemma2_{side}.npy")
        new = cache.copy()[p["keep"]]
        assert new.shape[0] == 131
        fp = OUT / f"{OUTTAG[tag]}_bge_multi_gemma2_{side}.npy"
        np.save(fp, new.astype(np.float32))
        df0 = adv0 if tag == "adv135" else nat0
        stale_ids = [int(df0.id.iloc[i]) for i in p["edited_idx"]]
        kept_pos = np.cumsum(p["keep"]) - 1  # 135-frame index -> 131-frame index
        provenance["files"][fp.name] = dict(
            source=f"{tag}_bge_multi_gemma2_{side}.npy",
            re_encoded_rows_135frame=[], status="PROVISIONAL-STALE" if stale_ids else "exact",
            stale_edited_ids=stale_ids,
            stale_rows_131frame=[int(kept_pos[i]) for i in p["edited_idx"]],
            note="edited rows still carry v1.0 embeddings; overwrite via GPU cluster job (the cluster job notes)")
        print(f"gemma2 {tag}_{side}: drop-only splice, {len(stale_ids)} stale edited rows {stale_ids}", flush=True)

json.dump(provenance, open(OUT / "PROVENANCE.json", "w"), indent=1)
json.dump(report, open(RES / "encode_validation.json", "w"), indent=1)
if failures:
    print("VALIDATION FAILURES:", failures, flush=True)
    sys.exit(1)
print("done", flush=True)
