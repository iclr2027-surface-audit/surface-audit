#!/usr/bin/env python3
"""GPU cluster job — re-encode the v1.1-edited cohort rows with BGE-Multi-Gemma2 and
splice them into the 131-row v1.1 test matrices.

Protocol: exact copy of audits/_exp_bge_multilingual_gemma2.py GemmaEncoder used
to build the published caches artifacts/embeddings/{adv135,natural135}_bge_multi_gemma2_{T,F}.npy
(last-token pool of last_hidden_state, L2 normalize, bf16, batch_size=1, max_length=512,
padding_side="left").

Validation: re-encodes 3 UNEDITED rows per column and requires min cosine >= 0.9995 (relaxed from 0.9999: bf16 cross-GPU jitter ~1e-4 observed across GPUs; recipe errors show as cos<0.99)
against the shipped v1.0 cache before splicing (same gate as the local families).

Run from the bundle root (see the cluster job notes). Needs in the bundle:
  hf_release/archive_v1.0/{SurfaceFlipped-135.csv,Natural-135.csv}
  hf_release/{SurfaceFlipped-131.csv,Natural-131.csv}
  artifacts/embeddings/{adv135,natural135}_bge_multi_gemma2_{T,F}.npy   (v1.0 caches, read-only)
  artifacts/hf_cache/  (or let HF download BAAI/bge-multilingual-gemma2, ~34 GB)

Writes (new files):
  out_v11/{adv131,natural131}_bge_multi_gemma2_{T,F}.npy   (131 rows, exact)
  out_v11/gemma2_v11_validation.json
"""
import gc, json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HF = ROOT / "artifacts" / "hf_cache"
os.environ.setdefault("HF_HOME", str(HF))
os.environ.setdefault("HF_HUB_CACHE", str(HF / "hub"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, pandas as pd, torch

MODEL_ID = "BAAI/bge-multilingual-gemma2"
EMB = ROOT / "artifacts" / "embeddings"
OUT = ROOT / "out_v11"
OUT.mkdir(exist_ok=True)

sf0 = pd.read_csv(ROOT / "hf_release" / "archive_v1.0" / "SurfaceFlipped-135.csv")
na0 = pd.read_csv(ROOT / "hf_release" / "archive_v1.0" / "Natural-135.csv")
sf1 = pd.read_csv(ROOT / "hf_release" / "SurfaceFlipped-131.csv")
na1 = pd.read_csv(ROOT / "hf_release" / "Natural-131.csv")
assert len(sf0) == len(na0) == 135 and len(sf1) == len(na1) == 131

keep_adv = sf0.id.isin(set(sf1.id)).to_numpy(); keep_nat = na0.id.isin(set(na1.id)).to_numpy()
assert keep_adv.sum() == 131 and keep_nat.sum() == 131
assert list(sf0.id[keep_adv]) == list(sf1.id) and list(na0.id[keep_nat]) == list(na1.id)

sf1i = sf1.set_index("id"); na1i = na1.set_index("id")
COLS = [("adv135", "T", sf0, sf1i, "true_text_surface_inverted", keep_adv),
        ("adv135", "F", sf0, sf1i, "false_text_surface_inverted", keep_adv),
        ("natural135", "T", na0, na1i, "true_answer_natural", keep_nat),
        ("natural135", "F", na0, na1i, "false_answer_natural", keep_nat)]
OUTTAG = {"adv135": "adv131", "natural135": "natural131"}


class GemmaEncoder:  # verbatim protocol of audits/_exp_bge_multilingual_gemma2.py (CUDA-first here)
    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        t0 = time.time(); print(f"loading {MODEL_ID} on {self.device} (bf16)...", flush=True)
        self.mdl = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        self.mdl.eval(); self.mdl.to(self.device)
        print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    def encode(self, texts, batch_size=1, max_length=512):
        out = []
        for s in range(0, len(texts), batch_size):
            enc = self.tok(texts[s:s + batch_size], padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                o = self.mdl(**enc)
            pooled = o.last_hidden_state[:, -1, :]
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            out.append(pooled.detach().cpu().to(torch.float32).numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


enc = GemmaEncoder()
report = {}
for tag, side, df0, df1i, col, keep in COLS:
    cache = np.load(EMB / f"{tag}_bge_multi_gemma2_{side}.npy"); assert cache.shape[0] == 135
    old = df0[col].astype(str).tolist()
    new = [str(df1i.loc[i, col]) if i in df1i.index else str(v) for i, v in zip(df0.id, old)]
    edited = [i for i in range(135) if keep[i] and old[i] != new[i]]
    val_idx = [i for i in range(135) if keep[i] and i not in edited][:3]
    V = enc.encode([old[i] for i in val_idx]); C = cache[val_idx]
    cos = np.sum(V * C, 1) / (np.linalg.norm(V, axis=1) * np.linalg.norm(C, axis=1) + 1e-12)
    ok = bool(cos.min() >= 0.9995)
    report[f"{tag}_{side}"] = dict(val_rows=val_idx, val_cos_min=float(cos.min()),
                                   n_edited=len(edited),
                                   edited_ids=[int(df0.id.iloc[i]) for i in edited], cosine_ok=ok)
    print(f"{tag}_{side}: val_cos_min={cos.min():.6f} ok={ok} n_edited={len(edited)}", flush=True)
    if not ok:
        json.dump(report, open(OUT / "gemma2_v11_validation.json", "w"), indent=1)
        raise SystemExit(f"VALIDATION FAILED for {tag}_{side}: min cos {cos.min():.6f} < 0.9995")
    newX = cache.copy()
    if edited:
        newX[np.array(edited)] = enc.encode([new[i] for i in edited])
    newX = newX[keep]; assert newX.shape[0] == 131
    np.save(OUT / f"{OUTTAG[tag]}_bge_multi_gemma2_{side}.npy", newX.astype(np.float32))
    json.dump(report, open(OUT / "gemma2_v11_validation.json", "w"), indent=1)

del enc.mdl, enc.tok; gc.collect()
print("done; wrote out_v11/{adv131,natural131}_bge_multi_gemma2_{T,F}.npy", flush=True)
