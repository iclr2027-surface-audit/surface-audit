#!/usr/bin/env python3
"""Zero-shot restyling check on the released SurfaceFlipped-131 cohort.

Same 131 questions rendered two ways -- 'plain' (true_answer_plain / false_answer_plain)
and 'inverted' (true_text_surface_inverted / false_text_surface_inverted) -- with the A/B
position of the true answer drawn ONCE per question (random.Random(42) over the file order)
and shared across renderings, so the two arms differ only in answer text.

Protocol otherwise identical to scripts/run_binary_choice_eval.py: same PROMPT_TEMPLATE,
greedy decoding, max_new_tokens=32.  Parsing is stricter than the panel eval: the first
line of the continuation must resolve to a single standalone letter; if both letters
appear standalone (an 'A\\nB' echo) or none does, the row is scored NONE (wrong).
Raw continuations are stored so the parse can be audited.

Usage:
  run_restyle_panel.py --models microsoft/Phi-3.5-mini-instruct Qwen/Qwen2.5-3B-Instruct
                       [--device auto|mps|cuda|cpu] [--out DIR] [--limit N]
Outputs (release copy): data/predictions/restyle/<model>__plain131.csv and <model>__inverted131.csv
         (model_name, pair_id, true_is_a, choice, correct, raw).  Resume-safe.
"""
import argparse, csv, gc, random, re, sys, time
from pathlib import Path
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_binary_choice_eval import PROMPT_TEMPLATE  # noqa: E402

_AB = re.compile(r"(?<![A-Za-z])([AB])(?![A-Za-z])")

def parse_strict(text):
    """First standalone A/B in the first line; an option echo ('A\\nB' / 'B\\nA', both letters
    standalone within the first 8 characters and nothing else on those lines) is NONE."""
    t = text.strip()
    t = re.sub(r"^(answer|choice)\s*[:\-]\s*", "", t, flags=re.I).strip()
    if not t:
        return None
    head = [m.group(1).upper() for m in _AB.finditer(t[:8])]
    if len(set(head)) > 1 and re.fullmatch(r"[AB][\s:.)-]*[AB][\s:.)-]*", t[:8].strip()[:5] or ""):
        return None
    first = t.splitlines()[0]
    first = re.sub(r"^\s*(answer|choice)\s*[:\-]\s*", "", first, flags=re.I)
    letters = [m.group(1).upper() for m in _AB.finditer(first)]
    if not letters:
        letters = head
    if not letters or len(set(letters)) > 1:
        return None
    return letters[0]

def load_arms(limit=None):
    df = pd.read_csv(ROOT / "hf_release" / "SurfaceFlipped-131.csv")
    if limit:
        df = df.head(limit)
    rng = random.Random(42)
    side = {int(r.id): (1 if rng.random() < 0.5 else 0) for r in df.itertuples()}
    arms = {
        "plain131": [(int(r.id), str(r.question), str(r.true_answer_plain), str(r.false_answer_plain)) for r in df.itertuples()],
        "inverted131": [(int(r.id), str(r.question), str(r.true_text_surface_inverted), str(r.false_text_surface_inverted)) for r in df.itertuples()],
    }
    return arms, side

def pick_device(arg):
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def eval_model(model_name, arms, side, device, out_dir, max_new_tokens=32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    safe = model_name.replace("/", "_")
    todo = [(a, out_dir / f"{safe}__{a}.csv") for a in arms if not (out_dir / f"{safe}__{a}.csv").exists()]
    if not todo:
        print(f"[skip] {model_name} (outputs exist)", flush=True); return
    print(f"[load] {model_name} on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for arm, out_csv in todo:
        rows, t0 = [], time.time()
        for pair_id, question, true_text, false_text in arms[arm]:
            true_is_a = side[pair_id]
            a, b = (true_text, false_text) if true_is_a else (false_text, true_text)
            prompt = PROMPT_TEMPLATE.format(question=question, answer_a=a, answer_b=b)
            inputs = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
            raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            choice = parse_strict(raw)
            correct = 0 if choice is None else int((choice == "A") == (true_is_a == 1))
            rows.append({"model_name": model_name, "pair_id": pair_id, "true_is_a": true_is_a,
                         "choice": choice or "NONE", "correct": correct, "raw": raw.replace("\n", "\\n")[:200]})
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        acc = sum(r["correct"] for r in rows) / len(rows); none_ct = sum(r["choice"] == "NONE" for r in rows)
        print(f"[done] {model_name} {arm}: acc={acc:.3f} unparsed={none_ct}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    del model; gc.collect()
    if device == "mps": torch.mps.empty_cache()
    if device == "cuda": torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(ROOT / "data" / "predictions" / "restyle"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    arms, side = load_arms(args.limit)
    device = pick_device(args.device)
    print(f"device={device} pairs={len(arms['plain131'])} true_is_a_rate={sum(side.values())/len(side):.3f}", flush=True)
    for m in args.models:
        try:
            eval_model(m, arms, side, device, out_dir)
        except Exception as e:
            print(f"[ERROR] {m}: {type(e).__name__}: {e}", flush=True)

if __name__ == "__main__":
    main()
