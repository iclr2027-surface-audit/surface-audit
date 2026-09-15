#!/usr/bin/env python3
"""
Generate surface-inverted eval candidates via Anthropic API.

Prompt source-of-truth: audits/_eval_dataset_{VERSION}_prompt.txt
After a successful run, the exact prompt sent to the API plus model
metadata is written to audits/_eval_dataset_{VERSION}_prompt_used.txt
in the same format as v2's _prompt_used.txt.

Primary model: claude-opus-4-5; fallback chain claude-opus-4-7,
claude-sonnet-4-6. T=0.7, max_tokens=32000, streaming.
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION = os.environ.get("EVAL_DATASET_VERSION", "v2")
PROMPT_VERSION = os.environ.get("EVAL_DATASET_PROMPT_VERSION", VERSION)
OUT_JSON = REPO_ROOT / "audits" / f"_eval_dataset_{VERSION}_raw.json"
PROMPT_FILE = REPO_ROOT / "audits" / f"_eval_dataset_{PROMPT_VERSION}_prompt.txt"
PROMPT_USED_FILE = REPO_ROOT / "audits" / f"_eval_dataset_{VERSION}_prompt_used.txt"

MODEL_CHAIN = ["claude-opus-4-5", "claude-opus-4-7", "claude-sonnet-4-6"]
TEMPERATURE = 0.7
MAX_TOKENS = 32000
N_TARGET = int(os.environ.get("EVAL_DATASET_N_TARGET", "40"))
RESUME_MIN_ROWS = max(1, N_TARGET // 2)
NO_TEMPERATURE_MODELS = {"claude-opus-4-7"}


def _load_dotenv(path: Path):
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _load_prompt() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(
            f"prompt file missing: {PROMPT_FILE.relative_to(REPO_ROOT)}\n"
            f"the prompt is the single source of truth; create it before running."
        )
    text = PROMPT_FILE.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"prompt file is empty: {PROMPT_FILE.relative_to(REPO_ROOT)}")
    return text


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```\s*$", "", t)
    return t.strip()


def _repair_truncated_array(t: str):
    """If the response was cut off mid-row, keep only complete top-level
    objects and close the array."""
    if not t.lstrip().startswith("["):
        return None
    start = t.index("[")
    body = t[start + 1:]
    depth = 0
    in_str = False
    esc = False
    last_complete = -1
    for i, ch in enumerate(body):
        if esc:
            esc = False; continue
        if ch == "\\" and in_str:
            esc = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_complete = i
    if last_complete < 0:
        return None
    repaired = "[" + body[:last_complete + 1] + "]"
    return json.loads(repaired)


def _try_parse_array(text: str):
    t = _strip_fence(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    repaired = _repair_truncated_array(t)
    if repaired is not None:
        print(f"  NOTE: response truncated; recovered {len(repaired)} complete rows", flush=True)
        return repaired
    raise ValueError("could not parse JSON array from response")


def main():
    _load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    PROMPT = _load_prompt()
    print(f"Loaded prompt from {PROMPT_FILE.relative_to(REPO_ROOT)} "
          f"({len(PROMPT)} chars)", flush=True)

    if OUT_JSON.exists():
        try:
            existing = json.loads(OUT_JSON.read_text())
            rows = existing.get("rows", existing) if isinstance(existing, dict) else existing
            if isinstance(rows, list) and len(rows) >= RESUME_MIN_ROWS:
                print(f"Resume: {OUT_JSON.name} already has {len(rows)} rows. Skipping generation.")
                return 0
        except Exception as e:
            print(f"WARN: existing raw JSON unreadable ({e}); regenerating.")

    from anthropic import Anthropic
    client = Anthropic()

    last_err = None
    for model_id in MODEL_CHAIN:
        use_temp = model_id not in NO_TEMPERATURE_MODELS
        print(f"Trying model={model_id}  T={TEMPERATURE if use_temp else 'default'}  "
              f"max_tokens={MAX_TOKENS}  (streaming)", flush=True)
        t0 = time.time()
        try:
            kwargs = {
                "model": model_id,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "user", "content": PROMPT}],
            }
            if use_temp:
                kwargs["temperature"] = TEMPERATURE
            with client.messages.stream(**kwargs) as stream:
                for _ in stream.text_stream:
                    pass
                resp = stream.get_final_message()
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"  FAILED: {last_err}", flush=True)
            continue
        elapsed = time.time() - t0
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        try:
            rows = _try_parse_array(text)
        except Exception as e:
            print(f"  PARSE FAILED: {e}", flush=True)
            print(f"  first 500 chars of response: {text[:500]!r}", flush=True)
            last_err = f"parse: {e}"
            continue

        usage = getattr(resp, "usage", None)
        in_tok  = getattr(usage, "input_tokens", None) if usage else None
        out_tok = getattr(usage, "output_tokens", None) if usage else None

        OUT_JSON.write_text(json.dumps({
            "model": model_id,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "elapsed_sec": round(elapsed, 2),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "n_rows": len(rows),
            "rows": rows,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        used_header = (
            "=== Model and parameters ===\n"
            f"model: {model_id}\n"
            f"temperature: {TEMPERATURE if use_temp else 'default (model does not accept temperature)'}\n"
            f"max_tokens: {MAX_TOKENS}\n"
            f"streaming: yes (client.messages.stream)\n"
            f"fallback_chain: {MODEL_CHAIN}\n"
            f"no_temperature_models: {sorted(NO_TEMPERATURE_MODELS)}\n"
            f"prompt_source: {PROMPT_FILE.relative_to(REPO_ROOT)}\n"
            f"elapsed_sec: {round(elapsed, 2)}\n"
            f"input_tokens: {in_tok}\n"
            f"output_tokens: {out_tok}\n"
            f"n_rows_returned: {len(rows)}\n"
            "\n=== Full prompt sent to API ===\n"
        )
        PROMPT_USED_FILE.write_text(used_header + PROMPT, encoding="utf-8")

        print(f"OK  model={model_id}  rows={len(rows)}  "
              f"elapsed={elapsed:.1f}s  tokens in/out={in_tok}/{out_tok}", flush=True)
        print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
        print(f"Wrote {PROMPT_USED_FILE.relative_to(REPO_ROOT)}")
        return 0

    print(f"ERROR: all models in chain failed. Last error: {last_err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
