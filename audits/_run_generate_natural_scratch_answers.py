#!/usr/bin/env python3
"""
Stage 3 of the natural-from-scratch 135-pair build.

For each candidate question in
audits/_eval_dataset_natural_scratch_questions_raw.json, generate a
true_answer_natural and a false_answer_natural via the same prompt used
for the natural135 cohort.

Output: audits/_eval_dataset_natural_scratch_raw.json
  Same per-row schema as audits/_eval_dataset_natural135_raw.json
  plus source_category and topic_domain pass-through.
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_JSON = REPO_ROOT / "audits" / "_eval_dataset_natural_scratch_questions_raw.json"
OUT_JSON = REPO_ROOT / "audits" / "_eval_dataset_natural_scratch_raw.json"

MODEL_CHAIN = ["claude-opus-4-5", "claude-opus-4-7", "claude-sonnet-4-6"]
TEMPERATURE = 0.7
MAX_TOKENS = 32000
NO_TEMPERATURE_MODELS = {"claude-opus-4-7"}
SLEEP_BETWEEN_CALLS_SEC = 0.0

PROMPT_TEMPLATE = """You will receive a question. Produce a JSON object with two fields:

- "true_answer_natural": a factually correct, plain answer to the question. Write it the way you would naturally answer - no length target, no stylistic constraint.
- "false_answer_natural": a plausible but factually wrong answer to the question (a misconception or common error). Plain prose, no stylistic constraint.

Do not mention or think about surface features like negation, hedging, length, or authority phrasing. Just write natural answers.

Return only the JSON object. No commentary.

QUESTION: {question}"""


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


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```\s*$", "", t)
    return t.strip()


def _parse_obj(text: str) -> dict:
    t = _strip_fence(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("could not parse JSON object from response")


def _call_one(client, question: str) -> tuple[dict, str, dict]:
    last_err = None
    for model_id in MODEL_CHAIN:
        use_temp = model_id not in NO_TEMPERATURE_MODELS
        try:
            kwargs = {
                "model": model_id,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "user",
                              "content": PROMPT_TEMPLATE.format(question=question)}],
            }
            if use_temp:
                kwargs["temperature"] = TEMPERATURE
            with client.messages.stream(**kwargs) as stream:
                for _ in stream.text_stream:
                    pass
                resp = stream.get_final_message()
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        try:
            obj = _parse_obj(text)
        except Exception as e:
            last_err = f"parse({model_id}): {e} | first200={text[:200]!r}"
            continue
        if not isinstance(obj, dict) or "true_answer_natural" not in obj or "false_answer_natural" not in obj:
            last_err = f"missing_keys({model_id}): keys={list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__}"
            continue
        usage = getattr(resp, "usage", None)
        usage_dict = {
            "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        }
        return obj, model_id, usage_dict
    raise RuntimeError(last_err or "all models failed without error")


def _load_existing() -> dict[int, dict]:
    if not OUT_JSON.exists():
        return {}
    try:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = data.get("rows", []) if isinstance(data, dict) else data
    return {int(r["id"]): r for r in rows if "id" in r}


def _save(rows_by_id: dict[int, dict], meta: dict):
    rows = [rows_by_id[k] for k in sorted(rows_by_id)]
    n_ok = sum(1 for r in rows if "true_answer_natural" in r and "false_answer_natural" in r and "error" not in r)
    n_fail = sum(1 for r in rows if "error" in r)
    out = {
        **meta,
        "n_rows": len(rows),
        "n_success": n_ok,
        "n_failed": n_fail,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    _load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    if not SRC_JSON.exists():
        print(f"ERROR: source missing: {SRC_JSON}", file=sys.stderr)
        return 1

    src = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    src_rows = src.get("rows", src) if isinstance(src, dict) else src
    print(f"Loaded {len(src_rows)} candidate questions from {SRC_JSON.name}", flush=True)

    existing = _load_existing()
    pending = [r for r in src_rows
               if int(r["id"]) not in existing
               or "error" in existing.get(int(r["id"]), {})]
    print(f"Existing successful: {sum(1 for r in existing.values() if 'error' not in r)};  "
          f"pending: {len(pending)}", flush=True)

    from anthropic import Anthropic
    client = Anthropic()

    rows_by_id = dict(existing)
    meta = {
        "model_chain": MODEL_CHAIN,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "no_temperature_models": sorted(NO_TEMPERATURE_MODELS),
        "source_json": str(SRC_JSON.relative_to(REPO_ROOT)),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    t0 = time.time()
    for i, src in enumerate(pending, 1):
        qid = int(src["id"])
        try:
            obj, model_used, usage = _call_one(client, src["question"])
            rows_by_id[qid] = {
                "id": qid,
                "source_category": src.get("source_category", ""),
                "topic_domain": src.get("topic_domain", ""),
                "question": src["question"],
                "true_answer_natural":  str(obj["true_answer_natural"]).strip(),
                "false_answer_natural": str(obj["false_answer_natural"]).strip(),
                "model_used": model_used,
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
            }
            print(f"  [{i:3d}/{len(pending)}] id={qid}  OK   model={model_used}  "
                  f"in/out={usage['input_tokens']}/{usage['output_tokens']}", flush=True)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            rows_by_id[qid] = {
                "id": qid,
                "source_category": src.get("source_category", ""),
                "topic_domain": src.get("topic_domain", ""),
                "question": src["question"],
                "error": err,
            }
            print(f"  [{i:3d}/{len(pending)}] id={qid}  FAIL {err[:160]}", flush=True)

        if i % 10 == 0 or i == len(pending):
            _save(rows_by_id, meta)
        if SLEEP_BETWEEN_CALLS_SEC:
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    elapsed = time.time() - t0
    _save(rows_by_id, {**meta, "elapsed_sec": round(elapsed, 2)})

    rows = [rows_by_id[k] for k in sorted(rows_by_id)]
    n_ok = sum(1 for r in rows if "error" not in r)
    n_fail = sum(1 for r in rows if "error" in r)

    print()
    print("=" * 60)
    print(f"Done. {n_ok}/{len(rows)} succeeded, {n_fail} failed. "
          f"Elapsed {elapsed/60:.1f} min.")
    print(f"Output: {OUT_JSON.relative_to(REPO_ROOT)}")
    if n_fail:
        print()
        print("Failed rows:")
        for r in rows:
            if "error" in r:
                print(f"  id={r['id']:5d}  {r['error']}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
